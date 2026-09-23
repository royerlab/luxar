#!/usr/bin/env python3
"""
Cyclomatic Complexity Ratchet

Enforces ``[tool.ruff.lint.mccabe] max-complexity`` as a baseline-driven
*ratchet*: pre-existing over-limit functions are tolerated via a checked-in
baseline (``scripts/complexity_baseline.json``), but a function that is NEWLY
over the limit — or an already-baselined one whose complexity INCREASED — fails
the check. Paid-down debt and functions that merely MOVED (same name, no greater
complexity, new file) also fail a full run until ``--update-baseline`` re-keys or
tightens the baseline. Restricted runs keep those findings advisory because
keys outside the explicit targets were not scanned.

Why a script instead of putting ``C901`` in ``[tool.ruff.lint] select``?
ruff has no baseline mechanism. A bare ``select`` entry would fail on all
pre-existing violations (195 at the time of writing), so it could not be turned
on at all without a large, unrelated refactor. The only ruff-native suppression
is ``per-file-ignores``, which is *file*-granular: silencing the 140 files that
currently hold a violation would also blind the guard to brand-new offenders
inside those very files — precisely the code most likely to grow. This checker
selects the rule explicitly and diffs the findings against the baseline, so the
limit applies to new code while existing debt stays visible and shrinkable.
Ruff's root resolved settings are fingerprinted in the baseline, nested Ruff
configs are refused, and existing baselined files must remain in Ruff's
``--show-files`` set. Configuration changes and partial scans therefore fail
closed instead of masquerading as paid-down complexity debt.

Usage:
    python scripts/check_complexity.py
    python scripts/check_complexity.py --update-baseline
    python scripts/check_complexity.py --baseline path/to/baseline.json
    python scripts/check_complexity.py packages/luxar/src   # explicit targets
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import ruff_ratchet
from arbol import aprint

# Repository root (this script lives in `<root>/scripts/`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Lint targets, kept in sync with the `lint` script in pyproject.toml.
# `scripts/benchmarks` is an explicit safety net against a broadened ignore rule.
DEFAULT_TARGETS: tuple[str, ...] = (
    "packages/luxar/src",
    "scripts",
    "scripts/benchmarks",
    "stats",
    "hatch_build.py",
)
RATCHETED_SELECT: tuple[str, ...] = ("C901",)

# Baseline path relative to the project root.
DEFAULT_BASELINE_RELPATH = "scripts/complexity_baseline.json"

BASELINE_COMMENT = (
    "Cyclomatic-complexity baseline for scripts/check_complexity.py. "
    "Regenerate with: hatch run check-complexity --update-baseline. Each key is "
    "'<repo-relative-path>::<function-name>' and maps to the descending-sorted "
    "complexities of the over-limit functions with that name in that file (a "
    "list, because one file may hold several same-named functions). A function "
    "not listed here, or one whose complexity exceeds its baselined value, "
    "fails the check. A missing, moved, or lower-complexity entry also fails "
    "until the baseline is refreshed. 'settings_fingerprint' records Ruff's "
    "normalized root resolved settings; a mismatch fails rather than silently "
    "retiring debt."
)

# ruff's C901 message, e.g. "`robust_download` is too complex (66 > 10)".
_MESSAGE_RE = re.compile(r"^`(?P<name>[^`]+)` is too complex \((?P<value>\d+) > \d+\)$")

# ruff's stderr line for a path it could not read at all, e.g.
# "warning: Failed to lint stats: No such file or directory (os error 2)".
_UNSCANNED_RE = re.compile(r"Failed to lint ")


@dataclass
class RatchetReport:
    """Classification of the current findings against the baseline.

    Every changed bucket fails a full run. ``improved`` and ``moved`` remain
    advisory on restricted runs, where omitted keys may simply be unscanned.
    """

    new: list[str] = field(default_factory=list)
    worsened: list[str] = field(default_factory=list)
    improved: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    # Entries of the form "<old key> -> <new key>": the same function name at no
    # greater complexity, under a new path (a module move, possibly with a tidy).
    moved: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ruff invocation and parsing
# ---------------------------------------------------------------------------


def run_ruff(targets: tuple[str, ...] | list[str], project_root: Path) -> str:
    """Run ``ruff check --select C901`` over ``targets`` and return its stdout.

    ruff exits 1 when it reports findings, which is the NORMAL case here; only
    other exit codes (or a missing ``ruff`` module) are treated as hard errors.

    The gate must fail CLOSED: ``python -m ruff`` with ruff not installed also
    exits 1, but with EMPTY stdout and the error on stderr. Left unchecked that
    reads as "no findings at all", every baselined key looks fixed, and the
    checker reports green. A real zero-findings run prints ``[]``, never blank,
    so requiring non-blank stdout on exit 1 cannot misfire.

    A path ruff could NOT read (missing, unreadable, a dangling symlink) is only
    a stderr ``Failed to lint <path>: ...`` WARNING: ruff lints the remaining
    targets and exits 0 or 1 exactly as if the scan had been complete. That is a
    fail-OPEN — the baselined keys under the unread path look vanished, so they
    are reported as paid-down debt (or, worse, become pairing candidates that let
    a genuinely new function through as a "move") and the gate goes green on a
    partial scan. So such a warning is a hard error here, whatever the exit code.
    Other stderr warnings (notably ``No Python files found under the given
    path(s)``, which is a whole-scan miss the caller catches — see ``main``) are
    echoed rather than discarded.
    """
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--select",
        ",".join(RATCHETED_SELECT),
        "--output-format",
        "json",
        *targets,
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
    except OSError as exc:  # pragma: no cover - environment failure
        raise RuntimeError(f"Could not run ruff ({' '.join(command)}): {exc}") from exc

    # Exit 0 = no findings, 1 = findings reported. Anything else is a failure
    # (bad arguments, unreadable file, ...) — as is an exit 1 with no report at
    # all, which is what `No module named ruff` looks like.
    empty_failure = proc.returncode == 1 and not proc.stdout.strip()
    if proc.returncode not in (0, 1) or empty_failure:
        detail = " but reported nothing" if empty_failure else ""
        raise RuntimeError(
            f"ruff failed (exit {proc.returncode}{detail}): {' '.join(command)}\n"
            f"{proc.stderr.strip()}"
        )
    stderr = proc.stderr.strip()

    # A target ruff could not read leaves a partial scan behind a normal exit
    # code, which reads as "those functions are all fixed". Fail closed instead.
    unscanned = [line for line in stderr.splitlines() if _UNSCANNED_RE.search(line)]
    if unscanned:
        raise RuntimeError(
            "ruff could not read every target, so the scan is PARTIAL and its "
            "findings cannot be diffed against the baseline:\n"
            + "\n".join(f"  {line}" for line in unscanned)
            + f"\nCommand: {' '.join(command)} (cwd {project_root})"
        )

    if stderr:
        aprint(f"⚠️  ruff wrote to stderr: {stderr}")
    return proc.stdout


def ruff_settings_fingerprint(target: str, project_root: Path) -> str:
    """Hash Ruff's normalized resolved settings for this ratchet."""
    return ruff_ratchet.settings_fingerprint(target, project_root, RATCHETED_SELECT)


def ensure_no_nested_ruff_configs(
    targets: tuple[str, ...] | list[str], project_root: Path
) -> None:
    """Reject hierarchical Ruff configs outside the root fingerprint."""
    ruff_ratchet.ensure_no_nested_configs(targets, project_root)


def list_ruff_files(
    targets: tuple[str, ...] | list[str], project_root: Path
) -> set[str]:
    """Return the repo-relative files Ruff says it will scan."""
    return ruff_ratchet.list_files(targets, project_root, RATCHETED_SELECT)


def ensure_baselined_files_were_scanned(
    baseline: Mapping[str, object], scanned_files: set[str], project_root: Path
) -> None:
    """Fail when an existing baselined file was silently excluded by Ruff."""
    ruff_ratchet.ensure_baselined_files_scanned(baseline, scanned_files, project_root)


def _existing_baseline_coverage_error_for_update(
    previous_baseline: dict[str, object],
    targets: tuple[str, ...],
    project_root: Path,
    restricted: bool,
) -> str | None:
    """Return a coverage error if a settings refresh would omit baseline keys."""
    if restricted:
        return None
    try:
        scanned_files = list_ruff_files(targets, project_root)
        ensure_baselined_files_were_scanned(
            previous_baseline, scanned_files, project_root
        )
    except RuntimeError as exc:
        return str(exc)
    return None


def parse_findings(
    stdout: str, project_root: Path = PROJECT_ROOT
) -> dict[str, list[int]]:
    """Parse ruff's JSON output into ``{"<rel-path>::<name>": [complexities]}``.

    Keys use repo-relative POSIX paths so the baseline is stable across
    machines, and deliberately carry NO line number: an unrelated edit above a
    function must not churn the baseline. Values are descending-sorted lists
    because one file can legitimately hold several over-limit functions with the
    same name (two methods on different classes, a nested ``def``).

    Only ``C901`` diagnostics are considered; ruff reports ``invalid-syntax``
    whatever is selected, and that is not this gate's business.
    """
    if not stdout.strip():
        return {}

    try:
        findings = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse ruff JSON output: {exc}") from exc

    entries: dict[str, list[int]] = {}
    for finding in findings:
        # ruff emits `invalid-syntax` diagnostics regardless of `--select`, so a
        # WIP file with a syntax error anywhere under the targets would otherwise
        # trip the message-shape guard below with a wrong diagnosis.
        if finding.get("code") not in RATCHETED_SELECT:
            continue
        message = finding["message"]
        match = _MESSAGE_RE.match(message)
        if match is None:
            # A ruff wording change must fail loudly rather than silently empty
            # the baseline and disable the gate.
            raise ValueError(
                f"Unexpected C901 message format from ruff: {message!r}. "
                "The parser in scripts/check_complexity.py needs updating."
            )
        path = Path(finding["filename"])
        try:
            relative = path.relative_to(project_root).as_posix()
        except ValueError:
            relative = path.as_posix()
        key = f"{relative}::{match.group('name')}"
        entries.setdefault(key, []).append(int(match.group("value")))

    return {key: sorted(values, reverse=True) for key, values in entries.items()}


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------


def load_baseline(
    path: Path, settings_fingerprint: str | None = None
) -> dict[str, list[int]]:
    """Load the baselined complexities from ``path``.

    Returns ``{}`` if the file does not exist. Raises a clear ``ValueError`` if
    it exists but is malformed (invalid JSON, missing/ill-typed ``functions``
    object, or a value that is not a list of ints).
    """
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Baseline file {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or "functions" not in data:
        raise ValueError(
            f"Baseline file {path} is malformed: expected a JSON object with a "
            "'functions' object."
        )

    recorded_fingerprint = data.get("settings_fingerprint")
    if not isinstance(recorded_fingerprint, str) or not recorded_fingerprint:
        raise ValueError(
            f"Baseline file {path} is malformed: 'settings_fingerprint' must "
            "be a non-empty string."
        )
    if (
        settings_fingerprint is not None
        and recorded_fingerprint != settings_fingerprint
    ):
        raise ValueError(
            f"Baseline file {path} was recorded for Ruff settings fingerprint "
            f"{recorded_fingerprint!r}, but this run resolves to "
            f"{settings_fingerprint!r}. A complexity setting changed which "
            "findings the ratchet can see. Re-run --update-baseline deliberately "
            "after reviewing `ruff check --show-settings`."
        )

    functions = data["functions"]
    if not isinstance(functions, dict):
        raise ValueError(
            f"Baseline file {path} is malformed: 'functions' must be an object "
            "mapping '<path>::<name>' to a list of complexities."
        )

    for key, values in functions.items():
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in values)
        ):
            raise ValueError(
                f"Baseline file {path} is malformed: entry {key!r} must be a "
                "non-empty list of integers."
            )

    return {key: sorted(values, reverse=True) for key, values in functions.items()}


def save_baseline(
    path: Path,
    entries: dict[str, list[int]],
    settings_fingerprint: str,
) -> None:
    """Write ``entries`` to ``path`` as deterministic, human-diffable JSON.

    Keys are sorted and each value descending-sorted; the file uses ``indent=2``
    and ends with a trailing newline.
    """
    payload = {
        "_comment": BASELINE_COMMENT,
        "settings_fingerprint": settings_fingerprint,
        "functions": {
            key: sorted(entries[key], reverse=True) for key in sorted(entries)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Ratchet
# ---------------------------------------------------------------------------


def _is_worse(current: list[int], baseline: list[int]) -> bool:
    """Whether ``current`` is worse than ``baseline`` (both descending-sorted)."""
    if len(current) > len(baseline):
        return True
    return any(value > baseline[i] for i, value in enumerate(current))


def _by_complexity(keys: list[str], values: dict[str, list[int]]) -> list[str]:
    """Order ``keys`` by complexity descending, then by key — deterministically."""
    return sorted(keys, key=lambda k: (tuple(-v for v in values[k]), k))


def _pair_moves(
    current: dict[str, list[int]],
    baseline: dict[str, list[int]],
    new_keys: list[str],
    vanished_keys: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Pair vanished baseline keys with ``new`` keys naming the same function.

    A pair is allowed when the new key is NOT WORSE than the vanished one, which
    covers both a plain relocation and the far commoner move-and-tidy. Pairing is
    one-to-one and greedy over both sides ordered by complexity descending, so it
    is deterministic and N vanished ``::f`` entries absorb at most N new ones.

    Returns ``(moved, unpaired_new, unpaired_vanished)``; ``moved`` entries read
    ``"<old key> -> <new key>"``.
    """
    by_name: dict[str, list[str]] = {}
    for key in _by_complexity(new_keys, current):
        by_name.setdefault(key.rpartition("::")[2], []).append(key)

    moved: list[str] = []
    paired: set[str] = set()
    unpaired_vanished: list[str] = []

    for old_key in _by_complexity(vanished_keys, baseline):
        candidates = by_name.get(old_key.rpartition("::")[2], ())
        match = next(
            (
                key
                for key in candidates
                if key not in paired and not _is_worse(current[key], baseline[old_key])
            ),
            None,
        )
        if match is None:
            unpaired_vanished.append(old_key)
        else:
            paired.add(match)
            moved.append(f"{old_key} -> {match}")

    unpaired_new = [key for key in new_keys if key not in paired]
    return sorted(moved), unpaired_new, sorted(unpaired_vanished)


def evaluate_ratchet(
    current: dict[str, list[int]],
    baseline: dict[str, list[int]],
    *,
    pair_moves: bool = True,
) -> RatchetReport:
    """Classify ``current`` findings against ``baseline``.

    - ``new``: a key absent from the baseline (a function newly over the limit).
    - ``worsened``: a key with MORE over-limit functions than baselined, or any
      i-th descending-sorted complexity above the baseline's i-th.
    - ``moved``: a vanished baseline key paired one-to-one with a ``new`` key
      naming the same function at no greater complexity — a module move, with or
      without a tidy-up on the way (see ``_pair_moves``). A full run fails until
      the baseline is regenerated to re-key it. Moves are advisory only when
      ``pair_moves=False`` for a PARTIAL scan (explicit target paths): every
      baseline key outside the scanned targets then looks vanished and would be
      an eligible pairing candidate, so a genuinely new function could be
      absorbed by a file that was never read.
    - ``improved``: a key that is strictly better (fewer entries or a lower
      complexity), plus unpaired keys that vanished entirely (fixed or deleted).
    - ``unchanged``: the rest (tolerated pre-existing debt).

    Known limitations, both of which bound total debt while getting the IDENTITY
    of a function wrong:

    1. A key's value is a sorted multiset, so two same-named over-limit functions
       in one file are indistinguishable. Baseline ``[20, 12]`` against current
       ``[20, 11]`` is classified ``improved`` even if what really happened is
       that the 12 grew to 20 while the 20 shrank to 11.
    2. ``moved`` pairs on the function NAME alone, so a genuinely new over-complex
       function is absorbed as a move if a baselined function with the same name
       and no lower complexity vanishes in the same run. The baseline really does
       hold same-name buckets (``main`` appears six times), so this is not
       hypothetical: the claim "this is the same function, relocated" can be
       wrong. What is never wrong is the bound — a pair cannot increase total
       complexity debt, which is the guarantee this ratchet makes.
    """
    report = RatchetReport()

    for key in sorted(current):
        values = current[key]
        if key not in baseline:
            report.new.append(key)
            continue
        base = baseline[key]
        if _is_worse(values, base):
            report.worsened.append(key)
        elif _is_worse(base, values):
            report.improved.append(key)
        else:
            report.unchanged.append(key)

    vanished = sorted(set(baseline) - set(current))
    if pair_moves:
        report.moved, report.new, vanished = _pair_moves(
            current, baseline, report.new, vanished
        )
    report.improved.extend(vanished)
    report.improved.sort()
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_counts(values: list[int]) -> str:
    """Render a complexity list compactly (``"20"`` / ``"20, 14"``)."""
    return ", ".join(str(v) for v in values)


def _baseline_is_populated(path: Path) -> bool:
    """Whether ``path`` holds a non-empty baseline.

    A malformed or unreadable baseline counts as populated: it cannot be PROVEN
    empty, and the only caller uses this to decide whether overwriting would
    destroy debt.
    """
    if not path.exists():
        return False
    try:
        return bool(load_baseline(path))
    except (ValueError, OSError):
        return True


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Ratchet cyclomatic complexity (ruff C901) against a baseline"
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Paths to check (default: the same paths as `hatch run lint`)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help=f"Path to the baseline file (default: {DEFAULT_BASELINE_RELPATH})",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Repository root (default: the parent of this script's directory)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="(Re)write the baseline from the current findings and exit 0",
    )
    return parser


def _update_baseline(
    baseline_path: Path,
    current: dict[str, list[int]],
    restricted: bool,
    settings_fingerprint: str,
) -> int:
    """Rewrite ``baseline_path`` from ``current``; returns the exit code."""
    # Refuse to replace a populated baseline with an empty one. Belt and braces
    # behind run_ruff's fail-closed check: any future path that yields zero
    # findings over a tree known to have debt would otherwise disable the gate
    # permanently, with a success message and exit 0.
    if not current and _baseline_is_populated(baseline_path):
        if restricted:
            aprint(
                f"❌ Refusing to overwrite {baseline_path} with an empty "
                "baseline: the RESTRICTED target set produced no findings, so "
                "writing it would drop every baselined key outside those paths. "
                "Re-run --update-baseline over the default targets (no target "
                "arguments) to record the whole tree."
            )
            return 2
        aprint(
            f"❌ Refusing to overwrite {baseline_path} with an empty baseline: "
            "ruff reported zero over-limit functions over a tree that has some "
            "baselined. This almost certainly means ruff did not run correctly. "
            "If the collapse to zero really is intended, delete the baseline "
            "file by hand and re-run."
        )
        return 2
    if restricted:
        aprint(
            "⚠️  Writing the baseline from a RESTRICTED target set. Debt outside "
            "those paths will be DROPPED from the baseline and will resurface as "
            "new findings on the next full run. Re-run without target arguments "
            "to record the whole tree."
        )
    save_baseline(baseline_path, current, settings_fingerprint)
    total = sum(len(v) for v in current.values())
    aprint(
        f"📝 Wrote complexity baseline with {total} over-limit functions "
        f"({len(current)} keys) to {baseline_path}"
    )
    return 0


def _print_regressions(
    report: RatchetReport,
    current: dict[str, list[int]],
    baseline: dict[str, list[int]],
    restricted: bool = False,
) -> None:
    """Print the failing keys, plus the keys that vanished in the same run.

    ``restricted`` carries the same meaning as in ``_print_report``: on a partial
    scan the vanished keys merely went unread, so the closing hint must not
    invite ``--update-baseline`` for them (writing one from a restricted run
    drops the rest of the tree's debt).
    """
    aprint("\n❌ Complexity regressions:\n")
    for key in report.new:
        aprint(f"  ❌ {key}: {_format_counts(current[key])} (not baselined)")
    for key in report.worsened:
        aprint(
            f"  ❌ {key}: {_format_counts(baseline[key])} → "
            f"{_format_counts(current[key])}"
        )

    # A partial move/rename shows up as new keys above AND vanished ones here, so
    # print both sides for a reader to recognise it as one edit (paired moves are
    # itemised by the caller). Only keys that are genuinely GONE qualify:
    # `improved` also holds keys that are still over the limit and merely got
    # simpler, and those did not vanish anywhere.
    gone = [key for key in report.improved if key not in current]
    if gone:
        aprint("\n   Baselined keys that vanished in the same run:\n")
        for key in gone:
            aprint(f"  ✨ {key}")

    hint = (
        "\n⚠️  Act on the ❌ regressions at the TOP of this report: if they are "
        "genuinely new complexity, simplify them (extract helpers, flatten "
        "branches) — do NOT reach for --update-baseline to silence real debt."
    )
    if gone and restricted:
        hint += (
            " The ✨ vanished keys just above are NOT fixed debt — this scan was "
            "restricted, so they simply went unread; --update-baseline would drop "
            "them from the baseline."
        )
    elif gone:
        hint += (
            " The ✨ vanished keys just above are the OPPOSITE — they dropped "
            "below the threshold or were re-keyed (a move/rename); "
            "--update-baseline IS the right response for those."
        )
    aprint(hint)


def _print_report(
    report: RatchetReport,
    current: dict[str, list[int]],
    baseline: dict[str, list[int]],
    restricted: bool = False,
) -> int:
    """Print the human summary of ``report``; returns the exit code.

    ``restricted`` marks a run over explicit target paths rather than the whole
    lint scope: keys outside those paths look vanished, so the full-run failure
    and ``--update-baseline`` remedy are replaced by a caveat (and ``main``
    disables move pairing upstream, so ``moved`` is empty there).
    """
    total = sum(len(v) for v in current.values())

    aprint("=" * 70)
    aprint("📐 COMPLEXITY RATCHET")
    aprint("=" * 70)
    aprint(f"🔢 Functions over the limit: {total}")
    aprint(f"🆕 New: {len(report.new)}")
    aprint(f"📈 Worsened: {len(report.worsened)}")
    aprint(f"🧱 Tolerated (baselined): {len(report.unchanged)}")
    aprint(f"🚚 Moved (same function, new file): {len(report.moved)}")
    aprint(f"✨ Improved: {len(report.improved)}")

    if report.moved:
        # Itemise the move count so it is auditable instead of a bare number.
        aprint(
            "\n   Some baselined functions relocated (total debt never "
            "increased); run --update-baseline to re-key the baseline:\n"
        )
        for entry in report.moved:
            aprint(f"  🚚 {entry}")

    if restricted and report.improved:
        aprint(
            "\n   ⚠️  Targets were RESTRICTED to explicit paths, so baselined "
            "keys outside them merely went unscanned — they were not fixed. Do "
            "NOT run --update-baseline from a restricted run."
        )
    if report.new or report.worsened:
        _print_regressions(report, current, baseline, restricted)
        return 1

    if not restricted and (report.moved or report.improved):
        aprint(
            "\n❌ Baseline no longer matches the current tree. This can follow "
            "your change, a dev merge, or a Ruff update. Run --update-baseline "
            "and commit the regenerated baseline."
        )
        return 1

    aprint("\n✅ No new complexity regressions.")
    return 0


def _refresh_baseline_after_load_error(
    error: ValueError | OSError,
    baseline_path: Path,
    current: dict[str, list[int]],
    targets: tuple[str, ...],
    project_root: Path,
    restricted: bool,
    settings_fingerprint: str,
) -> int:
    """Refresh mismatched baseline metadata without hiding omitted files."""
    previous_baseline = ruff_ratchet.baseline_entries_for_update(
        baseline_path, "functions"
    )
    coverage_error = _existing_baseline_coverage_error_for_update(
        previous_baseline, targets, project_root, restricted
    )
    if coverage_error:
        aprint(f"❌ {coverage_error}")
        return 2
    aprint(
        f"⚠️  {error}\n"
        "   Re-recording the baseline for the current Ruff settings because "
        "--update-baseline was requested deliberately."
    )
    result = _update_baseline(baseline_path, current, restricted, settings_fingerprint)
    if result == 0:
        retired = ruff_ratchet.format_retired_keys(previous_baseline, current)
        if retired:
            aprint(f"⚠️  {retired}")
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the complexity ratchet; returns the process exit code."""
    args = _build_parser().parse_args(argv)

    project_root = (
        Path(args.project_root).resolve() if args.project_root else PROJECT_ROOT
    )
    baseline_path = (
        Path(args.baseline)
        if args.baseline
        else project_root / DEFAULT_BASELINE_RELPATH
    )
    targets = tuple(args.targets) if args.targets else DEFAULT_TARGETS
    restricted = targets != DEFAULT_TARGETS

    try:
        current = parse_findings(run_ruff(targets, project_root), project_root)
        settings_fingerprint = ruff_settings_fingerprint(targets[0], project_root)
        ensure_no_nested_ruff_configs(targets, project_root)
    except (RuntimeError, ValueError) as exc:
        aprint(f"❌ {exc}")
        return 2

    if not baseline_path.exists():
        aprint(
            f"ℹ️  No baseline found at {baseline_path}. Run with "
            "--update-baseline to create one (existing debt will be tolerated)."
        )

    try:
        baseline = load_baseline(baseline_path, settings_fingerprint)
    except (ValueError, OSError) as exc:
        if args.update_baseline:
            return _refresh_baseline_after_load_error(
                exc,
                baseline_path,
                current,
                targets,
                project_root,
                restricted,
                settings_fingerprint,
            )
        aprint(f"❌ {exc}")
        return 2

    if not restricted and baseline:
        try:
            scanned_files = list_ruff_files(targets, project_root)
            ensure_baselined_files_were_scanned(baseline, scanned_files, project_root)
        except RuntimeError as exc:
            aprint(f"❌ {exc}")
            return 2

    if args.update_baseline:
        return _update_baseline(
            baseline_path, current, restricted, settings_fingerprint
        )

    # Zero findings against a populated baseline. Over the DEFAULT targets that
    # can only mean the scan covered nothing (ruff signals a mistyped path or an
    # empty directory with exit 0, `[]` and a stderr warning), and without this
    # the gating path would report every baselined key as fixed and exit GREEN.
    # Over explicit targets it is unremarkable — a subtree may simply be clean —
    # so it warns instead; a typo stays diagnosable through ruff's echoed stderr.
    if not current and baseline:
        if not restricted:
            aprint(
                f"❌ ruff reported zero over-limit functions, but {baseline_path} "
                f"baselines {len(baseline)}. The scan covered nothing — check "
                "for a mistyped target path, a wrong --project-root, a partial "
                "checkout, or a ruff that did not run. Any stderr warning is "
                "echoed above."
            )
            return 2
        aprint(
            f"⚠️  No over-limit functions under the restricted target set, while "
            f"{baseline_path} baselines {len(baseline)}. Expected if the scanned "
            "paths are clean — but check any ruff stderr warning above for a "
            "path that did not match."
        )

    # Move pairing is disabled on a partial scan: every unscanned baseline key
    # looks vanished, and would otherwise be an eligible pairing candidate for a
    # genuinely new function (a false green on a documented workflow).
    report = evaluate_ratchet(current, baseline, pair_moves=not restricted)
    return _print_report(report, current, baseline, restricted)


if __name__ == "__main__":
    sys.exit(main())
