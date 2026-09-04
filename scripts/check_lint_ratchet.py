#!/usr/bin/env python3
"""
Defect-Rule Lint Ratchet

Enforces ruff's ``flake8-bugbear`` (``B``) family plus ``RUF012`` as a
baseline-driven *ratchet*: pre-existing violations are tolerated via a
checked-in baseline (``scripts/lint_baseline.json``), but a NEW violation — or
an extra one in an already-baselined file — fails the check. Regenerate the
baseline with ``--update-baseline``; paid-down debt is reported as advisory
(exit 0) so the baseline can be tightened the same way.

WHY THESE RULES
---------------
They are the *defect-bearing* ones, not the cosmetic ones. Each describes a way
working-looking code is silently wrong:

===========  ====================================================================
``B006``     A mutable default argument is shared across every call.
``B007``     A loop control variable that is never used (usually a stale rename).
``B008``     A function call evaluated once, at import, as a default.
``B023``     A closure that captures a loop variable by reference, not by value.
``B028``     ``warnings.warn`` without ``stacklevel``, so it blames the wrong line.
``B034``     ``re.split``/``re.sub`` positional ``maxsplit``/``count`` — also a
             ``DeprecationWarning`` from Python 3.13 on.
``B904``     ``raise`` inside ``except`` without ``from``, which drops the cause.
``B905``     ``zip()`` without ``strict=``, which silently truncates to the
             shortest input.
``RUF012``   A mutable class attribute shared by every instance.
===========  ====================================================================

The rest of ruff's catalogue that this repository already enforces lives in
``[tool.ruff.lint] select`` and is checked by ``hatch run lint`` directly.

WHY A SCRIPT INSTEAD OF ``[tool.ruff.lint] select``
---------------------------------------------------
The same reason ``C901`` is ratcheted by ``scripts/check_complexity.py``: ruff
has no baseline mechanism. A bare ``select = ["B", "RUF012"]`` would fail on all
pre-existing violations (473 at the time of writing, 290 of them ``B905``), so
it could not be turned on at all without a large, unrelated, and — for ``B905``
specifically — *behaviour-changing* sweep: ``strict=True`` RAISES on mismatched
lengths, so it is a decision per call site, not a mechanical edit. The only
ruff-native suppression is ``per-file-ignores``, which is *file*-granular and
would blind the guard to brand-new violations inside the 200-odd files that
already hold one.

Note what this gate does NOT need to tolerate: ``B008`` sits at zero, because
``[tool.ruff.lint.flake8-bugbear] extend-immutable-calls`` in ``pyproject.toml``
declares ``typer.Option``/``typer.Argument`` immutable. All 83 hits were that
Typer idiom, which is not a defect. Remove that setting and this gate goes red —
which is the correct response, not a bug.

A ``# noqa: <code>`` with a rationale is a legitimate way to shrink the
baseline: ruff honours it, so the finding disappears and the entry can be
dropped. Prefer it to ``--update-baseline`` when the rule is a false positive at
that specific site.

Usage:
    python scripts/check_lint_ratchet.py
    python scripts/check_lint_ratchet.py --update-baseline
    python scripts/check_lint_ratchet.py --baseline path/to/baseline.json
    python scripts/check_lint_ratchet.py packages/luxar/src   # explicit targets
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from arbol import aprint

# Repository root (this script lives in `<root>/scripts/`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Lint targets, kept in sync with the `lint` script in pyproject.toml and with
# scripts/check_complexity.py.
DEFAULT_TARGETS: tuple[str, ...] = (
    "packages/luxar/src",
    "scripts",
    "scripts/benchmarks",
    "stats",
    "hatch_build.py",
)

# The ratcheted rule selection, passed to `ruff --select`. Recorded IN the
# baseline and compared on every run: dropping a rule from here would otherwise
# make every one of its findings look like paid-down debt, quietly retiring the
# rule with a green tick and an "improved" message.
RATCHETED_SELECT: tuple[str, ...] = ("B", "RUF012")

# Baseline path relative to the project root.
DEFAULT_BASELINE_RELPATH = "scripts/lint_baseline.json"

BASELINE_COMMENT = (
    "Defect-rule lint baseline for scripts/check_lint_ratchet.py. Regenerate "
    "with: hatch run check-lint-ratchet --update-baseline. Each key is "
    "'<repo-relative-path>::<ruff code>' and maps to the NUMBER of violations "
    "of that code in that file. A key absent from here, or one whose count "
    "exceeds its baselined value, fails the check. 'rules' records the "
    "--select used; a mismatch fails rather than silently retiring a rule."
)

# A finding this gate is allowed to see. Anything else (notably
# `invalid-syntax`, which ruff emits whatever is selected with a code outside
# the selection) means the scan did not read the tree it claims to have read.

# ruff's stderr line for a path it could not read at all, e.g.
# "warning: Failed to lint stats: No such file or directory (os error 2)".
_UNSCANNED_RE = re.compile(r"Failed to lint ")


def is_ratcheted_code(code: str) -> bool:
    """Return whether ``code`` is selected by ``RATCHETED_SELECT``."""
    for selector in RATCHETED_SELECT:
        if selector[-1].isdigit():
            if code == selector:
                return True
        elif re.fullmatch(rf"{re.escape(selector)}\d+", code):
            return True
    return False


@dataclass
class RatchetReport:
    """Classification of the current findings against the baseline.

    ``new`` and ``worsened`` are the FAILING sets; ``improved`` is advisory
    (debt paid down — regenerate the baseline to lock it in).
    """

    new: list[str] = field(default_factory=list)
    worsened: list[str] = field(default_factory=list)
    improved: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ruff invocation and parsing
# ---------------------------------------------------------------------------


def run_ruff(targets: tuple[str, ...] | list[str], project_root: Path) -> str:
    """Run ``ruff check --select B,RUF012`` over ``targets`` and return stdout.

    ruff exits 1 when it reports findings, which is the NORMAL case here; only
    other exit codes (or a missing ``ruff`` module) are treated as hard errors.

    Three fail-CLOSED guards, each covering a way a broken scan reads as a clean
    one (the same three ``scripts/check_complexity.py`` documents, for the same
    reasons):

    1. ``python -m ruff`` with ruff not installed also exits 1, but with EMPTY
       stdout and the error on stderr. Left unchecked that reads as "no findings
       at all", every baselined key looks fixed, and the checker reports green.
       A real zero-findings run prints ``[]``, never blank.
    2. A path ruff could NOT read (missing, unreadable, a dangling symlink) is
       only a stderr ``Failed to lint <path>`` WARNING; ruff lints the remaining
       targets and exits exactly as if the scan had been complete. The baselined
       keys under the unread path then look fixed. So such a warning is a hard
       error here, whatever the exit code.
    3. Any other stderr output is echoed rather than discarded — notably ``No
       Python files found under the given path(s)``, a whole-scan miss the
       caller catches separately in ``main``.
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

    empty_failure = proc.returncode == 1 and not proc.stdout.strip()
    if proc.returncode not in (0, 1) or empty_failure:
        detail = " but reported nothing" if empty_failure else ""
        raise RuntimeError(
            f"ruff failed (exit {proc.returncode}{detail}): {' '.join(command)}\n"
            f"{proc.stderr.strip()}"
        )
    stderr = proc.stderr.strip()

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


def list_ruff_files(
    targets: tuple[str, ...] | list[str], project_root: Path
) -> set[str]:
    """Return the repo-relative files ruff says it will scan."""
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--show-files",
        "--select",
        ",".join(RATCHETED_SELECT),
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

    if proc.returncode != 0:
        raise RuntimeError(
            f"ruff --show-files failed (exit {proc.returncode}): "
            f"{' '.join(command)}\n{proc.stderr.strip()}"
        )

    unscanned = [
        line for line in proc.stderr.splitlines() if _UNSCANNED_RE.search(line)
    ]
    if unscanned:
        raise RuntimeError(
            "ruff could not enumerate every target, so the scan is PARTIAL:\n"
            + "\n".join(f"  {line}" for line in unscanned)
        )

    files: set[str] = set()
    for line in proc.stdout.splitlines():
        path = Path(line)
        try:
            files.add(path.relative_to(project_root).as_posix())
        except ValueError:
            files.add(path.as_posix())
    return files


def ensure_baselined_files_were_scanned(
    baseline: dict[str, int], scanned_files: set[str], project_root: Path
) -> None:
    """Fail when an existing baselined file was silently excluded by ruff."""
    omitted = sorted(
        path
        for path in {key.rpartition("::")[0] for key in baseline}
        if (project_root / path).exists() and path not in scanned_files
    )
    if omitted:
        shown = "\n".join(f"  {path}" for path in omitted[:20])
        suffix = f"\n  ... and {len(omitted) - 20} more" if len(omitted) > 20 else ""
        raise RuntimeError(
            "ruff omitted existing baselined files, so the scan is PARTIAL. "
            "Check Ruff excludes and the checkout:\n"
            f"{shown}{suffix}"
        )


def parse_findings(stdout: str, project_root: Path = PROJECT_ROOT) -> dict[str, int]:
    r"""Parse ruff's JSON output into ``{"<rel-path>::<CODE>": count}``.

    Keys use repo-relative POSIX paths so the baseline is stable across
    machines, and deliberately carry NO line number: an unrelated edit above a
    violation must not churn the baseline. Values are plain counts — unlike
    complexity there is no per-finding magnitude to compare, so "how many of
    this rule does this file break" is the whole state.

    A finding whose code is outside ``RATCHETED_SELECT`` is a hard error.
    ruff emits ``invalid-syntax`` diagnostics regardless of ``--select``, and a
    file it could not parse is a file it did not lint — which would otherwise
    look like that file's baselined debt had been paid off.
    """
    if not stdout.strip():
        return {}

    try:
        findings = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse ruff JSON output: {exc}") from exc

    entries: dict[str, int] = {}
    for finding in findings:
        code = finding.get("code")
        path = Path(finding["filename"])
        try:
            relative = path.relative_to(project_root).as_posix()
        except ValueError:
            relative = path.as_posix()

        if code is None or not is_ratcheted_code(code):
            location = finding.get("location") or {}
            raise ValueError(
                f"ruff reported a {code!r} diagnostic at {relative}:"
                f"{location.get('row', '?')} — {finding.get('message', '')!r}. "
                "That is not one of the ratcheted rules, so the scan cannot be "
                "diffed against the baseline (a file ruff could not parse is a "
                "file it did not lint). Fix the file, then re-run."
            )

        key = f"{relative}::{code}"
        entries[key] = entries.get(key, 0) + 1

    return entries


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------


def load_baseline(path: Path) -> dict[str, int]:
    """Load the baselined violation counts from ``path``.

    Returns ``{}`` if the file does not exist. Raises a clear ``ValueError`` if
    it exists but is malformed, or if its recorded ``rules`` disagree with
    ``RATCHETED_SELECT`` — a mismatch means the baseline describes a different
    question from the one this run asked, so the two cannot be compared.
    """
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Baseline file {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or "violations" not in data:
        raise ValueError(
            f"Baseline file {path} is malformed: expected a JSON object with a "
            "'violations' object."
        )

    recorded = data.get("rules")
    expected = list(RATCHETED_SELECT)
    if recorded != expected:
        raise ValueError(
            f"Baseline file {path} was recorded for rules {recorded!r}, but this "
            f"run selects {expected!r}. Counts for a rule that is no longer "
            "selected would look like paid-down debt, so the rule would retire "
            "itself with a green tick. Re-run --update-baseline deliberately if "
            "the selection change is intended."
        )

    violations = data["violations"]
    if not isinstance(violations, dict):
        raise ValueError(
            f"Baseline file {path} is malformed: 'violations' must be an object "
            "mapping '<path>::<code>' to a positive integer count."
        )

    for key, value in violations.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(
                f"Baseline file {path} is malformed: entry {key!r} must be a "
                "positive integer count."
            )

    return dict(violations)


def save_baseline(path: Path, entries: dict[str, int]) -> None:
    """Write ``entries`` to ``path`` as deterministic, human-diffable JSON."""
    payload = {
        "_comment": BASELINE_COMMENT,
        "rules": list(RATCHETED_SELECT),
        "violations": {key: entries[key] for key in sorted(entries)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Ratchet
# ---------------------------------------------------------------------------


def evaluate_ratchet(
    current: dict[str, int], baseline: dict[str, int]
) -> RatchetReport:
    """Classify ``current`` findings against ``baseline``.

    - ``new``: a key absent from the baseline (a file that newly breaks a rule).
    - ``worsened``: a baselined key with a HIGHER count.
    - ``improved``: a baselined key with a lower count, or one that vanished.
    - ``unchanged``: the rest (tolerated pre-existing debt).

    There is deliberately NO move pairing, unlike ``check_complexity.py``. That
    checker pairs a vanished key with a new one naming the same function so a
    relocation does not read as new debt; the cost is that a genuinely new
    over-complex function can be absorbed by a same-named one that vanished in
    the same run — a false GREEN, bounded only by "total debt never increased".
    Here the identity available is a bare ``(file, rule)`` pair, which is far
    weaker evidence of "the same violation, relocated" than a function name. So
    a file move fails as ``new`` and is fixed by ``--update-baseline``: a false
    RED that names the file, rather than a false green that hides one.
    """
    report = RatchetReport()

    for key in sorted(current):
        count = current[key]
        if key not in baseline:
            report.new.append(key)
        elif count > baseline[key]:
            report.worsened.append(key)
        elif count < baseline[key]:
            report.improved.append(key)
        else:
            report.unchanged.append(key)

    report.improved.extend(sorted(set(baseline) - set(current)))
    report.improved.sort()
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        description=(
            "Ratchet ruff's defect-bearing rules "
            f"({', '.join(RATCHETED_SELECT)}) against a baseline"
        )
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
    baseline_path: Path, current: dict[str, int], restricted: bool
) -> int:
    """Rewrite ``baseline_path`` from ``current``; returns the exit code."""
    if restricted:
        aprint(
            f"❌ Refusing to overwrite {baseline_path} from a RESTRICTED target "
            "set: writing a partial scan would drop every baselined key outside "
            "those paths. Re-run --update-baseline without target arguments to "
            "record the whole tree."
        )
        return 2

    # Refuse to replace a populated baseline with an empty one. Belt and braces
    # behind run_ruff's fail-closed checks: any future path that yields zero
    # findings over a tree known to have debt would otherwise disable the gate
    # permanently, with a success message and exit 0.
    if not current and _baseline_is_populated(baseline_path):
        aprint(
            f"❌ Refusing to overwrite {baseline_path} with an empty baseline: "
            "ruff reported zero violations over a tree that has some baselined. "
            "This almost certainly means ruff did not run correctly. If the "
            "collapse to zero really is intended, delete the baseline file by "
            "hand and re-run."
        )
        return 2
    save_baseline(baseline_path, current)
    total = sum(current.values())
    aprint(
        f"📝 Wrote lint baseline with {total} violations across "
        f"{len(current)} file/rule keys to {baseline_path}"
    )
    return 0


def _by_rule(keys: list[str], values: dict[str, int]) -> dict[str, int]:
    """Total the violations in ``keys`` per ruff code, for the summary line."""
    totals: dict[str, int] = {}
    for key in keys:
        code = key.rpartition("::")[2]
        totals[code] = totals.get(code, 0) + values[key]
    return dict(sorted(totals.items()))


def _print_regressions(
    report: RatchetReport,
    current: dict[str, int],
    baseline: dict[str, int],
    restricted: bool = False,
) -> None:
    """Print the failing keys, plus the keys that vanished in the same run."""
    aprint("\n❌ Lint regressions:\n")
    for key in report.new:
        aprint(f"  ❌ {key}: {current[key]} (not baselined)")
    for key in report.worsened:
        aprint(f"  ❌ {key}: {baseline[key]} → {current[key]}")

    # A file move shows up as new keys above AND vanished ones here, so print
    # both sides for a reader to recognise it as one edit. Only keys that are
    # genuinely GONE qualify: `improved` also holds keys whose count merely fell.
    gone = [key for key in report.improved if key not in current]
    if gone:
        aprint("\n   Baselined keys that vanished in the same run:\n")
        for key in gone:
            aprint(f"  ✨ {key}")

    aprint(
        "\n⚠️  Act on the ❌ regressions at the TOP of this report. Each names a "
        "rule documented in scripts/check_lint_ratchet.py — fix the code, or "
        "add a `# noqa: <code>` WITH a rationale if it is a false positive at "
        "that site. Do NOT reach for --update-baseline to silence real debt."
    )
    if gone and restricted:
        aprint(
            "   The ✨ vanished keys are NOT fixed debt — this scan was "
            "restricted, so they simply went unread; --update-baseline would "
            "drop them from the baseline."
        )
    elif gone:
        aprint(
            "   The ✨ vanished keys are the OPPOSITE — they were fixed or "
            "re-keyed (a file move); --update-baseline IS right for those."
        )


def _print_report(
    report: RatchetReport,
    current: dict[str, int],
    baseline: dict[str, int],
    restricted: bool = False,
) -> int:
    """Print the human summary of ``report``; returns the exit code.

    ``restricted`` marks a run over explicit target paths rather than the whole
    lint scope: keys outside those paths look vanished, so the advisory that
    would otherwise invite ``--update-baseline`` is replaced by a caveat.
    """
    total = sum(current.values())

    aprint("=" * 70)
    aprint("🐛 DEFECT-RULE LINT RATCHET")
    aprint("=" * 70)
    aprint(f"🔢 Violations: {total} across {len(current)} file/rule keys")
    per_rule = _by_rule(list(current), current)
    if per_rule:
        aprint("   " + "  ".join(f"{code}={n}" for code, n in per_rule.items()))
    aprint(f"🆕 New: {len(report.new)}")
    aprint(f"📈 Worsened: {len(report.worsened)}")
    aprint(f"🧱 Tolerated (baselined): {len(report.unchanged)}")
    aprint(f"✨ Improved: {len(report.improved)}")

    if restricted and report.improved:
        aprint(
            "\n   ⚠️  Targets were RESTRICTED to explicit paths, so baselined "
            "keys outside them merely went unscanned — they were not fixed. Do "
            "NOT run --update-baseline from a restricted run."
        )
    elif report.improved:
        aprint(
            "\n   Nice — some lint debt was paid down. Run --update-baseline to "
            "tighten the baseline so it can't come back."
        )

    if report.new or report.worsened:
        _print_regressions(report, current, baseline, restricted)
        return 1

    aprint("\n✅ No new lint regressions.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the lint ratchet; returns the process exit code."""
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
    except (RuntimeError, ValueError) as exc:
        aprint(f"❌ {exc}")
        return 2

    if not baseline_path.exists():
        aprint(
            f"ℹ️  No baseline found at {baseline_path}. Run with "
            "--update-baseline to create one (existing debt will be tolerated)."
        )

    try:
        baseline = load_baseline(baseline_path)
    except (ValueError, OSError) as exc:
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
        return _update_baseline(baseline_path, current, restricted)

    # Zero findings against a populated baseline. Over the DEFAULT targets that
    # can only mean the scan covered nothing (ruff signals a mistyped path or an
    # empty directory with exit 0, `[]` and a stderr warning), and without this
    # the gating path would report every baselined key as fixed and exit GREEN.
    # Over explicit targets it is unremarkable — a subtree may simply be clean —
    # so it warns instead; a typo stays diagnosable through ruff's echoed stderr.
    if not current and baseline:
        if not restricted:
            aprint(
                f"❌ ruff reported zero violations, but {baseline_path} baselines "
                f"{len(baseline)} keys. The scan covered nothing — check for a "
                "mistyped target path, a wrong --project-root, a partial "
                "checkout, or a ruff that did not run. Any stderr warning is "
                "echoed above."
            )
            return 2
        aprint(
            f"⚠️  No violations under the restricted target set, while "
            f"{baseline_path} baselines {len(baseline)}. Expected if the scanned "
            "paths are clean — but check any ruff stderr warning above for a "
            "path that did not match."
        )

    report = evaluate_ratchet(current, baseline)
    return _print_report(report, current, baseline, restricted)


if __name__ == "__main__":
    sys.exit(main())
