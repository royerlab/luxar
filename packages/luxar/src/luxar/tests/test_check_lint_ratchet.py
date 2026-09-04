"""Tests for the defect-rule lint ratchet (``scripts/check_lint_ratchet.py``).

The gate is only worth having if it has been SEEN to fail. Three arms are
exercised end to end against real ruff runs over a temporary tree — a brand-new
violation, an extra violation in an already-baselined file, and a paid-down one
— plus every fail-closed guard that could turn a broken scan into a green tick.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _load_checker() -> ModuleType:
    """Import ``scripts/check_lint_ratchet.py`` as a module by file path."""
    script_path = PROJECT_ROOT / "scripts/check_lint_ratchet.py"
    spec = importlib.util.spec_from_file_location("check_lint_ratchet", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before executing: `@dataclass` resolves its own module through
    # ``sys.modules`` and raises AttributeError if the module is absent.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _finding(filename: str, code: str, row: int = 1) -> dict[str, object]:
    """Build one synthetic ruff finding."""
    return {
        "code": code,
        "filename": str(PROJECT_ROOT / filename),
        "message": f"synthetic {code}",
        "location": {"row": row, "column": 1},
    }


# ---------------------------------------------------------------------------
# run_ruff — the fail-closed detection logic
# ---------------------------------------------------------------------------


def _stub_ruff(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str, stderr: str = ""
) -> None:
    """Replace ``subprocess.run`` inside the checker with a canned ruff result."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Stand-in for a ruff invocation with a fixed outcome."""
        return subprocess.CompletedProcess([], returncode, stdout, stderr)

    monkeypatch.setattr(checker.subprocess, "run", fake_run)


def test_run_ruff_accepts_findings_and_a_clean_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exit 1 with a report, and exit 0 with `[]`, are both normal outcomes."""
    _stub_ruff(monkeypatch, 1, "[]")
    assert checker.run_ruff(("x",), PROJECT_ROOT) == "[]"

    _stub_ruff(monkeypatch, 0, "[]")
    assert checker.run_ruff(("x",), PROJECT_ROOT) == "[]"


def test_run_ruff_rejects_exit_1_with_no_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`No module named ruff` exits 1 with blank stdout — it must NOT read green."""
    _stub_ruff(monkeypatch, 1, "", "No module named ruff")

    with pytest.raises(RuntimeError, match="but reported nothing"):
        checker.run_ruff(("x",), PROJECT_ROOT)


def test_run_ruff_rejects_an_unexpected_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anything other than 0/1 is a ruff failure, not an empty result."""
    _stub_ruff(monkeypatch, 2, "", "error: unexpected argument")

    with pytest.raises(RuntimeError, match="ruff failed"):
        checker.run_ruff(("x",), PROJECT_ROOT)


def test_run_ruff_rejects_a_partial_scan_despite_a_normal_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unread target is a WARNING to ruff and a hard error here.

    ruff lints whatever it could read and exits 0, so the baselined keys under
    the unread path look fixed. That is the fail-open this guard closes — note
    the exit code below is the *success* one.
    """
    _stub_ruff(
        monkeypatch,
        0,
        "[]",
        "warning: Failed to lint stats: No such file or directory (os error 2)",
    )

    with pytest.raises(RuntimeError, match="PARTIAL"):
        checker.run_ruff(("stats",), PROJECT_ROOT)


def test_run_ruff_echoes_other_stderr_without_failing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-fatal warning is surfaced, not swallowed — `main` needs to see it."""
    _stub_ruff(monkeypatch, 0, "[]", "warning: No Python files found")

    assert checker.run_ruff(("x",), PROJECT_ROOT) == "[]"
    assert "No Python files found" in _ANSI_ESCAPE.sub("", capsys.readouterr().out)


# ---------------------------------------------------------------------------
# parse_findings
# ---------------------------------------------------------------------------


def test_parse_findings_counts_per_file_and_rule() -> None:
    """Keys are `<rel-path>::<CODE>` and values are plain occurrence counts."""
    stdout = json.dumps(
        [
            _finding("a.py", "B905", row=1),
            _finding("a.py", "B905", row=9),
            _finding("a.py", "B904"),
            _finding("b/c.py", "RUF012"),
        ]
    )

    assert checker.parse_findings(stdout, PROJECT_ROOT) == {
        "a.py::B905": 2,
        "a.py::B904": 1,
        "b/c.py::RUF012": 1,
    }


def test_parse_findings_is_empty_for_a_clean_run() -> None:
    """`[]` and blank both mean "nothing found" without raising."""
    assert checker.parse_findings("[]", PROJECT_ROOT) == {}
    assert checker.parse_findings("   ", PROJECT_ROOT) == {}


@pytest.mark.parametrize(
    "code",
    [
        # What ruff 0.16 actually emits, verified against a real run rather than
        # assumed: the code is the STRING "invalid-syntax", not a null.
        pytest.param("invalid-syntax", id="ruff-0.16-string-code"),
        # Kept as a second arm because the JSON schema does allow a null code,
        # and the guard must not depend on which of the two a ruff version picks.
        pytest.param(None, id="null-code"),
    ],
)
def test_parse_findings_rejects_an_unparseable_file(code: str | None) -> None:
    """A syntax error means ruff did NOT lint that file — fail, don't ignore.

    ruff reports it whatever ``--select`` says. Skipping it would let a file
    that could not be parsed look like a file whose baselined debt had been paid
    off — the scan silently covers less of the tree than it claims to.
    """
    stdout = json.dumps(
        [
            {
                "code": code,
                "filename": "x.py",
                "message": "Expected a parameter or the end of the parameter list",
                "location": {"row": 1, "column": 12},
            }
        ]
    )

    with pytest.raises(ValueError, match="not one of the ratcheted rules"):
        checker.parse_findings(stdout, PROJECT_ROOT)


def test_parse_findings_rejects_a_rule_outside_the_selection() -> None:
    """A code the selection cannot produce means the command drifted."""
    stdout = json.dumps([_finding("a.py", "E501")])

    with pytest.raises(ValueError, match="E501"):
        checker.parse_findings(stdout, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------


def _baseline_payload(violations: dict[str, int], rules: list[str] | None = None):
    """Build a baseline document, defaulting to the checker's own rule list."""
    return {
        "rules": list(checker.RATCHETED_SELECT) if rules is None else rules,
        "violations": violations,
    }


def test_load_baseline_missing_file_is_empty(tmp_path: Path) -> None:
    """An absent baseline is empty, not an error — `main` reports it separately."""
    assert checker.load_baseline(tmp_path / "nope.json") == {}


def test_load_baseline_rejects_a_baseline_recorded_for_other_rules(
    tmp_path: Path,
) -> None:
    """Dropping a rule from the selection must not read as paid-down debt.

    This is the anti-decay guard specific to a MULTI-rule ratchet: shrink
    ``RATCHETED_SELECT`` and every finding of the dropped rule vanishes from the
    scan, so the ratchet reports "improved", exits 0, and the rule retires
    itself with a green tick and a compliment.
    """
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(_baseline_payload({"a.py::B905": 1}, rules=["B"])))

    with pytest.raises(ValueError, match="retire itself"):
        checker.load_baseline(path)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"rules": ["B", "RUF012"]}, id="no-violations-key"),
        pytest.param(_baseline_payload({"a.py::B905": 0}), id="zero-count"),
        pytest.param(_baseline_payload({"a.py::B905": True}), id="bool-count"),
        pytest.param(_baseline_payload({"a.py::B905": "2"}), id="string-count"),
        pytest.param({"rules": ["B", "RUF012"], "violations": []}, id="list-not-dict"),
    ],
)
def test_load_baseline_raises_on_a_malformed_file(
    tmp_path: Path, payload: object
) -> None:
    """A malformed baseline is an error, never a silently empty one."""
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="malformed"):
        checker.load_baseline(path)


def test_load_baseline_raises_on_invalid_json(tmp_path: Path) -> None:
    """A truncated file must not be read as "no debt"."""
    path = tmp_path / "baseline.json"
    path.write_text("{not json")

    with pytest.raises(ValueError, match="not valid JSON"):
        checker.load_baseline(path)


def test_save_baseline_round_trips_and_is_deterministic(tmp_path: Path) -> None:
    """Written keys are sorted, the rule list is recorded, and re-writing is stable."""
    path = tmp_path / "baseline.json"
    entries = {"z.py::B905": 2, "a.py::B904": 1}

    checker.save_baseline(path, entries)
    first = path.read_text()
    assert checker.load_baseline(path) == entries

    checker.save_baseline(path, entries)
    assert path.read_text() == first

    document = json.loads(first)
    assert document["rules"] == list(checker.RATCHETED_SELECT)
    assert list(document["violations"]) == ["a.py::B904", "z.py::B905"]
    assert first.endswith("\n")


def test_baseline_is_populated_never_reads_a_bad_file_as_empty(
    tmp_path: Path,
) -> None:
    """An unreadable baseline counts as populated, so it cannot be overwritten."""
    path = tmp_path / "baseline.json"
    path.write_text("{not json")

    assert checker._baseline_is_populated(path) is True


# ---------------------------------------------------------------------------
# evaluate_ratchet
# ---------------------------------------------------------------------------


def test_evaluate_ratchet_classifies_every_transition() -> None:
    """New, worsened, improved, vanished and unchanged each land in one bucket."""
    current = {
        "new.py::B905": 1,
        "worse.py::B904": 5,
        "better.py::B905": 1,
        "same.py::RUF012": 3,
    }
    baseline = {
        "worse.py::B904": 4,
        "better.py::B905": 2,
        "same.py::RUF012": 3,
        "gone.py::B028": 1,
    }

    report = checker.evaluate_ratchet(current, baseline)

    assert report.new == ["new.py::B905"]
    assert report.worsened == ["worse.py::B904"]
    assert report.improved == ["better.py::B905", "gone.py::B028"]
    assert report.unchanged == ["same.py::RUF012"]


def test_evaluate_ratchet_does_not_pair_a_file_move() -> None:
    """A move fails as `new` — deliberately, unlike the complexity ratchet.

    ``check_complexity.py`` pairs a vanished key with a new one of the same
    FUNCTION NAME. Here the only identity available is ``(file, rule)``, which is
    far too weak: pairing on it would let a genuinely new violation be absorbed
    by any unrelated file that lost one of the same rule in the same run. So a
    move is a false RED that names the file, not a false green that hides one.
    """
    report = checker.evaluate_ratchet({"new/loc.py::B905": 2}, {"old/loc.py::B905": 2})

    assert report.new == ["new/loc.py::B905"]
    assert report.improved == ["old/loc.py::B905"]


def test_evaluate_ratchet_is_clean_against_its_own_findings() -> None:
    """The identity case: current == baseline yields no regressions at all."""
    entries = {"a.py::B905": 7, "b.py::B904": 1}

    report = checker.evaluate_ratchet(entries, dict(entries))

    assert not report.new and not report.worsened and not report.improved
    assert report.unchanged == ["a.py::B905", "b.py::B904"]


def test_evaluate_ratchet_empty_baseline_makes_everything_new() -> None:
    """A deleted baseline must not read as "everything is fine"."""
    report = checker.evaluate_ratchet({"a.py::B905": 1, "b.py::B904": 1}, {})

    assert report.new == ["a.py::B905", "b.py::B904"]


# ---------------------------------------------------------------------------
# main() — the CLI gate itself, driven against real ruff runs
# ---------------------------------------------------------------------------

_CLEAN_MODULE = '"""Fixture module."""\n\n\ndef total(n: int) -> int:\n    return n\n'

_ONE_VIOLATION = (
    '"""Fixture module."""\n\n\n'
    "def pairs(a: list[int], b: list[int]) -> list[tuple[int, int]]:\n"
    "    return list(zip(a, b))\n"
)

_TWO_VIOLATIONS = _ONE_VIOLATION + (
    "\n\ndef more(a: list[int], b: list[int]) -> list[tuple[int, int]]:\n"
    "    return list(zip(b, a))\n"
)


def _write_tree(tmp_path: Path, source: str) -> None:
    """Write the single-module fixture tree under ``tmp_path``."""
    (tmp_path / "sample.py").write_text(source)


def _run_main(tmp_path: Path, baseline: Path, *extra: str) -> int:
    """Drive ``checker.main`` over ``tmp_path`` with an isolated baseline."""
    return checker.main(
        [
            "sample.py",
            "--project-root",
            str(tmp_path),
            "--baseline",
            str(baseline),
            *extra,
        ]
    )


def _clean_output(capsys: pytest.CaptureFixture[str]) -> str:
    """Captured stdout with arbol's ANSI colour codes stripped."""
    return _ANSI_ESCAPE.sub("", capsys.readouterr().out)


def _unrestrict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fixture target count as the FULL scope, not a restricted run.

    ``_run_main`` always passes an explicit target, which the checker rightly
    treats as restricted; this pins ``DEFAULT_TARGETS`` to the same value so the
    unrestricted reporting path can be exercised too.
    """
    monkeypatch.setattr(checker, "DEFAULT_TARGETS", ("sample.py",))


def _require_ruff() -> None:
    """Skip when ruff is absent; these arms need a real scan to mean anything."""
    pytest.importorskip("ruff", reason="ruff is not installed in this environment")


def test_main_fails_on_a_violation_that_is_not_baselined(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ARM 1 — a brand-new violation exits 1 and names the file and rule."""
    _require_ruff()
    _write_tree(tmp_path, _ONE_VIOLATION)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({})))

    assert _run_main(tmp_path, baseline) == 1

    output = _clean_output(capsys)
    assert "sample.py::B905" in output
    assert "not baselined" in output


def test_main_fails_when_a_baselined_file_gains_a_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ARM 2 — tolerating one violation must not tolerate a second."""
    _require_ruff()
    _write_tree(tmp_path, _TWO_VIOLATIONS)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({"sample.py::B905": 1})))

    assert _run_main(tmp_path, baseline) == 1
    assert "sample.py::B905: 1 → 2" in _clean_output(capsys)


def test_main_reports_a_fix_as_advisory_and_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """ARM 3 — paid-down debt is green, and says the baseline can be tightened."""
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, _ONE_VIOLATION)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({"sample.py::B905": 2})))

    assert _run_main(tmp_path, baseline) == 0

    output = _clean_output(capsys)
    assert "Improved: 1" in output
    assert "--update-baseline" in output


def test_main_passes_when_the_violation_is_baselined(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pre-existing debt at its recorded count is tolerated."""
    _require_ruff()
    _write_tree(tmp_path, _ONE_VIOLATION)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({"sample.py::B905": 1})))

    assert _run_main(tmp_path, baseline) == 0
    assert "No new lint regressions" in _clean_output(capsys)


def test_main_update_baseline_records_the_current_counts(tmp_path: Path) -> None:
    """`--update-baseline` writes exactly what the scan found, plus the rules."""
    _require_ruff()
    _write_tree(tmp_path, _TWO_VIOLATIONS)
    baseline = tmp_path / "baseline.json"

    assert _run_main(tmp_path, baseline, "--update-baseline") == 0

    document = json.loads(baseline.read_text())
    assert document["violations"] == {"sample.py::B905": 2}
    assert document["rules"] == list(checker.RATCHETED_SELECT)


def test_main_refuses_to_wipe_a_populated_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-finding full scan must not silently empty the baseline."""
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, _CLEAN_MODULE)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({"other.py::B905": 4})))

    assert _run_main(tmp_path, baseline, "--update-baseline") == 2
    assert "Refusing to overwrite" in _clean_output(capsys)
    assert json.loads(baseline.read_text())["violations"] == {"other.py::B905": 4}


def test_main_refusal_diagnoses_a_clean_restricted_update(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The restricted refusal explains that debt OUTSIDE the paths would be lost."""
    _require_ruff()
    _write_tree(tmp_path, _CLEAN_MODULE)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({"other.py::B905": 4})))

    assert _run_main(tmp_path, baseline, "--update-baseline") == 2
    assert "RESTRICTED" in _clean_output(capsys)


def test_main_restricted_update_baseline_warns_about_dropped_debt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Writing from a restricted scan succeeds but says what it just dropped."""
    _require_ruff()
    _write_tree(tmp_path, _ONE_VIOLATION)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({"other.py::B905": 4})))

    assert _run_main(tmp_path, baseline, "--update-baseline") == 0

    output = _clean_output(capsys)
    assert "RESTRICTED" in output and "DROPPED" in output
    assert json.loads(baseline.read_text())["violations"] == {"sample.py::B905": 1}


def test_main_fails_closed_when_a_full_scan_covers_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero findings over the FULL scope, against a populated baseline, is exit 2.

    Without this the gating path reports every baselined key as fixed and exits
    green — the loudest possible fail-open.
    """
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, _CLEAN_MODULE)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({"other.py::B905": 4})))

    assert _run_main(tmp_path, baseline) == 2
    assert "covered nothing" in _clean_output(capsys)


def test_main_only_warns_when_a_restricted_scan_is_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A clean subtree is unremarkable on a restricted run — warn, do not fail."""
    _require_ruff()
    _write_tree(tmp_path, _CLEAN_MODULE)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({"other.py::B905": 4})))

    assert _run_main(tmp_path, baseline) == 0
    assert "restricted target set" in _clean_output(capsys)


def test_main_restricted_run_does_not_invite_updating_the_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unscanned keys are not fixed debt, so the advisory must be a caveat."""
    _require_ruff()
    _write_tree(tmp_path, _ONE_VIOLATION)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(_baseline_payload({"sample.py::B905": 1, "other.py::B904": 2}))
    )

    assert _run_main(tmp_path, baseline) == 0

    output = _clean_output(capsys)
    assert "RESTRICTED" in output
    assert "Nice — some lint debt was paid down" not in output


def test_main_reports_a_ruff_failure_as_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ruff that could not run is exit 2, distinct from "found regressions"."""
    _stub_ruff(monkeypatch, 1, "", "No module named ruff")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({"a.py::B905": 1})))

    assert _run_main(tmp_path, baseline) == 2
    assert "reported nothing" in _clean_output(capsys)


# ---------------------------------------------------------------------------
# Drift guards
# ---------------------------------------------------------------------------


def test_default_targets_match_the_lint_scripts_ruff_paths() -> None:
    """`DEFAULT_TARGETS` must cover exactly what `hatch run lint` lints.

    Read as TEXT rather than with ``tomllib``: the assertion is about the literal
    argument list written in the ``lint`` script, so parsing would only add a
    round-trip without making the comparison any stronger.
    """
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
    lint_block = re.search(r"^lint = \[\n(.*?)^\]", pyproject, re.MULTILINE | re.DOTALL)
    assert lint_block is not None, (
        "Could not locate the `lint` script in pyproject.toml"
    )

    command = re.search(r'"ruff check ([^"]+)"', lint_block.group(1))
    assert command is not None, "The `lint` script no longer runs `ruff check`"

    assert tuple(command.group(1).split()) == checker.DEFAULT_TARGETS


def test_the_ratchet_is_actually_wired_into_lint() -> None:
    """A gate nobody runs is not a gate.

    ``check-lint-ratchet`` must appear in the ``lint`` script — that is what
    ``hatch run lint``, ``check-static``, ``make check-all`` and CI all reach.
    """
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
    lint_block = re.search(r"^lint = \[\n(.*?)^\]", pyproject, re.MULTILINE | re.DOTALL)
    assert lint_block is not None
    assert '"check-lint-ratchet"' in lint_block.group(1)


def test_typer_defaults_are_declared_immutable() -> None:
    """The B008 exemption the baseline depends on must stay in pyproject.

    All 83 B008 findings in this tree were ``typer.Option``/``typer.Argument`` in
    a parameter default — the documented Typer idiom, not a defect. Remove the
    setting and they all come back as `new`; this asserts the intent so the
    failure is diagnosable rather than a surprise wall of findings.
    """
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
    section = re.search(
        r"^\[tool\.ruff\.lint\.flake8-bugbear\]\n(.*?)(?=^\[|\Z)",
        pyproject,
        re.MULTILINE | re.DOTALL,
    )
    assert section is not None, "flake8-bugbear settings are gone from pyproject.toml"
    assert "typer.Option" in section.group(1)
    assert "typer.Argument" in section.group(1)


def test_the_committed_baseline_holds_no_b008_debt() -> None:
    """The corollary: with the exemption in place, B008 is gated at ZERO.

    If a B008 entry ever appears here, the exemption stopped working and 83
    Typer false positives were baselined as real debt.
    """
    baseline = checker.load_baseline(PROJECT_ROOT / checker.DEFAULT_BASELINE_RELPATH)
    assert baseline, "the committed baseline is missing or empty"
    assert not [key for key in baseline if key.endswith("::B008")]


# ---------------------------------------------------------------------------
# Repo-level gate (this is what actually enforces the ratchet in CI)
# ---------------------------------------------------------------------------


def test_repository_has_no_lint_regressions() -> None:
    """The real tree must not add a violation of any ratcheted rule.

    Deliberately fails closed: a missing baseline is NOT skipped, it reports
    every violation as new — deleting the baseline must not turn the only
    enforcing gate green.
    """
    _require_ruff()
    baseline_path = PROJECT_ROOT / checker.DEFAULT_BASELINE_RELPATH
    baseline = checker.load_baseline(baseline_path)

    current = checker.parse_findings(
        checker.run_ruff(checker.DEFAULT_TARGETS, PROJECT_ROOT), PROJECT_ROOT
    )
    # Same fail-closed guard `main()` applies: a scan that covered nothing would
    # otherwise report every baselined key as fixed and pass this gate green.
    assert baseline, f"{baseline_path} is missing or empty — the ratchet has no floor"
    assert current, (
        "ruff reported no violation anywhere under "
        f"{checker.DEFAULT_TARGETS}, while {baseline_path} baselines "
        f"{len(baseline)}. The scan covered nothing."
    )

    report = checker.evaluate_ratchet(current, baseline)

    hint = (
        "Fix the code, or add a `# noqa: <code>` with a rationale if the rule is "
        "a false positive at that site. If the change is legitimate (a file "
        "move, say), run `hatch run check-lint-ratchet --update-baseline`."
    )
    assert not report.new, f"New lint violations: {report.new}. {hint}"
    assert not report.worsened, (
        f"Files that gained violations: {report.worsened}. {hint}"
    )
