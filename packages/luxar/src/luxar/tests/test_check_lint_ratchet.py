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
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _load_checker() -> ModuleType:
    """Import ``scripts/check_lint_ratchet.py`` as a module by file path."""
    script_path = PROJECT_ROOT / "scripts/check_lint_ratchet.py"
    sys.path.insert(0, str(script_path.parent))
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
_REAL_RUFF_SETTINGS_FINGERPRINT = checker.ruff_settings_fingerprint
_TEST_SETTINGS_FINGERPRINT = "test-settings-fingerprint"


@pytest.fixture(autouse=True)
def _stable_settings_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit fixtures independent of Ruff's large resolved-settings dump."""
    monkeypatch.setattr(
        checker,
        "ruff_settings_fingerprint",
        lambda _target, _project_root: _TEST_SETTINGS_FINGERPRINT,
    )


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


def test_ruff_settings_fingerprint_is_target_and_checkout_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only resolved settings matter, not Ruff's target banner or checkout path."""
    first_root = tmp_path / "first-checkout"
    second_root = tmp_path / "second-checkout"
    outputs = iter(
        [
            f'Resolved settings for: "{first_root}/a.py"\n'
            f'root = "{first_root}"\nvalue = 1\n',
            f'Resolved settings for: "{second_root}/b.py"\n'
            f'root = "{second_root}"\nvalue = 1\n',
        ]
    )

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, next(outputs), "")

    monkeypatch.setattr(checker.subprocess, "run", fake_run)

    first = _REAL_RUFF_SETTINGS_FINGERPRINT("a.py", first_root)
    second = _REAL_RUFF_SETTINGS_FINGERPRINT("b.py", second_root)

    assert first == second


def test_ruff_settings_fingerprint_rejects_empty_resolved_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful Ruff process that fingerprints nothing must fail closed."""
    _stub_ruff(monkeypatch, 0, 'Resolved settings for: "sample.py"\n')

    with pytest.raises(RuntimeError, match="no resolved settings"):
        _REAL_RUFF_SETTINGS_FINGERPRINT("sample.py", tmp_path)


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


def test_parse_findings_derives_accepted_codes_from_the_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parser must follow the configured selectors rather than a stale regex."""
    monkeypatch.setattr(checker, "RATCHETED_SELECT", ("B", "XYZ123"))

    parsed = checker.parse_findings(json.dumps([_finding("sample.py", "XYZ123")]))

    assert parsed == {"sample.py::XYZ123": 1}


@pytest.mark.parametrize("code", ["E501", "C901"])
def test_parse_findings_rejects_a_rule_outside_the_selection(code: str) -> None:
    """A code the selection cannot produce means the command drifted."""
    stdout = json.dumps([_finding("a.py", code)])

    with pytest.raises(ValueError, match=code):
        checker.parse_findings(stdout, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------


def _baseline_payload(violations: dict[str, int], rules: list[str] | None = None):
    """Build a baseline document, defaulting to the checker's own rule list."""
    return {
        "rules": list(checker.RATCHETED_SELECT) if rules is None else rules,
        "settings_fingerprint": _TEST_SETTINGS_FINGERPRINT,
        "violations": violations,
    }


def test_load_baseline_missing_file_is_empty(tmp_path: Path) -> None:
    """An absent baseline is empty, not an error — `main` reports it separately."""
    assert (
        checker.load_baseline(tmp_path / "nope.json", _TEST_SETTINGS_FINGERPRINT) == {}
    )


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
        checker.load_baseline(path, _TEST_SETTINGS_FINGERPRINT)


def test_load_baseline_rejects_changed_resolved_settings(tmp_path: Path) -> None:
    """A config change cannot make baselined findings disappear as improvements."""
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(_baseline_payload({"a.py::B905": 1})))

    with pytest.raises(ValueError, match="lint setting changed"):
        checker.load_baseline(path, "different-settings")


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"rules": list(checker.RATCHETED_SELECT)}, id="no-violations-key"),
        pytest.param(
            {"rules": list(checker.RATCHETED_SELECT), "violations": {}},
            id="no-settings-fingerprint",
        ),
        pytest.param(_baseline_payload({"a.py::B905": 0}), id="zero-count"),
        pytest.param(_baseline_payload({"a.py::B905": True}), id="bool-count"),
        pytest.param(_baseline_payload({"a.py::B905": "2"}), id="string-count"),
        pytest.param(
            {
                "rules": list(checker.RATCHETED_SELECT),
                "settings_fingerprint": _TEST_SETTINGS_FINGERPRINT,
                "violations": [],
            },
            id="list-not-dict",
        ),
    ],
)
def test_load_baseline_raises_on_a_malformed_file(
    tmp_path: Path, payload: object
) -> None:
    """A malformed baseline is an error, never a silently empty one."""
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="malformed"):
        checker.load_baseline(path, _TEST_SETTINGS_FINGERPRINT)


def test_load_baseline_raises_on_invalid_json(tmp_path: Path) -> None:
    """A truncated file must not be read as "no debt"."""
    path = tmp_path / "baseline.json"
    path.write_text("{not json")

    with pytest.raises(ValueError, match="not valid JSON"):
        checker.load_baseline(path, _TEST_SETTINGS_FINGERPRINT)


def test_save_baseline_round_trips_and_is_deterministic(tmp_path: Path) -> None:
    """Written keys are sorted, the rule list is recorded, and re-writing is stable."""
    path = tmp_path / "baseline.json"
    entries = {"z.py::B905": 2, "a.py::B904": 1}

    checker.save_baseline(path, entries, _TEST_SETTINGS_FINGERPRINT)
    first = path.read_text()
    assert checker.load_baseline(path, _TEST_SETTINGS_FINGERPRINT) == entries

    checker.save_baseline(path, entries, _TEST_SETTINGS_FINGERPRINT)
    assert path.read_text() == first

    document = json.loads(first)
    assert document["rules"] == list(checker.RATCHETED_SELECT)
    assert document["settings_fingerprint"] == _TEST_SETTINGS_FINGERPRINT
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
    """Fail clearly when the dependency that enforces this gate is absent."""
    assert importlib.util.find_spec("ruff") is not None, (
        "ruff is required for the lint-ratchet tests"
    )


def test_ruff_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping Ruff from the test environment must fail, never skip green."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(AssertionError, match="ruff is required"):
        _require_ruff()


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


def test_main_fails_on_a_new_blind_exception_handler(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A new silent broad catch cannot bypass the defect-rule ratchet."""
    _require_ruff()
    _write_tree(
        tmp_path,
        "try:\n    raise RuntimeError('boom')\nexcept Exception:\n    pass\n",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({})))

    assert _run_main(tmp_path, baseline) == 1

    output = _clean_output(capsys)
    assert "sample.py::BLE001" in output
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


def test_main_fails_when_a_fix_leaves_the_baseline_overdeclared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """ARM 3 — paid-down debt must be recorded before the gate passes."""
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, _ONE_VIOLATION)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({"sample.py::B905": 2})))

    assert _run_main(tmp_path, baseline) == 1

    output = _clean_output(capsys)
    assert "Improved: 1" in output
    assert "Baseline no longer matches the current tree" in output
    assert "your change, a dev merge, or a Ruff update" in output
    assert output.count("--update-baseline") == 1


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


def test_main_update_baseline_records_the_current_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--update-baseline` writes exactly what the scan found, plus the rules."""
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, _TWO_VIOLATIONS)
    baseline = tmp_path / "baseline.json"

    assert _run_main(tmp_path, baseline, "--update-baseline") == 0

    document = json.loads(baseline.read_text())
    assert document["violations"] == {"sample.py::B905": 2}
    assert document["rules"] == list(checker.RATCHETED_SELECT)
    assert document["settings_fingerprint"] == _TEST_SETTINGS_FINGERPRINT


@pytest.mark.parametrize(
    "existing",
    [
        pytest.param(
            json.dumps(_baseline_payload({"sample.py::B905": 1}, rules=["B"])),
            id="rules-mismatch",
        ),
        pytest.param(
            json.dumps(
                {
                    "rules": list(checker.RATCHETED_SELECT),
                    "settings_fingerprint": "stale-settings",
                    "violations": {"sample.py::B905": 1},
                }
            ),
            id="settings-mismatch",
        ),
        pytest.param("{truncated", id="invalid-json"),
    ],
)
def test_main_update_baseline_replaces_an_unloadable_baseline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    existing: str,
) -> None:
    """An explicit full-tree update repairs stale or malformed baseline metadata."""
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, _ONE_VIOLATION)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(existing)

    assert _run_main(tmp_path, baseline, "--update-baseline") == 0

    document = json.loads(baseline.read_text())
    assert document["violations"] == {"sample.py::B905": 1}
    assert document["rules"] == list(checker.RATCHETED_SELECT)
    assert document["settings_fingerprint"] == _TEST_SETTINGS_FINGERPRINT
    assert "Re-recording the baseline for the current Ruff settings" in _clean_output(
        capsys
    )


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


def test_main_refuses_to_wipe_an_unloadable_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair mode still refuses a zero-finding overwrite it cannot prove safe."""
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, _CLEAN_MODULE)
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{truncated")

    assert _run_main(tmp_path, baseline, "--update-baseline") == 2
    assert "Refusing to overwrite" in _clean_output(capsys)
    assert baseline.read_text() == "{truncated"


def test_main_update_baseline_diagnoses_a_non_utf8_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair mode diagnoses undecodable debt without rewriting it."""
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, _CLEAN_MODULE)
    baseline = tmp_path / "baseline.json"
    original = b'\xff\xfe{"violations": {}}'
    baseline.write_bytes(original)

    assert _run_main(tmp_path, baseline, "--update-baseline") == 2

    assert "utf-8" in _clean_output(capsys)
    assert baseline.read_bytes() == original


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


def test_main_restricted_update_baseline_refuses_to_drop_debt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A partial baseline is never useful, so a restricted rewrite is refused."""
    _require_ruff()
    _write_tree(tmp_path, _ONE_VIOLATION)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_payload({"other.py::B905": 4})))

    assert _run_main(tmp_path, baseline, "--update-baseline") == 2

    output = _clean_output(capsys)
    assert "Refusing" in output and "RESTRICTED" in output
    assert json.loads(baseline.read_text())["violations"] == {"other.py::B905": 4}


def test_settings_refresh_does_not_report_retirement_when_write_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refused refresh must not claim that baseline keys were retired."""
    _require_ruff()
    _write_tree(tmp_path, _ONE_VIOLATION)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(_baseline_payload({"other.py::B905": 4}, rules=["B"]))
    )

    assert _run_main(tmp_path, baseline, "--update-baseline") == 2

    output = _clean_output(capsys)
    assert "Refusing" in output and "RESTRICTED" in output
    assert "Retiring" not in output
    assert json.loads(baseline.read_text())["violations"] == {"other.py::B905": 4}


def test_main_rejects_a_baselined_file_excluded_from_the_full_scan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing baselined file omitted by ruff is a partial scan, not a fix."""
    _require_ruff()
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_text(_ONE_VIOLATION)
    (tmp_path / "pkg" / "sub" / "b.py").write_text(_ONE_VIOLATION)
    (tmp_path / "pyproject.toml").write_text('[tool.ruff]\nexclude = ["pkg/sub"]\n')
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(_baseline_payload({"pkg/a.py::B905": 1, "pkg/sub/b.py::B905": 1}))
    )
    monkeypatch.setattr(checker, "DEFAULT_TARGETS", ("pkg",))

    result = checker.main(
        ["--project-root", str(tmp_path), "--baseline", str(baseline)]
    )

    assert result == 2
    output = _clean_output(capsys)
    assert "PARTIAL" in output
    assert "pkg/sub/b.py" in output
    assert "remove those files' keys from the baseline by hand" in output


def test_update_baseline_rejects_a_file_omitted_by_settings_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fingerprint refresh must not erase keys hidden by the new settings."""
    _require_ruff()
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_text(_ONE_VIOLATION)
    (tmp_path / "pkg" / "sub" / "b.py").write_text(_ONE_VIOLATION)
    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(checker, "DEFAULT_TARGETS", ("pkg",))
    monkeypatch.setattr(
        checker, "ruff_settings_fingerprint", _REAL_RUFF_SETTINGS_FINGERPRINT
    )
    args = ["--project-root", str(tmp_path), "--baseline", str(baseline)]

    assert checker.main([*args, "--update-baseline"]) == 0
    mismatched_document = json.loads(baseline.read_text())
    mismatched_document["rules"] = ["B"]
    original = json.dumps(mismatched_document)
    baseline.write_text(original)
    capsys.readouterr()
    (tmp_path / "pyproject.toml").write_text('[tool.ruff]\nexclude = ["pkg/sub"]\n')

    assert checker.main([*args, "--update-baseline"]) == 2
    output = _clean_output(capsys)
    assert "pkg/sub/b.py::B905" in output
    assert baseline.read_text() == original


def test_settings_refresh_reports_retired_lint_keys(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deliberate refresh names same-file debt omitted by current settings."""
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, _ONE_VIOLATION)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            _baseline_payload(
                {"sample.py::B905": 1, "sample.py::RUF012": 1}, rules=["B"]
            )
        )
    )

    assert _run_main(tmp_path, baseline, "--update-baseline") == 0

    output = _clean_output(capsys)
    assert "Retiring 1 baseline key" in output
    assert "sample.py::RUF012" in output
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
    assert "Some lint debt was paid down" not in output


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
    ``hatch run lint``, ``check-static`` and ``make check-all`` all reach.
    CI gates through ``test_repository_has_no_lint_regressions`` below.
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
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    immutable_calls = pyproject["tool"]["ruff"]["lint"]["flake8-bugbear"][
        "extend-immutable-calls"
    ]
    assert immutable_calls == ["typer.Option", "typer.Argument"]


@pytest.mark.parametrize(
    ("pyproject", "extra_config"),
    [
        ('[tool.ruff]\ntarget-version = "py39"\n', None),
        (
            '[tool.ruff]\ntarget-version = "py312"\n'
            '[tool.ruff.lint]\nexclude = ["sample.py"]\n',
            None,
        ),
        (
            '[tool.ruff]\ntarget-version = "py312"\n'
            '[tool.ruff.per-file-target-version]\n"sample.py" = "py39"\n',
            None,
        ),
        (
            '[tool.ruff]\ntarget-version = "py312"\nextend = "extra.toml"\n',
            '[lint.per-file-ignores]\n"sample.py" = ["B905"]\n',
        ),
        (
            '[tool.ruff]\ntarget-version = "py312"\n'
            "[tool.ruff.lint]\npreview = true\n"
            "[tool.ruff.lint.per-file-ignores]\n"
            '"sample.py" = ["zip-without-explicit-strict"]\n',
            None,
        ),
    ],
    ids=(
        "target-version",
        "lint-exclude",
        "per-file-target-version",
        "extended-config",
        "rule-name",
    ),
)
def test_resolved_settings_drift_fails_before_debt_can_disappear(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    pyproject: str,
    extra_config: str | None,
) -> None:
    """Every confirmed Ruff suppression route invalidates the old baseline."""
    _require_ruff()
    monkeypatch.setattr(
        checker, "ruff_settings_fingerprint", _REAL_RUFF_SETTINGS_FINGERPRINT
    )
    _write_tree(
        tmp_path,
        "list(zip([1], [2]))\n"
        "try:\n    raise ValueError\nexcept ValueError:\n    raise RuntimeError\n",
    )
    (tmp_path / "pyproject.toml").write_text('[tool.ruff]\ntarget-version = "py312"\n')
    original_fingerprint = _REAL_RUFF_SETTINGS_FINGERPRINT("sample.py", tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "rules": list(checker.RATCHETED_SELECT),
                "settings_fingerprint": original_fingerprint,
                "violations": {"sample.py::B904": 1, "sample.py::B905": 1},
            }
        )
    )

    (tmp_path / "pyproject.toml").write_text(pyproject)
    if extra_config is not None:
        (tmp_path / "extra.toml").write_text(extra_config)

    assert _run_main(tmp_path, baseline) == 2
    assert "lint setting changed" in _clean_output(capsys)


@pytest.mark.parametrize(
    ("config_path", "config"),
    [
        ("ruff.toml", '[lint]\nexclude = ["deep.py"]\n'),
        (".ruff.toml", '[lint]\nexclude = ["deep.py"]\n'),
        (
            "pyproject.toml",
            '[tool.ruff.lint]\nexclude = ["deep.py"]\n',
        ),
    ],
)
def test_nested_ruff_config_fails_before_debt_can_disappear(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    config_path: str,
    config: str,
) -> None:
    """A hierarchical Ruff config must not hide debt outside the fingerprint."""
    _require_ruff()
    monkeypatch.setattr(
        checker, "ruff_settings_fingerprint", _REAL_RUFF_SETTINGS_FINGERPRINT
    )
    monkeypatch.setattr(checker, "DEFAULT_TARGETS", ("pkg",))
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "keep.py").write_text(_ONE_VIOLATION)
    (tmp_path / "pkg" / "sub" / "deep.py").write_text(_ONE_VIOLATION)
    (tmp_path / "pyproject.toml").write_text('[tool.ruff]\ntarget-version = "py312"\n')
    settings_fingerprint = _REAL_RUFF_SETTINGS_FINGERPRINT("pkg", tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "rules": list(checker.RATCHETED_SELECT),
                "settings_fingerprint": settings_fingerprint,
                "violations": {
                    "pkg/keep.py::B905": 1,
                    "pkg/sub/deep.py::B905": 1,
                },
            }
        )
    )
    (tmp_path / "pkg" / "sub" / config_path).write_text(config)

    assert (
        checker.main(["--project-root", str(tmp_path), "--baseline", str(baseline)])
        == 2
    )
    output = _clean_output(capsys)
    assert "Nested Ruff configuration" in output
    assert f"pkg/sub/{config_path}" in output


def test_nested_pyproject_without_ruff_settings_is_allowed(tmp_path: Path) -> None:
    """An unrelated nested package manifest is not a Ruff configuration."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "1.0"\n'
    )

    assert checker.find_nested_ruff_configs(("pkg",), tmp_path) == []


def test_root_ruff_config_is_allowed(tmp_path: Path) -> None:
    """The fingerprint intentionally covers the repository-root config."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "ruff.toml").write_text('[lint]\nselect = ["B"]\n')

    assert checker.find_nested_ruff_configs(("pkg",), tmp_path) == []


def test_nested_ruff_config_is_found_under_every_target(tmp_path: Path) -> None:
    """A later lint target cannot escape discovery through the first target."""
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    config = tmp_path / "second" / ".ruff.toml"
    config.write_text('[lint]\nselect = ["B"]\n')

    assert checker.find_nested_ruff_configs(("first", "second"), tmp_path) == [config]


def test_root_file_target_does_not_scan_unrelated_sibling_trees(
    tmp_path: Path,
) -> None:
    """A repository-root file target must not expand discovery to the whole tree."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "hatch_build.py").write_text("")
    unrelated = tmp_path / "node_modules" / "dependency" / "pyproject.toml"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("[tool.ruff]\nline-length = 200\n")

    assert checker.find_nested_ruff_configs(("pkg", "hatch_build.py"), tmp_path) == []


def test_file_target_still_checks_ancestor_configs(tmp_path: Path) -> None:
    """A file target must still inherit Ruff config from its own directory."""
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir()
    target.write_text("")
    config = tmp_path / "pkg" / "ruff.toml"
    config.write_text('[lint]\nselect = ["B"]\n')

    assert checker.find_nested_ruff_configs(("pkg/module.py",), tmp_path) == [config]


def test_unreadable_nested_pyproject_has_contextual_error(tmp_path: Path) -> None:
    """A non-UTF-8 manifest must fail with the nested-config diagnostic."""
    config = tmp_path / "pkg" / "sub" / "pyproject.toml"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"\xff\xfe[tool.ruff]\n")

    with pytest.raises(
        RuntimeError, match=r"Could not inspect nested .*pyproject.toml"
    ):
        checker.find_nested_ruff_configs(("pkg",), tmp_path)


def test_the_committed_baseline_holds_no_b008_debt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The corollary: with the exemption in place, B008 is gated at ZERO.

    If a B008 entry ever appears here, the exemption stopped working and 83
    Typer false positives were baselined as real debt.
    """
    monkeypatch.setattr(
        checker, "ruff_settings_fingerprint", _REAL_RUFF_SETTINGS_FINGERPRINT
    )
    settings_fingerprint = _REAL_RUFF_SETTINGS_FINGERPRINT(
        checker.DEFAULT_TARGETS[0], PROJECT_ROOT
    )
    baseline = checker.load_baseline(
        PROJECT_ROOT / checker.DEFAULT_BASELINE_RELPATH, settings_fingerprint
    )
    assert baseline, "the committed baseline is missing or empty"
    assert not [key for key in baseline if key.endswith("::B008")]


def test_repository_settings_fingerprint_is_target_independent() -> None:
    """The representative target must not affect this repository's fingerprint."""
    _require_ruff()
    assert _REAL_RUFF_SETTINGS_FINGERPRINT(
        checker.DEFAULT_TARGETS[0], PROJECT_ROOT
    ) == _REAL_RUFF_SETTINGS_FINGERPRINT("stats", PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Repo-level gate (this is what actually enforces the ratchet in CI)
# ---------------------------------------------------------------------------


def test_repository_has_no_lint_regressions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real tree and baseline must describe the same lint debt.

    Deliberately fails closed: a missing baseline is NOT skipped, it reports
    every violation as new — deleting the baseline must not turn the only
    enforcing gate green.
    """
    _require_ruff()
    monkeypatch.setattr(
        checker, "ruff_settings_fingerprint", _REAL_RUFF_SETTINGS_FINGERPRINT
    )
    baseline_path = PROJECT_ROOT / checker.DEFAULT_BASELINE_RELPATH
    settings_fingerprint = _REAL_RUFF_SETTINGS_FINGERPRINT(
        checker.DEFAULT_TARGETS[0], PROJECT_ROOT
    )
    baseline = checker.load_baseline(baseline_path, settings_fingerprint)

    current = checker.parse_findings(
        checker.run_ruff(checker.DEFAULT_TARGETS, PROJECT_ROOT), PROJECT_ROOT
    )
    checker.ensure_no_nested_ruff_configs(checker.DEFAULT_TARGETS, PROJECT_ROOT)
    scanned_files = checker.list_ruff_files(checker.DEFAULT_TARGETS, PROJECT_ROOT)
    # Same fail-closed guard `main()` applies: a scan that covered nothing would
    # otherwise report every baselined key as fixed and pass this gate green.
    assert baseline, f"{baseline_path} is missing or empty — the ratchet has no floor"
    assert current, (
        "ruff reported no violation anywhere under "
        f"{checker.DEFAULT_TARGETS}, while {baseline_path} baselines "
        f"{len(baseline)}. The scan covered nothing."
    )
    checker.ensure_baselined_files_were_scanned(baseline, scanned_files, PROJECT_ROOT)

    report = checker.evaluate_ratchet(current, baseline)

    regression_hint = (
        "Fix the code, or add a `# noqa: <code>` with a rationale if the rule is "
        "a false positive at that site. If the change is legitimate (a file "
        "move, say), run `hatch run check-lint-ratchet --update-baseline`."
    )
    baseline_hint = "Run `hatch run check-lint-ratchet --update-baseline`."
    assert not report.new, f"New lint violations: {report.new}. {regression_hint}"
    assert not report.worsened, (
        f"Files that gained violations: {report.worsened}. {regression_hint}"
    )
    assert not report.improved, (
        f"Over-declared baseline entries: {report.improved}. {baseline_hint}"
    )
