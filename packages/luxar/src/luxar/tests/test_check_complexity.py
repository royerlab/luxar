"""Tests for the cyclomatic-complexity ratchet (``scripts/check_complexity.py``)."""

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
    """Import ``scripts/check_complexity.py`` as a module by file path."""
    script_path = PROJECT_ROOT / "scripts/check_complexity.py"
    sys.path.insert(0, str(script_path.parent))
    spec = importlib.util.spec_from_file_location("check_complexity", script_path)
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
    """Keep unit fixtures independent of Ruff's resolved-settings dump."""
    monkeypatch.setattr(
        checker,
        "ruff_settings_fingerprint",
        lambda _target, _project_root: _TEST_SETTINGS_FINGERPRINT,
    )


def _baseline_payload(functions: object) -> str:
    """Return a complexity baseline document for unit fixtures."""
    return json.dumps(
        {
            "settings_fingerprint": _TEST_SETTINGS_FINGERPRINT,
            "functions": functions,
        }
    )


def _finding(filename: str, name: str, value: int) -> dict[str, object]:
    """Build one synthetic ruff C901 finding."""
    return {
        "code": "C901",
        "filename": str(PROJECT_ROOT / filename),
        "message": f"`{name}` is too complex ({value} > 10)",
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
    """Exit 2 (bad arguments) is an error even when stdout parses."""
    _stub_ruff(monkeypatch, 2, "[]", "error: unexpected argument")

    with pytest.raises(RuntimeError, match="exit 2"):
        checker.run_ruff(("x",), PROJECT_ROOT)


def test_run_ruff_rejects_a_target_it_could_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Failed to lint <path>` means a PARTIAL scan, which must not read green."""
    _stub_ruff(monkeypatch, 0, "[]", "warning: Failed to lint nope.py: No such file")

    with pytest.raises(RuntimeError, match="scan is PARTIAL"):
        checker.run_ruff(("nope.py",), PROJECT_ROOT)


def test_run_ruff_rejects_a_partial_scan_that_still_reported_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One unreadable target beside a productive one is the dangerous shape.

    ruff lints what it can and exits 1 with a real report, so neither the exit
    code nor the empty-stdout check notices — yet every baselined key under the
    unread path now looks fixed, and could even absorb a genuinely new function
    as a `moved` one. Only the stderr line gives it away.
    """
    _stub_ruff(
        monkeypatch,
        1,
        json.dumps([_finding("scripts/b.py", "beta", 15)]),
        "warning: Failed to lint packages/luxar/src: No such file or directory",
    )

    with pytest.raises(RuntimeError, match="packages/luxar/src"):
        checker.run_ruff(checker.DEFAULT_TARGETS, PROJECT_ROOT)


def test_run_ruff_echoes_an_unrelated_stderr_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A warning that is not a read failure is echoed, not fatal.

    `No Python files found` is legitimate over an explicit target subtree; the
    whole-scan case is caught by `main`'s zero-findings guard instead.
    """
    _stub_ruff(monkeypatch, 0, "[]", "warning: No Python files found under the path(s)")

    assert checker.run_ruff(("empty/",), PROJECT_ROOT) == "[]"
    assert "No Python files found" in _clean_output(capsys)


# ---------------------------------------------------------------------------
# parse_findings
# ---------------------------------------------------------------------------


def test_parse_findings_builds_relative_keys_and_merges_same_name() -> None:
    """Keys are ``<rel-path>::<name>``; same-named functions share one key."""
    payload = json.dumps(
        [
            _finding("packages/luxar/src/luxar/a.py", "alpha", 12),
            _finding("scripts/b.py", "beta", 15),
            # Two same-named functions in one file (two classes / a nested def):
            # they collapse to a single, descending-sorted entry.
            _finding("scripts/b.py", "render", 11),
            _finding("scripts/b.py", "render", 31),
        ]
    )

    assert checker.parse_findings(payload, PROJECT_ROOT) == {
        "packages/luxar/src/luxar/a.py::alpha": [12],
        "scripts/b.py::beta": [15],
        "scripts/b.py::render": [31, 11],
    }


def test_parse_findings_handles_empty_output() -> None:
    """No findings (empty or blank stdout) yields an empty mapping."""
    assert checker.parse_findings("", PROJECT_ROOT) == {}
    assert checker.parse_findings("[]", PROJECT_ROOT) == {}


def test_parse_findings_raises_on_unexpected_message() -> None:
    """A ruff wording change must fail loudly, not silently empty the baseline."""
    payload = json.dumps(
        [
            {
                "code": "C901",
                "filename": str(PROJECT_ROOT / "scripts/b.py"),
                "message": "`beta` has a cyclomatic complexity of 15",
            }
        ]
    )

    with pytest.raises(ValueError, match="Unexpected C901 message format"):
        checker.parse_findings(payload, PROJECT_ROOT)


def test_parse_findings_raises_on_unparseable_json() -> None:
    """Non-JSON stdout must raise, not quietly become an empty (green) scan."""
    with pytest.raises(ValueError, match="Could not parse ruff JSON output"):
        checker.parse_findings("not json at all", PROJECT_ROOT)


def test_parse_findings_ignores_non_c901_diagnostics() -> None:
    """ruff reports `invalid-syntax` whatever is selected; it must be skipped."""
    payload = json.dumps(
        [
            {
                "code": "invalid-syntax",
                "filename": str(PROJECT_ROOT / "scripts/wip.py"),
                "message": "Expected a parameter or the end of the parameter list",
            },
            _finding("scripts/b.py", "beta", 15),
        ]
    )

    assert checker.parse_findings(payload, PROJECT_ROOT) == {"scripts/b.py::beta": [15]}


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------


def test_load_baseline_missing_file_is_empty(tmp_path: Path) -> None:
    """A missing baseline is an empty baseline (everything counts as new)."""
    assert checker.load_baseline(tmp_path / "nope.json") == {}


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{not json", "not valid JSON"),
        ('{"oops": {}}', "'functions' object"),
        (_baseline_payload([]), "must be an object"),
        (_baseline_payload({"a.py::f": 12}), "non-empty list of integers"),
        (_baseline_payload({"a.py::f": []}), "non-empty list of integers"),
        (_baseline_payload({"a.py::f": ["12"]}), "non-empty list of integers"),
    ],
)
def test_load_baseline_raises_on_malformed_file(
    tmp_path: Path, content: str, match: str
) -> None:
    """Every malformed baseline shape raises a clear ValueError."""
    path = tmp_path / "baseline.json"
    path.write_text(content)

    with pytest.raises(ValueError, match=match):
        checker.load_baseline(path)


def test_load_baseline_sorts_values_descending(tmp_path: Path) -> None:
    """An ascending on-disk list loads descending (the comparison's invariant)."""
    path = tmp_path / "baseline.json"
    path.write_text(_baseline_payload({"a.py::f": [11, 31, 20]}))

    assert checker.load_baseline(path) == {"a.py::f": [31, 20, 11]}


def test_save_baseline_round_trips_and_is_deterministic(tmp_path: Path) -> None:
    """The written JSON is sorted, newline-terminated, and reloads identically."""
    path = tmp_path / "baseline.json"
    entries = {
        "scripts/b.py::render": [11, 31],
        "packages/luxar/src/luxar/a.py::alpha": [12],
    }

    checker.save_baseline(path, entries)

    assert checker.load_baseline(path) == {
        "scripts/b.py::render": [31, 11],
        "packages/luxar/src/luxar/a.py::alpha": [12],
    }
    text = path.read_text()
    assert text.endswith("\n")
    data = json.loads(text)
    assert list(data["functions"]) == sorted(data["functions"])
    assert data["functions"]["scripts/b.py::render"] == [31, 11]
    assert data["settings_fingerprint"] == _TEST_SETTINGS_FINGERPRINT
    assert "_comment" in data


def test_baseline_is_populated_never_reads_a_bad_file_as_empty(
    tmp_path: Path,
) -> None:
    """The overwrite guard must treat anything it cannot read as populated."""
    missing = tmp_path / "missing.json"
    empty = tmp_path / "empty.json"
    empty.write_text(_baseline_payload({}))
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json")
    populated = tmp_path / "populated.json"
    checker.save_baseline(populated, {"a.py::f": [12]})

    assert checker._baseline_is_populated(missing) is False
    assert checker._baseline_is_populated(empty) is False
    assert checker._baseline_is_populated(malformed) is True
    assert checker._baseline_is_populated(populated) is True
    # A directory raises OSError from read_text(); it must not escape.
    assert checker._baseline_is_populated(tmp_path) is True


# ---------------------------------------------------------------------------
# evaluate_ratchet
# ---------------------------------------------------------------------------


def test_evaluate_ratchet_classifies_every_transition() -> None:
    """new / worsened / improved / unchanged are each derived correctly."""
    baseline = {
        "a.py::same": [12],
        "a.py::rose": [12],
        "a.py::extra": [12],
        "a.py::fell": [20],
        "a.py::gone": [14],
    }
    current = {
        "a.py::same": [12],
        "a.py::rose": [13],
        # Gained a second same-named offender: more debt under one key.
        "a.py::extra": [12, 11],
        "a.py::fell": [11],
        "a.py::brand_new": [11],
    }

    report = checker.evaluate_ratchet(current, baseline)

    assert report.new == ["a.py::brand_new"]
    assert report.worsened == ["a.py::extra", "a.py::rose"]
    assert report.improved == ["a.py::fell", "a.py::gone"]
    assert report.unchanged == ["a.py::same"]


def test_evaluate_ratchet_treats_a_file_move_as_advisory() -> None:
    """A relocated function is `moved`, not `new` + `improved` (no false red)."""
    baseline = {"old/mod.py::f": [14], "old/mod.py::g": [12]}
    current = {"new/mod.py::f": [14], "new/mod.py::g": [12]}

    report = checker.evaluate_ratchet(current, baseline)

    assert report.new == []
    assert report.worsened == []
    assert report.improved == []
    assert report.moved == [
        "old/mod.py::f -> new/mod.py::f",
        "old/mod.py::g -> new/mod.py::g",
    ]


def test_evaluate_ratchet_move_pairing_is_one_to_one() -> None:
    """Two vanished `::f` absorb at most two new `::f`; the third stays new."""
    baseline = {"a.py::f": [14], "b.py::f": [14]}
    current = {"x.py::f": [14], "y.py::f": [14], "z.py::f": [14]}

    report = checker.evaluate_ratchet(current, baseline)

    assert len(report.moved) == 2
    assert len(report.new) == 1
    assert report.improved == []


def test_evaluate_ratchet_move_requires_the_new_key_to_be_no_worse() -> None:
    """A relocated function that ALSO got more complex is still a regression."""
    report = checker.evaluate_ratchet({"new/mod.py::f": [20]}, {"old/mod.py::f": [14]})

    assert report.moved == []
    assert report.new == ["new/mod.py::f"]
    assert report.improved == ["old/mod.py::f"]


def test_evaluate_ratchet_pairs_a_move_that_also_simplifies() -> None:
    """Move-and-tidy — the commonest extract-module shape — must not be a red."""
    report = checker.evaluate_ratchet(
        {"cli/new.py::handle": [12]}, {"cli/old.py::handle": [20]}
    )

    assert report.new == []
    assert report.improved == []
    assert report.moved == ["cli/old.py::handle -> cli/new.py::handle"]


def test_evaluate_ratchet_move_pairing_is_deterministic() -> None:
    """Pairing walks both sides complexity-descending, not in dict order.

    With several same-named candidates the result must not depend on how the
    mappings happen to be built, or the baseline would churn between runs.
    """
    baseline = {"a.py::f": [12], "b.py::f": [20]}
    current = {"c.py::f": [20], "d.py::f": [11]}
    shuffled_baseline = {"b.py::f": [20], "a.py::f": [12]}
    shuffled_current = {"d.py::f": [11], "c.py::f": [20]}

    report = checker.evaluate_ratchet(current, baseline)

    assert report.new == []
    assert report.moved == ["a.py::f -> d.py::f", "b.py::f -> c.py::f"]
    assert (
        checker.evaluate_ratchet(shuffled_current, shuffled_baseline).moved
        == report.moved
    )


def test_evaluate_ratchet_clean_against_own_baseline() -> None:
    """An unchanged snapshot has no failing sets."""
    current = {"a.py::f": [12], "b.py::g": [30, 11]}

    report = checker.evaluate_ratchet(current, dict(current))

    assert report.new == []
    assert report.worsened == []
    assert report.improved == []
    assert sorted(report.unchanged) == ["a.py::f", "b.py::g"]


def test_evaluate_ratchet_empty_baseline_makes_everything_new() -> None:
    """Without a baseline every over-limit function is a regression."""
    report = checker.evaluate_ratchet({"a.py::f": [12]}, {})

    assert report.new == ["a.py::f"]
    assert report.worsened == []


# ---------------------------------------------------------------------------
# main() — the CLI gate itself
# ---------------------------------------------------------------------------


def _write_tree(tmp_path: Path, *, complex_function: bool) -> str:
    """Write a one-module tree under ``tmp_path``; return the target filename."""
    branches = (
        "\n".join(f"        if i % {p} == 0:\n            total += {p}" for p in PRIMES)
        if complex_function
        else "        total += i"
    )
    (tmp_path / "sample.py").write_text(
        '"""Fixture module."""\n\n\ndef tangled(n: int) -> int:\n'
        '    """Fixture function."""\n'
        "    total = 0\n"
        "    for i in range(n):\n"
        f"{branches}\n"
        "    return total\n"
    )
    return "sample.py"


PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)


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


def _require_ruff() -> None:
    """Fail clearly when the dependency that enforces this gate is absent."""
    assert importlib.util.find_spec("ruff") is not None, (
        "ruff is required for the complexity-ratchet tests"
    )


def test_ruff_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping Ruff from the test environment must fail, never skip green."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(AssertionError, match="ruff is required"):
        _require_ruff()


def _unrestrict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fixture target count as the FULL scope, not a restricted run.

    ``_run_main`` always passes an explicit target, which the checker rightly
    treats as restricted; this pins ``DEFAULT_TARGETS`` to the same value so the
    unrestricted reporting path can be exercised too.
    """
    monkeypatch.setattr(checker, "DEFAULT_TARGETS", ("sample.py",))


def test_main_fails_on_a_function_that_is_not_baselined(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI exits 1 and names the offending function."""
    _require_ruff()
    _write_tree(tmp_path, complex_function=True)

    exit_code = _run_main(tmp_path, tmp_path / "baseline.json")
    output = _clean_output(capsys)

    assert exit_code == 1
    assert "sample.py::tangled" in output
    assert "🆕 New: 1" in output


def test_main_passes_when_the_function_is_baselined(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same tree with a matching baseline is tolerated (exit 0)."""
    _require_ruff()
    _write_tree(tmp_path, complex_function=True)
    baseline = tmp_path / "baseline.json"

    assert _run_main(tmp_path, baseline, "--update-baseline") == 0
    capsys.readouterr()

    assert _run_main(tmp_path, baseline) == 0
    assert "No new complexity regressions" in _clean_output(capsys)


def test_main_rejects_ruff_settings_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A config edit that suppresses C901 must fail, not retire the debt."""
    _require_ruff()
    monkeypatch.setattr(
        checker, "ruff_settings_fingerprint", _REAL_RUFF_SETTINGS_FINGERPRINT
    )
    _write_tree(tmp_path, complex_function=True)
    baseline = tmp_path / "baseline.json"

    assert _run_main(tmp_path, baseline, "--update-baseline") == 0
    capsys.readouterr()

    (tmp_path / "pyproject.toml").write_text(
        "[tool.ruff.lint.mccabe]\nmax-complexity = 40\n"
    )

    assert _run_main(tmp_path, baseline) == 2
    assert "settings" in _clean_output(capsys)


def test_main_rejects_an_existing_baselined_file_omitted_by_ruff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A partial scan must fail instead of reporting the omitted key improved."""
    _require_ruff()
    monkeypatch.setattr(checker, "DEFAULT_TARGETS", ("pkg",))
    for directory in (tmp_path / "pkg" / "kept", tmp_path / "pkg" / "sub"):
        directory.mkdir(parents=True)
        _write_tree(directory, complex_function=True)
    baseline = tmp_path / "baseline.json"
    args = [
        "pkg",
        "--project-root",
        str(tmp_path),
        "--baseline",
        str(baseline),
    ]

    assert checker.main([*args, "--update-baseline"]) == 0
    capsys.readouterr()
    (tmp_path / "pyproject.toml").write_text('[tool.ruff]\nexclude = ["pkg/sub"]\n')

    assert checker.main(args) == 2
    output = _clean_output(capsys)
    assert "scan is PARTIAL" in output
    assert "pkg/sub/sample.py" in output


def test_main_rejects_nested_ruff_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A per-subtree Ruff configuration cannot escape the root fingerprint."""
    _require_ruff()
    target = tmp_path / "pkg" / "sample.py"
    target.parent.mkdir()
    target.write_text("def simple() -> int:\n    return 1\n")
    (target.parent / "ruff.toml").write_text('[lint]\nselect = ["C901"]\n')

    exit_code = checker.main(
        [
            "pkg",
            "--project-root",
            str(tmp_path),
            "--baseline",
            str(tmp_path / "baseline.json"),
        ]
    )

    assert exit_code == 2
    assert "Nested Ruff configuration" in _clean_output(capsys)


def test_main_update_baseline_writes_the_expected_entry(tmp_path: Path) -> None:
    """`--update-baseline` on a fresh path records the over-limit function."""
    _require_ruff()
    _write_tree(tmp_path, complex_function=True)
    baseline = tmp_path / "fresh.json"

    assert _run_main(tmp_path, baseline, "--update-baseline") == 0

    entries = checker.load_baseline(baseline)
    assert list(entries) == ["sample.py::tangled"]
    assert entries["sample.py::tangled"][0] > 10


def test_main_refuses_to_wipe_a_populated_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A full scan collapsing to zero findings must not disable the ratchet."""
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, complex_function=False)
    baseline = tmp_path / "baseline.json"
    checker.save_baseline(baseline, {"sample.py::tangled": [13]})

    exit_code = _run_main(tmp_path, baseline, "--update-baseline")
    output = _clean_output(capsys)

    assert exit_code == 2
    assert "Refusing to overwrite" in output
    assert "ruff did not run correctly" in output
    # The populated baseline is untouched.
    assert checker.load_baseline(baseline) == {"sample.py::tangled": [13]}


def test_main_refusal_diagnoses_a_clean_restricted_update(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A restricted scan finding nothing is not evidence that ruff misbehaved."""
    _require_ruff()
    _write_tree(tmp_path, complex_function=False)
    baseline = tmp_path / "baseline.json"
    checker.save_baseline(baseline, {"other/mod.py::elsewhere": [13]})

    exit_code = _run_main(tmp_path, baseline, "--update-baseline")
    output = _clean_output(capsys)

    assert exit_code == 2
    assert "RESTRICTED target set produced no findings" in output
    assert "over the default targets" in output
    # Never the wrong diagnosis, nor the advice that would drop the debt.
    assert "ruff did not run correctly" not in output
    assert "delete the baseline file by hand" not in output
    assert checker.load_baseline(baseline) == {"other/mod.py::elsewhere": [13]}


def test_main_reports_a_move_as_advisory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A baselined function that relocated does not fail the gate."""
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, complex_function=True)
    baseline = tmp_path / "baseline.json"

    assert _run_main(tmp_path, baseline, "--update-baseline") == 0
    capsys.readouterr()
    # Simulate the move by re-keying the baseline to the function's old home,
    # and make it a move-AND-SIMPLIFY: the new complexity is strictly lower.
    moved_from = checker.load_baseline(baseline)["sample.py::tangled"]
    checker.save_baseline(baseline, {"old/sample.py::tangled": [moved_from[0] + 5]})

    exit_code = _run_main(tmp_path, baseline)
    output = _clean_output(capsys)

    assert exit_code == 0
    assert "🚚 Moved (same function, new file): 1" in output
    assert "🆕 New: 0" in output
    assert "✨ Improved: 0" in output
    assert "run --update-baseline to re-key the baseline" in output
    # Auditable on the GREEN path, not just when the run fails.
    assert "🚚 old/sample.py::tangled -> sample.py::tangled" in output
    # A pair can lower debt, so the advisory must not claim it never changes.
    assert "Total debt is unchanged" not in output


def test_main_does_not_pair_moves_on_a_restricted_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A partial scan makes every unscanned key look vanished, so it must not
    absorb a genuinely new over-complex function as a `moved` one."""
    _require_ruff()
    _write_tree(tmp_path, complex_function=True)
    baseline = tmp_path / "baseline.json"
    # Same function NAME, higher complexity, in a file this run never scans:
    # a perfect pairing candidate that the restricted run must refuse to use.
    checker.save_baseline(baseline, {"never/scanned.py::tangled": [40]})

    exit_code = _run_main(tmp_path, baseline)
    output = _clean_output(capsys)

    assert exit_code == 1
    assert "🚚 Moved (same function, new file): 0" in output
    assert "🆕 New: 1" in output
    assert "sample.py::tangled" in output


def test_main_fails_on_a_worsened_baselined_function(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A baselined function that got MORE complex is a regression, exit 1."""
    _require_ruff()
    _write_tree(tmp_path, complex_function=True)
    baseline = tmp_path / "baseline.json"
    checker.save_baseline(baseline, {"sample.py::tangled": [11]})

    exit_code = _run_main(tmp_path, baseline)
    output = _clean_output(capsys)

    assert exit_code == 1
    assert "📈 Worsened: 1" in output
    assert "🆕 New: 0" in output
    assert "sample.py::tangled: 11 → " in output


def test_main_fails_closed_when_a_full_scan_covers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Over the DEFAULT targets, zero findings against a populated baseline is
    an error — that is the gating path, and it can only mean nothing was scanned."""
    _require_ruff()
    _unrestrict(monkeypatch)
    _write_tree(tmp_path, complex_function=False)
    baseline = tmp_path / "baseline.json"
    checker.save_baseline(baseline, {"sample.py::tangled": [13]})

    exit_code = _run_main(tmp_path, baseline)
    output = _clean_output(capsys)

    assert exit_code == 2
    assert "The scan covered nothing" in output
    assert "No new complexity regressions" not in output


def test_main_only_warns_when_a_restricted_scan_is_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Scanning a subtree with no over-limit functions is legitimate, not an error."""
    _require_ruff()
    _write_tree(tmp_path, complex_function=False)
    baseline = tmp_path / "baseline.json"
    checker.save_baseline(baseline, {"sample.py::tangled": [13]})

    exit_code = _run_main(tmp_path, baseline)
    output = _clean_output(capsys)

    assert exit_code == 0
    assert "No over-limit functions under the restricted target set" in output
    assert "The scan covered nothing" not in output
    assert "No new complexity regressions" in output


def test_main_restricted_run_does_not_advise_updating_the_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keys outside an explicit target set went unscanned, not fixed."""
    _require_ruff()
    _write_tree(tmp_path, complex_function=True)
    baseline = tmp_path / "baseline.json"

    assert _run_main(tmp_path, baseline, "--update-baseline") == 0
    capsys.readouterr()
    # A key from elsewhere in the tree: unscanned by this restricted run.
    entries = checker.load_baseline(baseline)
    entries["other/mod.py::elsewhere"] = [14]
    checker.save_baseline(baseline, entries)

    exit_code = _run_main(tmp_path, baseline)
    output = _clean_output(capsys)

    assert exit_code == 0
    assert "Targets were RESTRICTED" in output
    assert "tighten the baseline" not in output


def test_main_restricted_update_baseline_warns_about_dropped_debt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Writing a baseline from a restricted scan must say what it drops."""
    _require_ruff()
    _write_tree(tmp_path, complex_function=True)
    baseline = tmp_path / "baseline.json"

    assert _run_main(tmp_path, baseline, "--update-baseline") == 0

    assert "RESTRICTED target set" in _clean_output(capsys)


def test_failure_report_lists_only_the_keys_that_truly_vanished(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The 'vanished' heading must not list a key that merely got simpler."""
    baseline = {"a.py::big": [20], "b.py::gone": [14]}
    current = {"a.py::big": [12], "c.py::newone": [11]}
    report = checker.evaluate_ratchet(current, baseline)
    assert report.improved == ["a.py::big", "b.py::gone"]

    exit_code = checker._print_report(report, current, baseline)
    vanished_block = _clean_output(capsys).partition("vanished in the same run")[2]

    assert exit_code == 1
    assert "b.py::gone" in vanished_block
    # Still over the limit at 12 — simplified, not vanished.
    assert "a.py::big" not in vanished_block


def test_failure_hint_points_at_the_regressions_not_the_vanished_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The closing hint must name the ❌ keys, not the ✨ list printed above it."""
    baseline = {"b.py::gone": [14]}
    current = {"c.py::newone": [11]}
    report = checker.evaluate_ratchet(current, baseline)
    assert report.new == ["c.py::newone"]

    assert checker._print_report(report, current, baseline) == 1
    output = _clean_output(capsys)

    assert "regressions at the TOP" in output
    # The ✨ keys are explicitly marked as the opposite of the failure.
    assert "IS the right response for those" in output
    # Never the old wording, which read as an accusation against the ✨ key.
    assert "function(s) above are genuinely new complexity" not in output


def test_failure_hint_omits_the_vanished_clause_when_nothing_vanished(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With no ✨ list printed, the hint must not refer to one."""
    baseline = {"a.py::big": [20]}
    current = {"a.py::big": [20], "b.py::newone": [11]}
    report = checker.evaluate_ratchet(current, baseline)

    assert checker._print_report(report, current, baseline) == 1
    output = _clean_output(capsys)

    assert "regressions at the TOP" in output
    assert "vanished" not in output


def test_restricted_failure_hint_does_not_invite_updating_the_baseline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On a partial scan the ✨ keys went unread, so they must not be tightened."""
    baseline = {"a.py::big": [20], "other/mod.py::elsewhere": [14]}
    current = {"a.py::big": [20], "a.py::newone": [11]}
    report = checker.evaluate_ratchet(current, baseline, pair_moves=False)

    assert checker._print_report(report, current, baseline, restricted=True) == 1
    output = _clean_output(capsys)

    assert "other/mod.py::elsewhere" in output
    assert "went unread" in output
    assert "IS the right response for those" not in output


def test_main_reports_a_ruff_failure_as_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ruff invocation that cannot run must fail CLOSED, never green."""
    baseline = tmp_path / "baseline.json"
    checker.save_baseline(baseline, {"sample.py::tangled": [13]})

    def broken_ruff(targets: object, project_root: object) -> str:
        """Stand-in for a ruff that could not be launched."""
        raise RuntimeError("ruff failed (exit 1 but reported nothing)")

    monkeypatch.setattr(checker, "run_ruff", broken_ruff)

    exit_code = _run_main(tmp_path, baseline)

    assert exit_code == 2
    assert "reported nothing" in _clean_output(capsys)


# ---------------------------------------------------------------------------
# Drift guard
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


# ---------------------------------------------------------------------------
# Repo-level gate (this is what actually enforces the ratchet in CI)
# ---------------------------------------------------------------------------


def test_repository_has_no_complexity_regressions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real tree must not add (or worsen) an over-limit function.

    Deliberately fails closed: a missing baseline is NOT skipped, it reports
    every function as new — deleting the baseline must not turn the only
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
        "ruff reported no over-limit function anywhere under "
        f"{checker.DEFAULT_TARGETS}, while {baseline_path} baselines "
        f"{len(baseline)}. The scan covered nothing."
    )
    checker.ensure_baselined_files_were_scanned(baseline, scanned_files, PROJECT_ROOT)

    report = checker.evaluate_ratchet(current, baseline)

    hint = (
        "Simplify the function(s) listed above (extract helpers, flatten "
        "branches). If the change is legitimate, run "
        "`hatch run check-complexity --update-baseline`."
    )
    assert not report.new, f"Newly over-complex functions: {report.new}. {hint}"
    assert not report.worsened, (
        f"Functions that got more complex: {report.worsened}. {hint}"
    )
