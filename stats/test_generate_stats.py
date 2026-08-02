"""Unit tests for stats/generate_stats.py measurement-completeness guard.

Run explicitly (not part of the default suite):
    hatch run pytest stats/test_generate_stats.py -q
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_stats as gs  # noqa: E402


def _section(
    *,
    coverage: float | None = 50.0,
    incomplete: str | None = None,
    incomplete_kind: str | None = None,
) -> dict[str, Any]:
    """A coverage-bearing (python/typescript) test section."""
    section: dict[str, Any] = {
        "test_files": 10,
        "test_count": 100,
        "test_passed": 100,
        "test_failed": 0,
        "coverage_percent": 0.0 if coverage is None else coverage,
        "incomplete": incomplete,
        "incomplete_kind": incomplete_kind,
    }
    return section


def _rust_section(
    *, incomplete: str | None = None, incomplete_kind: str | None = None
) -> dict[str, Any]:
    return {
        "test_files": 5,
        "test_count": 50,
        "test_passed": 50,
        "test_failed": 0,
        "incomplete": incomplete,
        "incomplete_kind": incomplete_kind,
    }


def _healthy_stats() -> dict[str, Any]:
    return {
        "python": _section(coverage=84.0),
        "typescript": _section(coverage=70.0),
        "rust": _rust_section(),
        "e2e": {"test_files": 3, "test_count": 0},
        "error": None,
    }


# ---------------------------------------------------------------------------
# validate_measurements
# ---------------------------------------------------------------------------


def test_no_tests_returns_empty_even_with_reasons() -> None:
    stats = _healthy_stats()
    stats["python"] = _section(
        coverage=0.0, incomplete="test run timed out", incomplete_kind="timeout"
    )
    stats["typescript"] = _section(
        coverage=0.0, incomplete="npx not found", incomplete_kind="tool_missing"
    )
    # run_tests=False: nothing was requested, so nothing can be incomplete.
    assert gs.validate_measurements(stats, run_tests=False, run_coverage=False) == []
    assert gs.validate_measurements(stats, run_tests=False, run_coverage=True) == []


def test_python_timeout_flagged() -> None:
    stats = _healthy_stats()
    stats["python"] = _section(
        coverage=0.0,
        incomplete="test run timed out after 90 minutes",
        incomplete_kind="timeout",
    )
    problems = gs.validate_measurements(stats, run_tests=True, run_coverage=True)
    assert len(problems) == 1
    assert problems[0].startswith("python:")
    assert "timed out" in problems[0]


def test_missing_toolchain_flagged() -> None:
    stats = _healthy_stats()
    stats["typescript"] = _section(
        coverage=0.0, incomplete="npx not found", incomplete_kind="tool_missing"
    )
    # Tool-missing is a test-run failure: flagged whenever run_tests is True,
    # regardless of run_coverage.
    problems = gs.validate_measurements(stats, run_tests=True, run_coverage=False)
    assert len(problems) == 1
    assert problems[0].startswith("typescript:")
    assert "npx not found" in problems[0]


def test_coverage_problem_gated_on_run_coverage() -> None:
    stats = _healthy_stats()
    # Same state: python coverage 0.0 after an attempted run recorded a
    # coverage-kind failure.
    stats["python"] = _section(
        coverage=0.0,
        incomplete="coverage requested but none was produced",
        incomplete_kind="coverage",
    )
    with_cov = gs.validate_measurements(stats, run_tests=True, run_coverage=True)
    assert len(with_cov) == 1
    assert with_cov[0].startswith("python:")

    without_cov = gs.validate_measurements(stats, run_tests=True, run_coverage=False)
    assert without_cov == []


def test_run_failed_flagged_regardless_of_coverage() -> None:
    stats = _healthy_stats()
    # A non-zero-and-not-1 exit means the run produced no numbers — a run-level
    # failure, NOT a coverage-only one, so it is flagged even under --no-coverage.
    stats["python"] = _section(
        coverage=0.0,
        incomplete="test run exited with code 2 (no measurement produced)",
        incomplete_kind="run_failed",
    )
    with_cov = gs.validate_measurements(stats, run_tests=True, run_coverage=True)
    assert len(with_cov) == 1
    assert with_cov[0].startswith("python:")

    without_cov = gs.validate_measurements(stats, run_tests=True, run_coverage=False)
    assert len(without_cov) == 1
    assert without_cov[0].startswith("python:")

    # Still not requested when --no-tests.
    assert gs.validate_measurements(stats, run_tests=False, run_coverage=True) == []


def test_rust_run_failed_flagged() -> None:
    stats = _healthy_stats()
    stats["rust"] = _rust_section(
        incomplete="test run exited with code 101 and produced no results",
        incomplete_kind="run_failed",
    )
    problems = gs.validate_measurements(stats, run_tests=True, run_coverage=False)
    assert len(problems) == 1
    assert problems[0].startswith("rust:")


def test_healthy_stats_never_flagged() -> None:
    stats = _healthy_stats()
    for run_tests in (True, False):
        for run_coverage in (True, False):
            assert (
                gs.validate_measurements(
                    stats, run_tests=run_tests, run_coverage=run_coverage
                )
                == []
            )


# ---------------------------------------------------------------------------
# run_failed RECORDING paths (drive the real runner logic)
# ---------------------------------------------------------------------------


def _fake_run(returncode: int, stdout: str = "", stderr: str = ""):
    """A stand-in for subprocess.run that ignores its args and returns a
    fixed CompletedProcess-shaped object."""

    def _run(*_args: Any, **_kwargs: Any) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        )

    return _run


def test_python_run_failed_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both the collect-only and the main run come back with a non-(0,1) exit and
    # no parseable coverage/pass lines -> the main run is flagged run_failed.
    monkeypatch.setattr(gs.subprocess, "run", _fake_run(returncode=2))
    test_stats = {"python": gs._empty_test_section()}
    gs._run_python_tests(Path("/nonexistent"), test_stats, run_coverage=True)
    assert test_stats["python"]["incomplete_kind"] == "run_failed"
    assert "code 2" in test_stats["python"]["incomplete"]


def test_python_returncode_1_is_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exit 1 = tests ran but some failed = a COMPLETE measurement. With coverage
    # off (so the 0.0-coverage rule doesn't apply), nothing must be flagged.
    monkeypatch.setattr(gs.subprocess, "run", _fake_run(returncode=1))
    test_stats = {"python": gs._empty_test_section()}
    gs._run_python_tests(Path("/nonexistent"), test_stats, run_coverage=False)
    assert test_stats["python"]["incomplete"] is None
    assert test_stats["python"]["incomplete_kind"] is None


def _rust_stats() -> dict[str, Any]:
    return {
        "rust": {
            "test_files": 0,
            "test_count": 0,
            "test_passed": 0,
            "test_failed": 0,
            "incomplete": None,
            "incomplete_kind": None,
        }
    }


def _rust_root(tmp_path: Path) -> Path:
    (tmp_path / "packages" / "luxar-viewer" / "src" / "wasm" / "rust").mkdir(
        parents=True
    )
    return tmp_path


def test_rust_run_failed_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 101 with no "test result:" line parses zero results -> a crash/compile
    # error that measured nothing.
    monkeypatch.setattr(gs.subprocess, "run", _fake_run(returncode=101, stdout=""))
    test_stats = _rust_stats()
    gs._run_rust_tests(_rust_root(tmp_path), test_stats)
    assert test_stats["rust"]["incomplete_kind"] == "run_failed"


def test_rust_failing_tests_are_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 101 but the summary parses failures (failed > 0) = a real run whose tests
    # failed = COMPLETE. Must NOT be flagged (locks in the `failed == 0` rule).
    stdout = "test result: FAILED. 3 passed; 2 failed; 0 ignored\n"
    monkeypatch.setattr(gs.subprocess, "run", _fake_run(returncode=101, stdout=stdout))
    test_stats = _rust_stats()
    gs._run_rust_tests(_rust_root(tmp_path), test_stats)
    assert test_stats["rust"]["incomplete"] is None
    assert test_stats["rust"]["incomplete_kind"] is None
    assert test_stats["rust"]["test_failed"] == 2


# ---------------------------------------------------------------------------
# weighted_coverage (defensive fallback documentation)
# ---------------------------------------------------------------------------


def test_weighted_coverage_loc_weighted_average() -> None:
    # Two measured languages: LOC-weighted mean.
    #   (80*1000 + 60*3000) / 4000 = 65.0
    assert gs.weighted_coverage(80.0, 1000, 60.0, 3000) == 65.0


def test_weighted_coverage_drops_single_zero_language() -> None:
    # A 0.0 language is treated as unmeasured and dropped (defensive fallback).
    assert gs.weighted_coverage(84.0, 1000, 0.0, 3000) == 84.0
    assert gs.weighted_coverage(0.0, 1000, 70.0, 3000) == 70.0
    # Both zero -> 0.0.
    assert gs.weighted_coverage(0.0, 1000, 0.0, 3000) == 0.0
