"""Guard the disk-headroom contract for the long Python CI jobs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[5]
WORKFLOWS = (
    (".github/workflows/ci.yml", "python-tests"),
    (".github/workflows/coverage.yml", "coverage"),
)
MINIMUM_FREE_KIB = 25 * 1024 * 1024


def _steps(workflow_path: str, job_name: str) -> list[dict[str, object]]:
    workflow = yaml.safe_load((REPO / workflow_path).read_text(encoding="utf-8"))
    return workflow["jobs"][job_name]["steps"]


def _named_step(steps: list[dict[str, object]], name: str) -> dict[str, object]:
    matches = [step for step in steps if step.get("name") == name]
    assert len(matches) == 1, f"expected exactly one {name!r} step"
    return matches[0]


@pytest.mark.parametrize(("workflow_path", "job_name"), WORKFLOWS)
def test_long_python_jobs_reclaim_disk_on_hosted_runners(
    workflow_path: str, job_name: str
) -> None:
    """The hosted cleanup must not run in an empty self-hosted container."""
    steps = _steps(workflow_path, job_name)
    reclaim = _named_step(steps, "Reclaim disk space")

    assert reclaim["if"] == "runner.environment == 'github-hosted'"
    script = str(reclaim["run"])
    assert "/usr/local/lib/android" in script
    assert "docker image prune --all --force" in script


@pytest.mark.parametrize(("workflow_path", "job_name"), WORKFLOWS)
def test_disk_guard_runs_before_checkout_and_environment_setup(
    workflow_path: str, job_name: str
) -> None:
    """Low disk must fail before checkout or the multi-gigabyte Hatch environment."""
    steps = _steps(workflow_path, job_name)
    reclaim_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Reclaim disk space"
    )
    guard_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Require 25 GiB free disk"
    )
    checkout_index = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    install_index = next(
        index for index, step in enumerate(steps) if step.get("name") == "Install Hatch"
    )

    assert reclaim_index < guard_index < checkout_index < install_index
    guard = steps[guard_index]
    if job_name == "python-tests":
        assert guard["if"] == "needs.changes.outputs.dom_py != 'false'"
    else:
        assert "if" not in guard


def _run_guard(
    tmp_path: Path, script: str, available_kib: int | str, df_exit: int = 0
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    fake_df = fake_bin / "df"
    fake_df.write_text(
        f"#!/bin/sh\nprintf 'Available\\n{available_kib}\\n'\nexit {df_exit}\n",
        encoding="utf-8",
    )
    fake_df.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    return subprocess.run(
        ["bash", "-e", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.parametrize(("workflow_path", "job_name"), WORKFLOWS)
def test_disk_guard_enforces_the_host_danger_line(
    tmp_path: Path, workflow_path: str, job_name: str
) -> None:
    """The guard accepts 25 GiB exactly and emits a clear annotation below it."""
    guard = _named_step(_steps(workflow_path, job_name), "Require 25 GiB free disk")
    script = str(guard["run"])

    accepted = _run_guard(tmp_path / "accepted", script, MINIMUM_FREE_KIB)
    assert accepted.returncode == 0, accepted.stderr

    rejected = _run_guard(tmp_path / "rejected", script, MINIMUM_FREE_KIB - 1)
    assert rejected.returncode == 1
    assert "::error title=Insufficient disk space::" in rejected.stdout
    assert "25 GiB required" in rejected.stdout


@pytest.mark.parametrize(("workflow_path", "job_name"), WORKFLOWS)
@pytest.mark.parametrize(
    ("df_output", "df_exit"),
    [("unavailable", 0), (MINIMUM_FREE_KIB, 1)],
)
def test_disk_guard_reports_measurement_failure(
    tmp_path: Path,
    workflow_path: str,
    job_name: str,
    df_output: int | str,
    df_exit: int,
) -> None:
    guard = _named_step(_steps(workflow_path, job_name), "Require 25 GiB free disk")
    result = _run_guard(tmp_path, str(guard["run"]), df_output, df_exit)

    assert result.returncode == 1
    assert "Available" in result.stdout
    assert "::error title=Disk check failed::" in result.stdout
    assert "::error title=Insufficient disk space::" not in result.stdout
