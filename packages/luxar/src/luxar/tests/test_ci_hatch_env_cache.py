"""Exercise the persistent Hatch cache setup used by CI jobs."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[5]
SCRIPT = REPO / "scripts/ci_hatch_env_cache.sh"
WORKFLOWS = (
    ("ci.yml", "python-tests"),
    ("ci.yml", "typescript-tests"),
    ("coverage.yml", "coverage"),
    ("cuda-nightly.yml", "cuda-native"),
)


def _run_cache(tmp_path: Path, week: str, *, enabled: bool = True) -> tuple[str, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_date = fake_bin / "date"
    fake_date.write_text(f"#!/bin/sh\nprintf '%s\\n' '{week}'\n", encoding="utf-8")
    fake_date.chmod(0o755)
    github_env = tmp_path / "github_env"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GITHUB_ENV"] = str(github_env)
    if enabled:
        env["LUXAR_CI_HATCH_BASE"] = str(tmp_path / "cache")
    else:
        env.pop("LUXAR_CI_HATCH_BASE", None)
    result = subprocess.run(
        ["bash", str(SCRIPT), "dependency-hash"],
        cwd=REPO,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return github_env.read_text() if github_env.exists() else "", tmp_path / "cache"


def test_cache_reuses_hot_directory_refreshes_weekly_and_prunes_old_keys(
    tmp_path: Path,
) -> None:
    first_env, base = _run_cache(tmp_path, "2026-W40")
    first = (
        base
        / f"py{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}-2026-W40-dependency-hash"
    )
    assert first_env == f"HATCH_DATA_DIR={first}\n"
    assert first.is_dir()

    old_time = time.time() - 9 * 86400
    os.utime(first, (old_time, old_time))
    reused_env, _ = _run_cache(tmp_path, "2026-W40")
    assert reused_env.endswith(f"HATCH_DATA_DIR={first}\n")
    assert first.stat().st_mtime > old_time

    expired = base / "py3.12.0-2026-W38-other-hash"
    expired.mkdir()
    os.utime(expired, (old_time, old_time))
    legacy = base / "py3.12.0-old-hash"
    legacy.mkdir()
    os.utime(legacy, (old_time, old_time))
    recent = base / "py3.12.0-2026-W39-other-hash"
    recent.mkdir()
    next_env, _ = _run_cache(tmp_path, "2026-W41")
    assert "2026-W41-dependency-hash" in next_env
    assert first.is_dir() and recent.is_dir()
    assert not expired.exists() and not legacy.exists()


def test_hosted_runner_does_not_set_hatch_cache(tmp_path: Path) -> None:
    github_env, base = _run_cache(tmp_path, "2026-W40", enabled=False)
    assert github_env == ""
    assert not base.exists()


@pytest.mark.parametrize(("workflow_name", "job_name"), WORKFLOWS)
def test_persistent_jobs_use_one_cache_script(
    workflow_name: str, job_name: str
) -> None:
    workflow = yaml.safe_load(
        (REPO / ".github/workflows" / workflow_name).read_text(encoding="utf-8")
    )
    steps = workflow["jobs"][job_name]["steps"]
    setup = next(
        step
        for step in steps
        if step.get("name") == "Reuse Hatch env on persistent runners"
    )
    assert setup["run"] == (
        'bash "$GITHUB_WORKSPACE/scripts/ci_hatch_env_cache.sh" '
        "\"${{ hashFiles('pyproject.toml') }}\""
    )
    install = next(step for step in steps if step.get("name") == "Install Hatch")
    assert setup.get("if") == install.get("if")


def test_audit_does_not_install_into_shared_hatch_environment() -> None:
    workflow = yaml.safe_load(
        (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["python-tests"]["steps"]
    audit = next(
        step for step in steps if step.get("name") == "Audit Python dependencies"
    )
    assert "pipx run" in audit["run"]
    assert "--path" in audit["run"]
    assert "hatch run python -m pip install" not in audit["run"]


def test_typescript_hatch_uses_the_keyed_interpreter() -> None:
    workflow = yaml.safe_load(
        (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["typescript-tests"]["steps"]
    setup_python = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    )
    install = next(step for step in steps if step.get("name") == "Install Hatch")
    assert setup_python["id"] == "setup-python"
    assert install["run"] == (
        'pipx install --python "${{ steps.setup-python.outputs.python-path }}" hatch'
    )


@pytest.mark.parametrize("job_name", ["docs-quality", "e2e-tests"])
def test_hosted_only_jobs_have_no_persistent_cache_step(job_name: str) -> None:
    workflow = yaml.safe_load(
        (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"][job_name]
    assert job["runs-on"] == "ubuntu-latest"
    assert all(
        "HATCH_DATA_DIR" not in str(step.get("run", ""))
        and "ci_hatch_env_cache.sh" not in str(step.get("run", ""))
        for step in job["steps"]
    )
