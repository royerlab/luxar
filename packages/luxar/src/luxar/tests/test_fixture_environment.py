"""Guard the lean Python environment used to generate viewer fixtures.

Fixture generation is part of every TypeScript test run.  Routing even one
entry point through Hatch's default development environment recreates the
multi-gigabyte CUDA/tooling install that this dedicated environment avoids.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[5]
FIXTURE_SCRIPTS = (
    "packages/luxar-viewer/tests/fixtures/generate_test_data.py",
    "packages/luxar-viewer/tests/fixtures/generate_expectations.py",
)
GUARD_PATH = "packages/luxar/src/luxar/tests/test_fixture_environment.py"
FIXTURE_COMMAND = re.compile(
    r"(?:hatch|\$\(HATCH\))\s+run(?P<args>[^\n]*?)"
    r"(?P<script>generate_(?:test_data|expectations)\.py)\b"
)


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    """Parsed project configuration."""
    with (REPO / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def test_fixture_environment_installs_only_cpu_gsplat_dependencies(
    pyproject: dict[str, Any],
) -> None:
    """The fixture env must not inherit the default dev/CUDA dependency closure."""
    fixtures = pyproject["tool"]["hatch"]["envs"]["fixtures"]

    assert fixtures["template"] == "fixtures"
    assert fixtures["features"] == ["gsplats"]
    assert fixtures["env-vars"]["PIP_EXTRA_INDEX_URL"] == (
        "https://download.pytorch.org/whl/cpu"
    )
    assert fixtures["env-vars"]["OMP_NUM_THREADS"] == "{env:OMP_NUM_THREADS:1}"
    assert fixtures["env-vars"]["MKL_NUM_THREADS"] == "{env:MKL_NUM_THREADS:1}"


def test_every_fixture_generator_uses_the_dedicated_environment() -> None:
    """No local, package, or CI entry point may silently rebuild `default`."""
    legacy_calls: list[str] = []
    dedicated_calls: set[str] = set()

    tracked_files = (
        subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    for relative_path in tracked_files:
        if not relative_path or relative_path == GUARD_PATH:
            continue
        path = REPO / relative_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in FIXTURE_COMMAND.finditer(text):
            command = match.group(0)
            script = match.group("script")
            if "fixtures:python" not in match.group("args"):
                legacy_calls.append(f"{relative_path}: {command}")
            else:
                dedicated_calls.add(script)

    assert not legacy_calls, "\n".join(legacy_calls)
    assert dedicated_calls == {Path(script).name for script in FIXTURE_SCRIPTS}


def test_typescript_ci_verifies_lean_cpu_torch_without_caching_pip() -> None:
    """CI must prove the resolver selected the lean CPU env and skip wheel caching."""
    workflow = yaml.safe_load(
        (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    guarded_jobs = ("python-tests", "typescript-tests", "e2e-tests")
    assert set(guarded_jobs) <= jobs.keys()

    verify_steps = [
        step
        for step in jobs["typescript-tests"]["steps"]
        if step.get("name") == "Verify fixture environment is lean and CPU-only"
    ]
    assert len(verify_steps) == 1
    assert verify_steps[0]["run"] == (
        "hatch run fixtures:python scripts/check_fixture_env.py"
    )

    # Four setup-python entries consumed 98% of the 13 GB cache budget and
    # evicted the pnpm/cargo caches that provide a larger wall-clock benefit.
    for job_name in guarded_jobs:
        setup_steps = [
            step
            for step in jobs[job_name]["steps"]
            if str(step.get("uses", "")).startswith("actions/setup-python@")
        ]
        assert setup_steps, f"{job_name} has no setup-python step"
        assert all("cache" not in step.get("with", {}) for step in setup_steps)
