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
GIT_GREP_PATTERN = (
    r"(hatch|\$\(HATCH\))[[:space:]]+"
    r"(((-e|--env)(=|[[:space:]]+)[[:alnum:]_.-]+[[:space:]]+)?run)"
    r"[^;&|]*"
    r"generate_(test_data|expectations)\.py"
)
FIXTURE_COMMAND = re.compile(
    r"(?:hatch|\$\(HATCH\))\s+"
    r"(?:(?:-e|--env)(?:\s+|=)(?P<selected_env>[\w.-]+)\s+)?"
    r"run\s+(?P<run_target>[\w.:-]+)[^;&|\n]*?"
    r"(?P<script>generate_(?:test_data|expectations)\.py)\b"
)


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    """Parsed project configuration."""
    with (REPO / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


@pytest.fixture(scope="module")
def fixture_command_lines() -> list[tuple[str, int, str]]:
    """Tracked lines that invoke a viewer fixture generator through Hatch."""
    result = subprocess.run(
        [
            "git",
            "grep",
            "-I",
            "-n",
            "-E",
            GIT_GREP_PATTERN,
            "--",
            ".",
            f":(exclude){GUARD_PATH}",
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        result.check_returncode()

    matches = []
    for output_line in result.stdout.splitlines():
        relative_path, line_number, line = output_line.split(":", 2)
        matches.append((relative_path, int(line_number), line))
    return matches


def _uses_dedicated_fixture_environment(match: re.Match[str]) -> bool:
    selected_env = match.group("selected_env")
    run_target = match.group("run_target")
    return (selected_env == "fixtures" and run_target == "python") or (
        selected_env is None and run_target == "fixtures:python"
    )


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


def test_every_fixture_generator_uses_the_dedicated_environment(
    fixture_command_lines: list[tuple[str, int, str]],
) -> None:
    """No local, package, or CI entry point may silently rebuild `default`."""
    legacy_calls: list[str] = []
    dedicated_calls: set[str] = set()

    for relative_path, line_number, line in fixture_command_lines:
        for match in FIXTURE_COMMAND.finditer(line):
            command = match.group(0)
            script = match.group("script")
            if not _uses_dedicated_fixture_environment(match):
                legacy_calls.append(f"{relative_path}:{line_number}: {command}")
            else:
                dedicated_calls.add(script)

    assert not legacy_calls, "\n".join(legacy_calls)
    assert dedicated_calls == {Path(script).name for script in FIXTURE_SCRIPTS}


@pytest.mark.parametrize(
    ("line", "expected_dedicated"),
    [
        (
            "hatch run fixtures:python pkg/generate_test_data.py",
            True,
        ),
        (
            "$(HATCH) run fixtures:python pkg/generate_expectations.py",
            True,
        ),
        (
            "hatch run fixtures:python scripts/check_fixture_env.py && "
            "hatch run python pkg/generate_test_data.py",
            False,
        ),
        (
            "hatch -e fixtures run python pkg/generate_test_data.py",
            True,
        ),
        (
            "hatch --env fixtures run python pkg/generate_expectations.py",
            True,
        ),
        (
            "hatch --env=fixtures run python pkg/generate_expectations.py",
            True,
        ),
        (
            "hatch -e default run python pkg/generate_test_data.py",
            False,
        ),
    ],
)
def test_fixture_command_parser_checks_the_selected_environment(
    line: str,
    expected_dedicated: bool,
) -> None:
    """Each supported Hatch spelling must be found and classified positionally."""
    grep_result = subprocess.run(
        ["grep", "-Eq", GIT_GREP_PATTERN],
        input=f"{line}\n",
        text=True,
        check=False,
    )
    assert grep_result.returncode == 0

    matches = list(FIXTURE_COMMAND.finditer(line))
    assert len(matches) == 1
    assert _uses_dedicated_fixture_environment(matches[0]) is expected_dedicated


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


def test_fixture_guard_inputs_trigger_python_ci(
    fixture_command_lines: list[tuple[str, int, str]],
) -> None:
    """Every non-Python input policed by this module must run python-tests."""
    workflow = yaml.safe_load(
        (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    classify_steps = [
        step
        for step in workflow["jobs"]["changes"]["steps"]
        if step.get("id") == "filter"
    ]
    assert len(classify_steps) == 1
    dom_py_pattern = re.search(
        r"grep -qE '([^']+)' && dom_py=true", classify_steps[0]["run"]
    )
    assert dom_py_pattern is not None

    guard_inputs = {
        relative_path
        for relative_path, _, _ in fixture_command_lines
        if not relative_path.endswith(".py")
        and relative_path != ".github/workflows/ci.yml"
    }
    for relative_path in sorted(guard_inputs):
        classified = subprocess.run(
            ["grep", "-Eq", dom_py_pattern.group(1)],
            input=f"{relative_path}\n",
            text=True,
            check=False,
        )
        assert classified.returncode == 0, f"dom_py does not classify {relative_path}"
