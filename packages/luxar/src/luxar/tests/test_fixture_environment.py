"""Guard the lean Python environment used to generate viewer fixtures.

Fixture generation is part of every TypeScript test run.  Routing even one
entry point through Hatch's default development environment recreates the
multi-gigabyte CUDA/tooling install that this dedicated environment avoids.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[5]
FIXTURE_SCRIPTS = (
    "packages/luxar-viewer/tests/fixtures/generate_test_data.py",
    "packages/luxar-viewer/tests/fixtures/generate_expectations.py",
)
COMMAND_FILES = (
    ".github/workflows/ci.yml",
    "Makefile",
    "packages/luxar-viewer/package.json",
    "packages/luxar-viewer/src/tests/global-setup.ts",
    "packages/luxar-viewer/tests/fixtures/README.md",
    *FIXTURE_SCRIPTS,
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

    for relative_path in COMMAND_FILES:
        text = (REPO / relative_path).read_text(encoding="utf-8")
        for script in FIXTURE_SCRIPTS:
            command_pattern = (
                rf"(?:hatch|\$\(HATCH\)) run ([^\n\"'&;]*?){re.escape(script)}"
            )
            for match in re.finditer(command_pattern, text):
                command = match.group(0)
                if "fixtures:python" not in command:
                    legacy_calls.append(f"{relative_path}: {command}")
                else:
                    dedicated_calls.add(script)

    assert not legacy_calls, "\n".join(legacy_calls)
    assert dedicated_calls == set(FIXTURE_SCRIPTS)


def test_typescript_ci_verifies_lean_cpu_torch_without_caching_pip() -> None:
    """CI must prove the resolver selected the lean CPU env and skip wheel caching."""
    workflow = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    python_job = workflow.split("\n  python-tests:", 1)[1].split(
        "\n  typescript-tests:", 1
    )[0]
    typescript_job = workflow.split("\n  typescript-tests:", 1)[1].split(
        "\n  release-readiness:", 1
    )[0]
    e2e_job = workflow.split("\n  e2e-tests:", 1)[1]

    assert "torch.__version__.endswith('+cpu')" in typescript_job
    assert "torch.version.cuda is None" in typescript_job
    assert "('napari', 'PyQt6', 'ruff', 'mypy')" in typescript_job
    assert "importlib.util.find_spec(name) is None" in typescript_job
    assert "cache: pip" not in python_job
    assert "cache: pip" not in typescript_job
    assert "cache: pip" not in e2e_job
