"""Guard the CI cache contract for the full-tree mypy passes."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[5]


def test_python_lint_restores_every_mypy_cache_with_exact_invalidation() -> None:
    """The one lint leg caches the root containing every declared mypy pass."""
    workflow = yaml.safe_load(
        (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["python-tests"]["steps"]
    lint_step = next(step for step in steps if step.get("run") == "hatch run lint")
    cache_steps = [
        step for step in steps if step.get("with", {}).get("path") == ".mypy_cache"
    ]

    assert len(cache_steps) == 1, (
        "python-tests must restore the whole .mypy_cache exactly once so the host "
        "and nested pinned-pass caches share one lifecycle"
    )
    cache_step = cache_steps[0]
    assert re.fullmatch(r"actions/cache@[0-9a-f]{40}", cache_step["uses"]), (
        "the cache action must be pinned to a full commit SHA"
    )
    assert cache_step.get("if") == lint_step.get("if"), (
        "the mypy cache must run on exactly the same Python 3.12 leg as lint"
    )

    key = cache_step["with"]["key"]
    assert "${{ runner.os }}" in key
    assert "${{ matrix.python-version }}" in key
    assert "${{ hashFiles('pyproject.toml') }}" in key, (
        "pyproject.toml contains both the mypy pin and the complete Python "
        "dependency declarations, so both must invalidate the cache"
    )
    assert "restore-keys" not in cache_step["with"], (
        "a partial-key fallback could restore cache data from another mypy or "
        "dependency set"
    )

    with (REPO / "pyproject.toml").open("rb") as stream:
        dev_dependencies = tomllib.load(stream)["project"]["optional-dependencies"][
            "dev"
        ]
    assert sum(dependency.startswith("mypy") for dependency in dev_dependencies) == 1
