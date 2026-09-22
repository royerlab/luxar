"""Guard the CI cache contract for the full-tree mypy passes."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[5]


def test_python_lint_restores_and_saves_every_mypy_cache() -> None:
    """The one lint leg caches the root and any nested per-pass directories."""
    workflow = yaml.safe_load(
        (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["python-tests"]["steps"]
    checkout_index = next(
        (
            index
            for index, step in enumerate(steps)
            if step.get("uses", "").startswith("actions/checkout@")
        ),
        None,
    )
    lint_index = next(
        (
            index
            for index, step in enumerate(steps)
            if (step.get("run") or "").strip() == "hatch run lint"
        ),
        None,
    )
    install_hatch_index = next(
        (
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Install Hatch"
        ),
        None,
    )

    assert checkout_index is not None, "python-tests must check out the repository"
    assert lint_index is not None, (
        "python-tests must run the declared hatch lint aggregate"
    )
    assert install_hatch_index is not None, "python-tests must install Hatch"
    lint_step = steps[lint_index]
    restore_steps = [
        (index, step)
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/cache/restore@")
        and step.get("with", {}).get("path") == ".mypy_cache"
    ]
    save_steps = [
        (index, step)
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/cache/save@")
        and step.get("with", {}).get("path") == ".mypy_cache"
    ]

    assert len(restore_steps) == 1, (
        "python-tests must restore the whole .mypy_cache exactly once"
    )
    assert len(save_steps) == 1, (
        "python-tests must save the whole .mypy_cache exactly once"
    )
    restore_index, restore_step = restore_steps[0]
    save_index, save_step = save_steps[0]
    assert re.fullmatch(r"actions/cache/restore@[0-9a-f]{40}", restore_step["uses"]), (
        "the cache restore action must be pinned to a full commit SHA"
    )
    assert re.fullmatch(r"actions/cache/save@[0-9a-f]{40}", save_step["uses"]), (
        "the cache save action must be pinned to a full commit SHA"
    )
    assert restore_index > checkout_index, (
        "checkout deletes gitignored paths, so mypy cache restore must follow it"
    )
    assert restore_index < install_hatch_index, (
        "mypy cache restore must precede environment installation"
    )
    assert save_index == lint_index + 1, "mypy cache save must immediately follow lint"
    assert restore_step.get("if") == lint_step.get("if"), (
        "the mypy cache restore must run on exactly the same Python 3.12 leg as lint"
    )
    assert save_step.get("if") == f"always() && {lint_step.get('if')}", (
        "the mypy cache save must run after failed lint on the same Python 3.12 leg"
    )

    key = restore_step["with"]["key"]
    assert "${{ runner.os }}" in key
    assert "${{ matrix.python-version }}" in key
    assert "${{ hashFiles('pyproject.toml') }}" in key, (
        "pyproject.toml contains the mypy configuration and dependency declarations"
    )
    assert "${{ github.sha }}" in key, "each commit must save refreshed mypy state"
    restore_prefix = restore_step["with"].get("restore-keys")
    assert restore_prefix, "mypy cache restore must fall back to a nearby cache entry"
    assert key.startswith(restore_prefix), (
        "the restore prefix must be a prefix of the primary mypy cache key"
    )
    assert save_step["with"]["key"] == key, (
        "mypy cache restore and save must use the same primary key"
    )
