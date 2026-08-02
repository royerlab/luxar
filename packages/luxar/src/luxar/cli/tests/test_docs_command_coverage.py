"""Drift guard: every public CLI command must appear in the docs reference.

Walks the live Typer application (``luxar.cli.main.app``) to enumerate the full
set of public (non-hidden) leaf-command paths, then asserts that each one's exact
invocation string ``luxar <path>`` is present in
``docs/guides/user/CLI_REFERENCE.md``. Adding a command to the CLI without
documenting it here makes this test fail, listing every undocumented command at
once.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import typer

from luxar.cli.main import app

_DOC_RELPATH = Path("docs/guides/user/CLI_REFERENCE.md")


def _command_name(command: typer.models.CommandInfo) -> str:
    """Resolve the invocation name of a registered command."""
    if command.name:
        return command.name
    assert command.callback is not None
    return command.callback.__name__.replace("_", "-")


def _leaf_paths(app_instance: typer.Typer, prefix: tuple[str, ...] = ()) -> list[str]:
    """Return space-joined paths of all non-hidden leaf commands under ``app``."""
    paths: list[str] = []
    for command in app_instance.registered_commands:
        if getattr(command, "hidden", False):
            continue
        paths.append(" ".join((*prefix, _command_name(command))))
    for group in app_instance.registered_groups:
        assert group.typer_instance is not None
        group_name = group.name or group.typer_instance.info.name
        assert group_name is not None
        paths.extend(_leaf_paths(group.typer_instance, (*prefix, group_name)))
    return paths


def _find_doc() -> Path | None:
    """Locate the CLI reference doc by walking up from this file."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _DOC_RELPATH
        if candidate.is_file():
            return candidate
    return None


def test_all_public_commands_are_documented() -> None:
    """Every non-hidden leaf command must be catalogued in the CLI reference."""
    doc = _find_doc()
    if doc is None:
        pytest.skip("docs tree not available")

    doc_text = doc.read_text(encoding="utf-8")
    paths = _leaf_paths(app)
    assert paths, "no commands discovered on the live Typer app"

    missing = sorted(
        path
        for path in paths
        if re.search(rf"luxar {re.escape(path)}(?![\w-])", doc_text) is None
    )
    assert not missing, (
        "CLI reference "
        f"({_DOC_RELPATH}) is missing these public commands:\n"
        + "\n".join(f"  - luxar {path}" for path in missing)
    )
