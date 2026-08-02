"""Drift guard: the docs CLI reference and the live Typer app must match.

Walks the live Typer application (``luxar.cli.main.app``) to enumerate the full
set of public (non-hidden) leaf-command paths, extracts every ``luxar ...``
invocation from the fenced code blocks of ``docs/guides/user/CLI_REFERENCE.md``,
and compares the two sets in BOTH directions:

* a live public command missing from the doc's code blocks fails (new command
  added without documenting it);
* a documented invocation that no longer resolves to a live public command or
  command group fails (command removed, renamed, or hidden without pruning the
  doc).
"""

from __future__ import annotations

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
        if getattr(group, "hidden", False) or getattr(
            group.typer_instance.info, "hidden", False
        ):
            continue
        group_name = group.name or group.typer_instance.info.name
        assert group_name is not None
        paths.extend(_leaf_paths(group.typer_instance, (*prefix, group_name)))
    return paths


def _documented_paths(doc_text: str) -> set[str]:
    """Extract ``luxar ...`` invocation paths from the doc's fenced code blocks.

    An invocation path is the whitespace-joined tokens after ``luxar`` up to the
    first option token (``-...``) or trailing ``#`` comment, so both catalog
    lines (``luxar gsplat fit   # ...``) and help examples
    (``luxar gsplat fit --help``) resolve to ``gsplat fit``. A bare ``luxar``
    (e.g. ``luxar --help``) yields the empty path and is dropped.
    """
    paths: set[str] = set()
    in_block = False
    for line in doc_text.splitlines():
        if line.lstrip().startswith("```"):
            in_block = not in_block
            continue
        tokens = line.split()
        if not in_block or not tokens or tokens[0] != "luxar":
            continue
        path: list[str] = []
        for token in tokens[1:]:
            if token.startswith("-") or token.startswith("#"):
                break
            path.append(token)
        if path:
            paths.add(" ".join(path))
    return paths


def _find_doc() -> Path | None:
    """Locate the CLI reference doc by walking up from this file."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _DOC_RELPATH
        if candidate.is_file():
            return candidate
    return None


def test_docs_and_cli_command_sets_match() -> None:
    """The doc catalog and the live public command tree must agree both ways."""
    doc = _find_doc()
    if doc is None:
        pytest.skip("docs tree not available")

    leaves = set(_leaf_paths(app))
    assert leaves, "no commands discovered on the live Typer app"
    # Group invocations like `luxar gsplat --help` are valid doc content: every
    # proper prefix of a public leaf path names a live command group.
    groups = {
        " ".join(parts[:i])
        for path in leaves
        for parts in [path.split()]
        for i in range(1, len(parts))
    }

    documented = _documented_paths(doc.read_text(encoding="utf-8"))
    assert documented, "no luxar invocations extracted from the CLI reference"

    missing = sorted(leaves - documented)
    assert not missing, (
        "CLI reference "
        f"({_DOC_RELPATH}) is missing these public commands:\n"
        + "\n".join(f"  - luxar {path}" for path in missing)
    )

    stale = sorted(documented - leaves - groups)
    assert not stale, (
        "CLI reference "
        f"({_DOC_RELPATH}) documents commands that no longer exist publicly:\n"
        + "\n".join(f"  - luxar {path}" for path in stale)
    )
