"""Spelling guard: public names are American (``optimize``, ``center``).

The pre-release lock-in pass settled the spelling of every public name on
American English (``luxar optimize``, ``--reveal-center``, ``reveal_center``,
``luxar.io.optimize``). Once the package is on PyPI a spelling is an API, so
this test keeps British forms from creeping back into the three surfaces a
user can type or import:

* every Typer command name, option spelling and help string reachable from
  the live ``luxar`` app (hidden commands and options included — pre-release
  there are no legacy aliases to exempt);
* every entry of every ``__all__`` in the ``luxar`` package (not only
  ``__init__.py`` files — a module-level ``__all__`` is just as public);
* the renamed modules themselves: the new names import, the old one is gone.

The walk resolves commands the way Click's own dispatch does (``get_command``
on a live ``click.Context``) rather than via ``isinstance(cmd, click.Group)``,
which is False for every Typer group in this environment — see
``_is_group`` in ``test_docs_command_coverage.py`` for the trap.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
from typing import Iterator

import click
import pytest
import typer

from luxar.cli.main import app

#: Case-insensitive substrings that must not appear in any public spelling.
#: ``optimis`` covers optimise/optimiser/optimisation; ``centre`` covers the
#: ``--reveal-centre`` family. Deliberately NOT ``colour`` etc.: only names
#: that were actually renamed are locked, so the test stays honest about scope.
_BRITISH_FRAGMENTS: tuple[str, ...] = ("optimis", "centre")

_LUXAR_ROOT = Path(__file__).resolve().parents[2]


def _british(text: str) -> str | None:
    """Return the offending fragment if ``text`` contains one, else ``None``."""
    low = text.lower()
    for fragment in _BRITISH_FRAGMENTS:
        if fragment in low:
            return fragment
    return None


def _walk_commands(
    node: object, ctx: click.Context, path: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], object]]:
    """Yield ``(path, command)`` for every command reachable from ``node``.

    Groups are recognised by duck typing on ``get_command``, exactly what Click
    calls at dispatch time; Typer's vendored click shim makes an
    ``isinstance(node, click.Group)`` check silently False for every group.
    """
    yield path, node
    if not hasattr(node, "get_command"):
        return
    for name in node.list_commands(ctx):  # type: ignore[attr-defined]
        child = node.get_command(ctx, name)  # type: ignore[attr-defined]
        assert child is not None, (
            f"{' '.join(path)} lists {name!r} but cannot resolve it"
        )
        yield from _walk_commands(
            child, click.Context(child, parent=ctx), (*path, name)
        )  # type: ignore[arg-type]


def _cli_surface() -> list[tuple[str, str, str]]:
    """Every ``(command path, kind, text)`` a user can see or type on the CLI."""
    root = typer.main.get_command(app)
    surface: list[tuple[str, str, str]] = []
    for path, command in _walk_commands(root, click.Context(root), ("luxar",)):
        where = " ".join(path)
        surface.append((where, "command name", path[-1]))
        surface.append((where, "command help", getattr(command, "help", None) or ""))
        for param in getattr(command, "params", []):
            spellings = [*param.opts, *param.secondary_opts]
            surface.append((where, "option spelling", " ".join(spellings)))
            surface.append((where, "option help", getattr(param, "help", None) or ""))
    return surface


def _all_entries() -> list[tuple[str, str]]:
    """Every string in every ``__all__`` (assignment or ``+=``) under ``luxar``."""
    entries: list[tuple[str, str]] = []
    for source in sorted(_LUXAR_ROOT.rglob("*.py")):
        if "tests" in source.relative_to(_LUXAR_ROOT).parts:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            targets: list[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            else:
                continue
            if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                continue
            for elt in getattr(node.value, "elts", []):
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    entries.append((str(source.relative_to(_LUXAR_ROOT)), elt.value))
    return entries


def test_cli_surface_is_american_spelled() -> None:
    surface = _cli_surface()
    # The walk must have reached the renamed commands, or a silent
    # false negative (an empty walk) would pass this test for free.
    seen = {where for where, _, _ in surface}
    assert "luxar optimize" in seen
    assert "luxar gsplat lod" in seen and "luxar mesh lod" in seen
    assert any(
        kind == "option spelling" and "--reveal-center" in text
        for _, kind, text in surface
    )

    offenders = [
        f"{where} [{kind}] contains {frag!r}: {text[:100]!r}"
        for where, kind, text in surface
        if (frag := _british(text)) is not None
    ]
    assert not offenders, "British spellings on the CLI surface:\n" + "\n".join(
        offenders
    )


def test_public_all_entries_are_american_spelled() -> None:
    entries = _all_entries()
    assert len(entries) > 100, "the __all__ scan found suspiciously few entries"
    assert ("io/optimize.py", "optimize_store") in entries
    offenders = [
        f"{module}: {name!r} contains {frag!r}"
        for module, name in entries
        if (frag := _british(name)) is not None
    ]
    assert not offenders, "British spellings in __all__:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "module_name",
    ["luxar.io.optimize", "luxar.cli.optimize_command"],
)
def test_renamed_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module_name",
    ["luxar.io.optimise", "luxar.cli.optimise_command"],
)
def test_british_module_names_are_gone(module_name: str) -> None:
    """A hard cut, no alias: pre-release there is nobody to keep the old name for."""
    assert importlib.util.find_spec(module_name) is None
