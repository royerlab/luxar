"""Static guards for the CLI-to-library boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import luxar

PROD_ROOT = Path(luxar.__file__).resolve().parent


def _production_files() -> list[Path]:
    return [
        path for path in sorted(PROD_ROOT.rglob("*.py")) if "tests" not in path.parts
    ]


def _registered_command_names(trees: dict[Path, ast.AST]) -> set[str]:
    names: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr in {"command", "callback"}
                    for decorator in node.decorator_list
                ):
                    names.add(node.name)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Call)
                and isinstance(node.func.func, ast.Attribute)
                and node.func.func.attr in {"command", "callback"}
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                names.add(node.args[0].id)
    return names


def test_production_code_does_not_call_registered_cli_functions() -> None:
    """Typer command defaults are sentinels, so commands are not library APIs."""
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in _production_files()
    }
    command_names = _registered_command_names(trees)
    offenders = [
        f"{path.relative_to(PROD_ROOT)}:{node.lineno}"
        for path, tree in trees.items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in command_names
    ]

    assert not offenders, (
        "production code calls registered Typer command functions directly; "
        f"extract a plain library boundary instead: {offenders}"
    )
