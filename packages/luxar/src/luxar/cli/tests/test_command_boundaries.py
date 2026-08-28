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


def _module_name(path: Path) -> str:
    relative = path.relative_to(PROD_ROOT).with_suffix("")
    parts = ("luxar", *relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _registered_commands(trees: dict[Path, ast.AST]) -> dict[str, set[str]]:
    commands: dict[str, set[str]] = {}
    for path, tree in trees.items():
        names: set[str] = set()
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
        if names:
            commands[_module_name(path)] = names
    return commands


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        if prefix is not None:
            return f"{prefix}.{node.attr}"
    return None


def _import_bindings(
    tree: ast.AST, registered: dict[str, set[str]]
) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    module_aliases: dict[str, str] = {}
    command_aliases: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                module_aliases[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                local = alias.asname or alias.name
                imported = f"{node.module}.{alias.name}"
                if imported in registered:
                    module_aliases[local] = imported
                elif alias.name in registered.get(node.module, set()):
                    command_aliases[local] = (node.module, alias.name)
    return module_aliases, command_aliases


def _is_registered_call(
    node: ast.Call,
    *,
    module: str,
    registered: dict[str, set[str]],
    module_aliases: dict[str, str],
    command_aliases: dict[str, tuple[str, str]],
) -> bool:
    if isinstance(node.func, ast.Name):
        target = command_aliases.get(node.func.id)
        return node.func.id in registered.get(module, set()) or (
            target is not None and target[1] in registered.get(target[0], set())
        )

    dotted = _dotted_name(node.func)
    if dotted is None or "." not in dotted:
        return False
    owner, name = dotted.rsplit(".", 1)
    root, _, suffix = owner.partition(".")
    imported_module = module_aliases.get(root)
    if imported_module is None:
        return False
    owner = imported_module + (f".{suffix}" if suffix else "")
    return name in registered.get(owner, set())


def _registered_command_calls(trees: dict[Path, ast.AST]) -> list[str]:
    registered = _registered_commands(trees)
    offenders: list[str] = []

    for path, tree in trees.items():
        module = _module_name(path)
        module_aliases, command_aliases = _import_bindings(tree, registered)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_registered_call(
                node,
                module=module,
                registered=registered,
                module_aliases=module_aliases,
                command_aliases=command_aliases,
            ):
                offenders.append(f"{path.relative_to(PROD_ROOT)}:{node.lineno}")

    return offenders


def test_registered_command_calls_find_imported_forms_without_name_collisions() -> None:
    commands = PROD_ROOT / "cli" / "sample_commands.py"
    consumer = PROD_ROOT / "cli" / "sample_consumer.py"
    trees = {
        commands: ast.parse(
            "def info_dataset(): pass\napp.command('info')(info_dataset)"
        ),
        consumer: ast.parse(
            "from luxar.cli import sample_commands\n"
            "from luxar.cli.sample_commands import info_dataset as imported_info\n"
            "def info_dataset(): pass\n"
            "info_dataset()\n"
            "sample_commands.info_dataset()\n"
            "imported_info()\n"
        ),
    }

    assert _registered_command_calls(trees) == [
        "cli/sample_consumer.py:5",
        "cli/sample_consumer.py:6",
    ]


def test_production_code_does_not_call_registered_cli_functions() -> None:
    """Typer command defaults are sentinels, so commands are not library APIs."""
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in _production_files()
    }
    offenders = _registered_command_calls(trees)

    assert not offenders, (
        "production code calls registered Typer command functions directly; "
        f"extract a plain library boundary instead: {offenders}"
    )
