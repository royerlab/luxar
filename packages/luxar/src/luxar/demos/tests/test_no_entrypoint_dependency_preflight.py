"""Guard: no demo may gate an optional dependency at its entry point.

A preflight inside ``main()`` — ``try: import umap / except ImportError:
aprint(...); sys.exit(1)`` — refuses to run on a machine that holds every
artifact it needs, because these demos cache their expensive results
(``cache_computed`` / ``cached_download``) and a warm cache never touches the
dependency that produced it. Six demos shipped that shape, and the ESM-3 demo
went further and printed advice ("a complete cached embeddings file skips the
model entirely") that its own preflight made impossible to follow.

The fix is :func:`luxar.demos.require_module`, called at the point of use. This
test is what keeps it fixed: it fails if a preflight reappears anywhere, so the
whole class stays closed rather than being re-fixed demo by demo.

Not covered on purpose: gates on a *mandatory* dependency, and soft checks that
degrade a feature instead of exiting (they do not call ``sys.exit``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DEMOS_DIR = Path(__file__).resolve().parent.parent
ENTRY_POINTS = {"main"}


def _demo_modules() -> list[Path]:
    return sorted(DEMOS_DIR.glob("demo_*.py"))


def _exits(node: ast.AST) -> bool:
    """Does this handler terminate the process instead of raising?"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Attribute) and func.attr == "exit":
                return True
            if isinstance(func, ast.Name) and func.id in {"exit", "quit"}:
                return True
        if isinstance(sub, ast.Raise) and isinstance(sub.exc, ast.Call):
            if isinstance(sub.exc.func, ast.Name) and sub.exc.func.id == "SystemExit":
                return True
    return False


def _preflights(path: Path) -> list[str]:
    """Import-guard-and-exit blocks inside an entry-point function."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name not in ENTRY_POINTS:
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Try):
                continue
            imports_something = any(
                isinstance(n, (ast.Import, ast.ImportFrom))
                or (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "__import__"
                )
                for n in ast.walk(node)
            )
            if not imports_something:
                continue
            for handler in node.handlers:
                if handler.type is None:
                    continue
                if "ImportError" not in ast.dump(handler.type):
                    continue
                if _exits(handler):
                    found.append(f"{path.name}:{node.lineno}")
    return found


@pytest.mark.parametrize("path", _demo_modules(), ids=lambda p: p.name)
def test_demo_has_no_entrypoint_dependency_preflight(path: Path) -> None:
    offenders = _preflights(path)
    assert not offenders, (
        f"{path.name} gates an optional dependency at its entry point "
        f"({', '.join(offenders)}). Move it to the point of use with "
        "`from luxar.demos import require_module` — see "
        "luxar/demos/_dependencies.py for why this is a rule."
    )


def test_the_guard_itself_detects_the_pattern(tmp_path: Path) -> None:
    """A guard that cannot fail is worth nothing — prove it catches the shape."""
    offender = tmp_path / "demo_offender.py"
    offender.write_text(
        "import sys\n"
        "def main():\n"
        "    try:\n"
        "        import umap\n"
        "    except ImportError:\n"
        "        print('missing')\n"
        "        sys.exit(1)\n"
    )
    assert _preflights(offender) == ["demo_offender.py:3"]


def test_the_guard_ignores_soft_optional_checks(tmp_path: Path) -> None:
    """Degrading a feature is fine; only refusing to run is the defect."""
    soft = tmp_path / "demo_soft.py"
    soft.write_text(
        "def main():\n"
        "    try:\n"
        "        from PIL import Image\n"
        "    except ImportError:\n"
        "        Image = None  # thumbnails disabled, demo still runs\n"
    )
    assert _preflights(soft) == []


def test_the_guard_ignores_point_of_use_gates(tmp_path: Path) -> None:
    """The sanctioned form must not trip the guard."""
    ok = tmp_path / "demo_ok.py"
    ok.write_text(
        "from luxar.demos import require_module\n"
        "def _compute():\n"
        "    UMAP = require_module('umap').UMAP\n"
        "    return UMAP\n"
        "def main():\n"
        "    _compute()\n"
    )
    assert _preflights(ok) == []


def test_guard_actually_scans_the_demo_suite() -> None:
    """Fail loudly if the glob silently matches nothing (e.g. after a move)."""
    modules = _demo_modules()
    assert len(modules) > 50, f"only found {len(modules)} demo modules in {DEMOS_DIR}"
