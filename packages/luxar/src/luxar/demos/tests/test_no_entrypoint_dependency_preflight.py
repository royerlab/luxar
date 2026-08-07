"""Guard: no demo may gate an optional dependency at its entry point.

A preflight inside ``main()`` — ``try: import umap / except ImportError:
aprint(...); sys.exit(1)`` — refuses to run on a machine that holds every
artifact it needs, because these demos cache their expensive results
(``cache_computed`` / ``cached_download``) and a warm cache never touches the
dependency that produced it. Six demos shipped that shape, and the ESM-3 demo
went further and printed advice ("a complete cached embeddings file skips the
model entirely") that its own preflight made impossible to follow.

The fix is :func:`luxar.demos.require_module`, called at the point of use. This
test is what keeps it fixed: it fails if a preflight reappears in any of the
shapes below, so the class stays closed rather than being re-fixed demo by demo.

The guard recognises the preflight in every equivalent disguise: it flags a
``try`` block whose BODY (a) loads a dependency — an ``import`` /
``from ... import`` / ``__import__`` / ``importlib.import_module`` *or* a
:func:`require_module` call — and that (b) has a handler naming ``ImportError``,
its subclass ``ModuleNotFoundError``, or :class:`~luxar.demos.MissingDependencyError`
(the error ``require_module`` raises), including a tuple like
``except (ImportError, ValueError):`` and an attribute form
``except builtins.ImportError:``, and that (c) exits the process (``sys.exit``,
``os._exit``, ``exit``/``quit``, or ``raise SystemExit``). The entry points are
``main()``, the module's executable top-level statements, and the
``if __name__ == "__main__":`` block (demos run as ``python -m ...``) — plus,
transitively, every module-local helper any of them calls by bare name. A
scanned module that defines NO entry point of its own — today exactly
``_graph_common.py``, ``_interop_common.py``, ``_roundtrip_common.py`` and
``registry.py``, whose callers live in the demos that import them — has every
module-level ``def`` treated as reachable instead. So a gate moved into one of
*those* modules stays guarded; the precondition is the absence of an entry
point, not the ``_common`` name (see the last bullet below for the case it does
not cover). Nested
``def``s inside a reachable function are scanned too, conservatively, whether
or not the closure is provably called: skipping them would reopen the bypass
of hiding the preflight in an immediately-invoked local closure.

Not covered on purpose:

- Gates on a *mandatory* dependency, and soft checks that degrade a feature
  instead of exiting (they do not exit the process).
- Exit gates spelled as an ``if``-guard on a dependency probe rather than a
  ``try``/``except`` — e.g. ``if not is_installed("x"): sys.exit(1)`` or
  ``if importlib.util.find_spec("x") is None: sys.exit(1)``.
- Indirection the deliberately-simple static analysis cannot resolve: helpers
  invoked via an alias or through an attribute / method call, and exception
  names bound to a local alias.
- Exits routed indirectly: through a local ``sys.exit`` wrapper function or a
  ``return`` (not an exit call in the handler), ``raise builtins.SystemExit``
  (attribute form), or an exit placed in a ``finally:`` rather than the
  ``except`` handler.
- ``except*`` (:class:`ast.TryStar`) handlers — moot on the Python 3.10 floor,
  where ``except*`` is a syntax error (3.11+).
- Reversed-operand or ``and``-compound spellings of the ``__name__`` guard;
  only the idiomatic ``if __name__ == "__main__":`` is recognised.
- A gate exported from a module that HAS its own entry point: a function in a
  ``main()``-bearing module that only its *importers* call is unreachable from
  that module's own ``main()``, so it is not scanned. This is a real shape —
  ``demo_gsplats_lod_tribolium.py``, ``demo_gsplats_lod_embryo_line.py`` and
  ``demo_gsplats_recipes_tribolium.py`` all import from
  ``demo_gsplats_3d_tribolium_embryo.py``, and
  ``demo_particle_collision_animated.py`` from ``demo_particle_collision.py``.
  Seeding every ``def`` unconditionally would close it, but at the cost of
  flagging genuinely dead code as an entry-point preflight, which
  ``test_the_guard_still_ignores_unreached_helpers_in_a_real_demo`` pins against.
  Putting shared code in a module without a ``main()`` (the ``_*_common.py``
  pattern) keeps it inside the guard.

These are accepted limits, not full soundness: within the idiomatic
``try``/``except`` preflight shape the guard is closed, and every demo today is
written in that shape (the parametrized real-demo scan proves it stays at zero).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ._scanned_modules import scanned_demo_modules

DEMOS_DIR = Path(__file__).resolve().parent.parent
ENTRY_POINTS = {"main"}


def _demo_modules() -> list[Path]:
    # Includes the shared `_*_common.py` helpers, not just `demo_*.py` — see
    # `_scanned_modules` for why the set is explicit rather than a blanket glob.
    return scanned_demo_modules(DEMOS_DIR)


def _exits(node: ast.AST) -> bool:
    """Does this handler terminate the process instead of raising?

    Recognises ``sys.exit(...)`` / ``os._exit(...)`` (attribute ``exit`` /
    ``_exit``), bare ``exit()`` / ``quit()``, ``raise SystemExit(...)``, and a
    bare ``raise SystemExit`` (no call).
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Attribute) and func.attr in {"exit", "_exit"}:
                return True
            if isinstance(func, ast.Name) and func.id in {"exit", "quit"}:
                return True
        if isinstance(sub, ast.Raise):
            exc = sub.exc
            if (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == "SystemExit"
            ):
                return True
            if isinstance(exc, ast.Name) and exc.id == "SystemExit":
                return True
    return False


_IMPORT_ERROR_NAMES = {"ImportError", "ModuleNotFoundError", "MissingDependencyError"}


def _handler_catches_import_error(handler: ast.ExceptHandler) -> bool:
    """Does this handler name an import-failure error?

    Walks the handler's type node for an :class:`ast.Name` (``ImportError``) or
    an :class:`ast.Attribute` (``builtins.ImportError``) whose leaf is in
    :data:`_IMPORT_ERROR_NAMES` — so ``ImportError``, its subclass
    ``ModuleNotFoundError``, and ``MissingDependencyError`` (raised by
    ``require_module``) all match, in both bare and tuple
    (``except (ImportError, X):``) form. A bare ``except:`` (``handler.type is
    None``) and a broad ``except Exception:`` are exempt.
    """
    if handler.type is None:
        return False
    for node in ast.walk(handler.type):
        if isinstance(node, ast.Name) and node.id in _IMPORT_ERROR_NAMES:
            return True
        if isinstance(node, ast.Attribute) and node.attr in _IMPORT_ERROR_NAMES:
            return True
    return False


def _walk_same_scope(node: ast.AST) -> "list[ast.AST]":
    """``node`` and its descendants, NOT descending into nested def/lambda.

    An ``import`` inside a nested ``def``/``lambda`` belongs to that inner
    scope, not to the ``try`` body, so it must not count as a body load.
    """
    result: list[ast.AST] = [node]
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        result.extend(_walk_same_scope(child))
    return result


def _is_dependency_call(call: ast.Call) -> bool:
    """Is this call a dependency load (``require_module`` / dynamic import)?"""
    func = call.func
    if isinstance(func, ast.Name) and func.id in {"__import__", "require_module"}:
        return True
    if isinstance(func, ast.Attribute) and func.attr in {
        "require_module",
        "import_module",
    }:
        return True
    return False


def _loads_dependency(try_node: ast.Try) -> bool:
    """Does the ``try`` BODY (only) load an optional dependency?

    A dependency load is an ``import`` / ``from ... import`` / ``__import__`` /
    ``importlib.import_module`` *or* a :func:`require_module` call (which itself
    raises ImportError / MissingDependencyError when the module is absent).
    Only ``try_node.body`` is scanned — a load in the ``except`` handler, the
    ``else``/``finally`` clauses, or a nested ``def``/``lambda`` does not count.
    A ``def``/``async def`` written directly in the body is skipped too: its
    body does not execute at ``try`` time, so its imports cannot raise there
    (a ``class`` body, which DOES execute, stays in scope).
    """
    for stmt in try_node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for n in _walk_same_scope(stmt):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                return True
            if isinstance(n, ast.Call) and _is_dependency_call(n):
                return True
    return False


def _module_defs(tree: ast.Module) -> dict[str, ast.AST]:
    """Map module-level function name -> its (async) FunctionDef node."""
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _local_callees(fn: ast.AST, defs: dict[str, ast.AST]) -> set[str]:
    """Names of module-level functions called by bare name within ``fn``."""
    names: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in defs
        ):
            names.add(node.func.id)
    return names


def _is_name_guard(node: ast.AST) -> bool:
    """Is this an ``if __name__ <op> ...:`` block (any operator)?"""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    left = node.test.left
    return isinstance(left, ast.Name) and left.id == "__name__"


def _is_main_guard(node: ast.AST) -> bool:
    """Is this a module-level ``if __name__ == "__main__":`` block?

    Requires the ``==`` operator so ``if __name__ != "__main__":`` (whose body
    runs on *import*, not as a script) is not mistaken for an entry point. The
    reversed-operand and ``and``-compound spellings are documented limits.
    """
    if not _is_name_guard(node):
        return False
    test = node.test  # narrowed by _is_name_guard
    assert isinstance(test, ast.Compare)
    if not (test.ops and isinstance(test.ops[0], ast.Eq)):
        return False
    return any(
        isinstance(cmp, ast.Constant) and cmp.value == "__main__"
        for cmp in test.comparators
    )


def _entry_reachable(tree: ast.Module) -> list[ast.AST]:
    """Scan roots: module top-level statements, entry-point functions, and the
    module-local helpers they call.

    Demos run as ``python -m luxar.demos.demo_<name>``, so the module's
    executable top-level statements run at entry — every statement that is not
    a ``def``/``async def``/``class`` becomes a root (this covers a top-level
    ``try``). A ``__name__`` guard is special-cased by direction: the positive
    ``if __name__ == "__main__":`` block runs as a script (scanned); the
    negated / other-operator forms do not run under ``python -m`` and are
    skipped. Bare-name calls to module-level defs — from ``main()``, the
    guard, or any ordinary top-level statement — are followed transitively,
    with a visited set to break cycles, so a preflight moved into a helper
    is still scanned no matter which entry point calls it.

    A module with NO entry point of its own — no ``main()``, no ``__main__``
    guard, nothing called at import time; today ``_graph_common.py``,
    ``_interop_common.py``, ``_roundtrip_common.py`` and ``registry.py`` —
    would otherwise have every one of its functions unreachable, so the walk
    would return nothing and the guard would pass vacuously. That is not a
    hypothetical: moving a gate out of five demos' ``main()``-reachable code
    into one such helper would silently *drop* it from this guard. For those
    modules every module-level ``def`` is therefore
    seeded as a root — the demos reach them by import, so the entry point is
    simply somewhere else.

    Note the precondition: this keys on the ABSENCE of an entry point, not on the
    filename. Should one of those modules ever gain a ``main()``, intra-module
    reachability resumes and an exported-only function stops being scanned — the
    limit recorded in the module docstring's last "not covered" bullet.
    """
    defs = _module_defs(tree)
    roots: list[ast.AST] = []
    seed_names: set[str] = {name for name in ENTRY_POINTS if name in defs}
    has_entry_point = bool(seed_names)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # defs reached via seeds; class bodies are not entry points
        if _is_name_guard(node):
            if _is_main_guard(node):
                has_entry_point = True
                roots.append(node)  # runs as a script on `python -m ...`
                seed_names |= _local_callees(node, defs)
            continue  # negated / other-operator guard runs on import — skip
        roots.append(node)  # ordinary top-level statement runs on `python -m ...`
        seed_names |= _local_callees(node, defs)
    if not has_entry_point:
        # No entry point in THIS module (a shared helper such as _graph_common /
        # _interop_common / _roundtrip_common / registry): its callers' entry
        # points are elsewhere, so treat every module-level function as
        # reachable rather than none.
        seed_names |= set(defs)
    scanned: dict[str, ast.AST] = {}
    stack = list(seed_names)
    while stack:
        name = stack.pop()
        if name in scanned:
            continue
        fn = defs[name]
        scanned[name] = fn
        for callee in _local_callees(fn, defs):
            if callee not in scanned:
                stack.append(callee)
    roots.extend(scanned.values())
    return roots


def _preflights(path: Path) -> list[str]:
    """Import-guard-and-exit blocks reachable from an entry point."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for root in _entry_reachable(tree):
        for node in ast.walk(root):
            if not isinstance(node, ast.Try):
                continue
            if not _loads_dependency(node):
                continue
            for handler in node.handlers:
                if _handler_catches_import_error(handler) and _exits(handler):
                    found.append(f"{path.name}:{node.lineno}")
    # One try can be reached via multiple roots (e.g. the __main__ guard is both
    # a top-level statement and a guard root) or match on two handlers — report
    # each offending location once, preserving discovery order.
    return list(dict.fromkeys(found))


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


def test_the_guard_scans_a_helper_module_with_no_main(tmp_path: Path) -> None:
    """A shared helper's functions must be reachable even without a ``main()``.

    Shaped exactly like ``_roundtrip_common.py``: no entry point, nothing run at
    import time, the gate inside a function the demos call. Rooting the walk at
    ``main()`` alone returned ``[]`` here, so moving a gate out of five demos
    into one helper would have dropped it from this guard entirely.
    """
    helper = tmp_path / "_helper_common.py"
    helper.write_text(
        "import sys\n"
        "def show_something(x):\n"
        "    try:\n"
        "        import matplotlib.pyplot as plt\n"
        "    except ImportError:\n"
        "        print('missing')\n"
        "        sys.exit(1)\n"
        "    return plt\n"
    )
    assert _preflights(helper) == ["_helper_common.py:3"]


def test_the_guard_still_ignores_unreached_helpers_in_a_real_demo(
    tmp_path: Path,
) -> None:
    """The helper seeding must not leak into modules that DO have an entry point.

    A demo with a ``main()`` keeps the reachability rule: a dead function nobody
    calls is not an entry-point preflight.
    """
    demo = tmp_path / "demo_with_dead_code.py"
    demo.write_text(
        "import sys\n"
        "def never_called():\n"
        "    try:\n"
        "        import umap\n"
        "    except ImportError:\n"
        "        sys.exit(1)\n"
        "def main():\n"
        "    return 1\n"
    )
    assert _preflights(demo) == []


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


def test_the_guard_detects_module_not_found_error(tmp_path: Path) -> None:
    """ModuleNotFoundError is an ImportError subclass — catching it is the defect."""
    offender = tmp_path / "demo_mnfe.py"
    offender.write_text(
        "import sys\n"
        "def main():\n"
        "    try:\n"
        "        import umap\n"
        "    except ModuleNotFoundError:\n"
        "        print('missing')\n"
        "        sys.exit(1)\n"
    )
    assert _preflights(offender) == ["demo_mnfe.py:3"]


def test_the_guard_detects_tuple_handler(tmp_path: Path) -> None:
    """A tuple handler naming ImportError still catches the import failure."""
    offender = tmp_path / "demo_tuple.py"
    offender.write_text(
        "import sys\n"
        "def main():\n"
        "    try:\n"
        "        import umap\n"
        "    except (ImportError, RuntimeError):\n"
        "        sys.exit(1)\n"
    )
    assert _preflights(offender) == ["demo_tuple.py:3"]


def test_the_guard_follows_module_local_helper(tmp_path: Path) -> None:
    """A preflight relocated into a helper called by main() is still caught."""
    offender = tmp_path / "demo_helper.py"
    offender.write_text(
        "import sys\n"
        "def _preflight():\n"
        "    try:\n"
        "        import umap\n"
        "    except ImportError:\n"
        "        sys.exit(1)\n"
        "def main():\n"
        "    _preflight()\n"
    )
    assert _preflights(offender) == ["demo_helper.py:3"]


def test_the_guard_detects_require_module_exit_gate(tmp_path: Path) -> None:
    """require_module in a try that exits on ImportError is the same preflight."""
    offender = tmp_path / "demo_reqmod.py"
    offender.write_text(
        "import sys\n"
        "from luxar.demos import require_module\n"
        "def main():\n"
        "    try:\n"
        "        require_module('umap')\n"
        "    except ImportError:\n"
        "        sys.exit(1)\n"
    )
    assert _preflights(offender) == ["demo_reqmod.py:4"]


def test_the_guard_ignores_require_module_soft_check(tmp_path: Path) -> None:
    """require_module in a try that degrades (no exit) stays allowed."""
    soft = tmp_path / "demo_reqmod_soft.py"
    soft.write_text(
        "from luxar.demos import require_module\n"
        "def main():\n"
        "    try:\n"
        "        umap = require_module('umap')\n"
        "    except ImportError:\n"
        "        umap = None  # feature disabled, demo still runs\n"
    )
    assert _preflights(soft) == []


def test_the_guard_ignores_import_only_in_handler(tmp_path: Path) -> None:
    """An import in the EXCEPT handler is not a body load — must not flag.

    The body only computes; the handler imports ``traceback`` before exiting.
    That is diagnostics-on-failure, not an optional-dependency preflight.
    """
    soft = tmp_path / "demo_handler_import.py"
    soft.write_text(
        "import sys\n"
        "def compute():\n"
        "    return 1\n"
        "def main():\n"
        "    try:\n"
        "        compute()\n"
        "    except ImportError:\n"
        "        import traceback\n"
        "        traceback.print_exc()\n"
        "        sys.exit(1)\n"
    )
    assert _preflights(soft) == []


def test_the_guard_scans_the_dunder_main_block(tmp_path: Path) -> None:
    """A helper called only from ``if __name__ == '__main__':`` is scanned."""
    offender = tmp_path / "demo_dunder_main.py"
    offender.write_text(
        "import sys\n"
        "def _preflight():\n"
        "    try:\n"
        "        import umap\n"
        "    except ImportError:\n"
        "        sys.exit(1)\n"
        "def main():\n"
        "    pass\n"
        'if __name__ == "__main__":\n'
        "    _preflight()\n"
        "    main()\n"
    )
    assert _preflights(offender) == ["demo_dunder_main.py:3"]


def test_the_guard_follows_top_level_helper_call(tmp_path: Path) -> None:
    """A helper invoked from a module top-level statement runs at entry too."""
    offender = tmp_path / "demo_top_call.py"
    offender.write_text(
        "import sys\n"
        "def _preflight():\n"
        "    try:\n"
        "        import umap\n"
        "    except ImportError:\n"
        "        sys.exit(1)\n"
        "def main():\n"
        "    pass\n"
        "_preflight()\n"
    )
    assert _preflights(offender) == ["demo_top_call.py:3"]


def test_the_guard_detects_immediately_invoked_nested_preflight(
    tmp_path: Path,
) -> None:
    """A preflight hidden in a local closure main() calls is still the defect.

    This pins the conservative walk into nested ``def`` bodies of reachable
    functions — restricting the scan to same-scope statements would let this
    shape through.
    """
    offender = tmp_path / "demo_nested_closure.py"
    offender.write_text(
        "import sys\n"
        "def main():\n"
        "    def _pf():\n"
        "        try:\n"
        "            import umap\n"
        "        except ImportError:\n"
        "            sys.exit(1)\n"
        "    _pf()\n"
    )
    assert _preflights(offender) == ["demo_nested_closure.py:4"]


def test_the_guard_detects_missing_dependency_error(tmp_path: Path) -> None:
    """MissingDependencyError is what require_module raises — must be caught."""
    offender = tmp_path / "demo_mde.py"
    offender.write_text(
        "import sys\n"
        "from luxar.demos import require_module, MissingDependencyError\n"
        "def main():\n"
        "    try:\n"
        "        require_module('umap')\n"
        "    except MissingDependencyError:\n"
        "        sys.exit(1)\n"
    )
    assert _preflights(offender) == ["demo_mde.py:4"]


def test_the_guard_detects_importlib_import_module(tmp_path: Path) -> None:
    """importlib.import_module is a dynamic import — a dependency load."""
    offender = tmp_path / "demo_importlib.py"
    offender.write_text(
        "import importlib\n"
        "import sys\n"
        "def main():\n"
        "    try:\n"
        "        importlib.import_module('umap')\n"
        "    except ImportError:\n"
        "        sys.exit(1)\n"
    )
    assert _preflights(offender) == ["demo_importlib.py:4"]


def test_the_guard_detects_module_level_preflight(tmp_path: Path) -> None:
    """A top-level try/except-exit runs on `python -m ...` — must be caught."""
    offender = tmp_path / "demo_module_level.py"
    offender.write_text(
        "import sys\n"
        "try:\n"
        "    import umap\n"
        "except ImportError:\n"
        "    sys.exit(1)\n"
        "def main():\n"
        "    pass\n"
    )
    assert _preflights(offender) == ["demo_module_level.py:2"]


def test_the_guard_reports_dunder_main_try_once(tmp_path: Path) -> None:
    """A try inside the __main__ guard is reachable via two roots — report once."""
    offender = tmp_path / "demo_guard_try.py"
    offender.write_text(
        "import sys\n"
        "def main():\n"
        "    pass\n"
        'if __name__ == "__main__":\n'
        "    try:\n"
        "        import umap\n"
        "    except ImportError:\n"
        "        sys.exit(1)\n"
    )
    assert _preflights(offender) == ["demo_guard_try.py:5"]


def test_the_guard_ignores_negated_name_guard(tmp_path: Path) -> None:
    """`if __name__ != "__main__":` runs on import, not as a script — not a gate."""
    soft = tmp_path / "demo_negated_guard.py"
    soft.write_text(
        "import sys\n"
        'if __name__ != "__main__":\n'
        "    try:\n"
        "        import umap\n"
        "    except ImportError:\n"
        "        sys.exit(1)\n"
    )
    assert _preflights(soft) == []


def test_the_guard_ignores_def_in_try_body(tmp_path: Path) -> None:
    """A def in the try body does not execute its imports at try time."""
    soft = tmp_path / "demo_def_in_try.py"
    soft.write_text(
        "import sys\n"
        "def main():\n"
        "    try:\n"
        "        def later():\n"
        "            import umap\n"
        "            return umap\n"
        "    except ImportError:\n"
        "        sys.exit(1)\n"
    )
    assert _preflights(soft) == []


def test_the_guard_still_flags_class_body_import_in_try(tmp_path: Path) -> None:
    """A class body DOES execute at try time — its import can raise, so flag it."""
    offender = tmp_path / "demo_class_in_try.py"
    offender.write_text(
        "import sys\n"
        "def main():\n"
        "    try:\n"
        "        class C:\n"
        "            import umap\n"
        "    except ImportError:\n"
        "        sys.exit(1)\n"
    )
    assert _preflights(offender) == ["demo_class_in_try.py:3"]
