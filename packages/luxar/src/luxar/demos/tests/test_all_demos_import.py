"""Import smoke test across every demo in ``luxar/demos/``.

Now that ``luxar.demos`` is a real importable package (the sys.modules alias was
removed), every ``demo_*.py`` can be imported directly. This parametrized test
imports each one and asserts it exposes a ``main`` entry point — cheaply catching
syntax errors, bad imports, and module-level regressions across the whole suite
without any network, GPU, or viewer.

A demo whose module-level imports need an optional extra that isn't installed is
skipped (not failed): the point is to catch *our* regressions, not missing
third-party packages. Anything else (SyntaxError, AttributeError, NameError, a
luxar-internal ImportError) is a real failure.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_DEMOS_DIR = Path(__file__).resolve().parents[1]
_DEMO_MODULES = sorted(
    p.stem for p in _DEMOS_DIR.glob("demo_*.py") if not p.name.startswith("_")
)

# Sanity: the glob must actually find the suite (guards against a bad path).
assert _DEMO_MODULES, f"no demo_*.py found under {_DEMOS_DIR}"


@pytest.mark.parametrize("module_name", _DEMO_MODULES)
def test_demo_imports_and_has_main(module_name: str) -> None:
    """Each demo imports cleanly and exposes a callable ``main``."""
    try:
        module = importlib.import_module(f"luxar.demos.{module_name}")
    except ModuleNotFoundError as exc:
        # An uninstalled optional extra (pandas/umap/anndata/esm/…) — not our bug.
        pytest.skip(f"optional dependency missing for {module_name}: {exc}")

    assert hasattr(module, "main"), f"{module_name} has no main() entry point"
    assert callable(module.main), f"{module_name}.main is not callable"
