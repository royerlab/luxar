"""Byte-compile smoke test for the gsplats research demos.

Unlike ``luxar/demos`` (importable ``main()`` modules that the public ``luxar
demo`` registry discovers), the research/dev scripts under ``luxar/gsplats/demos``,
``luxar/gsplats/seeds/demos`` and ``luxar/gsplats/multiscale/demos`` are
run-on-import: most have no ``if __name__ == "__main__"`` guard and execute a
full fit at module scope, so importing them in a test would trigger minutes of
work (and sometimes a viewer). This test byte-compiles each one instead, which
catches syntax and parse regressions across the demo code cheaply, with no
network, GPU, or viewer. (It does not catch a renamed ``luxar.gsplats`` import —
that would need execution — but these trees are source-tree-only and excluded
from the wheel, so they no longer ship.)
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

# .../luxar/gsplats/demos/tests/  ->  parents[2] == .../luxar/gsplats
_GSPLATS_DIR = Path(__file__).resolve().parents[2]
_DEMO_DIRS = [
    _GSPLATS_DIR / "demos",
    _GSPLATS_DIR / "seeds" / "demos",
    _GSPLATS_DIR / "multiscale" / "demos",
]
_DEMO_SCRIPTS = sorted(
    p for d in _DEMO_DIRS for p in d.glob("demo_*.py") if not p.name.startswith("_")
)

# Sanity: the glob must actually find the demos (guards against a bad path).
assert _DEMO_SCRIPTS, f"no demo_*.py found under {[str(d) for d in _DEMO_DIRS]}"


@pytest.mark.parametrize("script", _DEMO_SCRIPTS, ids=lambda p: p.stem)
def test_demo_byte_compiles(script: Path, tmp_path: Path) -> None:
    """Each gsplats demo parses and byte-compiles without error."""
    # Write the .pyc into tmp_path so the test never touches the source tree
    # (keeps it hermetic on a read-only checkout).
    cfile = str(tmp_path / (script.stem + ".pyc"))
    try:
        py_compile.compile(str(script), cfile=cfile, doraise=True)
    except py_compile.PyCompileError as exc:  # pragma: no cover - failure path
        pytest.fail(f"{script.name} failed to compile: {exc}")
