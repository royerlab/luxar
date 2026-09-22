"""Offline tests for the README architecture diagrams.

The drawings are built in memory (no rsvg-convert needed); what is asserted is
that both figures build in both themes at their declared sizes, that the code
panel names only API that exists, and that the layer labels name real modules.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "make_architecture_diagram.py"
SPEC = importlib.util.spec_from_file_location("make_architecture_diagram", SCRIPT)
assert SPEC and SPEC.loader
diag = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diag
SPEC.loader.exec_module(diag)
REPO_ROOT = SCRIPT.parents[2]
PY_PKG = REPO_ROOT / "packages/luxar/src/luxar"
VIEWER_SRC = REPO_ROOT / "packages/luxar-viewer/src"


@pytest.mark.parametrize("theme", [diag.DARK, diag.LIGHT])
def test_both_figures_build_at_their_declared_size(theme: diag.Theme) -> None:
    pipeline = diag.build(theme).as_svg()
    layers = diag.build_layers(theme).as_svg()
    assert f'width="{diag.W}"' in pipeline and f'height="{diag.H}"' in pipeline
    assert (
        f'width="{diag.LAYERS_W}"' in layers and f'height="{diag.LAYERS_H}"' in layers
    )
    for svg in (pipeline, layers):
        assert theme.bg in svg  # background painted in the theme's colour
        assert "Helvetica Neue" in svg and "Menlo" in svg


def test_code_panel_names_only_real_api() -> None:
    svg = diag.build(diag.DARK).as_svg()
    import luxar  # noqa: PLC0415 - the point is to check the shipped package

    for name in ("Dimensions", "Dimension", "LuxarZarrCompiler"):
        assert name in svg
        assert hasattr(luxar, name), name
    from luxar.core import Scene  # noqa: PLC0415

    for method in ("create_scene",):
        assert hasattr(luxar.LuxarZarrCompiler, method), method
    for method in ("add_gsplats", "add_points", "add_lines", "add_mesh"):
        assert method in svg
        assert hasattr(Scene, method), method


def test_layer_labels_name_real_modules() -> None:
    svg = diag.build_layers(diag.DARK).as_svg()
    for module in (
        "core/",
        "gsplats/",
        "mesh/",
        "io/",
        "encoding/",
        "cli/",
        "control/",
        "demos/",
    ):
        assert module in svg
        assert (PY_PKG / module.rstrip("/")).is_dir(), module
    for module in (
        "core",
        "ui",
        "themes",
        "controls",
        "input",
        "rendering",
        "scene",
        "workers",
        "wasm",
        "data",
        "cache",
        "config",
        "audio",
    ):
        assert (VIEWER_SRC / module).is_dir(), module
    assert (REPO_ROOT / "packages/luxar-launcher").is_dir()


def test_non_ascii_runs_of_spaces_are_preserved() -> None:
    svg = diag.build(diag.DARK).as_svg()
    # The tree column aligns descriptions with runs of spaces, which SVG would
    # collapse; the text helper converts them to non-breaking spaces.
    assert re.search(r"zarr\.json  ", svg)
