"""Pins the Cells3D multichannel demo's hand-tuned per-channel display windows.

Tuned in the hosted viewer's Layers panel on 2026-09-10; a window is stored as
``intensity = 1/(hi-lo)``, ``offset = -lo/(hi-lo)`` on these colormapped nodes.
Read off the module rather than a built scene: building one needs the Cells3D
fits.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_3d_cells3d_multichannel.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "_cells3d_demo_under_test", _DEMO_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_channel_windows_are_the_tuned_ones() -> None:
    demo = _load()
    windows = {ch["name"]: ch["window"] for ch in demo.CHANNELS}
    assert windows == {"Membranes": (0.006, 0.053), "Nuclei": (0.009, 0.114)}


def test_window_attrs_round_trip() -> None:
    demo = _load()
    attrs = demo.window_attrs((0.006, 0.053))
    assert attrs["intensity"] == pytest.approx(1.0 / 0.047)
    assert attrs["offset"] == pytest.approx(-0.006 / 0.047)
    # The viewer recovers [lo, hi] as [-offset/intensity, (1-offset)/intensity].
    lo = -attrs["offset"] / attrs["intensity"]
    hi = (1.0 - attrs["offset"]) / attrs["intensity"]
    assert (lo, hi) == pytest.approx((0.006, 0.053))


def test_add_gsplats_receives_the_window() -> None:
    tree = ast.parse(_DEMO_PATH.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_gsplats"
        ):
            splats = [kw for kw in node.keywords if kw.arg is None]
            assert splats and "window_attrs" in ast.unparse(splats[0].value)
            return
    raise AssertionError("no add_gsplats call found in the demo")
