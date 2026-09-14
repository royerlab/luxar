"""The 3D Hilbert demo opens on order 4, the first order with visible structure."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import zarr

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_hilbert_curve_3d.py"


def _load():
    saved = list(sys.argv)
    sys.argv = ["demo", "--no-serve"]
    try:
        spec = importlib.util.spec_from_file_location(
            "_hilbert_demo_under_test", _DEMO_PATH
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def test_opening_slot_is_order_four_when_available() -> None:
    demo = _load()
    assert demo.opening_slot([1, 2, 3, 4, 5]) == 3
    assert demo.opening_slot([1, 2, 3, 4]) == 3
    # Clamped when the scene is built shallower than order 4.
    assert demo.opening_slot([1, 2]) == 1


def test_hidden_order_ladder_opens_on_an_eighth_and_bounds_commits() -> None:
    demo = _load()
    for n_vertices in (524_286, 4_194_302):
        counts = demo.hilbert_ladder(n_vertices)["counts"]
        increments = [
            2 * (count - previous)
            for previous, count in zip([0, *counts[:-1]], counts, strict=True)
        ]

        assert counts[0] * 2 == max(65_536, (n_vertices + 7) // 8)
        assert counts[-1] * 2 == n_vertices
        assert max(increments) <= 900_000


def test_scene_opens_on_order_four(tmp_path) -> None:
    demo = _load()
    out = tmp_path / "hilbert.luxar.zarr"
    demo.build_scene(out, max_order=4)
    cfg = dict(zarr.open_group(str(out), mode="r").attrs)["viewer_config"]
    assert cfg["dimensions"]["current_step"] == [3.0, 0.0, 0.0, 0.0]
    assert cfg["dimensions"]["selected_dimension"] == 0
