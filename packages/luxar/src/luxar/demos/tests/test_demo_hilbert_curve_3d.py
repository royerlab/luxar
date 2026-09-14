"""The 3D Hilbert demo opens on order 4, the first order with visible structure."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import zarr

from luxar.utils.lod_breakpoints import parse_stream_chunk, stream_cuts

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
    n_vertices = 524_286
    first_chunk = parse_stream_chunk(demo.hilbert_ladder(n_vertices)["counts"])
    cuts = stream_cuts(n_vertices, first_chunk)
    increments = [
        count - previous for previous, count in zip([0, *cuts[:-1]], cuts, strict=True)
    ]

    assert first_chunk == 65_536
    assert max(increments) <= 900_000


def test_scene_opens_on_order_four(tmp_path) -> None:
    demo = _load()
    out = tmp_path / "hilbert.luxar.zarr"
    demo.build_scene(out, max_order=4)
    cfg = dict(zarr.open_group(str(out), mode="r").attrs)["viewer_config"]
    assert cfg["dimensions"]["current_step"] == [3.0, 0.0, 0.0, 0.0]
    assert cfg["dimensions"]["selected_dimension"] == 0
