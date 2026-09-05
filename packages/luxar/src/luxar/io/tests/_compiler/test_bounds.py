"""Direct unit tests for luxar.io._compiler.bounds."""

from __future__ import annotations

import numpy as np

from luxar.io._compiler.bounds import compute_position_bounds, update_scene_bounds


def test_compute_position_bounds_basic() -> None:
    pos = np.array([[0.0, 1.0, 2.0], [4.0, -1.0, 2.0]], dtype=np.float32)
    b = compute_position_bounds(pos)
    assert b["min"] == [0.0, -1.0, 2.0]
    assert b["max"] == [4.0, 1.0, 2.0]


def test_compute_position_bounds_empty() -> None:
    pos = np.zeros((0, 3), dtype=np.float32)
    b = compute_position_bounds(pos)
    assert b == {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}


def test_update_scene_bounds_first_node_initializes() -> None:
    out = update_scene_bounds(None, {"min": [0.0, 0.0], "max": [1.0, 1.0]})
    assert out == {"min": [0.0, 0.0], "max": [1.0, 1.0]}


def test_update_scene_bounds_takes_union() -> None:
    cur = {"min": [0.0, 0.0], "max": [1.0, 1.0]}
    out = update_scene_bounds(cur, {"min": [-1.0, 0.5], "max": [0.5, 3.0]})
    assert out["min"] == [-1.0, 0.0]
    assert out["max"] == [1.0, 3.0]


def test_update_scene_bounds_extends_dimensionality() -> None:
    cur = {"min": [0.0, 0.0], "max": [1.0, 1.0]}
    out = update_scene_bounds(cur, {"min": [0.0, 0.0, 5.0], "max": [1.0, 1.0, 9.0]})
    assert out["min"] == [0.0, 0.0, 5.0]
    assert out["max"] == [1.0, 1.0, 9.0]


def test_every_geometry_type_contributes_to_world_bounds() -> None:
    """Each contract geometry type's leaf bounds must reach the scene bounds.

    ``expand_bounds_with_transforms`` selects leaves by testing the node's
    ``type`` against the contract vocabulary. It previously tested a literal
    ``("points", "lines", "gsplats")`` tuple, so a geometry type added to
    ``geometry_types`` but missed there would be skipped **silently** — its
    geometry would never widen the scene's world bounds, and the viewer derives
    near/far clipping from those bounds. Parametrised over the vocabulary so a
    new type is covered without editing this test.
    """
    import zarr

    from luxar.io._compiler.bounds import expand_bounds_with_transforms
    from luxar.typing_utils._format_contract import GEOMETRY_TYPES

    store = zarr.group()
    store.attrs["scene_dimensions"] = {
        "dimensions": [
            {
                "name": "x",
                "unit": "um",
                "range": [0.0, 100.0],
                "step": 1.0,
                "display": True,
            },
            {
                "name": "y",
                "unit": "um",
                "range": [0.0, 100.0],
                "step": 1.0,
                "display": True,
            },
            {
                "name": "z",
                "unit": "um",
                "range": [0.0, 100.0],
                "step": 1.0,
                "display": True,
            },
        ]
    }

    # One leaf per geometry type, each occupying a disjoint unit box at a
    # distinct offset so a skipped type is visible in the union.
    for i, gtype in enumerate(GEOMETRY_TYPES):
        child = store.create_group(f"leaf_{gtype}")
        child.attrs["type"] = gtype
        lo = float(i * 10)
        child.attrs["position_bounds"] = {
            "min": [lo, lo, lo],
            "max": [lo + 1.0, lo + 1.0, lo + 1.0],
        }

    seed = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    out = expand_bounds_with_transforms(store, seed)

    assert out is not None
    # The last type's box is the farthest out; if any type were skipped the max
    # would fall short of it.
    last_lo = float((len(GEOMETRY_TYPES) - 1) * 10)
    for axis in range(3):
        assert out["max"][axis] >= last_lo + 1.0, (
            f"axis {axis}: world bounds {out['max']} does not cover the "
            f"{GEOMETRY_TYPES[-1]!r} leaf at {last_lo}"
        )
