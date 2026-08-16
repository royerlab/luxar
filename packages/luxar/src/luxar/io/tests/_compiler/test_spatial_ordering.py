"""Tests for luxar.io._compiler.spatial_ordering: direct unit tests, plus one
end-to-end compiler round trip through the scene API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.io._compiler.context import OrderingCtx
from luxar.io._compiler.spatial_ordering.lines import build_lines_ordering
from luxar.io._compiler.spatial_ordering.points import (
    build_points_ordering,
    write_points_ordering_to_zarr,
)
from luxar.io._ordering.bounds import _BARRIER_BOUND_EPS
from luxar.io.reader import DEFAULT_COMP
from luxar.typing_utils.constants import DEFAULT_POINT_RADIUS


def _disabled() -> OrderingCtx:
    return OrderingCtx(enable_spatial_index=False, ordering_method="morton")


def _enabled() -> OrderingCtx:
    return OrderingCtx(enable_spatial_index=True, ordering_method="morton")


def test_build_points_ordering_none_when_disabled() -> None:
    pos = np.random.rand(10, 3).astype(np.float32)
    assert build_points_ordering(pos, 10, 3, None, _disabled(), zarr.group()) is None


def test_build_points_ordering_none_without_scene_dimensions() -> None:
    pos = np.random.rand(10, 3).astype(np.float32)
    # enabled, but the store carries no scene_dimensions attr → skipped
    assert build_points_ordering(pos, 10, 3, None, _enabled(), zarr.group()) is None


def test_build_lines_ordering_none_when_disabled() -> None:
    verts = np.random.rand(6, 3).astype(np.float32)
    segs = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.uint32)
    out = build_lines_ordering(verts, segs, 1.0, 6, 3, 3, _disabled(), zarr.group())
    assert out is None


def test_write_points_ordering_to_zarr_roundtrip() -> None:
    g = zarr.group()
    chunk_bounds = np.array(
        [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], dtype=np.float32
    )  # (1, 3, 2)
    ordering_data = {
        "ordering": "morton",
        "slice_dims": [],
        "ordering_dims": [0, 1, 2],
        "ordering_min": [0.0, 0.0, 0.0],
        "ordering_max": [1.0, 1.0, 1.0],
        "ordering_bits_per_dim": 10,
        "chunk_size": 1024,
        "chunk_bounds": chunk_bounds,
    }
    write_points_ordering_to_zarr(g, ordering_data, DEFAULT_COMP)
    assert g.attrs["ordering"] == "morton"
    assert g.attrs["chunk_size"] == 1024
    assert "chunk_bounds" in g
    np.testing.assert_allclose(g["chunk_bounds"][:], chunk_bounds)


def test_stored_chunk_bounds_carry_the_default_radius_without_radii(
    tmp_path: Path,
) -> None:
    """End-to-end: a radii-less node's ON-DISK chunk_bounds carry the default pad.

    Every other test of this behaviour calls ``compute_chunk_bounds_points``
    directly; this one drives the whole authoring path and then reads the stored
    array back. The writer path that sees ``radii=None`` is
    ``write_points_multi_lod``, reached whenever ``add_points`` builds an
    additive ladder before the authoring default is materialized — the
    ``additive_lod=`` used here, the default stream ladder ``substitutive_lod=``
    composes for its finest child, or ``partition=`` parts carrying a ladder of
    their own. A plain ``add_points`` (and a bare ``partition=``, whose parts
    recurse back through it) materializes ``DEFAULT_POINT_RADIUS`` first.

    The data pins one point at the spatial origin and one at ``(4, 4, 4)``, so
    whichever additive level each lands in, the tightest stored bound across all
    levels is exactly ``[0 - r, 4 + r]`` per spatial axis. The old code padded a
    no-radii chunk by ``max(1% of the chunk's range, 0.01) <= 0.04`` here, so the
    ``-0.5`` / ``4.5`` asserted below cannot be produced by it.
    """
    dims = Dimensions(
        [
            Dimension("time", unit="s", display=False, discrete=True),
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ]
    )

    rng = np.random.default_rng(0)
    positions = np.zeros((64, 4), dtype=np.float32)
    positions[:, 0] = rng.integers(0, 4, 64)
    positions[:, 1:] = (rng.random((64, 3)) * 4.0).astype(np.float32)
    # Pin the per-axis extremes so the expected bounds are exact.
    positions[0] = [0.0, 0.0, 0.0, 0.0]
    positions[1] = [3.0, 4.0, 4.0, 4.0]

    out = tmp_path / "no_radii.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=True) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_points("pts", positions, additive_lod={"n_lods": 3})

    node = zarr.open_group(out, mode="r")["pts"]
    levels = [node[k] for k in sorted(node.group_keys())]
    assert levels, "expected additive sub-LOD levels under the points node"

    for level in levels:
        # No max_radius is stamped without radii — this is what makes the
        # reader fall back to defaultMaxRadius=0.1, and why the bound has to be
        # honest on its own.
        assert level.attrs["has_radii"] is False
        assert "max_radius" not in level.attrs

    bounds = np.concatenate([level["chunk_bounds"][:] for level in levels], axis=0)
    assert bounds.shape[1:] == (4, 2)

    # Spatial axes (1, 2, 3): padded by exactly DEFAULT_POINT_RADIUS = 0.5.
    for dim in (1, 2, 3):
        assert bounds[:, dim, 0].min() == pytest.approx(0.0 - DEFAULT_POINT_RADIUS)
        assert bounds[:, dim, 1].max() == pytest.approx(4.0 + DEFAULT_POINT_RADIUS)

    # Discrete/barrier axis 0: only the float-boundary epsilon, never the radius.
    assert bounds[:, 0, 0].min() == pytest.approx(0.0 - _BARRIER_BOUND_EPS)
    assert bounds[:, 0, 1].max() == pytest.approx(3.0 + _BARRIER_BOUND_EPS)
