"""Direct unit tests for luxar.io._compiler.spatial_ordering."""

from __future__ import annotations

import numpy as np
import zarr

from luxar.io._compiler.context import OrderingCtx
from luxar.io._compiler.spatial_ordering.lines import build_lines_ordering
from luxar.io._compiler.spatial_ordering.points import (
    build_points_ordering,
    write_points_ordering_to_zarr,
)
from luxar.io.reader import DEFAULT_COMP


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
