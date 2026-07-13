"""Smoke tests for pure helpers in demo_ppi_flow_field.

These tests cover deterministic numerical helpers that don't touch the network,
the cache, or matplotlib. Network-fetching code paths (``ensure_data``) are
intentionally not exercised here. The heavy graph deps (networkx/umap/scipy) are
imported lazily inside the demo's functions, so importing the module only needs
pandas — hence the ``importorskip`` guard before the demo import.
"""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

# The ``luxar.demos`` package is now importable directly (the sys.modules alias
# that used to shadow it was removed), so we import the demo's helpers normally.
from luxar.demos.demo_ppi_flow_field import (  # noqa: E402
    _grid_points_for_flat_indices,
    array_hash,
    network_hash,
)
from luxar.utils.fields import FlowField, cubic_bounds, trilinear_vector  # noqa: E402

compute_cubic_bounds = cubic_bounds
_trilinear_vector_batch = trilinear_vector


class TestGridPointsForFlatIndices:
    """`_grid_points_for_flat_indices` decodes C-order grid indices to xyz."""

    def test_corner_indices_round_trip(self) -> None:
        n = 4
        spacing = 0.5
        grid_min = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # All eight cube corners as flat C-order indices on a 4×4×4 grid.
        flats = np.array(
            [
                0,  # ( 0,  0,  0)
                n - 1,  # ( 0,  0,  3)
                (n - 1) * n,  # ( 0,  3,  0)
                (n - 1) * n + (n - 1),  # ( 0,  3,  3)
                (n - 1) * n * n,  # ( 3,  0,  0)
                n * n * n - 1,  # ( 3,  3,  3)
            ],
            dtype=np.int64,
        )

        coords = _grid_points_for_flat_indices(flats, grid_min, spacing, n)
        expected = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0 + 3 * spacing],
                [1.0, 2.0 + 3 * spacing, 3.0],
                [1.0, 2.0 + 3 * spacing, 3.0 + 3 * spacing],
                [1.0 + 3 * spacing, 2.0, 3.0],
                [
                    1.0 + 3 * spacing,
                    2.0 + 3 * spacing,
                    3.0 + 3 * spacing,
                ],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(coords, expected)

    def test_returns_float32_for_dtype_safety_downstream(self) -> None:
        coords = _grid_points_for_flat_indices(
            np.array([0, 1], dtype=np.int64),
            np.zeros(3, dtype=np.float32),
            1.0,
            4,
        )
        assert coords.dtype == np.float32


class TestTrilinearVectorBatch:
    """`_trilinear_vector_batch` interpolates a vector field at world points."""

    def _constant_field(self) -> FlowField:
        n = 4
        spacing = 1.0
        vectors = np.full((n, n, n, 3), 7.0, dtype=np.float32)
        return FlowField(
            vectors=vectors,
            grid_min=np.zeros(3, dtype=np.float32),
            grid_max=np.full(3, n - 1, dtype=np.float32),
            spacing=spacing,
            cache_key="test",
        )

    def test_constant_field_returns_constant(self) -> None:
        flow = self._constant_field()
        points = np.array([[1.0, 1.0, 1.0], [1.5, 2.5, 0.5]], dtype=np.float32)
        result = _trilinear_vector_batch(flow, points)
        np.testing.assert_allclose(result, np.full((2, 3), 7.0), atol=1e-6)

    def test_out_of_bounds_returns_nan(self) -> None:
        flow = self._constant_field()
        points = np.array([[-1.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float32)
        result = _trilinear_vector_batch(flow, points)
        assert np.isnan(result).all()

    def test_at_corner_returns_corner_value(self) -> None:
        # Non-constant field so we can verify which voxel is sampled.
        n = 4
        vectors = np.zeros((n, n, n, 3), dtype=np.float32)
        vectors[2, 1, 2] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        flow = FlowField(
            vectors=vectors,
            grid_min=np.zeros(3, dtype=np.float32),
            grid_max=np.full(3, n - 1, dtype=np.float32),
            spacing=1.0,
            cache_key="test",
        )
        # Exactly at grid index (2, 1, 2) → world (2.0, 1.0, 2.0). Index must be
        # strictly inside [0, n-1) so the trilinear corner can be sampled
        # (the helper marks anything on the upper boundary as out-of-bounds).
        result = _trilinear_vector_batch(
            flow, np.array([[2.0, 1.0, 2.0]], dtype=np.float32)
        )
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0], atol=1e-6)


class TestComputeCubicBounds:
    def test_returns_padded_cube_around_coords(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 2.0]], dtype=np.float32)
        lo, hi = compute_cubic_bounds(coords, pad_fraction=0.1)
        side_lo = hi - lo
        # Cubic: every axis the same span.
        np.testing.assert_allclose(side_lo, np.full(3, side_lo[0]), atol=1e-6)
        # Padded > raw range (10 along x).
        assert side_lo[0] > 10.0


class TestHashHelpersAreDeterministic:
    """Hashes of identical inputs must match; different inputs must differ."""

    def test_array_hash_deterministic(self) -> None:
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert array_hash(a) == array_hash(a)
        assert array_hash(a) != array_hash(a + 1.0)

    def test_network_hash_deterministic_for_identical_inputs(self) -> None:
        nodes = ["A", "B", "C"]
        edges = pd.DataFrame({"sym_a": ["A", "B"], "sym_b": ["B", "C"]})
        h1 = network_hash(nodes, edges)
        h2 = network_hash(list(nodes), edges.copy())
        assert h1 == h2

    def test_network_hash_changes_with_edges(self) -> None:
        nodes = ["A", "B"]
        e1 = pd.DataFrame({"sym_a": ["A"], "sym_b": ["B"]})
        e2 = pd.DataFrame({"sym_a": ["B"], "sym_b": ["A"]})
        assert network_hash(nodes, e1) != network_hash(nodes, e2)
