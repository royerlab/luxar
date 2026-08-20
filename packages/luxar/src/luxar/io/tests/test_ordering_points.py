"""Tests for Points chunk bounds computation with discrete dimensions.

This module specifically tests compute_chunk_bounds_points to ensure that:
1. Spatial dimensions have radius expansion applied
2. Discrete dimensions do NOT have radius expansion (only a tiny
   float-boundary epsilon, ``_BARRIER_BOUND_EPS`` — the reader's query
   tolerance owns the reach)

This is critical for correct discrete dimension filtering (e.g., time steps,
orbital indices, channels) - a point at orbital=0 should NOT appear when
querying orbital=3.
"""

import numpy as np
import pytest

from luxar.core.dimensions import Dimension
from luxar.io.ordering import (
    _BARRIER_BOUND_EPS,
    compute_chunk_bounds_points,
    sort_points_compound,
)
from luxar.typing_utils.constants import DEFAULT_POINT_RADIUS


class TestChunkBoundsPoints:
    """Test chunk bounding box computation for Points."""

    def test_chunk_bounds_basic_no_radii(self) -> None:
        """Test basic chunk bounds computation without radii."""
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=np.float32,
        )

        bounds = compute_chunk_bounds_points(positions, radii=None, chunk_size=2)

        # Should have 2 chunks
        assert bounds.shape == (2, 2, 2)

        # Check bounds are valid (min < max)
        assert np.all(bounds[:, :, 0] <= bounds[:, :, 1])

    def test_no_radii_expands_by_the_default_render_radius(self) -> None:
        """A chunk with no radii is expanded by exactly DEFAULT_POINT_RADIUS.

        A points node that stores no radii array is still DRAWN with the
        renderer's default radius, so the write-side bound must carry that
        extent. The old code used a "1% of the chunk's coordinate range, floor
        0.01" fudge instead — 50x too tight here — leaving the reader to rely on
        the viewer's tolerance over-reach to not miss a point.
        """
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )

        bounds = compute_chunk_bounds_points(positions, radii=None, chunk_size=2)

        assert bounds.shape == (1, 2, 2)
        for dim in range(2):
            assert bounds[0, dim, 0] == pytest.approx(0.0 - DEFAULT_POINT_RADIUS)
            assert bounds[0, dim, 1] == pytest.approx(1.0 + DEFAULT_POINT_RADIUS)

    def test_no_radii_expansion_does_not_scale_with_coordinate_range(self) -> None:
        """The no-radii expansion is absolute, not a fraction of the extent.

        The removed "1% of coordinate range" margin grew with the data, so the
        same authored scene bounded differently depending on its units. Two
        chunks whose ranges differ by three orders of magnitude must get the
        same expansion.
        """
        small = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        large = np.array([[0.0, 0.0], [1000.0, 1000.0]], dtype=np.float32)

        small_bounds = compute_chunk_bounds_points(small, radii=None, chunk_size=2)
        large_bounds = compute_chunk_bounds_points(large, radii=None, chunk_size=2)

        small_pad = small[:, 0].min() - small_bounds[0, 0, 0]
        large_pad = large[:, 0].min() - large_bounds[0, 0, 0]
        assert small_pad == pytest.approx(large_pad)
        assert large_pad == pytest.approx(DEFAULT_POINT_RADIUS)

    def test_no_radii_matches_explicit_default_scalar_radius(self) -> None:
        """radii=None and radii=DEFAULT_POINT_RADIUS give identical bounds.

        The writer and the renderer agree by construction: absent radii simply
        mean the default radius.
        """
        rng = np.random.default_rng(7)
        positions = (rng.random((37, 3)) * 12.0).astype(np.float32)

        implicit = compute_chunk_bounds_points(positions, radii=None, chunk_size=8)
        explicit = compute_chunk_bounds_points(
            positions, radii=DEFAULT_POINT_RADIUS, chunk_size=8
        )

        np.testing.assert_array_equal(implicit, explicit)

    def test_no_radii_pad_survives_the_float32_store_at_large_coordinates(
        self,
    ) -> None:
        """A 0.5 pad must still widen the bound where it is under half an ULP.

        ``chunk_bounds`` is float32. Past ``|x| ~ 2**23`` the ULP exceeds 1, so a
        round-to-nearest store of ``min - 0.5`` / ``max + 0.5`` lands back on the
        unpadded coordinate and the stored bound is TIGHTER than the disc the
        renderer draws. The removed scale-relative fudge was no safer here — 1%
        of this chunk's 10-unit range is 0.1 against a half-ULP of 1.0, so it
        rounded away too; the outward store is what closes the hole, for the
        authored-radius paths as much as for this one. A points node authored in
        nm over a ~10 mm field sits exactly here.
        """
        # ULP is 2.0 at 2e7, so both pads round away without the fix.
        lo, hi = 2.0e7, 2.0e7 + 10.0
        positions = np.array([[lo, lo, lo], [hi, hi, hi]], dtype=np.float32)

        bounds = compute_chunk_bounds_points(positions, radii=None, chunk_size=2)

        assert bounds.dtype == np.float32
        assert bounds.shape == (1, 3, 2)
        for dim in range(3):
            assert bounds[0, dim, 0] < positions[:, dim].min()
            assert bounds[0, dim, 1] > positions[:, dim].max()

    def test_authored_radius_pad_survives_the_float32_store_too(self) -> None:
        """The outward store is not specific to the no-radii default.

        An authored radius is padded in exactly the same place, so a small
        radius on large coordinates lost its pad the same way. Pins both the
        broadcast-scalar and the per-point array paths.
        """
        lo, hi = 2.0e7, 2.0e7 + 10.0
        positions = np.array([[lo, lo, lo], [hi, hi, hi]], dtype=np.float32)
        per_point = np.full(2, 0.05, dtype=np.float32)

        for radii in (0.05, per_point):
            bounds = compute_chunk_bounds_points(positions, radii=radii, chunk_size=2)
            for dim in range(3):
                assert bounds[0, dim, 0] < positions[:, dim].min()
                assert bounds[0, dim, 1] > positions[:, dim].max()

    def test_barrier_epsilon_survives_the_float32_store_at_large_coordinates(
        self,
    ) -> None:
        """The barrier arm is padded the same way, and loses it the same way.

        The two large-coordinate tests above both pass ``radii=`` and no
        ``slice_dims``, so they only exercise the SPATIAL arm. A categorical axis
        with large values (a millisecond timestamp, an acquisition index offset
        into an experiment) takes ``coords.min() - _BARRIER_BOUND_EPS`` instead:
        1e-3 against a half-ULP of 1.0 at ``|x| = 2e7``, so a round-to-nearest
        store collapses it back onto the exact category value and a query
        landing a float ULP off it misses the chunk. Mirrors the lines and
        gsplats barrier tests.
        """
        base = 2.0e7
        positions = np.array([[base, 5.0, 5.0], [base, 6.0, 6.0]], dtype=np.float32)

        bounds = compute_chunk_bounds_points(
            positions, radii=0.05, chunk_size=2, slice_dims=[0]
        )

        assert bounds.dtype == np.float32
        assert np.float64(bounds[0, 0, 0]) < np.float64(base)
        assert np.float64(bounds[0, 0, 1]) > np.float64(base)
        # ...and the spatial axes still carry their (much larger) radius pad.
        for dim in (1, 2):
            assert bounds[0, dim, 0] == pytest.approx(5.0 - 0.05)
            assert bounds[0, dim, 1] == pytest.approx(6.0 + 0.05)

    def test_chunk_bounds_basic_with_radii(self) -> None:
        """Test chunk bounds with uniform radii."""
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=np.float32,
        )
        radii = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        bounds = compute_chunk_bounds_points(positions, radii=radii, chunk_size=2)

        # Should have 2 chunks
        assert bounds.shape == (2, 2, 2)

        # First chunk: points [0,0] and [1,1], with radii 0.5
        # Min should be 0-0.5=-0.5, max should be 1+0.5=1.5
        assert bounds[0, 0, 0] == pytest.approx(-0.5, abs=0.01)  # X min
        assert bounds[0, 0, 1] == pytest.approx(1.5, abs=0.01)  # X max
        assert bounds[0, 1, 0] == pytest.approx(-0.5, abs=0.01)  # Y min
        assert bounds[0, 1, 1] == pytest.approx(1.5, abs=0.01)  # Y max

    def test_chunk_bounds_scalar_radii(self) -> None:
        """Test chunk bounds with scalar radii (no full array expansion)."""
        positions = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=np.float32,
        )

        bounds = compute_chunk_bounds_points(positions, radii=0.5, chunk_size=2)

        # Should have 2 chunks
        assert bounds.shape == (2, 2, 2)

        # First chunk: points [0,0] and [1,1], with radii 0.5
        assert bounds[0, 0, 0] == pytest.approx(-0.5, abs=0.01)  # X min
        assert bounds[0, 0, 1] == pytest.approx(1.5, abs=0.01)  # X max
        assert bounds[0, 1, 0] == pytest.approx(-0.5, abs=0.01)  # Y min
        assert bounds[0, 1, 1] == pytest.approx(1.5, abs=0.01)  # Y max

    def test_chunk_bounds_discrete_dimensions_no_expansion(self) -> None:
        """CRITICAL TEST: Discrete dimensions should NOT expand by radius.

        This test verifies the fix for the discrete dimension filtering bug:
        - A point at discrete=0 with radius=5 should have tight (epsilon-padded)
          chunk bounds of ~0 for the discrete dimension, NOT [-5, 5].
        - Without this fix, chunks with discrete=0 would match queries
          for discrete=3, returning wrong data.
        """
        # 4D data: [time, z, y, x] where time is discrete (dim 0)
        # Points at time=0, time=1, time=2
        positions = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],  # time=0
                [0.0, 1.0, 1.0, 1.0],  # time=0
                [1.0, 0.0, 0.0, 0.0],  # time=1
                [1.0, 1.0, 1.0, 1.0],  # time=1
                [2.0, 0.0, 0.0, 0.0],  # time=2
                [2.0, 1.0, 1.0, 1.0],  # time=2
            ],
            dtype=np.float32,
        )
        # Large radii to highlight the bug
        radii = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)

        # Dimension 0 (time) is discrete
        slice_dims = [0]

        bounds = compute_chunk_bounds_points(
            positions, radii=radii, chunk_size=2, slice_dims=slice_dims
        )

        # Should have 3 chunks (6 points / 2 per chunk)
        assert bounds.shape == (3, 4, 2)

        # Check chunk 0 (points at time=0): discrete bounds are exactly
        # ±_BARRIER_BOUND_EPS (NOT radius-expanded, NOT the legacy ±0.5 pad —
        # pins the write-side half of the barrier over-fetch fix).
        assert bounds[0, 0, 0] == pytest.approx(-_BARRIER_BOUND_EPS)
        assert bounds[0, 0, 1] == pytest.approx(_BARRIER_BOUND_EPS)

        # Check chunk 1 (points at time=1): bounds exactly 1 ± epsilon
        assert bounds[1, 0, 0] == pytest.approx(1.0 - _BARRIER_BOUND_EPS)
        assert bounds[1, 0, 1] == pytest.approx(1.0 + _BARRIER_BOUND_EPS)

        # Check chunk 2 (points at time=2): bounds exactly 2 ± epsilon
        assert bounds[2, 0, 0] == pytest.approx(2.0 - _BARRIER_BOUND_EPS)
        assert bounds[2, 0, 1] == pytest.approx(2.0 + _BARRIER_BOUND_EPS)

        # But spatial dimensions (1, 2, 3) SHOULD be expanded by radius
        # First chunk: spatial coords [0,0,0] and [1,1,1], radii=5
        # Min should be 0-5=-5, max should be 1+5=6
        assert bounds[0, 1, 0] == pytest.approx(-5.0, abs=0.01)  # Z min
        assert bounds[0, 1, 1] == pytest.approx(6.0, abs=0.01)  # Z max

    def test_chunk_bounds_multiple_discrete_dims(self) -> None:
        """Test with multiple discrete dimensions (e.g., time and channel)."""
        # 5D: [time, channel, z, y, x]
        # time (dim 0) and channel (dim 1) are discrete
        positions = np.array(
            [
                [0.0, 0.0, 5.0, 5.0, 5.0],  # time=0, channel=0
                [0.0, 1.0, 5.0, 5.0, 5.0],  # time=0, channel=1
                [1.0, 0.0, 5.0, 5.0, 5.0],  # time=1, channel=0
                [1.0, 1.0, 5.0, 5.0, 5.0],  # time=1, channel=1
            ],
            dtype=np.float32,
        )
        radii = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32)

        # Dimensions 0 (time) and 1 (channel) are discrete
        slice_dims = [0, 1]

        bounds = compute_chunk_bounds_points(
            positions, radii=radii, chunk_size=4, slice_dims=slice_dims
        )

        # Single chunk with all 4 points
        assert bounds.shape == (1, 5, 2)

        # Time bounds: exactly [0, 1] padded by the epsilon only
        assert bounds[0, 0, 0] == pytest.approx(-_BARRIER_BOUND_EPS)  # time min
        assert bounds[0, 0, 1] == pytest.approx(1.0 + _BARRIER_BOUND_EPS)  # time max

        # Channel bounds: exactly [0, 1] padded by the epsilon only
        assert bounds[0, 1, 0] == pytest.approx(-_BARRIER_BOUND_EPS)  # channel min
        assert bounds[0, 1, 1] == pytest.approx(1.0 + _BARRIER_BOUND_EPS)  # channel max

        # Spatial dimensions (z, y, x) should be expanded by radius
        # All at coord 5, radius 3 -> [2, 8]
        for dim in [2, 3, 4]:
            assert bounds[0, dim, 0] == pytest.approx(2.0, abs=0.01)
            assert bounds[0, dim, 1] == pytest.approx(8.0, abs=0.01)

    def test_chunk_bounds_no_discrete_dims(self) -> None:
        """Test that all dimensions expand by radius when no discrete dims."""
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        radii = np.array([2.0, 2.0], dtype=np.float32)

        # No discrete dimensions - all are spatial
        bounds = compute_chunk_bounds_points(
            positions, radii=radii, chunk_size=2, slice_dims=[]
        )

        # Single chunk
        assert bounds.shape == (1, 3, 2)

        # All dims should be expanded: coords [0,0,0] to [1,1,1] with r=2
        # -> bounds [-2, 3] in all dims
        for dim in range(3):
            assert bounds[0, dim, 0] == pytest.approx(-2.0, abs=0.01)
            assert bounds[0, dim, 1] == pytest.approx(3.0, abs=0.01)

    def test_chunk_bounds_slice_dims_none(self) -> None:
        """Test that slice_dims=None behaves like empty list (all spatial)."""
        positions = np.array([[5.0, 5.0]], dtype=np.float32)
        radii = np.array([2.0], dtype=np.float32)

        bounds = compute_chunk_bounds_points(
            positions, radii=radii, chunk_size=1, slice_dims=None
        )

        # Should expand in all dims
        for dim in range(2):
            assert bounds[0, dim, 0] == pytest.approx(3.0, abs=0.01)
            assert bounds[0, dim, 1] == pytest.approx(7.0, abs=0.01)

    def test_chunk_bounds_no_radii_with_discrete_dims(self) -> None:
        """Test discrete dimension handling when no radii are provided."""
        # Points at time=0 and time=5
        positions = np.array(
            [
                [0.0, 10.0, 10.0],  # time=0
                [5.0, 10.0, 10.0],  # time=5
            ],
            dtype=np.float32,
        )

        # Dimension 0 (time) is discrete
        slice_dims = [0]

        bounds = compute_chunk_bounds_points(
            positions, radii=None, chunk_size=2, slice_dims=slice_dims
        )

        # Single chunk
        assert bounds.shape == (1, 3, 2)

        # Time (discrete) should have tight bounds: exactly [0, 5] padded by
        # the epsilon only. Reached with radii=None, which since the write-side
        # fix runs through the SAME code site as an explicit scalar radius (the
        # two branches were collapsed into one) — so this pins that the default
        # render radius does not leak onto a categorical axis.
        assert bounds[0, 0, 0] == pytest.approx(-_BARRIER_BOUND_EPS)  # time min
        assert bounds[0, 0, 1] == pytest.approx(5.0 + _BARRIER_BOUND_EPS)  # time max

        # ...while the spatial axes DO carry the renderer's default radius.
        for dim in (1, 2):
            assert bounds[0, dim, 0] == pytest.approx(10.0 - DEFAULT_POINT_RADIUS)
            assert bounds[0, dim, 1] == pytest.approx(10.0 + DEFAULT_POINT_RADIUS)

    def test_chunk_bounds_varying_radii(self) -> None:
        """Test with varying radii per point."""
        positions = np.array(
            [
                [0.0, 5.0, 5.0],  # time=0
                [0.0, 5.0, 5.0],  # time=0
            ],
            dtype=np.float32,
        )
        # Different radii
        radii = np.array([1.0, 3.0], dtype=np.float32)

        slice_dims = [0]  # time is discrete

        bounds = compute_chunk_bounds_points(
            positions, radii=radii, chunk_size=2, slice_dims=slice_dims
        )

        # Time bounds: tight (discrete) — exactly ±epsilon
        assert bounds[0, 0, 0] == pytest.approx(-_BARRIER_BOUND_EPS)
        assert bounds[0, 0, 1] == pytest.approx(_BARRIER_BOUND_EPS)

        # Spatial dims: should use max radius extent
        # coords [5,5] with radii [1,3] -> min=5-3=2, max=5+3=8
        assert bounds[0, 1, 0] == pytest.approx(2.0, abs=0.01)
        assert bounds[0, 1, 1] == pytest.approx(8.0, abs=0.01)


class TestChunkBoundsDiscreteFiltering:
    """Tests verifying discrete filtering won't cause cross-contamination."""

    def test_discrete_chunks_dont_overlap(self) -> None:
        """Verify chunks at different discrete values don't overlap in that dimension.

        This is the core correctness property: if we have points at time=0 and
        points at time=3, their chunk bounds in the time dimension should NOT
        overlap, even with large radii.
        """
        # Points at time=0, 1, 2, 3, 4
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        # Very large radii (would cause huge overlap if applied to discrete dim)
        radii = np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

        slice_dims = [0]  # time is discrete

        # One chunk per point
        bounds = compute_chunk_bounds_points(
            positions, radii=radii, chunk_size=1, slice_dims=slice_dims
        )

        # 5 chunks, one per point
        assert bounds.shape == (5, 3, 2)

        # Each chunk's time bounds should NOT overlap with neighbors
        for i in range(4):
            chunk_max = bounds[i, 0, 1]
            next_chunk_min = bounds[i + 1, 0, 0]
            # With epsilon padding, chunks at t=0 and t=1 have bounds
            # [-eps, eps] and [1-eps, 1+eps] with a clean ~1-step gap
            # (1 - 2·eps ≈ 0.998). The legacy ±0.5 padding gave gap = 0, so
            # requiring >= 0.9 pins the epsilon fix (fails on pre-fix bounds).
            gap = next_chunk_min - chunk_max
            assert gap >= 0.9, (
                f"Chunks {i} and {i + 1} overlap too much in discrete dim: "
                f"chunk {i} max={chunk_max}, chunk {i + 1} min={next_chunk_min}, gap={gap}"
            )

    def test_query_at_discrete_value_finds_correct_chunks(self) -> None:
        """Simulate querying and verify only correct chunks match."""
        # Points at orbital values 0, 1, 2, 3, 4
        positions = np.array(
            [
                [0.0, 5.0, 5.0],  # orbital=0
                [1.0, 5.0, 5.0],  # orbital=1
                [2.0, 5.0, 5.0],  # orbital=2
                [3.0, 5.0, 5.0],  # orbital=3
                [4.0, 5.0, 5.0],  # orbital=4
            ],
            dtype=np.float32,
        )
        radii = np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        slice_dims = [0]  # orbital is discrete

        bounds = compute_chunk_bounds_points(
            positions, radii=radii, chunk_size=1, slice_dims=slice_dims
        )

        # Simulate query for orbital=2 with tolerance 0.5 — the most generous
        # reader tolerance still in the wild (the viewer now uses 0.25 × step).
        # With epsilon-padded bounds even this must select ONLY the target
        # category; with the legacy ±0.5 padding it also touched chunks 1 and 3
        # (pad + tolerance summed to a full step), so the exact-match assertion
        # below pins the write-side over-fetch fix.
        query_value = 2.0
        query_tolerance = 0.5
        query_min = query_value - query_tolerance
        query_max = query_value + query_tolerance

        # Find which chunks intersect the query in the orbital dimension
        matching_chunks = []
        for chunk_idx in range(bounds.shape[0]):
            chunk_min = bounds[chunk_idx, 0, 0]
            chunk_max = bounds[chunk_idx, 0, 1]
            # AABB intersection test
            if not (chunk_max < query_min or chunk_min > query_max):
                matching_chunks.append(chunk_idx)

        # Must match EXACTLY chunk 2 (orbital=2) — no neighbour bleed.
        assert matching_chunks == [2], (
            f"Query for orbital=2 must select only its own chunk, got {matching_chunks}"
        )


class TestSortPointsCompound:
    """Unit coverage for sort_points_compound — the Points sort mirroring
    TestSortSplatsSpatial (GSplats) and TestSortSegmentsCompound (Lines)."""

    def test_sort_indices_are_a_permutation(self) -> None:
        rng = np.random.default_rng(0)
        pos = rng.random((50, 3)).astype(np.float32)
        dims = [Dimension(n, display=True) for n in ("x", "y", "z")]
        idx, meta = sort_points_compound(pos, dims, method="morton")
        # Pure permutation — no point dropped or duplicated.
        np.testing.assert_array_equal(np.sort(idx), np.arange(50))
        assert meta["ordering"] in ("morton", "hilbert")

    def test_both_curves_permute(self) -> None:
        rng = np.random.default_rng(1)
        pos = rng.random((32, 3)).astype(np.float32)
        dims = [Dimension(n, display=True) for n in ("x", "y", "z")]
        for method in ("morton", "hilbert"):
            idx, _ = sort_points_compound(pos, dims, method=method)
            np.testing.assert_array_equal(np.sort(idx), np.arange(32))

    def test_discrete_dim_groups_before_spatial(self) -> None:
        # A non-displayed (=> discrete) "time" axis is a barrier: after ordering,
        # points must be grouped by their time value (non-decreasing) so a chunk
        # never straddles two timepoints — the compound-ordering contract.
        rng = np.random.default_rng(2)
        spatial = rng.random((9, 2)).astype(np.float32)
        times = np.array([2, 0, 1, 2, 0, 1, 2, 0, 1], dtype=np.float32)
        pos = np.column_stack([times, spatial]).astype(np.float32)
        dims = [
            Dimension("Time", display=False, range=(0, 2)),
            Dimension("y", display=True),
            Dimension("x", display=True),
        ]
        idx, _ = sort_points_compound(pos, dims)
        ordered_times = pos[idx, 0]
        assert np.all(np.diff(ordered_times) >= 0), ordered_times
