"""Tests for tiled Gaussian splat fitting."""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.tiling import compute_tile_specs, cosine_window

# Fitting tests require torch
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Tiling geometry tests (pure NumPy, always run) ───────────────


class TestComputeTileSpecs:
    """Tests for compute_tile_specs()."""

    def test_coverage_3d(self) -> None:
        """Every voxel must be covered by at least one tile."""
        shape = (100, 80, 120)
        specs = compute_tile_specs(shape, tile_size=40, overlap=10)

        covered = np.zeros(shape, dtype=bool)
        for spec in specs:
            covered[spec.slices] = True

        assert covered.all(), "Some voxels are not covered by any tile"

    def test_coverage_2d(self) -> None:
        """2D coverage check."""
        shape = (50, 70)
        specs = compute_tile_specs(shape, tile_size=30, overlap=8)

        covered = np.zeros(shape, dtype=bool)
        for spec in specs:
            covered[spec.slices] = True

        assert covered.all()

    def test_single_tile_when_volume_smaller(self) -> None:
        """Volume smaller than tile_size should produce exactly 1 tile."""
        shape = (32, 32, 32)
        specs = compute_tile_specs(shape, tile_size=64, overlap=16)

        assert len(specs) == 1
        spec = specs[0]
        assert spec.index == 0
        assert spec.shape == shape
        assert spec.border_low == (True, True, True)
        assert spec.border_high == (True, True, True)
        assert spec.overlap_low == (0, 0, 0)
        assert spec.overlap_high == (0, 0, 0)
        assert spec.origin == (0.0, 0.0, 0.0)

    def test_deterministic(self) -> None:
        """Same inputs must always produce identical output (Slurm requirement)."""
        shape = (100, 100, 100)
        specs1 = compute_tile_specs(shape, tile_size=40, overlap=10)
        specs2 = compute_tile_specs(shape, tile_size=40, overlap=10)

        assert len(specs1) == len(specs2)
        for s1, s2 in zip(specs1, specs2):
            assert s1 == s2

    def test_sequential_indices(self) -> None:
        """Tile indices must be sequential 0..N-1."""
        specs = compute_tile_specs((80, 80, 80), tile_size=50, overlap=10)
        indices = [s.index for s in specs]
        assert indices == list(range(len(specs)))

    def test_border_flags_interior_tile(self) -> None:
        """Interior tiles should not have border flags set."""
        shape = (100, 100, 100)
        specs = compute_tile_specs(shape, tile_size=40, overlap=5)

        interior = [
            s
            for s in specs
            if all(not lo and not hi for lo, hi in zip(s.border_low, s.border_high))
        ]
        assert len(interior) > 0, "Expected at least one interior tile"

    def test_anisotropic_tile_size(self) -> None:
        """Tuple tile_size should work for anisotropic volumes."""
        shape = (50, 200, 200)
        specs = compute_tile_specs(shape, tile_size=(50, 100, 100), overlap=(0, 20, 20))

        covered = np.zeros(shape, dtype=bool)
        for spec in specs:
            covered[spec.slices] = True
        assert covered.all()

    def test_validation_errors(self) -> None:
        """Invalid parameters should raise ValueError."""
        with pytest.raises(ValueError, match="tile_size must be > 0"):
            compute_tile_specs((100,), tile_size=0, overlap=0)

        with pytest.raises(ValueError, match="overlap must be >= 0"):
            compute_tile_specs((100,), tile_size=10, overlap=-1)

        with pytest.raises(ValueError, match="overlap.*must be < tile_size"):
            compute_tile_specs((100,), tile_size=10, overlap=10)

        with pytest.raises(ValueError, match="overlap.*must be <= tile_size/2"):
            compute_tile_specs((100,), tile_size=10, overlap=6)

    def test_max_overlap_at_half_tile(self) -> None:
        """overlap = tile_size // 2 should be accepted (boundary case)."""
        # overlap=5 for tile_size=10 is the maximum allowed (5*2 == 10)
        specs = compute_tile_specs((100,), tile_size=10, overlap=5)
        assert len(specs) > 1

    def test_zero_overlap(self) -> None:
        """Zero overlap should produce non-overlapping tiles."""
        shape = (100,)
        specs = compute_tile_specs(shape, tile_size=30, overlap=0)

        for i in range(len(specs) - 1):
            assert specs[i].slices[0].stop <= specs[i + 1].slices[0].start

    def test_numpy_integer_inputs(self) -> None:
        """numpy integer types (e.g., np.int64) should be accepted."""
        specs = compute_tile_specs(
            (100, 100), tile_size=np.int64(40), overlap=np.int32(10)
        )
        assert len(specs) > 0
        # Shape should be plain Python int tuples
        assert all(isinstance(s, int) for s in specs[0].shape)

    def test_overlap_values_correct(self) -> None:
        """Per-tile overlap_low/overlap_high should match actual geometry."""
        specs = compute_tile_specs((100,), tile_size=40, overlap=10)

        for i, spec in enumerate(specs):
            if i == 0:
                assert spec.overlap_low[0] == 0
            else:
                prev = specs[i - 1]
                expected = prev.slices[0].stop - spec.slices[0].start
                assert spec.overlap_low[0] == expected

            if i == len(specs) - 1:
                assert spec.overlap_high[0] == 0
            else:
                nxt = specs[i + 1]
                expected = spec.slices[0].stop - nxt.slices[0].start
                assert spec.overlap_high[0] == expected


class TestCosineWindow:
    """Tests for cosine_window()."""

    def _partition_of_unity(
        self, volume_shape: tuple[int, ...], tile_size: int, overlap: int
    ) -> None:
        """Helper: verify overlapping windows sum to 1.0 everywhere."""
        specs = compute_tile_specs(volume_shape, tile_size, overlap)
        accumulated = np.zeros(volume_shape, dtype=np.float64)

        for spec in specs:
            w = cosine_window(spec)
            accumulated[spec.slices] += w.astype(np.float64)

        np.testing.assert_allclose(accumulated, 1.0, atol=1e-6)

    def test_partition_of_unity_1d(self) -> None:
        self._partition_of_unity((100,), tile_size=40, overlap=10)

    def test_partition_of_unity_2d(self) -> None:
        self._partition_of_unity((60, 80), tile_size=30, overlap=8)

    def test_partition_of_unity_3d(self) -> None:
        self._partition_of_unity((50, 50, 50), tile_size=30, overlap=8)

    def test_partition_of_unity_edge_case(self) -> None:
        """Volume size not evenly divisible by stride."""
        self._partition_of_unity((57,), tile_size=20, overlap=5)
        self._partition_of_unity((43, 37), tile_size=15, overlap=4)

    def test_partition_of_unity_max_overlap(self) -> None:
        """Partition-of-unity must hold at the maximum allowed overlap (tile_size/2)."""
        self._partition_of_unity((100,), tile_size=20, overlap=10)
        self._partition_of_unity((50, 50), tile_size=16, overlap=8)
        self._partition_of_unity((40, 40, 40), tile_size=20, overlap=10)

    def test_boundary_no_taper(self) -> None:
        """Single tile covering the whole volume should have uniform window."""
        shape = (20, 20, 20)
        specs = compute_tile_specs(shape, tile_size=64, overlap=5)
        assert len(specs) == 1
        w = cosine_window(specs[0])
        np.testing.assert_allclose(w, 1.0)

    def test_interior_tile_has_ramps(self) -> None:
        """Interior tiles should have values < 1.0 at edges."""
        specs = compute_tile_specs((100, 100, 100), tile_size=40, overlap=8)
        # Find an interior tile
        interior = [
            s
            for s in specs
            if all(not lo and not hi for lo, hi in zip(s.border_low, s.border_high))
        ]
        assert len(interior) > 0
        w = cosine_window(interior[0])
        # Corners should be near zero
        assert w[0, 0, 0] < 0.1
        # Center should be 1.0
        mid = tuple(s // 2 for s in interior[0].shape)
        assert w[mid] == pytest.approx(1.0)

    def test_window_values_in_range(self) -> None:
        """Window values must be in (0, 1]."""
        specs = compute_tile_specs((100,), tile_size=40, overlap=10)
        for spec in specs:
            w = cosine_window(spec)
            assert np.all(w > 0), "Window has zero or negative values"
            assert np.all(w <= 1.0 + 1e-7), "Window exceeds 1.0"


# ── Fitting tests (require torch) ────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestFitTile:
    """Tests for fit_tile()."""

    def test_global_coordinates(self) -> None:
        """Tile centers should be translated to global coordinates."""
        from luxar.gsplats.fit_tiled_gsplats import fit_tile

        volume = np.zeros((64, 64, 64), dtype=np.float32)
        volume[20:44, 20:44, 20:44] = 1.0

        # Use tile_size > volume to get a single tile
        specs = compute_tile_specs(volume.shape, tile_size=128, overlap=8)
        assert len(specs) == 1

        result = fit_tile(
            volume,
            specs[0],
            seeds=10,
            n_iters=50,
            verbose=False,
        )

        assert result.n_splats > 0
        for d in range(3):
            assert np.all(result.centers[:, d] >= -1)
            assert np.all(result.centers[:, d] <= volume.shape[d] + 1)

    def test_offset_tile(self) -> None:
        """A tile not at origin should have centers offset by origin."""
        from luxar.gsplats.fit_tiled_gsplats import fit_tile

        volume = np.zeros((80, 40, 40), dtype=np.float32)
        volume[50:70, 10:30, 10:30] = 1.0

        specs = compute_tile_specs(volume.shape, tile_size=40, overlap=8)
        signal_tile = None
        for spec in specs:
            if spec.slices[0].start >= 32:
                signal_tile = spec
                break

        assert signal_tile is not None
        result = fit_tile(
            volume,
            signal_tile,
            seeds=10,
            n_iters=50,
            verbose=False,
        )

        if result.n_splats > 0:
            assert np.all(result.centers[:, 0] >= signal_tile.origin[0] - 5)

    def test_rejects_explicit_seed_array(self) -> None:
        """Explicit seed arrays should be rejected."""
        from luxar.gsplats.fit_tiled_gsplats import fit_tile

        volume = np.ones((32, 32, 32), dtype=np.float32)
        specs = compute_tile_specs(volume.shape, tile_size=32, overlap=0)

        with pytest.raises(ValueError, match="Explicit seed coordinate arrays"):
            fit_tile(
                volume,
                specs[0],
                seeds=np.array([[16, 16, 16]]),
                n_iters=10,
                verbose=False,
            )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestFitTiled:
    """Tests for fit_tiled()."""

    def test_small_volume_round_trip(self) -> None:
        """Tiled fitting on a small volume should produce reasonable results."""
        from luxar.gsplats.fit_tiled_gsplats import fit_tiled

        rng = np.random.RandomState(42)
        volume = np.zeros((64, 64, 64), dtype=np.float32)
        for _ in range(5):
            cx, cy, cz = rng.randint(10, 54, size=3)
            volume[cx - 5 : cx + 5, cy - 5 : cy + 5, cz - 5 : cz + 5] = rng.uniform(
                0.5, 1.0
            )

        result = fit_tiled(
            volume,
            tile_size=40,
            overlap=10,
            seeds=20,
            n_iters=100,
            verbose=False,
        )

        assert result.n_splats > 0
        assert result.stats.get("tiled_fitting") is True
        assert result.stats.get("num_tiles", 0) > 1
        assert "splats_per_tile" in result.stats

    def test_verbose_propagates_to_per_tile_fitting(self) -> None:
        """verbose=False should propagate to fit_gaussian_splats per tile."""
        from unittest.mock import patch

        from luxar.gsplats.fit_tiled_gsplats import fit_tiled

        volume = np.random.RandomState(0).rand(20, 20, 20).astype(np.float32)
        with patch("luxar.gsplats.fit_tiled_gsplats.fit_gaussian_splats") as mock_fit:
            # Make mock return a valid GSplatData
            from luxar.gsplats.gsplat_data import GSplatData

            mock_fit.return_value = GSplatData(
                centers=np.zeros((1, 3), dtype=np.float32),
                amplitudes=np.ones((1,), dtype=np.float32),
                cholesky_factors=np.eye(3, dtype=np.float32)[
                    np.tril_indices(3)
                ].reshape(1, -1),
                stats={"time_seconds": 0.0},
            )

            fit_tiled(volume, tile_size=20, overlap=0, verbose=False)

            # Every call to fit_gaussian_splats should have verbose=False
            for call in mock_fit.call_args_list:
                assert call.kwargs.get("verbose") is False, (
                    "verbose=False not propagated to fit_gaussian_splats"
                )

    def test_degenerates_to_single_tile(self) -> None:
        """Volume smaller than tile_size should behave like non-tiled fitting."""
        from luxar.gsplats.fit_tiled_gsplats import fit_tiled

        volume = np.random.RandomState(0).rand(20, 20, 20).astype(np.float32)
        result = fit_tiled(
            volume,
            tile_size=64,
            overlap=8,
            seeds=10,
            n_iters=50,
            verbose=False,
        )

        assert result.n_splats > 0
        assert result.stats.get("num_tiles") == 1


# ── Tiled + Progressive tests ──────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTiledProgressive:
    """Tests for tiled fitting with progressive=True.

    ``test_tiled_progressive_basic`` and ``test_tiled_progressive_stats``
    share the ``shared_tiled_progressive_fit`` module-scoped fixture (see
    ``conftest.py``) — they originally ran nearly identical fits and
    checked disjoint facets of the result.  The fixture uses
    ``residual_pass_min_iters=30`` to bypass the production 500-iter floor.
    """

    @pytest.mark.slow
    def test_tiled_progressive_basic(self, shared_tiled_progressive_fit) -> None:
        """Basic tiled+progressive produces multi-LOD result."""
        result = shared_tiled_progressive_fit.result
        assert result.n_splats > 0
        assert result.n_additive_sublods >= 1  # At least one LOD
        assert result.stats.get("progressive") is True

    def test_merge_lods_across_tiles(self) -> None:
        """LOD merge correctly combines LODs from multiple tiles."""
        from luxar.gsplats.fit_tiled_gsplats import _merge_lods_across_tiles
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        rng = np.random.RandomState(42)

        # Create 3 fake tile results with 2 LODs each
        tiles = []
        for _ in range(3):
            lods = []
            for level in range(2):
                n = rng.randint(5, 15)
                lods.append(
                    AdditiveSubLOD(
                        centers=rng.rand(n, 2).astype(np.float32),
                        amplitudes=rng.rand(n).astype(np.float32),
                        cholesky_factors=np.tile(
                            np.array([1, 0, 1], dtype=np.float32), (n, 1)
                        ),
                    )
                )
            tiles.append(GSplatData.from_additive_sublods(lods))

        merged = _merge_lods_across_tiles(tiles)

        # Should have 2 LODs
        assert merged.n_additive_sublods == 2
        # LOD 0 = sum of all tiles' LOD 0 splats
        expected_lod0 = sum(t.additive_sublod(0).n_splats for t in tiles)
        assert merged.additive_sublod(0).n_splats == expected_lod0
        # LOD 1 = sum of all tiles' LOD 1 splats
        expected_lod1 = sum(t.additive_sublod(1).n_splats for t in tiles)
        assert merged.additive_sublod(1).n_splats == expected_lod1
        # Total
        assert merged.n_splats == expected_lod0 + expected_lod1

    def test_merge_lods_mismatched_counts(self) -> None:
        """LOD merge handles tiles with different LOD counts (pad to max)."""
        from luxar.gsplats.fit_tiled_gsplats import _merge_lods_across_tiles
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        rng = np.random.RandomState(42)

        # Tile A: 3 LODs, Tile B: 1 LOD
        tile_a = GSplatData.from_additive_sublods(
            [
                AdditiveSubLOD(
                    centers=rng.rand(5, 2).astype(np.float32),
                    amplitudes=rng.rand(5).astype(np.float32),
                    cholesky_factors=np.tile(
                        np.array([1, 0, 1], dtype=np.float32), (5, 1)
                    ),
                )
                for _ in range(3)
            ]
        )
        tile_b = GSplatData(
            centers=rng.rand(8, 2).astype(np.float32),
            amplitudes=rng.rand(8).astype(np.float32),
            cholesky_factors=np.tile(np.array([1, 0, 1], dtype=np.float32), (8, 1)),
        )  # single LOD

        merged = _merge_lods_across_tiles([tile_a, tile_b])

        # Should pad to max LODs = 3
        assert merged.n_additive_sublods == 3
        # LOD 0 should have splats from both tiles
        assert merged.additive_sublod(0).n_splats == 5 + 8
        # LOD 1 and 2 should only have tile_a's splats
        assert merged.additive_sublod(1).n_splats == 5
        assert merged.additive_sublod(2).n_splats == 5

    @pytest.mark.slow
    def test_tiled_progressive_stats(self, shared_tiled_progressive_fit) -> None:
        """Stats reflect progressive tiled fitting."""
        result = shared_tiled_progressive_fit.result
        assert result.stats.get("tiled_fitting") is True
        assert result.stats.get("progressive") is True
        assert result.stats.get("num_tiles", 0) > 1

    def test_tiled_non_progressive_still_works(self) -> None:
        """progressive=False (default) still produces single-LOD result."""
        from luxar.gsplats.fit_tiled_gsplats import fit_tiled

        V = np.random.RandomState(42).rand(32, 32).astype(np.float32)
        result = fit_tiled(
            V,
            tile_size=16,
            overlap=4,
            progressive=False,
            seeds=50,
            n_iters=30,
            verbose=False,
        )
        assert result.n_additive_sublods == 1
