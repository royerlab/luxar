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


@pytest.mark.slow  # ~50s: progressive tiled CPU fitting (CI runs `-m "not slow"`)
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


# ── Background floor handling in tiled fitting ───────────────────


def _empty_result(ndim: int):
    """A fresh 0-splat GSplatData (fresh stats dict per call)."""
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.utils.trils import tril_size

    return GSplatData(
        centers=np.zeros((0, ndim), dtype=np.float32),
        amplitudes=np.zeros((0,), dtype=np.float32),
        cholesky_factors=np.zeros((0, tril_size(ndim)), dtype=np.float32),
        stats={"time_seconds": 0.0},
    )


def _recording_stub(records: list):
    """Stub fitter that records the tile array + floor kwarg it was handed."""

    def stub(tile_data, **kwargs):
        result = _empty_result(tile_data.ndim)
        records.append(
            {
                "data": np.array(tile_data, copy=True),
                "floor": kwargs.get("floor"),
                "result": result,
            }
        )
        return result

    return stub


def _one_splat_stub(records: list):
    """Stub fitter returning a 1-splat result (merge paths need non-empty)."""

    def stub(tile_data, **kwargs):
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.utils.trils import tril_size

        ndim = tile_data.ndim
        chol = np.zeros((1, tril_size(ndim)), dtype=np.float32)
        k = 0  # identity, packed lower-triangular row-major
        for i in range(ndim):
            for j in range(i + 1):
                if i == j:
                    chol[0, k] = 1.0
                k += 1
        result = GSplatData(
            centers=np.full((1, ndim), 1.0, dtype=np.float32),
            amplitudes=np.ones((1,), dtype=np.float32),
            cholesky_factors=chol,
            stats={"time_seconds": 0.0},
        )
        records.append({"floor": kwargs.get("floor"), "result": result})
        return result

    return stub


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestTiledFloorHandling:
    """The floor must be resolved once per run and subtracted BEFORE windowing."""

    def _pedestal_volume(
        self, pedestal: float = 100.0, base: float = 0.5
    ) -> np.ndarray:
        """96x96 volume: pedestal + small base + a bump straddling a tile seam."""
        yy, xx = np.meshgrid(np.arange(96), np.arange(96), indexing="ij")
        # tile_size=48, overlap=16 -> the bump sits inside overlap zones
        bump = 50.0 * np.exp(-(((yy - 40) ** 2 + (xx - 48) ** 2) / (2 * 8.0**2)))
        # `base` keeps every floor-subtracted tile above the near-zero skip,
        # so tiles map 1:1 onto the recorded stub calls.
        return (pedestal + base + bump).astype(np.float32)

    def test_partition_of_unity_with_floor(self, monkeypatch) -> None:
        """Recorded tiles must sum to clip(V - m, 0) everywhere, incl. overlaps.

        Subtracting the floor AFTER windowing breaks this: an overlap zone
        (w_A + w_B = 1) merges to V - 2m instead of V - m, and clip erases a
        seam-centred bump entirely.
        """
        import luxar.gsplats.fit_tiled_gsplats as ftg

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))

        pedestal = 100.0
        volume = self._pedestal_volume(pedestal=pedestal)
        ftg.fit_tiled(volume, tile_size=48, overlap=16, floor=pedestal, verbose=False)

        specs = compute_tile_specs(volume.shape, tile_size=48, overlap=16)
        assert len(records) == len(specs)

        accumulated = np.zeros(volume.shape, dtype=np.float64)
        for spec, rec in zip(specs, records):
            assert rec["data"].shape == spec.shape
            accumulated[spec.slices] += rec["data"].astype(np.float64)

        expected = np.clip(volume.astype(np.float64) - pedestal, 0.0, None)
        np.testing.assert_allclose(accumulated, expected, atol=1e-3)

    def test_inner_fit_receives_floor_none_standard(self, monkeypatch) -> None:
        """No live floor spec may reach fit_gaussian_splats (standard branch)."""
        import luxar.gsplats.fit_tiled_gsplats as ftg

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))

        volume = self._pedestal_volume()
        ftg.fit_tiled(volume, tile_size=48, overlap=16, floor="auto", verbose=False)

        assert len(records) > 1
        for rec in records:
            assert rec["floor"] == "none"

    def test_inner_fit_receives_floor_none_progressive(self, monkeypatch) -> None:
        """No live floor spec may reach the progressive fitter either."""
        import luxar.gsplats.fit_progressive_gsplats as fpg
        import luxar.gsplats.fit_tiled_gsplats as ftg

        records: list = []
        monkeypatch.setattr(
            fpg, "fit_progressive_gaussian_splats", _recording_stub(records)
        )

        volume = self._pedestal_volume()
        ftg.fit_tiled(
            volume,
            tile_size=48,
            overlap=16,
            floor="auto",
            progressive=True,
            verbose=False,
        )

        assert len(records) > 1
        for rec in records:
            assert rec["floor"] == "none"

    def test_one_shared_floor_level_across_tiles(self, monkeypatch) -> None:
        """Every tile subtracts the identical, whole-volume-resolved level."""
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))

        # Mostly a noisy pedestal ~100, plus one densely signalled interior
        # region — a per-tile "auto" there would estimate a wildly different
        # (signal-level) floor and subtract real signal.
        rng = np.random.RandomState(0)
        volume = rng.normal(100.0, 1.0, size=(96, 96)).astype(np.float32)
        volume[40:80, 40:80] += 150.0

        ftg.fit_tiled(volume, tile_size=48, overlap=16, floor="auto", verbose=False)

        assert len(records) > 1
        floors = [rec["result"].stats["applied_floor"] for rec in records]
        assert all(f is not None for f in floors)
        assert all(f == floors[0] for f in floors)
        expected = resolve_volume_floor(volume, "auto")
        assert floors[0] == pytest.approx(expected)

    def test_numeric_floor_subtracted_verbatim(self, monkeypatch) -> None:
        """An explicit numeric floor is subtracted exactly as given."""
        import luxar.gsplats.fit_tiled_gsplats as ftg

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))

        volume = self._pedestal_volume(pedestal=100.0)
        ftg.fit_tiled(volume, tile_size=48, overlap=16, floor=100.0, verbose=False)

        specs = compute_tile_specs(volume.shape, tile_size=48, overlap=16)
        for spec, rec in zip(specs, records):
            w = cosine_window(spec)
            expected = np.clip(volume[spec.slices] - 100.0, 0.0, None) * w
            np.testing.assert_allclose(rec["data"], expected, atol=1e-4)
            assert rec["result"].stats["applied_floor"] == 100.0

    def test_negative_background_level_is_subtracted(self, monkeypatch) -> None:
        """A negative resolved level (dark-frame-corrected data) IS subtracted.

        Floor suppression means "put the background at 0": if the background
        sits at -2, then V - (-2) = V + 2 is correct — and it is what the
        non-tiled path's ``image_min = max(resolved_floor, image_min)`` does
        with a negative level. Dropping a negative level here would silently
        fall back to per-tile hard-min normalization of the windowed tile —
        exactly the seam-producing bug tiled floor resolution exists to fix.
        A constant offset subtracted before windowing also preserves the
        partition-of-unity invariant.
        """
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        rng = np.random.RandomState(5)
        volume = rng.normal(-2.0, 1.0, size=(96, 96)).astype(np.float32)
        yy, xx = np.meshgrid(np.arange(96), np.arange(96), indexing="ij")
        volume += (
            50.0 * np.exp(-(((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 8.0**2)))
        ).astype(np.float32)

        level = resolve_volume_floor(volume, "auto")
        assert level is not None and level < 0.0

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        # Must complete without raising (the resolved negative level is a
        # contract-level value for fit_tile, not a user spec to re-validate).
        ftg.fit_tiled(volume, tile_size=48, overlap=16, floor="auto", verbose=False)

        specs = compute_tile_specs(volume.shape, tile_size=48, overlap=16)
        assert len(records) == len(specs)

        # Every tile subtracts the SAME negative level ...
        floors = [rec["result"].stats["applied_floor"] for rec in records]
        assert all(f == level for f in floors)

        # ... and the recorded tiles still sum to clip(V - m, 0) everywhere.
        accumulated = np.zeros(volume.shape, dtype=np.float64)
        for spec, rec in zip(specs, records):
            accumulated[spec.slices] += rec["data"].astype(np.float64)
        expected = np.clip(volume.astype(np.float64) - level, 0.0, None)
        np.testing.assert_allclose(accumulated, expected, atol=1e-3)

    def test_too_high_numeric_floor_refused_not_erasing(self, monkeypatch) -> None:
        """A numeric floor above the volume max is refused, not silently fatal.

        Matches the non-tiled path's semantics (warn + ignore): every tile is
        still fitted on unmodified data instead of the whole dataset silently
        emptying and the save step erroring.
        """
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        volume = self._pedestal_volume(pedestal=100.0)
        assert resolve_volume_floor(volume, 10_000.0, guard_numeric=True) is None
        # Pre-resolved levels handed down to tiles keep the read-free
        # short-circuit (no guard) by default.
        assert resolve_volume_floor(_ExplodingArray(), 10_000.0) == 10_000.0

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        ftg.fit_tiled(volume, tile_size=48, overlap=16, floor=10_000.0, verbose=False)

        specs = compute_tile_specs(volume.shape, tile_size=48, overlap=16)
        assert len(records) == len(specs)  # every tile still fitted
        for spec, rec in zip(specs, records):
            w = cosine_window(spec)
            np.testing.assert_allclose(rec["data"], volume[spec.slices] * w, rtol=1e-5)
            assert rec["result"].stats["applied_floor"] is None

    def test_fit_tile_resolves_live_spec_against_whole_volume(
        self, monkeypatch
    ) -> None:
        """fit_tile with a LIVE spec must resolve it on the WHOLE volume.

        Exercises fit_tile's own spec-resolution branch (the --tile k/M worker
        path, which fit_tiled bypasses by pre-resolving): a regression back to
        per-tile estimation would subtract a different, tile-local level.
        """
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        rng = np.random.RandomState(6)
        volume = rng.normal(100.0, 1.0, size=(96, 96)).astype(np.float32)
        volume[40:80, 40:80] += 150.0  # bright/dense region skews one tile

        whole = resolve_volume_floor(volume, "auto")
        assert whole is not None

        # Pick a tile that overlaps the bright region (so the fitter runs)
        # AND whose tile-local estimate differs from the whole-volume one.
        specs = compute_tile_specs(volume.shape, tile_size=48, overlap=16)
        chosen = None
        for s in specs:
            if float(volume[s.slices].max()) < 200.0:
                continue
            local = resolve_volume_floor(np.asarray(volume[s.slices]), "auto")
            if local is not None and abs(local - whole) > 1e-9:
                chosen = (s, local)
                break
        assert chosen is not None, "no bright tile with a distinct local estimate"
        spec, local = chosen

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        result = ftg.fit_tile(volume, spec, floor="auto", verbose=False)

        applied = result.stats["applied_floor"]
        assert applied == pytest.approx(whole)
        assert applied != pytest.approx(local)
        # And the tile handed to the fitter had the WHOLE-volume level removed.
        w = cosine_window(spec)
        expected = np.clip(volume[spec.slices] - whole, 0.0, None) * w
        np.testing.assert_allclose(records[-1]["data"], expected, atol=1e-4)

    def test_all_background_tile_skipped_as_empty(self) -> None:
        """A tile wholly at/below the global floor clips to zero and is
        skipped with a 0-splat result (no fitter call, no error)."""
        import luxar.gsplats.fit_tiled_gsplats as ftg

        volume = np.full((96, 96), 100.0, dtype=np.float32)
        volume[64:, 64:] += 50.0  # signal elsewhere keeps the floor meaningful

        specs = compute_tile_specs(volume.shape, tile_size=48, overlap=16)
        background_spec = specs[0]  # origin (0, 0): constant pedestal only
        assert float(volume[background_spec.slices].max()) == 100.0

        result = ftg.fit_tile(volume, background_spec, floor=100.0, verbose=False)
        assert result.n_splats == 0
        assert result.stats.get("skipped") is True
        assert result.stats["applied_floor"] == 100.0

    def test_applied_floor_recorded_on_merge_paths(self, monkeypatch) -> None:
        """Flat merge stamps stats['applied_floor']; the partition path stamps
        the returned node's meta['applied_floor'] (in-memory bookkeeping)."""
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        volume = self._pedestal_volume(pedestal=100.0)
        expected = resolve_volume_floor(volume, "auto")
        assert expected is not None

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _one_splat_stub(records))

        flat = ftg.fit_tiled(
            volume,
            tile_size=48,
            overlap=16,
            floor="auto",
            cull_retention=None,
            verbose=False,
        )
        assert flat.stats["applied_floor"] == pytest.approx(expected)

        node = ftg.fit_tiled(
            volume,
            tile_size=48,
            overlap=16,
            floor="auto",
            cull_retention=None,
            partition=True,
            verbose=False,
        )
        assert node.meta["applied_floor"] == pytest.approx(expected)

    def test_floor_none_subtracts_nothing(self, monkeypatch) -> None:
        """floor='none' leaves the tile untouched (behaviour unchanged)."""
        import luxar.gsplats.fit_tiled_gsplats as ftg

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))

        volume = self._pedestal_volume()
        ftg.fit_tiled(volume, tile_size=48, overlap=16, floor="none", verbose=False)

        specs = compute_tile_specs(volume.shape, tile_size=48, overlap=16)
        assert len(records) == len(specs)
        for spec, rec in zip(specs, records):
            w = cosine_window(spec)
            np.testing.assert_allclose(rec["data"], volume[spec.slices] * w, rtol=1e-5)
            assert rec["floor"] == "none"
            assert rec["result"].stats["applied_floor"] is None


class _CountingArray:
    """Array shim that counts the number of elements returned by reads."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr
        self.voxels_read = 0

    @property
    def shape(self) -> tuple:
        return self._arr.shape

    @property
    def ndim(self) -> int:
        return self._arr.ndim

    def __getitem__(self, key):
        out = self._arr[key]
        self.voxels_read += int(np.asarray(out).size)
        return out


class _ExplodingArray:
    """Array shim that fails on any read — data must never be touched."""

    shape = (1024, 1024, 1024)
    ndim = 3

    def __getitem__(self, key):
        raise AssertionError("volume data must not be read for this spec")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestResolveVolumeFloor:
    """Unit tests for the bounded whole-volume floor resolver."""

    def _pedestal_volume(self) -> np.ndarray:
        rng = np.random.RandomState(1)
        vol = rng.normal(100.0, 2.0, size=(32, 64, 64)).astype(np.float32)
        vol[10:20, 20:40, 20:40] += 300.0  # bright signal region
        return vol

    def test_recovers_known_pedestal(self) -> None:
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        f = resolve_volume_floor(self._pedestal_volume(), "auto")
        assert f is not None
        assert f == pytest.approx(100.0, abs=5.0)

    def test_deterministic_across_calls_and_access_patterns(self) -> None:
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        vol = self._pedestal_volume()
        f1 = resolve_volume_floor(vol, "auto")
        f2 = resolve_volume_floor(vol, "auto")
        f3 = resolve_volume_floor(_CountingArray(vol), "auto")
        assert f1 == f2 == f3

    def test_memory_bound_respected(self, monkeypatch) -> None:
        import luxar.gsplats.fitting.preprocessing as pp

        budget = 4096
        monkeypatch.setattr(pp, "FLOOR_SAMPLE_BUDGET_VOXELS", budget)
        rng = np.random.RandomState(2)
        base = rng.normal(100.0, 5.0, size=(256, 16, 16)).astype(np.float32)
        shim = _CountingArray(base)  # 65,536 voxels >> budget

        f = pp.resolve_volume_floor(shim, "p10")
        assert f is not None
        assert 0 < shim.voxels_read <= budget

    def test_bounded_sample_is_deterministic(self, monkeypatch) -> None:
        import luxar.gsplats.fitting.preprocessing as pp

        monkeypatch.setattr(pp, "FLOOR_SAMPLE_BUDGET_VOXELS", 4096)
        rng = np.random.RandomState(3)
        base = rng.normal(100.0, 5.0, size=(256, 16, 16)).astype(np.float32)
        f1 = pp.resolve_volume_floor(base, "p10")
        f2 = pp.resolve_volume_floor(_CountingArray(base), "p10")
        assert f1 == f2

    def test_guard_floor_at_or_above_max_returns_none(self) -> None:
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        rng = np.random.RandomState(4)
        vol = rng.normal(100.0, 5.0, size=(16, 16, 16)).astype(np.float32)
        # p100 resolves to the sampled max -> would erase all signal
        assert resolve_volume_floor(vol, "p100") is None

    def test_numeric_specs_short_circuit(self) -> None:
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        shim = _ExplodingArray()
        assert resolve_volume_floor(shim, 42.0) == 42.0
        assert resolve_volume_floor(shim, "17.5") == 17.5
        assert resolve_volume_floor(shim, "none") is None
        assert resolve_volume_floor(shim, None) is None
        assert resolve_volume_floor(shim, 0.0) is None  # 0 disables
        assert resolve_volume_floor(shim, -3.0) == -3.0  # negative level, verbatim

    def test_negative_background_level_is_resolved(self) -> None:
        """auto/pN can go negative on dark-frame-corrected float data; the
        negative level is returned (subtracting it shifts the background up
        to 0, matching the non-tiled path), not discarded."""
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        rng = np.random.RandomState(7)
        vol = rng.normal(-2.0, 1.0, size=(32, 32, 32)).astype(np.float32)
        vol[10:20, 10:20, 10:20] += 100.0
        auto = resolve_volume_floor(vol, "auto")
        assert auto is not None and auto < 0.0
        assert auto == pytest.approx(-2.0, abs=1.0)  # near the background mode
        p10 = resolve_volume_floor(vol, "p10")
        assert p10 is not None and p10 < 0.0
        assert p10 == pytest.approx(float(np.percentile(vol, 10.0)))

    def test_numeric_guard_refuses_floor_above_max(self) -> None:
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        rng = np.random.RandomState(8)
        vol = rng.normal(100.0, 5.0, size=(16, 16, 16)).astype(np.float32)
        assert resolve_volume_floor(vol, 1e6, guard_numeric=True) is None
        # Below the max: passes the guard and is returned verbatim.
        assert resolve_volume_floor(vol, 50.0, guard_numeric=True) == 50.0
        # Without the guard the numeric short-circuits (never sampled).
        assert resolve_volume_floor(_ExplodingArray(), 1e6) == 1e6

    def test_memory_bound_respected_small_leading_axis(self, monkeypatch) -> None:
        """A small leading axis must not defeat the budget: the sampler slabs
        along the LONGEST axis, not hard-coded axis 0."""
        import luxar.gsplats.fitting.preprocessing as pp

        budget = 4096
        monkeypatch.setattr(pp, "FLOOR_SAMPLE_BUDGET_VOXELS", budget)
        rng = np.random.RandomState(9)
        base = rng.normal(100.0, 5.0, size=(1, 256, 256)).astype(np.float32)
        shim = _CountingArray(base)  # 65,536 voxels >> budget

        f = pp.resolve_volume_floor(shim, "p10")
        assert f is not None
        assert 0 < shim.voxels_read <= budget

    def test_single_block_sample_reads_the_middle(self) -> None:
        """When the budget allows only one block, it is taken from the middle
        of the sampled axis: the first slab of a real stack is systematically
        atypical (vignetting, no sample in frame, axial gradients)."""
        import luxar.gsplats.fitting.preprocessing as pp

        # 64 slabs of 16x16 = 256 voxels each; budget 256 -> exactly 1 slab.
        vol = np.broadcast_to(
            np.arange(64, dtype=np.float32)[:, None, None], (64, 16, 16)
        ).copy()
        sample = pp._sample_volume_for_floor(vol, 256)
        assert sample is not None
        assert sample.size == 256
        # The single slab comes from the middle of axis 0, not index 0.
        assert np.unique(sample).tolist() == [31.0]


def _single_tile_ctx(tile: str, tile_size: int = 48, overlap: int = 16):
    """A minimal FitPipelineCtx for exercising the single-tile worker path.

    Only the fields ``fit_single_tile`` reads are given meaningful values;
    everything else is None/inert.
    """
    import dataclasses

    from luxar.cli.gsplat_ops.fitting_fit_utils import FitPipelineCtx

    kwargs: dict = {f.name: None for f in dataclasses.fields(FitPipelineCtx)}
    kwargs.update(
        tile=tile,
        tile_size=tile_size,
        tile_overlap=overlap,
        progressive=False,
        max_splats_per_pass=5000,
        psnr_patience=0.5,
        max_passes=None,
    )
    return FitPipelineCtx(**kwargs)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestSingleTileWorkerFloor:
    """The standalone ``--tile k/M`` worker resolves the user's floor spec
    itself, guarded — it is that worker's own user-spec entry point."""

    def _pedestal_volume(self) -> np.ndarray:
        rng = np.random.RandomState(11)
        vol = rng.normal(100.0, 1.0, size=(96, 96)).astype(np.float32)
        vol[40:80, 40:80] += 150.0
        return vol

    def test_live_spec_resolved_against_whole_volume(self, monkeypatch) -> None:
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.cli.gsplat_ops.fitting_fit_utils import fit_single_tile
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        volume = self._pedestal_volume()
        expected = resolve_volume_floor(volume, "auto")
        assert expected is not None

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        result = fit_single_tile(
            _single_tile_ctx("0/4"), volume, {"floor": "auto"}, None
        )
        assert result.stats["applied_floor"] == pytest.approx(expected)

    def test_too_high_numeric_floor_is_guarded(self, monkeypatch) -> None:
        """A numeric floor above the volume max is refused HERE (the guard),
        not taken at face value by fit_tile — which would silently clip the
        whole tile to zero and skip the fit."""
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.cli.gsplat_ops.fitting_fit_utils import fit_single_tile

        volume = self._pedestal_volume()
        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        result = fit_single_tile(
            _single_tile_ctx("0/4"), volume, {"floor": 10_000.0}, None
        )
        # Refused level -> the tile is fitted on unmodified (windowed) data.
        assert len(records) == 1
        assert result.stats["applied_floor"] is None
        specs = compute_tile_specs(volume.shape, 48, 16)
        w = cosine_window(specs[0])
        np.testing.assert_allclose(
            records[0]["data"], volume[specs[0].slices] * w, rtol=1e-5
        )

    def test_null_floor_config_means_disabled(self, monkeypatch) -> None:
        """``floor: null`` in a --config YAML disables the floor on the
        single-tile worker, exactly as on the sequential and non-tiled paths
        (it must not be remapped to 'auto')."""
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.cli.gsplat_ops.fitting_fit_utils import fit_single_tile

        volume = self._pedestal_volume()
        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        result = fit_single_tile(_single_tile_ctx("0/4"), volume, {"floor": None}, None)
        assert len(records) == 1
        assert records[0]["floor"] == "none"
        assert result.stats["applied_floor"] is None
        specs = compute_tile_specs(volume.shape, 48, 16)
        w = cosine_window(specs[0])
        np.testing.assert_allclose(
            records[0]["data"], volume[specs[0].slices] * w, rtol=1e-5
        )
