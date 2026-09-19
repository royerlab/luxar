"""Tests for tiled Gaussian splat fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from luxar.gsplats.tiling import compute_tile_specs, cosine_window

if TYPE_CHECKING:
    from pathlib import Path

    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.tiling import TileSpec

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

    @pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
    def test_denoise_receives_global_norm_range(self, monkeypatch) -> None:
        """Per-tile denoise must be handed the WHOLE-volume range (issue #754).

        Guards the dict-key ↔ kwarg-name contract:
        ``_denoise_params['norm_range']`` must reach
        ``denoise_volume_array(norm_range=...)`` unchanged — a rename on either
        side would otherwise pass silently. The spy wraps the real function
        (capture then delegate) so the fit still runs.
        """
        import luxar.gsplats.preprocessing.denoise_pipeline as _dp
        from luxar.gsplats.fit_tiled_gsplats import fit_tile

        volume = np.zeros((16, 32, 32), dtype=np.float32)
        volume[4:12, 8:24, 8:24] = 1.0
        # A deliberate whole-volume range distinct from any tile's own min/max.
        global_range = (0.0, 5.0)

        real_denoise = _dp.denoise_volume_array
        captured: dict = {}

        def _spy(tile, *args, **kwargs):
            captured["norm_range"] = kwargs.get("norm_range", "MISSING")
            return real_denoise(tile, *args, **kwargs)

        monkeypatch.setattr(_dp, "denoise_volume_array", _spy)

        specs = compute_tile_specs(volume.shape, tile_size=64, overlap=8)
        assert len(specs) == 1

        fit_tile(
            volume,
            specs[0],
            seeds=10,
            n_iters=20,
            verbose=False,
            _denoise_h=0.05,
            _denoise_params={
                "patch_size": 3,
                "search_distance": 5,
                "backend": "skimage",
                "device": None,
                "use_2d": False,
                "norm_range": global_range,
            },
        )

        assert captured.get("norm_range") == global_range

    @pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
    def test_progressive_respects_output_space_real(self) -> None:
        """Regression for issue #733: the progressive branch must honour
        ``output_space='real'`` with an anisotropic ``voxel_size``.

        Previously the progressive path silently returned voxel-space centers
        and Cholesky factors while the non-progressive path returned physical
        coordinates — a silent factor-of-``voxel_size`` distortion. Here the
        volume is anisotropic (5x along axis 0), so a correct conversion makes
        the real-space axis-0 extent (and the L[0,0] Cholesky scale) ~5x the
        voxel-space one, and matches the non-progressive real result.
        """
        from luxar.gsplats.fit_tiled_gsplats import fit_tile

        voxel_size = (5.0, 1.0, 1.0)
        volume = np.zeros((16, 16, 16), dtype=np.float32)
        volume[3:6, 4:8, 4:8] = 1.0
        volume[10:13, 8:12, 8:12] = 1.0

        # tile_size > volume → single tile isolates fit_tile's own conversion.
        specs = compute_tile_specs(volume.shape, tile_size=64, overlap=8)
        assert len(specs) == 1

        def _fit(progressive: bool, output_space: str):
            return fit_tile(
                volume,
                specs[0],
                voxel_size=voxel_size,
                output_space=output_space,
                progressive=progressive,
                seeds=200,
                n_iters=100,
                verbose=False,
            )

        prog_real = _fit(progressive=True, output_space="real")
        prog_voxel = _fit(progressive=True, output_space="voxel")
        nonprog_real = _fit(progressive=False, output_space="real")

        assert prog_real.n_splats > 0
        assert prog_voxel.n_splats > 0
        assert nonprog_real.n_splats > 0

        def _extent(res, axis: int) -> float:
            c = res.centers[:, axis]
            return float(c.max() - c.min())

        def _axis0_extent(res) -> float:
            return _extent(res, 0)

        # Real-space axis-0 extent must be ~voxel_size[0]x the voxel-space one
        # (physical conversion applied), NOT ~1x (silent voxel-space output).
        assert _axis0_extent(prog_real) == pytest.approx(
            _axis0_extent(prog_voxel) * voxel_size[0], rel=0.25
        )
        # An unscaled axis (voxel_size 1.0) must stay ~1x — a wrong fix applying
        # the scalar voxel_size[0]=5 to ALL axes would inflate it to ~5x.
        assert _extent(prog_real, 1) == pytest.approx(
            _extent(prog_voxel, 1) * voxel_size[1], rel=0.25
        )
        # ...and it must match the non-progressive real result (both physical).
        assert _axis0_extent(prog_real) == pytest.approx(
            _axis0_extent(nonprog_real), rel=0.3
        )
        # The L[0,0] Cholesky entry is scaled by voxel_size[0] too.
        assert float(prog_real.cholesky_factors[:, 0].mean()) == pytest.approx(
            float(prog_voxel.cholesky_factors[:, 0].mean()) * voxel_size[0],
            rel=0.25,
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
        """Basic tiled+progressive produces a flat, non-empty result."""
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
        floors = [rec["result"].stats["floor"] for rec in records]
        assert all(f is not None for f in floors)
        assert all(f == floors[0] for f in floors)
        expected = resolve_volume_floor(volume, "auto")
        assert floors[0] == pytest.approx(expected)

    def test_expensive_floor_resolution_happens_exactly_once(self, monkeypatch) -> None:
        """The string/"auto" floor spec is resolved exactly ONCE per run.

        ``fit_tiled`` resolves the user's floor spec (here ``"auto"``) against
        the whole volume a single time, then hands each tile the resolved
        NUMERIC level; every per-tile ``resolve_volume_floor_denoised`` call
        therefore gets a cheap numeric short-circuit, never the expensive "auto"
        sampling resolution. ``test_one_shared_floor_level_across_tiles`` only
        asserts the tiles agree on a floor VALUE — it would still pass if a
        regression re-resolved ``"auto"`` per tile (they'd agree, just
        expensively) — so it does not pin this. We spy on the ``spec``
        argument of every ``resolve_volume_floor_denoised`` call and require
        exactly one string spec (``"auto"``), the rest numeric.

        BOTH resolution doors are spied: the wrapper the tiled paths call, and
        the raw ``resolve_volume_floor`` underneath it. A future per-tile call
        going straight to the raw one (a per-tile RAW pedestal — worse than the
        expense) would be invisible to the wrapper spy alone.
        """
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.gsplats.fitting import preprocessing as pp

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))

        real_resolve = ftg.resolve_volume_floor_denoised
        specs: list = []

        def spy(volume, floor, **kwargs):
            specs.append(floor)
            return real_resolve(volume, floor, **kwargs)

        monkeypatch.setattr(ftg, "resolve_volume_floor_denoised", spy)

        real_raw = pp.resolve_volume_floor
        raw_specs: list = []

        def raw_spy(volume, floor, **kwargs):
            raw_specs.append(floor)
            return real_raw(volume, floor, **kwargs)

        monkeypatch.setattr(pp, "resolve_volume_floor", raw_spy)

        rng = np.random.RandomState(0)
        volume = rng.normal(100.0, 1.0, size=(96, 96)).astype(np.float32)
        volume[40:80, 40:80] += 150.0

        ftg.fit_tiled(volume, tile_size=48, overlap=16, floor="auto", verbose=False)

        # Multiple tiles ran, but the string spec was resolved only once.
        assert len(records) > 1
        string_specs = [s for s in specs if isinstance(s, str)]
        assert len(string_specs) == 1
        assert string_specs[0] == "auto"

        # Denoise is off here, so every wrapper call delegates to the raw
        # resolver exactly once — and only ONE of those carries the string spec.
        assert len(raw_specs) == len(specs)
        assert [s for s in raw_specs if isinstance(s, str)] == ["auto"]

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
            assert rec["result"].stats["floor"] == 100.0

    def test_pre_resolved_floor_skips_run_level_resolution(self, monkeypatch) -> None:
        """The CLI divisor scan hands ``fit_tiled`` its resolved level once."""
        import luxar.gsplats.fit_tiled_gsplats as ftg

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))

        real_resolve = ftg.resolve_volume_floor_denoised
        run_level_guard_values: list[object] = []

        def spy(volume, floor, **kwargs):
            if "guard_numeric" in kwargs:
                run_level_guard_values.append(kwargs["guard_numeric"])
            return real_resolve(volume, floor, **kwargs)

        monkeypatch.setattr(ftg, "resolve_volume_floor_denoised", spy)

        volume = self._pedestal_volume(pedestal=100.0)
        ftg.fit_tiled(
            volume,
            tile_size=48,
            overlap=16,
            floor=100.0,
            _floor_resolved=True,
            verbose=False,
        )

        assert records
        assert run_level_guard_values == []

    @pytest.mark.parametrize(
        ("floor", "applied_floor", "pedestal"),
        [("none", None, 0.0), (3.0, 3.0, 2.0)],
    )
    def test_nonempty_scan_matches_fitted_tiles(
        self, monkeypatch, floor, applied_floor, pedestal
    ) -> None:
        """Every tile counted for the seed divisor must enter the fitter."""
        import luxar.gsplats.fit_tiled_gsplats as ftg

        volume = np.full((96, 96), pedestal, dtype=np.float32)
        volume[:24, :24] = 10.0
        specs = compute_tile_specs(volume.shape, tile_size=24, overlap=4)

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        ftg.fit_tiled(
            volume,
            tile_size=24,
            overlap=4,
            floor=floor,
            _floor_resolved=True,
            verbose=False,
        )

        predicted = ftg.count_nonempty_tiles(volume, specs, applied_floor)
        assert predicted == 4
        assert len(records) == predicted

    def test_negative_background_level_is_subtracted(self, monkeypatch) -> None:
        """A negative resolved level (dark-frame-corrected data) IS subtracted.

        Floor suppression means "put the background at 0": if the background
        sits at -2, then V - (-2) = V + 2 is correct. The tiled path must apply
        that concrete level before apodization. Dropping a negative level here would silently
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
        floors = [rec["result"].stats["floor"] for rec in records]
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
            assert rec["result"].stats["floor"] is None

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

        applied = result.stats["floor"]
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
        assert result.stats["floor"] == 100.0

    def test_floor_recorded_on_merge_paths(self, monkeypatch) -> None:
        """Flat merge stamps stats['floor']; the partition path stamps the
        returned root node's meta['floor'], which the tree writer promotes into
        the store's pipeline/ group on save (#1175)."""
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
        assert flat.stats["floor"] == pytest.approx(expected)

        node = ftg.fit_tiled(
            volume,
            tile_size=48,
            overlap=16,
            floor="auto",
            cull_retention=None,
            partition=True,
            verbose=False,
        )
        assert node.meta["floor"] == pytest.approx(expected)

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
            assert rec["result"].stats["floor"] is None


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


class _ShapeOnlyArray:
    """Lazy array shim that records requested regions without storing a volume."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.read_regions: list[tuple[tuple[int, int, int], ...]] = []
        self.voxels_read = 0

    def __getitem__(self, key: object) -> np.ndarray:
        if key is Ellipsis:
            indices = (slice(None),) * self.ndim
        else:
            assert isinstance(key, tuple)
            assert len(key) <= self.ndim
            indices = key + (slice(None),) * (self.ndim - len(key))

        region: list[tuple[int, int, int]] = []
        read_shape: list[int] = []
        for index, axis_len in zip(indices, self.shape, strict=True):
            assert isinstance(index, slice)
            start, stop, step = index.indices(axis_len)
            region.append((start, stop, step))
            read_shape.append(len(range(start, stop, step)))

        shape = tuple(read_shape)
        self.read_regions.append(tuple(region))
        self.voxels_read += int(np.prod(shape, dtype=np.int64))
        return np.zeros(shape, dtype=np.float32)


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

    @pytest.mark.parametrize(
        ("shape", "budget"),
        [
            ((50, 20, 128, 128), 4096),
            ((97, 89, 83, 79), 1000),
        ],
    )
    def test_memory_bound_when_one_slab_exceeds_budget(
        self, shape: tuple[int, ...], budget: int
    ) -> None:
        """Oversized cross-sections are cropped recursively before reading."""
        import luxar.gsplats.fitting.preprocessing as pp

        first = _ShapeOnlyArray(shape)
        second = _ShapeOnlyArray(shape)
        sample = pp._sample_volume_for_floor(first, budget)
        repeated = pp._sample_volume_for_floor(second, budget)

        assert sample is not None
        assert repeated is not None
        assert 0 < sample.size == first.voxels_read <= budget
        assert repeated.size == second.voxels_read <= budget
        assert first.read_regions == second.read_regions

    def test_non_positive_sample_budget_is_rejected(self) -> None:
        import luxar.gsplats.fitting.preprocessing as pp

        with pytest.raises(ValueError, match="at least 1 voxel"):
            pp._sample_volume_for_floor(_ExplodingArray(), 0)

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

    from luxar.cli.gsplat_ops.fitting.fit_utils import FitPipelineCtx

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
    """The standalone ``--tile k/M`` worker resolves the floor spec itself.

    A volume-derived spec (``auto``/``pNN``) becomes a level here for the first
    time, so it is resolved against the WHOLE volume and guarded. A CONCRETE
    numeric is applied unvetoed (#1174): it is normally a level a parent already
    resolved and guarded for all the workers, and re-guarding it per sub-volume is
    the pedestal disagreement the shared resolution removes — a too-high one is
    announced loudly instead of being silently dropped.
    """

    def _pedestal_volume(self) -> np.ndarray:
        rng = np.random.RandomState(11)
        vol = rng.normal(100.0, 1.0, size=(96, 96)).astype(np.float32)
        vol[40:80, 40:80] += 150.0
        return vol

    def test_live_spec_resolved_against_whole_volume(self, monkeypatch) -> None:
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.cli.gsplat_ops.fitting.fit_utils import fit_single_tile
        from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

        volume = self._pedestal_volume()
        expected = resolve_volume_floor(volume, "auto")
        assert expected is not None

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        result = fit_single_tile(
            _single_tile_ctx("0/4"), volume, {"floor": "auto"}, None
        )
        assert result.stats["floor"] == pytest.approx(expected)

    def test_too_high_numeric_floor_is_applied_but_announced(
        self, monkeypatch, capsys
    ) -> None:
        """A numeric floor above this worker's max is APPLIED, loudly (#1174).

        Behaviour change: this used to be guarded here and reported-and-IGNORED,
        so two uniform sub-paths disagreed about the same flag — the sequential
        in-process fit applied its one resolved level while a ``--tile k/M`` / ``-j
        N`` worker quietly dropped it. The number now wins (no worker may disagree
        with its siblings about the pedestal), and the resulting all-zero tile is
        explained up front instead of surfacing as a bare "empty store".
        """
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.cli.gsplat_ops.fitting.fit_utils import fit_single_tile

        volume = self._pedestal_volume()
        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        result = fit_single_tile(
            _single_tile_ctx("0/4"), volume, {"floor": 10_000.0}, None
        )
        assert result.stats["floor"] == pytest.approx(10_000.0)
        # Subtracting it clips the whole tile to zero, so the fit is skipped
        # outright and the tile yields 0 splats — exactly the outcome the warning
        # has to name up front (level, sampled max, consequence).
        assert records == []
        assert result.centers.shape[0] == 0
        out = capsys.readouterr().out
        assert "⚠" in out and "10000" in out
        assert "0 SPLATS" in out and "skipped by the merge" in out

    def test_null_floor_config_means_disabled(self, monkeypatch) -> None:
        """``floor: null`` in a --config YAML disables the floor on the
        single-tile worker, exactly as on the sequential and non-tiled paths
        (it must not be remapped to 'auto')."""
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.cli.gsplat_ops.fitting.fit_utils import fit_single_tile

        volume = self._pedestal_volume()
        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        result = fit_single_tile(_single_tile_ctx("0/4"), volume, {"floor": None}, None)
        assert len(records) == 1
        assert records[0]["floor"] == "none"
        assert result.stats["floor"] is None
        specs = compute_tile_specs(volume.shape, 48, 16)
        w = cosine_window(specs[0])
        np.testing.assert_allclose(
            records[0]["data"], volume[specs[0].slices] * w, rtol=1e-5
        )


# Denoise settings for the #1178 wiring tests: small, CPU-pinned, explicit — a
# per-tile pass costs a few ms and the numbers are machine-independent.
_D_H = 0.15
_D_BASE_PARAMS = {
    "patch_size": 3,
    "search_distance": 2,
    "backend": "pytorch",
    "device": "cpu",
    "use_2d": False,
}


def _d_params(volume: np.ndarray) -> dict:
    """PRODUCTION-shaped per-tile denoise params for ``volume``.

    ``norm_range`` is the whole-volume ``(min, max)``, which ``resolve_denoise_h``
    always records and ``assemble_fit_config`` always forwards: a fixed ``h`` is
    not scale-invariant, so every tile and the floor probe must normalize against
    the same range. Leaving it out measures a configuration production never runs
    (and can flip the sign of the measured denoise shift).
    """
    return {
        **_D_BASE_PARAMS,
        "norm_range": (float(volume.min()), float(volume.max())),
    }


def _skewed_pedestal_volume() -> np.ndarray:
    """96x96: right-SKEWED noise pedestal (~100) + a bright blob.

    Skew is what makes the denoised histogram mode differ measurably from the
    raw one — with symmetric noise the #1178 bug is invisible.
    """
    rng = np.random.default_rng(1178)
    yy, xx = np.meshgrid(np.arange(96), np.arange(96), indexing="ij")
    vol = 100.0 + rng.gamma(2.0, 3.0, size=(96, 96))
    vol = vol + 180.0 * np.exp(-(((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 10.0**2)))
    return vol.astype(np.float32)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestTiledFloorOnDenoisedBasis:
    """With ``--denoise`` the tiled level is a property of the DENOISED data.

    Every tile is denoised BEFORE the level is subtracted, so resolving the
    level on the raw volume removed a different pedestal than ``--tiling none``
    does on the same input (#1178). All three tiled doors — ``fit_tiled``,
    a standalone ``fit_tile``, and the ``--tile k/M`` / ``-j N`` worker — must
    reach the denoised-basis level.
    """

    @staticmethod
    def _levels(volume: np.ndarray) -> "tuple[float, float]":
        """``(raw_basis_level, denoised_basis_level)`` for this volume."""
        from luxar.gsplats.fitting.preprocessing import (
            resolve_volume_floor,
            resolve_volume_floor_denoised,
        )

        raw = resolve_volume_floor(volume, "auto", guard_numeric=True)
        denoised = resolve_volume_floor_denoised(
            volume,
            "auto",
            denoise_h=_D_H,
            denoise_params=_d_params(volume),
            guard_numeric=True,
        )
        assert raw is not None and denoised is not None
        # Guard the test's own premise: the two bases must actually differ here.
        assert abs(denoised - raw) > 1.0
        return float(raw), float(denoised)

    def test_fit_tiled_applies_the_denoised_basis_level(self, monkeypatch) -> None:
        import luxar.gsplats.fit_tiled_gsplats as ftg

        volume = _skewed_pedestal_volume()
        raw, denoised = self._levels(volume)

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        ftg.fit_tiled(
            volume,
            tile_size=48,
            overlap=16,
            floor="auto",
            verbose=False,
            _denoise_h=_D_H,
            _denoise_params=_d_params(volume),
        )

        assert len(records) > 1
        floors = [rec["result"].stats["floor"] for rec in records]
        assert all(f == pytest.approx(denoised) for f in floors)
        assert floors[0] != pytest.approx(raw, rel=1e-3)

    def test_standalone_fit_tile_applies_the_denoised_basis_level(
        self, monkeypatch
    ) -> None:
        import luxar.gsplats.fit_tiled_gsplats as ftg

        volume = _skewed_pedestal_volume()
        raw, denoised = self._levels(volume)

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        specs = compute_tile_specs(volume.shape, 48, 16)
        result = ftg.fit_tile(
            volume,
            specs[0],
            floor="auto",
            verbose=False,
            _denoise_h=_D_H,
            _denoise_params=_d_params(volume),
        )
        assert result.stats["floor"] == pytest.approx(denoised)
        assert result.stats["floor"] != pytest.approx(raw, rel=1e-3)

    def test_single_tile_worker_applies_the_denoised_basis_level(
        self, monkeypatch
    ) -> None:
        """The ``--tile k/M`` / ``-j N`` worker agrees with the sequential path."""
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.cli.gsplat_ops.fitting.fit_utils import fit_single_tile

        volume = _skewed_pedestal_volume()
        raw, denoised = self._levels(volume)

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        result = fit_single_tile(
            _single_tile_ctx("0/4"),
            volume,
            {
                "floor": "auto",
                "_denoise_h": _D_H,
                "_denoise_params": _d_params(volume),
            },
            None,
        )
        assert result.stats["floor"] == pytest.approx(denoised)
        assert result.stats["floor"] != pytest.approx(raw, rel=1e-3)
        # The tile really was denoised: the worker only PEEKED at the keys, so
        # they survived into `fit_tile` (which pops them).
        assert len(records) == 1
        w = cosine_window(compute_tile_specs(volume.shape, 48, 16)[0])
        raw_tile = (
            np.clip(
                volume[compute_tile_specs(volume.shape, 48, 16)[0].slices] - denoised,
                0.0,
                None,
            )
            * w
        )
        assert not np.allclose(records[0]["data"], raw_tile, atol=1e-3)

    def test_verbose_run_logs_the_denoised_basis_level(
        self, monkeypatch, capsys
    ) -> None:
        """A verbose caller is TOLD which basis the level came from.

        The correction is otherwise invisible: only the failure notes print
        unconditionally, so a run that corrected its level said nothing about it
        and a ``-j N`` fleet had no way to show its workers agreed.
        """
        import luxar.gsplats.fit_tiled_gsplats as ftg

        volume = _skewed_pedestal_volume()
        raw, denoised = self._levels(volume)

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        capsys.readouterr()  # drop the probe chatter from `_levels`
        ftg.fit_tiled(
            volume,
            tile_size=48,
            overlap=16,
            floor="auto",
            verbose=True,
            _denoise_h=_D_H,
            _denoise_params=_d_params(volume),
        )

        out = capsys.readouterr().out
        assert "DENOISED basis" in out
        assert f"{denoised:.6g}" in out
        assert f"raw {raw:.6g}" in out  # the level it was corrected FROM

    def test_denoise_off_keeps_the_raw_basis_level(self, monkeypatch) -> None:
        """No denoise keys -> exactly the pre-#1178 level, unchanged."""
        import luxar.gsplats.fit_tiled_gsplats as ftg

        volume = _skewed_pedestal_volume()
        raw, _denoised = self._levels(volume)

        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        ftg.fit_tiled(volume, tile_size=48, overlap=16, floor="auto", verbose=False)

        floors = [rec["result"].stats["floor"] for rec in records]
        assert floors and all(f == pytest.approx(raw) for f in floors)

    def test_denoise_off_logs_nothing_new_about_the_floor(
        self, monkeypatch, capsys
    ) -> None:
        """With ``--denoise`` off the log surface is byte-identical to before #1178.

        The floor resolution now goes through a wrapper that CAN log, so the
        ``verbose=`` forwarding is gated on denoising actually being active. Only
        the one summary line ``fit_tiled`` always printed may appear.
        """
        import luxar.gsplats.fit_tiled_gsplats as ftg

        volume = _skewed_pedestal_volume()
        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        capsys.readouterr()
        ftg.fit_tiled(volume, tile_size=48, overlap=16, floor="auto", verbose=True)

        out = capsys.readouterr().out
        assert "Resolved whole-volume background floor" not in out
        assert "DENOISED basis" not in out
        assert "Floor suppression: subtracting background level" in out

    @pytest.mark.parametrize("spec", ["110", 110.0, "none"])
    def test_an_absolute_floor_logs_nothing_new_even_with_denoise_on(
        self, monkeypatch, capsys, spec
    ) -> None:
        """A user absolute is not "resolved", so nothing new is announced for it.

        ``--floor 110`` reaches this function as the STRING ``"110"`` (the CLI
        passes the flag through verbatim), which an ``isinstance(str)`` gate would
        wave through — and the resulting line duplicates the summary line below
        it. The gate is on the VOLUME-DERIVED predicate for that reason.
        """
        import luxar.gsplats.fit_tiled_gsplats as ftg

        volume = _skewed_pedestal_volume()
        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        capsys.readouterr()
        ftg.fit_tiled(
            volume,
            tile_size=48,
            overlap=16,
            floor=spec,
            verbose=True,
            _denoise_h=_D_H,
            _denoise_params=_d_params(volume),
        )

        out = capsys.readouterr().out
        assert "Resolved whole-volume background floor" not in out
        assert "DENOISED basis" not in out

    def test_the_per_tile_door_never_announces_the_level(
        self, monkeypatch, capsys
    ) -> None:
        """``fit_tile`` resolves per TILE, so it must not log the level.

        Whatever it printed would be printed once per tile. The two callers that
        resolve once per RUN do the announcing — ``fit_tiled`` (above) and the
        standalone worker (``TestSingleTileWorkerFloor``) — and both hand
        ``fit_tile`` a concrete number afterwards.
        """
        import luxar.gsplats.fit_tiled_gsplats as ftg

        volume = _skewed_pedestal_volume()
        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))
        specs = compute_tile_specs(volume.shape, 48, 16)
        capsys.readouterr()
        ftg.fit_tile(
            volume,
            specs[0],
            floor="auto",
            verbose=True,
            _denoise_h=_D_H,
            _denoise_params=_d_params(volume),
        )

        out = capsys.readouterr().out
        assert "Resolved whole-volume background floor" not in out
        assert "DENOISED basis" not in out

    def test_the_standalone_worker_announces_only_with_denoise_on(
        self, monkeypatch, capsys
    ) -> None:
        """The ``--tile k/M`` worker's own new line, and only where it is new.

        With ``--denoise`` on it resolves a level nothing else in its log shows,
        so it says which basis that level came from. With ``--denoise`` off there
        is nothing new to report and its log stays as it was — which also matters
        because ``build_worker_cmd`` hardcodes ``--quiet`` and the parent discards
        a successful worker's stdout, so this line is for a human running one
        worker by hand, not for a fleet.
        """
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.cli.gsplat_ops.fitting.fit_utils import fit_single_tile

        volume = _skewed_pedestal_volume()
        records: list = []
        monkeypatch.setattr(ftg, "fit_gaussian_splats", _recording_stub(records))

        capsys.readouterr()
        fit_single_tile(
            _single_tile_ctx("0/4"),
            volume,
            {"floor": "auto", "verbose": True},
            None,
        )
        out = capsys.readouterr().out
        assert "DENOISED basis" not in out
        # Both lines, not just the denoised one: with denoise off the wrapper
        # delegates to `resolve_volume_floor`, so an ungated `verbose=` shows up as
        # the plain "Resolved whole-volume background floor" line instead.
        assert "Resolved whole-volume background floor" not in out

        fit_single_tile(
            _single_tile_ctx("0/4"),
            volume,
            {
                "floor": "auto",
                "verbose": True,
                "_denoise_h": _D_H,
                "_denoise_params": _d_params(volume),
            },
            None,
        )
        assert "DENOISED basis" in capsys.readouterr().out


class TestGridBspTree:
    """Split planes over a uniform tile grid (`grid_bsp_tree`).

    Approximate by nature: apodized tiles keep their overlap band, so parts
    genuinely intersect and no exact order exists. What must hold is that every
    tile is named exactly once and every cut lands INSIDE the band it separates —
    that is what bounds the misordering to the band instead of letting whole
    tiles swap.
    """

    @staticmethod
    def _labels(node: dict) -> list[int]:
        if "part" in node:
            return [node["part"]]
        return TestGridBspTree._labels(node["left"]) + TestGridBspTree._labels(
            node["right"]
        )

    def test_every_tile_is_named_exactly_once(self) -> None:
        from luxar.gsplats.tiling import grid_bsp_tree

        specs = compute_tile_specs((100, 80, 80), 40, 8)
        tree = grid_bsp_tree(specs)
        assert tree is not None
        assert sorted(self._labels(tree)) == list(range(len(specs)))

    def test_labels_are_flat_tile_indices_not_dfs_order(self) -> None:
        """A median split does not visit tiles in row-major order, so the labels
        must be explicit — DFS-implied numbering would silently mislabel parts."""
        from luxar.gsplats.tiling import grid_bsp_tree

        specs = compute_tile_specs((100, 80, 80), 40, 8)
        tree = grid_bsp_tree(specs)
        assert tree is not None
        assert self._labels(tree) != list(range(len(specs)))

    def test_each_cut_lies_within_the_band_it_separates(self) -> None:
        from luxar.gsplats.tiling import grid_bsp_tree

        specs = compute_tile_specs((100, 80, 80), 40, 8)
        tree = grid_bsp_tree(specs)
        assert tree is not None
        by_index = {s.index: s for s in specs}

        def check(node: dict) -> None:
            if "part" in node:
                return
            axis, split = node["axis"], node["split"]
            lows = [by_index[i] for i in self._labels(node["left"])]
            highs = [by_index[i] for i in self._labels(node["right"])]
            # Every low tile starts before the cut and every high tile ends
            # after it, and the cut sits inside their shared band: it is at or
            # after the highest low-side start and at or before the lowest
            # high-side end.
            assert max(s.origin[axis] for s in lows) <= split
            assert min(s.origin[axis] + s.shape[axis] for s in highs) >= split
            assert min(s.origin[axis] for s in highs) <= split
            assert max(s.origin[axis] + s.shape[axis] for s in lows) >= split
            check(node["left"])
            check(node["right"])

        check(tree)

    def test_a_single_tile_is_a_bare_leaf(self) -> None:
        from luxar.gsplats.tiling import grid_bsp_tree

        assert grid_bsp_tree(compute_tile_specs((30, 30, 30), 64, 8)) == {"part": 0}

    def test_no_specs_yields_no_tree(self) -> None:
        from luxar.gsplats.tiling import grid_bsp_tree

        assert grid_bsp_tree([]) is None

    def test_a_grid_split_beyond_the_third_axis_is_refused(self) -> None:
        """Stacked grid axes past the third are never displayed."""
        from luxar.gsplats.tiling import grid_bsp_tree

        specs = compute_tile_specs((16, 16, 16, 64), (16, 16, 16, 16), 0)
        assert len({s.grid_index[3] for s in specs}) > 1
        assert grid_bsp_tree(specs) is None


class TestGridBspTreeScale:
    """The ``scale`` parameter: the grid's frame need not be the splats' frame.

    Under ``--downscale`` the parallel tiled path computes its grid on the
    POST-downscale shape (so parent and workers agree on the tile count) while
    each worker rescales its splats back to full resolution. ``scale`` is what
    lifts the planes into that same full-resolution frame (issue #1587).
    """

    @staticmethod
    def _splits(node: dict) -> list[tuple[int, float]]:
        """Every internal node's ``(axis, split)`` in a fixed traversal order."""
        if "part" in node:
            return []
        return (
            [(int(node["axis"]), float(node["split"]))]
            + TestGridBspTreeScale._splits(node["left"])
            + TestGridBspTreeScale._splits(node["right"])
        )

    def test_scale_none_is_identical_to_no_scale_at_all(self) -> None:
        from luxar.gsplats.tiling import grid_bsp_tree

        specs = compute_tile_specs((100, 80, 80), 40, 8)
        assert grid_bsp_tree(specs, scale=None) == grid_bsp_tree(specs)
        # An all-ones scale is likewise a no-op.
        assert grid_bsp_tree(specs, scale=(1, 1, 1)) == grid_bsp_tree(specs)

    def test_a_uniform_scale_multiplies_every_split(self) -> None:
        from luxar.gsplats.tiling import grid_bsp_tree

        specs = compute_tile_specs((25, 20, 20), 10, 2)
        plain = grid_bsp_tree(specs)
        scaled = grid_bsp_tree(specs, scale=(4, 4, 4))
        assert plain is not None and scaled is not None
        # Same structure and labels...
        assert self._labels_of(plain) == self._labels_of(scaled)
        # ...but every plane four times further out.
        for (ax_a, s_a), (ax_b, s_b) in zip(self._splits(plain), self._splits(scaled)):
            assert ax_a == ax_b
            assert s_b == pytest.approx(4.0 * s_a)

    def test_per_axis_factors_scale_their_own_axis_only(self) -> None:
        from luxar.gsplats.tiling import grid_bsp_tree

        factors = (1, 2, 4)
        specs = compute_tile_specs((25, 20, 20), 10, 2)
        plain = grid_bsp_tree(specs)
        scaled = grid_bsp_tree(specs, scale=factors)
        assert plain is not None and scaled is not None
        seen = set()
        for (ax_a, s_a), (ax_b, s_b) in zip(self._splits(plain), self._splits(scaled)):
            assert ax_a == ax_b
            assert s_b == pytest.approx(factors[ax_a] * s_a)
            seen.add(ax_a)
        # The non-uniformity is actually exercised: more than one axis splits,
        # and at least one of them has a factor != 1.
        assert len(seen) > 1
        assert any(factors[ax] != 1 for ax in seen)

    def test_a_wrong_length_scale_is_refused(self) -> None:
        from luxar.gsplats.tiling import grid_bsp_tree

        specs = compute_tile_specs((25, 20, 20), 10, 2)
        with pytest.raises(ValueError, match="scale has length 2"):
            grid_bsp_tree(specs, scale=(4, 4))

    @pytest.mark.parametrize(
        "bad",
        [
            (0, 1, 1),
            (1, -2, 1),
            (1, 1, 0.0),
            (1, float("nan"), 1),
            (float("inf"), 1, 1),
        ],
    )
    def test_a_non_positive_scale_is_refused(self, bad: tuple[float, ...]) -> None:
        """A 0 collapses every plane onto the origin, a negative one mirrors the
        frame, and a ``NaN``/``inf`` (which a bare ``<= 0`` test lets through)
        puts a non-number in the tree — all three would emit a tree that cannot
        order the parts."""
        from luxar.gsplats.tiling import grid_bsp_tree

        specs = compute_tile_specs((25, 20, 20), 10, 2)
        with pytest.raises(ValueError, match="strictly positive"):
            grid_bsp_tree(specs, scale=bad)

    @staticmethod
    def _labels_of(node: dict) -> list[int]:
        if "part" in node:
            return [int(node["part"])]
        return TestGridBspTreeScale._labels_of(
            node["left"]
        ) + TestGridBspTreeScale._labels_of(node["right"])


class TestResolveGridScale:
    """``resolve_grid_scale`` composes the two terms that move the splats.

    The tile grid is always in voxels of the array that was tiled; ``--downscale``
    and a real-space ``voxel_size`` each move the SPLATS relative to it, and a
    downscaled parallel fit with a ``voxel_size`` applies both (#1587).
    """

    def test_no_terms_means_no_scale(self) -> None:
        from luxar.gsplats.tiling import resolve_grid_scale

        assert resolve_grid_scale(3) is None
        # Explicit no-ops are still no-ops (so the caller passes None onward).
        assert resolve_grid_scale(3, downscale_factors=(1, 1, 1)) is None
        assert resolve_grid_scale(3, voxel_size=1.0) is None

    def test_downscale_only(self) -> None:
        from luxar.gsplats.tiling import resolve_grid_scale

        assert resolve_grid_scale(3, downscale_factors=(1, 2, 4)) == (1.0, 2.0, 4.0)

    def test_voxel_size_only(self) -> None:
        from luxar.gsplats.tiling import resolve_grid_scale

        assert resolve_grid_scale(3, voxel_size=(4.0, 1.0, 1.0)) == (4.0, 1.0, 1.0)
        # A scalar broadcasts, like everywhere else voxel_size is accepted.
        assert resolve_grid_scale(3, voxel_size=2.5) == (2.5, 2.5, 2.5)

    def test_the_two_terms_multiply(self) -> None:
        """The parallel worker emits ``voxel_size * f * origin``, so only the
        PRODUCT puts the planes back on the parts."""
        from luxar.gsplats.tiling import resolve_grid_scale

        assert resolve_grid_scale(
            3, downscale_factors=(2, 2, 2), voxel_size=(4.0, 1.0, 1.0)
        ) == (8.0, 2.0, 2.0)

    def test_voxel_output_space_drops_the_voxel_size_term(self) -> None:
        """With ``output_space="voxel"`` the centers stay in voxels, so applying
        the spacing would move the planes OFF the parts."""
        from luxar.gsplats.tiling import resolve_grid_scale

        assert (
            resolve_grid_scale(3, voxel_size=(4.0, 1.0, 1.0), output_space="voxel")
            is None
        )
        # ...but the downscale term is unconditional — the worker rescales
        # its centers whatever space they are expressed in.
        assert resolve_grid_scale(
            3,
            downscale_factors=(2, 2, 2),
            voxel_size=(4.0, 1.0, 1.0),
            output_space="voxel",
        ) == (2.0, 2.0, 2.0)

    @pytest.mark.parametrize("bad", ["physical", "Real", "", None])
    def test_out_of_vocabulary_output_space_raises(self, bad: "Any") -> None:
        """A typo (or a YAML ``output_space: null``) must NOT be read as "voxel".

        Silently dropping the spacing term is exactly the #1587 mismatch this
        function exists to close, and it would be invisible: the tree still
        builds, just in the wrong frame.
        """
        from luxar.gsplats.tiling import resolve_grid_scale

        with pytest.raises(ValueError, match="output_space"):
            resolve_grid_scale(3, voxel_size=(4.0, 1.0, 1.0), output_space=bad)
        # Refused even with nothing for the term to apply to, so a caller
        # cannot discover the typo only once a spacing is configured.
        with pytest.raises(ValueError, match="output_space"):
            resolve_grid_scale(3, output_space=bad)

    def test_wrong_length_voxel_size_is_refused_by_name(self) -> None:
        """Matching ``fitting.validation``: a length-1 spacing on a 3D grid is a
        mistake, not something to broadcast, and the message must say so."""
        from luxar.gsplats.tiling import resolve_grid_scale

        for bad in ((0.5,), (1.0, 1.0), (1.0, 1.0, 1.0, 1.0)):
            with pytest.raises(ValueError, match="voxel_size must have length 3"):
                resolve_grid_scale(3, voxel_size=bad)
        # A genuine scalar still broadcasts (the documented spelling).
        assert resolve_grid_scale(3, voxel_size=0.5) == (0.5, 0.5, 0.5)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"voxel_size": float("nan")},
            {"voxel_size": (4.0, float("inf"), 1.0)},
            {"voxel_size": (4.0, 0.0, 1.0)},
            {"voxel_size": -1.0},
            {"downscale_factors": (2, 0, 2)},
            {"downscale_factors": (2, 2, 2), "voxel_size": float("nan")},
        ],
    )
    def test_a_frame_the_tree_cannot_state_is_refused(self, kwargs: "Any") -> None:
        """``voxel_size`` is not validated upstream for finiteness at all.

        ``fitting.validation`` tests ``voxel_size <= 0``, which is False for
        ``NaN``. An unrefused ``NaN``/``0`` would become a ``NaN``/collapsed
        split plane in the serialized tree — a silently unorderable partition —
        or, on the batch path, get recorded on the manifest for a merge days
        later to trip over. Refused where the offending term can still be named.

        The ``downscale_factors`` cases are defence in depth for a direct
        caller: both in-tree producers now normalise first (the single-fit
        ``-j N`` path through ``normalize_downscale``, and the batch planner,
        which since #1624 validates the key itself and does not pass this term at
        all). This is a public entry point, so it still refuses rather than
        trusting them.
        """
        from luxar.gsplats.tiling import resolve_grid_scale

        with pytest.raises(ValueError, match="finite and strictly positive"):
            resolve_grid_scale(3, **kwargs)


def _core_region(
    spec: "TileSpec", factors: tuple[float, ...], n: int, seed: int
) -> "GSplatData":
    """``n`` splats inside ``spec``'s CORE, in the SPLATS' own frame.

    ``factors`` maps the spec's voxel frame onto that frame — the same product
    of downscale and voxel size :func:`resolve_grid_scale` builds.

    The core is the tile minus its overlap band, so the synthetic parts are
    genuinely disjoint and an exact separating plane must exist — which is what
    lets a test assert separation rather than the weaker "cut lands in the band"
    property that overlapping tiles are limited to.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    lo = np.array(
        [
            (spec.origin[d] + spec.overlap_low[d]) * factors[d]
            for d in range(len(factors))
        ],
        dtype=float,
    )
    hi = np.array(
        [
            (spec.origin[d] + spec.shape[d] - spec.overlap_high[d]) * factors[d]
            for d in range(len(factors))
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0  # packed-3D diagonal; must be strictly positive
    return GSplatData(
        centers=rng.uniform(lo, hi, size=(n, 3)).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )


# Torch-gated for an IMPORT reason, not a fitting one: nothing below fits
# anything, but `merge_tile_results` lives in `fit_tiled_gsplats`, which imports
# `fit_gsplats`, which imports torch at module scope. Verified by running the
# class with `import torch` blocked (ModuleNotFoundError at collection).
@pytest.mark.skipif(not HAS_TORCH, reason="fit_tiled_gsplats imports torch")
class TestDownscaledPartitionPlanes:
    """``merge_tile_results`` must place the split planes in the SPLATS' frame.

    The parallel tiled path hands in a grid computed on the downscaled shape
    while its workers write splats already rescaled to full resolution. Without
    ``grid_scale`` every plane is a factor too small and no longer lies between
    the parts it separates (#1587).
    """

    FACTORS = (4.0, 4.0, 4.0)
    GRID_SHAPE = (24, 24, 24)  # the post-downscale shape the parent computes
    TILE_SIZE = 16
    OVERLAP = 4

    def _specs(self) -> "list[TileSpec]":
        return compute_tile_specs(self.GRID_SHAPE, self.TILE_SIZE, self.OVERLAP)

    def _merge(
        self,
        grid_scale: "tuple[float, ...] | None",
        splat_factors: "tuple[float, ...] | None" = None,
    ) -> "Any":
        """Merge synthetic tiles placed at ``splat_factors`` (default: FACTORS)
        with the planes scaled by ``grid_scale``."""
        from luxar.gsplats.fit_tiled_gsplats import merge_tile_results

        specs = self._specs()
        assert len(specs) > 1
        factors = self.FACTORS if splat_factors is None else splat_factors
        results = [_core_region(s, factors, 6, 100 + s.index) for s in specs]
        return merge_tile_results(
            results,
            volume_shape=self.GRID_SHAPE,
            tile_size=self.TILE_SIZE,
            overlap=self.OVERLAP,
            num_tiles=len(specs),
            progressive=False,
            cull_retention=None,
            elapsed=0.0,
            verbose=False,
            partition=True,
            grid_scale=grid_scale,
        )

    @staticmethod
    def _part_centers(node) -> list:
        return [child.additive_sublods[0].centers for child in node.children]

    @staticmethod
    def _leaf_parts(sub: dict) -> list[int]:
        if "part" in sub:
            return [int(sub["part"])]
        return TestDownscaledPartitionPlanes._leaf_parts(
            sub["left"]
        ) + TestDownscaledPartitionPlanes._leaf_parts(sub["right"])

    @classmethod
    def _separation_failures(cls, tree: dict, centers: list) -> list[str]:
        """Internal nodes whose plane does not separate their two subtrees."""
        bad: list[str] = []

        def walk(node: dict) -> None:
            if "part" in node:
                return
            axis, split = int(node["axis"]), float(node["split"])
            left = np.concatenate(
                [centers[i][:, axis] for i in cls._leaf_parts(node["left"])]
            )
            right = np.concatenate(
                [centers[i][:, axis] for i in cls._leaf_parts(node["right"])]
            )
            if left.max() > split or right.min() < split:
                bad.append(
                    f"axis={axis} split={split} left.max={left.max()} "
                    f"right.min={right.min()}"
                )
            walk(node["left"])
            walk(node["right"])

        walk(tree)
        return bad

    def test_planes_separate_the_parts_when_factors_are_threaded(self) -> None:
        node = self._merge(self.FACTORS)
        assert node.bsp_tree is not None
        centers = self._part_centers(node)
        assert len(centers) == len(self._specs())
        assert self._separation_failures(node.bsp_tree, centers) == []

    def test_a_downscale_and_a_voxel_size_compose(self) -> None:
        """A downscaled parallel fit with a ``voxel_size`` applies BOTH factors
        to its splats, so only their product separates the parts (#1587)."""
        from luxar.gsplats.tiling import resolve_grid_scale

        downscale = (2, 2, 2)
        voxel_size = (4.0, 1.0, 1.0)
        combined = resolve_grid_scale(
            3, downscale_factors=downscale, voxel_size=voxel_size
        )
        assert combined == (8.0, 2.0, 2.0)

        good = self._merge(combined, splat_factors=combined)
        assert good.bsp_tree is not None
        assert self._separation_failures(good.bsp_tree, self._part_centers(good)) == []

        # Either term ALONE leaves every plane in the wrong frame.
        for partial in (tuple(float(f) for f in downscale), voxel_size):
            half = self._merge(partial, splat_factors=combined)
            assert half.bsp_tree is not None
            failures = self._separation_failures(
                half.bsp_tree, self._part_centers(half)
            )
            assert failures, f"scale={partial} should not separate the parts"

    def test_omitting_the_factors_misplaces_every_plane(self) -> None:
        """The regression guard: the un-threaded tree is in the wrong frame."""
        good = self._merge(self.FACTORS)
        bad = self._merge(None)
        assert good.bsp_tree is not None and bad.bsp_tree is not None

        # Every plane is short by the downscale factor...
        good_splits = TestGridBspTreeScale._splits(good.bsp_tree)
        bad_splits = TestGridBspTreeScale._splits(bad.bsp_tree)
        assert len(bad_splits) == len(good_splits) > 0
        for (ax_g, s_g), (ax_b, s_b) in zip(good_splits, bad_splits):
            assert ax_g == ax_b
            assert s_g == pytest.approx(self.FACTORS[ax_g] * s_b)

        # ...and consequently separates nothing.
        failures = self._separation_failures(bad.bsp_tree, self._part_centers(bad))
        assert len(failures) == len(bad_splits), failures


@pytest.mark.skipif(not HAS_TORCH, reason="fit_tiled_parallel imports torch")
class TestParallelTiledGridScaleThreading:
    """``fit_tiled_parallel`` must FORWARD its ``grid_scale`` to the merge.

    The dispatcher resolves the factor and ``merge_tile_results`` consumes it,
    but the hop between them is a plain keyword: dropping it reintroduces #1587
    one frame later, with both endpoints still passing their own tests. This
    drives the real orchestrator through its ``worker_cmd_builder`` seam, with
    the subprocess launch replaced by an in-process fake that writes each tile's
    store directly — no torch fitting, no child processes, so it runs in the
    default (``-m "not slow"``) gate.
    """

    GRID_SHAPE = (24, 12, 12)  # 2 tiles on axis 0 -> exactly one split plane
    TILE_SIZE = 16
    OVERLAP = 4
    # What a `--downscale 2` fit with `voxel_size=(4, 1, 1)` puts on the splats.
    SCALE = (8.0, 2.0, 2.0)

    def _specs(self) -> "list[TileSpec]":
        return compute_tile_specs(self.GRID_SHAPE, self.TILE_SIZE, self.OVERLAP)

    def _run(
        self,
        tmp_path: "Path",
        monkeypatch: pytest.MonkeyPatch,
        grid_scale: "tuple[float, ...] | None",
    ) -> "Any":
        import subprocess

        from luxar.gsplats.fit_tiled_parallel import fit_tiled_parallel

        specs = self._specs()
        assert len(specs) == 2

        def _fake_run(cmd: "list[str]", **kwargs: "Any") -> "Any":
            # Stand in for the `fit --tile i/M` worker: write the tile's splats
            # in the SPLATS' frame (already rescaled / in physical units), which
            # is exactly what the real worker does before exiting 0.
            index, out_path = int(cmd[1]), cmd[2]
            _core_region(specs[index], self.SCALE, 6, 100 + index).save(out_path)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(
            "luxar.gsplats.fit_tiled_parallel.subprocess.run", _fake_run
        )
        return fit_tiled_parallel(
            num_tiles=len(specs),
            jobs=1,
            tmp_dir=tmp_path / "tiles",
            worker_cmd_builder=lambda i, m, p: ["worker", str(i), str(p)],
            volume_shape=self.GRID_SHAPE,
            tile_size=self.TILE_SIZE,
            overlap=self.OVERLAP,
            progressive=False,
            cull_retention=None,
            verbose=False,
            partition=True,
            grid_scale=grid_scale,
        )

    def test_the_scale_reaches_the_split_planes(
        self, tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node = self._run(tmp_path, monkeypatch, self.SCALE)
        assert node.bsp_tree is not None
        centers = TestDownscaledPartitionPlanes._part_centers(node)
        assert (
            TestDownscaledPartitionPlanes._separation_failures(node.bsp_tree, centers)
            == []
        )

    def test_without_the_scale_the_planes_separate_nothing(
        self, tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-vacuity control: the assertion above is only a guard because the
        un-scaled tree genuinely fails it."""
        node = self._run(tmp_path, monkeypatch, None)
        assert node.bsp_tree is not None
        centers = TestDownscaledPartitionPlanes._part_centers(node)
        assert TestDownscaledPartitionPlanes._separation_failures(
            node.bsp_tree, centers
        )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestVoxelSizePartitionPlanes:
    """A real ``fit_tiled`` partition with a ``voxel_size``: planes follow the splats.

    The second, independent term of the #1587 frame mismatch, and the one that
    needs no ``--downscale`` at all: the tile grid is in VOXELS while
    ``output_space="real"`` (the CLI default) offsets every tile's splats by
    ``origin * voxel_size``. This runs the SEQUENTIAL path end to end, so it
    covers the resolution inside ``fit_tiled`` and not just the merge keyword.
    """

    VOXEL_SIZE = (4.0, 1.0, 1.0)
    SHAPE = (24, 12, 12)
    TILE_SIZE = 16
    OVERLAP = 4

    @staticmethod
    @pytest.fixture(scope="class")
    def fits() -> "dict[str, Any]":
        """The two fits this class compares, run once (each is a real fit).

        A 2-tile grid on axis 0 — the axis ``VOXEL_SIZE`` scales — keeps the
        cost of a real fit inside the default gate. The unscaled reference is
        NOT a third fit: the voxel-frame planes are a pure function of the tile
        grid, so :func:`grid_bsp_tree` yields them for free (see
        ``_voxel_frame_splits``).
        """
        from luxar.gsplats.fit_tiled_gsplats import fit_tiled

        cls = TestVoxelSizePartitionPlanes
        rng = np.random.RandomState(3)
        volume = rng.random_sample(cls.SHAPE).astype(np.float32)

        def run(**kwargs: "Any") -> "Any":
            return fit_tiled(
                volume,
                tile_size=cls.TILE_SIZE,
                overlap=cls.OVERLAP,
                partition=True,
                seeds=8,
                n_iters=10,
                floor="none",
                cull_retention=None,
                verbose=False,
                **kwargs,
            )

        return {
            "real": run(voxel_size=cls.VOXEL_SIZE, output_space="real"),
            "voxel": run(voxel_size=cls.VOXEL_SIZE, output_space="voxel"),
        }

    @classmethod
    def _voxel_frame_splits(cls) -> list:
        """The planes an unscaled fit of the same grid writes — no fit needed."""
        from luxar.gsplats.tiling import grid_bsp_tree

        tree = grid_bsp_tree(compute_tile_specs(cls.SHAPE, cls.TILE_SIZE, cls.OVERLAP))
        assert tree is not None
        return TestGridBspTreeScale._splits(tree)

    @staticmethod
    def _plane_is_between_its_sides(node: "Any") -> "list[str]":
        """Internal nodes whose plane does NOT sit between the two sides' splats.

        Weaker than exact separation — apodized tiles overlap, so a few splats
        legitimately cross — but it is exactly what the frame mismatch breaks:
        a plane stated in voxels while the splats are physical lands below the
        whole object on the scaled axis.
        """
        centers = TestDownscaledPartitionPlanes._part_centers(node)
        bad: "list[str]" = []

        def walk(sub: dict) -> None:
            if "part" in sub:
                return
            axis, split = int(sub["axis"]), float(sub["split"])
            leaves = TestDownscaledPartitionPlanes._leaf_parts
            left = np.concatenate([centers[i][:, axis] for i in leaves(sub["left"])])
            right = np.concatenate([centers[i][:, axis] for i in leaves(sub["right"])])
            if not (float(left.mean()) <= split <= float(right.mean())):
                bad.append(
                    f"axis={axis} split={split} left.mean={left.mean()} "
                    f"right.mean={right.mean()}"
                )
            walk(sub["left"])
            walk(sub["right"])

        walk(node.bsp_tree)
        return bad

    def test_planes_are_stated_in_physical_units(self, fits: "dict[str, Any]") -> None:
        node = fits["real"]
        assert node.bsp_tree is not None
        # The scaled axis really is scaled: the axis-0 planes are ~4x the
        # voxel-frame ones the same grid would give.
        scaled_splits = TestGridBspTreeScale._splits(node.bsp_tree)
        plain_splits = self._voxel_frame_splits()
        assert len(scaled_splits) == len(plain_splits) > 0
        for (ax_s, s_s), (ax_p, s_p) in zip(scaled_splits, plain_splits):
            assert ax_s == ax_p
            assert s_s == pytest.approx(self.VOXEL_SIZE[ax_s] * s_p)
        # ...and every plane lands between the parts it separates.
        assert self._plane_is_between_its_sides(node) == []

    def test_voxel_output_space_keeps_the_planes_in_voxels(
        self, fits: "dict[str, Any]"
    ) -> None:
        """``output_space="voxel"`` leaves the centers in voxels, so the planes
        must NOT pick up the spacing."""
        voxelled = fits["voxel"]
        assert (
            TestGridBspTreeScale._splits(voxelled.bsp_tree)
            == self._voxel_frame_splits()
        )
        assert self._plane_is_between_its_sides(voxelled) == []
