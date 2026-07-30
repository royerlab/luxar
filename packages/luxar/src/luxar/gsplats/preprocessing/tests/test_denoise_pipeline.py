"""Tests for denoise_pipeline module."""

import numpy as np
import pytest

from luxar.gsplats.preprocessing.denoise_pipeline import (
    _pick_sample_timepoints,
    denoise_volume_array,
    denormalize_volume,
    normalize_volume,
)


class TestNormalization:
    def test_roundtrip(self):
        vol = np.random.rand(10, 20, 20).astype(np.float32) * 1000 + 50
        norm, vmin, vmax = normalize_volume(vol)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0
        recovered = denormalize_volume(norm, vmin, vmax)
        np.testing.assert_allclose(recovered, vol, atol=1e-3)

    def test_constant_volume(self):
        vol = np.full((5, 10, 10), 42.0, dtype=np.float32)
        norm, vmin, vmax = normalize_volume(vol)
        assert np.all(norm == 0.0)
        recovered = denormalize_volume(norm, vmin, vmax)
        np.testing.assert_allclose(recovered, 42.0)

    def test_already_01(self):
        vol = np.random.rand(5, 10, 10).astype(np.float32)
        norm, vmin, vmax = normalize_volume(vol)
        # Roundtrip should recover original, not necessarily identity
        recovered = denormalize_volume(norm, vmin, vmax)
        np.testing.assert_allclose(recovered, vol, atol=1e-3)

    def test_negative_values(self):
        vol = np.random.rand(5, 10, 10).astype(np.float32) * 200 - 100
        norm, vmin, vmax = normalize_volume(vol)
        assert norm.min() >= -1e-6
        assert norm.max() <= 1.0 + 1e-6

    def test_explicit_value_range(self):
        # A sub-array normalized against a wider global range must NOT
        # occupy the full [0,1] — it should reflect its position in the
        # global range and return the supplied vmin/vmax.
        sub = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        norm, vmin, vmax = normalize_volume(sub, value_range=(0.0, 10.0))
        assert vmin == 0.0
        assert vmax == 10.0
        np.testing.assert_allclose(norm, sub / 10.0)
        # It does not span [0,1] (its own extent would have).
        assert norm.max() < 1.0
        assert norm.min() > 0.0

    def test_value_range_default_unchanged(self):
        # value_range=None must use the volume's OWN min/max...
        vol = np.random.rand(5, 8, 8).astype(np.float32) * 50 + 3
        _norm, vmin, vmax = normalize_volume(vol, value_range=None)
        assert (vmin, vmax) == (float(vol.min()), float(vol.max()))
        # ...whereas an explicit range overrides it (and differs here).
        _norm2, rmin, rmax = normalize_volume(vol, value_range=(0.0, 100.0))
        assert (rmin, rmax) == (0.0, 100.0)
        assert (rmin, rmax) != (vmin, vmax)


class TestSharedRangeScalePrecondition:
    """Check the shared range preserves the scale relationship a fixed ``h``
    relies on.

    This is the *precondition* for consistent tiled denoising (not the denoise
    step itself): a fixed NLM ``h`` is not scale-invariant, so tiles must share
    one intensity scale for it to mean the same strength everywhere. This test
    verifies (torch-free) that normalizing against the WHOLE-volume range keeps
    tiles on one shared scale, whereas per-tile normalization collapses each to
    [0,1] and destroys that relationship.
    """

    def test_shared_range_preserves_amplitude_ratio(self):
        # Two "tiles" drawn from a common global range [0, 100]:
        #   dim tile: amplitudes ~10 (low dynamic range)
        #   bright tile: amplitudes ~100 (10x brighter)
        global_range = (0.0, 100.0)
        dim_tile = np.array([[1.0, 5.0], [10.0, 3.0]], dtype=np.float32)
        bright_tile = 10.0 * dim_tile  # same shape, 10x amplitudes

        # Shared (global) normalization: the 10x ratio is preserved.
        dim_norm, _, _ = normalize_volume(dim_tile, value_range=global_range)
        bright_norm, _, _ = normalize_volume(bright_tile, value_range=global_range)
        ratio = bright_norm / dim_norm
        np.testing.assert_allclose(ratio, 10.0, rtol=1e-5)

        # Per-tile normalization (value_range=None) collapses BOTH to the
        # SAME [0,1] array (bright is just 10x dim), destroying the amplitude
        # relationship the shared range preserved — which is exactly why a
        # fixed h drifts tile-to-tile.
        dim_self, _, _ = normalize_volume(dim_tile)
        bright_self, _, _ = normalize_volume(bright_tile)
        np.testing.assert_allclose(dim_self, bright_self, rtol=1e-5)
        assert not np.allclose(dim_norm, dim_self)


def _make_denoise_ctx(volume, denoise_h):
    """Build a minimal valid FitPipelineCtx with denoise enabled."""
    from pathlib import Path

    from luxar.cli.gsplat_ops.fitting_fit_utils import FitPipelineCtx

    return FitPipelineCtx(
        input_path=Path("in.zarr"),
        output_path=Path("out.gsplats.zarr"),
        seeds=None,
        iters=None,
        device=None,
        preset=None,
        loss=None,
        config=None,
        compress=None,
        channel=None,
        timepoint=None,
        array_key=None,
        axes=None,
        lr=None,
        floor=None,
        seed_method=None,
        verbose=False,
        downscale=None,
        resolved_tiling="uniform",
        flat=False,
        tile_size=256,
        tile_overlap=32,
        tile=None,
        jobs="1",
        keep_tiles=False,
        allow_empty_tile=False,
        recipe=None,
        recipe_n_lods=None,
        recipe_additive_method=None,
        recipe_breakpoints=None,
        recipe_target_ms=None,
        recipe_bandwidth_mbps=None,
        recipe_bytes_per_splat=None,
        recipe_compression_factor=None,
        recipe_levels=None,
        recipe_substitutive_method=None,
        recipe_coarsen_dims=None,
        cal=None,
        k_star_ref=None,
        n_features_ref=None,
        feature_threshold=None,
        feature_metric=None,
        target_features=None,
        plan_only=False,
        plan_box=None,
        progressive=False,
        max_splats_per_pass=50000,
        psnr_patience=0.1,
        max_passes=None,
        cull_retention=None,
        denoise=True,
        denoise_h=denoise_h,
        denoise_2d=False,
        denoise_patch_size=3,
        denoise_search_distance=5,
        denoise_backend="auto",
    )


class TestDenoiseConfigInjection:
    def test_manual_h_records_global_range_and_injects_it(self):
        from luxar.cli.gsplat_ops.fitting_fit_utils import (
            assemble_fit_config,
            resolve_denoise_h,
        )

        rng = np.random.RandomState(0)
        volume = (rng.rand(6, 16, 16) * 500 + 20).astype(np.float32)
        ctx = _make_denoise_ctx(volume, denoise_h=0.05)

        effective_h = resolve_denoise_h(ctx, volume)
        assert effective_h == 0.05
        expected_range = (float(volume.min()), float(volume.max()))
        assert ctx.denoise_norm_range == expected_range

        ctx.denoise_effective_h = effective_h
        fit_config, _seeds, _ds = assemble_fit_config(ctx, is_tiled=True)
        assert fit_config["_denoise_h"] == 0.05
        assert fit_config["_denoise_params"]["norm_range"] == expected_range

    def test_auto_calibration_records_global_range(self, monkeypatch):
        # The default --denoise flow (no --denoise-h) takes the auto branch:
        # calibrate_nlm_h is heavy external compute, so patch it and assert the
        # WHOLE-volume range is still captured for per-tile normalization.
        import luxar.gsplats.preprocessing as _preproc
        from luxar.cli.gsplat_ops.fitting_fit_utils import resolve_denoise_h

        rng = np.random.RandomState(1)
        volume = (rng.rand(6, 16, 16) * 300 + 10).astype(np.float32)

        def _fake_calibrate(*_args, **_kwargs):
            return 0.037

        # resolve_denoise_h does `from luxar.gsplats.preprocessing import
        # calibrate_nlm_h`, so patch the attribute on that package module.
        monkeypatch.setattr(_preproc, "calibrate_nlm_h", _fake_calibrate)

        ctx = _make_denoise_ctx(volume, denoise_h=None)
        effective_h = resolve_denoise_h(ctx, volume)
        assert effective_h == 0.037
        assert ctx.denoise_norm_range == (float(volume.min()), float(volume.max()))

    def test_no_denoise_returns_none_range(self):
        from luxar.cli.gsplat_ops.fitting_fit_utils import resolve_denoise_h

        volume = np.zeros((4, 8, 8), dtype=np.float32)
        ctx = _make_denoise_ctx(volume, denoise_h=0.05)
        ctx.denoise = False
        assert resolve_denoise_h(ctx, volume) is None
        assert ctx.denoise_norm_range is None


class TestPickSampleTimepoints:
    def test_more_samples_than_available(self):
        result = _pick_sample_timepoints(3, 10)
        assert len(result) == 3

    def test_equidistant(self):
        result = _pick_sample_timepoints(100, 5)
        assert len(result) == 5
        assert result[0] == 0
        assert result[-1] == 99

    def test_single_timepoint(self):
        result = _pick_sample_timepoints(1, 5)
        assert len(result) == 1
        assert result[0] == 0

    def test_with_indices(self):
        result = _pick_sample_timepoints(100, 3, timepoint_indices=[10, 20, 30, 40, 50])
        assert len(result) == 3
        assert all(t in [10, 20, 30, 40, 50] for t in result)

    def test_five_from_ten(self):
        result = _pick_sample_timepoints(10, 5)
        assert len(result) == 5
        # Should be roughly equidistant
        assert result[0] == 0
        assert result[-1] == 9


class TestDenoiseVolumeArray:
    @pytest.fixture
    def noisy_volume(self):
        rng = np.random.RandomState(42)
        clean = np.zeros((16, 32, 32), dtype=np.float32)
        # Add some structure
        clean[4:12, 8:24, 8:24] = 1.0
        noisy = clean + rng.normal(0, 0.1, clean.shape).astype(np.float32)
        return noisy, clean

    def test_3d_default(self, noisy_volume):
        noisy, clean = noisy_volume
        denoised = denoise_volume_array(noisy, h=0.05, backend="skimage")
        assert denoised.shape == noisy.shape
        assert denoised.dtype == np.float32

    def test_2d_mode(self, noisy_volume):
        noisy, clean = noisy_volume
        denoised = denoise_volume_array(noisy, h=0.05, use_2d=True, backend="skimage")
        assert denoised.shape == noisy.shape

    def test_preserves_scale(self, noisy_volume):
        noisy, _ = noisy_volume
        # Scale to uint16-like range
        scaled = noisy * 10000 + 5000
        denoised = denoise_volume_array(scaled, h=0.05, backend="skimage")
        # Output should be in similar range (not [0,1])
        assert denoised.min() > 1000
        assert denoised.max() > 5000

    def test_2d_input(self):
        rng = np.random.RandomState(42)
        noisy = rng.rand(32, 32).astype(np.float32)
        denoised = denoise_volume_array(noisy, h=0.05, backend="skimage")
        assert denoised.shape == (32, 32)

    def test_norm_range_equalizes_denoise_strength(self):
        """End-to-end: a fixed h + shared norm_range gives matched smoothing.

        Two tiles carry the SAME absolute noise but different dynamic ranges
        (a dim tile spanning ~[0, d], a bright tile spanning ~[0, G]). With a
        shared whole-volume range the fixed ``h`` removes the same fraction of
        noise in both. With per-tile [0,1] normalization the dim tile's noise
        is stretched ~G/d more, so the fixed ``h`` under-smooths it — the drift.

        We measure a scale-invariant "smoothing effectiveness" (residual std /
        input std in a flat interior); the ratio is unaffected by the tile's
        own amplitude, so it isolates the effective strength.
        """
        pytest.importorskip("torch")

        rng = np.random.RandomState(7)
        d, big_g, sigma = 5.0, 50.0, 0.5
        shape = (12, 48, 48)
        # A bright slab (full-depth in z) sets the tile's dynamic range; the
        # measurement region is STRICTLY inside it (edges > search+patch away)
        # so we probe pure flat-region noise, isolating effective smoothing.
        block = (slice(None), slice(10, 38), slice(10, 38))
        meas = (slice(4, 8), slice(18, 30), slice(18, 30))

        def make_tile(amp):
            clean = np.zeros(shape, dtype=np.float32)
            clean[block] = amp
            return clean + rng.normal(0, sigma, shape).astype(np.float32)

        dim = make_tile(d)
        bright = make_tile(big_g)
        global_range = (0.0, big_g)

        def effectiveness(vol, **kw):
            in_std = float(vol[meas].std())
            den = denoise_volume_array(vol, h=0.06, backend="skimage", **kw)
            return float(den[meas].std()) / in_std

        # Shared global range: effective strength matches across tiles.
        s_dim = effectiveness(dim, norm_range=global_range)
        s_bright = effectiveness(bright, norm_range=global_range)
        shared_ratio = max(s_dim, s_bright) / max(min(s_dim, s_bright), 1e-9)

        # Per-tile range: effective strength diverges (the drift the fix removes).
        p_dim = effectiveness(dim)
        p_bright = effectiveness(bright)
        pertile_ratio = max(p_dim, p_bright) / max(min(p_dim, p_bright), 1e-9)

        # Shared-range smoothing is far more consistent tile-to-tile.
        assert shared_ratio < pertile_ratio
        # The bright tile's own range ≈ the global range, so it is essentially
        # unchanged between the two schemes...
        assert abs(p_bright - s_bright) / s_bright < 0.1
        # ...while the low-dynamic-range dim tile is smoothed markedly more
        # weakly under per-tile normalization (the drift the fix removes).
        assert p_dim > 1.5 * s_dim
