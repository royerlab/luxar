"""Unit tests for blind-spot CV utilities and the calibration driver."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from luxar.gsplats.calibration import (
    CalibrationResult,
    HeldOutPeak,
    NoiseFloor,
    RDModel,
    SplatDensity,
    build_k_grid,
    calibrate,
    count_features,
    cv_mask,
    donut_median_fill,
    estimate_noise_floor,
    find_k_star,
    fit_rd_model,
    foreground_mask_otsu,
    held_out_gain_db,
    held_out_psnr,
    held_out_psnr_foreground,
    predict_zero_baseline_mse,
    select_calibration_region,
)

# -----------------------------------------------------------------------------
# cv_mask
# -----------------------------------------------------------------------------


class TestCvMask:
    def test_determinism_same_seed(self):
        m1 = cv_mask((10, 10, 10), seed=42)
        m2 = cv_mask((10, 10, 10), seed=42)
        np.testing.assert_array_equal(m1, m2)

    def test_different_seeds_differ(self):
        m1 = cv_mask((50, 50), seed=42)
        m2 = cv_mask((50, 50), seed=43)
        assert not np.array_equal(m1, m2)

    def test_shape_and_dtype(self):
        m = cv_mask((4, 5, 6))
        assert m.shape == (4, 5, 6)
        assert m.dtype == bool

    def test_fraction_within_binomial_ci(self):
        # For 1M draws at p=0.05 the std is sqrt(N*p*(1-p)) ≈ 218; ±5σ → ±1090
        N = 100 * 100 * 100
        m = cv_mask((100, 100, 100), fraction=0.05, seed=42)
        n_true = int(m.sum())
        expected = N * 0.05
        assert abs(n_true - expected) < 5 * math.sqrt(N * 0.05 * 0.95)

    def test_invalid_fraction(self):
        with pytest.raises(ValueError):
            cv_mask((10, 10), fraction=0.0)
        with pytest.raises(ValueError):
            cv_mask((10, 10), fraction=1.0)


# -----------------------------------------------------------------------------
# donut_median_fill
# -----------------------------------------------------------------------------


class TestDonutFill:
    def test_constant_volume_unchanged(self):
        # On a constant volume the donut median equals the centre value
        V = np.full((8, 8, 8), 0.42, dtype=np.float32)
        mask = cv_mask(V.shape, fraction=0.1, seed=0)
        out = donut_median_fill(V, mask)
        np.testing.assert_allclose(out, V, atol=1e-6)

    def test_unmasked_voxels_untouched(self):
        rng = np.random.default_rng(0)
        V = rng.random((8, 8, 8), dtype=np.float32)
        mask = cv_mask(V.shape, fraction=0.1, seed=1)
        out = donut_median_fill(V, mask)
        np.testing.assert_array_equal(out[~mask], V[~mask])

    def test_gradient_volume_local_average(self):
        # Linear gradient: donut median ≈ centre value (since median of symmetric
        # neighbours is the centre under a linear field). Tolerance allows for
        # the parity asymmetry of an even-sized donut (8 neighbours in 2D).
        Y, X = np.meshgrid(
            np.arange(20, dtype=np.float32),
            np.arange(20, dtype=np.float32),
            indexing="ij",
        )
        V = (Y + X) / 38.0
        # Mask interior cells only to avoid reflect-padding artefacts at edges
        mask = np.zeros_like(V, dtype=bool)
        mask[5:15, 5:15] = cv_mask((10, 10), fraction=0.5, seed=2)
        out = donut_median_fill(V, mask)
        # On a linear field, donut median = centre (within numerical precision)
        np.testing.assert_allclose(out[mask], V[mask], atol=0.06)

    def test_2d(self):
        V = np.arange(25, dtype=np.float32).reshape(5, 5)
        mask = np.zeros_like(V, dtype=bool)
        mask[2, 2] = True
        out = donut_median_fill(V, mask)
        # 8 neighbours of V[2,2]=12 are {6,7,8,11,13,16,17,18}, median = 12
        assert math.isclose(out[2, 2], 12.0, abs_tol=1e-6)
        np.testing.assert_array_equal(out[~mask], V[~mask])

    def test_4d(self):
        rng = np.random.default_rng(0)
        V = rng.random((4, 4, 4, 4), dtype=np.float32)
        mask = cv_mask(V.shape, fraction=0.1, seed=3)
        out = donut_median_fill(V, mask)
        assert out.shape == V.shape
        assert out.dtype == V.dtype

    def test_no_masked_voxels(self):
        V = np.ones((4, 4), dtype=np.float32)
        mask = np.zeros_like(V, dtype=bool)
        out = donut_median_fill(V, mask)
        np.testing.assert_array_equal(out, V)
        assert out is not V  # must be a copy

    def test_shape_mismatch_raises(self):
        V = np.zeros((4, 4))
        mask = np.zeros((4, 5), dtype=bool)
        with pytest.raises(ValueError):
            donut_median_fill(V, mask)

    def test_radius_bounds(self):
        V = np.zeros((4, 4))
        mask = np.zeros_like(V, dtype=bool)
        with pytest.raises(ValueError):
            donut_median_fill(V, mask, radius=0)


# -----------------------------------------------------------------------------
# held_out_psnr
# -----------------------------------------------------------------------------


class TestHeldOutPSNR:
    def test_perfect_reconstruction_inf(self):
        V = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
        mask = cv_mask(V.shape, fraction=0.05, seed=0)
        psnr = held_out_psnr(V, V, mask)
        assert math.isinf(psnr)

    def test_known_mse(self):
        V = np.zeros((4, 4), dtype=np.float32)
        V_hat = np.zeros_like(V)
        mask = np.zeros_like(V, dtype=bool)
        mask[0, 0] = True
        V_hat[0, 0] = 0.1
        # data_range from V: 0 → fall back to data_range=1.0
        psnr = held_out_psnr(V_hat, V, mask, data_range=1.0)
        # MSE = 0.01, PSNR = 10*log10(1/0.01) = 20
        assert math.isclose(psnr, 20.0, abs_tol=1e-6)

    def test_empty_mask_nan(self):
        V = np.zeros((4, 4))
        V_hat = np.zeros_like(V)
        mask = np.zeros_like(V, dtype=bool)
        psnr = held_out_psnr(V_hat, V, mask)
        assert math.isnan(psnr)

    def test_shape_mismatch_raises(self):
        V = np.zeros((4, 4))
        with pytest.raises(ValueError):
            held_out_psnr(np.zeros((4, 5)), V, np.zeros_like(V, dtype=bool))


# -----------------------------------------------------------------------------
# build_k_grid
# -----------------------------------------------------------------------------


class TestBuildKGrid:
    def test_exp_default_matches_manuscript(self):
        # n=10, [1k, 512k] log-spaced → manuscript {1K, 2K, 4K, ..., 512K}
        grid = build_k_grid(n_points=10, k_min=1_000, k_max=512_000, progression="exp")
        expected = [
            1_000,
            2_000,
            4_000,
            8_000,
            16_000,
            32_000,
            64_000,
            128_000,
            256_000,
            512_000,
        ]
        # Allow off-by-one rounding tolerance per point
        assert len(grid) == len(expected)
        for got, want in zip(grid, expected):
            assert abs(got - want) <= max(1, want // 100)

    def test_endpoints_pinned(self):
        grid = build_k_grid(n_points=5, k_min=100, k_max=10_000, progression="exp")
        assert grid[0] == 100
        assert grid[-1] == 10_000

    def test_power_progression_denser_at_low_k(self):
        # power=2 makes K_i = k_min + (k_max - k_min) * (i/(N-1))^2
        # First gap is small, last gap is large.
        grid = build_k_grid(
            n_points=5, k_min=100, k_max=10_000, progression="power", power=2
        )
        assert grid[0] == 100
        assert grid[-1] == 10_000
        # Last gap > first gap (denser at low K under quadratic spacing)
        gaps = np.diff(grid)
        assert gaps[-1] > gaps[0]

    def test_explicit_passes_through(self):
        out = build_k_grid(explicit=[1_000, 4_000, 16_000])
        assert out == [1_000, 4_000, 16_000]

    def test_explicit_takes_precedence(self):
        out = build_k_grid(explicit=[100, 200, 300], n_points=10, k_max=1_000_000)
        assert out == [100, 200, 300]

    def test_explicit_validation(self):
        with pytest.raises(ValueError):
            build_k_grid(explicit=[1000])  # need at least 2
        with pytest.raises(ValueError):
            build_k_grid(explicit=[1000, -5])

    def test_invalid_progression(self):
        with pytest.raises(ValueError):
            build_k_grid(progression="cubic")  # not a valid name

    def test_n_points_too_small(self):
        with pytest.raises(ValueError):
            build_k_grid(n_points=1)

    def test_kmax_le_kmin(self):
        with pytest.raises(ValueError):
            build_k_grid(k_min=1000, k_max=1000)


# -----------------------------------------------------------------------------
# find_k_star
# -----------------------------------------------------------------------------


class TestFindKStar:
    def test_clear_peak_in_middle(self):
        ks = [1, 2, 4, 8, 16, 32, 64]
        # bell-shape, peak at index 3 (k=8); both flanks well below
        psnr = [22.0, 26.0, 30.0, 34.0, 33.5, 32.0, 30.0]
        out = find_k_star(ks, psnr)
        assert out.type == "peak"
        assert out.k_star == 8
        assert out.confidence_db > 0

    def test_signal_limited_monotone_rising(self):
        ks = [1, 4, 16, 64, 256]
        psnr = [26.0, 30.0, 34.0, 38.0, 42.0]
        out = find_k_star(ks, psnr)
        assert out.type == "signal_limited"
        assert out.k_star == 256
        assert out.confidence_db == pytest.approx(16.0)

    def test_plateau_returns_smallest_within_tolerance(self):
        ks = [1, 2, 4, 8, 16, 32]
        # Sharp rise, then a flat top within 0.3 dB. Post-mean is < 0.1 dB
        # below the peak, so the peak rule rejects this curve and we fall
        # through to plateau detection.
        psnr = [30.0, 35.0, 39.5, 39.78, 39.80, 39.79]
        out = find_k_star(ks, psnr)
        # peak = 39.80 at idx 4; pre_mean = (30+35+39.5+39.78)/4 = 36.07
        # post_mean = 39.79 — gap = 0.01, fails 0.1 dB cutoff → not "peak"
        # threshold = 39.50; smallest above-threshold idx = 2 → k=4
        assert out.type == "plateau"
        assert out.k_star == 4

    def test_handles_nan_in_psnr(self):
        ks = [1, 2, 4, 8]
        psnr = [22.0, float("nan"), 30.0, 28.0]
        out = find_k_star(ks, psnr)
        # Should still pick a finite peak/plateau
        assert out.k_star in ks
        assert math.isfinite(out.confidence_db)

    def test_flat_top_with_max_at_last_k_is_plateau(self):
        # Regression for the real deconvolved-tile cal: curve rises then goes
        # flat, but its noisy maximum lands on the LAST K. Total rise > 0.3 dB,
        # yet the tail is flat (<0.1 dB) → must be 'plateau', not 'signal_limited'
        # (the latter would inflate the K*-derived splat density).
        ks = [16000, 64000, 128000, 256000, 512000]
        psnr = [42.15, 43.29, 43.50, 43.49, 43.52]  # last step +0.03 dB (flat)
        out = find_k_star(ks, psnr)
        assert out.type == "plateau"
        assert out.k_star == 64000  # diminishing-returns onset, not 512k

    def test_still_climbing_tail_stays_signal_limited(self):
        # A genuinely starved curve (large last step) is unchanged.
        ks = [16000, 64000, 128000, 256000, 512000]
        psnr = [45.4, 45.9, 46.2, 47.8, 49.3]  # last step +1.5 dB
        out = find_k_star(ks, psnr)
        assert out.type == "signal_limited"
        assert out.k_star == 512000

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            find_k_star([1, 2, 4], [10.0, 20.0])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            find_k_star([10], [25.0])


# -----------------------------------------------------------------------------
# estimate_noise_floor
# -----------------------------------------------------------------------------


class TestNoiseFloor:
    def test_gaussian_noise_recovered(self):
        # Synthetic: smooth signal + known-σ noise → sigma_hat within 15 %
        rng = np.random.default_rng(0)
        Y, X, Z = np.meshgrid(
            np.linspace(0, 1, 64),
            np.linspace(0, 1, 64),
            np.linspace(0, 1, 64),
            indexing="ij",
        )
        signal = 0.5 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        sigma_true = 0.02
        V = (signal + sigma_true * rng.standard_normal(signal.shape)).astype(np.float32)
        V = np.clip(V, 0, 1)  # match [0, 1] normalisation contract
        nf = estimate_noise_floor(V)
        # Both high-pass estimators should agree within 25 % of σ_true.
        # Background MAD undershoots when the signal contains low-percentile
        # voxels too — the ensemble median is robust to that one outlier.
        assert abs(nf.sigma_laplacian - sigma_true) / sigma_true < 0.25
        assert abs(nf.sigma_haar - sigma_true) / sigma_true < 0.25
        assert abs(nf.sigma_hat - sigma_true) / sigma_true < 0.25
        # PSNR ceiling for [0,1] data with σ ≈ 0.02 is around 34 dB
        assert 30 < nf.psnr_max_db < 40

    def test_zero_volume_handles_gracefully(self):
        V = np.zeros((16, 16, 16), dtype=np.float32)
        nf = estimate_noise_floor(V)
        # All estimators should report 0 noise; PSNR ceiling = +inf
        assert nf.sigma_hat == 0.0 or math.isnan(nf.sigma_hat)
        assert math.isinf(nf.psnr_max_db) or math.isnan(nf.psnr_max_db)

    # [P5][P12] σ̂ tracks σ_true across multiple magnitudes — guards against
    # estimator clipping/saturation at small or large noise levels.
    @pytest.mark.parametrize("sigma_true", [0.005, 0.02, 0.05])
    def test_sigma_recovered_across_magnitudes(self, sigma_true: float) -> None:
        """σ̂ from a noisy smooth signal recovers σ_true within 30% across a
        range of magnitudes."""
        rng = np.random.default_rng(123)
        Y, X, Z = np.meshgrid(
            np.linspace(0, 1, 48),
            np.linspace(0, 1, 48),
            np.linspace(0, 1, 48),
            indexing="ij",
        )
        signal = 0.5 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        V = (signal + sigma_true * rng.standard_normal(signal.shape)).astype(np.float32)
        V = np.clip(V, 0, 1)
        nf = estimate_noise_floor(V)
        # High-pass estimators should track σ_true linearly; allow 30% slack
        # at extremes where the signal's high-frequency content interferes
        # mildly with the Laplacian estimator.
        assert abs(nf.sigma_hat - sigma_true) / sigma_true < 0.30, (
            f"σ_true={sigma_true} σ̂={nf.sigma_hat:.5f} "
            f"rel-err={abs(nf.sigma_hat - sigma_true) / sigma_true:.3f}"
        )

    # [P5][P8] σ̂ should not depend on signal morphology at fixed σ — verifies
    # the ensemble median decouples noise from underlying structure.
    def test_sigma_invariant_to_signal_shape(self) -> None:
        """Same σ_true on different smooth signals yields σ̂ values within
        50% of each other."""
        rng = np.random.default_rng(7)
        sigma_true = 0.03
        shape = (48, 48, 48)
        Y, X, Z = np.meshgrid(
            np.linspace(0, 1, shape[0]),
            np.linspace(0, 1, shape[1]),
            np.linspace(0, 1, shape[2]),
            indexing="ij",
        )
        # Three structurally-different smooth signals
        signals = {
            "sinusoid": 0.5 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y),
            "radial": 0.5
            + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) / 0.1),
            "ramp": 0.2 + 0.6 * X,
        }
        sigmas = []
        for sig in signals.values():
            V = (sig + sigma_true * rng.standard_normal(shape)).astype(np.float32)
            V = np.clip(V, 0, 1)
            sigmas.append(estimate_noise_floor(V).sigma_hat)
        ratio = max(sigmas) / min(sigmas)
        assert ratio < 1.5, (
            f"σ̂ should be shape-invariant; got {sigmas} (ratio {ratio:.3f})"
        )
        # And all values still within 35% of σ_true
        for s in sigmas:
            assert abs(s - sigma_true) / sigma_true < 0.35, (
                f"σ̂={s} far from σ_true={sigma_true}"
            )


# -----------------------------------------------------------------------------
# CalibrationResult round-trip
# -----------------------------------------------------------------------------


class TestCalibrationResult:
    def test_json_round_trip(self, tmp_path: Path):
        result = CalibrationResult(
            k_values_requested=[100, 500, 2000],
            k_values_effective=[98, 488, 1942],
            held_out_psnr_db=[25.1, 27.3, 26.0],
            train_psnr_db=[25.4, 28.0, 28.6],
            held_out_mse=[3.1e-3, 2.0e-3, 2.5e-3],
            full_psnr_db=[25.4, 27.9, 28.5],
            full_ssim=[0.7, 0.85, 0.86],
            held_out_peak=HeldOutPeak(k_star=500, type="peak", confidence_db=1.3),
            noise_floor=NoiseFloor(
                sigma_hat=0.01,
                sigma_laplacian=0.011,
                sigma_haar=0.009,
                sigma_background=0.005,
                psnr_max_db=40.0,
            ),
            fit_times_seconds=[1.2, 2.1, 4.4],
            splat_paths=None,
            mask_seed=42,
            mask_fraction=0.05,
            donut_radius=1,
            fit_config={"preset": "draft"},
            volume_shape=[32, 32, 32],
            volume_dtype="float32",
            timestamp="2026-05-06T12:00:00+00:00",
        )
        out = tmp_path / "cal.json"
        result.to_json(out)
        loaded = CalibrationResult.from_json(out)
        assert loaded.k_values_requested == result.k_values_requested
        assert loaded.held_out_peak.k_star == result.held_out_peak.k_star
        assert loaded.held_out_peak.type == result.held_out_peak.type
        assert math.isclose(loaded.noise_floor.sigma_hat, 0.01)
        assert loaded.fit_config == {"preset": "draft"}

    def test_nan_inf_become_null(self, tmp_path: Path):
        result = CalibrationResult(
            k_values_requested=[10],
            k_values_effective=[10],
            held_out_psnr_db=[float("inf")],
            train_psnr_db=[float("nan")],
            held_out_mse=[0.0],
            full_psnr_db=[float("inf")],
            full_ssim=[1.0],
            held_out_peak=HeldOutPeak(
                k_star=10, type="signal_limited", confidence_db=0.0
            ),
            noise_floor=NoiseFloor(
                sigma_hat=0.0,
                sigma_laplacian=0.0,
                sigma_haar=0.0,
                sigma_background=0.0,
                psnr_max_db=float("inf"),
            ),
            fit_times_seconds=[0.1],
            splat_paths=None,
            mask_seed=42,
            mask_fraction=0.05,
            donut_radius=1,
            fit_config={},
            volume_shape=[4, 4, 4],
            volume_dtype="float32",
            timestamp="2026-05-06T12:00:00+00:00",
        )
        out = tmp_path / "cal.json"
        result.to_json(out)
        raw = json.loads(out.read_text())
        assert raw["held_out_psnr_db"] == [None]
        assert raw["train_psnr_db"] == [None]
        # Round-trip: nan/inf come back as nan / inf appropriately
        loaded = CalibrationResult.from_json(out)
        assert math.isnan(loaded.held_out_psnr_db[0])
        assert math.isnan(loaded.train_psnr_db[0])
        assert math.isinf(loaded.noise_floor.psnr_max_db)


# -----------------------------------------------------------------------------
# End-to-end calibrate (small, CPU)
# -----------------------------------------------------------------------------


class TestCalibrateSmoke:
    def test_small_volume_cpu(self, tmp_path: Path):
        # Smoke test: run a 3-K sweep on a tiny synthetic volume on CPU
        rng = np.random.default_rng(0)
        # Smooth structured signal so even a few splats can land somewhere meaningful
        Y, X, Z = np.meshgrid(
            np.linspace(0, 1, 16),
            np.linspace(0, 1, 16),
            np.linspace(0, 1, 16),
            indexing="ij",
        )
        signal = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) * 8)
        V = (signal + 0.02 * rng.standard_normal(signal.shape)).astype(np.float32)
        V = np.clip(V, 0, 1)

        result = calibrate(
            V,
            k_grid=[20, 80, 200],
            fit_kwargs={
                "n_iters": 50,
                "device": "cpu",
                "verbose": False,
                "early_stop_patience": 50,
                "use_cuda": False,
                "use_metal": False,
            },
        )
        assert isinstance(result, CalibrationResult)
        assert len(result.held_out_psnr_db) == 3
        assert len(result.train_psnr_db) == 3
        assert all(math.isfinite(p) or math.isinf(p) for p in result.held_out_psnr_db)
        assert result.held_out_peak.k_star in (20, 80, 200)
        assert result.noise_floor.sigma_hat >= 0
        # JSON serialisation works on the actual produced result
        out_json = tmp_path / "cal.json"
        result.to_json(out_json)
        round_trip = CalibrationResult.from_json(out_json)
        assert round_trip.held_out_peak.k_star == result.held_out_peak.k_star

    def test_default_behaviour_unchanged(self):
        # Reproducibility guardrail: with default flags, K* is EXACTLY the legacy
        # min--max blind-spot peak and no metric switch engages. The regime-robust
        # additions never alter the manuscript default path.
        rng = np.random.default_rng(1)
        Y, X, Z = np.meshgrid(
            np.linspace(0, 1, 16),
            np.linspace(0, 1, 16),
            np.linspace(0, 1, 16),
            indexing="ij",
        )
        signal = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) * 8)
        V = np.clip(signal + 0.05 * rng.standard_normal(signal.shape), 0, 1).astype(
            np.float32
        )
        result = calibrate(
            V,
            k_grid=[20, 80, 200],
            fit_kwargs={
                "n_iters": 50,
                "device": "cpu",
                "verbose": False,
                "early_stop_patience": 50,
                "use_cuda": False,
                "use_metal": False,
            },
        )
        # Default metric, no selected-peak override
        assert result.k_star_metric == "psnr_minmax"
        assert result.held_out_peak_selected is None
        # K* is byte-identical to recomputing find_k_star on the min--max curve
        legacy = find_k_star(result.k_values_requested, result.held_out_psnr_db)
        assert result.held_out_peak.k_star == legacy.k_star
        assert result.held_out_peak.type == legacy.type
        # Transferable density still emitted (additive, doesn't affect K*)
        assert result.splat_density is not None
        assert result.splat_density["feature_method"] == "peaks"


# -----------------------------------------------------------------------------
# Content-aware metrics (regime-robust extensions)
# -----------------------------------------------------------------------------


def _sparse_blobs(shape=(40, 40, 40), centers=None, sigma2=3.0):
    """Mostly-zero volume with a few Gaussian blobs (sparse fluorescence-like)."""
    if centers is None:
        centers = [(8, 8, 8), (10, 12, 9), (7, 11, 13)]
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    V = np.zeros(shape, np.float32)
    for cz, cy, cx in centers:
        V += np.exp(
            -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / sigma2)
        ).astype(np.float32)
    return np.clip(V, 0, 1)


class TestContentAwareMetric:
    def test_gain_zero_when_no_better_than_baseline(self):
        V = _sparse_blobs()
        mask = cv_mask(V.shape, 0.05, 42)
        base = predict_zero_baseline_mse(V, mask)
        # A predict-zero "reconstruction": held MSE == baseline -> 0 dB gain.
        assert math.isclose(held_out_gain_db(base, base), 0.0, abs_tol=1e-9)
        # Perfect reconstruction -> infinite gain.
        assert math.isinf(held_out_gain_db(0.0, base))

    def test_minmax_psnr_inflated_but_gain_flat_for_predict_zero(self):
        # The crux: on sparse data the min--max PSNR of a predict-zero recon is
        # HIGH (background-dominated) while gain-over-baseline is ~0 dB.
        V = _sparse_blobs()
        mask = cv_mask(V.shape, 0.05, 42)
        zeros = np.zeros_like(V)
        base = predict_zero_baseline_mse(V, mask)
        held_mse = float(np.mean((zeros[mask] - V[mask]) ** 2))
        psnr_minmax = held_out_psnr(zeros, V, mask)
        gain = held_out_gain_db(held_mse, base)
        assert psnr_minmax > 20.0  # deceptively high
        assert abs(gain) < 1e-6  # but truthfully ~0 over baseline

    def test_foreground_psnr_restricts_to_signal(self):
        V = _sparse_blobs()
        mask = cv_mask(V.shape, 0.05, 42)
        fg = foreground_mask_otsu(V)
        assert fg.sum() > 0 and fg.sum() < V.size  # a real subset
        # Perfect recon -> inf in both; a wrong recon -> finite foreground PSNR.
        assert math.isinf(held_out_psnr_foreground(V, V, mask, fg))
        wrong = V + 0.5
        p = held_out_psnr_foreground(wrong, V, mask, fg)
        assert math.isfinite(p)


class TestCountFeatures:
    def test_recovers_separated_blobs(self):
        # 8 well-separated blobs on a coarse grid -> ~8 peaks.
        centers = [
            (z, y, x)
            for z in (8, 24)
            for y in (8, 24)
            for x in (8, 24)
        ]
        V = _sparse_blobs((32, 32, 32), centers=centers, sigma2=2.0)
        n = count_features(V, method="peaks")
        assert 6 <= n <= 10  # recovers ~8

    def test_methods_run_and_are_nonnegative(self):
        V = _sparse_blobs()
        assert count_features(V, method="peaks") >= 0
        assert count_features(V, method="edges") >= 0
        assert count_features(V, method="intensity") >= 0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            count_features(_sparse_blobs(), method="bogus")


class TestSelectRegion:
    def test_densest_picks_content_corner(self):
        V = _sparse_blobs((48, 48, 48))  # blobs near the origin corner
        crop, reg = select_calibration_region(V, region_size=24, strategy="densest")
        assert reg.origin == [0, 0, 0]
        assert reg.size == [24, 24, 24]
        assert crop.shape == (24, 24, 24)
        assert reg.n_features >= 1

    def test_small_volume_returns_whole(self):
        V = _sparse_blobs((16, 16, 16))
        crop, reg = select_calibration_region(V, region_size=32)
        assert reg.strategy == "whole"
        assert crop.shape == V.shape

    def test_unknown_strategy_raises(self):
        V = _sparse_blobs((48, 48, 48))
        with pytest.raises(ValueError):
            select_calibration_region(V, region_size=24, strategy="bogus")


class TestRDModel:
    def test_recovers_known_exponent(self):
        ks = [1000, 4000, 16000, 64000, 256000]
        floor, a, beta = 0.02, 5.0, 0.5
        errs = [floor + a * k**-beta for k in ks]
        rd = fit_rd_model(ks, errs)
        assert rd is not None
        assert abs(rd.beta - beta) < 0.05
        assert abs(rd.floor - floor) < 0.01
        assert rd.converged_fraction > 0.9  # broad sweep -> converged

    def test_k_for_error_inverts(self):
        rd = RDModel(floor=0.01, a=4.0, beta=0.5, rmse=0.0, n_points=5, converged_fraction=1.0)
        k = rd.k_for_error(0.05)
        assert math.isfinite(k)
        assert math.isclose(rd.predict_error(k), 0.05, rel_tol=1e-6)
        assert math.isinf(rd.k_for_error(0.005))  # below floor -> unreachable

    def test_too_few_points_returns_none(self):
        assert fit_rd_model([1000, 2000], [0.1, 0.05]) is None

    def test_not_converged_when_still_climbing(self):
        # A curve far from its floor across the sampled range -> partial convergence.
        ks = [100, 200, 400]
        floor, a, beta = 0.0, 1.0, 0.3
        errs = [floor + a * k**-beta for k in ks]
        rd = fit_rd_model(ks, errs)
        assert rd is not None
        assert rd.converged_fraction < 0.95


class TestSplatDensity:
    def test_predict_k_power_law_and_cap(self):
        d = SplatDensity(
            feature_method="peaks",
            n_features_reference=100,
            k_star_reference=1000,
            saturation_exponent=0.5,
            saturation_cap=4000,
            splats_per_feature=10.0,
        )
        assert d.predict_k(100) == 1000  # ratio 1
        assert d.predict_k(400) == 2000  # 1000 * 4^0.5
        assert d.predict_k(1_000_000) == 4000  # capped


# -----------------------------------------------------------------------------
# Backward-compatibility: old cal.json (no new keys) must still load
# -----------------------------------------------------------------------------


class TestBackwardCompat:
    def test_old_json_loads_with_defaults(self, tmp_path: Path):
        # A minimal pre-extension cal.json (none of the new keys present).
        old = {
            "k_values_requested": [100, 500],
            "k_values_effective": [98, 488],
            "held_out_psnr_db": [25.1, 27.3],
            "train_psnr_db": [25.4, 28.0],
            "held_out_mse": [3.1e-3, 2.0e-3],
            "full_psnr_db": [25.4, 27.9],
            "full_ssim": [0.7, 0.85],
            "held_out_peak": {"k_star": 500, "type": "peak", "confidence_db": 1.3},
            "noise_floor": {
                "sigma_hat": 0.01,
                "sigma_laplacian": 0.011,
                "sigma_haar": 0.009,
                "sigma_background": 0.005,
                "psnr_max_db": 40.0,
            },
            "fit_times_seconds": [1.2, 2.1],
            "splat_paths": None,
            "mask_seed": 42,
            "mask_fraction": 0.05,
            "donut_radius": 1,
            "fit_config": {"preset": "n2s"},
            "volume_shape": [32, 32, 32],
            "volume_dtype": "float32",
            "timestamp": "2026-05-06T12:00:00+00:00",
        }
        p = tmp_path / "old_cal.json"
        p.write_text(json.dumps(old))
        loaded = CalibrationResult.from_json(p)
        # Existing fields intact
        assert loaded.held_out_peak.k_star == 500
        # New fields default gracefully
        assert loaded.k_star_metric == "psnr_minmax"
        assert loaded.held_out_peak_selected is None
        assert loaded.splat_density is None
        assert loaded.rd_model is None
        assert loaded.not_converged is False
        assert loaded.held_out_gain_db == []
        assert loaded.calibration_region is None
        assert math.isnan(loaded.predict_zero_baseline_mse)
