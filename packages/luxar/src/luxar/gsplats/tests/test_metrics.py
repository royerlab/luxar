"""Tests for quality metrics (PSNR, SSIM, MSE, rel_l2, max_abs_error)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from luxar.gsplats.metrics import (
    _should_tile_ssim,
    _ssim_nd,
    _ssim_nd_tiled,
    compute_foreground_psnr,
    compute_psnr,
    compute_quality_metrics,
    compute_ssim,
    otsu_threshold,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def identical_2d(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.rand(64, 64, device=device)
    return t, t.clone()


@pytest.fixture
def identical_3d(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.rand(16, 16, 16, device=device)
    return t, t.clone()


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


class TestSSIM:
    def test_identical_2d(
        self, identical_2d: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        a, b = identical_2d
        ssim = compute_ssim(a, b)
        assert ssim == pytest.approx(1.0, abs=1e-5)

    def test_identical_3d(
        self, identical_3d: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        a, b = identical_3d
        ssim = compute_ssim(a, b)
        assert ssim == pytest.approx(1.0, abs=1e-5)

    def test_noise_low_ssim(self, device: torch.device) -> None:
        torch.manual_seed(0)
        a = torch.rand(64, 64, device=device)
        b = torch.rand(64, 64, device=device)
        ssim = compute_ssim(a, b)
        assert ssim < 0.3  # uncorrelated noise => low SSIM

    def test_constant_images(self, device: torch.device) -> None:
        a = torch.ones(32, 32, device=device) * 0.5
        b = torch.ones(32, 32, device=device) * 0.5
        ssim = compute_ssim(a, b)
        assert ssim == pytest.approx(1.0, abs=1e-5)

    def test_small_volume_no_crash(self, device: torch.device) -> None:
        """Window auto-shrinks when input is smaller than default window_size=11."""
        torch.manual_seed(42)
        a = torch.rand(8, 8, 8, device=device)
        b = a + 0.1 * torch.randn_like(a)
        ssim = compute_ssim(a, b)
        assert -1.0 <= ssim <= 1.0

    def test_tiny_2d_no_crash(self, device: torch.device) -> None:
        """Even 3x3 images should not crash."""
        a = torch.rand(3, 3, device=device)
        b = a.clone()
        ssim = compute_ssim(a, b)
        assert ssim == pytest.approx(1.0, abs=1e-4)

    def test_shape_mismatch_raises(self, device: torch.device) -> None:
        a = torch.rand(32, 32, device=device)
        b = torch.rand(32, 64, device=device)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_ssim(a, b)

    def test_inputs_not_modified(self, device: torch.device) -> None:
        """In-place optimisations in _ssim_nd must not mutate pred/target."""
        torch.manual_seed(99)
        a = torch.rand(32, 32, 32, device=device)
        b = torch.rand(32, 32, 32, device=device)
        a_copy = a.clone()
        b_copy = b.clone()
        compute_ssim(a, b)
        assert torch.equal(a, a_copy)
        assert torch.equal(b, b_copy)


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------


class TestPSNR:
    def test_identical_infinite(
        self, identical_2d: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        a, b = identical_2d
        psnr = compute_psnr(a, b)
        assert psnr == float("inf")

    def test_known_mse(self, device: torch.device) -> None:
        """PSNR for data_range=1, MSE=0.01 should be ~20 dB."""
        a = torch.zeros(100, 100, device=device)
        b = torch.ones(100, 100, device=device) * 0.1  # MSE = 0.01
        psnr = compute_psnr(a, b, data_range=1.0)
        assert psnr == pytest.approx(20.0, abs=0.1)

    def test_shape_mismatch_raises(self, device: torch.device) -> None:
        a = torch.rand(10, 10, device=device)
        b = torch.rand(10, 20, device=device)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_psnr(a, b)


# ---------------------------------------------------------------------------
# compute_quality_metrics (aggregate)
# ---------------------------------------------------------------------------


class TestQualityMetrics:
    def test_returns_all_keys(
        self, identical_2d: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        a, b = identical_2d
        m = compute_quality_metrics(a, b)
        expected_keys = {
            "mse",
            "psnr_db",
            "ssim",
            "rel_l2",
            "max_abs_error",
            "foreground_psnr_db",
            "foreground_threshold",
            "foreground_fraction",
            "psnr_robust_db",
            "foreground_psnr_robust_db",
            "foreground_ssim",
            "data_range",
            "robust_data_range",
        }
        assert set(m.keys()) == expected_keys

    def test_identical_metrics(
        self, identical_3d: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        a, b = identical_3d
        m = compute_quality_metrics(a, b)
        assert m["mse"] == pytest.approx(0.0, abs=1e-10)
        assert m["psnr_db"] == float("inf")
        assert m["ssim"] == pytest.approx(1.0, abs=1e-5)
        assert m["rel_l2"] == pytest.approx(0.0, abs=1e-10)
        assert m["max_abs_error"] == pytest.approx(0.0, abs=1e-10)

    def test_noisy_metrics_range(self, device: torch.device) -> None:
        """Audit W6 fix: previous `> 0` bounds were so loose a mutant
        returning constant 999 for every metric would pass. Pin each
        metric to a derivable bound for the controlled noise (σ=0.05
        Gaussian additive on Uniform[0,1])."""
        torch.manual_seed(42)
        a = torch.rand(32, 32, 32, device=device)
        b = a + 0.05 * torch.randn_like(a)
        m = compute_quality_metrics(a, b)

        # MSE of N(0, σ²) noise on values in [0, 1] is ~σ² = 0.0025.
        # Allow ±50% slop for the finite-sample variance over 32³ voxels.
        assert 0.0015 < m["mse"] < 0.0045, f"MSE outside expected band: {m['mse']}"

        # PSNR_dB = 10 log10(MAX² / MSE), MAX ≈ 1 → ~ 26 dB; allow [23, 29].
        assert 23.0 < m["psnr_db"] < 29.0, f"PSNR outside expected band: {m['psnr_db']}"

        # SSIM strictly between 0 and 1 (degenerate sentinels would fail).
        assert 0.0 < m["ssim"] < 1.0

        # rel_l2 = ||a - b|| / ||a||. For uniform a with mean 0.5 and σ=0.05
        # noise on 32³ voxels: ~ sqrt(MSE) / sqrt(E[a²]) ≈ 0.05 / 0.58 ≈ 0.09.
        # Allow [0.04, 0.20] to absorb sample variance and SSIM-coupled drift.
        assert 0.04 < m["rel_l2"] < 0.20, f"rel_l2 outside expected band: {m['rel_l2']}"

        # Max abs error: extreme tail of N(0, 0.05²) over 32³ ≈ 32768 draws.
        # Expected magnitude ~ 0.05 * sqrt(2 ln 32768) ≈ 0.05 * 4.55 ≈ 0.23.
        # Allow generous [0.10, 0.50].
        assert 0.10 < m["max_abs_error"] < 0.50, (
            f"max_abs_error outside expected band: {m['max_abs_error']}"
        )

    def test_ssim_matches_skimage(self) -> None:
        """Validate SSIM matches scikit-image to high precision."""
        import numpy as np
        from skimage.metrics import structural_similarity as skimage_ssim

        np.random.seed(123)
        a_np = np.random.rand(32, 32).astype(np.float32)
        b_np = a_np + 0.1 * np.random.randn(32, 32).astype(np.float32)
        dr = float(b_np.max() - b_np.min())

        ours = compute_ssim(
            torch.from_numpy(a_np), torch.from_numpy(b_np), data_range=dr
        )
        ref = skimage_ssim(
            a_np,
            b_np,
            data_range=dr,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
        )
        assert ours == pytest.approx(ref, abs=1e-4)


# ---------------------------------------------------------------------------
# _ssim_nd return type
# ---------------------------------------------------------------------------


class TestSSIMReturnType:
    def test_returns_tuple(self, device: torch.device) -> None:
        a = torch.rand(32, 32, device=device)
        result = _ssim_nd(a, a.clone(), window_size=11, data_range=1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        ssim_sum, num_voxels = result
        assert isinstance(ssim_sum, float)
        assert isinstance(num_voxels, int)
        assert num_voxels > 0

    def test_mean_matches_compute_ssim(self, device: torch.device) -> None:
        """ssim_sum / num_voxels should equal compute_ssim output."""
        torch.manual_seed(42)
        a = torch.rand(32, 32, 32, device=device)
        b = a + 0.1 * torch.randn_like(a)
        dr = float((b.max() - b.min()).item())

        ssim_sum, num_voxels = _ssim_nd(a, b, window_size=11, data_range=dr)
        expected = compute_ssim(a, b, data_range=dr)
        assert ssim_sum / num_voxels == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Tiled SSIM
# ---------------------------------------------------------------------------


class TestTiledSSIM:
    def test_tiled_matches_non_tiled_3d(self, device: torch.device) -> None:
        """Tiled SSIM must match non-tiled to < 1e-4."""
        torch.manual_seed(42)
        a = torch.rand(64, 64, 64, device=device)
        b = a + 0.1 * torch.randn_like(a)
        dr = 1.0

        s_ref, n_ref = _ssim_nd(a, b, window_size=11, data_range=dr)
        s_tiled, n_tiled = _ssim_nd_tiled(
            a, b, window_size=11, data_range=dr, tile_size=32
        )

        assert n_tiled == n_ref
        assert s_tiled / n_tiled == pytest.approx(s_ref / n_ref, abs=1e-4)

    def test_tiled_matches_non_tiled_2d(self, device: torch.device) -> None:
        torch.manual_seed(42)
        a = torch.rand(128, 128, device=device)
        b = a + 0.1 * torch.randn_like(a)
        dr = 1.0

        s_ref, n_ref = _ssim_nd(a, b, window_size=11, data_range=dr)
        s_tiled, n_tiled = _ssim_nd_tiled(
            a, b, window_size=11, data_range=dr, tile_size=48
        )

        assert n_tiled == n_ref
        assert s_tiled / n_tiled == pytest.approx(s_ref / n_ref, abs=1e-4)

    def test_volume_smaller_than_tile(self, device: torch.device) -> None:
        """When volume fits in one tile, result is identical to non-tiled."""
        torch.manual_seed(42)
        a = torch.rand(16, 16, 16, device=device)
        b = a + 0.05 * torch.randn_like(a)
        dr = 1.0

        s_ref, n_ref = _ssim_nd(a, b, window_size=11, data_range=dr)
        s_tiled, n_tiled = _ssim_nd_tiled(
            a, b, window_size=11, data_range=dr, tile_size=128
        )

        assert s_tiled == pytest.approx(s_ref, abs=1e-7)
        assert n_tiled == n_ref

    def test_tiled_identical_volumes(self, device: torch.device) -> None:
        """Identical volumes give SSIM ~1.0 even through tiled path."""
        a = torch.rand(64, 64, 64, device=device)
        s, n = _ssim_nd_tiled(
            a, a.clone(), window_size=11, data_range=1.0, tile_size=32
        )
        assert s / n == pytest.approx(1.0, abs=1e-5)

    def test_num_voxels_consistency(self, device: torch.device) -> None:
        """Total voxels from tiled must equal non-tiled valid region."""
        torch.manual_seed(42)
        a = torch.rand(48, 48, 48, device=device)
        b = a + 0.05 * torch.randn_like(a)
        dr = 1.0

        _, n_ref = _ssim_nd(a, b, window_size=11, data_range=dr)
        _, n_tiled = _ssim_nd_tiled(a, b, window_size=11, data_range=dr, tile_size=24)
        assert n_tiled == n_ref

    def test_non_cubic_volume(self, device: torch.device) -> None:
        """Tiling works correctly for non-cubic volumes."""
        torch.manual_seed(42)
        a = torch.rand(100, 40, 60, device=device)
        b = a + 0.1 * torch.randn_like(a)
        dr = 1.0

        s_ref, n_ref = _ssim_nd(a, b, window_size=11, data_range=dr)
        s_tiled, n_tiled = _ssim_nd_tiled(
            a, b, window_size=11, data_range=dr, tile_size=32
        )

        assert n_tiled == n_ref
        assert s_tiled / n_tiled == pytest.approx(s_ref / n_ref, abs=1e-4)


# ---------------------------------------------------------------------------
# Auto-tiling heuristic
# ---------------------------------------------------------------------------


class TestShouldTile:
    def test_cpu_tiles_only_past_host_budget(self) -> None:
        # CPU/MPS free memory is not queryable, so a fixed budget decides: a
        # small volume stays on the untiled path, a large one tiles (untiled,
        # 65 Mvox costs ~1.8 GB of temporaries; tiled, ~0.37 GB).
        cpu = torch.device("cpu")
        assert _should_tile_ssim((64, 64, 64), cpu) is False
        assert _should_tile_ssim((1000, 1000, 1000), cpu) is True

    def test_auto_tiled_cpu_matches_untiled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The auto-selected tiled path must agree with the untiled kernel.
        # A tiny budget forces tiling on a tiny volume (no large allocation).
        import luxar.gsplats.metrics as metrics_mod

        monkeypatch.setattr(metrics_mod, "_SSIM_HOST_PEAK_BUDGET_BYTES", 1024)
        g = torch.Generator().manual_seed(0)
        # One axis longer than the default 128-voxel tile so the volume splits.
        a = torch.rand(16, 300, 24, generator=g)
        b = (a + 0.05 * torch.rand(16, 300, 24, generator=g)).clamp(0, 1)
        assert _should_tile_ssim(a.shape, a.device) is True
        s_ref, n_ref = _ssim_nd(a, b, window_size=11, data_range=1.0)
        auto = compute_ssim(a, b, data_range=1.0)
        assert auto == pytest.approx(s_ref / n_ref, abs=1e-6)

    def test_small_volume_no_tile(self, device: torch.device) -> None:
        # 32^3 * 4 bytes * 8 ≈ 1 MB — should never exceed 50% GPU memory
        assert _should_tile_ssim((32, 32, 32), device) is False


# ---------------------------------------------------------------------------
# Foreground PSNR
# ---------------------------------------------------------------------------


class TestOtsuThreshold:
    """The reimplementation must agree with scikit-image, or the metric drifts.

    ``otsu_threshold`` exists so the foreground definition carries no optional
    dependency. Calibration shares this implementation, so matching
    scikit-image keeps feature thresholds, intensity counts, and content scans
    on the same scale where scikit-image is installed. This pins that
    equivalence rather than trusting the port.
    """

    @pytest.mark.parametrize(
        "name",
        ["bimodal", "sparse", "uniform", "blob", "wide_range"],
    )
    def test_matches_skimage_exactly(self, name: str) -> None:
        skimage_filters = pytest.importorskip("skimage.filters")
        import numpy as np

        rng = np.random.default_rng(0)
        arrays = {
            "bimodal": np.concatenate(
                [rng.normal(0.1, 0.02, 20000), rng.normal(0.8, 0.05, 2000)]
            ).astype(np.float32),
            # The case that matters for Luxar: a mostly-empty light-sheet stack.
            "sparse": np.concatenate(
                [
                    np.zeros(49000, np.float32),
                    rng.uniform(0.3, 1.0, 1000).astype(np.float32),
                ]
            ),
            "uniform": rng.uniform(0, 1, 20000).astype(np.float32),
            "blob": np.exp(
                -((np.mgrid[0:30, 0:30, 0:30] - 15).__pow__(2).sum(axis=0) / 40)
            ).astype(np.float32),
            "wide_range": rng.integers(0, 4096, 20000).astype(np.float32),
        }
        arr = arrays[name]
        expected = float(skimage_filters.threshold_otsu(arr))
        actual = otsu_threshold(torch.from_numpy(arr.ravel()))
        # Exact, not "within a bin": the port picks the same histogram bin, so
        # only float noise should separate them. A one-bin tolerance is too
        # slack to be worth writing -- it survives swapping the cumulative
        # class indices, which is precisely the porting error to guard against.
        bin_width = (float(arr.max()) - float(arr.min())) / 256
        assert abs(actual - expected) <= bin_width * 1e-3, (
            f"{name}: ours {actual} vs skimage {expected} "
            f"(should be the same bin; bin width {bin_width})"
        )

    @staticmethod
    def _histc_otsu(flat: torch.Tensor, bins: int = 256) -> float:
        """The pre-subsampling implementation (full-volume ``torch.histc``)."""
        lo, hi = float(flat.min()), float(flat.max())
        counts = torch.histc(flat, bins=bins, min=lo, max=hi)
        edges = torch.linspace(lo, hi, bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0
        w1 = torch.cumsum(counts, 0)
        w2 = torch.flip(torch.cumsum(torch.flip(counts, [0]), 0), [0])
        cw = counts * centres
        m1 = torch.cumsum(cw, 0) / w1.clamp(min=1e-12)
        m2 = torch.flip(torch.cumsum(torch.flip(cw, [0]), 0), [0]) / w2.clamp(min=1e-12)
        var = torch.nan_to_num(w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2)
        return float(centres[int(torch.argmax(var))])

    @pytest.mark.parametrize("n", [50_000, 12_000_000])
    def test_matches_full_histc_implementation(self, n: int) -> None:
        """Bincount counting is exact below the cap; subsampling stays in-bin."""
        from luxar.gsplats import metrics

        gen = torch.Generator().manual_seed(1)
        background = torch.rand(n, generator=gen) * 0.2
        signal = 0.5 + torch.rand(n // 20, generator=gen) * 0.5
        flat = torch.cat([background, signal])
        flat = flat[torch.randperm(flat.numel(), generator=gen)]
        expected = self._histc_otsu(flat)
        actual = otsu_threshold(flat)
        bin_width = float(flat.max() - flat.min()) / 256
        if flat.numel() <= metrics._OTSU_MAX_SAMPLES:
            assert abs(actual - expected) <= bin_width * 1e-3
        else:
            assert abs(actual - expected) <= bin_width * 1.001

    def test_subsample_does_not_alias_with_the_fastest_axis(self) -> None:
        """Stride == X used to read only column 0 of every row (0.0023 vs 0.199)."""
        from luxar.gsplats import metrics
        from luxar.gsplats.calibration.content import _otsu_threshold

        x = 4
        rows = metrics._OTSU_MAX_SAMPLES  # numel = cap * X -> old stride == X
        gen = torch.Generator().manual_seed(2)
        vol = torch.zeros((rows, x))
        signal = torch.rand((rows, x - 1), generator=gen)
        vol[:, 1:] = torch.where(signal < 0.5, 0.1 * signal, 0.6 + 0.4 * signal)
        expected = self._histc_otsu(vol.reshape(-1))
        bin_width = float(vol.max() - vol.min()) / 256
        assert abs(otsu_threshold(vol) - expected) <= bin_width * 1.001
        assert abs(_otsu_threshold(vol.numpy()) - expected) <= bin_width * 1.001
        assert expected > 0.04  # the old strided read collapsed to 0.0

    def test_counts_are_bounded_cpu_samples(self) -> None:
        """The histogram is counted on CPU over a bounded strided sample."""
        from luxar.gsplats import metrics

        flat = torch.rand(metrics._OTSU_MAX_SAMPLES * 3)
        counts = metrics._otsu_bin_counts(
            flat[::3], float(flat.min()), float(flat.max()), 256
        )
        assert counts.device.type == "cpu"
        assert int(counts.sum()) == metrics._OTSU_MAX_SAMPLES

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_large_volume_does_not_fail(self) -> None:
        """>16 Mvox used to raise "64-bit indexing" from MPS histc."""
        vol = torch.zeros(20_000_000, device="mps")
        vol[::50] = 1.0
        assert 0.0 <= otsu_threshold(vol) < 1.0

    def test_constant_volume_selects_nothing(self) -> None:
        """A flat volume has no foreground; the strict ``>`` must yield none."""
        const = torch.full((100,), 3.0)
        assert otsu_threshold(const) == 3.0
        psnr, _, fraction = compute_foreground_psnr(const, const)
        assert fraction == 0.0
        assert psnr != psnr  # nan: undefined, not a free "perfect" score


class TestForegroundPSNR:
    def test_ignores_background_error(self) -> None:
        """Error placed only in background must not move the foreground score.

        This is the whole point of the metric, and it is the assertion that
        fails if the mask is ever dropped or inverted.
        """
        target = torch.zeros(32, 32, 32)
        target[10:20, 10:20, 10:20] = 1.0

        exact = target.clone()
        dirty = target.clone()
        dirty[0:5, 0:5, 0:5] = 0.4  # background-only corruption

        fg_exact, _, _ = compute_foreground_psnr(exact, target)
        fg_dirty, _, _ = compute_foreground_psnr(dirty, target)
        assert fg_exact == float("inf")
        assert fg_dirty == float("inf"), "background error leaked into foreground"

        # ... while the global PSNR does degrade, proving the two differ.
        assert compute_psnr(dirty, target) < float("inf")

    def test_is_harsher_than_global_on_sparse_data(self) -> None:
        """On a sparse volume the global figure flatters; foreground must not."""
        target = torch.zeros(64, 64, 64)
        target[30:34, 30:34, 30:34] = 1.0
        pred = target.clone()
        pred[30:34, 30:34, 30:34] = 0.1  # lose 90% of the signal

        global_psnr = compute_psnr(pred, target)
        fg_psnr, _, fraction = compute_foreground_psnr(pred, target)
        assert global_psnr > 30.0, "sanity: global PSNR is indeed flattering here"
        assert fg_psnr < 5.0, "foreground PSNR must expose the lost signal"
        assert fraction < 0.001

    def test_data_range_comes_from_the_whole_volume(self) -> None:
        """Using the foreground's own range would silently inflate the score.

        Foreground values here span [0.9, 1.0] but the volume spans [0, 1]; the
        result must reflect the wider reference.
        """
        target = torch.zeros(16, 16, 16)
        target[4:8, 4:8, 4:8] = torch.linspace(0.9, 1.0, 64).reshape(4, 4, 4)
        pred = target + 0.01

        fg_psnr, _, _ = compute_foreground_psnr(pred, target)
        expected = 10.0 * math.log10(1.0**2 / (0.01**2))
        assert fg_psnr == pytest.approx(expected, abs=0.2)

        narrow = compute_foreground_psnr(pred, target, data_range=0.1)[0]
        assert narrow < fg_psnr - 15.0, "the two conventions must be far apart"

    def test_foreground_defined_on_target_not_prediction(self) -> None:
        """A fit that invents signal is scored where the signal really is."""
        target = torch.zeros(16, 16, 16)
        target[2:5, 2:5, 2:5] = 1.0
        hallucinating = target.clone()
        hallucinating[10:14, 10:14, 10:14] = 1.0  # signal that is not there

        _, _, fraction = compute_foreground_psnr(hallucinating, target)
        real_fraction = float((target > 0).float().mean())
        assert fraction == pytest.approx(real_fraction, abs=1e-6)

    def test_explicit_threshold_overrides_otsu(self) -> None:
        target = torch.zeros(10, 10, 10)
        target[0:5] = 0.5
        target[5:] = 1.0
        # A threshold above 0.5 must halve the foreground.
        _, thr, fraction = compute_foreground_psnr(target, target, threshold=0.75)
        assert thr == 0.75
        assert fraction == pytest.approx(0.5)

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_foreground_psnr(torch.zeros(4, 4), torch.zeros(4, 5))

    def test_aggregate_reports_the_trio(self) -> None:
        """compute_quality_metrics must carry all three, or the stamp is partial."""
        target = torch.zeros(16, 16, 16)
        target[4:8, 4:8, 4:8] = 1.0
        pred = target * 0.8

        metrics = compute_quality_metrics(pred, target)
        for key in (
            "foreground_psnr_db",
            "foreground_threshold",
            "foreground_fraction",
        ):
            assert key in metrics, f"{key} missing from the aggregate"

        standalone = compute_foreground_psnr(pred, target)[0]
        assert metrics["foreground_psnr_db"] == pytest.approx(standalone)
        assert metrics["foreground_psnr_db"] < metrics["psnr_db"]


class TestChunkedForegroundPSNRParity:
    """The progressive fitter has its own chunked copy; the two must agree.

    ``fit_progressive_gsplats`` cannot hold two copies of a large volume on the
    GPU, so it reimplements the metric chunk-wise. Two implementations of one
    published number is exactly the kind of pair that drifts apart silently,
    so the equivalence is pinned rather than assumed.
    """

    @pytest.mark.parametrize("chunk_voxels", [7, 1000, 10_000_000])
    def test_matches_the_direct_implementation(self, chunk_voxels: int) -> None:
        import numpy as np

        from luxar.gsplats.fit_progressive_gsplats import (
            _compute_foreground_psnr_chunked,
        )

        rng = np.random.default_rng(1)
        target = np.zeros((16, 16, 16), np.float32)
        target[4:10, 4:10, 4:10] = rng.uniform(0.5, 1.0, (6, 6, 6)).astype(np.float32)
        pred = (target + rng.normal(0, 0.05, target.shape)).astype(np.float32)

        direct, thr_d, frac_d = compute_foreground_psnr(
            torch.from_numpy(pred), torch.from_numpy(target)
        )
        chunked, thr_c, frac_c = _compute_foreground_psnr_chunked(
            torch.from_numpy(pred), target, chunk_voxels=chunk_voxels
        )

        # A chunk size of 7 forces many partial batches, so this also covers the
        # accumulation being over the right denominator (foreground count, not
        # total voxels) in every chunk.
        assert chunked == pytest.approx(direct, abs=1e-4)
        assert thr_c == pytest.approx(thr_d)
        assert frac_c == pytest.approx(frac_d)

    def test_empty_foreground_is_nan_not_zero(self) -> None:
        import numpy as np

        from luxar.gsplats.fit_progressive_gsplats import (
            _compute_foreground_psnr_chunked,
        )

        flat = np.full((8, 8, 8), 2.0, np.float32)
        psnr, _, fraction = _compute_foreground_psnr_chunked(
            torch.from_numpy(flat), flat
        )
        assert fraction == 0.0
        assert psnr != psnr  # nan


class TestSeparableSSIM:
    """`_ssim_nd` filters separably; it must equal the n-D outer-product kernel."""

    @staticmethod
    def _dense_reference(
        a: torch.Tensor, b: torch.Tensor, window_size: int, data_range: float
    ) -> float:
        import torch.nn.functional as F

        from luxar.gsplats.metrics import _gaussian_kernel_1d

        k = _gaussian_kernel_1d(window_size, 1.5, a.device).double()
        kernel = k
        for _ in range(a.ndim - 1):
            kernel = kernel.unsqueeze(-1) * k
        kernel = kernel[None, None]
        conv = F.conv2d if a.ndim == 2 else F.conv3d
        p, t = a.double()[None, None], b.double()[None, None]
        mu_p, mu_t = conv(p, kernel), conv(t, kernel)
        s_pp = conv(p * p, kernel) - mu_p**2
        s_tt = conv(t * t, kernel) - mu_t**2
        s_pt = conv(p * t, kernel) - mu_p * mu_t
        c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        ssim = ((2 * mu_p * mu_t + c1) * (2 * s_pt + c2)) / (
            (mu_p**2 + mu_t**2 + c1) * (s_pp + s_tt + c2)
        )
        return float(ssim.mean())

    @pytest.mark.parametrize("shape", [(48, 41), (20, 26, 23)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("window_size", [7, 11])
    def test_matches_dense_kernel(
        self, shape: tuple[int, ...], dtype: torch.dtype, window_size: int
    ) -> None:
        g = torch.Generator().manual_seed(1)
        a = torch.rand(shape, generator=g, dtype=dtype)
        b = (a + 0.1 * torch.randn(shape, generator=g, dtype=dtype)).clamp(0, 1)
        s, n = _ssim_nd(a, b, window_size=window_size, data_range=1.0)
        ref = self._dense_reference(a, b, window_size, 1.0)
        assert s / n == pytest.approx(ref, abs=1e-6)

    def test_4d_averages_3d_subvolumes(self) -> None:
        g = torch.Generator().manual_seed(2)
        a = torch.rand(3, 16, 18, 17, generator=g)
        b = (a + 0.05 * torch.randn(3, 16, 18, 17, generator=g)).clamp(0, 1)
        refs = [self._dense_reference(a[i], b[i], 11, 1.0) for i in range(3)]
        assert compute_ssim(a, b, data_range=1.0) == pytest.approx(
            sum(refs) / 3, abs=1e-6
        )

    def test_no_im2col_blowup_on_cpu(self) -> None:
        """A 96^3 CPU SSIM stays near a few volumes of memory and is fast.

        The former conv3d path lowered the 11^3 kernel to im2col (~5 KB per
        output voxel: ~4.6 GB here, 37 GB at 200^3); the separable filter
        holds only a handful of float32 volumes.
        """
        import sys
        import time

        resource = pytest.importorskip("resource")
        # ru_maxrss is a high-water mark: KiB on Linux, bytes on macOS.
        unit = 1 if sys.platform == "darwin" else 1024
        a = torch.rand(96, 96, 96)
        b = a + 0.01
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
        t0 = time.perf_counter()
        _ssim_nd(a, b, window_size=11, data_range=1.0)
        elapsed = time.perf_counter() - t0
        grown = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit - before
        assert elapsed < 10.0
        assert grown < 500 * 1024**2


# ---------------------------------------------------------------------------
# Robust peak, streamed sums, foreground SSIM
# ---------------------------------------------------------------------------


def _blobs(shape=(24, 32, 32), seed: int = 3) -> torch.Tensor:
    import numpy as np

    rng = np.random.default_rng(seed)
    v = np.zeros(shape, np.float32)
    for _ in range(12):
        z, y, x = (rng.integers(3, s - 3) for s in shape)
        v[z - 2 : z + 3, y - 2 : y + 3, x - 2 : x + 3] += rng.uniform(50, 100)
    return torch.from_numpy(v)


class TestRobustPeak:
    def test_matches_numpy_nearest_rank_quantile(self) -> None:
        import numpy as np

        from luxar.gsplats.metrics import robust_peak

        values = torch.from_numpy(
            np.random.default_rng(0).gamma(2.0, 3.0, 50_000).astype(np.float32)
        )
        expected = np.quantile(values.numpy(), 0.9999, method="inverted_cdf")
        assert robust_peak(values) == pytest.approx(float(expected))
        assert robust_peak(values, 0.5) == pytest.approx(
            float(np.quantile(values.numpy(), 0.5, method="inverted_cdf"))
        )

    def test_streams_across_chunks_exactly(self, monkeypatch) -> None:
        """A running top-k over chunks must give the one-shot answer."""
        import luxar.gsplats.metrics as metrics_mod

        target = _blobs()
        one_shot = metrics_mod.robust_peak(target, 0.999)
        monkeypatch.setattr(metrics_mod, "_STREAM_CHUNK_VOXELS", 997)
        assert metrics_mod.robust_peak(target, 0.999) == one_shot


class TestRobustPSNR:
    def test_a_hot_pixel_moves_full_range_psnr_but_not_the_robust_one(self) -> None:
        """The headline must not be decided by one cosmic ray."""
        target = _blobs()
        rng = torch.Generator().manual_seed(0)
        pred = target + torch.randn(target.shape, generator=rng) * 2.0
        clean = compute_quality_metrics(pred, target)

        hot_target = target.clone()
        hot_target[0, 0, 0] = 50_000.0
        hot_pred = pred.clone()
        hot_pred[0, 0, 0] = 50_000.0  # reproduced exactly: the error is unchanged
        hot = compute_quality_metrics(hot_pred, hot_target)

        # One voxel's noise of 24,576 left the error sum: ~4e-5 relative.
        assert hot["mse"] == pytest.approx(clean["mse"], rel=1e-3)
        assert hot["robust_data_range"] == pytest.approx(clean["robust_data_range"])
        assert hot["psnr_robust_db"] == pytest.approx(clean["psnr_robust_db"], abs=1e-2)
        # Full range jumps by ~20 log10(50000 / 300): the hot pixel IS the peak.
        assert hot["psnr_db"] - clean["psnr_db"] > 40.0

    def test_robust_peak_is_quantile_minus_minimum(self) -> None:
        from luxar.gsplats.metrics import robust_peak

        target = _blobs() + 7.0  # a floor the peak is measured from
        m = compute_quality_metrics(target * 0.98, target)
        assert m["robust_data_range"] == pytest.approx(robust_peak(target) - 7.0)
        assert m["data_range"] == pytest.approx(float(target.max() - 7.0))
        mse = float(((target * 0.98 - target) ** 2).mean())
        assert m["psnr_robust_db"] == pytest.approx(
            10 * math.log10(m["robust_data_range"] ** 2 / mse), abs=1e-4
        )

    def test_degenerate_robust_range_falls_back_to_full(self) -> None:
        """>99.99% of voxels at the minimum: no robust peak to speak of."""
        target = torch.zeros(40, 40, 40)
        target[5, 5, 5] = 10.0
        m = compute_quality_metrics(target * 0.5, target)
        assert m["robust_data_range"] == m["data_range"] == 10.0


class TestStreamedSums:
    def test_chunked_reductions_match_whole_volume_ones(self, monkeypatch) -> None:
        import luxar.gsplats.metrics as metrics_mod

        target = _blobs()
        pred = target * 0.9 + 1.5
        monkeypatch.setattr(metrics_mod, "_STREAM_CHUNK_VOXELS", 1013)
        # Plain Otsu on the target, so the foreground trio is comparable with
        # the direct `compute_foreground_psnr`.
        m = metrics_mod.compute_quality_metrics(
            pred,
            target,
            foreground_field=target,
            foreground_threshold=otsu_threshold(target),
        )
        diff = (pred - target).double()
        assert m["mse"] == pytest.approx(float((diff**2).mean()), rel=1e-5)
        assert m["max_abs_error"] == pytest.approx(float(diff.abs().max()), rel=1e-6)
        rel = float(diff.norm() / target.double().norm())
        assert m["rel_l2"] == pytest.approx(rel, rel=1e-5)
        fg_psnr, thr, frac = compute_foreground_psnr(pred, target)
        assert m["foreground_psnr_db"] == pytest.approx(fg_psnr, abs=1e-3)
        assert m["foreground_threshold"] == pytest.approx(thr)
        assert m["foreground_fraction"] == pytest.approx(frac)


class TestForegroundSSIM:
    def test_identical_volumes_score_one_everywhere(self) -> None:
        target = _blobs()
        m = compute_quality_metrics(target.clone(), target)
        assert m["foreground_ssim"] == pytest.approx(1.0, abs=1e-5)

    def test_foreground_ssim_is_not_diluted_by_matching_background(self) -> None:
        """Noise on the signal only: the empty majority flatters global SSIM."""
        target = _blobs()
        noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(1))
        pred = target + (target > 0).float() * noise * 15.0
        m = compute_quality_metrics(pred, target)
        assert m["foreground_ssim"] < m["ssim"] - 0.01

    def test_explicit_field_and_threshold_define_the_foreground(self) -> None:
        target = _blobs()
        mask = torch.zeros_like(target)
        mask[:, :16, :] = 1.0
        m = compute_quality_metrics(
            target * 0.9, target, foreground_field=mask, foreground_threshold=0.0
        )
        assert m["foreground_fraction"] == pytest.approx(0.5)
        assert m["foreground_threshold"] == 0.0

    def test_tiled_foreground_ssim_matches_untiled(self) -> None:
        from luxar.gsplats.metrics import _ssim_sums, _ssim_sums_tiled

        target = _blobs((40, 44, 36))
        pred = target * 0.8 + 3.0
        dr = float(target.max() - target.min())
        thr = 10.0
        direct = _ssim_sums(pred, target, 11, dr, target, thr)
        tiled = _ssim_sums_tiled(pred, target, 11, dr, 20, target, thr)
        assert tiled[1] == direct[1] and tiled[3] == direct[3]
        assert tiled[0] == pytest.approx(direct[0], rel=1e-5)
        assert tiled[2] == pytest.approx(direct[2], rel=1e-5)

    def test_field_shape_mismatch_rejected(self) -> None:
        target = _blobs()
        with pytest.raises(ValueError, match="foreground field"):
            compute_quality_metrics(
                target, target, foreground_field=torch.zeros(3, 3, 3)
            )


class TestSharedForegroundDefinition:
    def test_torch_tent_blur_matches_the_scipy_one(self) -> None:
        import numpy as np

        from luxar.gsplats.metrics import _tent_blur
        from luxar.gsplats.seeds.utils import soft_blur_nd

        v = np.random.default_rng(0).random((7, 1, 9, 5)).astype(np.float32)
        t = torch.from_numpy(v)
        np.testing.assert_allclose(
            _tent_blur(t).numpy(), soft_blur_nd(v), rtol=1e-6, atol=1e-6
        )
        assert torch.equal(t, torch.from_numpy(v))  # input untouched

    def test_a_hot_pixel_does_not_empty_the_default_foreground(self) -> None:
        """Uncapped Otsu put every blob below the threshold (0.00% foreground)."""
        target = _blobs()
        clean = compute_quality_metrics(target * 0.9, target)
        hot = target.clone()
        hot[0, 0, 0] = 1e6
        spiked = compute_quality_metrics(hot * 0.9, hot)
        assert clean["foreground_fraction"] > 0.01
        # Within a few percent: the spike's own blurred footprint is foreground,
        # and the capped histogram shifts by one order statistic.
        assert spiked["foreground_fraction"] == pytest.approx(
            clean["foreground_fraction"], rel=0.05
        )

    def test_empty_foreground_has_nan_ssim(self) -> None:
        target = _blobs()
        m = compute_quality_metrics(
            target,
            target,
            foreground_field=torch.zeros_like(target),
            foreground_threshold=0.5,
        )
        assert m["foreground_fraction"] == 0.0
        assert math.isnan(m["foreground_ssim"])
        flat = torch.full((12, 12, 12), 3.0)
        assert math.isnan(compute_quality_metrics(flat, flat)["foreground_ssim"])

    def test_integer_targets_are_scored_as_float(self) -> None:
        target = (_blobs() * 10).to(torch.int32)
        pred = target.to(torch.float32) + 0.5
        m = compute_quality_metrics(pred, target)
        assert m["mse"] == pytest.approx(0.25)
        assert m["max_abs_error"] == pytest.approx(0.5)


def test_progressive_foreground_matches_shared_hot_pixel_rule() -> None:
    from luxar.gsplats.fit_progressive_gsplats import _compute_foreground_psnr_chunked
    from luxar.gsplats.metrics import compute_quality_metrics

    rng = np.random.default_rng(5)
    target = rng.normal(0.0, 0.02, (24, 32, 32)).astype(np.float32)
    target[4:12, 4:12, 4:12] += 0.8
    target[0, 0, 0] = 1000.0
    rendered = torch.from_numpy(target * 0.95)
    expected = compute_quality_metrics(rendered, torch.from_numpy(target))
    actual = _compute_foreground_psnr_chunked(rendered, target, chunk_voxels=1000)

    assert actual[0] == pytest.approx(expected["foreground_psnr_db"], rel=1e-5)
    assert actual[1] == pytest.approx(expected["foreground_psnr_robust_db"], rel=1e-5)
    assert actual[2] == pytest.approx(expected["foreground_threshold"], rel=1e-5)
    assert actual[3] == pytest.approx(
        expected["foreground_fraction"], abs=1 / target.size
    )
