"""Tests for quality metrics (PSNR, SSIM, MSE, rel_l2, max_abs_error)."""

from __future__ import annotations

import pytest
import torch

from luxar.gsplats.metrics import (
    _should_tile_ssim,
    _ssim_nd,
    _ssim_nd_tiled,
    compute_psnr,
    compute_quality_metrics,
    compute_ssim,
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
        expected_keys = {"mse", "psnr_db", "ssim", "rel_l2", "max_abs_error"}
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
    def test_cpu_never_tiles(self) -> None:
        assert _should_tile_ssim((1000, 1000, 1000), torch.device("cpu")) is False

    def test_small_volume_no_tile(self, device: torch.device) -> None:
        # 32^3 * 4 bytes * 8 ≈ 1 MB — should never exceed 50% GPU memory
        assert _should_tile_ssim((32, 32, 32), device) is False
