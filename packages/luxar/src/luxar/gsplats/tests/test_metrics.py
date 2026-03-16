"""Tests for quality metrics (PSNR, SSIM, MSE, rel_l2, max_abs_error)."""

from __future__ import annotations

import pytest
import torch

from luxar.gsplats.metrics import (
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
        torch.manual_seed(42)
        a = torch.rand(32, 32, 32, device=device)
        b = a + 0.05 * torch.randn_like(a)
        m = compute_quality_metrics(a, b)
        assert m["mse"] > 0
        assert m["psnr_db"] > 0
        assert 0.0 < m["ssim"] < 1.0
        assert m["rel_l2"] > 0
        assert m["max_abs_error"] > 0

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
