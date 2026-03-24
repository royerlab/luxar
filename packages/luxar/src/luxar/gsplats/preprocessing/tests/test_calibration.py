"""Tests for Noise2Self J-invariant NLM calibration."""

from __future__ import annotations

import torch

from luxar.gsplats.preprocessing import calibrate_nlm_h, denoise_nlm

from .conftest import psnr


class TestCalibrationBasic:
    """Basic calibration behaviour."""

    def test_returns_float(self, noisy_2d):
        _, noisy = noisy_2d
        h = calibrate_nlm_h(
            noisy,
            h_range=[0.01, 0.03, 0.05],
            patch_size=3,
            search_distance=2,
            backend="pytorch",
        )
        assert isinstance(h, float)
        assert h > 0

    def test_selects_from_h_range(self, noisy_2d):
        _, noisy = noisy_2d
        h_range = [0.01, 0.03, 0.05, 0.08]
        h = calibrate_nlm_h(
            noisy,
            h_range=h_range,
            patch_size=3,
            search_distance=2,
            backend="pytorch",
        )
        assert h in h_range

    def test_deterministic(self, noisy_2d):
        """Running calibration twice gives the same result."""
        _, noisy = noisy_2d
        kwargs = dict(
            h_range=[0.01, 0.03, 0.05],
            patch_size=3,
            search_distance=2,
            backend="pytorch",
        )
        h1 = calibrate_nlm_h(noisy, **kwargs)
        h2 = calibrate_nlm_h(noisy, **kwargs)
        assert h1 == h2


class TestCalibrationQuality:
    """Calibrated h should produce good denoising."""

    def test_calibrated_h_improves_psnr(self, noisy_2d):
        """Calibrated h should denoise better than extreme h values."""
        clean, noisy = noisy_2d
        h_opt = calibrate_nlm_h(
            noisy,
            h_range=[0.01, 0.02, 0.03, 0.05, 0.08],
            patch_size=3,
            search_distance=2,
            backend="pytorch",
        )

        denoised_opt = denoise_nlm(
            noisy, h=h_opt, patch_size=3, search_distance=2, backend="pytorch"
        )
        psnr_opt = psnr(clean, denoised_opt)

        # Very small h (under-smoothing) should be worse
        denoised_small = denoise_nlm(
            noisy, h=0.001, patch_size=3, search_distance=2, backend="pytorch"
        )
        psnr_small = psnr(clean, denoised_small)

        assert psnr_opt >= psnr_small, (
            f"Calibrated h={h_opt} gave PSNR {psnr_opt:.2f} dB, "
            f"but h=0.001 gave {psnr_small:.2f} dB"
        )


class TestCalibration3D:
    """Calibration with 3D volumes."""

    def test_3d_with_2d_slice(self, noisy_3d):
        """Calibrating on a 2D slice of a 3D volume."""
        _, noisy = noisy_3d
        h = calibrate_nlm_h(
            noisy,
            h_range=[0.01, 0.03, 0.05],
            patch_size=3,
            search_distance=2,
            use_2d_slice=True,
            backend="pytorch",
        )
        assert isinstance(h, float)
        assert h > 0

    def test_3d_slice_index(self, noisy_3d):
        """Custom slice index should work."""
        _, noisy = noisy_3d
        h = calibrate_nlm_h(
            noisy,
            h_range=[0.01, 0.03, 0.05],
            patch_size=3,
            search_distance=2,
            use_2d_slice=True,
            slice_index=0,
            backend="pytorch",
        )
        assert isinstance(h, float)

    def test_3d_full_volume_calibration(self):
        """Calibrate on the full 3D volume (slower, use small data)."""
        clean = torch.zeros(8, 8, 8)
        # Simple blob
        clean[2:6, 2:6, 2:6] = 1.0
        noisy = clean + torch.randn_like(clean) * 0.1

        h = calibrate_nlm_h(
            noisy,
            h_range=[0.05, 0.10, 0.15],
            patch_size=3,
            search_distance=2,
            use_2d_slice=False,
            backend="pytorch",
        )
        assert isinstance(h, float)
        assert h > 0
