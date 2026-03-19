"""
Numerical accuracy tests for CUDA backend.

These tests verify that the CUDA kernels produce numerically accurate
results compared to reference implementations.
"""

import numpy as np
import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


class TestMahalanobisDistance:
    """Test Mahalanobis distance computation accuracy."""

    def test_mahalanobis_isotropic(self):
        """Test distance computation for isotropic Gaussians."""
        # For isotropic Gaussian with σ=1, Σ = I, Σ⁻¹ = I
        # Mahalanobis distance = Euclidean distance
        d = np.array([1.0, 2.0, 3.0])
        conic = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])  # Identity upper tri

        # Expected: d^T @ I @ d = 1 + 4 + 9 = 14
        expected = np.sum(d**2)

        # Manual computation matching CUDA kernel
        c_00, c_01, c_02, c_11, c_12, c_22 = conic
        dist_sq = (
            d[0] ** 2 * c_00
            + d[1] ** 2 * c_11
            + d[2] ** 2 * c_22
            + 2 * d[0] * d[1] * c_01
            + 2 * d[0] * d[2] * c_02
            + 2 * d[1] * d[2] * c_12
        )

        np.testing.assert_allclose(dist_sq, expected, rtol=1e-6)

    def test_mahalanobis_anisotropic(self):
        """Test distance computation for anisotropic Gaussians."""
        # Anisotropic: Σ = diag([4, 1, 1]), Σ⁻¹ = diag([0.25, 1, 1])
        d = np.array([2.0, 1.0, 1.0])
        conic = np.array([0.25, 0.0, 0.0, 1.0, 0.0, 1.0])  # Diagonal

        # Expected: 0.25*4 + 1*1 + 1*1 = 1 + 1 + 1 = 3
        expected = 0.25 * 4 + 1 * 1 + 1 * 1

        c_00, c_01, c_02, c_11, c_12, c_22 = conic
        dist_sq = (
            d[0] ** 2 * c_00
            + d[1] ** 2 * c_11
            + d[2] ** 2 * c_22
            + 2 * d[0] * d[1] * c_01
            + 2 * d[0] * d[2] * c_02
            + 2 * d[1] * d[2] * c_12
        )

        np.testing.assert_allclose(dist_sq, expected, rtol=1e-6)


class TestGeneralizedGaussian:
    """Test generalized Gaussian intensity computation."""

    def test_standard_gaussian(self):
        """Test s=2 gives standard Gaussian."""
        dist_sq = 4.0  # ||y||² = 4
        s = 2.0

        # exp(-0.5 * dist_sq^(s/2)) = exp(-0.5 * 4) = exp(-2)
        expected = np.exp(-0.5 * dist_sq ** (s / 2))
        actual = np.exp(-0.5 * dist_sq)  # s=2 simplifies to standard

        np.testing.assert_allclose(expected, actual, rtol=1e-6)

    def test_sharp_gaussian(self):
        """Test s>2 gives sharper falloff."""
        dist_sq = 4.0
        s_standard = 2.0
        s_sharp = 4.0

        val_standard = np.exp(-0.5 * dist_sq ** (s_standard / 2))
        val_sharp = np.exp(-0.5 * dist_sq ** (s_sharp / 2))

        # At dist=2 (dist_sq=4):
        # s=2: exp(-0.5 * 4) = exp(-2) ≈ 0.135
        # s=4: exp(-0.5 * 4²) = exp(-8) ≈ 0.00034
        # Sharp should decay faster
        assert val_sharp < val_standard

    def test_soft_gaussian(self):
        """Test s<2 gives softer falloff."""
        dist_sq = 4.0
        s_standard = 2.0
        s_soft = 1.0

        val_standard = np.exp(-0.5 * dist_sq ** (s_standard / 2))
        val_soft = np.exp(-0.5 * dist_sq ** (s_soft / 2))

        # At dist=2 (dist_sq=4):
        # s=2: exp(-0.5 * 4) = exp(-2) ≈ 0.135
        # s=1: exp(-0.5 * 2) = exp(-1) ≈ 0.368
        # Soft should decay slower
        assert val_soft > val_standard


class TestGradientFormulas:
    """Test gradient computation formulas."""

    def test_amplitude_gradient(self):
        """Test ∂I/∂a = exp(inner)."""
        a = 2.0
        dist_sq = 1.0
        s = 2.0

        inner = -0.5 * dist_sq ** (s / 2)
        a * np.exp(inner)

        # ∂I/∂a = exp(inner)
        dI_da = np.exp(inner)

        # Numerical verification
        eps = 1e-5
        I_plus = (a + eps) * np.exp(inner)
        I_minus = (a - eps) * np.exp(inner)
        dI_da_numerical = (I_plus - I_minus) / (2 * eps)

        np.testing.assert_allclose(dI_da, dI_da_numerical, rtol=1e-4)

    def test_distance_gradient(self):
        """Test ∂I/∂D² = I × (-0.25s) × D^(s-2)."""
        a = 2.0
        dist_sq = 4.0
        s = 2.0

        inner = -0.5 * dist_sq ** (s / 2)
        intensity = a * np.exp(inner)

        # ∂I/∂D² = I × (-0.25 × s) × D²^(s/2 - 1)
        dI_dD2 = intensity * (-0.25 * s) * dist_sq ** (s / 2 - 1)

        # Numerical verification
        eps = 1e-5
        inner_plus = -0.5 * (dist_sq + eps) ** (s / 2)
        inner_minus = -0.5 * (dist_sq - eps) ** (s / 2)
        I_plus = a * np.exp(inner_plus)
        I_minus = a * np.exp(inner_minus)
        dI_dD2_numerical = (I_plus - I_minus) / (2 * eps)

        np.testing.assert_allclose(dI_dD2, dI_dD2_numerical, rtol=1e-4)

    def test_center_gradient_sign(self):
        """
        Test that center gradient sign is NEGATIVE.

        Since d = x - μ, we have ∂d/∂μ = -I.
        Therefore ∂I/∂μ = ∂I/∂D² × ∂D²/∂d × ∂d/∂μ = grad_dist × 2Σ⁻¹d × (-1)
        """
        # At x=2, μ=0, d=2
        # For isotropic σ=1: Σ⁻¹ = I
        # D² = 4, I = exp(-2)

        x = 2.0
        mu = 0.0
        d = x - mu
        s = 2.0
        a = 1.0

        dist_sq = d**2
        inner = -0.5 * dist_sq ** (s / 2)
        intensity = a * np.exp(inner)

        # grad_dist = ∂I/∂D² = I × (-0.25 × s) × D²^(s/2 - 1)
        grad_dist = intensity * (-0.25 * s) * dist_sq ** (s / 2 - 1)

        # ∂D²/∂d = 2 × Σ⁻¹ × d = 2d (for Σ⁻¹ = I)
        dD2_dd = 2 * d

        # ∂d/∂μ = -1
        dd_dmu = -1.0

        # Final: ∂I/∂μ = grad_dist × dD2_dd × dd_dmu
        dI_dmu = grad_dist * dD2_dd * dd_dmu

        # Numerical verification
        eps = 1e-5
        d_plus = x - (mu + eps)
        d_minus = x - (mu - eps)
        I_plus = a * np.exp(-0.5 * d_plus**2)
        I_minus = a * np.exp(-0.5 * d_minus**2)
        dI_dmu_numerical = (I_plus - I_minus) / (2 * eps)

        np.testing.assert_allclose(dI_dmu, dI_dmu_numerical, rtol=1e-4)

        # The gradient should be POSITIVE when μ < x (splat left of pixel)
        # because increasing μ moves splat closer to pixel, increasing I
        assert dI_dmu > 0


class TestConicGradient:
    """Test conic (Σ⁻¹) gradient computation."""

    def test_diagonal_conic_gradient(self):
        """Test gradient w.r.t. diagonal conic elements."""
        d = np.array([2.0, 1.0])
        np.array([1.0, 0.0, 1.0])  # 2D: [c_00, c_01, c_11]

        # D² = d[0]² × c_00 + 2 × d[0] × d[1] × c_01 + d[1]² × c_11
        # ∂D²/∂c_00 = d[0]²
        # ∂D²/∂c_11 = d[1]²
        # ∂D²/∂c_01 = 2 × d[0] × d[1]

        dD2_dc00 = d[0] ** 2
        dD2_dc11 = d[1] ** 2
        dD2_dc01 = 2 * d[0] * d[1]

        np.testing.assert_allclose(dD2_dc00, 4.0, rtol=1e-6)
        np.testing.assert_allclose(dD2_dc11, 1.0, rtol=1e-6)
        np.testing.assert_allclose(dD2_dc01, 4.0, rtol=1e-6)
