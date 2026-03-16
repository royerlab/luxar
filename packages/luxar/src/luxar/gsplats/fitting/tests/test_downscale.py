"""Tests for volume downscaling utilities."""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.fitting.downscale import (
    downscale_volume,
    normalize_downscale,
    rescale_centers,
    rescale_cholesky_packed,
)

# --- normalize_downscale ---


class TestNormalizeDownscale:
    def test_none_returns_none(self):
        assert normalize_downscale(None, 3) is None

    def test_int_broadcast(self):
        assert normalize_downscale(4, 3) == (4, 4, 4)

    def test_int_broadcast_2d(self):
        assert normalize_downscale(2, 2) == (2, 2)

    def test_tuple_passthrough(self):
        assert normalize_downscale((1, 4, 4), 3) == (1, 4, 4)

    def test_list_passthrough(self):
        assert normalize_downscale([2, 3], 2) == (2, 3)

    def test_all_ones_returns_none(self):
        """All-ones means no-op, should return None."""
        assert normalize_downscale(1, 3) is None
        assert normalize_downscale((1, 1, 1), 3) is None

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="2 elements but volume has 3"):
            normalize_downscale((2, 4), 3)

    def test_negative_factor_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            normalize_downscale(-1, 2)

    def test_zero_factor_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            normalize_downscale(0, 2)

    def test_numpy_int(self):
        """Should accept numpy integer types."""
        assert normalize_downscale(np.int64(4), 3) == (4, 4, 4)


# --- downscale_volume ---


class TestDownscaleVolume:
    def test_shape_isotropic_2d(self):
        V = np.random.rand(64, 64).astype(np.float32)
        result = downscale_volume(V, (2, 2))
        assert result.shape == (32, 32)

    def test_shape_isotropic_3d(self):
        V = np.random.rand(32, 32, 32).astype(np.float32)
        result = downscale_volume(V, (4, 4, 4))
        assert result.shape == (8, 8, 8)

    def test_shape_anisotropic(self):
        V = np.random.rand(64, 128, 128).astype(np.float32)
        result = downscale_volume(V, (1, 4, 4))
        assert result.shape == (64, 32, 32)

    def test_preserves_low_frequency(self):
        """Downscaling should preserve smooth (low-frequency) content."""
        # Create a smooth 2D sine wave
        x = np.linspace(0, 2 * np.pi, 128)
        y = np.linspace(0, 2 * np.pi, 128)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        V = np.sin(xx) * np.cos(yy)
        V = V.astype(np.float32)

        V_down = downscale_volume(V, (4, 4))

        # Reconstruct reference at downscaled resolution
        x_down = np.linspace(0, 2 * np.pi, 32)
        y_down = np.linspace(0, 2 * np.pi, 32)
        xx_d, yy_d = np.meshgrid(x_down, y_down, indexing="ij")
        V_ref = np.sin(xx_d) * np.cos(yy_d)

        # Correlation should be very high (>0.95)
        corr = np.corrcoef(V_down.ravel(), V_ref.ravel())[0, 1]
        assert corr > 0.95, f"Low-frequency content not preserved: corr={corr:.3f}"

    def test_no_blur_for_factor_one(self):
        """Axes with factor=1 should not be blurred."""
        V = np.random.rand(8, 8).astype(np.float32)
        result = downscale_volume(V, (1, 2))
        assert result.shape == (8, 4)

    def test_output_dtype_float32(self):
        V = np.random.rand(16, 16).astype(np.float64)
        result = downscale_volume(V, (2, 2))
        assert result.dtype == np.float32


# --- rescale_centers ---


class TestRescaleCenters:
    def test_isotropic(self):
        centers = np.array([[5.0, 10.0, 15.0]], dtype=np.float32)
        result = rescale_centers(centers, (4, 4, 4))
        np.testing.assert_array_equal(result, [[20.0, 40.0, 60.0]])

    def test_anisotropic(self):
        centers = np.array([[5.0, 10.0, 15.0]], dtype=np.float32)
        result = rescale_centers(centers, (1, 2, 4))
        np.testing.assert_array_equal(result, [[5.0, 20.0, 60.0]])

    def test_batch(self):
        centers = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result = rescale_centers(centers, (2, 3))
        expected = np.array([[2.0, 6.0], [6.0, 12.0], [10.0, 18.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_empty(self):
        centers = np.zeros((0, 3), dtype=np.float32)
        result = rescale_centers(centers, (2, 2, 2))
        assert result.shape == (0, 3)


# --- rescale_cholesky_packed ---


class TestRescaleCholeskyPacked:
    def test_2d(self):
        """2D: packed = [L00, L10, L11] → scales by [f0, f1, f1]."""
        packed = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = rescale_cholesky_packed(packed, (2, 4))
        # L00 * 2, L10 * 4, L11 * 4
        np.testing.assert_array_equal(result, [[2.0, 8.0, 12.0]])

    def test_3d(self):
        """3D: packed = [L00, L10, L11, L20, L21, L22] → scales by [f0, f1, f1, f2, f2, f2]."""
        packed = np.ones((1, 6), dtype=np.float32)
        result = rescale_cholesky_packed(packed, (1, 2, 3))
        expected = np.array([[1.0, 2.0, 2.0, 3.0, 3.0, 3.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_batch(self):
        packed = np.ones((5, 3), dtype=np.float32)
        result = rescale_cholesky_packed(packed, (3, 3))
        assert result.shape == (5, 3)
        np.testing.assert_array_equal(result, np.full((5, 3), 3.0))

    def test_empty(self):
        packed = np.zeros((0, 6), dtype=np.float32)
        result = rescale_cholesky_packed(packed, (2, 2, 2))
        assert result.shape == (0, 6)


# --- Integration: explicit seeds with downscale ---


class TestExplicitSeedsWithDownscale:
    """Verify that explicit seed centers are rescaled to downscaled coords."""

    def test_explicit_seeds_rescaled(self):
        """Seed centers in original coords should be divided by downscale factors."""
        import torch

        from luxar.gsplats.fitting.config import FitConfig
        from luxar.gsplats.fitting.preprocessing import preprocess_data

        V = np.random.RandomState(42).rand(32, 32).astype(np.float32)
        # Seed at center of original volume
        seeds = np.array([[16.0, 16.0]], dtype=np.float32)

        config = FitConfig(
            V=V,
            seeds=seeds,
            norm_percentile=0.0,
            init_sigma_vox=None,
            sigma_min_diag=[0.289, 0.289],
            sigma_max_diag=None,
            truncate=3.0,
            n_iters=10,
            lr=0.01,
            max_abs_error=None,
            rel_l2_target=None,
            gradient_clip=1.0,
            loss_type="l1",
            asymmetric_penalty=10.0,
            l1_amp=None,
            l1_diag=None,
            l1_sharpness=None,
            scheduler_type="plateau",
            patience=25,
            lr_reduction_factor=0.98,
            early_stop_patience=300,
            enable_dynamic_ops=False,
            dynamic_config=None,
            dynamic_ops_verbose=False,
            napari_movie=False,
            movie_every=1,
            movie_max_frames=None,
            device=torch.device("cpu"),
            verbose=False,
            downscale=(2, 2),
        )

        result = preprocess_data(config)
        # Seed at (16, 16) in original -> (8, 8) in downscaled 2x volume
        np.testing.assert_allclose(result.seed_centers, [[8.0, 8.0]], atol=0.01)
        # Volume should be downscaled
        assert result.V_tensor.shape == (16, 16)
        # Downscale factors should be stored
        assert result.downscale_factors == (2, 2)
