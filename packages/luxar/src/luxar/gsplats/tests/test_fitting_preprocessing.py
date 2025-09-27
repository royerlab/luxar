"""
Tests for fitting preprocessing module.
"""

import numpy as np
import pytest
import torch

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

if HAS_TORCH:
    from luxar.gsplats.dynamic_ops import DynamicOpsConfig
    from luxar.gsplats.fitting.config import FitConfig
    from luxar.gsplats.fitting.preprocessing import preprocess_data


@pytest.fixture
def mock_config_2d():
    """Create a mock 2D configuration for testing."""
    V = np.random.rand(32, 32).astype(np.float32)
    centers = np.random.rand(10, 2).astype(np.float32)

    return FitConfig(
        V=V,
        centers_overcomplete=centers,
        norm_percentile=0.0,
        init_sigma_vox=1.5,
        sigma_min_diag=[0.1, 0.1],
        sigma_max_diag=None,
        truncate=3.0,
        n_iters=100,
        lr=0.01,
        max_abs_error=None,
        gradient_clip=1.0,
        loss_type="l1",
        asymmetric_penalty=10.0,
        l1_amp=0.001,
        scheduler_type="plateau",
        patience=10,
        factor=0.5,
        enable_dynamic_ops=True,
        dynamic_config=DynamicOpsConfig(),
        dynamic_ops_verbose=False,
        napari_movie=False,
        movie_every=1,
        movie_max_frames=100,
        device=torch.device("cpu"),
        verbose=False
    )


class TestPreprocessData:
    """Test data preprocessing functionality."""

    def test_basic_preprocessing(self, mock_config_2d):
        """Test basic data preprocessing."""
        result = preprocess_data(mock_config_2d)

        assert result.V_normalized.shape == mock_config_2d.V.shape
        assert result.centers_overcomplete.shape == (10, 2)
        assert result.d == 2
        assert result.N == 10
        assert result.intensity_range > 0
        assert result.effective_lr > 0

    def test_normalization_full_range(self, mock_config_2d):
        """Test full range normalization."""
        # Create data with known range
        V = np.array([[0.1, 0.5], [0.3, 0.9]], dtype=np.float32)
        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 0.0

        result = preprocess_data(mock_config_2d)

        assert result.image_min == pytest.approx(0.1, abs=1e-6)
        assert result.image_max == pytest.approx(0.9, abs=1e-6)
        assert result.intensity_range == pytest.approx(0.8, abs=1e-6)
        # Check normalization
        assert np.min(result.V_normalized) >= 0.0
        assert np.max(result.V_normalized) <= 1.0

    def test_normalization_percentile(self, mock_config_2d):
        """Test percentile-based normalization."""
        # Create data with outliers
        V = np.array([[0.0, 0.5], [0.3, 10.0]], dtype=np.float32)  # 10.0 is outlier
        mock_config_2d.V = V
        mock_config_2d.norm_percentile = 25.0  # Should ignore extremes

        result = preprocess_data(mock_config_2d)

        # Should not use the extreme values
        assert result.image_min != 0.0
        assert result.image_max != 10.0

    def test_uniform_image_handling(self, mock_config_2d):
        """Test handling of nearly uniform images."""
        # Create nearly uniform image
        V = np.full((16, 16), 0.5, dtype=np.float32)
        mock_config_2d.V = V

        result = preprocess_data(mock_config_2d)

        assert result.intensity_range == 1.0  # Should be set to avoid division by zero
        assert np.allclose(result.V_normalized, 0.5)

    def test_auto_candidate_generation(self, mock_config_2d):
        """Test automatic candidate generation."""
        mock_config_2d.centers_overcomplete = None  # Trigger auto-generation

        result = preprocess_data(mock_config_2d)

        assert result.centers_overcomplete is not None
        assert result.centers_overcomplete.shape[1] == 2  # 2D centers
        assert result.N > 0

    def test_gradient_dilution_2d(self, mock_config_2d):
        """Test gradient dilution for 2D (should be minimal)."""
        result = preprocess_data(mock_config_2d)

        # 2D should have minimal scaling
        assert result.gradient_dilution_factor == pytest.approx(1.0, rel=0.1)
        assert result.dimensional_complexity is None  # Not used for 2D
        assert result.effective_lr == result.gradient_dilution_factor * mock_config_2d.lr

    def test_gradient_dilution_4d(self, mock_config_2d):
        """Test gradient dilution for 4D (should be significant)."""
        # Create 4D configuration
        V_4d = np.random.rand(8, 8, 8, 8).astype(np.float32)
        centers_4d = np.random.rand(10, 4).astype(np.float32)

        mock_config_2d.V = V_4d
        mock_config_2d.centers_overcomplete = centers_4d
        mock_config_2d.sigma_min_diag = [0.1] * 4

        result = preprocess_data(mock_config_2d)

        assert result.d == 4
        assert result.gradient_dilution_factor > 1.0  # Should be scaled up
        assert result.dimensional_complexity is not None  # Should be set for 4D
        assert result.effective_lr > mock_config_2d.lr  # Should be boosted

    def test_convergence_threshold_auto(self, mock_config_2d):
        """Test automatic convergence threshold setting."""
        mock_config_2d.max_abs_error = None

        result = preprocess_data(mock_config_2d)

        assert result.max_abs_error == 0.01  # 1% of normalized range

    def test_convergence_threshold_custom(self, mock_config_2d):
        """Test custom convergence threshold."""
        mock_config_2d.max_abs_error = 0.005

        result = preprocess_data(mock_config_2d)

        assert result.max_abs_error == 0.005

    def test_tensor_conversion(self, mock_config_2d):
        """Test conversion to PyTorch tensor."""
        result = preprocess_data(mock_config_2d)

        assert isinstance(result.V_tensor, torch.Tensor)
        assert result.V_tensor.device == mock_config_2d.device
        assert result.V_tensor.dtype == torch.float32
        assert result.V_tensor.shape == mock_config_2d.V.shape
