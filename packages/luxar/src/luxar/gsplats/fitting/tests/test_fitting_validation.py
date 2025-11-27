"""
Tests for fitting validation module.
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
    from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
    from luxar.gsplats.fitting.validation import prepare_fit_config


class MockGaussianSplatFitter:
    """Mock fitter for testing validation."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.enable_dynamic_ops = True
        self.dynamic_config = DynamicOpsConfig()


class TestPrepareConfig:
    """Test configuration preparation and validation."""

    def test_basic_config_preparation(self) -> None:
        """Test basic configuration preparation."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V)
        assert config.V.shape == (16, 16)
        assert config.device == torch.device("cpu")
        assert config.enable_dynamic_ops
        assert config.n_iters == 1000  # default
        assert config.lr == 0.01  # default

    def test_input_validation_empty_image(self) -> None:
        """Test validation of empty image."""
        fitter = MockGaussianSplatFitter()
        V = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="Input image V cannot be empty"):
            prepare_fit_config(fitter, V)

    def test_input_validation_scalar_image(self) -> None:
        """Test validation of scalar image."""
        fitter = MockGaussianSplatFitter()
        V = np.array(5.0, dtype=np.float32)

        with pytest.raises(
            ValueError, match="Input image V must have at least 1 dimension"
        ):
            prepare_fit_config(fitter, V)

    def test_candidates_validation_wrong_dimensions(self) -> None:
        """Test validation of candidate centers with wrong dimensions."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)
        centers = np.random.rand(10, 3).astype(
            np.float32
        )  # Wrong: 3D centers for 2D image

        with pytest.raises(ValueError, match="seeds must have 2 columns"):
            prepare_fit_config(fitter, V, seeds=centers)

    def test_candidates_validation_wrong_shape(self) -> None:
        """Test validation of candidate centers with wrong shape."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)
        centers = np.random.rand(10).astype(np.float32)  # Wrong: 1D array

        with pytest.raises(ValueError, match="seeds must be a 2D array"):
            prepare_fit_config(fitter, V, seeds=centers)

    def test_parameter_validation_negative_sigma(self) -> None:
        """Test validation of negative sigma."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="init_sigma_vox must be positive"):
            prepare_fit_config(fitter, V, init_sigma_vox=-1.0)

    def test_parameter_validation_negative_iterations(self) -> None:
        """Test validation of negative iterations."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="n_iters must be positive"):
            prepare_fit_config(fitter, V, n_iters=-10)

    def test_parameter_validation_negative_lr(self) -> None:
        """Test validation of negative learning rate."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="lr must be positive"):
            prepare_fit_config(fitter, V, lr=-0.01)

    def test_parameter_validation_invalid_loss_type(self) -> None:
        """Test validation of invalid loss type."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(
            ValueError, match="loss_type must be 'mse', 'poisson', or 'l1'"
        ):
            prepare_fit_config(fitter, V, loss_type="invalid")

    def test_sigma_constraints_validation_wrong_length(self) -> None:
        """Test validation of sigma constraints with wrong length."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="sigma_min_diag must have length 2"):
            prepare_fit_config(
                fitter, V, sigma_min_diag=[0.1, 0.1, 0.1]
            )  # Wrong: 3 values for 2D  # type: ignore[arg-type]

    def test_sigma_constraints_validation_negative_values(self) -> None:
        """Test validation of negative sigma constraints."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(
            ValueError, match="All sigma_min_diag values must be positive"
        ):
            prepare_fit_config(fitter, V, sigma_min_diag=[0.1, -0.1])

    def test_sigma_max_less_than_min(self) -> None:
        """Test validation when sigma_max is less than sigma_min."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(
            ValueError, match="sigma_max_diag must be greater than sigma_min_diag"
        ):
            prepare_fit_config(
                fitter, V, sigma_min_diag=[1.0, 1.0], sigma_max_diag=[0.5, 0.5]
            )  # type: ignore[arg-type]

    def test_l1_regularization_default(self) -> None:
        """Test that L1 regularization is None by default."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, lr=0.02)
        # L1 defaults set in preprocessing.py after gradient dilution calc
        assert config.l1_amp is None
        assert config.l1_diag is None
        assert config.l1_sharpness is None

    def test_l1_diag_regularization_custom(self) -> None:
        """Test custom L1 diagonal regularization parameter."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, lr=0.01, l1_diag=0.005)
        assert config.l1_diag == 0.005  # Custom value preserved

    def test_custom_parameters_preserved(self) -> None:
        """Test that custom parameters are preserved correctly."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)
        centers = np.random.rand(5, 2).astype(np.float32)

        config = prepare_fit_config(
            fitter,
            V,  # type: ignore[arg-type]
            seeds=centers,
            n_iters=500,
            lr=0.05,
            loss_type="mse",
            truncate=2.5,
            verbose=False,
        )

        assert config.seeds.shape == (5, 2)
        assert config.n_iters == 500
        assert config.lr == 0.05
        assert config.loss_type == "mse"
        assert config.truncate == 2.5
        assert config.verbose is False
