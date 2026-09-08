"""
Tests for fitting validation module.
"""

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

if HAS_TORCH:
    from luxar.gsplats.fitting.config import FitParameters
    from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
    from luxar.gsplats.fitting.validation import (
        prepare_fit_config as _prepare_fit_config,
    )


def prepare_fit_config(fitter, V, **kwargs):
    """Build raw parameters while keeping validation tests concise."""
    return _prepare_fit_config(fitter, FitParameters(V=V, **kwargs))


class MockGaussianSplatFitter:
    """Mock fitter for testing validation."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.enable_dynamic_ops = True
        self.dynamic_config = DynamicOpsConfig()
        self.use_metal = False
        self.use_cuda = False


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

    def test_sigma_min_diag_float_broadcast(self) -> None:
        """Test that sigma_min_diag accepts a float and broadcasts it."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, sigma_min_diag=0.2)
        assert config.sigma_min_diag == [0.2, 0.2]

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

    def test_seeds_as_int_valid(self) -> None:
        """Test seeds parameter as valid integer count."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        # Should accept positive integers
        config = prepare_fit_config(fitter, V, seeds=100)
        assert config.seeds == 100

        # Should accept 1
        config = prepare_fit_config(fitter, V, seeds=1)
        assert config.seeds == 1

    def test_seeds_as_int_invalid(self) -> None:
        """Test seeds parameter as invalid integer."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        # Should reject zero
        with pytest.raises(ValueError, match="seeds as int must be positive"):
            prepare_fit_config(fitter, V, seeds=0)

        # Should reject negative
        with pytest.raises(ValueError, match="seeds as int must be positive"):
            prepare_fit_config(fitter, V, seeds=-10)

    def test_seeds_as_float_valid(self) -> None:
        """Test seeds parameter as valid compression ratio."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        # Should accept compression ratios in (0, 1]
        config = prepare_fit_config(fitter, V, seeds=0.01)
        assert config.seeds == 0.01

        config = prepare_fit_config(fitter, V, seeds=1.0)
        assert config.seeds == 1.0

        config = prepare_fit_config(fitter, V, seeds=0.5)
        assert config.seeds == 0.5

    def test_seeds_as_float_invalid(self) -> None:
        """Test seeds parameter as invalid compression ratio."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        # Should reject values > 1.0
        with pytest.raises(ValueError, match="compression ratio"):
            prepare_fit_config(fitter, V, seeds=1.5)

        # Should reject zero
        with pytest.raises(ValueError, match="compression ratio"):
            prepare_fit_config(fitter, V, seeds=0.0)

        # Should reject negative
        with pytest.raises(ValueError, match="compression ratio"):
            prepare_fit_config(fitter, V, seeds=-0.1)


class TestFloorValidation:
    """Tests for the ``floor`` (background suppression) parameter."""

    def test_default_floor_is_auto(self) -> None:
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)
        config = prepare_fit_config(fitter, V)
        assert config.floor == "auto"

    @pytest.mark.parametrize("value", ["auto", "none", "p10", "0.5", 0.5, 0, None])
    def test_valid_floor_values_accepted(self, value) -> None:
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)
        config = prepare_fit_config(fitter, V, floor=value)
        assert config.floor == value

    @pytest.mark.parametrize(
        "value", ["-1", "pX", "abc", -2.0, float("nan"), float("inf"), "nan", "inf"]
    )
    def test_invalid_floor_values_rejected(self, value) -> None:
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)
        with pytest.raises(ValueError):
            prepare_fit_config(fitter, V, floor=value)


class TestBoundaryPenaltyValidation:
    """Tests for boundary_penalty parameter validation."""

    def test_negative_boundary_penalty_raises_error(self) -> None:
        """Negative boundary_penalty raises ValueError."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="boundary_penalty must be non-negative"):
            prepare_fit_config(fitter, V, boundary_penalty=-0.1)

    def test_positive_boundary_penalty_accepted(self) -> None:
        """Positive boundary_penalty is accepted."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, boundary_penalty=0.5)
        assert config.boundary_penalty == 0.5

    def test_zero_boundary_penalty_accepted(self) -> None:
        """Zero boundary_penalty is accepted (effectively disabled)."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, boundary_penalty=0.0)
        assert config.boundary_penalty == 0.0

    def test_none_boundary_penalty_accepted(self) -> None:
        """None boundary_penalty is accepted (default, disabled)."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, boundary_penalty=None)
        assert config.boundary_penalty is None

    def test_clip_to_bounds_accepted(self) -> None:
        """clip_to_bounds boolean values are accepted."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, clip_to_bounds=True)
        assert config.clip_to_bounds is True

        config = prepare_fit_config(fitter, V, clip_to_bounds=False)
        assert config.clip_to_bounds is False


class TestVoxelSizeValidation:
    """Tests for voxel_size parameter validation."""

    def test_voxel_size_none_default(self) -> None:
        """Default voxel_size is None."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(8, 16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V)
        assert config.voxel_size is None

    def test_voxel_size_scalar_broadcasts_to_array(self) -> None:
        """Scalar voxel_size broadcasts to per-dim array."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(8, 16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, voxel_size=0.381)
        assert config.voxel_size is not None
        assert config.voxel_size.shape == (3,)
        assert config.voxel_size.dtype == np.float32
        np.testing.assert_allclose(config.voxel_size, [0.381, 0.381, 0.381])

    def test_voxel_size_tuple_accepted(self) -> None:
        """Tuple voxel_size (as demos use) is accepted."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(8, 16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, voxel_size=(0.29, 0.26, 0.26))
        np.testing.assert_allclose(config.voxel_size, [0.29, 0.26, 0.26])

    def test_voxel_size_list_accepted(self) -> None:
        """List voxel_size is accepted."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(8, 16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, voxel_size=[5.0, 1.0, 1.0])
        np.testing.assert_allclose(config.voxel_size, [5.0, 1.0, 1.0])

    def test_voxel_size_wrong_length_raises(self) -> None:
        """Mismatched length raises ValueError."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(8, 16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="voxel_size must have length 3"):
            prepare_fit_config(fitter, V, voxel_size=[1.0, 1.0])

    def test_voxel_size_negative_raises(self) -> None:
        """Negative voxel_size raises ValueError."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(8, 16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="voxel_size values must be positive"):
            prepare_fit_config(fitter, V, voxel_size=[5.0, -1.0, 1.0])

    def test_voxel_size_zero_raises(self) -> None:
        """Zero voxel_size raises ValueError."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(8, 16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="voxel_size values must be positive"):
            prepare_fit_config(fitter, V, voxel_size=[5.0, 0.0, 1.0])

    def test_voxel_size_scalar_negative_raises(self) -> None:
        """Negative scalar voxel_size raises ValueError."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(8, 16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="voxel_size must be positive"):
            prepare_fit_config(fitter, V, voxel_size=-1.0)


class TestOutputSpaceValidation:
    """Tests for output_space parameter validation."""

    def test_output_space_default_is_real(self) -> None:
        """Default output_space is 'real'."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V)
        assert config.output_space == "real"

    def test_output_space_real_accepted(self) -> None:
        """output_space='real' is accepted."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, output_space="real")
        assert config.output_space == "real"

    def test_output_space_voxel_accepted(self) -> None:
        """output_space='voxel' is accepted."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, output_space="voxel")
        assert config.output_space == "voxel"

    def test_output_space_invalid_raises(self) -> None:
        """Invalid output_space raises ValueError."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="output_space must be 'real' or 'voxel'"):
            prepare_fit_config(fitter, V, output_space="physical")


class TestSigmaMaxDiagFraction:
    """Tests for sigma_max_diag scalar fraction feature."""

    def test_sigma_max_diag_fraction_2d(self) -> None:
        """Scalar sigma_max_diag as fraction of volume extent (2D)."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(50, 200).astype(np.float32)

        config = prepare_fit_config(fitter, V, sigma_max_diag=1 / 16)
        expected = [50 / 16, 200 / 16]
        assert len(config.sigma_max_diag) == 2
        np.testing.assert_allclose(config.sigma_max_diag, expected)

    def test_sigma_max_diag_fraction_3d_anisotropic(self) -> None:
        """Scalar sigma_max_diag on anisotropic 3D volume."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(10, 100, 100).astype(np.float32)

        config = prepare_fit_config(fitter, V, sigma_max_diag=0.25)
        expected = [10 * 0.25, 100 * 0.25, 100 * 0.25]
        np.testing.assert_allclose(config.sigma_max_diag, expected)

    def test_sigma_max_diag_fraction_negative_raises(self) -> None:
        """Negative fraction raises ValueError."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(
            ValueError, match="sigma_max_diag fraction must be positive"
        ):
            prepare_fit_config(fitter, V, sigma_max_diag=-0.1)

    def test_sigma_max_diag_sequence_still_works(self) -> None:
        """Per-axis sequence sigma_max_diag still works."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, sigma_max_diag=[5.0, 8.0])
        assert list(config.sigma_max_diag) == [5.0, 8.0]

    def test_rel_l2_target_negative_raises(self) -> None:
        """Negative rel_l2_target raises ValueError."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="rel_l2_target must be positive"):
            prepare_fit_config(fitter, V, rel_l2_target=-0.1)

    def test_rel_l2_target_zero_raises(self) -> None:
        """Zero rel_l2_target raises ValueError."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        with pytest.raises(ValueError, match="rel_l2_target must be positive"):
            prepare_fit_config(fitter, V, rel_l2_target=0.0)

    def test_rel_l2_target_valid(self) -> None:
        """Valid rel_l2_target is accepted."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V, rel_l2_target=0.1)
        assert config.rel_l2_target == 0.1

    def test_rel_l2_target_none_default(self) -> None:
        """rel_l2_target defaults to None."""
        fitter = MockGaussianSplatFitter()
        V = np.random.rand(16, 16).astype(np.float32)

        config = prepare_fit_config(fitter, V)
        assert config.rel_l2_target is None
