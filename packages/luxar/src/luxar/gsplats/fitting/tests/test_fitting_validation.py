"""
Tests for fitting validation module.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from luxar.conftest import (
    ValidationCase,
    assert_validation_failure,
    assert_validation_precedence,
    ordered_validation_pairs,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

if HAS_TORCH:
    from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
    from luxar.gsplats.gsplat_data import GSplatData

    from .conftest import prepare_fit_config


class MockGaussianSplatFitter:
    """Mock fitter for testing validation."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.enable_dynamic_ops = True
        self.dynamic_config = DynamicOpsConfig()
        self.use_metal = False
        self.use_cuda = False


VALIDATION_PRECEDENCE = (
    ValidationCase(
        "source_shape", {"source_shape": []}, ValueError, "source_shape cannot be empty"
    ),
    ValidationCase(
        "source_stored_bytes",
        {"source_stored_bytes": 0},
        ValueError,
        "source_stored_bytes must be a positive integer, got 0",
    ),
    ValidationCase(
        "empty_volume",
        {"V": np.empty((0, 2), dtype=np.float32)},
        ValueError,
        "Input image V cannot be empty",
    ),
    ValidationCase(
        "downscale",
        {"downscale": 0},
        ValueError,
        "downscale factors must be >= 1, got 0",
    ),
    ValidationCase("seeds", {"seeds": 0}, ValueError, "seeds as int must be positive"),
    ValidationCase(
        "init_sigma_vox",
        {"init_sigma_vox": 0},
        ValueError,
        "init_sigma_vox must be positive if specified",
    ),
    ValidationCase("n_iters", {"n_iters": 0}, ValueError, "n_iters must be positive"),
    ValidationCase("lr", {"lr": 0}, ValueError, "lr must be positive"),
    ValidationCase(
        "loss_type",
        {"loss_type": "invalid"},
        ValueError,
        "loss_type must be 'mse', 'poisson', or 'l1'",
    ),
    ValidationCase(
        "l1_amp",
        {"l1_amp": -1},
        ValueError,
        "l1_amp must be non-negative if specified",
    ),
    ValidationCase(
        "l1_diag",
        {"l1_diag": -1},
        ValueError,
        "l1_diag must be non-negative if specified",
    ),
    ValidationCase(
        "asymmetric_penalty",
        {"asymmetric_penalty": 0.5},
        ValueError,
        "asymmetric_penalty must be >= 1.0 (values < 1.0 would invert the penalty)",
    ),
    ValidationCase(
        "gradient_clip",
        {"gradient_clip": 0},
        ValueError,
        "gradient_clip must be positive if specified",
    ),
    ValidationCase("patience", {"patience": 0}, ValueError, "patience must be >= 1"),
    ValidationCase(
        "lr_reduction_factor",
        {"lr_reduction_factor": 1},
        ValueError,
        "lr_reduction_factor must be in range (0, 1)",
    ),
    ValidationCase(
        "early_stop_patience",
        {"early_stop_patience": 0},
        ValueError,
        "early_stop_patience must be >= 1 if specified",
    ),
    ValidationCase(
        "scheduler_type",
        {"scheduler_type": "invalid"},
        ValueError,
        "scheduler_type must be 'plateau' or 'exponential'",
    ),
    ValidationCase(
        "truncate", {"truncate": 0}, ValueError, "truncate must be positive"
    ),
    ValidationCase(
        "max_abs_error",
        {"max_abs_error": 0},
        ValueError,
        "max_abs_error must be positive if specified",
    ),
    ValidationCase(
        "rel_l2_target",
        {"rel_l2_target": 0},
        ValueError,
        "rel_l2_target must be positive if specified",
    ),
    ValidationCase(
        "movie_max_frames",
        {"movie_max_frames": 0},
        ValueError,
        "movie_max_frames must be positive or None",
    ),
    ValidationCase(
        "movie_every", {"movie_every": 0}, ValueError, "movie_every must be >= 1"
    ),
    ValidationCase(
        "norm_percentile",
        {"norm_percentile": 50},
        ValueError,
        "norm_percentile must be in range [0.0, 50.0), got 50",
    ),
    ValidationCase("floor", {"floor": -1}, ValueError, "floor must be >= 0, got -1.0"),
    ValidationCase(
        "sigma_min_diag",
        {"sigma_min_diag": 0},
        ValueError,
        "sigma_min_diag must be positive if specified",
    ),
    ValidationCase(
        "sigma_max_diag",
        {"sigma_max_diag": 0},
        ValueError,
        "sigma_max_diag fraction must be positive",
    ),
    ValidationCase(
        "amp_max",
        {"amp_max": 0},
        ValueError,
        "amp_max must be positive if specified",
    ),
    ValidationCase(
        "norm_range",
        {"norm_range": (1, 1)},
        ValueError,
        "norm_range must satisfy image_max > image_min, got (1, 1)",
    ),
    ValidationCase(
        "max_eccentricity",
        {"max_eccentricity": 0},
        ValueError,
        "max_eccentricity must be >= 1.0 (ratio of longest to shortest axis)",
    ),
    ValidationCase(
        "voxel_footprint_correction",
        {"voxel_footprint_correction": -1},
        ValueError,
        "voxel_footprint_correction sigma must be positive",
    ),
    ValidationCase(
        "boundary_penalty",
        {"boundary_penalty": -1},
        ValueError,
        "boundary_penalty must be non-negative if specified",
    ),
    ValidationCase(
        "voxel_size", {"voxel_size": 0}, ValueError, "voxel_size must be positive"
    ),
    ValidationCase(
        "output_space",
        {"output_space": "invalid"},
        ValueError,
        "output_space must be 'real' or 'voxel'",
    ),
    ValidationCase(
        "sort_splats_interval",
        {"sort_splats_interval": 0},
        ValueError,
        "sort_splats_interval must be >= 1",
    ),
    ValidationCase(
        "iter_callback",
        {"iter_callback": 1},
        ValueError,
        "iter_callback must be callable or None",
    ),
    ValidationCase(
        "iter_callback_every",
        {"iter_callback_every": 0},
        ValueError,
        "iter_callback_every must be >= 1",
    ),
)


def _validate_with_defaults(overrides: Mapping[str, Any]) -> None:
    kwargs = dict(overrides)
    volume = kwargs.pop("V", np.ones((2, 3), dtype=np.float32))
    prepare_fit_config(MockGaussianSplatFitter(), volume, **kwargs)


@pytest.mark.parametrize("case", VALIDATION_PRECEDENCE, ids=lambda case: case.name)
def test_prepare_fit_config_preserves_validation_failures(case: ValidationCase) -> None:
    """Each recorded invalid input keeps its exact exception type and message."""
    assert_validation_failure(_validate_with_defaults, case)


@pytest.mark.parametrize(
    ("earlier", "later"),
    ordered_validation_pairs(VALIDATION_PRECEDENCE),
    ids=lambda case: case.name,
)
def test_prepare_fit_config_preserves_validation_precedence(
    earlier: ValidationCase, later: ValidationCase
) -> None:
    """Every earlier invalid parameter wins when a later rule also fails."""
    assert_validation_precedence(_validate_with_defaults, earlier, later)


def test_prepare_fit_config_rejects_wrong_dimension_gsplat_seeds() -> None:
    """Warm-start splats must match the input volume dimensionality."""
    seeds = GSplatData(
        centers=np.ones((1, 3), dtype=np.float32),
        amplitudes=np.ones(1, dtype=np.float32),
        cholesky_factors=np.ones((1, 6), dtype=np.float32),
    )
    case = ValidationCase(
        "gsplat_seed_dimensions",
        {"seeds": seeds},
        ValueError,
        "GSplatData centers must have 2 columns to match image dimensions",
    )

    assert_validation_failure(_validate_with_defaults, case)


class TestPrepareConfig:
    """Test configuration preparation and validation."""

    def test_seed_kwargs_are_copied_at_the_config_boundary(self) -> None:
        """Preprocessing mutations must not leak back into caller parameters."""
        fitter = MockGaussianSplatFitter()
        seed_kwargs = {"num_scales": 3}

        config = prepare_fit_config(
            fitter,
            np.ones((4, 4), dtype=np.float32),
            seed_kwargs=seed_kwargs,
        )

        assert config.seed_kwargs == seed_kwargs
        assert config.seed_kwargs is not seed_kwargs
        config.seed_kwargs["device"] = "cpu"
        assert seed_kwargs == {"num_scales": 3}

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
        assert config.movie_max_frames == 10000

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
    def test_specimen_floor_is_valid(self) -> None:
        fitter = MockGaussianSplatFitter()
        config = prepare_fit_config(
            fitter, np.ones((4, 4), dtype=np.float32), floor="specimen"
        )
        assert config.floor == "specimen"

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
