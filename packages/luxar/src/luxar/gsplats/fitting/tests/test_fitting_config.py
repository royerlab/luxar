"""
Tests for fitting configuration dataclasses.
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
    from luxar.gsplats.fitting.config import (
        ConstraintConfig,
        FitConfig,
        LossConfig,
        ModelComponents,
        OptimConfig,
        OptimizationResults,
        PreprocessedData,
    )
    from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig


def _make_fit_config(**overrides) -> "FitConfig":
    """Helper to create a FitConfig with sensible defaults, overridable per-field."""
    defaults = dict(
        V=np.random.rand(32, 32).astype(np.float32),
        seeds=None,
        norm_percentile=0.0,
        init_sigma_vox=1.5,
        sigma_min_diag=[0.1, 0.1],
        sigma_max_diag=None,
        truncate=3.0,
        n_iters=100,
        lr=0.01,
        max_abs_error=0.01,
        rel_l2_target=None,
        gradient_clip=1.0,
        loss_type="l1",
        asymmetric_penalty=10.0,
        l1_amp=0.001,
        l1_diag=0.0001,
        scheduler_type="plateau",
        patience=10,
        lr_reduction_factor=0.5,
        early_stop_patience=None,
        enable_dynamic_ops=True,
        dynamic_config=DynamicOpsConfig(),
        dynamic_ops_verbose=False,
        napari_movie=False,
        movie_every=1,
        movie_max_frames=100,
        device=torch.device("cpu"),
        verbose=False,
    )
    defaults.update(overrides)
    return FitConfig(**defaults)


class TestFitConfig:
    """Test FitConfig dataclass."""

    def test_fit_config_creation(self) -> None:
        """Test basic FitConfig creation."""
        config = _make_fit_config(verbose=True)

        assert config.V.shape == (32, 32)
        assert config.n_iters == 100
        assert config.lr == 0.01
        assert config.device == torch.device("cpu")

    def test_fit_config_stores_all_required_fields(self) -> None:
        """Verify every required field is stored correctly."""
        V = np.zeros((8, 8), dtype=np.float32)
        config = _make_fit_config(
            V=V,
            n_iters=500,
            lr=0.05,
            loss_type="mse",
            truncate=4.0,
            patience=20,
            lr_reduction_factor=0.9,
        )
        assert config.V is V
        assert config.n_iters == 500
        assert config.lr == 0.05
        assert config.loss_type == "mse"
        assert config.truncate == 4.0
        assert config.patience == 20
        assert config.lr_reduction_factor == 0.9

    def test_fit_config_default_optional_fields(self) -> None:
        """Verify optional fields with defaults are set correctly."""
        config = _make_fit_config()
        # Fields with default values
        assert config.use_metal is True
        assert config.use_cuda is True
        assert config.floor == "auto"
        assert config.seed_method == "auto"
        assert config.seed_kwargs is None
        assert config.init_L is None
        assert config.init_amps is None
        assert config.amp_max is None
        assert config.max_eccentricity is None
        assert config.voxel_footprint_correction is False
        assert config.clip_to_bounds is False
        assert config.voxel_size is None
        assert config.output_space == "real"
        assert config.boundary_penalty is None
        assert config.sort_splats_enabled is True
        assert config.sort_splats_interval == 1000
        assert config.downscale is None

    def test_fit_config_accepts_negative_n_iters(self) -> None:
        """FitConfig is a plain dataclass with no validation -- negative n_iters
        is accepted at construction time (validation happens in fit_gaussian_splats)."""
        config = _make_fit_config(n_iters=-1)
        assert config.n_iters == -1

    def test_fit_config_accepts_zero_lr(self) -> None:
        """FitConfig is a plain dataclass -- zero lr is accepted at construction
        time (validation happens in fit_gaussian_splats)."""
        config = _make_fit_config(lr=0.0)
        assert config.lr == 0.0

    def test_fit_config_empty_volume(self) -> None:
        """FitConfig accepts an empty volume array (validation is deferred)."""
        config = _make_fit_config(V=np.array([], dtype=np.float32))
        assert config.V.size == 0

    def test_fit_config_with_all_optional_overrides(self) -> None:
        """Test FitConfig with all optional fields explicitly set."""
        config = _make_fit_config(
            seed_method="edges",
            seed_kwargs={"n_seeds": 200},
            init_L=np.zeros((5, 2, 2), dtype=np.float32),
            init_amps=np.ones(5, dtype=np.float32),
            amp_max=10.0,
            max_eccentricity=5.0,
            voxel_footprint_correction=0.5,
            clip_to_bounds=True,
            voxel_size=np.array([1.0, 2.0]),
            output_space="voxel",
            boundary_penalty=0.1,
            sort_splats_enabled=False,
            sort_splats_interval=500,
            downscale=(2, 2),
        )
        assert config.seed_method == "edges"
        assert config.seed_kwargs == {"n_seeds": 200}
        assert config.init_L.shape == (5, 2, 2)
        assert config.init_amps.shape == (5,)
        assert config.amp_max == 10.0
        assert config.max_eccentricity == 5.0
        assert config.voxel_footprint_correction == 0.5
        assert config.clip_to_bounds is True
        assert config.output_space == "voxel"
        assert config.boundary_penalty == 0.1
        assert config.sort_splats_enabled is False
        assert config.sort_splats_interval == 500
        assert config.downscale == (2, 2)

    def test_fit_config_is_mutable(self) -> None:
        """FitConfig is a regular (non-frozen) dataclass, so fields are mutable."""
        config = _make_fit_config(n_iters=100)
        config.n_iters = 200
        assert config.n_iters == 200


class TestOptimConfig:
    """Test OptimConfig frozen dataclass."""

    def test_default_values(self) -> None:
        """Test OptimConfig default values match expected constants."""
        cfg = OptimConfig()
        assert cfg.n_iters == 1000
        assert cfg.lr == 0.01
        assert cfg.gradient_clip == 1.0
        assert cfg.scheduler_type == "plateau"
        assert cfg.patience == 25
        assert cfg.lr_reduction_factor == 0.98
        assert cfg.early_stop_patience == 300
        assert cfg.sort_splats_enabled is True
        assert cfg.sort_splats_interval == 1000

    def test_custom_values(self) -> None:
        """Test OptimConfig with custom values."""
        cfg = OptimConfig(n_iters=500, lr=0.05, early_stop_patience=None)
        assert cfg.n_iters == 500
        assert cfg.lr == 0.05
        assert cfg.early_stop_patience is None

    def test_frozen(self) -> None:
        """OptimConfig is frozen -- mutation raises."""
        cfg = OptimConfig()
        with pytest.raises(AttributeError):
            cfg.n_iters = 999  # type: ignore[misc]


class TestLossConfig:
    """Test LossConfig frozen dataclass."""

    def test_default_values(self) -> None:
        """Test LossConfig default values."""
        cfg = LossConfig()
        assert cfg.loss_type == "l1"
        assert cfg.asymmetric_penalty == 1.0
        assert cfg.l1_amp is None
        assert cfg.l1_diag is None
        assert cfg.boundary_penalty is None

    def test_custom_values(self) -> None:
        """Test LossConfig with custom values."""
        cfg = LossConfig(loss_type="mse", l1_amp=0.01, l1_diag=0.001)
        assert cfg.loss_type == "mse"
        assert cfg.l1_amp == 0.01
        assert cfg.l1_diag == 0.001

    def test_frozen(self) -> None:
        """LossConfig is frozen -- mutation raises."""
        cfg = LossConfig()
        with pytest.raises(AttributeError):
            cfg.loss_type = "poisson"  # type: ignore[misc]


class TestConstraintConfig:
    """Test ConstraintConfig frozen dataclass."""

    def test_default_values(self) -> None:
        """Test ConstraintConfig default values."""
        cfg = ConstraintConfig()
        assert cfg.sigma_min_diag is None
        assert cfg.sigma_max_diag is None
        assert cfg.amp_max is None
        assert cfg.max_eccentricity == 10.0
        assert cfg.truncate == 2.75
        assert cfg.voxel_size is None
        assert cfg.output_space == "real"
        assert cfg.boundary_penalty is None
        assert cfg.clip_to_bounds is False

    def test_custom_values(self) -> None:
        """Test ConstraintConfig with custom values."""
        cfg = ConstraintConfig(
            sigma_min_diag=[0.5, 0.5],
            sigma_max_diag=[5.0, 5.0],
            amp_max=2.0,
            truncate=4.0,
            clip_to_bounds=True,
        )
        assert cfg.sigma_min_diag == [0.5, 0.5]
        assert cfg.sigma_max_diag == [5.0, 5.0]
        assert cfg.amp_max == 2.0
        assert cfg.truncate == 4.0
        assert cfg.clip_to_bounds is True

    def test_frozen(self) -> None:
        """ConstraintConfig is frozen -- mutation raises."""
        cfg = ConstraintConfig()
        with pytest.raises(AttributeError):
            cfg.truncate = 5.0  # type: ignore[misc]


class TestPreprocessedData:
    """Test PreprocessedData dataclass."""

    def test_preprocessed_data_creation(self) -> None:
        """Test basic PreprocessedData creation."""
        V_normalized = np.random.rand(16, 16).astype(np.float32)
        V_tensor = torch.tensor(V_normalized)
        centers = np.random.rand(10, 2).astype(np.float32)

        data = PreprocessedData(
            V_normalized=V_normalized,
            V_tensor=V_tensor,
            seed_centers=centers,
            image_min=0.0,
            image_max=1.0,
            intensity_range=1.0,
            d=2,
            N=10,
            max_abs_error=0.01,
        )

        assert data.V_normalized.shape == (16, 16)
        assert data.seed_centers.shape == (10, 2)
        assert data.d == 2
        assert data.N == 10

    def test_preprocessed_data_optional_fields_default_none(self) -> None:
        """Verify optional fields default to None."""
        data = PreprocessedData(
            V_normalized=np.zeros((4, 4), dtype=np.float32),
            V_tensor=torch.zeros(4, 4),
            seed_centers=np.zeros((1, 2), dtype=np.float32),
            image_min=0.0,
            image_max=1.0,
            intensity_range=1.0,
            d=2,
            N=1,
            max_abs_error=0.01,
        )
        assert data.rel_l2_target is None
        assert data.l1_amp is None
        assert data.l1_diag is None
        assert data.floor is None
        assert data.init_L is None
        assert data.init_amps is None
        assert data.downscale_factors is None

    def test_preprocessed_data_with_init_arrays(self) -> None:
        """Test PreprocessedData with pre-initialized parameters."""
        init_L = np.eye(2, dtype=np.float32).reshape(1, 2, 2)
        init_amps = np.array([1.0], dtype=np.float32)

        data = PreprocessedData(
            V_normalized=np.zeros((4, 4), dtype=np.float32),
            V_tensor=torch.zeros(4, 4),
            seed_centers=np.zeros((1, 2), dtype=np.float32),
            image_min=0.0,
            image_max=1.0,
            intensity_range=1.0,
            d=2,
            N=1,
            max_abs_error=0.01,
            init_L=init_L,
            init_amps=init_amps,
        )
        assert data.init_L is not None
        assert data.init_L.shape == (1, 2, 2)
        assert data.init_amps is not None
        assert data.init_amps.shape == (1,)


class TestOptimizationResults:
    """Test OptimizationResults dataclass."""

    def test_optimization_results_creation(self) -> None:
        """Test basic OptimizationResults creation."""
        centers = torch.rand(5, 2)
        Ls = torch.rand(5, 2, 2)
        amps = torch.rand(5)

        results = OptimizationResults(
            centers=centers,
            Ls=Ls,
            amps=amps,
            converged_early=True,
            early_stopped=False,
            actual_iters=50,
            best_iteration=45,
            best_loss=0.1,
            best_max_abs_error=0.005,
            best_rel_l2=0.1,
            movie_frames=None,
            start_time=0.0,
            end_time=1.0,
        )

        assert results.centers.shape == (5, 2)
        assert results.amps.shape == (5,)
        assert results.converged_early
        assert results.actual_iters == 50

    def test_optimization_results_timing(self) -> None:
        """Verify timing fields are stored and duration can be derived."""
        results = OptimizationResults(
            centers=torch.rand(1, 2),
            Ls=torch.rand(1, 2, 2),
            amps=torch.rand(1),
            converged_early=False,
            early_stopped=True,
            actual_iters=10,
            best_iteration=8,
            best_loss=0.5,
            best_max_abs_error=0.05,
            best_rel_l2=0.3,
            movie_frames=None,
            start_time=100.0,
            end_time=105.5,
        )
        duration = results.end_time - results.start_time
        assert abs(duration - 5.5) < 1e-6
        assert results.early_stopped is True
        assert results.converged_early is False

    def test_optimization_results_with_movie_frames(self) -> None:
        """Test OptimizationResults with movie frames populated."""
        frames = {"centers": [torch.rand(2, 2)], "amps": [torch.rand(2)]}
        results = OptimizationResults(
            centers=torch.rand(2, 2),
            Ls=torch.rand(2, 2, 2),
            amps=torch.rand(2),
            converged_early=False,
            early_stopped=False,
            actual_iters=5,
            best_iteration=3,
            best_loss=0.2,
            best_max_abs_error=0.01,
            best_rel_l2=0.05,
            movie_frames=frames,
            start_time=0.0,
            end_time=1.0,
        )
        assert results.movie_frames is not None
        assert "centers" in results.movie_frames


class TestModelComponents:
    """Test ModelComponents dataclass."""

    def test_model_components_creation(self) -> None:
        """Test basic ModelComponents creation."""
        components = ModelComponents(
            model=None,
            optimizer=None,
            scheduler=None,
        )

        assert components.model is None
        assert components.optimizer is None
        assert components.scheduler is None

    def test_model_components_with_real_optimizer(self) -> None:
        """Test ModelComponents stores a real optimizer object."""
        param = torch.nn.Parameter(torch.zeros(3))
        optimizer = torch.optim.Adam([param], lr=0.01)

        components = ModelComponents(
            model=None,
            optimizer=optimizer,
            scheduler=None,
        )
        assert isinstance(components.optimizer, torch.optim.Adam)
