"""
Tests for fitting configuration dataclasses.
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
    from luxar.gsplats.fitting.config import (
        FitConfig,
        ModelComponents,
        OptimizationResults,
        PreprocessedData,
    )
    from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig


class TestFitConfig:
    """Test FitConfig dataclass."""

    def test_fit_config_creation(self) -> None:
        """Test basic FitConfig creation."""
        V = np.random.rand(32, 32).astype(np.float32)
        device = torch.device("cpu")
        dynamic_config = DynamicOpsConfig()

        config = FitConfig(
            V=V,
            seeds=None,
            norm_percentile=0.0,
            init_sigma_vox=1.5,
            sigma_min_diag=[0.1, 0.1],
            sigma_max_diag=None,
            truncate=3.0,
            n_iters=100,
            lr=0.01,
            max_abs_error=0.01,
            gradient_clip=1.0,
            loss_type="l1",
            asymmetric_penalty=10.0,
            l1_amp=0.001,
            l1_diag=0.0001,
            l1_sharpness=0.0,
            scheduler_type="plateau",
            patience=10,
            lr_reduction_factor=0.5,
            early_stop_patience=None,
            enable_dynamic_ops=True,
            dynamic_config=dynamic_config,
            dynamic_ops_verbose=False,
            napari_movie=False,
            movie_every=1,
            movie_max_frames=100,
            device=device,
            verbose=True,
        )

        assert config.V.shape == (32, 32)
        assert config.n_iters == 100
        assert config.lr == 0.01
        assert config.device == device


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


class TestOptimizationResults:
    """Test OptimizationResults dataclass."""

    def test_optimization_results_creation(self) -> None:
        """Test basic OptimizationResults creation."""
        centers = torch.rand(5, 2)
        Ls = torch.rand(5, 2, 2)
        amps = torch.rand(5)
        sharpness = torch.full((5,), 2.0)  # Standard Gaussian sharpness

        results = OptimizationResults(
            centers=centers,
            Ls=Ls,
            amps=amps,
            sharpness=sharpness,
            converged_early=True,
            actual_iters=50,
            best_iteration=45,
            best_loss=0.1,
            best_max_abs_error=0.005,
            movie_frames=None,
            start_time=0.0,
            end_time=1.0,
        )

        assert results.centers.shape == (5, 2)
        assert results.amps.shape == (5,)
        assert results.sharpness.shape == (5,)
        assert results.converged_early
        assert results.actual_iters == 50


class TestModelComponents:
    """Test ModelComponents dataclass."""

    def test_model_components_creation(self) -> None:
        """Test basic ModelComponents creation."""
        # Create mock components
        components = ModelComponents(
            model=None,  # Would be GaussianSplatModel in real usage
            optimizer=None,  # Would be torch optimizer
            scheduler=None,  # Would be LR scheduler
        )

        # Test that structure is correct
        assert hasattr(components, "model")
        assert hasattr(components, "optimizer")
        assert hasattr(components, "scheduler")
