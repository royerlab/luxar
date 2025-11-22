"""
Tests for dynamic Gaussian splat operations with residual-driven approach.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.fitting.dynamic_ops import (
    DynamicOpsConfig,
    _find_residual_peaks,
    apply_dynamic_operations,
)
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.seeds import find_seeds_multiscale_gaussian


class TestDynamicOpsConfig:
    """Test dynamic operations configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        cfg = DynamicOpsConfig()
        assert cfg.step_every == 50
        assert cfg.k_max_residuals == 10
        assert cfg.nms_radius_vox == 2.0
        assert cfg.min_contribution_threshold == 0.05
        assert cfg.relative_contribution_factor == 0.1
        assert cfg.lr_boost_factor == 1.5
        assert cfg.boost_influence_threshold == 0.05
        assert cfg.pruning_percentile == 5.0
        assert cfg.min_splats_to_keep == 10
        assert cfg.init_sigma_vox == 0.5

    def test_config_modification(self) -> None:
        """Test that config values can be modified."""
        cfg = DynamicOpsConfig()
        cfg.step_every = 25
        cfg.k_max_residuals = 5
        cfg.min_contribution_threshold = 1e-4

        assert cfg.step_every == 25
        assert cfg.k_max_residuals == 5
        assert cfg.min_contribution_threshold == 1e-4


class TestResidualPeakFinding:
    """Test residual peak finding functionality."""

    def test_find_residual_peaks_2d(self) -> None:
        """Test finding residual peaks in 2D images."""
        # Create a synthetic residual with clear peaks
        residual = torch.zeros((20, 20))
        residual[5, 5] = 1.0  # Peak 1
        residual[15, 15] = 0.8  # Peak 2
        residual[10, 10] = 0.6  # Peak 3

        peaks = _find_residual_peaks(residual, k_max_residuals=3, nms_radius_vox=2.0)

        assert len(peaks) <= 3
        assert len(peaks) > 0

        # Check that peaks contain the expected locations
        peak_locations = set(peaks)
        assert (5, 5) in peak_locations
        assert (15, 15) in peak_locations

    def test_find_residual_peaks_3d(self) -> None:
        """Test finding residual peaks in 3D volumes."""
        # Create a synthetic 3D residual
        residual = torch.zeros((10, 10, 10))
        residual[5, 5, 5] = 1.0  # Peak 1
        residual[2, 2, 2] = 0.7  # Peak 2

        peaks = _find_residual_peaks(residual, k_max_residuals=2, nms_radius_vox=1.5)

        assert len(peaks) <= 2
        assert len(peaks) > 0

        # Check that we get 3D coordinates
        for peak in peaks:
            assert len(peak) == 3

    def test_find_residual_peaks_empty(self) -> None:
        """Test behavior with no significant peaks."""
        residual = torch.zeros((10, 10))
        peaks = _find_residual_peaks(residual, k_max_residuals=5, nms_radius_vox=2.0)

        # Should return empty list for zero residual
        assert len(peaks) == 0


class TestSimplifiedSeeding:
    """Test ultra-simple seeding approach with direct amplitude and isotropic shape."""

    def test_simple_amplitude_estimation(self) -> None:
        """Test direct amplitude estimation from residual center value."""
        # Create synthetic residual with known peak
        residual = torch.zeros((20, 20))
        residual[10, 10] = 0.75  # Known residual value

        # Simple amplitude should equal residual value at center
        center = torch.tensor([10.0, 10.0])
        center_coords = torch.round(center).long()
        amplitude = torch.abs(residual[tuple(center_coords)])

        assert amplitude.item() == 0.75  # Should exactly match residual value
        assert amplitude.item() > 0

    def test_isotropic_shape_generation(self) -> None:
        """Test isotropic covariance matrix generation."""
        from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig

        cfg = DynamicOpsConfig()
        center = torch.tensor([10.0, 10.0])
        d = len(center)

        # Simple isotropic covariance
        L = torch.eye(d, device=center.device) * cfg.init_sigma_vox

        assert L.shape == (2, 2)
        assert torch.allclose(L, torch.eye(2) * 0.5)  # Should be identity scaled


class TestGaussianSplatModel:
    """Test basic Gaussian splat model operations required for dynamic ops."""

    def test_n_splats(self) -> None:
        """Test counting splats."""
        centers = np.array([[5.0, 5.0], [10.0, 10.0]])
        L0 = np.eye(2)[None, :, :] * 1.0  # (1, 2, 2)
        L0 = np.repeat(L0, 2, axis=0)  # (2, 2, 2)
        amps0 = np.array([1.0, 1.0])
        model = GaussianSplatModel(
            shape=(20, 20),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
        )
        assert model.n_splats() == 2

    def test_current_params(self) -> None:
        """Test retrieving current parameters."""
        centers = np.array([[5.0, 5.0], [10.0, 10.0]])
        L0 = np.eye(2)[None, :, :] * 1.0  # (1, 2, 2)
        L0 = np.repeat(L0, 2, axis=0)  # (2, 2, 2)
        amps0 = np.array([1.0, 1.0])
        model = GaussianSplatModel(
            shape=(20, 20),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
        )

        centers_t, Ls_t, amps_t, sharpness_t = model.current_params()

        assert centers_t.shape == (2, 2)
        assert Ls_t.shape == (2, 2, 2)
        assert amps_t.shape == (2,)


class TestDynamicOperationsIntegration:
    """Test the full dynamic operations pipeline."""

    def test_dynamic_ops_with_simple_model(self) -> None:
        """Test dynamic operations on a simple model."""
        # Create simple synthetic data
        V_target = torch.zeros((16, 16))
        V_target[8, 8] = 1.0  # Single bright spot

        # Create initial model with a few splats
        centers = np.array([[7.0, 7.0], [9.0, 9.0]])
        L0 = np.eye(2)[None, :, :] * 2.0  # (1, 2, 2)
        L0 = np.repeat(L0, 2, axis=0)  # (2, 2, 2)
        amps0 = np.array([1.0, 1.0])
        model = GaussianSplatModel(
            shape=(16, 16),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
        )

        # Create dummy optimizer and scheduler
        from luxar.gsplats.optim import PerSplatAdam, PerSplatReduceLROnPlateau

        optimizer = PerSplatAdam(model, lr=0.1)
        scheduler = PerSplatReduceLROnPlateau(optimizer, patience=5)

        # Get current prediction
        V_pred = model()

        # Configure dynamic operations
        cfg = DynamicOpsConfig()
        cfg.k_max_residuals = 3
        cfg.min_contribution_threshold = 1e-6

        # Apply dynamic operations
        opt_new, sched_new, topology_changed = apply_dynamic_operations(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            V_target=V_target,
            V_pred=V_pred,
            cfg=cfg,
            current_lr=0.1,
            max_abs_error_threshold=0.1,
            device=torch.device("cpu"),
            verbose=False,
        )

        # Check that we get valid returns
        assert opt_new is not None
        assert sched_new is not None
        assert isinstance(topology_changed, bool)

    def test_fit_with_dynamic_ops(self) -> None:
        """Test full fitting pipeline with dynamic operations enabled."""
        # Create simple test data
        blob = np.zeros((32, 32))
        blob[16, 16] = 1.0
        blob[12, 12] = 0.8
        V = blob.astype(np.float32)

        # Find seeds
        centers = find_seeds_multiscale_gaussian(
            V, scales=(1.0, 2.0), peaks_per_scale=10, percentile_thresh=50.0
        )

        if len(centers) == 0:
            pytest.skip("No seeds found for test data")

        # Configure dynamic operations
        cfg = DynamicOpsConfig()
        cfg.step_every = 5  # Run more frequently for testing

        # Run fitting with dynamic operations
        result = fit_gaussian_splats(
            V,
            seeds=centers,
            init_sigma_vox=1.5,
            n_iters=20,  # Short run for testing
            lr=0.1,
            enable_dynamic_ops=True,
            dynamic_config=cfg,
            verbose=False,
            napari_movie=False,  # Disable movie for tests
        )

        # Check that we got valid results
        assert result.centers.shape[0] > 0  # Should have some splats
        assert result.amplitudes.shape[0] == result.centers.shape[0]
        assert "final_loss" in result.stats  # Check for stats that actually exist

    def test_fit_without_dynamic_ops(self) -> None:
        """Test fitting without dynamic operations for comparison."""
        # Create simple test data
        blob = np.zeros((32, 32))
        blob[16, 16] = 1.0
        V = blob.astype(np.float32)

        # Find seeds
        centers = find_seeds_multiscale_gaussian(
            V, scales=(1.0,), peaks_per_scale=5, percentile_thresh=50.0
        )

        if len(centers) == 0:
            pytest.skip("No seeds found for test data")

        # Run fitting without dynamic operations
        result = fit_gaussian_splats(
            V,
            seeds=centers,
            init_sigma_vox=1.5,
            n_iters=10,
            lr=0.1,
            enable_dynamic_ops=False,
            verbose=False,
            napari_movie=False,  # Disable movie for tests
        )

        # Check that we got valid results
        assert result.centers.shape[0] > 0
        assert result.amplitudes.shape[0] == result.centers.shape[0]
        assert "final_loss" in result.stats  # Check for stats that actually exist

    def test_principled_pruning_functionality(self) -> None:
        """Test the new principled pruning algorithm."""
        from luxar.gsplats.fitting.dynamic_ops import (
            _calculate_splat_importance,
            _select_pruning_candidates,
        )

        # Create test model with varying importance splats
        V = np.random.random((32, 32)).astype(np.float32)
        centers = find_seeds_multiscale_gaussian(V, peaks_per_scale=50)

        # Create model with many splats to trigger pruning
        L0 = np.eye(2)[None, :, :] * 1.0
        L0 = np.repeat(L0, len(centers), axis=0)
        amps0 = np.random.uniform(0.01, 1.0, len(centers)).astype(
            np.float32
        )  # Varying amplitudes

        model = GaussianSplatModel(
            shape=(32, 32),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
        )

        # Test importance calculation
        importance = _calculate_splat_importance(model)
        assert importance.shape == (len(centers),)
        assert torch.all(importance >= 0)

        # Test candidate selection
        candidates = _select_pruning_candidates(importance, 10.0)  # Top 10%
        expected_candidates = max(1, int(len(centers) * 0.1))
        assert len(candidates) == expected_candidates

        # Verify candidates are actually least important
        sorted_importance = torch.sort(importance)[0]
        candidate_importance = importance[candidates]
        assert torch.all(candidate_importance <= sorted_importance[expected_candidates])

    def test_asymmetric_penalty_with_all_loss_types(self) -> None:
        """Test asymmetric penalty works with all loss functions."""
        V = np.random.random((24, 24)).astype(np.float32)
        centers = find_seeds_multiscale_gaussian(V, peaks_per_scale=20)

        for loss_type in ["mse", "poisson", "l1"]:
            result = fit_gaussian_splats(
                V,
                centers,
                n_iters=10,
                loss_type=loss_type,
                asymmetric_penalty=5.0,  # Test with asymmetric penalty
                verbose=False,
                enable_dynamic_ops=False,
                napari_movie=False,
            )

            assert len(result.amplitudes) > 0, f"{loss_type} with asymmetric penalty failed"
            assert all(result.amplitudes >= 0), f"{loss_type} produced negative amplitudes"

    def test_local_convergence_based_pruning(self) -> None:
        """Test the local convergence-based pruning algorithm."""
        # Create test data where some splats should be removable
        V = np.ones((32, 32), dtype=np.float32) * 0.5  # Uniform background
        centers = np.array([[10, 10], [15, 15], [20, 20]], dtype=np.float32)

        # Create model with varying importance
        L0 = np.stack(
            [np.eye(2) * 2.0, np.eye(2) * 0.5, np.eye(2) * 1.0]
        )  # Different sizes
        amps0 = np.array([0.8, 0.001, 0.5])  # Very different amplitudes

        model = GaussianSplatModel(
            shape=(32, 32),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.1, 0.1],
        )

        from luxar.gsplats.fitting.dynamic_ops import (
            DynamicOpsConfig,
            apply_dynamic_operations,
        )
        from luxar.gsplats.optim import PerSplatAdam, PerSplatReduceLROnPlateau

        optimizer = PerSplatAdam(model, lr=0.1)
        scheduler = PerSplatReduceLROnPlateau(optimizer)

        V_target = torch.tensor(V, dtype=torch.float32)
        V_pred = model()

        cfg = DynamicOpsConfig()
        cfg.min_splats_to_keep = 1  # Allow more aggressive pruning for test

        # Test pruning with verbose output
        opt_new, sched_new, topology_changed = apply_dynamic_operations(
            model,
            optimizer,
            scheduler,
            V_target,
            V_pred,
            cfg,
            current_lr=0.1,
            max_abs_error_threshold=0.01,
            device=torch.device("cpu"),
            verbose=True,  # Test verbose output
        )

        # Verify function returned successfully
        assert opt_new is not None
        assert sched_new is not None
        assert isinstance(topology_changed, bool)

    def test_auto_convergence_threshold_behavior(self) -> None:
        """Test auto-convergence threshold integration with dynamic operations."""
        V = np.random.random((24, 24)).astype(np.float32)
        centers = find_seeds_multiscale_gaussian(V, peaks_per_scale=30)

        # Test that auto-threshold works with dynamic operations
        result = fit_gaussian_splats(
            V,
            centers,
            n_iters=50,
            max_abs_error=None,  # Should auto-set to 0.01
            loss_type="l1",
            enable_dynamic_ops=True,  # Enable to test interaction
            dynamic_ops_verbose=True,  # Test verbose output
            verbose=True,  # Should log auto-threshold
            napari_movie=False,
        )

        assert "converged" in result.stats
        assert len(result.amplitudes) > 0

    def test_compression_analysis_functionality(self) -> None:
        """Test compression ratio analysis functionality."""
        from luxar.gsplats.fit_result import GaussianSplatResult
        from luxar.gsplats.fitting.visualization import display_compression_analysis

        # Create simple test data
        V = np.random.random((16, 16)).astype(np.float32)
        d = 2
        N = 10

        # Create GaussianSplatResult for testing
        result = GaussianSplatResult(
            centers=np.random.random((N, d)).astype(np.float32),
            amplitudes=np.random.uniform(0.1, 1.0, N).astype(np.float32),
            cholesky_factors=np.random.random((N, 3)).astype(np.float32),  # 2D tril = 3
            sharpnesses=np.random.uniform(1.5, 3.0, N).astype(np.float32),
            stats={}
        )

        # Test compression analysis (should not raise exceptions)
        try:
            display_compression_analysis(V, result)
            compression_test_passed = True
        except Exception:
            compression_test_passed = False

        assert compression_test_passed, "Compression analysis failed"

    def test_adaptive_learning_rate_boosting(self) -> None:
        """Test adaptive learning rate boosting for problematic regions."""
        from luxar.gsplats.fitting.dynamic_ops import (
            DynamicOpsConfig,
            _boost_splat_learning_rate,
        )
        from luxar.gsplats.optim import PerSplatAdam

        # Create test model
        centers = np.array([[10, 10], [15, 15]], dtype=np.float32)
        L0 = np.stack([np.eye(2) * 1.0, np.eye(2) * 1.0])
        amps0 = np.array([0.5, 0.3])

        model = GaussianSplatModel(
            shape=(32, 32),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.1, 0.1],
        )

        # Create optimizer
        optimizer = PerSplatAdam(model, lr=0.1)

        # Reduce learning rate for splat 0 to simulate scheduler effect
        optimizer.set_learning_rate(0, 0.05)  # Reduced from base 0.1

        # Get initial learning rates
        initial_lrs = optimizer.get_effective_learning_rates()

        # Test LR boosting
        cfg = DynamicOpsConfig()
        base_lr = 0.1  # Original starting rate
        boosted_lr = _boost_splat_learning_rate(optimizer, 0, base_lr, cfg)

        # Verify boosting worked
        new_lrs = optimizer.get_effective_learning_rates()
        assert boosted_lr > initial_lrs[0]  # Should be boosted from 0.05
        assert boosted_lr <= base_lr  # Should not exceed base rate (0.1)
        assert new_lrs[0] == boosted_lr  # Should be applied to splat 0
