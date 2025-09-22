"""
Tests for dynamic Gaussian splat operations with residual-driven approach.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.candidates import find_candidates_overcomplete_nd
from luxar.gsplats.dynamic_ops import (
    DynamicOpsConfig,
    _estimate_amplitude_from_residual,
    _find_residual_peaks,
    apply_dynamic_operations,
)
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel


class TestDynamicOpsConfig:
    """Test dynamic operations configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = DynamicOpsConfig()
        assert cfg.step_every == 50
        assert cfg.k_max_residuals == 10
        assert cfg.nms_radius_vox == 2.0
        assert cfg.min_contribution_threshold == 0.05
        assert cfg.relative_contribution_factor == 0.1
        assert cfg.learning_rate_threshold == 1e-6
        assert cfg.init_sigma_vox == 1.5
        assert cfg.split_size_threshold == 3.0
        assert cfg.split_elongation_threshold == 4.0

    def test_config_modification(self):
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

    def test_find_residual_peaks_2d(self):
        """Test finding residual peaks in 2D images."""
        # Create a synthetic residual with clear peaks
        residual = torch.zeros((20, 20))
        residual[5, 5] = 1.0   # Peak 1
        residual[15, 15] = 0.8 # Peak 2
        residual[10, 10] = 0.6 # Peak 3

        peaks = _find_residual_peaks(residual, k_max_residuals=3, nms_radius_vox=2.0)

        assert len(peaks) <= 3
        assert len(peaks) > 0

        # Check that peaks contain the expected locations
        peak_locations = set(peaks)
        assert (5, 5) in peak_locations
        assert (15, 15) in peak_locations

    def test_find_residual_peaks_3d(self):
        """Test finding residual peaks in 3D volumes."""
        # Create a synthetic 3D residual
        residual = torch.zeros((10, 10, 10))
        residual[5, 5, 5] = 1.0   # Peak 1
        residual[2, 2, 2] = 0.7   # Peak 2

        peaks = _find_residual_peaks(residual, k_max_residuals=2, nms_radius_vox=1.5)

        assert len(peaks) <= 2
        assert len(peaks) > 0

        # Check that we get 3D coordinates
        for peak in peaks:
            assert len(peak) == 3

    def test_find_residual_peaks_empty(self):
        """Test behavior with no significant peaks."""
        residual = torch.zeros((10, 10))
        peaks = _find_residual_peaks(residual, k_max_residuals=5, nms_radius_vox=2.0)

        # Should return empty list for zero residual
        assert len(peaks) == 0


class TestAmplitudeEstimation:
    """Test amplitude estimation functionality."""

    def test_amplitude_estimation_2d(self):
        """Test amplitude estimation in 2D."""
        shape = (20, 20)
        center = torch.tensor([10.0, 10.0])
        L = torch.eye(2) * 1.5  # Isotropic covariance

        # Create synthetic residual
        residual = torch.ones(shape) * 0.5

        amplitude = _estimate_amplitude_from_residual(shape, center, L, residual)

        assert isinstance(amplitude, torch.Tensor)
        assert amplitude.item() >= 0.0  # Should be non-negative

    def test_amplitude_estimation_3d(self):
        """Test amplitude estimation in 3D."""
        shape = (10, 10, 10)
        center = torch.tensor([5.0, 5.0, 5.0])
        L = torch.eye(3) * 2.0

        residual = torch.ones(shape) * 0.3

        amplitude = _estimate_amplitude_from_residual(shape, center, L, residual)

        assert isinstance(amplitude, torch.Tensor)
        assert amplitude.item() >= 0.0


class TestGaussianSplatModel:
    """Test basic Gaussian splat model operations required for dynamic ops."""

    def test_n_splats(self):
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
            sigma_min_diag=[0.5, 0.5]
        )
        assert model.n_splats() == 2

    def test_current_params(self):
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
            sigma_min_diag=[0.5, 0.5]
        )

        centers_t, Ls_t, amps_t = model.current_params()

        assert centers_t.shape == (2, 2)
        assert Ls_t.shape == (2, 2, 2)
        assert amps_t.shape == (2,)


class TestDynamicOperationsIntegration:
    """Test the full dynamic operations pipeline."""

    def test_dynamic_ops_with_simple_model(self):
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
            sigma_min_diag=[0.5, 0.5]
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
            device=torch.device('cpu'),
            verbose=False,
        )

        # Check that we get valid returns
        assert opt_new is not None
        assert sched_new is not None
        assert isinstance(topology_changed, bool)

    def test_fit_with_dynamic_ops(self):
        """Test full fitting pipeline with dynamic operations enabled."""
        # Create simple test data
        blob = np.zeros((32, 32))
        blob[16, 16] = 1.0
        blob[12, 12] = 0.8
        V = blob.astype(np.float32)

        # Find candidates
        centers = find_candidates_overcomplete_nd(
            V, scales=(1.0, 2.0), peaks_per_scale=10, percentile_thresh=50.0
        )

        if len(centers) == 0:
            pytest.skip("No candidates found for test data")

        # Configure dynamic operations
        cfg = DynamicOpsConfig()
        cfg.step_every = 5  # Run more frequently for testing

        # Run fitting with dynamic operations
        params, amps, stats = fit_gaussian_splats(
            V,
            centers_overcomplete=centers,
            init_sigma_vox=1.5,
            n_iters=20,  # Short run for testing
            lr=0.1,
            enable_dynamic_ops=True,
            dynamic_config=cfg,
            verbose=False,
            napari_movie=False,  # Disable movie for tests
        )

        # Check that we got valid results
        assert params.shape[0] > 0  # Should have some splats
        assert amps.shape[0] == params.shape[0]
        assert 'final_loss' in stats  # Check for stats that actually exist

    def test_fit_without_dynamic_ops(self):
        """Test fitting without dynamic operations for comparison."""
        # Create simple test data
        blob = np.zeros((32, 32))
        blob[16, 16] = 1.0
        V = blob.astype(np.float32)

        # Find candidates
        centers = find_candidates_overcomplete_nd(
            V, scales=(1.0,), peaks_per_scale=5, percentile_thresh=50.0
        )

        if len(centers) == 0:
            pytest.skip("No candidates found for test data")

        # Run fitting without dynamic operations
        params, amps, stats = fit_gaussian_splats(
            V,
            centers_overcomplete=centers,
            init_sigma_vox=1.5,
            n_iters=10,
            lr=0.1,
            enable_dynamic_ops=False,
            verbose=False,
            napari_movie=False,  # Disable movie for tests
        )

        # Check that we got valid results
        assert params.shape[0] > 0
        assert amps.shape[0] == params.shape[0]
        assert 'final_loss' in stats  # Check for stats that actually exist
