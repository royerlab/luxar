"""
Tests for dynamic Gaussian splat operations.

Tests the four core operations: prune, seed, merge, split as well as
integration with the GaussianSplatModel and fitting process.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.candidates import find_candidates_overcomplete_nd
from luxar.gsplats.dynamic_ops import (
    DynamicOpsConfig,
    _estimate_amp_from_residual,
    _local_maxima,
    _moment_match_merge,
    _structure_tensor_eigs_nd,
    _sym_kl_gaussians,
)
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel


class TestDynamicOpsConfig:
    """Test dynamic operations configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = DynamicOpsConfig()
        assert cfg.amp_abs_min == 1e-4
        assert cfg.step_every == 50
        assert cfg.do_prune is True
        assert cfg.do_seed is True  # Current default
        assert cfg.do_merge is False  # Disabled by default
        assert cfg.do_split is False  # Disabled by default


class TestStructureTensor:
    """Test structure tensor computation for anisotropic seeding."""

    def test_structure_tensor_2d(self):
        """Test 2D structure tensor computation."""
        # Create a simple edge pattern
        res = torch.zeros((32, 32))
        res[10:20, :] = 1.0  # horizontal edge

        eigvals, eigvecs = _structure_tensor_eigs_nd(res)

        assert eigvals is not None
        assert eigvecs is not None
        assert eigvals.shape == (32, 32)
        assert eigvecs.shape == (2, 32, 32)

        # Check that eigenvectors are normalized
        norms = torch.sqrt(eigvecs[0] ** 2 + eigvecs[1] ** 2)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_structure_tensor_3d_placeholder(self):
        """Test 3D structure tensor (returns placeholders)."""
        res = torch.zeros((16, 16, 16))
        res[8:10, :, :] = 1.0

        eigvals, eigvecs = _structure_tensor_eigs_nd(res)

        # 3D returns None placeholders for now
        assert eigvals is None
        assert eigvecs is None


class TestLocalMaxima:
    """Test local maxima detection for seeding."""

    def test_local_maxima_2d(self):
        """Test local maxima detection in 2D."""
        # Create image with known peaks
        res = torch.zeros((32, 32))
        res[10, 10] = 1.0
        res[20, 20] = 0.8
        res[5, 25] = 0.6

        coords = _local_maxima(res, k=5, min_dist=3, thr=0.5)

        assert len(coords) == 3
        assert (10, 10) in coords
        assert (20, 20) in coords
        assert (5, 25) in coords

    def test_local_maxima_threshold(self):
        """Test threshold filtering."""
        res = torch.zeros((16, 16))
        res[5, 5] = 1.0
        res[10, 10] = 0.3  # Below threshold

        coords = _local_maxima(res, k=5, min_dist=2, thr=0.5)

        assert len(coords) == 1
        assert (5, 5) in coords
        assert (10, 10) not in coords

    def test_local_maxima_3d(self):
        """Test local maxima detection in 3D."""
        res = torch.zeros((16, 16, 16))
        res[8, 8, 8] = 1.0
        res[4, 4, 4] = 0.7

        coords = _local_maxima(res, k=5, min_dist=3, thr=0.5)

        assert len(coords) == 2
        assert (8, 8, 8) in coords
        assert (4, 4, 4) in coords


class TestMomentMatchMerge:
    """Test moment matching for merging Gaussians."""

    def test_moment_match_merge(self):
        """Test merging two Gaussians."""
        device = torch.device("cpu")

        # Two simple 2D Gaussians
        mu1 = torch.tensor([1.0, 1.0], device=device)
        mu2 = torch.tensor([2.0, 2.0], device=device)
        L1 = torch.eye(2, device=device) * 0.5
        L2 = torch.eye(2, device=device) * 0.7
        a1 = torch.tensor(0.4, device=device)
        a2 = torch.tensor(0.6, device=device)

        mu, L, a = _moment_match_merge(mu1, L1, a1, mu2, L2, a2)

        # Check merged amplitude
        assert torch.allclose(a, a1 + a2)

        # Check merged center is weighted average
        expected_mu = (a1 * mu1 + a2 * mu2) / (a1 + a2)
        assert torch.allclose(mu, expected_mu, atol=1e-5)

        # Check that L is lower triangular
        assert torch.allclose(L.triu(diagonal=1), torch.zeros_like(L.triu(diagonal=1)))

        # Check that merged covariance is positive definite
        Sigma = L @ L.T
        eigvals = torch.linalg.eigvals(Sigma)
        # Use real part in case of numerical errors creating tiny imaginary components
        assert torch.all(eigvals.real > 0)

    def test_degenerate_merge(self):
        """Test merging when total amplitude is zero."""
        device = torch.device("cpu")

        mu1 = torch.tensor([1.0, 1.0], device=device)
        mu2 = torch.tensor([2.0, 2.0], device=device)
        L1 = torch.eye(2, device=device)
        L2 = torch.eye(2, device=device)
        a1 = torch.tensor(0.0, device=device)
        a2 = torch.tensor(0.0, device=device)

        mu, L, a = _moment_match_merge(mu1, L1, a1, mu2, L2, a2)

        assert torch.allclose(a, torch.tensor(0.0))


class TestSymmetricKL:
    """Test symmetric KL divergence computation."""

    def test_symmetric_kl_identical(self):
        """Test KL divergence between identical Gaussians."""
        device = torch.device("cpu")

        mu = torch.tensor([1.0, 2.0], device=device)
        L = torch.eye(2, device=device) * 0.5

        kl = _sym_kl_gaussians(mu, L, mu, L)

        assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)

    def test_symmetric_kl_different(self):
        """Test KL divergence between different Gaussians."""
        device = torch.device("cpu")

        mu1 = torch.tensor([0.0, 0.0], device=device)
        mu2 = torch.tensor([1.0, 1.0], device=device)
        L1 = torch.eye(2, device=device) * 0.5
        L2 = torch.eye(2, device=device) * 1.0

        kl = _sym_kl_gaussians(mu1, L1, mu2, L2)

        assert kl > 0
        assert torch.isfinite(kl)

    def test_symmetric_property(self):
        """Test that KL divergence is symmetric."""
        device = torch.device("cpu")

        mu1 = torch.tensor([0.0, 0.0], device=device)
        mu2 = torch.tensor([2.0, 1.0], device=device)
        L1 = torch.eye(2, device=device) * 0.3
        L2 = torch.diag(torch.tensor([0.8, 0.5], device=device))

        kl12 = _sym_kl_gaussians(mu1, L1, mu2, L2)
        kl21 = _sym_kl_gaussians(mu2, L2, mu1, L1)

        assert torch.allclose(kl12, kl21, atol=1e-5)


class TestAmplitudeEstimation:
    """Test amplitude estimation from residual."""

    def test_amplitude_estimation(self):
        """Test amplitude estimation for new splat."""
        device = torch.device("cpu")
        shape = (32, 32)

        # Create a simple residual pattern
        residual = torch.zeros(shape, device=device)
        residual[15:17, 15:17] = 0.5  # Small bright region

        center = torch.tensor([16.0, 16.0], device=device)
        L = torch.eye(2, device=device) * 1.0

        amp = _estimate_amp_from_residual(shape, center, L, residual)

        assert amp > 0
        assert torch.isfinite(amp)


class TestGaussianSplatModelDynamicMethods:
    """Test dynamic management methods in GaussianSplatModel."""

    @staticmethod
    def create_test_model(n_splats=5):
        """Create a test model with known parameters."""
        shape = (32, 32)
        centers0 = np.random.uniform(5, 25, (n_splats, 2)).astype(np.float32)
        L0 = np.stack([np.eye(2) * 1.5] * n_splats).astype(np.float32)
        amps0 = np.random.uniform(0.1, 1.0, n_splats).astype(np.float32)
        sigma_min_diag = [0.5, 0.5]

        return GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            device=torch.device("cpu"),
        )

    def test_n_splats(self):
        """Test n_splats method."""
        model = self.create_test_model(7)
        assert model.n_splats() == 7

    def test_prune(self):
        """Test pruning operation."""
        model = self.create_test_model(5)
        model.n_splats()

        # Prune every other splat
        keep_mask = torch.tensor([True, False, True, False, True])
        model.prune_(keep_mask)

        assert model.n_splats() == 3
        assert model.raw_mu.shape[0] == 3
        assert model.raw_L_diag.shape[0] == 3
        assert model.raw_a.shape[0] == 3

    def test_append(self):
        """Test appending new splats."""
        model = self.create_test_model(3)
        original_n = model.n_splats()

        # Create new splats to append
        device = torch.device("cpu")
        centers_new = torch.tensor([[10.0, 10.0], [20.0, 20.0]], device=device)
        Ls_new = torch.stack([torch.eye(2) * 1.0, torch.eye(2) * 1.2], dim=0).to(device)
        amps_new = torch.tensor([0.5, 0.7], device=device)

        model.append_(centers_new, Ls_new, amps_new)

        assert model.n_splats() == original_n + 2
        assert model.raw_mu.shape[0] == 5

    def test_replace_with(self):
        """Test complete parameter replacement."""
        model = self.create_test_model(3)

        # Create new parameters
        device = torch.device("cpu")
        centers_new = torch.tensor([[5.0, 5.0], [15.0, 15.0]], device=device)
        Ls_new = torch.stack([torch.eye(2) * 0.8, torch.eye(2) * 1.1], dim=0).to(device)
        amps_new = torch.tensor([0.3, 0.8], device=device)

        model.replace_with(centers_new, Ls_new, amps_new)

        assert model.n_splats() == 2

        # Verify parameters are correctly set
        centers, Ls, amps = model.current_params()
        assert centers.shape[0] == 2
        assert torch.allclose(amps, amps_new, atol=1e-3)


class TestIntegration:
    """Test integration with the full fitting process."""

    @staticmethod
    def create_test_data():
        """Create synthetic test data."""
        shape = (32, 32)
        data = np.zeros(shape, dtype=np.float32)

        # Add a few Gaussian blobs
        for _ in range(3):
            center_y = np.random.randint(8, 24)
            center_x = np.random.randint(8, 24)
            yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
            sigma = np.random.uniform(2, 4)
            amplitude = np.random.uniform(0.3, 0.8)
            blob = amplitude * np.exp(
                -((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2 * sigma**2)
            )
            data += blob

        return np.clip(data, 0, 1)

    def test_fit_with_dynamic_ops(self):
        """Test fitting with dynamic operations enabled."""
        # Create test data
        image = self.create_test_data()

        # Find initial candidates
        candidates = find_candidates_overcomplete_nd(
            image,
            scales=(1.0, 2.0),
            peaks_per_scale=20,
            percentile_thresh=70,
            min_dist=2.0,
        )

        # Configure dynamic operations
        cfg = DynamicOpsConfig()
        cfg.step_every = 5  # Run more frequently for testing
        cfg.max_add_per_step = 10
        cfg.max_merges_per_step = 10

        # Fit with dynamic operations
        params, amps, stats = fit_gaussian_splats(
            image,
            centers_overcomplete=candidates,
            n_iters=30,
            lr=0.1,
            verbose=False,
            early_stopping=False,  # Disable for predictable testing
            enable_dynamic_ops=True,
            dynamic_config=cfg,
        )

        assert params.shape[0] > 0  # Should have some splats
        assert amps.shape[0] == params.shape[0]
        assert np.all(amps >= 0)  # Amplitudes should be non-negative
        assert np.all(np.isfinite(params))  # Parameters should be finite
        assert np.all(np.isfinite(amps))

        # Check that we got reasonable reconstruction
        from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians_numpy

        reconstruction = render_gaussians_numpy(image.shape, params, amps)
        mse = np.mean((image - reconstruction) ** 2)
        assert mse < 0.2  # Should achieve reasonable reconstruction

    def test_fit_without_dynamic_ops(self):
        """Test that fitting works without dynamic operations (baseline)."""
        image = self.create_test_data()

        candidates = find_candidates_overcomplete_nd(
            image,
            scales=(1.0, 2.0),
            peaks_per_scale=15,
            percentile_thresh=75,
            min_dist=2.0,
        )

        # Fit without dynamic operations
        params, amps, stats = fit_gaussian_splats(
            image,
            centers_overcomplete=candidates,
            n_iters=30,
            lr=0.1,
            verbose=False,
            early_stopping=False,
            enable_dynamic_ops=False,  # Disabled
        )

        assert params.shape[0] == len(candidates)  # Should preserve initial count
        assert amps.shape[0] == params.shape[0]
        assert np.all(amps >= 0)
        assert np.all(np.isfinite(params))
        assert np.all(np.isfinite(amps))


class TestConfigurability:
    """Test configuration options for dynamic operations."""

    def test_disable_individual_operations(self):
        """Test disabling individual dynamic operations."""
        cfg = DynamicOpsConfig()

        # Test enabling/disabling each operation
        cfg.do_prune = False
        assert not cfg.do_prune

        cfg.do_seed = False
        assert not cfg.do_seed

        cfg.do_merge = False
        assert not cfg.do_merge

        cfg.do_split = False
        assert not cfg.do_split

    def test_parameter_bounds(self):
        """Test parameter bounds are reasonable."""
        cfg = DynamicOpsConfig()

        assert cfg.amp_abs_min > 0
        assert cfg.merge_dist_vox > 0
        assert cfg.split_shrink > 0 and cfg.split_shrink < 1
        assert cfg.residual_quantile > 0 and cfg.residual_quantile < 1
        assert cfg.step_every > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
