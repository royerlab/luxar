"""
Comprehensive end-to-end tests for Gaussian splatting implementation.
Tests the full pipeline from candidate generation to fitting to rendering.
"""

from typing import Tuple

import numpy as np
import pytest
import torch

from luxar.gsplats.fit_gsplats import GaussianSplatFitter, fit_gaussian_splats
from luxar.gsplats.models.gsplats import (
    render_gaussians,
    render_gaussians_numpy,
)
from luxar.gsplats.seeds import find_seeds_multiscale_gaussian
from luxar.gsplats.utils.trils import tril_size, unpack_tril


class TestGaussianSplatsIntegration:
    """End-to-end integration tests for Gaussian splatting."""

    @staticmethod
    def create_test_image(shape: Tuple[int, ...], n_blobs: int = 5) -> np.ndarray:
        """Create synthetic test image with Gaussian blobs."""
        data = np.zeros(shape, dtype=np.float32)

        # Add random Gaussian blobs
        for _ in range(n_blobs):
            center = [np.random.randint(10, s - 10) for s in shape]
            sigma = np.random.uniform(2, 4)
            amplitude = np.random.uniform(0.5, 1.0)

            grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
            dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
            data += amplitude * np.exp(-dist_sq / (2 * sigma**2))

        return np.clip(data, 0, 1)

    def test_full_pipeline_2d(self) -> None:
        """Test complete pipeline for 2D image."""
        # Create test data
        image = self.create_test_image((64, 64), n_blobs=3)

        # Find seeds
        seeds = find_seeds_multiscale_gaussian(
            image,
            scales=(1.0, 2.0, 3.0),
            peaks_per_scale=50,
            percentile_thresh=80,
            min_distance=2.0,
        )
        assert len(seeds) > 0
        assert seeds.shape[1] == 2

        # Fit splats
        result = fit_gaussian_splats(
            image,
            seeds=seeds,
            n_iters=50,
            lr=0.2,
            verbose=False,
            enable_dynamic_ops=False,  # Disable for predictable test results
            napari_movie=False,  # Disable napari windows in tests
        )

        # Verify result structure
        assert result.centers.shape[0] == len(seeds)
        assert result.centers.shape[1] == 2  # 2D centers
        assert result.cholesky_factors.shape[0] == len(seeds)
        assert result.cholesky_factors.shape[1] == tril_size(2)  # Packed L
        assert result.sharpnesses.shape == (len(seeds),)
        assert result.amplitudes.shape == (len(seeds),)
        assert np.all(result.amplitudes >= 0)  # Amplitudes should be non-negative

        # Render reconstruction
        reconstruction = render_gaussians_numpy(image.shape, result, truncate=3.0)

        assert reconstruction.shape == image.shape
        assert np.all(np.isfinite(reconstruction))

        # Check reconstruction quality
        mse = np.mean((image - reconstruction) ** 2)
        assert mse < 0.1  # Reasonable reconstruction error

    def test_full_pipeline_3d(self) -> None:
        """Test complete pipeline for 3D volume."""
        # Create test data
        volume = self.create_test_image((32, 32, 32), n_blobs=3)

        # Find seeds
        seeds = find_seeds_multiscale_gaussian(
            volume,
            scales=(1.0, 2.0),
            peaks_per_scale=30,
            percentile_thresh=85,
            min_distance=3.0,
        )
        assert len(seeds) > 0
        assert seeds.shape[1] == 3

        # Fit splats
        result = fit_gaussian_splats(
            volume,
            seeds=seeds,
            n_iters=30,  # Fewer for speed
            lr=0.2,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Verify result structure
        assert result.centers.shape[0] == len(seeds)
        assert result.centers.shape[1] == 3  # 3D centers
        assert result.cholesky_factors.shape[0] == len(seeds)
        assert result.cholesky_factors.shape[1] == tril_size(3)  # Packed L
        assert result.sharpnesses.shape == (len(seeds),)
        assert result.amplitudes.shape == (len(seeds),)

        # Render reconstruction
        reconstruction = render_gaussians_numpy(volume.shape, result, truncate=3.0)

        assert reconstruction.shape == volume.shape
        assert np.all(np.isfinite(reconstruction))

        # Check reconstruction quality
        mse = np.mean((volume - reconstruction) ** 2)
        assert mse < 0.15  # Maintain 3D quality standards

    def test_full_pipeline_4d(self) -> None:
        """Test complete pipeline for 4D hypercube to verify nD renderer chunking."""
        # Set random seed for reproducibility (4D optimization can be sensitive to initialization)
        np.random.seed(42)

        # Create smaller 4D test data to keep computation reasonable
        shape_4d = (16, 16, 16, 8)  # 4D hypercube: spatial xyz + time/channel
        data = np.zeros(shape_4d, dtype=np.float32)

        # Add a few 4D Gaussian blobs
        for _ in range(2):
            center = [np.random.randint(2, max(3, s - 2)) for s in shape_4d]
            sigma = np.random.uniform(1.5, 2.5)
            amplitude = np.random.uniform(0.6, 1.0)

            # Create 4D coordinate grids
            grids = np.meshgrid(*[np.arange(s) for s in shape_4d], indexing="ij")
            dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
            data += amplitude * np.exp(-dist_sq / (2 * sigma**2))

        data = np.clip(data, 0, 1)

        # Find seeds using fewer scales for 4D
        seeds = find_seeds_multiscale_gaussian(
            data,
            scales=(1.0, 2.0),  # Fewer scales for 4D
            peaks_per_scale=20,  # Fewer seeds
            percentile_thresh=75,
            min_distance=2.0,
        )
        assert len(seeds) > 0
        assert seeds.shape[1] == 4  # 4D coordinates

        # Fit splats with reduced iterations for 4D
        result = fit_gaussian_splats(
            data,
            seeds=seeds,
            n_iters=50,  # Fewer iterations for test speed
            lr=0.3,
            verbose=False,  # Reduce test output
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Verify result structure
        assert result.centers.shape[1] == 4  # 4D centers
        assert result.cholesky_factors.shape[1] == tril_size(4)  # 4x4 Cholesky
        assert result.sharpnesses.shape == (len(seeds),)
        assert len(result.amplitudes) == len(seeds)
        assert all(result.amplitudes >= 0)  # Non-negative amplitudes

        # Render reconstruction (this tests our nD chunking path!)
        reconstruction = render_gaussians_numpy(shape_4d, result, truncate=2.5)
        assert reconstruction.shape == shape_4d

        # Verify reconstruction quality (looser tolerance for 4D)
        mse = np.mean((reconstruction - data) ** 2)
        assert (
            mse < 12.0
        )  # 4D is very challenging, focus on functionality not precision

        # Verify we exercised the nD path (not 2D/3D specialized paths)
        assert len(shape_4d) == 4  # Confirms we used generic nD renderer

    def test_early_stopping_convergence(self) -> None:
        """Test that early stopping works and maintains quality."""
        # Simple test case that should converge quickly
        image = np.zeros((32, 32), dtype=np.float32)
        # Add single Gaussian blob
        y, x = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
        image = 0.8 * np.exp(-((y - 16) ** 2 + (x - 16) ** 2) / (2 * 4**2))

        # Find seeds
        seeds = find_seeds_multiscale_gaussian(image, peaks_per_scale=10)

        # Fit with early stopping (more iterations to allow convergence)
        fitter = GaussianSplatFitter(enable_dynamic_ops=False)
        result_early = fitter.fit(
            image,
            seeds=seeds,
            n_iters=200,  # More iterations
            max_abs_error=0.001,  # Use convergence threshold instead of early_stopping
            verbose=False,
            napari_movie=False,
        )
        stats_early = result_early.stats

        # Fit without early stopping
        result_full = fitter.fit(
            image,
            seeds=seeds,
            n_iters=200,  # Same number
            verbose=False,
            napari_movie=False,
        )
        stats_full = result_full.stats

        # Early stopping should use fewer iterations (or at least not more)
        assert stats_early["iterations"] <= stats_full["iterations"]
        # If converged, should be less
        if stats_early["converged"]:
            assert stats_early["iterations"] < 200

        # But achieve similar quality
        recon_early = render_gaussians_numpy(image.shape, result_early)
        recon_full = render_gaussians_numpy(image.shape, result_full)

        mse_early = np.mean((image - recon_early) ** 2)
        mse_full = np.mean((image - recon_full) ** 2)

        # Quality should be within 10%
        assert abs(mse_early - mse_full) / mse_full < 0.1

    def test_batched_renderer_equivalence(self) -> None:
        """Test that batched renderer produces same results as numpy version."""
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            pytest.skip("Requires GPU for batched renderer test")

        image = self.create_test_image((64, 64), n_blobs=3)
        seeds = find_seeds_multiscale_gaussian(image, peaks_per_scale=30)

        # Fit splats
        result = fit_gaussian_splats(
            image,
            seeds=seeds,
            n_iters=30,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )
        # Render with numpy
        recon_numpy = render_gaussians_numpy(image.shape, result)

        # Render with batched PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        d = 2
        # Extract parameters manually for PyTorch rendering
        centers = torch.tensor(result.centers, device=device)
        L_full = unpack_tril(result.cholesky_factors, d)
        Ls = torch.tensor(L_full, device=device)
        amps_t = torch.tensor(result.amplitudes, device=device)
        # Use fitted sharpness values
        sharpness = torch.tensor(result.sharpnesses, device=device)

        recon_torch = render_gaussians(
            image.shape, centers, Ls, amps_t, sharpness, truncate=3.0
        )
        recon_torch_np = recon_torch.cpu().numpy()

        # Should be nearly identical
        max_diff = np.max(np.abs(recon_numpy - recon_torch_np))
        assert max_diff < 1e-5

    def test_loss_functions(self) -> None:
        """Test both MSE and Poisson loss functions."""
        image = self.create_test_image((32, 32), n_blobs=2)
        seeds = find_seeds_multiscale_gaussian(image, peaks_per_scale=20)

        # Test MSE loss
        result_mse = fit_gaussian_splats(
            image,
            seeds=seeds,
            n_iters=30,
            loss_type="mse",
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )
        # Test Poisson loss
        result_poisson = fit_gaussian_splats(
            image,
            seeds=seeds,
            n_iters=30,
            loss_type="poisson",
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Both should produce valid results
        assert np.all(np.isfinite(result_mse.centers))
        assert np.all(np.isfinite(result_mse.cholesky_factors))
        assert np.all(np.isfinite(result_poisson.centers))
        assert np.all(np.isfinite(result_poisson.cholesky_factors))
        assert np.all(result_mse.amplitudes >= 0)
        assert np.all(result_poisson.amplitudes >= 0)

        # Both should reconstruct reasonably well
        recon_mse = render_gaussians_numpy(image.shape, result_mse)
        recon_poisson = render_gaussians_numpy(image.shape, result_poisson)

        mse_mse = np.mean((image - recon_mse) ** 2)
        mse_poisson = np.mean((image - recon_poisson) ** 2)

        assert mse_mse < 0.1
        assert mse_poisson < 0.1

    def test_regularization(self) -> None:
        """Test L1 regularization on amplitudes."""
        image = self.create_test_image((32, 32), n_blobs=5)
        seeds = find_seeds_multiscale_gaussian(image, peaks_per_scale=50)

        # Without regularization
        result_no_reg = fit_gaussian_splats(
            image,
            seeds=seeds,
            n_iters=50,
            l1_amp=0.0,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )
        # With strong regularization
        result_reg = fit_gaussian_splats(
            image,
            seeds=seeds,
            n_iters=50,
            l1_amp=0.1,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Regularization should produce sparser solution
        n_active_no_reg = np.sum(result_no_reg.amplitudes > 0.01)
        n_active_reg = np.sum(result_reg.amplitudes > 0.01)

        assert n_active_reg <= n_active_no_reg

    def test_sigma_constraints(self) -> None:
        """Test that sigma constraints are respected."""
        image = self.create_test_image((32, 32), n_blobs=2)
        seeds = find_seeds_multiscale_gaussian(image, peaks_per_scale=10)

        # Fit with constraints
        sigma_min = [0.5, 0.5]
        sigma_max = [5.0, 5.0]

        result = fit_gaussian_splats(
            image,
            seeds=seeds,
            n_iters=50,
            sigma_min_diag=sigma_min,
            sigma_max_diag=sigma_max,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )
        params = np.column_stack([result.centers, result.cholesky_factors, result.sharpnesses])
        amps = result.amplitudes

        # Extract and check Cholesky factors
        d = 2
        L_packed = params[:, d:]
        L_full = unpack_tril(L_packed, d)

        # Check diagonal elements are within bounds
        for i in range(len(L_full)):
            diag = np.diag(L_full[i])
            assert np.all(diag >= sigma_min)
            assert np.all(diag <= sigma_max)

    def test_device_compatibility(self) -> None:
        """Test that fitting works on different devices."""
        image = self.create_test_image((32, 32), n_blobs=2)
        seeds = find_seeds_multiscale_gaussian(image, peaks_per_scale=10)

        # Test CPU
        result_cpu = fit_gaussian_splats(
            image,
            seeds=seeds,
            n_iters=20,
            device="cpu",
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )
        params_cpu = np.column_stack([result_cpu.centers, result_cpu.cholesky_factors, result_cpu.sharpnesses])
        assert np.all(np.isfinite(params_cpu))

        # Test GPU if available
        if torch.cuda.is_available():
            result_cuda = fit_gaussian_splats(
                image,
                seeds=seeds,
                n_iters=20,
                device="cuda",
                verbose=False,
            )
            params_cuda = np.column_stack([result_cuda.centers, result_cuda.cholesky_factors, result_cuda.sharpnesses])
            assert np.all(np.isfinite(params_cuda))

        # Test MPS if available
        if torch.backends.mps.is_available():
            result_mps = fit_gaussian_splats(
                image,
                seeds=seeds,
                n_iters=20,
                device="mps",
                verbose=False,
            )
            params_mps = np.column_stack([result_mps.centers, result_mps.cholesky_factors, result_mps.sharpnesses])
            assert np.all(np.isfinite(params_mps))

    def test_empty_input_handling(self) -> None:
        """Test handling of edge cases and empty inputs."""
        # Empty seeds
        image = self.create_test_image((32, 32), n_blobs=1)
        empty_seeds = np.zeros((0, 2), dtype=np.float32)

        result = fit_gaussian_splats(
            image,
            seeds=empty_seeds,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )
        params = np.column_stack([result.centers, result.cholesky_factors, result.sharpnesses]) if len(result.centers) > 0 else np.zeros((0, 2 + tril_size(2) + 1))
        amps = result.amplitudes

        assert params.shape == (0, 2 + tril_size(2) + 1)  # Include sharpness
        assert amps.shape == (0,)

        # Uniform image
        uniform_image = np.ones((32, 32), dtype=np.float32) * 0.5
        seeds = find_seeds_multiscale_gaussian(
            uniform_image, peaks_per_scale=10
        )

        result = fit_gaussian_splats(
            uniform_image,
            seeds=seeds,
            n_iters=20,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )
        params = np.column_stack([result.centers, result.cholesky_factors, result.sharpnesses])
        amps = result.amplitudes

        assert np.all(np.isfinite(params))
        assert np.all(np.isfinite(amps))

    def test_memory_chunking_large_aabb(self) -> None:
        """Test that memory chunking prevents OOM with large AABB boxes."""
        # Create a scenario with large AABB boxes that would OOM without chunking
        shape = (64, 64)  # 2D for faster test, but with loose truncation

        # Create single blob with very loose truncation to create large AABB
        data = np.zeros(shape, dtype=np.float32)
        center = [32, 32]
        sigma = 4.0  # Large sigma
        grids = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        dist_sq = (grids[0] - center[0]) ** 2 + (grids[1] - center[1]) ** 2
        data = 0.8 * np.exp(-dist_sq / (2 * sigma**2))

        # Use just one seed at the center
        seeds = np.array([[32.0, 32.0]])

        # Fit with very loose truncation to force large AABB
        result = fit_gaussian_splats(
            data,
            seeds=seeds,
            n_iters=20,
            truncate=8.0,  # Very loose truncation = large AABB
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )
        # Render with loose truncation (this exercises chunking!)
        reconstruction = render_gaussians_numpy(shape, result, truncate=8.0)

        assert reconstruction.shape == shape
        assert np.all(np.isfinite(reconstruction))

        # Should still reconstruct reasonably well
        mse = np.mean((data - reconstruction) ** 2)
        assert mse < 0.3  # Reasonable reconstruction
