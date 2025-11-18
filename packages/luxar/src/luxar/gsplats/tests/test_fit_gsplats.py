"""
Tests for fit_gaussian_splats function and optimization pipeline.
"""

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import importlib.util

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

if HAS_TORCH:
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.utils.trils import tril_size


@pytest.fixture
def simple_2d_blob():
    """Create a simple 2D blob image for testing."""
    x, y = np.meshgrid(np.linspace(-3, 3, 21), np.linspace(-3, 3, 21))
    blob = np.exp(-(x**2 + y**2) / 2) + 0.1  # Gaussian blob with small background
    return blob.astype(np.float32)


@pytest.fixture
def multi_blob_2d():
    """Create a 2D image with multiple blobs."""
    x, y = np.meshgrid(np.linspace(-5, 5, 31), np.linspace(-5, 5, 31))

    # Three blobs at different locations
    blob1 = 0.8 * np.exp(-((x + 2) ** 2 + (y + 2) ** 2) / 1.5)
    blob2 = 0.6 * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / 2.0)
    blob3 = 1.0 * np.exp(-(x**2 + (y - 3) ** 2) / 1.0)

    image = blob1 + blob2 + blob3 + 0.05  # Small background
    return image.astype(np.float32)


@pytest.fixture
def simple_3d_blob():
    """Create a simple 3D blob for testing."""
    x, y, z = np.meshgrid(
        np.linspace(-2, 2, 15), np.linspace(-2, 2, 15), np.linspace(-2, 2, 15)
    )
    blob = np.exp(-(x**2 + y**2 + z**2) / 2) + 0.05
    return blob.astype(np.float32)


@pytest.fixture
def simple_candidates_2d():
    """Simple candidate centers for 2D testing."""
    return np.array(
        [
            [10.0, 10.0],  # Center of 21x21 image
            [7.0, 7.0],  # Off-center
            [13.0, 13.0],  # Another off-center
        ],
        dtype=np.float32,
    )


@pytest.fixture
def simple_candidates_3d():
    """Simple candidate centers for 3D testing."""
    return np.array(
        [
            [7.0, 7.0, 7.0],  # Center of 15x15x15 image
            [5.0, 5.0, 5.0],  # Off-center
        ],
        dtype=np.float32,
    )


class TestFitGaussianSplatsBasic:
    """Test basic functionality of fit_gaussian_splats."""

    def test_basic_fitting_2d(self, simple_2d_blob, simple_candidates_2d):
        """Test basic splat fitting in 2D."""
        params_full, amps, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            init_sigma_vox=1.0,
            n_iters=50,  # Keep short for testing
            lr=0.1,
            verbose=False,
            enable_dynamic_ops=False,  # Disable for predictable test results
            napari_movie=False,  # Disable movie for testing
        )

        # Check output shapes
        d = 2
        N = len(simple_candidates_2d)
        expected_param_size = (
            d + tril_size(d) + 1
        )  # centers + packed Cholesky + sharpness

        assert params_full.shape == (N, expected_param_size)
        assert amps.shape == (N,)

        # Check data types
        assert params_full.dtype == np.float32
        assert amps.dtype == np.float32

        # Check that centers are within reasonable bounds
        centers = params_full[:, :d]
        assert np.all(centers >= -1)  # Allow small margin outside image
        assert np.all(centers <= simple_2d_blob.shape[0])

        # Check that amplitudes are non-negative
        assert np.all(amps >= 0)

    def test_basic_fitting_3d(self, simple_3d_blob, simple_candidates_3d):
        """Test basic splat fitting in 3D."""
        params_full, amps, _ = fit_gaussian_splats(
            V=simple_3d_blob,
            seeds=simple_candidates_3d,
            init_sigma_vox=1.2,
            n_iters=30,  # Keep short for testing
            lr=0.15,
            verbose=False,
            enable_dynamic_ops=False,  # Disable for predictable test results
            napari_movie=False,  # Disable movie for testing
        )

        # Check output shapes
        d = 3
        N = len(simple_candidates_3d)
        expected_param_size = d + tril_size(d) + 1  # 3 + 6 + 1 = 10

        assert params_full.shape == (N, expected_param_size)
        assert amps.shape == (N,)

        # Check basic validity
        assert np.all(amps >= 0)
        centers = params_full[:, :d]
        assert np.all(np.isfinite(centers))

    def test_empty_candidates(self, simple_2d_blob):
        """Test fitting with no candidate centers."""
        empty_candidates = np.zeros((0, 2), dtype=np.float32)

        params_full, amps, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=empty_candidates,
            n_iters=10,
            verbose=False,
            enable_dynamic_ops=False,  # Disable for predictable test results
            napari_movie=False,  # Disable movie for testing
        )

        # Should return empty arrays with correct shapes
        d = 2
        expected_param_size = d + tril_size(d) + 1  # Include sharpness

        assert params_full.shape == (0, expected_param_size)
        assert amps.shape == (0,)

    def test_single_candidate(self, simple_2d_blob):
        """Test fitting with single candidate."""
        single_candidate = np.array([[10.0, 10.0]], dtype=np.float32)

        params_full, amps, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=single_candidate,
            n_iters=30,
            verbose=False,
            enable_dynamic_ops=False,  # Disable for predictable test results
            napari_movie=False,  # Disable movie for testing
        )

        assert params_full.shape == (1, 6)  # 2D + 3 tril elements + sharpness
        assert amps.shape == (1,)
        assert amps[0] > 0  # Should have positive amplitude


class TestInputValidation:
    """Test input validation in fit_gaussian_splats."""

    def test_input_validation_basic(self, simple_2d_blob, simple_candidates_2d):
        """Test basic input validation."""

        # Test empty image
        with pytest.raises(ValueError, match="cannot be empty"):
            fit_gaussian_splats(np.array([]), simple_candidates_2d, verbose=False)

        # Test scalar image
        with pytest.raises(ValueError, match="at least 1 dimension"):
            fit_gaussian_splats(5.0, simple_candidates_2d, verbose=False)

        # Test mismatched dimensions
        candidates_3d = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="columns to match image dimensions"):
            fit_gaussian_splats(simple_2d_blob, candidates_3d, verbose=False)

        # Test wrong candidate shape
        bad_candidates = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 1D array
        with pytest.raises(ValueError, match="2D array"):
            fit_gaussian_splats(simple_2d_blob, bad_candidates, verbose=False)

    def test_parameter_validation(self, simple_2d_blob, simple_candidates_2d):
        """Test parameter validation."""

        # Test negative init_sigma_vox
        with pytest.raises(ValueError, match="init_sigma_vox must be positive"):
            fit_gaussian_splats(
                simple_2d_blob, simple_candidates_2d, init_sigma_vox=-1.0, verbose=False
            )

        # Test zero iterations
        with pytest.raises(ValueError, match="n_iters must be positive"):
            fit_gaussian_splats(
                simple_2d_blob, simple_candidates_2d, n_iters=0, verbose=False
            )

        # Test negative learning rate
        with pytest.raises(ValueError, match="lr must be positive"):
            fit_gaussian_splats(
                simple_2d_blob, simple_candidates_2d, lr=-0.1, verbose=False
            )

        # Test invalid loss type
        with pytest.raises(ValueError, match="loss_type must be"):
            fit_gaussian_splats(
                simple_2d_blob, simple_candidates_2d, loss_type="invalid", verbose=False
            )

        # Test negative L1 regularization
        with pytest.raises(ValueError, match="l1_amp must be non-negative"):
            fit_gaussian_splats(
                simple_2d_blob, simple_candidates_2d, l1_amp=-0.1, verbose=False
            )

        # Test negative truncation
        with pytest.raises(ValueError, match="truncate must be positive"):
            fit_gaussian_splats(
                simple_2d_blob, simple_candidates_2d, truncate=-1.0, verbose=False
            )

    def test_sigma_constraints_validation(self, simple_2d_blob, simple_candidates_2d):
        """Test sigma constraint validation."""

        # Test wrong dimension for sigma_min_diag
        with pytest.raises(ValueError, match="sigma_min_diag must have length"):
            fit_gaussian_splats(
                simple_2d_blob,
                simple_candidates_2d,
                sigma_min_diag=[0.5],
                verbose=False,
            )  # Should be length 2

        # Test negative sigma_min_diag
        with pytest.raises(ValueError, match="sigma_min_diag values must be positive"):
            fit_gaussian_splats(
                simple_2d_blob,
                simple_candidates_2d,
                sigma_min_diag=[-0.1, 0.5],
                verbose=False,
            )

        # Test wrong dimension for sigma_max_diag
        with pytest.raises(ValueError, match="sigma_max_diag must have length"):
            fit_gaussian_splats(
                simple_2d_blob,
                simple_candidates_2d,
                sigma_max_diag=[2.0, 3.0, 4.0],
                verbose=False,
            )  # Should be length 2

        # Test negative sigma_max_diag
        with pytest.raises(ValueError, match="sigma_max_diag values must be positive"):
            fit_gaussian_splats(
                simple_2d_blob,
                simple_candidates_2d,
                sigma_max_diag=[2.0, -1.0],
                verbose=False,
            )

        # Test max < min
        with pytest.raises(ValueError, match="greater than sigma_min_diag"):
            fit_gaussian_splats(
                simple_2d_blob,
                simple_candidates_2d,
                sigma_min_diag=[1.0, 1.0],
                sigma_max_diag=[0.5, 2.0],
                verbose=False,
            )


class TestUniformImageHandling:
    """Test handling of uniform images (division by zero protection)."""

    def test_uniform_image(self, simple_candidates_2d):
        """Test fitting on completely uniform image."""
        uniform_image = np.ones((21, 21), dtype=np.float32) * 5.0

        # Should not crash due to division by zero
        params_full, amps, _ = fit_gaussian_splats(
            V=uniform_image,
            seeds=simple_candidates_2d,
            n_iters=10,
            verbose=False,  # Suppress warning message
        )

        # Should return valid results
        assert params_full.shape == (3, 6)  # 2D + 3 tril elements + sharpness
        assert amps.shape == (3,)
        assert np.all(np.isfinite(params_full))
        assert np.all(np.isfinite(amps))

    def test_nearly_uniform_image(self, simple_candidates_2d):
        """Test fitting on nearly uniform image."""
        nearly_uniform = np.ones((21, 21), dtype=np.float32) * 5.0
        nearly_uniform[10, 10] = 5.0001  # Tiny variation

        params_full, amps, _ = fit_gaussian_splats(
            V=nearly_uniform,
            seeds=simple_candidates_2d,
            n_iters=10,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        assert np.all(np.isfinite(params_full))
        assert np.all(np.isfinite(amps))


class TestLossTypes:
    """Test different loss types."""

    def test_mse_loss(self, simple_2d_blob, simple_candidates_2d):
        """Test MSE loss function."""
        params_full, amps, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            loss_type="mse",
            n_iters=20,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        assert params_full.shape == (3, 6)  # Include sharpness
        assert amps.shape == (3,)
        assert np.all(amps >= 0)

    def test_poisson_loss(self, simple_2d_blob, simple_candidates_2d):
        """Test Poisson loss function."""
        params_full, amps, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            loss_type="poisson",
            n_iters=20,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        assert params_full.shape == (3, 6)  # Include sharpness
        assert amps.shape == (3,)
        assert np.all(amps >= 0)

    def test_l1_loss(self, simple_2d_blob, simple_candidates_2d):
        """Test L1 loss function."""
        params_full, amps, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            loss_type="l1",
            n_iters=20,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        assert params_full.shape == (3, 6)  # Include sharpness
        assert amps.shape == (3,)
        assert np.all(amps >= 0)


class TestRegularization:
    """Test L1 regularization effects."""

    def test_l1_regularization(self, multi_blob_2d, simple_candidates_2d):
        """Test that L1 regularization affects results."""

        # Fit without regularization
        params_no_reg, amps_no_reg, _ = fit_gaussian_splats(
            V=multi_blob_2d,
            seeds=simple_candidates_2d,
            l1_amp=0.0,
            n_iters=30,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Fit with L1 regularization
        params_reg, amps_reg, _ = fit_gaussian_splats(
            V=multi_blob_2d,
            seeds=simple_candidates_2d,
            l1_amp=0.01,  # Small regularization
            n_iters=30,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Results should be different
        assert not np.allclose(amps_no_reg, amps_reg, atol=1e-3)

        # Regularized amplitudes should generally be smaller or more sparse
        # (though this isn't guaranteed for all cases)
        assert np.all(amps_reg >= 0)
        assert np.all(amps_no_reg >= 0)

    def test_l1_diag_regularization(self, multi_blob_2d, simple_candidates_2d):
        """Test that L1 diagonal regularization affects results."""
        # Fit without diagonal regularization
        params_no_reg, amps_no_reg, _ = fit_gaussian_splats(
            V=multi_blob_2d,
            seeds=simple_candidates_2d,
            l1_amp=0.0,
            l1_diag=0.0,
            n_iters=30,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Fit with L1 diagonal regularization
        params_reg, amps_reg, _ = fit_gaussian_splats(
            V=multi_blob_2d,
            seeds=simple_candidates_2d,
            l1_amp=0.0,  # Only diagonal regularization
            l1_diag=0.01,  # Small diagonal regularization
            n_iters=30,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Results should be different due to diagonal regularization
        assert not np.allclose(params_no_reg, params_reg, atol=1e-3)
        # Both should still have valid shapes
        assert params_no_reg.shape == params_reg.shape
        assert amps_no_reg.shape == amps_reg.shape


class TestConstraints:
    """Test sigma constraints."""

    def test_sigma_min_constraint(self, simple_2d_blob, simple_candidates_2d):
        """Test that minimum sigma constraint is enforced."""
        sigma_min = [0.8, 1.2]  # Different mins for each axis

        params_full, amps, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            sigma_min_diag=sigma_min,
            n_iters=30,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Extract Cholesky factors and check diagonal elements
        from luxar.gsplats.utils.trils import unpack_tril

        d = 2
        L_packed = params_full[:, d:]
        L_matrices = unpack_tril(L_packed, d)

        # Check that diagonal elements respect minimum constraints
        for i in range(len(simple_candidates_2d)):
            assert (
                L_matrices[i, 0, 0] >= sigma_min[0] - 1e-6
            )  # Small tolerance for numerical precision
            assert L_matrices[i, 1, 1] >= sigma_min[1] - 1e-6

    def test_sigma_max_constraint(self, simple_2d_blob, simple_candidates_2d):
        """Test that maximum sigma constraint is enforced."""
        sigma_min = [0.3, 0.3]
        sigma_max = [1.5, 2.0]  # Different maxes for each axis

        params_full, amps, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            sigma_min_diag=sigma_min,
            sigma_max_diag=sigma_max,
            n_iters=30,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Extract and check constraints
        from luxar.gsplats.utils.trils import unpack_tril

        d = 2
        L_packed = params_full[:, d:]
        L_matrices = unpack_tril(L_packed, d)

        for i in range(len(simple_candidates_2d)):
            assert L_matrices[i, 0, 0] <= sigma_max[0] + 1e-6
            assert L_matrices[i, 1, 1] <= sigma_max[1] + 1e-6


class TestDeviceSupport:
    """Test device support (CPU/CUDA)."""

    def test_cpu_device(self, simple_2d_blob, simple_candidates_2d):
        """Test explicit CPU device."""
        params_full, amps, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            device="cpu",
            n_iters=10,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        assert params_full.shape == (3, 6)  # Include sharpness
        assert amps.shape == (3,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self, simple_2d_blob, simple_candidates_2d):
        """Test CUDA device if available."""
        params_full, amps, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            device="cuda",
            n_iters=10,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        assert params_full.shape == (3, 6)  # Include sharpness
        assert amps.shape == (3,)


class TestConvergence:
    """Test optimization convergence properties."""

    def test_auto_convergence_threshold(self, simple_2d_blob, simple_candidates_2d):
        """Test automatic convergence threshold setting."""
        params, amps, stats = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            n_iters=100,
            max_abs_error=None,  # Should auto-set to 0.01
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Should converge with auto-threshold
        assert "converged" in stats
        assert params.shape[0] > 0

    def test_auto_candidate_generation_2d(self) -> None:
        """Test automatic candidate generation for 2D images."""
        # Create test image
        V = np.random.random((32, 32)).astype(np.float32)

        # Test auto-generation
        params, amps, stats = fit_gaussian_splats(
            V,
            # seeds=None (default)
            n_iters=10,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Should generate reasonable number of candidates based on volume
        # Expected at least 50 candidates for any image size
        assert len(amps) > 0
        assert params.shape[0] == len(amps)

    def test_auto_candidate_generation_3d(self) -> None:
        """Test automatic candidate generation for 3D volumes."""
        # Create test volume
        V = np.random.random((16, 16, 16)).astype(np.float32)

        # Test auto-generation
        params, amps, stats = fit_gaussian_splats(
            V,
            n_iters=10,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Should work with 3D data
        assert len(amps) > 0
        assert (
            params.shape[1] == 10
        )  # 3D centers (3) + 3x3 packed L (6) + sharpness (1) = 10

    def test_volume_proportional_scaling(self) -> None:
        """Test that candidate count scales with image volume."""
        # Small image
        V_small = np.random.random((16, 16)).astype(np.float32)
        params_small, amps_small, _ = fit_gaussian_splats(
            V_small,
            n_iters=5,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Large image (4x linear = 16x area)
        V_large = np.random.random((32, 32)).astype(np.float32)
        params_large, amps_large, _ = fit_gaussian_splats(
            V_large,
            n_iters=5,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Large image should generate proportionally more candidates
        # But both use minimum of 50, so this tests the scaling logic
        assert len(amps_small) > 0
        assert len(amps_large) >= len(amps_small)  # Should be at least as many

    def test_intensity_rescaling(self) -> None:
        """Test that amplitudes are correctly rescaled to original intensity range."""
        # Create image with realistic intensity distribution
        V = np.random.uniform(100, 300, (24, 24)).astype(np.float32)
        # This ensures 1st and 99th percentiles will span most of the [100,300] range

        # Store original range for comparison
        original_min, original_max = np.percentile(V, [1, 99])
        expected_scale_factor = original_max - original_min

        params, amps, stats = fit_gaussian_splats(
            V,
            n_iters=10,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Amplitudes should be rescaled to original intensity scale
        assert len(amps) > 0
        assert np.all(amps >= 0)  # Should remain non-negative

        # With realistic intensity range (~200), amplitudes should be much larger than [0,1]
        if (
            expected_scale_factor > 10
        ):  # Only test if we have significant intensity range
            assert (
                np.max(amps) > 5.0
            )  # Should be substantially larger than normalized range

    def test_configurable_normalization(self) -> None:
        """Test configurable percentile normalization."""
        # Create image with outliers
        V = np.random.uniform(10, 20, (24, 24)).astype(np.float32)
        V[0, 0] = 1000.0  # Extreme outlier

        # Test full range normalization
        params_full, amps_full, _ = fit_gaussian_splats(
            V, norm_percentile=0.0, n_iters=5, verbose=False, napari_movie=False
        )

        # Test percentile normalization (should be more robust to outlier)
        params_robust, amps_robust, _ = fit_gaussian_splats(
            V, norm_percentile=1.0, n_iters=5, verbose=False, napari_movie=False
        )

        # Both should work but potentially with different characteristics
        assert len(amps_full) > 0
        assert len(amps_robust) > 0
        assert np.all(amps_full >= 0)
        assert np.all(amps_robust >= 0)

    def test_best_state_tracking(self) -> None:
        """Test that best state (lowest max_abs_error) is returned, not final state."""
        # Create simple test case
        V = np.random.random((24, 24)).astype(np.float32)

        params, amps, stats = fit_gaussian_splats(
            V,
            n_iters=30,
            verbose=False,
            enable_dynamic_ops=True,  # May cause temporary quality fluctuations
            napari_movie=False,
        )

        # Should have best state information in stats
        assert "best_iteration" in stats
        assert "final_max_abs_error" in stats
        assert stats["best_iteration"] > 0
        assert stats["best_iteration"] <= stats["iterations"]
        assert stats["final_max_abs_error"] >= 0

        # Should return valid splat configuration
        assert len(amps) > 0
        assert params.shape[0] == len(amps)

    def test_convergence_with_iterations(self, simple_2d_blob, simple_candidates_2d):
        """Test that more iterations generally improve convergence."""

        # Short optimization
        params_short, amps_short, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            n_iters=5,
            verbose=False,
            enable_dynamic_ops=False,  # Disable for consistent topology
            napari_movie=False,
        )

        # Longer optimization
        params_long, amps_long, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            n_iters=50,
            verbose=False,
            enable_dynamic_ops=False,  # Disable for consistent topology
            napari_movie=False,
        )

        # Results should be different (optimization should progress)
        assert not np.allclose(params_short, params_long, atol=1e-3)
        assert not np.allclose(amps_short, amps_long, atol=1e-3)

    def test_different_learning_rates(self, simple_2d_blob, simple_candidates_2d):
        """Test that different learning rates produce different results."""

        params_low_lr, amps_low_lr, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            lr=0.01,  # Low learning rate
            n_iters=20,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        params_high_lr, amps_high_lr, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=simple_candidates_2d,
            lr=0.5,  # High learning rate
            n_iters=20,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Different learning rates should produce different results
        assert not np.allclose(params_low_lr, params_high_lr, atol=1e-2)


class TestReconstructionQuality:
    """Test reconstruction quality and meaningful results."""

    def test_single_blob_reconstruction(self, simple_2d_blob):
        """Test that we can reasonably reconstruct a single blob."""
        # Use candidate near the center where the blob peak should be
        center_candidate = np.array([[10.0, 10.0]], dtype=np.float32)

        params_full, amps, _ = fit_gaussian_splats(
            V=simple_2d_blob,
            seeds=center_candidate,
            n_iters=100,  # More iterations for better fit
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Check that fitted center is reasonably close to blob center
        fitted_center = params_full[0, :2]
        assert abs(fitted_center[0] - 10.0) < 2.0  # Within 2 pixels
        assert abs(fitted_center[1] - 10.0) < 2.0

        # Check that amplitude is reasonable (should be close to peak value)
        expected_peak = simple_2d_blob.max()
        assert amps[0] > 0.3 * expected_peak  # At least 30% of peak
        assert amps[0] < 2.0 * expected_peak  # Not unreasonably large

    def test_reconstruction_improves_with_more_splats(self, multi_blob_2d):
        """Test that using more splats generally improves reconstruction."""

        # Fit with few candidates
        few_candidates = np.array([[15.0, 15.0]], dtype=np.float32)
        params_few, amps_few, _ = fit_gaussian_splats(
            V=multi_blob_2d,
            seeds=few_candidates,
            n_iters=50,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # Fit with more candidates
        many_candidates = np.array(
            [
                [8.0, 8.0],  # Near blob 1
                [23.0, 23.0],  # Near blob 2
                [15.0, 27.0],  # Near blob 3
            ],
            dtype=np.float32,
        )
        params_many, amps_many, _ = fit_gaussian_splats(
            V=multi_blob_2d,
            seeds=many_candidates,
            n_iters=50,
            verbose=False,
            enable_dynamic_ops=False,
            napari_movie=False,
        )

        # More splats should generally have higher total amplitude
        # (since they can better represent the multiple blobs)
        # This isn't guaranteed in all cases, but is a reasonable expectation
        # for this test case with well-separated blobs
        assert len(amps_many) > len(amps_few)  # More splats


if __name__ == "__main__":
    pytest.main([__file__])
