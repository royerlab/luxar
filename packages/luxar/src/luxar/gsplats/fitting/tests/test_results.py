"""
Tests for fitting/results.py module.
"""

import time

import numpy as np
import pytest
import torch

from luxar.gsplats.fitting.config import (
    FitConfig,
    OptimizationResults,
    PreprocessedData,
)
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fitting.results import _clip_to_bounds, finalize_results

from .conftest import prepare_fit_config


@pytest.fixture
def basic_optimization_results():
    """Create basic optimization results for testing."""
    N = 5
    d = 2

    centers = torch.rand(N, d, dtype=torch.float32)
    Ls = torch.rand(N, d, d, dtype=torch.float32)
    # Amplitudes well above default max_abs_error (0.01) to avoid culling in basic tests
    amps = torch.rand(N, dtype=torch.float32) * 0.5 + 0.05

    start_time = time.time()
    end_time = start_time + 10.0

    return OptimizationResults(
        centers=centers,
        Ls=Ls,
        amps=amps,
        converged_early=True,
        early_stopped=False,
        actual_iters=50,
        best_iteration=45,
        best_loss=0.001,
        best_max_abs_error=0.005,
        best_rel_l2=0.1,
        movie_frames=None,
        start_time=start_time,
        end_time=end_time,
    )


@pytest.fixture
def basic_config():
    """Create basic FitConfig for testing."""
    V = np.random.rand(32, 32).astype(np.float32)
    return FitConfig(
        V=V,
        seeds=None,
        norm_percentile=0.0,
        init_sigma_vox=2.0,
        sigma_min_diag=[0.5, 0.5],
        sigma_max_diag=[10.0, 10.0],
        truncate=3.0,
        n_iters=100,
        lr=0.01,
        max_abs_error=0.01,
        rel_l2_target=None,
        gradient_clip=None,
        loss_type="mse",
        asymmetric_penalty=None,
        l1_amp=None,
        l1_diag=None,
        scheduler_type="plateau",
        patience=10,
        lr_reduction_factor=0.5,
        early_stop_patience=None,
        enable_dynamic_ops=False,
        dynamic_config=DynamicOpsConfig(),
        dynamic_ops_verbose=False,
        napari_movie=False,
        movie_every=1,
        movie_max_frames=100,
        device=torch.device("cpu"),
        verbose=False,
    )


@pytest.fixture
def basic_preprocessed_data():
    """Create basic preprocessed data."""
    return PreprocessedData(
        d=2,
        N=5,
        seed_centers=np.random.rand(5, 2).astype(np.float32),
        V_normalized=np.random.rand(32, 32).astype(np.float32),
        V_tensor=torch.rand(32, 32),
        image_min=0.0,
        image_max=2.5,
        intensity_range=2.5,  # Important for amplitude rescaling test
        max_abs_error=0.01,
    )


def test_finalize_results_basic(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Test basic result finalization."""
    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    # Check individual component shapes
    N = basic_optimization_results.centers.shape[0]
    d = basic_optimization_results.centers.shape[1]
    tril_size = d * (d + 1) // 2

    assert result.centers.shape == (N, d)
    assert result.cholesky_factors.shape == (N, tril_size)
    assert result.amplitudes.shape == (N,)
    assert isinstance(result.stats, dict)


def test_amplitude_rescaling(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Test that amplitudes are rescaled to original intensity range."""
    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    # Amplitudes should be rescaled by intensity_range
    original_amps = basic_optimization_results.amps.cpu().numpy()
    expected_amps = original_amps * basic_preprocessed_data.intensity_range

    assert np.allclose(result.amplitudes, expected_amps, rtol=1e-5)


def test_parameter_packing(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Test that centers and packed Cholesky are correctly stored separately."""
    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    N = basic_optimization_results.centers.shape[0]
    d = basic_optimization_results.centers.shape[1]

    # Centers should match exactly
    expected_centers = basic_optimization_results.centers.cpu().numpy()
    assert np.allclose(result.centers, expected_centers, rtol=1e-5)

    # Cholesky factors should be packed correctly
    tril_size = d * (d + 1) // 2
    assert result.cholesky_factors.shape == (N, tril_size)

    # Sharpness should match exactly


def test_stats_dictionary_structure(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Test that stats dictionary has all required fields."""
    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    # Required fields
    required_fields = [
        "time_seconds",
        "iterations",
        "best_iteration",
        "final_loss",
        "final_max_abs_error",
        "final_rel_l2",
        "converged",
        "early_stopped",
        "n_splats",
    ]

    for field in required_fields:
        assert field in result.stats

    # Check values make sense
    assert result.stats["time_seconds"] > 0
    assert result.stats["iterations"] == basic_optimization_results.actual_iters
    assert result.stats["best_iteration"] == basic_optimization_results.best_iteration
    assert result.stats["final_loss"] == basic_optimization_results.best_loss
    assert result.stats["n_splats"] == len(result.amplitudes)


def test_convergence_flag(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Test convergence flag in result.stats."""
    # Test converged case
    basic_optimization_results.converged_early = True

    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    assert result.stats["converged"] is True

    # Test non-converged case
    basic_optimization_results.converged_early = False

    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    assert result.stats["converged"] is False


def test_movie_frames_included(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Test that movie frames are included in result.stats when enabled."""
    # Add movie frames to optimization results
    movie_frames = {
        "target": [np.random.rand(32, 32) for _ in range(3)],
        "reconstruction": [np.random.rand(32, 32) for _ in range(3)],
        "residual": [np.random.rand(32, 32) for _ in range(3)],
        "splat_centers": [np.random.rand(5, 2) for _ in range(3)],
        "iterations": [10, 20, 30],
    }
    basic_optimization_results.movie_frames = movie_frames
    basic_config.napari_movie = True

    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    assert result.stats["movie_frames"] is not None
    assert result.stats["movie_frames"] == movie_frames
    assert result.stats["movie_shape"] == basic_config.V.shape


def test_movie_frames_excluded(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Test that movie frames are None when disabled."""
    basic_optimization_results.movie_frames = None
    basic_config.napari_movie = False

    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    assert result.stats["movie_frames"] is None


def test_data_types(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Test that output arrays are float32."""
    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    assert result.centers.dtype == np.float32
    assert result.cholesky_factors.dtype == np.float32
    assert result.amplitudes.dtype == np.float32


def test_3d_data(basic_config, basic_preprocessed_data) -> None:
    """Test finalization works for 3D data."""
    N = 4
    d = 3

    centers = torch.rand(N, d, dtype=torch.float32)
    Ls = torch.rand(N, d, d, dtype=torch.float32)
    # Amplitudes well above max_abs_error to avoid culling in this test
    amps = torch.rand(N, dtype=torch.float32) + 0.05

    optimization_results = OptimizationResults(
        centers=centers,
        Ls=Ls,
        amps=amps,
        converged_early=True,
        early_stopped=False,
        actual_iters=30,
        best_iteration=25,
        best_loss=0.002,
        best_max_abs_error=0.008,
        best_rel_l2=0.15,
        movie_frames=None,
        start_time=time.time(),
        end_time=time.time() + 5,
    )

    # Update preprocessed data for 3D
    basic_preprocessed_data.d = 3
    basic_preprocessed_data.N = N

    result = finalize_results(
        optimization_results, basic_config, basic_preprocessed_data
    )

    # Check component shapes for 3D
    tril_size = 3 * (3 + 1) // 2  # 6 for 3D

    assert result.centers.shape == (N, d)
    assert result.cholesky_factors.shape == (N, tril_size)
    assert result.amplitudes.shape == (N,)


def test_positive_amplitudes(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Test that output amplitudes are positive."""
    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    # All amplitudes should be positive
    assert np.all(result.amplitudes >= 0)


def test_time_seconds_calculation(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Test that time_seconds is calculated correctly."""
    expected_time = (
        basic_optimization_results.end_time - basic_optimization_results.start_time
    )

    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    assert abs(result.stats["time_seconds"] - expected_time) < 1e-6


# =============================================================================
# Voxel Footprint Correction Tests
# =============================================================================


class TestVoxelFootprintCorrection:
    """Tests for voxel footprint correction feature."""

    def test_correction_disabled_by_default(
        self, basic_optimization_results, basic_config, basic_preprocessed_data
    ) -> None:
        """Test that correction is disabled by default and Cholesky is unchanged."""
        # Default is False
        assert basic_config.voxel_footprint_correction is False

        # Get original Ls
        original_Ls = basic_optimization_results.Ls.cpu().numpy().copy()

        result = finalize_results(
            basic_optimization_results, basic_config, basic_preprocessed_data
        )

        # Unpack the result and verify it matches the original
        from luxar.gsplats.utils.trils import pack_tril

        expected_packed = pack_tril(original_Ls)
        assert np.allclose(result.cholesky_factors, expected_packed, rtol=1e-5)

    def test_correction_enabled_with_true(
        self, basic_optimization_results, basic_config, basic_preprocessed_data
    ) -> None:
        """Test correction with True uses 1-voxel box footprint (sigma = sqrt(1/12))."""
        basic_config.voxel_footprint_correction = True
        d = 2  # dimension from fixture
        sigma = np.sqrt(1.0 / 12.0)  # ~0.289 voxels
        variance = sigma * sigma  # = 1/12

        # Get original Ls and compute expected correction
        original_Ls = basic_optimization_results.Ls.cpu().numpy()
        Sigma_original = original_Ls @ original_Ls.transpose(0, 2, 1)
        Sigma_corrected = Sigma_original + variance * np.eye(d, dtype=original_Ls.dtype)
        L_expected = np.linalg.cholesky(Sigma_corrected)

        result = finalize_results(
            basic_optimization_results, basic_config, basic_preprocessed_data
        )

        # Unpack and verify
        from luxar.gsplats.utils.trils import pack_tril

        expected_packed = pack_tril(L_expected)
        assert np.allclose(result.cholesky_factors, expected_packed, rtol=1e-5)

    def test_correction_enabled_with_custom_sigma(
        self, basic_optimization_results, basic_config, basic_preprocessed_data
    ) -> None:
        """Test correction with custom sigma in voxel units."""
        custom_sigma = 0.5  # Half-voxel blur
        basic_config.voxel_footprint_correction = custom_sigma
        d = 2  # dimension from fixture
        variance = custom_sigma * custom_sigma  # = 0.25

        # Get original Ls and compute expected correction
        original_Ls = basic_optimization_results.Ls.cpu().numpy()
        Sigma_original = original_Ls @ original_Ls.transpose(0, 2, 1)
        Sigma_corrected = Sigma_original + variance * np.eye(d, dtype=original_Ls.dtype)
        L_expected = np.linalg.cholesky(Sigma_corrected)

        result = finalize_results(
            basic_optimization_results, basic_config, basic_preprocessed_data
        )

        # Unpack and verify
        from luxar.gsplats.utils.trils import pack_tril

        expected_packed = pack_tril(L_expected)
        assert np.allclose(result.cholesky_factors, expected_packed, rtol=1e-5)

    def test_correction_enabled_with_integer_sigma(
        self, basic_optimization_results, basic_config, basic_preprocessed_data
    ) -> None:
        """Test correction with integer sigma value (should work like float)."""
        int_sigma = 1  # Integer, not float - means 1 voxel sigma
        basic_config.voxel_footprint_correction = int_sigma
        d = 2  # dimension from fixture
        variance = float(int_sigma) * float(int_sigma)  # = 1.0

        # Get original Ls and compute expected correction
        original_Ls = basic_optimization_results.Ls.cpu().numpy()
        Sigma_original = original_Ls @ original_Ls.transpose(0, 2, 1)
        # Integer sigma should be used as-is (converted to float, then squared)
        Sigma_corrected = Sigma_original + variance * np.eye(d, dtype=original_Ls.dtype)
        L_expected = np.linalg.cholesky(Sigma_corrected)

        result = finalize_results(
            basic_optimization_results, basic_config, basic_preprocessed_data
        )

        # Unpack and verify
        from luxar.gsplats.utils.trils import pack_tril

        expected_packed = pack_tril(L_expected)
        assert np.allclose(result.cholesky_factors, expected_packed, rtol=1e-5)

    def test_correction_inflates_covariances(
        self, basic_optimization_results, basic_config, basic_preprocessed_data
    ) -> None:
        """Test that correction actually inflates covariances (increases diagonal)."""
        basic_config.voxel_footprint_correction = True
        # True means sigma = sqrt(1/12), so variance = 1/12

        # Get original Ls and compute original covariance diagonals
        original_Ls = basic_optimization_results.Ls.cpu().numpy()
        Sigma_original = original_Ls @ original_Ls.transpose(0, 2, 1)
        original_diag = Sigma_original[:, range(2), range(2)]  # (N, d) diagonal

        result = finalize_results(
            basic_optimization_results, basic_config, basic_preprocessed_data
        )

        # Unpack result and compute new covariance diagonals
        from luxar.gsplats.utils.trils import unpack_tril

        L_new = unpack_tril(result.cholesky_factors, 2)
        Sigma_new = L_new @ L_new.transpose(0, 2, 1)
        new_diag = Sigma_new[:, range(2), range(2)]

        # All diagonal elements should be inflated by at least 1/12 (sigma^2 for default)
        assert np.all(new_diag >= original_diag + 1.0 / 12.0 - 1e-6)

    def test_correction_3d_data(self, basic_config, basic_preprocessed_data) -> None:
        """Test correction works correctly for 3D data."""
        N = 4
        d = 3
        sigma = np.sqrt(1.0 / 12.0)  # Default for True
        variance = sigma * sigma  # = 1/12

        # Create 3D optimization results
        centers = torch.rand(N, d, dtype=torch.float32)
        Ls = torch.rand(N, d, d, dtype=torch.float32)
        # Make Ls lower triangular and positive definite
        Ls = torch.tril(Ls)
        Ls[:, range(d), range(d)] = torch.abs(Ls[:, range(d), range(d)]) + 0.5
        amps = torch.rand(N, dtype=torch.float32) + 0.05

        optimization_results = OptimizationResults(
            centers=centers,
            Ls=Ls,
            amps=amps,
            converged_early=True,
            early_stopped=False,
            actual_iters=30,
            best_iteration=25,
            best_loss=0.002,
            best_max_abs_error=0.008,
            best_rel_l2=0.15,
            movie_frames=None,
            start_time=time.time(),
            end_time=time.time() + 5,
        )

        # Update config and preprocessed data for 3D
        basic_config.voxel_footprint_correction = True
        basic_preprocessed_data.d = d
        basic_preprocessed_data.N = N

        # Compute expected
        Ls_np = Ls.numpy()
        Sigma_original = Ls_np @ Ls_np.transpose(0, 2, 1)
        Sigma_corrected = Sigma_original + variance * np.eye(d, dtype=Ls_np.dtype)
        L_expected = np.linalg.cholesky(Sigma_corrected)

        result = finalize_results(
            optimization_results, basic_config, basic_preprocessed_data
        )

        # Verify shape and values
        from luxar.gsplats.utils.trils import pack_tril

        tril_size = d * (d + 1) // 2
        assert result.cholesky_factors.shape == (N, tril_size)

        expected_packed = pack_tril(L_expected)
        assert np.allclose(result.cholesky_factors, expected_packed, rtol=1e-5)

    def test_correction_5d_data(self, basic_config, basic_preprocessed_data) -> None:
        """Test correction works correctly for 5D data (n-dimensional support)."""
        N = 3
        d = 5
        sigma = np.sqrt(1.0 / 12.0)  # Default for True
        variance = sigma * sigma  # = 1/12

        # Create 5D optimization results
        centers = torch.rand(N, d, dtype=torch.float32)
        Ls = torch.rand(N, d, d, dtype=torch.float32)
        # Make Ls lower triangular and positive definite
        Ls = torch.tril(Ls)
        Ls[:, range(d), range(d)] = torch.abs(Ls[:, range(d), range(d)]) + 0.5
        amps = torch.rand(N, dtype=torch.float32) + 0.05

        optimization_results = OptimizationResults(
            centers=centers,
            Ls=Ls,
            amps=amps,
            converged_early=True,
            early_stopped=False,
            actual_iters=30,
            best_iteration=25,
            best_loss=0.002,
            best_max_abs_error=0.008,
            best_rel_l2=0.15,
            movie_frames=None,
            start_time=time.time(),
            end_time=time.time() + 5,
        )

        # Update config and preprocessed data for 5D
        basic_config.voxel_footprint_correction = True
        basic_preprocessed_data.d = d
        basic_preprocessed_data.N = N

        # Compute expected
        Ls_np = Ls.numpy()
        Sigma_original = Ls_np @ Ls_np.transpose(0, 2, 1)
        Sigma_corrected = Sigma_original + variance * np.eye(d, dtype=Ls_np.dtype)
        L_expected = np.linalg.cholesky(Sigma_corrected)

        result = finalize_results(
            optimization_results, basic_config, basic_preprocessed_data
        )

        # Verify shape and values
        from luxar.gsplats.utils.trils import pack_tril

        tril_size = d * (d + 1) // 2
        assert result.cholesky_factors.shape == (N, tril_size)

        expected_packed = pack_tril(L_expected)
        assert np.allclose(result.cholesky_factors, expected_packed, rtol=1e-5)

    def test_correction_preserves_positive_definiteness(
        self, basic_optimization_results, basic_config, basic_preprocessed_data
    ) -> None:
        """Test that correction maintains positive definite covariances."""
        basic_config.voxel_footprint_correction = True

        result = finalize_results(
            basic_optimization_results, basic_config, basic_preprocessed_data
        )

        # Unpack and verify eigenvalues are positive
        from luxar.gsplats.utils.trils import unpack_tril

        L_new = unpack_tril(result.cholesky_factors, 2)
        Sigma_new = L_new @ L_new.transpose(0, 2, 1)

        for i in range(len(L_new)):
            eigenvalues = np.linalg.eigvalsh(Sigma_new[i])
            assert np.all(eigenvalues > 0), f"Non-positive eigenvalue in covariance {i}"


class TestVoxelFootprintCorrectionValidation:
    """Tests for voxel footprint correction validation."""

    def test_negative_sigma_raises_error(self) -> None:
        """Test that negative sigma raises ValueError."""
        from luxar.gsplats.fit_gsplats import GaussianSplatFitter

        fitter = GaussianSplatFitter(device="cpu")
        V = np.random.rand(32, 32).astype(np.float32)

        with pytest.raises(
            ValueError, match="voxel_footprint_correction sigma must be positive"
        ):
            prepare_fit_config(
                fitter,
                V,
                voxel_footprint_correction=-0.1,
            )

    def test_zero_sigma_raises_error(self) -> None:
        """Test that zero sigma raises ValueError."""
        from luxar.gsplats.fit_gsplats import GaussianSplatFitter

        fitter = GaussianSplatFitter(device="cpu")
        V = np.random.rand(32, 32).astype(np.float32)

        with pytest.raises(
            ValueError, match="voxel_footprint_correction sigma must be positive"
        ):
            prepare_fit_config(
                fitter,
                V,
                voxel_footprint_correction=0.0,
            )

    def test_false_disabled_no_error(self) -> None:
        """Test that False (disabled) is accepted without error."""
        from luxar.gsplats.fit_gsplats import GaussianSplatFitter

        fitter = GaussianSplatFitter(device="cpu")
        V = np.random.rand(32, 32).astype(np.float32)

        # Should not raise
        config = prepare_fit_config(
            fitter,
            V,
            voxel_footprint_correction=False,
        )
        assert config.voxel_footprint_correction is False

    def test_true_enabled_no_error(self) -> None:
        """Test that True is accepted without error."""
        from luxar.gsplats.fit_gsplats import GaussianSplatFitter

        fitter = GaussianSplatFitter(device="cpu")
        V = np.random.rand(32, 32).astype(np.float32)

        # Should not raise
        config = prepare_fit_config(
            fitter,
            V,
            voxel_footprint_correction=True,
        )
        assert config.voxel_footprint_correction is True

    def test_positive_float_no_error(self) -> None:
        """Test that positive float is accepted without error."""
        from luxar.gsplats.fit_gsplats import GaussianSplatFitter

        fitter = GaussianSplatFitter(device="cpu")
        V = np.random.rand(32, 32).astype(np.float32)

        # Should not raise
        config = prepare_fit_config(
            fitter,
            V,
            voxel_footprint_correction=0.25,
        )
        assert config.voxel_footprint_correction == 0.25

    def test_positive_int_no_error(self) -> None:
        """Test that positive int is accepted without error."""
        from luxar.gsplats.fit_gsplats import GaussianSplatFitter

        fitter = GaussianSplatFitter(device="cpu")
        V = np.random.rand(32, 32).astype(np.float32)

        # Should not raise - integer should be accepted
        config = prepare_fit_config(
            fitter,
            V,
            voxel_footprint_correction=1,  # integer, not float
        )
        assert config.voxel_footprint_correction == 1

    def test_negative_int_raises_error(self) -> None:
        """Test that negative int raises ValueError."""
        from luxar.gsplats.fit_gsplats import GaussianSplatFitter

        fitter = GaussianSplatFitter(device="cpu")
        V = np.random.rand(32, 32).astype(np.float32)

        with pytest.raises(
            ValueError, match="voxel_footprint_correction sigma must be positive"
        ):
            prepare_fit_config(
                fitter,
                V,
                voxel_footprint_correction=-1,  # negative integer
            )


# =============================================================================
# (Post-fit noise-floor culling removed — now handled by GSplatData.cull())
# =============================================================================

# =============================================================================
# Clip-to-Bounds Tests
# =============================================================================


class TestClipToBounds:
    """Tests for boundary clipping in post-processing."""

    def test_clip_shrinks_out_of_bounds_splats(self) -> None:
        """Splat near edge with large L is shrunk to fit within bounds."""
        N, d = 1, 2
        shape = (16, 16)
        truncate = 3.0

        # Splat at position (2, 8) with sigma=5 → radius = 3*5 = 15 > 2 (dist to edge)
        centers = np.array([[2.0, 8.0]], dtype=np.float32)
        Ls = np.zeros((N, d, d), dtype=np.float32)
        Ls[0, 0, 0] = 5.0  # Large sigma in dim 0
        Ls[0, 1, 1] = 1.0  # Normal sigma in dim 1

        Ls_clipped = _clip_to_bounds(centers, Ls, shape, truncate)

        # After clipping: truncate * sqrt(Sigma_00) <= 2.0 (dist to edge in dim 0)
        sigma_diag_clipped = np.sum(Ls_clipped * Ls_clipped, axis=2)  # (N, d)
        radii_clipped = truncate * np.sqrt(sigma_diag_clipped)

        dist_to_edge = np.minimum(
            centers, np.array(shape, dtype=np.float32) - 1.0 - centers
        )
        # Radius should be <= dist_to_edge (within tolerance)
        assert np.all(radii_clipped <= dist_to_edge + 1e-5)

        # Dim 0 should have been shrunk
        assert Ls_clipped[0, 0, 0] < Ls[0, 0, 0]
        # Dim 1 should be unchanged (well within bounds)
        np.testing.assert_allclose(Ls_clipped[0, 1, 1], Ls[0, 1, 1], atol=1e-6)

    def test_clip_preserves_in_bounds_splats(self) -> None:
        """Splat well inside the volume is not modified."""
        N, d = 1, 2
        shape = (32, 32)
        truncate = 3.0

        centers = np.array([[16.0, 16.0]], dtype=np.float32)
        Ls = np.zeros((N, d, d), dtype=np.float32)
        Ls[0, 0, 0] = 2.0
        Ls[0, 1, 1] = 2.0

        Ls_clipped = _clip_to_bounds(centers, Ls, shape, truncate)

        # truncate * sqrt(4) = 6 << 16 (dist to edge) → no change
        np.testing.assert_allclose(Ls_clipped, Ls, atol=1e-7)

    def test_clip_preserves_orientation(self) -> None:
        """Row element ratios within L are preserved after clipping."""
        N, d = 1, 2
        shape = (16, 16)
        truncate = 3.0

        centers = np.array([[2.0, 8.0]], dtype=np.float32)
        Ls = np.zeros((N, d, d), dtype=np.float32)
        Ls[0, 0, 0] = 5.0
        Ls[0, 1, 0] = 2.0  # off-diagonal
        Ls[0, 1, 1] = 4.0

        Ls_clipped = _clip_to_bounds(centers, Ls, shape, truncate)

        # Check ratios within row 1 are preserved
        if abs(Ls[0, 1, 0]) > 1e-8:
            original_ratio = Ls[0, 1, 1] / Ls[0, 1, 0]
            clipped_ratio = Ls_clipped[0, 1, 1] / Ls_clipped[0, 1, 0]
            np.testing.assert_allclose(clipped_ratio, original_ratio, rtol=1e-5)

    def test_clip_disabled_by_default(
        self, basic_optimization_results, basic_config, basic_preprocessed_data
    ) -> None:
        """clip_to_bounds=False leaves L unchanged."""
        assert basic_config.clip_to_bounds is False

        original_Ls = basic_optimization_results.Ls.cpu().numpy().copy()

        result = finalize_results(
            basic_optimization_results, basic_config, basic_preprocessed_data
        )

        from luxar.gsplats.utils.trils import pack_tril

        expected_packed = pack_tril(original_Ls)
        np.testing.assert_allclose(result.cholesky_factors, expected_packed, rtol=1e-5)

    def test_clip_3d_data(self) -> None:
        """Clip works for 3D data."""
        N, d = 2, 3
        shape = (10, 20, 30)
        truncate = 3.0

        # One splat near edge, one inside
        centers = np.array([[1.0, 10.0, 15.0], [5.0, 10.0, 15.0]], dtype=np.float32)
        Ls = np.zeros((N, d, d), dtype=np.float32)
        for i in range(d):
            Ls[:, i, i] = 3.0  # sigma=3, radius=9

        Ls_clipped = _clip_to_bounds(centers, Ls, shape, truncate)

        # Splat 0 in dim 0: dist_to_edge = 1, radius=9 → must shrink
        assert Ls_clipped[0, 0, 0] < Ls[0, 0, 0]
        # Splat 1 in dim 0: dist_to_edge = min(5, 4) = 4, radius=9 → must shrink
        assert Ls_clipped[1, 0, 0] < Ls[1, 0, 0]
        # Splat 0 in dim 1: dist_to_edge = min(10, 9) = 9, radius=9 → borderline, ≈ unchanged
        np.testing.assert_allclose(Ls_clipped[0, 1, 1], Ls[0, 1, 1], atol=0.1)

    def test_clip_center_at_boundary(self) -> None:
        """Splat center at position 0 should have L shrunk to near-zero."""
        N, d = 1, 2
        shape = (16, 16)
        truncate = 3.0

        centers = np.array([[0.0, 8.0]], dtype=np.float32)
        Ls = np.zeros((N, d, d), dtype=np.float32)
        Ls[0, 0, 0] = 2.0
        Ls[0, 1, 1] = 2.0

        Ls_clipped = _clip_to_bounds(centers, Ls, shape, truncate)

        # dist_to_edge in dim 0 = min(0, 15) = 0 → max_sigma_sq = 0 → scale = 0
        assert abs(Ls_clipped[0, 0, 0]) < 1e-6
        # dim 1 should be fine (dist=7)
        assert Ls_clipped[0, 1, 1] > 0

    def test_clip_before_voxel_footprint_correction(self) -> None:
        """When both enabled, clip runs first, then voxel footprint correction inflates."""
        N, d = 1, 2
        shape = (16, 16)

        # Splat near edge
        centers = torch.tensor([[2.0, 8.0]], dtype=torch.float32)
        Ls = torch.zeros(N, d, d, dtype=torch.float32)
        Ls[0, 0, 0] = 5.0
        Ls[0, 1, 1] = 1.0
        amps = torch.tensor([0.5], dtype=torch.float32)

        opt = OptimizationResults(
            centers=centers,
            Ls=Ls,
            amps=amps,
            converged_early=True,
            early_stopped=False,
            actual_iters=50,
            best_iteration=45,
            best_loss=0.001,
            best_max_abs_error=0.005,
            best_rel_l2=0.1,
            movie_frames=None,
            start_time=0.0,
            end_time=1.0,
        )

        V = np.random.rand(*shape).astype(np.float32)
        config = FitConfig(
            V=V,
            seeds=None,
            norm_percentile=0.0,
            init_sigma_vox=2.0,
            sigma_min_diag=[0.5, 0.5],
            sigma_max_diag=[10.0, 10.0],
            truncate=3.0,
            n_iters=100,
            lr=0.01,
            max_abs_error=0.01,
            rel_l2_target=None,
            gradient_clip=None,
            loss_type="mse",
            asymmetric_penalty=None,
            l1_amp=None,
            l1_diag=None,
            scheduler_type="plateau",
            patience=10,
            lr_reduction_factor=0.5,
            early_stop_patience=None,
            enable_dynamic_ops=False,
            dynamic_config=DynamicOpsConfig(),
            dynamic_ops_verbose=False,
            napari_movie=False,
            movie_every=1,
            movie_max_frames=100,
            device=torch.device("cpu"),
            verbose=False,
            clip_to_bounds=True,
            voxel_footprint_correction=True,
        )

        ppd = PreprocessedData(
            d=2,
            N=1,
            seed_centers=np.array([[2.0, 8.0]], dtype=np.float32),
            V_normalized=V,
            V_tensor=torch.from_numpy(V),
            image_min=0.0,
            image_max=1.0,
            intensity_range=1.0,
            max_abs_error=0.01,
        )

        result = finalize_results(opt, config, ppd)

        # Should have 1 splat (no culling since 0.5 >> threshold)
        assert len(result.amplitudes) == 1
        # The result should be valid (finite values)
        assert np.all(np.isfinite(result.cholesky_factors))


# =============================================================================
# Voxel Size Output Conversion Tests
# =============================================================================


class TestVoxelSizeOutputConversion:
    """Tests for voxel_size + output_space coordinate conversion in finalize_results."""

    def _make_config_and_data(self, d, voxel_size=None, output_space="real"):
        """Helper to create config and data for conversion tests."""
        shape = tuple([32] * d)
        V = np.random.rand(*shape).astype(np.float32)
        N = 3

        config = FitConfig(
            V=V,
            seeds=None,
            norm_percentile=0.0,
            init_sigma_vox=2.0,
            sigma_min_diag=[0.5] * d,
            sigma_max_diag=[10.0] * d,
            truncate=3.0,
            n_iters=100,
            lr=0.01,
            max_abs_error=0.01,
            rel_l2_target=None,
            gradient_clip=None,
            loss_type="mse",
            asymmetric_penalty=None,
            l1_amp=None,
            l1_diag=None,
            scheduler_type="plateau",
            patience=10,
            lr_reduction_factor=0.5,
            early_stop_patience=None,
            enable_dynamic_ops=False,
            dynamic_config=DynamicOpsConfig(),
            dynamic_ops_verbose=False,
            napari_movie=False,
            movie_every=1,
            movie_max_frames=100,
            device=torch.device("cpu"),
            verbose=False,
            voxel_size=voxel_size,
            output_space=output_space,
        )

        # Deterministic centers and Ls for predictable output
        centers = torch.tensor([[5.0] * d, [15.0] * d, [25.0] * d], dtype=torch.float32)
        Ls = torch.zeros(N, d, d, dtype=torch.float32)
        for i in range(d):
            Ls[:, i, i] = 1.0 + 0.1 * i  # slightly different per axis
        amps = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float32)

        start_time = time.time()
        opt = OptimizationResults(
            centers=centers,
            Ls=Ls,
            amps=amps,
            converged_early=True,
            early_stopped=False,
            actual_iters=50,
            best_iteration=45,
            best_loss=0.001,
            best_max_abs_error=0.005,
            best_rel_l2=0.1,
            movie_frames=None,
            start_time=start_time,
            end_time=start_time + 1.0,
        )
        ppd = PreprocessedData(
            d=d,
            N=N,
            seed_centers=np.random.rand(N, d).astype(np.float32),
            V_normalized=V,
            V_tensor=torch.from_numpy(V),
            image_min=0.0,
            image_max=1.0,
            intensity_range=1.0,
            max_abs_error=0.01,
        )
        return config, opt, ppd

    def test_no_voxel_size_no_conversion(self):
        """voxel_size=None produces voxel-space output regardless of output_space."""
        config, opt, ppd = self._make_config_and_data(3)
        result = finalize_results(opt, config, ppd)
        # Centers should be in voxel range [0, 32)
        assert result.centers.max() < 32

    def test_voxel_size_real_scales_centers(self):
        """output_space='real' with voxel_size scales centers correctly."""
        vs = np.array([5.0, 1.0, 1.0], dtype=np.float32)

        # Get voxel-space result
        config_vox, opt, ppd = self._make_config_and_data(
            3, voxel_size=vs, output_space="voxel"
        )
        result_vox = finalize_results(opt, config_vox, ppd)

        # Get real-space result (re-create opt since finalize_results modifies tensors)
        config_real, opt2, ppd2 = self._make_config_and_data(
            3, voxel_size=vs, output_space="real"
        )
        result_real = finalize_results(opt2, config_real, ppd2)

        # Real centers = voxel centers * voxel_size
        np.testing.assert_allclose(
            result_real.centers, result_vox.centers * vs, rtol=1e-5
        )

    def test_voxel_size_real_scales_cholesky(self):
        """output_space='real' scales packed Cholesky factors correctly."""
        vs = np.array([5.0, 2.0, 1.0], dtype=np.float32)
        d = 3

        config_vox, opt, ppd = self._make_config_and_data(
            d, voxel_size=vs, output_space="voxel"
        )
        result_vox = finalize_results(opt, config_vox, ppd)

        config_real, opt2, ppd2 = self._make_config_and_data(
            d, voxel_size=vs, output_space="real"
        )
        result_real = finalize_results(opt2, config_real, ppd2)

        # Build expected scale factors for packed tril: row i → vs[i]
        tril_scales = np.concatenate([[vs[i]] * (i + 1) for i in range(d)])
        np.testing.assert_allclose(
            result_real.cholesky_factors,
            result_vox.cholesky_factors * tril_scales,
            rtol=1e-5,
        )

    def test_voxel_size_amplitudes_unchanged(self):
        """Amplitudes are NOT scaled by voxel_size (not spatial quantities)."""
        vs = np.array([5.0, 2.0, 1.0], dtype=np.float32)

        config_vox, opt, ppd = self._make_config_and_data(
            3, voxel_size=vs, output_space="voxel"
        )
        result_vox = finalize_results(opt, config_vox, ppd)

        config_real, opt2, ppd2 = self._make_config_and_data(
            3, voxel_size=vs, output_space="real"
        )
        result_real = finalize_results(opt2, config_real, ppd2)

        np.testing.assert_allclose(
            result_real.amplitudes, result_vox.amplitudes, rtol=1e-5
        )

    def test_voxel_size_voxel_output_no_scaling(self):
        """output_space='voxel' suppresses coordinate conversion."""
        vs = np.array([5.0, 1.0, 1.0], dtype=np.float32)

        config_vox, opt, ppd = self._make_config_and_data(
            3, voxel_size=vs, output_space="voxel"
        )
        result = finalize_results(opt, config_vox, ppd)

        # Centers should be in voxel range, not physical
        assert result.centers[:, 0].max() < 32  # not 32*5=160

    def test_2d_output_conversion(self):
        """Output conversion works for 2D data."""
        vs = np.array([3.0, 1.5], dtype=np.float32)

        config_vox, opt, ppd = self._make_config_and_data(
            2, voxel_size=vs, output_space="voxel"
        )
        result_vox = finalize_results(opt, config_vox, ppd)

        config_real, opt2, ppd2 = self._make_config_and_data(
            2, voxel_size=vs, output_space="real"
        )
        result_real = finalize_results(opt2, config_real, ppd2)

        np.testing.assert_allclose(
            result_real.centers, result_vox.centers * vs, rtol=1e-5
        )


# ═══════════════════════════════════════════════════════════════════════
# Post-fit quality metrics (PSNR, SSIM, MSE)
# ═══════════════════════════════════════════════════════════════════════


def test_postfit_metrics_present(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Post-fit quality metrics should be stored in stats dict."""
    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )
    for key in ("psnr_db", "ssim", "mse"):
        assert key in result.stats, f"Missing post-fit metric: {key}"
    assert result.stats["psnr_db"] > 0
    # SSIM ranges from -1 to 1 (can be negative for anti-correlated signals)
    assert -1.0 <= result.stats["ssim"] <= 1.0
    assert result.stats["mse"] >= 0.0


def test_postfit_metrics_present_for_physical_coords() -> None:
    """Physical output coordinates must not cost a fit its quality metrics.

    This used to assert the opposite — that the metrics were skipped — which
    quietly made a defect a contract: ``output_space="real"`` is the default, so
    every fit passing a ``voxel_size`` produced an archive with no PSNR. The
    metrics are now scored on the pre-conversion (voxel-grid) arrays. See
    ``test_results_quality_real_space.py`` for the equivalence check.
    """
    V = np.random.rand(16, 16).astype(np.float32)
    config = FitConfig(
        V=V,
        seeds=None,
        norm_percentile=0.0,
        init_sigma_vox=2.0,
        sigma_min_diag=[0.5, 0.5],
        sigma_max_diag=[10.0, 10.0],
        truncate=3.0,
        n_iters=100,
        lr=0.01,
        max_abs_error=0.01,
        rel_l2_target=None,
        gradient_clip=None,
        loss_type="mse",
        asymmetric_penalty=None,
        l1_amp=None,
        l1_diag=None,
        scheduler_type="plateau",
        patience=10,
        lr_reduction_factor=0.5,
        early_stop_patience=None,
        enable_dynamic_ops=False,
        dynamic_config=DynamicOpsConfig(),
        dynamic_ops_verbose=False,
        napari_movie=False,
        movie_every=1,
        movie_max_frames=100,
        device=torch.device("cpu"),
        verbose=False,
        output_space="real",
        voxel_size=np.array([0.5, 0.5]),
    )

    opt = OptimizationResults(
        centers=torch.rand(5, 2),
        Ls=torch.eye(2).unsqueeze(0).expand(5, -1, -1),
        amps=torch.rand(5),
        converged_early=False,
        early_stopped=False,
        actual_iters=10,
        best_iteration=5,
        best_loss=0.01,
        best_max_abs_error=0.05,
        best_rel_l2=0.1,
        movie_frames=None,
        start_time=0.0,
        end_time=1.0,
    )

    ppd = PreprocessedData(
        d=2,
        N=5,
        seed_centers=np.random.rand(5, 2).astype(np.float32),
        V_normalized=np.random.rand(16, 16).astype(np.float32),
        V_tensor=torch.rand(16, 16),
        image_min=0.0,
        image_max=1.0,
        intensity_range=1.0,
        max_abs_error=0.01,
    )

    voxel_centers = opt.centers.numpy().copy()
    result = finalize_results(opt, config, ppd)
    for key in ("psnr_db", "ssim", "mse", "foreground_psnr_db"):
        assert key in result.stats, f"missing post-fit metric: {key}"
    # And the caller still gets physical coordinates back: voxel_size is 0.5 per
    # axis, so the returned centers must be exactly HALF the voxel indices they
    # came from, not the indices the scoring copy used. Pinned against those
    # indices rather than a bound derived from the 16x16 shape: `torch.rand`
    # centers all sit below 1.0, so any such bound holds whether the physical
    # conversion ran or not.
    np.testing.assert_allclose(result.centers, 0.5 * voxel_centers, rtol=1e-6)


def test_finalize_reports_relocation_and_candidate_scale_diagnostics(
    basic_config, basic_preprocessed_data
) -> None:
    """Fit stats distinguish relocation history from colliding scale candidates."""
    basic_config.init_sigma_vox = 0.5
    basic_config.sigma_min_diag = [0.3, 0.3]
    basic_config.enable_dynamic_ops = True
    basic_config.n_iters = 20_000
    basic_config.dynamic_config.step_every = 50
    basic_config.dynamic_config.k_max_residuals = 40
    basic_config.dynamic_config.init_sigma_vox = 0.5
    basic_config.clip_to_bounds = False
    basic_config.voxel_footprint_correction = True

    Ls = torch.tensor(
        [
            [[0.5, 0.0], [0.0, 0.5]],
            [[0.5, 0.0], [0.006, 0.5]],
            [[0.3, 0.0], [0.0, 0.3]],
            [[0.7, 0.0], [0.0, 0.7]],
            [[0.5, 0.0], [0.0, 0.7]],
        ],
        dtype=torch.float32,
    )
    optimization_results = OptimizationResults(
        centers=torch.full((5, 2), 16.0),
        Ls=Ls,
        amps=torch.ones(5),
        converged_early=False,
        early_stopped=False,
        actual_iters=20_000,
        best_iteration=19_900,
        best_loss=0.01,
        best_max_abs_error=0.05,
        best_rel_l2=0.1,
        movie_frames=None,
        start_time=0.0,
        end_time=1.0,
        relocation_statistics={"total_relocations": 17, "unique_splats": 11},
    )

    result = finalize_results(
        optimization_results, basic_config, basic_preprocessed_data
    )

    assert result.stats["configured_iterations"] == 20_000
    assert result.stats["dynamic_ops_step_every"] == 50
    assert result.stats["dynamic_ops_k_max_residuals"] == 40
    assert result.stats["dynamic_ops_relocation_events"] == 17
    assert result.stats["dynamic_ops_unique_splats_relocated"] == 11
    assert result.stats["scale_diagnostic_tolerance_vox"] == 0.01
    assert result.stats["fit_init_sigma_diag_vox"] == [0.5, 0.5]
    assert result.stats["relocation_init_sigma_vox"] == 0.5
    assert result.stats["sigma_min_diag_vox"] == pytest.approx([0.3, 0.3])
    assert result.stats["splats_near_fit_init_sigma_count"] == 2
    assert result.stats["splats_near_fit_init_sigma_fraction"] == 0.4
    assert result.stats["splats_near_relocation_init_sigma_count"] == 2
    assert result.stats["splats_near_relocation_init_sigma_fraction"] == 0.4
    assert result.stats["splats_near_sigma_min_count"] == 1
    assert result.stats["splats_near_sigma_min_fraction"] == 0.2
