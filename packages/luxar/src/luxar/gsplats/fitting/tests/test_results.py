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
from luxar.gsplats.fitting.results import finalize_results


@pytest.fixture
def basic_optimization_results():
    """Create basic optimization results for testing."""
    N = 5
    d = 2

    centers = torch.rand(N, d, dtype=torch.float32)
    Ls = torch.rand(N, d, d, dtype=torch.float32)
    amps = torch.rand(N, dtype=torch.float32) * 0.5
    sharpness = torch.rand(N, dtype=torch.float32) * 2.0 + 1.0

    start_time = time.time()
    end_time = start_time + 10.0

    return OptimizationResults(
        centers=centers,
        Ls=Ls,
        amps=amps,
        sharpness=sharpness,
        converged_early=True,
        actual_iters=50,
        best_iteration=45,
        best_loss=0.001,
        best_max_abs_error=0.005,
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
        gradient_clip=None,
        loss_type="mse",
        asymmetric_penalty=None,
        l1_amp=None,
        l1_diag=None,
        l1_sharpness=None,
        scheduler_type="plateau",
        patience=10,
        lr_reduction_factor=0.5,
        early_stop_patience=None,        enable_dynamic_ops=False,
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
    assert result.sharpnesses.shape == (N,)
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
    expected_sharpness = basic_optimization_results.sharpness.cpu().numpy()
    assert np.allclose(result.sharpnesses, expected_sharpness, rtol=1e-5)


def test_sharpness_statistics(
    basic_optimization_results, basic_config, basic_preprocessed_data
) -> None:
    """Test that sharpness statistics are computed correctly."""
    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    # Check that all sharpness stats are present
    assert "sharpness_min" in result.stats
    assert "sharpness_max" in result.stats
    assert "sharpness_mean" in result.stats
    assert "sharpness_std" in result.stats
    assert "sharpness_median" in result.stats

    # Verify they are reasonable values
    sharpness_np = basic_optimization_results.sharpness.cpu().numpy()
    assert abs(result.stats["sharpness_min"] - float(np.min(sharpness_np))) < 1e-5
    assert abs(result.stats["sharpness_max"] - float(np.max(sharpness_np))) < 1e-5
    assert abs(result.stats["sharpness_mean"] - float(np.mean(sharpness_np))) < 1e-5


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
        "converged",
        "n_splats",
        "sharpness_min",
        "sharpness_max",
        "sharpness_mean",
        "sharpness_std",
        "sharpness_median",
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
    basic_optimization_results.actual_iters = 50
    basic_config.n_iters = 100

    result = finalize_results(
        basic_optimization_results, basic_config, basic_preprocessed_data
    )

    assert result.stats["converged"] is True

    # Test non-converged case
    basic_optimization_results.actual_iters = 100
    basic_config.n_iters = 100

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
    assert result.sharpnesses.dtype == np.float32
    assert result.amplitudes.dtype == np.float32


def test_3d_data(basic_config, basic_preprocessed_data) -> None:
    """Test finalization works for 3D data."""
    N = 4
    d = 3

    centers = torch.rand(N, d, dtype=torch.float32)
    Ls = torch.rand(N, d, d, dtype=torch.float32)
    amps = torch.rand(N, dtype=torch.float32)
    sharpness = torch.rand(N, dtype=torch.float32) * 2.0 + 1.0

    optimization_results = OptimizationResults(
        centers=centers,
        Ls=Ls,
        amps=amps,
        sharpness=sharpness,
        converged_early=True,
        actual_iters=30,
        best_iteration=25,
        best_loss=0.002,
        best_max_abs_error=0.008,
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
    assert result.sharpnesses.shape == (N,)
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
