"""
Tests for fitting/optimization.py module.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.fitting.config import (
    DynamicOpsConfig,
    FitConfig,
    PreprocessedData,
)
from luxar.gsplats.fitting.initialization import initialize_optimization
from luxar.gsplats.fitting.losses import create_loss_function
from luxar.gsplats.fitting.optimization import (
    _compute_max_abs_error,
    run_optimization_loop,
)


@pytest.fixture
def simple_2d_setup():
    """Create a simple 2D setup for optimization testing."""
    # Create a simple target with a few bright spots
    V = np.zeros((32, 32), dtype=np.float32)
    V[10:12, 10:12] = 1.0
    V[20:22, 20:22] = 0.8

    config = FitConfig(
        V=V,
        seeds=None,
        norm_percentile=0.0,
        init_sigma_vox=2.0,
        sigma_min_diag=[0.5, 0.5],
        sigma_max_diag=[10.0, 10.0],
        truncate=3.0,
        n_iters=50,
        lr=0.1,
        max_abs_error=0.05,
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

    # Create seed centers at bright spots
    seed_centers = np.array([[10.5, 10.5], [20.5, 20.5]], dtype=np.float32)

    V_normalized = V / (V.max() + 1e-12)
    V_tensor = torch.from_numpy(V_normalized).to(config.device)

    preprocessed_data = PreprocessedData(
        d=2,
        N=2,
        seed_centers=seed_centers,
        V_normalized=V_normalized,
        V_tensor=V_tensor,
        image_min=0.0,
        image_max=float(V.max()),
        intensity_range=float(V.max()),
        max_abs_error=0.05,
    )

    return config, preprocessed_data


def test_compute_max_abs_error() -> None:
    """Test max absolute error computation."""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.1, 1.8], [3.2, 3.9]])

    max_error = _compute_max_abs_error(pred, target)

    # Maximum error should be 0.2 (at position [1, 0])
    assert abs(max_error - 0.2) < 1e-6


def test_run_optimization_loop_basic(simple_2d_setup) -> None:
    """Test basic optimization loop execution."""
    config, preprocessed_data = simple_2d_setup

    # Initialize components
    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    # Run optimization
    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Check results structure
    assert results.centers is not None
    assert results.Ls is not None
    assert results.amps is not None
    assert results.actual_iters > 0
    assert results.best_iteration > 0
    assert results.best_loss >= 0
    assert results.best_max_abs_error >= 0
    assert results.end_time > results.start_time


def test_convergence_early_stopping(simple_2d_setup) -> None:
    """Test early stopping when convergence criterion is met."""
    config, preprocessed_data = simple_2d_setup

    # Set very loose convergence criterion to trigger early stopping
    preprocessed_data.max_abs_error = 0.5  # Very loose

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Should converge before max iterations
    assert results.converged_early
    assert results.actual_iters < config.n_iters


def test_iteration_limit(simple_2d_setup) -> None:
    """Test that optimization stops at n_iters if not converged."""
    config, preprocessed_data = simple_2d_setup

    # Set very tight convergence criterion to prevent early stopping
    preprocessed_data.max_abs_error = 1e-10  # Very tight

    # Also reduce iterations to make test faster
    config.n_iters = 10

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Should reach iteration limit
    assert not results.converged_early
    assert results.actual_iters == config.n_iters


def test_best_state_tracking(simple_2d_setup) -> None:
    """Test that best state is tracked and can differ from final state."""
    config, preprocessed_data = simple_2d_setup
    config.n_iters = 20

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Best iteration should be tracked
    assert results.best_iteration > 0
    assert results.best_iteration <= results.actual_iters

    # Best error should be reasonable
    assert results.best_max_abs_error >= 0


def test_gradient_clipping(simple_2d_setup) -> None:
    """Test that gradient clipping is applied when configured."""
    config, preprocessed_data = simple_2d_setup
    config.gradient_clip = 1.0  # Enable gradient clipping
    config.n_iters = 10

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    # Should run without errors
    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    assert results.actual_iters == 10


def test_gradient_clipping_none(simple_2d_setup) -> None:
    """Test that no gradient clipping works when set to None."""
    config, preprocessed_data = simple_2d_setup
    config.gradient_clip = None  # Disable gradient clipping
    config.n_iters = 10

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    # Should run without errors
    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    assert results.actual_iters == 10


def test_learning_rate_scheduling(simple_2d_setup) -> None:
    """Test that LR scheduler is called during optimization."""
    config, preprocessed_data = simple_2d_setup
    config.n_iters = 20

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    # Get initial learning rate from standard Adam
    initial_lr = components.optimizer.param_groups[0]["lr"]
    assert initial_lr > 0

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # LR may have changed (if loss plateaued)
    # At minimum, scheduler should have been called without errors
    assert results.actual_iters > 0


def test_movie_recording_enabled(simple_2d_setup) -> None:
    """Test movie frame recording when enabled."""
    config, preprocessed_data = simple_2d_setup
    config.napari_movie = True
    config.movie_every = 5
    config.movie_max_frames = 100
    config.n_iters = 15

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Movie frames should be recorded
    assert results.movie_frames is not None
    assert "target" in results.movie_frames
    assert "reconstruction" in results.movie_frames
    assert "residual" in results.movie_frames
    assert "splat_centers" in results.movie_frames
    assert "iterations" in results.movie_frames

    # Should have recorded frames at iterations 5, 10, 15
    assert len(results.movie_frames["target"]) >= 2


def test_movie_recording_disabled(simple_2d_setup) -> None:
    """Test that no movie is recorded when disabled."""
    config, preprocessed_data = simple_2d_setup
    config.napari_movie = False
    config.n_iters = 10

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # No movie frames
    assert results.movie_frames is None


def test_dynamic_operations_disabled(simple_2d_setup) -> None:
    """Test optimization with dynamic operations disabled."""
    config, preprocessed_data = simple_2d_setup
    config.enable_dynamic_ops = False
    config.n_iters = 10

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    initial_n_splats = components.model.n_splats()

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Number of splats should not have changed
    final_n_splats = len(results.amps)
    assert final_n_splats == initial_n_splats


def test_dynamic_operations_enabled(simple_2d_setup) -> None:
    """Test optimization with dynamic operations enabled."""
    config, preprocessed_data = simple_2d_setup
    config.enable_dynamic_ops = True
    # Create DynamicOpsConfig and set attributes
    dynamic_config = DynamicOpsConfig()
    dynamic_config.step_every = 5
    dynamic_config.seeding_error_threshold = 0.2
    dynamic_config.seeding_radius = 2.0
    dynamic_config.pruning_amp_threshold = 0.01
    dynamic_config.pruning_enabled = True
    dynamic_config.seeding_enabled = True
    config.dynamic_config = dynamic_config
    config.n_iters = 20

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Dynamic ops may have changed splat count
    # (could be more or less depending on seeding/pruning)
    final_n_splats = len(results.amps)
    # Just verify it ran without crashing
    assert final_n_splats > 0


def test_loss_decreases_over_iterations(simple_2d_setup) -> None:
    """Test that loss generally decreases during optimization."""
    config, preprocessed_data = simple_2d_setup
    config.n_iters = 30
    config.verbose = False

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    # Get initial loss
    with torch.no_grad():
        initial_pred = components.model()
        initial_loss = loss_fn(initial_pred).item()

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Final loss should be lower than initial
    assert results.best_loss < initial_loss


def test_verbose_logging(simple_2d_setup) -> None:
    """Test that verbose logging doesn't cause errors."""
    config, preprocessed_data = simple_2d_setup
    config.verbose = True
    config.n_iters = 10

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    # Should run without errors even with verbose output
    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    assert results.actual_iters == 10


def test_convergence_rel_l2_target(simple_2d_setup) -> None:
    """Test convergence via rel_l2_target criterion."""
    config, preprocessed_data = simple_2d_setup
    config.n_iters = 50

    # Set very tight max_abs_error so it won't trigger
    preprocessed_data.max_abs_error = 1e-10
    # Set very loose rel_l2_target so it triggers quickly
    preprocessed_data.rel_l2_target = 0.99

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Should converge via rel_l2 before hitting iteration limit
    assert results.converged_early
    assert results.actual_iters < config.n_iters
    assert results.best_rel_l2 >= 0


def test_rel_l2_target_none_no_effect(simple_2d_setup) -> None:
    """Test that rel_l2_target=None doesn't change behavior."""
    config, preprocessed_data = simple_2d_setup
    config.n_iters = 10

    # Set very tight max_abs_error so it won't trigger
    preprocessed_data.max_abs_error = 1e-10
    # Ensure rel_l2_target is None (default)
    preprocessed_data.rel_l2_target = None

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Should NOT converge (neither criterion met)
    assert not results.converged_early
    assert results.actual_iters == config.n_iters
    assert results.best_rel_l2 >= 0


def test_different_loss_types_in_optimization(simple_2d_setup) -> None:
    """Test optimization works with different loss types."""
    config, preprocessed_data = simple_2d_setup
    config.n_iters = 10

    for loss_type in ["mse", "l1", "poisson"]:
        config.loss_type = loss_type

        components = initialize_optimization(config, preprocessed_data)
        loss_fn = create_loss_function(config, preprocessed_data, components.model)

        results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

        # Should complete successfully for all loss types
        assert results.actual_iters == 10
        assert results.best_loss >= 0
