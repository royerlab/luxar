"""
Tests for fitting/optimization.py module.
"""

import numpy as np
import pytest
import torch
import zarr

from luxar.gsplats.fitting.config import (
    DynamicOpsConfig,
    FitConfig,
    ModelComponents,
    PreprocessedData,
)
from luxar.gsplats.fitting.initialization import initialize_optimization
from luxar.gsplats.fitting.losses import create_loss_function
from luxar.gsplats.fitting.optimization import (
    _compute_max_abs_error,
    _EarlyStopRule,
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
    assert results.relocation_statistics == {
        "total_relocations": 0,
        "unique_splats": 0,
    }


def test_run_optimization_loop_preserves_relocation_counts(simple_2d_setup) -> None:
    """Fit results retain both event and distinct-splat relocation counts."""
    config, preprocessed_data = simple_2d_setup
    config.enable_dynamic_ops = True
    config.n_iters = 2
    config.max_abs_error = 1e-10
    preprocessed_data.max_abs_error = 1e-10
    config.dynamic_config.step_every = 1
    config.dynamic_config.k_max_residuals = 1
    config.dynamic_config.enable_tiled_seeding = False
    config.dynamic_config.relocation_percentile = 100.0
    config.dynamic_config.max_relocations_per_step = 1
    config.dynamic_config.min_splats_to_keep = 0

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    assert results.relocation_statistics["total_relocations"] >= 1
    assert 1 <= results.relocation_statistics["unique_splats"] <= 2
    assert (
        results.relocation_statistics["unique_splats"]
        <= results.relocation_statistics["total_relocations"]
    )


def test_best_state_relocation_mask_excludes_later_sorted_relocations(
    simple_2d_setup,
) -> None:
    config, data = simple_2d_setup
    config.enable_dynamic_ops = True
    config.sort_splats_enabled = True
    config.sort_splats_interval = 1
    config.n_iters = 3
    config.max_abs_error = 1e-10
    data.max_abs_error = 1e-10
    config.dynamic_config.step_every = 1
    config.dynamic_config.k_max_residuals = 1
    config.dynamic_config.enable_tiled_seeding = False
    config.dynamic_config.relocation_percentile = 100.0
    config.dynamic_config.max_relocations_per_step = 1
    config.dynamic_config.min_splats_to_keep = 0

    components = initialize_optimization(config, data)
    # Equal loss selects the first snapshot; dynamic ops run after that snapshot.
    results = run_optimization_loop(
        components, lambda pred: pred.sum() * 0, config, data
    )
    assert results.best_iteration == 1
    assert results.relocation_statistics["total_relocations"] > 0
    assert results.relocated_mask is not None
    assert not results.relocated_mask.any()


def test_best_state_relocation_mask_tracks_returned_sorted_splats(
    simple_2d_setup,
) -> None:
    config, data = simple_2d_setup
    config.enable_dynamic_ops = True
    config.sort_splats_enabled = True
    config.sort_splats_interval = 1
    config.n_iters = 2
    config.max_abs_error = 1e-10
    data.max_abs_error = 1e-10
    config.dynamic_config.step_every = 1
    config.dynamic_config.k_max_residuals = 1
    config.dynamic_config.enable_tiled_seeding = False
    config.dynamic_config.relocation_percentile = 100.0
    config.dynamic_config.max_relocations_per_step = 1
    config.dynamic_config.min_splats_to_keep = 0

    components = initialize_optimization(config, data)
    loss_fn = create_loss_function(config, data, components.model)
    results = run_optimization_loop(components, loss_fn, config, data)

    # The post-final-step parameters are a candidate too (they are scored once
    # after the loop); the mask must follow the returned splats through every
    # sort, even when one splat is relocated twice.
    assert results.best_iteration == 2
    assert results.relocated_mask is not None
    assert results.relocation_statistics["total_relocations"] > 0
    assert (
        results.relocated_mask.sum() == results.relocation_statistics["unique_splats"]
    )
    Ls = results.Ls.detach().cpu().numpy()
    sigma = np.sqrt(np.sum(Ls**2, axis=2))
    near_reseed = np.all(np.abs(sigma - 0.5) <= 0.01, axis=1)
    # The just-reseeded splat is flagged, and no unrelocated splat sits at the
    # reseed sigma (a misaligned mask would flag the wrong one).
    assert np.count_nonzero(near_reseed & results.relocated_mask) >= 1
    assert not np.any(near_reseed & ~results.relocated_mask)


def test_convergence_early_stopping(simple_2d_setup) -> None:
    """Test early stopping when convergence criterion is met."""
    config, preprocessed_data = simple_2d_setup

    # Set very loose convergence criterion to trigger early stopping
    preprocessed_data.max_abs_error = (
        0.8  # Very loose (must be achievable in 50 CPU iters)
    )

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


class _ScalarSplatModel(torch.nn.Module):
    """Tiny model exposing the splat-model methods used by the optimizer."""

    shape = (1,)

    def __init__(self, value: float) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.param

    def current_params(self):
        centers = self.param.reshape(1, 1)
        Ls = torch.ones((1, 1, 1), dtype=self.param.dtype, device=self.param.device)
        amps = self.param.clone()
        return centers, Ls, amps

    @torch.no_grad()
    def replace_with(self, centers, _Ls, _amps) -> None:
        self.param = torch.nn.Parameter(centers.reshape(1).detach().clone())

    def n_splats(self) -> int:
        return 1


def test_best_state_scores_are_recomputed_from_restored_state() -> None:
    """Reported loss/metrics must match the restored best parameters exactly."""
    config = FitConfig(
        V=np.zeros((1,), dtype=np.float32),
        seeds=None,
        norm_percentile=0.0,
        init_sigma_vox=1.0,
        sigma_min_diag=[0.1],
        sigma_max_diag=None,
        truncate=3.0,
        n_iters=101,
        lr=0.01,
        max_abs_error=0.0,
        rel_l2_target=None,
        gradient_clip=None,
        loss_type="mse",
        asymmetric_penalty=None,
        l1_amp=None,
        l1_diag=None,
        scheduler_type="none",
        patience=10,
        lr_reduction_factor=0.5,
        early_stop_patience=None,
        enable_dynamic_ops=False,
        dynamic_config=DynamicOpsConfig(),
        dynamic_ops_verbose=False,
        napari_movie=False,
        movie_every=1,
        movie_max_frames=None,
        device=torch.device("cpu"),
        verbose=False,
    )
    target = torch.zeros(1, dtype=torch.float32)
    preprocessed_data = PreprocessedData(
        d=1,
        N=1,
        seed_centers=np.array([[0.0]], dtype=np.float32),
        V_normalized=np.zeros((1,), dtype=np.float32),
        V_tensor=target,
        image_min=0.0,
        image_max=0.0,
        intensity_range=1.0,
        max_abs_error=0.0,
    )
    model = _ScalarSplatModel(value=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    components = ModelComponents(model=model, optimizer=optimizer, scheduler=None)

    def loss_fn(pred: torch.Tensor) -> torch.Tensor:
        return torch.sum((pred - target) ** 2)

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    restored_value = results.centers.reshape(-1)[0]
    expected_loss = float((restored_value**2).item())
    expected_max_abs_error = float(abs(restored_value).item())
    expected_rel_l2 = expected_max_abs_error / 1e-12

    assert results.best_loss == pytest.approx(expected_loss)
    assert results.best_max_abs_error == pytest.approx(expected_max_abs_error)
    assert results.best_rel_l2 == pytest.approx(expected_rel_l2)


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
    assert len(results.movie_frames["reconstruction"]) >= 2
    # Target is stored once (single array, not per-frame list)
    assert isinstance(results.movie_frames["target"], np.ndarray)


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


def test_iter_callback_invoked(simple_2d_setup) -> None:
    """The iter_callback fires at the configured cadence with expected args."""
    config, preprocessed_data = simple_2d_setup
    config.n_iters = 100

    captured = []

    def callback(iteration, pred, info):
        captured.append(
            {
                "iteration": iteration,
                "pred_shape": tuple(pred.shape),
                "info_keys": set(info.keys()),
                "loss": info["loss"],
            }
        )

    config.iter_callback = callback
    config.iter_callback_every = 25

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)
    run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Expect at least 4 calls (n_iters=100, every=25 -> iters 25, 50, 75, 100)
    # and at most as many as eval iterations.
    assert len(captured) >= 4, f"Expected >=4 callback calls, got {len(captured)}"
    expected_keys = {"loss", "best_loss", "max_abs_error", "rel_l2", "n_splats"}
    for c in captured:
        assert expected_keys.issubset(c["info_keys"]), (
            f"info dict missing keys: {expected_keys - c['info_keys']}"
        )
        assert c["pred_shape"] == preprocessed_data.V_normalized.shape, (
            f"pred shape mismatch: {c['pred_shape']} vs {preprocessed_data.V_normalized.shape}"
        )
        assert c["iteration"] % 25 == 0, (
            f"callback fired at iter {c['iteration']}, not a multiple of 25"
        )
        assert c["loss"] >= 0


def test_iter_callback_none_zero_overhead(simple_2d_setup) -> None:
    """When iter_callback is None, the loop runs unchanged."""
    config, preprocessed_data = simple_2d_setup
    config.n_iters = 30
    config.iter_callback = None

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)
    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # No assertion on side-effects; just confirm it runs without error.
    assert results.actual_iters > 0


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
    dynamic_config.nms_radius_vox = 2.0
    config.dynamic_config = dynamic_config
    config.n_iters = 20

    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)

    # Dynamic ops use fixed-pool relocation (no topology changes)
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


# ---------------------------------------------------------------------------
# Sync-free best-state tracking and plateau scheduling
# ---------------------------------------------------------------------------


class TestDeviceBestState:
    """``_DeviceBestState`` must pick exactly what ``if loss < best`` picked."""

    def test_matches_host_comparison_semantics(self) -> None:
        from luxar.gsplats.fitting.optimization import _DeviceBestState

        losses = [5.0, 4.0, 4.0, 6.0, 3.5, float("nan"), 3.5, 2.0, 2.5]
        tracker = _DeviceBestState(torch.device("cpu"))
        metrics = torch.tensor([9.0, 9.0])
        best, best_it = float("inf"), 0
        for it, value in enumerate(losses, start=1):
            params = torch.full((3, 2), float(it))
            tracker.update(
                it,
                torch.tensor(value),
                {
                    "centers": params,
                    "Ls": params,
                    "amps": params[:, 0],
                    "relocated": None,
                },
                metrics * it,
            )
            if value < best:  # the old host-side rule
                best, best_it = value, it
        state = tracker.as_best_state()
        assert state is not None
        assert state["iteration"] == best_it == 8
        assert state["loss"] == best == 2.0
        assert torch.equal(state["centers"], torch.full((3, 2), 8.0))
        assert state["max_abs_error"] == 72.0  # last-known metrics AT iteration 8
        assert state["relocated"] is None

    def test_never_improving_yields_no_state(self) -> None:
        from luxar.gsplats.fitting.optimization import _DeviceBestState

        tracker = _DeviceBestState(torch.device("cpu"))
        p = torch.zeros(2, 2)
        for it in (1, 2):
            tracker.update(
                it,
                torch.tensor(float("nan")),
                {"centers": p, "Ls": p, "amps": p[:, 0], "relocated": None},
                torch.zeros(2),
            )
        assert tracker.as_best_state() is None

    def test_pool_resize_falls_back_to_host_decision(self) -> None:
        from luxar.gsplats.fitting.optimization import _DeviceBestState

        tracker = _DeviceBestState(torch.device("cpu"))
        small, big = torch.zeros(2, 2), torch.ones(3, 2)
        tracker.update(
            1,
            torch.tensor(1.0),
            {"centers": small, "Ls": small, "amps": small[:, 0]},
            torch.zeros(2),
        )
        tracker.update(
            2,
            torch.tensor(0.5),
            {"centers": big, "Ls": big, "amps": big[:, 0]},
            torch.zeros(2),
        )
        state = tracker.as_best_state()
        assert state is not None and state["centers"].shape == (3, 2)


class TestPlateauFeeder:
    """Buffered replay makes the same plateau decisions as per-iteration steps."""

    @staticmethod
    def _scheduler(patience: int):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=1.0)
        return opt, torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=patience, factor=0.5
        )

    def test_replay_reaches_the_same_lr_sequence(self) -> None:
        from luxar.gsplats.fitting.optimization import _PlateauFeeder

        gen = torch.Generator().manual_seed(0)
        losses = (1.0 + 0.3 * torch.rand(400, generator=gen)).tolist()
        ref_opt, ref = self._scheduler(15)
        reference = []
        for value in losses:
            ref.step(value)
            reference.append(ref_opt.param_groups[0]["lr"])

        opt, sched = self._scheduler(15)
        feeder = _PlateauFeeder(sched, eval_interval=25)
        assert feeder.interval == 5
        seen = []
        for i, value in enumerate(losses, start=1):
            feeder.push(torch.tensor(value), force_flush=(i % 25 == 0))
            if not feeder.pending:  # just flushed
                seen.append((i, opt.param_groups[0]["lr"]))
        # After every flush the LR equals the per-iteration reference at that
        # iteration: the decisions are identical, only applied at the flush.
        for i, lr in seen:
            assert lr == reference[i - 1]
        assert ref_opt.param_groups[0]["lr"] < 1.0  # reductions were exercised

    def test_interval_is_bounded_by_eval_interval_and_patience(self) -> None:
        from luxar.gsplats.fitting.optimization import _PlateauFeeder

        assert _PlateauFeeder(self._scheduler(15)[1], 25).interval == 5
        assert _PlateauFeeder(self._scheduler(15)[1], 1).interval == 1
        assert _PlateauFeeder(self._scheduler(2)[1], 25).interval == 1
        assert _PlateauFeeder(self._scheduler(300)[1], 25).interval == 25


def test_loop_has_no_per_iteration_best_loss_sync(
    simple_2d_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best loss is read back on eval iterations only, not every iteration."""
    from luxar.gsplats.fitting import optimization

    config, preprocessed_data = simple_2d_setup
    config.n_iters = 200  # eval interval 25
    components = initialize_optimization(config, preprocessed_data)
    loss_fn = create_loss_function(config, preprocessed_data, components.model)

    calls = {"sync": 0}
    real_sync = optimization._DeviceBestState.sync

    def counting_sync(self):
        calls["sync"] += 1
        return real_sync(self)

    monkeypatch.setattr(optimization._DeviceBestState, "sync", counting_sync)
    results = run_optimization_loop(components, loss_fn, config, preprocessed_data)
    # 5 warm-up evals + every 25th iteration + the final read-back (x2).
    assert calls["sync"] <= 5 + results.actual_iters // 25 + 3
    assert 0 < results.best_iteration <= results.actual_iters


def _scalar_setup(n_iters: int, lr: float, start: float, threshold: float = 1e-12):
    config = FitConfig(
        V=np.zeros((1,), dtype=np.float32),
        seeds=None,
        norm_percentile=0.0,
        init_sigma_vox=1.0,
        sigma_min_diag=[0.1],
        sigma_max_diag=None,
        truncate=3.0,
        n_iters=n_iters,
        lr=lr,
        max_abs_error=threshold,
        rel_l2_target=None,
        gradient_clip=None,
        loss_type="mse",
        asymmetric_penalty=None,
        l1_amp=None,
        l1_diag=None,
        scheduler_type="none",
        patience=10,
        lr_reduction_factor=0.5,
        early_stop_patience=None,
        enable_dynamic_ops=False,
        dynamic_config=DynamicOpsConfig(),
        dynamic_ops_verbose=False,
        napari_movie=False,
        movie_every=1,
        movie_max_frames=None,
        device=torch.device("cpu"),
        verbose=False,
    )
    target = torch.zeros(1, dtype=torch.float32)
    preprocessed = PreprocessedData(
        d=1,
        N=1,
        seed_centers=np.array([[0.0]], dtype=np.float32),
        V_normalized=np.zeros((1,), dtype=np.float32),
        V_tensor=target,
        image_min=0.0,
        image_max=0.0,
        intensity_range=1.0,
        max_abs_error=threshold,
    )
    model = _ScalarSplatModel(value=start)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    components = ModelComponents(model=model, optimizer=optimizer, scheduler=None)

    def loss_fn(pred: torch.Tensor) -> torch.Tensor:
        return torch.sum((pred - target) ** 2)

    return components, loss_fn, config, preprocessed


def test_best_state_is_the_iterate_its_loss_was_computed_from() -> None:
    """A warm start whose steps only make things worse must return the start.

    SGD with lr=1.5 on p**2 overshoots and diverges from p=0.1: the losses are
    0.01, 0.04, 0.16, ... so iteration 1 ranks best -- and the parameters that
    produced that loss are the STARTING ones. Snapshotting after
    ``optimizer.step()`` returned the post-first-step p=-0.2 (loss 0.04)
    instead, which is how warm-started refits came back worse than their seed.
    """
    components, loss_fn, config, preprocessed = _scalar_setup(
        n_iters=6, lr=1.5, start=0.1
    )
    results = run_optimization_loop(components, loss_fn, config, preprocessed)
    assert results.best_iteration == 1
    assert float(results.centers.reshape(-1)[0]) == pytest.approx(0.1)
    # The restored state's own score equals the loss it was ranked by.
    assert results.best_loss == pytest.approx(0.01)


def test_last_pre_step_state_is_not_reported_as_final_step(capsys) -> None:
    components, loss_fn, config, preprocessed = _scalar_setup(
        n_iters=1, lr=1.5, start=0.1
    )
    config.verbose = True
    results = run_optimization_loop(components, loss_fn, config, preprocessed)
    assert results.best_iteration == results.actual_iters == 1
    assert results.best_from_final_step is False
    assert float(results.centers.reshape(-1)[0]) == pytest.approx(0.1)
    assert "Restored best state from iteration 1" in capsys.readouterr().out


def test_zero_iterations_has_no_final_step() -> None:
    components, loss_fn, config, preprocessed = _scalar_setup(
        n_iters=0, lr=0.25, start=1.0
    )
    results = run_optimization_loop(components, loss_fn, config, preprocessed)
    assert results.actual_iters == 0
    assert results.best_from_final_step is False


def test_convergence_break_returns_the_iterate_that_converged() -> None:
    """``converged=True`` must mean the RETURNED splats meet the threshold.

    The convergence check scores the post-step parameters; restoring the
    pre-step best snapshot instead returned p=0.03125 > 0.03 while reporting
    convergence.
    """
    components, loss_fn, config, preprocessed = _scalar_setup(
        n_iters=50, lr=0.25, start=1.0, threshold=0.03
    )
    results = run_optimization_loop(components, loss_fn, config, preprocessed)
    assert results.converged_early
    assert results.best_from_final_step is True
    assert results.best_max_abs_error < 0.03
    assert abs(float(results.centers.reshape(-1)[0])) < 0.03


def test_last_step_is_a_candidate_at_the_iteration_limit(capsys) -> None:
    """The parameters after the final step are scored, not silently dropped."""
    components, loss_fn, config, preprocessed = _scalar_setup(
        n_iters=3, lr=0.25, start=1.0
    )
    config.verbose = True
    results = run_optimization_loop(components, loss_fn, config, preprocessed)
    assert results.best_from_final_step is True
    assert "Best state is from final iteration" in capsys.readouterr().out
    # p: 1 -> 0.5 -> 0.25 -> 0.125; only the post-step-3 iterate scores 0.0156.
    assert float(results.centers.reshape(-1)[0]) == pytest.approx(0.125)
    assert results.best_loss == pytest.approx(0.125**2)


@pytest.mark.slow
def test_warm_start_from_a_good_fit_does_not_get_worse() -> None:
    """Refitting from an existing fit (seeds=GSplatData) must not lose quality."""
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats

    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[0:24, 0:24, 0:24]
    vol = np.zeros((24, 24, 24), np.float32)
    for _ in range(6):
        c = rng.uniform(5, 19, 3)
        vol += np.exp(-((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / 6)
    kwargs = dict(device="cpu", verbose=False, cull_retention=0.0, use_cuda=False)
    first = fit_gaussian_splats(vol, seeds=60, n_iters=300, **kwargs)
    # The seed's own score as the refit sees it (n_iters=1 restores the start;
    # the GSplatData -> seed round trip itself is not what is under test).
    start = fit_gaussian_splats(vol, seeds=first, n_iters=1, lr=0.05, **kwargs)
    # lr=0.05 overshoots from a converged fit, so every step is worse than the
    # start: the best state must BE the start, not the iterate after step 1
    # (which scored 24 dB here before the fix, against ~60 dB for the seed).
    refit = fit_gaussian_splats(vol, seeds=first, n_iters=60, lr=0.05, **kwargs)
    assert refit.stats["psnr_db"] >= start.stats["psnr_db"] - 0.01
    assert refit.stats["psnr_db"] > first.stats["psnr_db"] - 1.0


class TestEarlyStopRule:
    """The patience rule, with and without a relative tolerance."""

    def test_plain_rule_waits_for_no_improvement_at_all(self) -> None:
        rule = _EarlyStopRule(patience=100, rel_tol=None)
        # Creeping by 1e-6 per eval never stops the plain rule ...
        for it in range(25, 1001, 25):
            assert not rule.should_stop(it, 1.0 - 1e-6 * it, 0)
        # ... only a full window without a new best does.
        assert rule.should_stop(1025, 0.999, 100) == "plain"

    def test_relative_rule_stops_on_a_crawl(self) -> None:
        rule = _EarlyStopRule(patience=100, rel_tol=1e-3)
        stopped = [
            it for it in range(25, 1001, 25) if rule.should_stop(it, 1.0 - 1e-6 * it, 0)
        ]
        # First eval with a full window behind it (it - 100 >= 25).
        assert stopped and stopped[0] == 125
        fresh = _EarlyStopRule(patience=100, rel_tol=1e-3)
        reasons = [fresh.should_stop(it, 1.0 - 1e-6 * it, 0) for it in (25, 50, 125)]
        assert reasons == [None, None, "relative"]

    def test_relative_rule_keeps_going_while_progress_is_real(self) -> None:
        rule = _EarlyStopRule(patience=100, rel_tol=1e-3)
        # 1% per window, ten times the tolerance.
        for it in range(25, 2001, 25):
            assert not rule.should_stop(it, 0.9999**it, 0)

    def test_relative_rule_needs_a_full_window(self) -> None:
        rule = _EarlyStopRule(patience=300, rel_tol=0.5)
        # Flat loss, but no eval is a full window old yet.
        assert not rule.should_stop(1, 1.0, 0)
        assert not rule.should_stop(250, 1.0, 0)
        assert rule.should_stop(325, 1.0, 0)

    def test_disabled_patience_disables_both(self) -> None:
        rule = _EarlyStopRule(patience=None, rel_tol=0.5)
        assert not rule.should_stop(10_000, 1.0, 10_000)


def test_relative_early_stop_ends_a_fit_sooner(tmp_path) -> None:
    """End to end: the tolerance cuts the tail of a real fit."""
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.io.load_gsplats import load_gsplats

    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[0:16, 0:24, 0:24]
    vol = np.zeros((16, 24, 24), np.float32)
    for _ in range(8):
        c = rng.uniform([3, 3, 3], [13, 21, 21])
        vol += np.exp(-((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / 6)
    vol += rng.normal(0, 0.02, vol.shape).astype(np.float32)
    kwargs = dict(
        seeds=40,
        n_iters=2000,
        early_stop_patience=200,
        device="cpu",
        verbose=False,
        use_cuda=False,
    )
    plain = fit_gaussian_splats(vol, **kwargs)
    relative = fit_gaussian_splats(vol, early_stop_rel_tol=1e-2, **kwargs)
    assert relative.stats["iterations"] < plain.stats["iterations"]
    assert relative.stats["early_stopped"]
    assert relative.stats["early_stop_reason"] == "relative"
    assert relative.stats["early_stop_rel_tol"] == 1e-2
    assert plain.stats["early_stop_rel_tol"] is None
    assert plain.stats["early_stop_reason"] in ("plain", None)

    for name, result in (("plain", plain), ("relative", relative)):
        path = tmp_path / f"{name}.gsplats.zarr"
        result.save(path)
        fitting = zarr.open_group(str(path), mode="r")["fitting"].attrs
        assert fitting["early_stop_reason"] == result.stats["early_stop_reason"]
        assert fitting["early_stop_rel_tol"] == result.stats["early_stop_rel_tol"]
        loaded = load_gsplats(path, include_stats=True)
        assert loaded.stats["early_stop_reason"] == result.stats["early_stop_reason"]
        assert loaded.stats["early_stop_rel_tol"] == result.stats["early_stop_rel_tol"]


def test_stop_message_names_the_rule_that_fired(capsys) -> None:
    """The verbose message must not describe a plain stop as a relative one."""
    _EarlyStopRule(patience=300, rel_tol=None).announce("plain", 900, 300)
    plain = capsys.readouterr().out
    assert "No improvement for 300 iterations" in plain
    assert "relative" not in plain
    _EarlyStopRule(patience=300, rel_tol=1e-3).announce("relative", 900, 25)
    relative = capsys.readouterr().out
    assert "improved by <= 0.001 (relative) over the last 300 iterations" in relative
