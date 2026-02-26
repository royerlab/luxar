"""
Optimization loop logic for Gaussian splat fitting.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Dict

import torch
from arbol import aprint

if TYPE_CHECKING:
    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

from luxar.gsplats.fitting.config import (
    FitConfig,
    ModelComponents,
    OptimizationResults,
    PreprocessedData,
)
from luxar.gsplats.fitting.dynamic_ops import (
    RecentlyRelocatedTracker,
    apply_dynamic_operations,
)


def _compute_max_abs_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute maximum absolute error between prediction and target.

    Parameters
    ----------
    pred : torch.Tensor
        Model prediction
    target : torch.Tensor
        Target values

    Returns
    -------
    float
        Maximum absolute error across all elements
    """
    return torch.max(torch.abs(pred - target)).item()


def run_optimization_loop(
    components: ModelComponents,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    config: FitConfig,
    preprocessed_data: PreprocessedData,
) -> OptimizationResults:
    """
    Run the main optimization loop.

    Parameters
    ----------
    components : ModelComponents
        Model, optimizer, and scheduler
    loss_fn : Callable
        Loss function that takes prediction and returns loss
    config : FitConfig
        Configuration for optimization
    preprocessed_data : PreprocessedData
        Preprocessed data including target tensor

    Returns
    -------
    OptimizationResults
        Results from optimization including best state and statistics
    """
    model = components.model
    optimizer = components.optimizer
    scheduler = components.scheduler
    V_t = preprocessed_data.V_tensor

    start_time = time.time()

    # Tracking
    best_loss = float("inf")

    # Movie recording setup (only if enabled)
    movie_frames = None
    if config.napari_movie:
        movie_frames = {
            "target": [],
            "reconstruction": [],
            "residual": [],
            "iterations": [],
            "splat_centers": [],
        }

    # Log convergence criteria before starting optimization
    if config.verbose:
        aprint(
            f"Convergence criterion: max absolute error < {preprocessed_data.max_abs_error:.6f}"
        )
        aprint(f"Maximum iterations: {config.n_iters}")

    # Initialize best state tracking for quality guarantee
    best_max_abs_error = float("inf")
    best_state = None
    best_iteration = 0
    iterations_since_improvement = 0  # Track patience for early stopping
    last_max_abs_error = float("inf")

    # Track initial splat count (used for dynamic ops logging)
    n_splats = model.n_splats() if hasattr(model, "n_splats") else preprocessed_data.N

    # Initialize relocation tracker for dynamic ops (prevents repeated relocation)
    relocation_tracker = None
    if config.enable_dynamic_ops:
        relocation_tracker = RecentlyRelocatedTracker(
            n_splats=n_splats,
            cooldown_steps=config.dynamic_config.relocation_cooldown_steps,
            device=V_t.device,
        )
        if config.verbose:
            aprint(
                f"Dynamic ops enabled: Splat relocation with cooldown "
                f"({config.dynamic_config.relocation_cooldown_steps} steps)"
            )

    # Main optimization loop
    converged_early = False
    early_stopped = False
    actual_iters = 0
    for it in range(1, config.n_iters + 1):
        actual_iters = it

        # Forward pass
        optimizer.zero_grad()
        pred = model()
        loss = loss_fn(pred)
        loss.backward()  # type: ignore[no-untyped-call]

        # Gradient clipping for stability
        if config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        # Learning rate scheduling
        if scheduler is not None:
            # ReduceLROnPlateau schedulers require metrics, others don't
            # Check by class name to handle both PyTorch and per-splat versions
            scheduler_name = type(scheduler).__name__
            if "Plateau" in scheduler_name:
                scheduler.step(loss.detach())  # Plateau schedulers need loss
            else:
                scheduler.step()  # Exponential and other schedulers don't need loss

        with torch.no_grad():
            pred_eval = model()
            loss_eval = loss_fn(pred_eval)

        # Tracking (use post-step metrics to match current model state)
        current_loss = float(loss_eval.item())

        # Movie frame recording (only if enabled and at specified intervals)
        if (
            config.napari_movie
            and movie_frames is not None
            and it % config.movie_every == 0
        ):
            _record_movie_frame(model, pred_eval, V_t, movie_frames, config, it)

        # Convergence check and best state tracking using maximum absolute error
        with torch.no_grad():
            current_max_abs_error = _compute_max_abs_error(pred_eval, V_t)
            last_max_abs_error = current_max_abs_error

            # Track best state based on loss (smoother signal for optimization progress)
            if current_loss < best_loss:
                # Save previous best for logging comparison
                previous_best = best_loss

                best_loss = current_loss
                best_max_abs_error = current_max_abs_error
                best_iteration = it
                iterations_since_improvement = 0  # Reset patience counter

                # Save current best state (deep copy to avoid mutations)
                centers, Ls, amps, sharpness = model.current_params()
                best_state = {
                    "centers": centers.detach().clone(),
                    "Ls": Ls.detach().clone(),
                    "amps": amps.detach().clone(),
                    "sharpness": sharpness.detach().clone(),
                    "iteration": it,
                    "max_abs_error": current_max_abs_error,
                    "loss": current_loss,
                }

                # Smart logging: significant improvements or early iterations
                if config.verbose and (it <= 10 or current_loss < previous_best * 0.95):
                    aprint(
                        f"    ★ New best state: iteration {it}, "
                        f"loss={current_loss:.6f}  max_abs_error={current_max_abs_error:.6f}"
                    )
            else:
                # No improvement this iteration
                iterations_since_improvement += 1

            # Check for convergence
            if current_max_abs_error < preprocessed_data.max_abs_error:
                converged_early = True
                if config.verbose:
                    aprint(f"✓ CONVERGENCE ACHIEVED at iteration {it}")
                    aprint(
                        f"  Max absolute error: {current_max_abs_error:.6f} < threshold: {preprocessed_data.max_abs_error:.6f}"
                    )
                break

            # Check for early stopping (patience-based)
            if config.early_stop_patience is not None:
                if iterations_since_improvement >= config.early_stop_patience:
                    early_stopped = True
                    if config.verbose:
                        aprint(f"⏹ EARLY STOPPING at iteration {it}")
                        aprint(
                            f"  No improvement for {iterations_since_improvement} iterations "
                            f"(patience: {config.early_stop_patience})"
                        )
                        aprint(
                            f"  Best state from iteration {best_iteration}: "
                            f"loss={best_loss:.6f}  max_abs_error={best_max_abs_error:.6f}"
                        )
                    break

        # Dynamic operations (splat relocation)
        if config.enable_dynamic_ops and it % config.dynamic_config.step_every == 0:
            apply_dynamic_operations(
                model,
                V_t,  # target
                pred_eval,  # current prediction
                config.dynamic_config,
                max_abs_error_threshold=preprocessed_data.max_abs_error,
                optimizer=optimizer,  # For resetting Adam state
                relocation_tracker=relocation_tracker,  # For cooldown tracking
                verbose=config.dynamic_ops_verbose,
            )
            # Advance tracker step counter (after relocation is complete)
            if relocation_tracker is not None:
                relocation_tracker.advance_step()

        # Logging (update N after potential dynamic ops)
        N = model.n_splats() if hasattr(model, "n_splats") else preprocessed_data.N
        if config.verbose and (it % max(1, config.n_iters // 10) == 0 or it <= 5):
            with torch.no_grad():
                rel = torch.linalg.norm((pred_eval - V_t).reshape(-1)) / (
                    torch.linalg.norm(V_t.reshape(-1)) + 1e-12
                )
            # Reuse current_max_abs_error computed earlier (line 149) - no redundant computation
            aprint(
                f"[{it:4d}/{config.n_iters}] loss={current_loss:.5g}  "
                f"relL2={float(rel):.4f}  maxAbsErr={current_max_abs_error:.5g}  N={N}"
            )

    end_time = time.time()

    # Log termination reason
    if config.verbose:
        if converged_early:
            aprint("✓ Optimization terminated: CONVERGENCE ACHIEVED")
        elif early_stopped:
            aprint("⏹ Optimization terminated: EARLY STOPPING (no improvement)")
            aprint(f"  No improvement for {config.early_stop_patience} iterations")
            aprint(
                f"  Best state from iteration {best_iteration}: "
                f"loss={best_loss:.6f}  max_abs_error={best_max_abs_error:.6f}"
            )
        else:
            aprint("⚠ Optimization terminated: ITERATION LIMIT REACHED")
            aprint(
                f"  Final max absolute error: {last_max_abs_error:.6f} "
                f"(threshold: {preprocessed_data.max_abs_error:.6f})"
            )

        # Log relocation statistics
        if relocation_tracker is not None:
            current_n_splats = (
                model.n_splats() if hasattr(model, "n_splats") else preprocessed_data.N
            )
            stats = relocation_tracker.get_statistics()
            aprint("\n📊 Dynamic Operations Summary:")
            aprint(f"  Total relocations: {stats['total_relocations']}")
            aprint(
                f"  Unique splats relocated: {stats['unique_splats']} / {current_n_splats}"
            )
            coverage_pct = (
                stats["unique_splats"] / max(1, current_n_splats)
            ) * 100
            aprint(
                f"  Coverage: {coverage_pct:.1f}% of splats were relocated at least once"
            )
            if stats["unique_splats"] > 0:
                avg_relocations = stats["total_relocations"] / stats["unique_splats"]
                aprint(
                    f"  Average relocations per relocated splat: {avg_relocations:.1f}"
                )

    # Get final parameters (will be overwritten by best state if available)
    if best_state is not None:
        if config.verbose:
            if best_iteration != actual_iters:
                aprint(
                    f"★ Restored best state from iteration {best_iteration} "
                    f"(loss={best_loss:.6f}  max_abs_error={best_max_abs_error:.6f})"
                )
            else:
                aprint("★ Best state is from final iteration")

        centers = best_state["centers"]
        Ls = best_state["Ls"]
        amps = best_state["amps"]
        sharpness = best_state["sharpness"]
        best_loss = best_state["loss"]
        best_max_abs_error = best_state["max_abs_error"]
    else:
        # Fallback to final state if no best state saved
        with torch.no_grad():
            centers, Ls, amps, sharpness = model.current_params()
            best_max_abs_error = _compute_max_abs_error(model(), V_t)

    return OptimizationResults(
        centers=centers,
        Ls=Ls,
        amps=amps,
        sharpness=sharpness,
        converged_early=converged_early,
        early_stopped=early_stopped,
        actual_iters=actual_iters,
        best_iteration=best_iteration,
        best_loss=best_loss,
        best_max_abs_error=best_max_abs_error,
        movie_frames=movie_frames,
        start_time=start_time,
        end_time=end_time,
    )


def _record_movie_frame(
    model: "GaussianSplatModel",
    pred: torch.Tensor,
    V_t: torch.Tensor,
    movie_frames: Dict[str, Any],
    config: FitConfig,
    iteration: int,
) -> None:
    """Record a frame for the optimization movie."""
    with torch.no_grad():
        # Memory-bounded recording: remove oldest frames if limit exceeded
        if len(movie_frames["target"]) >= config.movie_max_frames:
            # Remove oldest frame (FIFO)
            for key in [
                "target",
                "reconstruction",
                "residual",
                "splat_centers",
                "iterations",
            ]:
                movie_frames[key].pop(0)

        # Store frames as numpy arrays (detached from computation graph)
        target_frame = V_t.cpu().numpy()
        pred_frame = pred.detach().cpu().numpy()
        residual_frame = torch.abs(V_t - pred.detach()).cpu().numpy()

        # Record current splat centers
        centers, _, _, _ = model.current_params()
        centers_frame = centers.detach().cpu().numpy()

        movie_frames["target"].append(target_frame)
        movie_frames["reconstruction"].append(pred_frame)
        movie_frames["residual"].append(residual_frame)
        movie_frames["splat_centers"].append(centers_frame)
        movie_frames["iterations"].append(iteration)
