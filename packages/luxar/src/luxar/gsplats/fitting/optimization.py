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

from luxar.gsplats.dynamic_ops import apply_dynamic_operations
from luxar.gsplats.fitting.config import (
    FitConfig,
    ModelComponents,
    OptimizationResults,
    PreprocessedData,
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
        Model, optimizer, scheduler, and coordinator
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

    # Main optimization loop
    converged_early = False
    actual_iters = 0
    for it in range(1, config.n_iters + 1):
        actual_iters = it

        # Forward pass
        optimizer.zero_grad()
        pred = model()
        loss = loss_fn(pred)
        loss.backward()

        # Gradient clipping for stability
        if config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        # Learning rate scheduling (detach to avoid warning)
        scheduler.step(loss.detach())

        # Tracking
        current_loss = loss.item()

        # Movie frame recording (only if enabled and at specified intervals)
        if (
            config.napari_movie
            and movie_frames is not None
            and it % config.movie_every == 0
        ):
            _record_movie_frame(model, pred, V_t, movie_frames, config, it)

        # Update best loss tracking
        if current_loss < best_loss:
            best_loss = current_loss

        # Convergence check and best state tracking using maximum absolute error
        with torch.no_grad():
            current_max_abs_error = _compute_max_abs_error(pred, V_t)

            # Track best state based on max absolute error (quality guarantee)
            if current_max_abs_error < best_max_abs_error:
                best_max_abs_error = current_max_abs_error
                best_iteration = it

                # Save current best state (deep copy to avoid mutations)
                centers, Ls, amps = model.current_params()
                best_state = {
                    "centers": centers.detach().clone(),
                    "Ls": Ls.detach().clone(),
                    "amps": amps.detach().clone(),
                    "iteration": it,
                    "max_abs_error": current_max_abs_error,
                    "loss": current_loss,
                }

                # Smart logging: significant improvements or early iterations
                if config.verbose and (
                    it <= 10 or current_max_abs_error < best_max_abs_error * 0.95
                ):
                    aprint(
                        f"    ★ New best state: iteration {it}, max_abs_error={current_max_abs_error:.6f}"
                    )

            # Check for convergence
            if current_max_abs_error < preprocessed_data.max_abs_error:
                converged_early = True
                if config.verbose:
                    aprint(f"✓ CONVERGENCE ACHIEVED at iteration {it}")
                    aprint(
                        f"  Max absolute error: {current_max_abs_error:.6f} < threshold: {preprocessed_data.max_abs_error:.6f}"
                    )
                break

        # Dynamic operations (seeding and pruning)
        if config.enable_dynamic_ops and it % config.dynamic_config.step_every == 0:
            optimizer, scheduler, topology_changed = apply_dynamic_operations(
                model,
                optimizer,
                scheduler,
                V_t,  # target
                pred,  # current prediction
                config.dynamic_config,
                preprocessed_data.effective_lr,  # Use gradient dilution compensated learning rate
                max_abs_error_threshold=preprocessed_data.max_abs_error,  # Now always has a sensible value
                device=config.device,
                verbose=config.dynamic_ops_verbose,
            )

        # Logging (update N after potential dynamic ops)
        N = model.n_splats() if hasattr(model, "n_splats") else preprocessed_data.N
        if config.verbose and (it % max(1, config.n_iters // 10) == 0 or it <= 5):
            with torch.no_grad():
                rel = torch.linalg.norm((pred - V_t).reshape(-1)) / (
                    torch.linalg.norm(V_t.reshape(-1)) + 1e-12
                )
                # Calculate max absolute error for display
                current_max_abs_error = _compute_max_abs_error(pred, V_t)
            aprint(
                f"[{it:4d}/{config.n_iters}] loss={current_loss:.5g}  "
                f"relL2={float(rel):.4f}  maxAbsErr={current_max_abs_error:.5g}  N={N}"
            )

    end_time = time.time()

    # Log termination reason
    if config.verbose:
        if converged_early:
            aprint("✓ Optimization terminated: CONVERGENCE ACHIEVED")
        else:
            aprint("⚠ Optimization terminated: ITERATION LIMIT REACHED")
            aprint(
                f"  Final max absolute error: {best_max_abs_error:.6f} (threshold: {preprocessed_data.max_abs_error:.6f})"
            )

    # Get final parameters (will be overwritten by best state if available)
    if best_state is not None:
        if config.verbose:
            improvement = (
                f" (improved from {best_max_abs_error:.6f} to {best_state['max_abs_error']:.6f})"
                if best_iteration != actual_iters
                else ""
            )
            aprint(
                f"★ Restored best state from iteration {best_iteration}{improvement}"
            )

        centers = best_state["centers"]
        Ls = best_state["Ls"]
        amps = best_state["amps"]
        best_loss = best_state["loss"]
        best_max_abs_error = best_state["max_abs_error"]
    else:
        # Fallback to final state if no best state saved
        with torch.no_grad():
            centers, Ls, amps = model.current_params()
            best_max_abs_error = _compute_max_abs_error(model(), V_t)

    return OptimizationResults(
        centers=centers,
        Ls=Ls,
        amps=amps,
        converged_early=converged_early,
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
        centers, _, _ = model.current_params()
        centers_frame = centers.detach().cpu().numpy()

        movie_frames["target"].append(target_frame)
        movie_frames["reconstruction"].append(pred_frame)
        movie_frames["residual"].append(residual_frame)
        movie_frames["splat_centers"].append(centers_frame)
        movie_frames["iterations"].append(iteration)
