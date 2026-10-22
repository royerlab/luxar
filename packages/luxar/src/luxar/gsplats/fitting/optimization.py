"""
Optimization loop logic for Gaussian splat fitting.

Speed optimisations (empirically validated)
--------------------------------------------
1. **Eval frequency** (−21.5%): The eval forward pass (for convergence checking
   and best-state metrics) runs every 25 iterations instead of every iteration.
   Training loss is used for scheduler and best-loss tracking on non-eval iters.

2. **GPU sync elimination**: no per-iteration host sync. The best loss AND the
   best-state snapshot are tracked on the device with ``torch.where``
   (:class:`_DeviceBestState`) and read back only on eval iterations; the
   ``ReduceLROnPlateau`` scheduler is fed from a device-side loss buffer
   replayed every few iterations (:class:`_PlateauFeeder`). The old loop's
   ``if loss < best`` and ``scheduler.step(loss)`` each forced a blocking
   GPU→host sync on EVERY iteration, which drains the queue and serializes
   CPU dispatch with GPU execution.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch
from arbol import aprint

if TYPE_CHECKING:
    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

from luxar.gsplats.fitting.config import (
    _EMPTY_RELOCATION_STATISTICS,
    FitConfig,
    ModelComponents,
    OptimizationResults,
    PreprocessedData,
)
from luxar.gsplats.fitting.dynamic_ops import (
    RecentlyRelocatedTracker,
    apply_dynamic_operations,
)
from luxar.gsplats.fitting.sorting import sort_splats_by_morton_order


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


def _completed_relocation_statistics(
    relocation_tracker: Optional[RecentlyRelocatedTracker],
) -> dict[str, int]:
    """Return persisted relocation counters without transient cooldown state."""
    if relocation_tracker is None:
        return _EMPTY_RELOCATION_STATISTICS.copy()
    tracker_statistics = relocation_tracker.get_statistics()
    return {
        "total_relocations": tracker_statistics["total_relocations"],
        "unique_splats": tracker_statistics["unique_splats"],
    }


def _relocated_snapshot(
    tracker: Optional[RecentlyRelocatedTracker],
) -> Optional[torch.Tensor]:
    """Copy of the ever-relocated mask in the live splat order (None: no tracker)."""
    if tracker is None:
        return None
    return (tracker.last_relocation_step >= 0).detach().clone()


def _returned_relocated_mask(
    best_state: Optional[dict[str, Any]],
    tracker: Optional[RecentlyRelocatedTracker],
) -> Optional[np.ndarray]:
    """Relocated mask aligned with the returned splats (snapshot first, else live)."""
    if best_state is not None and best_state.get("relocated") is not None:
        return np.asarray(best_state["relocated"].cpu().numpy())
    if tracker is not None:
        return (tracker.last_relocation_step >= 0).cpu().numpy()
    return None


def _compute_eval_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[float, float]:
    """Compute max absolute error and relative L2 in one pass.

    Shares the ``(pred - target)`` diff tensor between both metrics,
    halving peak GPU memory compared to calling them separately.
    For a 767³ volume this saves ~1.8 GB.

    Returns
    -------
    tuple[float, float]
        (max_abs_error, rel_l2)
    """
    diff = pred - target
    max_abs_err = torch.max(torch.abs(diff)).item()
    rel_l2 = float(
        torch.linalg.norm(diff.reshape(-1))
        / (torch.linalg.norm(target.reshape(-1)) + 1e-12)
    )
    del diff
    return max_abs_err, rel_l2


class _DeviceBestState:
    """Best-loss state tracked on the device, without a host sync per iteration.

    Each :meth:`update` folds the current loss and parameters into the running
    best with ``torch.where`` on a 0-d device condition, so selecting the best
    iterate never blocks on the GPU. The scalar bookkeeping (best loss, best
    iteration, whether any iterate improved) is read back by :meth:`sync`, which
    the loop calls only on eval iterations. Semantics match the old
    ``if loss < best: snapshot`` exactly: a strict improvement replaces the
    WHOLE snapshot (parameters, relocation mask, the last-known eval metrics
    and the iteration), anything else keeps it.
    """

    def __init__(self, device: torch.device) -> None:
        """Start with no best state: loss +inf, iteration 0, no snapshot."""
        self.device = device
        self.loss = torch.tensor(float("inf"), device=device)
        self.iteration = torch.tensor(0, device=device, dtype=torch.long)
        self.from_final_step = torch.tensor(False, device=device)
        self.metrics = torch.full((2,), float("inf"), device=device)
        self.tensors: Optional[Dict[str, Optional[torch.Tensor]]] = None

    def update(
        self,
        it: int,
        loss: torch.Tensor,
        current: Dict[str, Optional[torch.Tensor]],
        current_metrics: torch.Tensor,
        *,
        final_step: bool = False,
    ) -> None:
        """Fold iteration ``it`` into the running best without a host sync.

        ``loss`` must have been computed from exactly the parameters in
        ``current`` (the loop calls this before ``optimizer.step()``); a
        strict improvement replaces the whole snapshot, anything else keeps it.
        """
        improved = loss < self.loss
        self.loss = torch.where(improved, loss, self.loss)
        self.iteration = torch.where(
            improved, torch.full_like(self.iteration, it), self.iteration
        )
        self.from_final_step = torch.where(
            improved,
            torch.full_like(self.from_final_step, final_step),
            self.from_final_step,
        )
        self.metrics = torch.where(improved, current_metrics, self.metrics)
        if self.tensors is None or not self._same_layout(current):
            # First iterate (or a pool resize): there is no snapshot of this
            # layout to blend into, so decide on the host -- rare by design.
            if self.tensors is None or bool(improved):
                self.tensors = {
                    k: (v.detach().clone() if v is not None else None)
                    for k, v in current.items()
                }
            return
        for key, value in current.items():
            best = self.tensors[key]
            if value is not None and best is not None:
                self.tensors[key] = torch.where(improved, value.detach(), best)

    def force(
        self,
        it: int,
        loss: torch.Tensor,
        current: Dict[str, Optional[torch.Tensor]],
        current_metrics: torch.Tensor,
    ) -> None:
        """Make ``current`` the snapshot unconditionally (a convergence break).

        The convergence check scored the live, post-step parameters; those
        are the ones that met the threshold, so they -- not an earlier
        lower-training-loss snapshot -- are what the fit must return.
        """
        self.loss = loss.detach().clone()
        self.iteration = torch.full_like(self.iteration, it)
        self.from_final_step = torch.tensor(True, device=self.device)
        self.metrics = current_metrics.clone()
        self.tensors = {
            k: (v.detach().clone() if v is not None else None)
            for k, v in current.items()
        }

    def _same_layout(self, current: Dict[str, Optional[torch.Tensor]]) -> bool:
        """True when ``current`` has the snapshot's keys, presence and shapes."""
        assert self.tensors is not None
        for key, value in current.items():
            best = self.tensors.get(key)
            if (value is None) != (best is None):
                return False
            if value is not None and best is not None and value.shape != best.shape:
                return False
        return True

    def sync(self) -> tuple[float, int]:
        """``(best_loss, best_iteration)`` on the host (one transfer)."""
        # The first copy waits for the queue; the second then has nothing to
        # wait for (MPS has no float64, so the two dtypes are not stacked).
        return float(self.loss.cpu()), int(self.iteration.cpu())

    def as_best_state(self) -> Optional[dict[str, Any]]:
        """The snapshot in the old ``best_state`` dict shape (None: never improved)."""
        best_loss, best_iteration = self.sync()
        if best_iteration == 0 or self.tensors is None:
            return None
        mae, rel_l2 = self.metrics.cpu().tolist()
        return {
            "centers": self.tensors["centers"],
            "Ls": self.tensors["Ls"],
            "amps": self.tensors["amps"],
            "iteration": best_iteration,
            "relocated": self.tensors.get("relocated"),
            "max_abs_error": mae,
            "rel_l2": rel_l2,
            "loss": best_loss,
        }


class _PlateauFeeder:
    """Feeds ``ReduceLROnPlateau`` every iteration's loss without a sync each.

    ``ReduceLROnPlateau.step`` converts its metric to a Python float, so
    stepping it per iteration forced a GPU→host sync per iteration. The losses
    are instead buffered on the device and replayed into the scheduler, in
    order and one ``step`` per iteration, every :attr:`interval` iterations
    (and on every eval iteration, which syncs anyway). The scheduler therefore
    sees exactly the same per-iteration loss sequence, with the same
    patience/cooldown accounting in iterations; the only change is that a
    learning-rate reduction takes effect when the buffer is replayed, at most
    ``interval - 1`` iterations after the iteration that triggered it.
    ``interval`` is a third of the patience (5 for the default 15), capped by
    the eval interval, so the delay stays well inside the patience window.
    """

    def __init__(self, scheduler: Any, eval_interval: int) -> None:
        """Wrap ``scheduler``; replay every ``max(1, min(eval_interval, patience // 3))``."""
        self.scheduler = scheduler
        patience = int(getattr(scheduler, "patience", 3))
        self.interval = max(1, min(eval_interval, patience // 3))
        self.pending: list[torch.Tensor] = []

    def push(self, loss: torch.Tensor, force_flush: bool) -> None:
        """Buffer one iteration's loss; replay when the buffer is full or forced."""
        self.pending.append(loss)
        if force_flush or len(self.pending) >= self.interval:
            self.flush()

    def flush(self) -> None:
        """Replay the buffered losses into the scheduler, in order (one host sync)."""
        if not self.pending:
            return
        values = torch.stack(self.pending).cpu().tolist()
        self.pending = []
        for value in values:
            self.scheduler.step(value)


class _EarlyStopRule:
    """Patience-based early stop, optionally on RELATIVE progress.

    The plain rule (``rel_tol=None``) stops once the best loss has not
    improved at all for ``patience`` iterations. With ``rel_tol`` set it also
    stops once the best loss improved by less than ``rel_tol`` (a fraction of
    the best loss ``patience`` iterations ago) over that window, so a fit
    creeping down by 1e-6 per eval does not run out its whole budget. Both are
    checked only on eval iterations, from the host-side best loss the loop
    already syncs there.
    """

    def __init__(self, patience: Optional[int], rel_tol: Optional[float]) -> None:
        """Remember the window and tolerance; no history yet."""
        self.patience = patience
        self.rel_tol = rel_tol
        self.history: list[tuple[int, float]] = []

    def should_stop(
        self, it: int, best_loss: float, iterations_since_improvement: int
    ) -> Optional[str]:
        """Record ``(it, best_loss)``; return which rule fired, or None.

        ``"plain"``: no new best loss for a whole window. ``"relative"``: the
        best loss improved, but by at most ``rel_tol`` of its value one window
        earlier.
        """
        if self.patience is None:
            return None
        if iterations_since_improvement >= self.patience:
            return "plain"
        if self.rel_tol is None:
            return None
        self.history.append((it, best_loss))
        # Best loss at the latest eval at or before the window start.
        start = it - self.patience
        while len(self.history) > 1 and self.history[1][0] <= start:
            self.history.pop(0)
        then_it, then_loss = self.history[0]
        if then_it > start or not np.isfinite(then_loss):
            return None
        gain = then_loss - best_loss
        return "relative" if gain <= self.rel_tol * abs(then_loss) else None

    def announce(self, reason: str, it: int, iterations_since_improvement: int) -> None:
        """Print which rule stopped the fit at iteration ``it``."""
        aprint(f"⏹ EARLY STOPPING at iteration {it}")
        if reason == "relative":
            aprint(
                f"  Best loss improved by <= {self.rel_tol:g} (relative) over the "
                f"last {self.patience} iterations"
            )
        else:
            aprint(
                f"  No improvement for {iterations_since_improvement} iterations "
                f"(patience: {self.patience})"
            )


def _stop_reason(converged: bool, rule_reason: Optional[str]) -> Optional[str]:
    """``early_stop_reason`` for the stats: converged / plain / relative / None."""
    return "converged" if converged else rule_reason


def _rank_final_iterate(
    tracker: _DeviceBestState,
    model: Any,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    relocation_tracker: Optional[RecentlyRelocatedTracker],
    actual_iters: int,
    *,
    converged_metrics: Optional[torch.Tensor],
    current_metrics: torch.Tensor,
) -> None:
    """Score the parameters left after the last step (never ranked in-loop).

    Every in-loop loss is computed BEFORE its step, so the final step's result
    would otherwise never be a candidate. On a convergence break
    (``converged_metrics`` given) those parameters are exactly what met the
    threshold, so they are returned outright; otherwise they compete with the
    best snapshot.
    """
    with torch.no_grad():
        final_pred = model()
        final_loss = loss_fn(final_pred).detach()
        del final_pred
        centers, Ls, amps = model.current_params()
        candidate = {
            "centers": centers,
            "Ls": Ls,
            "amps": amps,
            "relocated": _relocated_snapshot(relocation_tracker),
        }
        if converged_metrics is not None:
            tracker.force(actual_iters, final_loss, candidate, converged_metrics)
        else:
            tracker.update(
                actual_iters, final_loss, candidate, current_metrics, final_step=True
            )


def _append_bounded_sample(curve: list[list[float]], sample: list[float]) -> None:
    """Keep endpoints and a sparse history without growing with iteration count."""
    curve.append(sample)
    if len(curve) > 256:
        curve[:] = [*curve[::2][:-1], curve[-1]]


def _record_relocations(curve: list[list[float]], tracker: Any, iteration: int) -> None:
    total = tracker.total_relocations
    if total and (not curve or total > curve[-1][1]):
        _append_bounded_sample(curve, [float(iteration), float(total)])


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
    movie_frames: Optional[Dict[str, Any]] = None
    if config.napari_movie:
        movie_frames = {
            "target": V_t.cpu().numpy(),  # Store once (target never changes)
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
        if preprocessed_data.rel_l2_target is not None:
            aprint(
                f"Convergence criterion: relL2 < {preprocessed_data.rel_l2_target:.6f}"
            )
        aprint(f"Maximum iterations: {config.n_iters}")

    # Initialize best state tracking for quality guarantee
    best_max_abs_error = float("inf")
    best_rel_l2 = float("inf")
    best_iteration = 0
    iterations_since_improvement = 0  # Track patience for early stopping
    rule_reason: Optional[str] = None
    stop_rule = _EarlyStopRule(config.early_stop_patience, config.early_stop_rel_tol)
    last_max_abs_error = float("inf")

    # Track initial splat count (used for dynamic ops logging)
    n_splats = model.n_splats() if hasattr(model, "n_splats") else preprocessed_data.N

    # Initialize relocation tracker for dynamic ops (prevents repeated relocation)
    relocation_tracker = None
    if config.enable_dynamic_ops:
        relocation_tracker = RecentlyRelocatedTracker(
            n_splats=n_splats,
            cooldown_steps=config.dynamic_config.relocation_cooldown_steps,
            device=str(V_t.device),
        )
        if config.verbose:
            aprint(
                f"Dynamic ops enabled: Splat relocation with cooldown "
                f"({config.dynamic_config.relocation_cooldown_steps} steps)"
            )

    # Initial Z-order sort for memory locality (iteration 0)
    if config.sort_splats_enabled:
        sort_splats_by_morton_order(model, optimizer, relocation_tracker)
        if config.verbose:
            aprint("Z-order sort applied (initial)")

    # Eval frequency: full forward pass for convergence/metrics only every N iters.
    # The training forward+backward is always needed, but the second (eval) forward
    # pass is pure overhead for monitoring.  Skipping it on most iterations saves
    # ~30% of per-iteration wall-clock time.
    # Eval interval: skip the expensive eval forward pass on most iterations.
    # For short runs (<100 iters), check every iteration to not miss convergence.
    # For long runs (progressive fitting: 3000 iters), check every 25.
    _EVAL_INTERVAL = 25 if config.n_iters >= 100 else 1

    # Main optimization loop
    converged_early = False
    early_stopped = False
    actual_iters = 0
    current_max_abs_error = float("inf")
    current_rel_l2 = float("inf")
    pred_eval = None  # lazily computed
    # Best loss + best-state snapshot live on the device; read back only on
    # eval iterations (see _DeviceBestState). No per-iteration host sync.
    tracker = _DeviceBestState(V_t.device)
    current_metrics_t = tracker.metrics.clone()
    loss_curve: list[list[float]] = []
    relocation_curve: list[list[float]] = []
    # Determine scheduler type once (avoid repeated string checks)
    _is_plateau_scheduler = (
        scheduler is not None and "Plateau" in type(scheduler).__name__
    )
    plateau = (
        _PlateauFeeder(scheduler, _EVAL_INTERVAL) if _is_plateau_scheduler else None
    )
    for it in range(1, config.n_iters + 1):
        actual_iters = it

        # Forward pass (training — always needed)
        optimizer.zero_grad()
        pred = model()
        loss = loss_fn(pred)
        loss.backward()  # type: ignore[no-untyped-call]

        # Gradient clipping
        if config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        # --- Best state tracking on the device (no CPU sync) ---
        # BEFORE optimizer.step(): `loss` was computed from the CURRENT
        # parameters, so they are what the loss ranks. Snapshotting after the
        # step paired every loss with the NEXT iterate; on a warm start the
        # first (best) loss then restored the post-first-step parameters
        # (A549 refit: foreground PSNR 22.6 -> 14.5 dB).
        loss_detached = loss.detach()
        with torch.no_grad():
            centers, Ls, amps = model.current_params()
            tracker.update(
                it,
                loss_detached,
                {
                    "centers": centers,
                    "Ls": Ls,
                    "amps": amps,
                    # Snapshot the ever-relocated mask IN THIS SNAPSHOT'S ORDER
                    # (#2734). A later Morton sort reorders the live tracker, so
                    # the tracker's final mask would both misalign with and
                    # mislabel these returned (best-loss) splats. Captured in
                    # lockstep with the params, it stays correct whatever sorts
                    # follow.
                    "relocated": _relocated_snapshot(relocation_tracker),
                },
                current_metrics_t,
            )
        del centers, Ls, amps

        optimizer.step()

        # --- Periodic full evaluation (expensive — involves second forward pass) ---
        need_dynamic = (
            config.enable_dynamic_ops
            and it % config.dynamic_config.step_every == 0
            # Tail cutoff (#2734): stop relocating past the configured fraction
            # of the schedule. Default 1.0 -> always true (no-op).
            and it <= config.dynamic_config.relocation_stop_fraction * config.n_iters
        )
        need_eval = it % _EVAL_INTERVAL == 0 or it <= 5 or need_dynamic

        # Learning rate scheduling (buffered for the plateau scheduler; see
        # _PlateauFeeder for the only semantic difference: reduction timing)
        if plateau is not None:
            plateau.push(loss_detached, force_flush=need_eval)
        elif scheduler is not None:
            scheduler.step()

        if need_eval:
            # Sync best loss/iteration to CPU (only on eval iterations)
            best_loss, best_iteration = tracker.sync()
            iterations_since_improvement = it - best_iteration

            with torch.no_grad():
                pred_eval = model()
                current_max_abs_error, current_rel_l2 = _compute_eval_metrics(
                    pred_eval, V_t
                )
                last_max_abs_error = current_max_abs_error
                current_metrics_t = torch.tensor(
                    [current_max_abs_error, current_rel_l2], device=V_t.device
                )
                _append_bounded_sample(
                    loss_curve, [float(it), best_loss, current_rel_l2]
                )

                # Update best_state metrics if this is at or near the best iteration
                if best_iteration > 0 and it - best_iteration < _EVAL_INTERVAL:
                    tracker.metrics = current_metrics_t.clone()
                    best_max_abs_error = current_max_abs_error
                    best_rel_l2 = current_rel_l2

            # Convergence check (either criterion suffices)
            if current_max_abs_error < preprocessed_data.max_abs_error:
                converged_early = True
                if config.verbose:
                    aprint(f"✓ CONVERGENCE ACHIEVED at iteration {it}")
                    aprint(
                        f"  Max absolute error: {current_max_abs_error:.6f} < threshold: {preprocessed_data.max_abs_error:.6f}"
                    )
                break
            elif (
                preprocessed_data.rel_l2_target is not None
                and current_rel_l2 < preprocessed_data.rel_l2_target
            ):
                converged_early = True
                if config.verbose:
                    aprint(f"✓ CONVERGENCE ACHIEVED at iteration {it}")
                    aprint(
                        f"  Relative L2 error: {current_rel_l2:.6f} < target: {preprocessed_data.rel_l2_target:.6f}"
                    )
                break

            # Early stopping (patience-based, checked on eval iterations)
            rule_reason = stop_rule.should_stop(
                it, best_loss, iterations_since_improvement
            )
            if rule_reason is not None:
                early_stopped = True
                if config.verbose:
                    stop_rule.announce(rule_reason, it, iterations_since_improvement)
                    aprint(
                        f"  Best state from iteration {best_iteration}: "
                        f"loss={best_loss:.6f}  max_abs_error={best_max_abs_error:.6f}"
                    )
                break

            # Dynamic operations (splat relocation)
            if need_dynamic:
                apply_dynamic_operations(
                    model,
                    V_t,
                    pred_eval,
                    config.dynamic_config,
                    max_abs_error_threshold=preprocessed_data.max_abs_error,
                    optimizer=optimizer,
                    relocation_tracker=relocation_tracker,
                    verbose=config.dynamic_ops_verbose,
                )
                if relocation_tracker is not None:
                    relocation_tracker.advance_step()
                    _record_relocations(relocation_curve, relocation_tracker, it)

        # Movie frame recording (runs on its own schedule, outside eval gate)
        if (
            config.napari_movie
            and movie_frames is not None
            and it % config.movie_every == 0
        ):
            with torch.no_grad():
                _movie_pred = (
                    pred_eval if (need_eval and pred_eval is not None) else model()
                )
            _record_movie_frame(model, _movie_pred, V_t, movie_frames, config, it)

        # User-supplied per-iteration callback (e.g. validation-set scoring).
        # Fires alongside eval so pred_eval is fresh; cadence is the larger
        # of _EVAL_INTERVAL and config.iter_callback_every.
        if (
            config.iter_callback is not None
            and need_eval
            and pred_eval is not None
            and it % max(_EVAL_INTERVAL, config.iter_callback_every) == 0
        ):
            with torch.no_grad():
                callback_info = {
                    "loss": float(loss_detached.item()),
                    "best_loss": best_loss,
                    "max_abs_error": current_max_abs_error,
                    "rel_l2": current_rel_l2,
                    "n_splats": (
                        model.n_splats()
                        if hasattr(model, "n_splats")
                        else preprocessed_data.N
                    ),
                }
                config.iter_callback(it, pred_eval.detach(), callback_info)

        # Free eval prediction to reclaim GPU memory for next training iteration
        if need_eval and pred_eval is not None:
            del pred_eval
            pred_eval = None

        # Periodic Z-order sort for memory locality
        if config.sort_splats_enabled and it % config.sort_splats_interval == 0:
            sort_splats_by_morton_order(model, optimizer, relocation_tracker)

        # Logging (only on eval iterations when we have fresh metrics)
        N = model.n_splats() if hasattr(model, "n_splats") else preprocessed_data.N
        if (
            need_eval
            and config.verbose
            and (it % max(1, config.n_iters // 10) == 0 or it <= 5)
        ):
            aprint(
                f"[{it:4d}/{config.n_iters}] loss={best_loss:.5g}  "
                f"relL2={current_rel_l2:.4f}  maxAbsErr={current_max_abs_error:.5g}  N={N}"
            )

    _rank_final_iterate(
        tracker,
        model,
        loss_fn,
        relocation_tracker,
        actual_iters,
        converged_metrics=(
            torch.tensor([current_max_abs_error, current_rel_l2], device=V_t.device)
            if converged_early
            else None
        ),
        current_metrics=current_metrics_t,
    )

    end_time = time.time()
    best_state = tracker.as_best_state()
    best_loss, best_iteration = tracker.sync()
    best_from_final_step = best_state is not None and bool(
        tracker.from_final_step.cpu()
    )

    # Log termination reason
    if config.verbose:
        if converged_early:
            aprint("✓ Optimization terminated: CONVERGENCE ACHIEVED")
        elif early_stopped:
            aprint(f"⏹ Optimization terminated: EARLY STOPPING ({rule_reason} rule)")
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
            coverage_pct = (stats["unique_splats"] / max(1, current_n_splats)) * 100
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
        centers = best_state["centers"]
        Ls = best_state["Ls"]
        amps = best_state["amps"]

        # Restore the selected state into the model and recompute all reported
        # quality values from that exact state. Periodic eval metrics can be
        # stale because full evaluation is intentionally skipped on most
        # iterations, and the saved best state may not be the final live model.
        if not hasattr(model, "replace_with"):
            raise TypeError(
                "Optimization model must implement replace_with() to restore "
                "and score the selected best state."
            )
        with torch.no_grad():
            model.replace_with(centers, Ls, amps)
            pred_best = model()
            best_loss = float(loss_fn(pred_best).detach().item())
            best_max_abs_error, best_rel_l2 = _compute_eval_metrics(pred_best, V_t)
        del pred_best

        if config.verbose:
            if not best_from_final_step:
                aprint(
                    f"★ Restored best state from iteration {best_iteration} "
                    f"(loss={best_loss:.6f}  max_abs_error={best_max_abs_error:.6f})"
                )
            else:
                aprint("★ Best state is from final iteration")
    else:
        # Fallback to final state if no best state saved
        with torch.no_grad():
            centers, Ls, amps = model.current_params()
            pred_final = model()
            best_max_abs_error, best_rel_l2 = _compute_eval_metrics(pred_final, V_t)

    relocation_statistics = _completed_relocation_statistics(relocation_tracker)

    # Ever-relocated mask (#2734), ALIGNED WITH THE RETURNED SPLATS. The returned
    # centers/Ls/amps are the best-loss snapshot, so use the mask captured into
    # best_state at that iteration (correct whatever Morton sorts followed). Fall
    # back to the live tracker only when no snapshot was ever recorded (degenerate
    # no-improvement fit, where the returned params are the live model). None when
    # dynamic ops were disabled.
    relocated_mask = _returned_relocated_mask(best_state, relocation_tracker)

    return OptimizationResults(
        centers=centers,
        Ls=Ls,
        amps=amps,
        converged_early=converged_early,
        early_stopped=early_stopped,
        early_stop_reason=_stop_reason(converged_early, rule_reason),
        early_stop_rel_tol=config.early_stop_rel_tol,
        loss_curve=loss_curve,
        relocation_curve=relocation_curve,
        actual_iters=actual_iters,
        best_iteration=best_iteration,
        best_from_final_step=best_from_final_step,
        best_loss=best_loss,
        best_max_abs_error=best_max_abs_error,
        best_rel_l2=best_rel_l2,
        movie_frames=movie_frames,
        start_time=start_time,
        end_time=end_time,
        relocation_statistics=relocation_statistics,
        relocated_mask=relocated_mask,
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
        if (
            config.movie_max_frames is not None
            and len(movie_frames["reconstruction"]) >= config.movie_max_frames
        ):
            # Remove oldest frame (FIFO) — target is stored once, not per-frame
            for key in [
                "reconstruction",
                "residual",
                "splat_centers",
                "iterations",
            ]:
                movie_frames[key].pop(0)

        # Store frames as numpy arrays (detached from computation graph)
        pred_frame = pred.detach().cpu().numpy()
        residual_frame = torch.abs(V_t - pred.detach()).cpu().numpy()

        # Record current splat centers
        centers, _, _ = model.current_params()
        centers_frame = centers.detach().cpu().numpy()

        movie_frames["reconstruction"].append(pred_frame)
        movie_frames["residual"].append(residual_frame)
        movie_frames["splat_centers"].append(centers_frame)
        movie_frames["iterations"].append(iteration)
