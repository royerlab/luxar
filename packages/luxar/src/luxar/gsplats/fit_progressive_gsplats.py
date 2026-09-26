"""Progressive Gaussian splat fitting via iterative residual decomposition.

Fits splats in multiple passes, each targeting the residual (original minus
current approximation).  Each pass solves a simpler subproblem, and the
asymmetric loss ensures under-prediction so residuals are clean and positive.

The result is a single flattened ``GSplatData`` containing all splats from
all passes.  To produce a principled LOD ladder for streaming, pass the
result to :func:`luxar.gsplats.lod.make_additive_lod`; the supplementary
document ``additive_lod`` shows that fitting passes do *not* yield useful
LODs, and that a post-fit greedy / self-energy ordering is the right way
to build them.

Optimised per-pass configuration
---------------------------------
Pass 0 and residual passes (1+) use different loss, penalty, and optimizer
settings.  Empirically tuned across 3 diverse microscopy datasets.  Key
findings:

**Pass 0 (dense volume)**:
  Uses the single-pass optimized defaults (L1 loss, symmetric penalty,
  truncation 2.75, no gradient clipping).  The only override is
  ``asymmetric_penalty = min(caller, 3.0)`` which adds a mild under-prediction
  bias to produce clean positive residuals for subsequent passes.

**Passes 1+ (sparse residual)**:
1. **Poisson loss** — the single largest improvement (+2.4 dB on single-image,
   +5 dB total on benchmark).  Poisson deviance naturally weights errors
   relative to signal level, ideal for sparse, count-like residuals.
2. **Asymmetric penalty (10×)** — prevents permanent overshoot.  In progressive
   fitting, overshoot is *locked in* (clamped residuals hide it from subsequent
   passes).  This is the opposite of single-pass where symmetric (1.0) is best.
3. **Higher LR (0.03)** — small splats fitting fine detail converge faster
   with a larger learning rate (base pipeline default 0.01 is conservative).
4. **No eccentricity limit** — allows splats to adapt shape freely to
   irregular features (elongated nuclei, curved edges).
5. **No sigma_max_diag constraint** — residual splats size freely; the old
   coarse-to-fine cascade was removed as it prevented capturing broad
   residual patterns without measurable quality benefit.
6. **Peaks seeding** — intensity-weighted sampling from non-zero voxels.
   Every seed lands on actual signal; no seeds wasted on background.

Speed optimisations
-------------------
7. **Disable dynamic ops** (−10.6%): Splat relocation is unnecessary in
   progressive fitting — peaks seeding already places seeds at residual maxima.
8. **Adaptive iteration count** (−9.4%): Later passes fit progressively
   smaller residuals and converge faster.  Budget: 100%, 95%, 90%, ..., 80%.
9. **gc.collect() between passes** — ensures the previous pass's model and
   optimizer tensors are freed before the next pass allocates, preventing
   peak GPU memory from stacking both passes' allocations.
10. **Center LR boost (1.5×)** — center positions are the most critical
    parameters for PSNR; a moderate boost accelerates convergence.
11. **Per-parameter-group LR** (−4.4%): Amplitudes get 3× base LR.

Additional optimisations in fitting sub-modules:
  - ``fitting/optimization.py``: eval frequency (−21.5%), GPU sync elim (−1.9%)
  - ``fitting/losses.py``: Poisson dedup (−4.9%), torch.compile (−14.4%)
"""

from __future__ import annotations

import gc
import math
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
from arbol import aprint, asection

from luxar.gsplats.fitting.results import (
    lift_normalization_stats,
    lift_source_grid_stats,
    stamp_voxels_per_splat,
)
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS


def _pin_pass0_norm_range(
    pass_i: int,
    image_max: float,
    applied_floor: "float | None",
    pass_kwargs: dict[str, Any],
) -> None:
    """Put progressive pass 0 on the floor-subtracted normalization basis."""
    if pass_i != 0 or applied_floor is None:
        return
    from luxar.gsplats.fitting.preprocessing import NORM_RANGE_MIN_SPAN

    shifted_max = float(image_max) - float(applied_floor)
    pass_kwargs["norm_range"] = (
        (0.0, shifted_max)
        if np.isfinite(shifted_max) and shifted_max > NORM_RANGE_MIN_SPAN
        else None
    )


def _compute_psnr_chunked(
    rendered_gpu: torch.Tensor,
    original_np: np.ndarray,
    chunk_voxels: int = 50_000_000,
) -> float:
    """Compute PSNR without loading the full original volume onto GPU.

    Iterates over chunks of ~50M voxels (~200 MB), loading each chunk of
    ``original_np`` to GPU, computing partial sum-of-squared-errors, and
    accumulating as Python float (float64 precision).

    Peak GPU overhead beyond ``rendered_gpu``: one ~200 MB chunk.

    Parameters
    ----------
    rendered_gpu : torch.Tensor
        Full rendered volume on GPU.
    original_np : np.ndarray
        Original volume as numpy array (CPU).
    chunk_voxels : int
        Number of voxels per chunk.

    Returns
    -------
    float
        PSNR in dB.  Returns ``float('inf')`` when MSE is zero.
    """
    n = original_np.size
    flat_rendered = rendered_gpu.reshape(-1)
    flat_original = original_np.ravel()

    sse = 0.0
    for i in range(0, n, chunk_voxels):
        end = min(i + chunk_voxels, n)
        chunk_orig = torch.from_numpy(flat_original[i:end]).to(rendered_gpu.device)
        chunk_sse = (flat_rendered[i:end] - chunk_orig).square_().sum().item()
        sse += chunk_sse
        del chunk_orig

    mse = sse / n
    if mse == 0.0:
        return float("inf")

    data_range = float(original_np.max()) - float(original_np.min())
    if data_range == 0.0:
        return float("inf")

    return 10.0 * math.log10(data_range**2 / mse)


def _compute_foreground_psnr_chunked(
    rendered_gpu: torch.Tensor,
    original_np: np.ndarray,
    chunk_voxels: int = 50_000_000,
) -> tuple[float, float, float]:
    """Foreground PSNR, chunked like :func:`_compute_psnr_chunked`.

    The progressive fitter works on volumes too large to hold several copies of,
    so it cannot call :func:`~luxar.gsplats.metrics.compute_foreground_psnr`
    directly: that one masks the whole volume at once (a full-size mask plus two
    boolean-indexed copies), on top of the target and the render already
    resident. Chunking bounds those to ``chunk_voxels``. Same definition: error
    averaged over ``original > otsu`` only, ``data_range`` from the whole volume.

    ``rendered_gpu`` follows :func:`_compute_psnr_chunked`'s signature, but the
    fitter calls this one with its CPU-cached render (the GPU copy is already
    freed by then), so in practice the work happens on the host.

    Returns ``(psnr_db, threshold, foreground_fraction)``.
    """
    from .metrics import otsu_threshold

    flat_original = original_np.ravel()
    # A view when already float32, so this does not double peak RAM.
    threshold = otsu_threshold(torch.from_numpy(flat_original))

    n = flat_original.size
    flat_rendered = rendered_gpu.reshape(-1)
    sse = 0.0
    n_fg = 0
    for i in range(0, n, chunk_voxels):
        end = min(i + chunk_voxels, n)
        chunk_orig = torch.from_numpy(flat_original[i:end]).to(rendered_gpu.device)
        mask = chunk_orig > threshold
        count = int(mask.sum().item())
        if count:
            sse += (
                (flat_rendered[i:end][mask] - chunk_orig[mask]).square_().sum().item()
            )
            n_fg += count
        del chunk_orig, mask

    fraction = n_fg / max(n, 1)
    if n_fg == 0:
        return float("nan"), threshold, fraction

    mse = sse / n_fg
    if mse == 0.0:
        return float("inf"), threshold, fraction

    data_range = float(original_np.max()) - float(original_np.min())
    if data_range == 0.0:
        return float("inf"), threshold, fraction

    return 10.0 * math.log10(data_range**2 / mse), threshold, fraction


def _final_foreground_score(
    cached_rendered_np: Optional[np.ndarray],
    V_original: np.ndarray,
) -> tuple[float, float, float]:
    """Foreground score for the last completed pass, or ``nan`` if there was none.

    Taken once at the end rather than per pass: only the final value is
    stamped, and the patience check steers on the global PSNR. The caller
    passes the render ``prev_psnr`` was measured from, so the two figures agree
    on which pass they describe.
    """
    if cached_rendered_np is None:  # no pass completed — nothing was rendered
        return float("nan"), float("nan"), 0.0
    return _compute_foreground_psnr_chunked(
        torch.from_numpy(cached_rendered_np), V_original
    )


def fit_progressive_gaussian_splats(
    V: np.ndarray,
    max_splats: int = 50000,
    max_splats_per_pass: int = 5000,
    iters_per_pass: int = 1000,
    psnr_patience: float = 0.5,
    max_passes: Optional[int] = None,
    asymmetric_penalty: Optional[
        float
    ] = 10.0,  # Intentionally higher than single-pass (1.0)
    enable_dynamic_ops: bool = False,
    cull_retention: float | None = 0.98,
    on_pass_complete: Optional[Callable[[int, AdditiveSubLOD, float], None]] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    truncate: float = DEFAULT_TRUNCATION_RADIUS,
    residual_pass_min_iters: int = 500,
    **kwargs: Any,
) -> GSplatData:
    """Fit Gaussian splats progressively via iterative residual decomposition.

    Each pass fits up to ``max_splats_per_pass`` splats to the current residual
    (``clamp(V - render(accumulated), min=0)``).  Passes continue until
    ``max_splats`` is reached, PSNR improvement drops below ``psnr_patience``
    dB, or the residual becomes negligible.

    Parameters
    ----------
    V : np.ndarray
        Input volume to approximate (any dimensionality).
    max_splats : int
        Maximum total number of splats across all passes.
    max_splats_per_pass : int
        Maximum number of splats to fit per pass.  The actual count may be
        lower due to post-fit culling and adaptive reduction when previous
        passes show high culling rates.
    iters_per_pass : int
        Optimization iterations per pass.
    psnr_patience : float
        Stop if ΔPSNR between consecutive passes < this value (in dB).
    max_passes : int, optional
        Maximum number of passes.  If None, continues until ``max_splats``
        is reached or PSNR patience triggers.
    asymmetric_penalty : float, optional
        Asymmetric loss penalty factor (default 10.0).
    enable_dynamic_ops : bool, default=False
        Whether to enable dynamic splat relocation within each pass. Off by
        default for progressive fitting: each pass seeds directly at the
        residual peaks, so relocation shows no measured quality benefit. Set
        to ``True`` to opt in.
    cull_retention : float or None, default=0.98
        Post-fit cumulative culling on the final accumulated result.  Keeps the
        top splats that account for this fraction of total amplitude (0--1).
        Set to ``None`` to disable.
    on_pass_complete : callable, optional
        Callback invoked after each pass:
        ``on_pass_complete(pass_index, lod_data, cumulative_psnr)``.
    device : str, optional
        Device for fitting and rendering (auto-detected if None).
    verbose : bool
        Whether to print progress information.
    truncate : float
        Truncation radius in standard deviations for rendering.
    residual_pass_min_iters : int, default=500
        Minimum optimizer iterations for *residual* passes (pass 1+).  Pass 0
        always honours ``iters_per_pass`` directly.  The default of 500 is
        the historical floor that protects fit quality when callers supply
        a small ``iters_per_pass`` (the decayed value can otherwise drop
        below what residual passes need to converge).  Lower this only for
        tests that need short runtime — production callers should leave it
        at the default.
    **kwargs
        Additional keyword arguments passed through to ``fit_gaussian_splats``.
        ``norm_range`` (a whole-volume intensity scale, as tiled fitting
        supplies) applies to pass 0 only: passes 1+ fit a residual that is by
        construction a small fraction of that range, and normalizing it against
        the range would put it under the absolute convergence tolerance and end
        the pass immediately. Residual passes keep their own per-pass scale.

    Returns
    -------
    GSplatData
        Single-LOD result containing all splats from all passes.  The
        per-pass intermediate LODs are surfaced through ``on_pass_complete``
        and the ``stats`` dict (``stats['n_passes']``,
        ``stats['pass_psnrs']``); to build a streamable LOD ladder, hand
        the result to :func:`luxar.gsplats.lod.make_additive_lod`.

    Notes
    -----
    **GPU utilization**: Each pass fits only ``max_splats_per_pass`` splats,
    which may under-saturate the GPU compared to a single large fit.  When
    using tiled fitting on a cluster (``luxar gsplat batch-fit submit``), combine
    ``--progressive`` with ``--parallel`` to run multiple tiles concurrently
    on the same GPU and fill the utilization gap.
    """
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.fitting.preprocessing import (
        _resolve_applied_norm_bounds_with_strategy,
    )
    from luxar.gsplats.fitting.validation import _validate_floor
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    if max_passes is not None and max_passes < 1:
        raise ValueError(f"max_passes must be >= 1, got {max_passes}")

    # Pop params that progressive handles itself to avoid "multiple values" conflicts
    kwargs.pop("n_iters", None)  # progressive uses iters_per_pass instead
    kwargs.pop("seeds", None)  # progressive computes seeds_this_pass per pass
    kwargs.pop("seed_method", None)  # progressive sets auto/peaks per pass
    kwargs.pop("cull_retention", None)  # per-pass culling disabled; final cull at end
    # Background floor / DC-offset suppression: subtract ONCE from the volume up
    # front (like calibrate()), then run EVERY pass with floor='none'. The
    # residual chain (target = clip(V - render, 0)) is built against this
    # subtracted volume, so the pedestal is never reintroduced; output amplitudes
    # stay background-relative, matching the single-pass fitter. Applying floor
    # only inside pass 0 would be WRONG — the residual would be built against the
    # raw volume and reinstate the pedestal for the floor='none' residual passes.
    floor_spec = kwargs.pop("floor", "auto")
    # The progressive path never routes through prepare_fit_config (per-pass fits
    # run with floor='none'), so validate the user's spec here — the same guard
    # every other entry point gets — before resolving it below.
    _validate_floor(floor_spec)
    # Force voxel-space output for internal passes: render_to_volume_tensor
    # expects voxel-space centers for correct residual computation.
    # We capture voxel_size/output_space to apply to the final result.
    caller_output_space: str = kwargs.pop("output_space", "real")
    caller_voxel_size_raw = kwargs.pop("voxel_size", None)
    # Normalize voxel_size to ndarray or None
    caller_voxel_size: Optional[np.ndarray] = None
    if caller_voxel_size_raw is not None:
        if isinstance(caller_voxel_size_raw, (int, float)):
            caller_voxel_size = np.full(V.ndim, float(caller_voxel_size_raw))
        else:
            caller_voxel_size = np.asarray(caller_voxel_size_raw, dtype=np.float64)

    start_time = time.time()
    V_original = V.astype(np.float32)
    _, image_max, applied_floor, floor_strategy = (
        _resolve_applied_norm_bounds_with_strategy(
            V_original,
            float(kwargs.get("norm_percentile", 0.0)),
            verbose,
            floor_spec,
            kwargs.get("norm_range"),
        )
    )
    if applied_floor is not None:
        V_original = np.clip(V_original - applied_floor, 0.0, None).astype(np.float32)

    accumulated_lods: list[AdditiveSubLOD] = []
    prev_psnr = 0.0
    stop_reason = "max_splats"
    # Cache the rendered volume (CPU numpy) from PSNR computation to avoid
    # re-rendering at the start of the next pass (same accumulated splats).
    # Stored on CPU to free GPU memory for the next pass's fitting.
    cached_rendered_np: Optional[np.ndarray] = None

    if verbose:
        with asection("Progressive Gaussian splat fitting"):
            aprint(f"Volume shape: {V.shape}")
            aprint(f"Max splats: {max_splats:,}, splats/pass: {max_splats_per_pass:,}")
            aprint(f"Iters/pass: {iters_per_pass}")
            aprint(f"PSNR patience: {psnr_patience} dB")
            aprint(
                f"Note: each pass fits {max_splats_per_pass:,} splats — "
                f"GPU may be under-saturated. For tiled batch jobs, "
                f"use --parallel to run multiple tiles per GPU."
            )

    pass_i = 0
    while True:
        pass_start = time.time()

        # --- Check stopping conditions ---
        total_so_far = sum(lod.n_splats for lod in accumulated_lods)
        if total_so_far >= max_splats:
            stop_reason = "max_splats"
            break
        if max_passes is not None and pass_i >= max_passes:
            stop_reason = "max_passes"
            break

        # --- Compute target for this pass ---
        if pass_i == 0:
            target = V_original
        else:
            # Reuse the cached render (CPU numpy) from the previous pass's
            # PSNR computation.  Residual is computed entirely on CPU to
            # keep GPU memory free for the upcoming fit.
            assert cached_rendered_np is not None  # guaranteed after pass 0
            target = np.clip(V_original - cached_rendered_np, 0, None).astype(
                np.float32
            )

            # Check if residual is negligible
            residual_max = float(target.max())
            if residual_max < 1e-6:
                if verbose:
                    aprint(
                        f"Pass {pass_i}: residual negligible ({residual_max:.2e}), stopping"
                    )
                stop_reason = "residual_negligible"
                break

            # NOTE: Residual thresholding disabled — with denoised data,
            # the entire residual is legitimate signal. Peaks seeding
            # already concentrates seeds on high-intensity regions.

        # --- Determine seed count (adapt based on previous culling rate) ---
        effective_max_splats_per_pass = max_splats_per_pass
        if len(accumulated_lods) > 0:
            last_lod = accumulated_lods[-1]
            requested = last_lod.stats.get("seeds_requested", 0)
            survived = last_lod.stats.get("splats_after_culling", 0)
            if requested > 0 and survived < requested:
                survival_rate = survived / requested
                if survival_rate < 0.5:
                    # Significant culling — reduce seeds to avoid waste
                    effective_max_splats_per_pass = max(
                        10, int(max_splats_per_pass * max(survival_rate * 1.2, 0.3))
                    )
                    if verbose:
                        aprint(
                            f"\nAdapting: previous pass culled "
                            f"{100 * (1 - survival_rate):.0f}% of splats, "
                            f"reducing seeds {max_splats_per_pass} → {effective_max_splats_per_pass}"
                        )

        seeds_this_pass = min(effective_max_splats_per_pass, max_splats - total_so_far)

        # --- Fit splats to target ---
        # Pass 0: use default seeding (auto). Passes 1+: use peaks seeding
        # to place seeds directly at local maxima of the sparse residual.
        pass_seed_method = "auto" if pass_i == 0 else "peaks"

        if verbose:
            aprint(f"\n{'=' * 60}")
            aprint(
                f"Pass {pass_i}: fitting {seeds_this_pass} splats to "
                f"{'volume' if pass_i == 0 else 'residual'} "
                f"(seed_method={pass_seed_method})"
            )

        # --- Per-pass configuration (see module docstring for rationale) ---

        # Asymmetric penalty: mild for pass 0 (capture more signal), full for
        # residual passes (prevent permanent overshoot in the residual chain).
        if asymmetric_penalty is not None:
            if pass_i == 0:
                pass_asymmetric_penalty: Optional[float] = min(asymmetric_penalty, 3.0)
            else:
                pass_asymmetric_penalty = asymmetric_penalty
        else:
            pass_asymmetric_penalty = None

        # Build per-pass kwargs, overriding caller defaults where needed.
        pass_kwargs = dict(kwargs)

        # No eccentricity limit: let splats adapt shape to irregular features.
        pass_kwargs["max_eccentricity"] = None

        # Floor is already subtracted from V_original up front (see above), so
        # every per-pass fit must run with floor='none' — re-estimating a floor
        # on the (already background-relative) full volume or its residuals would
        # wrongly eat signal.
        pass_kwargs["floor"] = "none"

        # When the effective baseline was subtracted up front, pin pass 0 to the
        # same zero-based bounds resolved by the single-pass fitter.
        _pin_pass0_norm_range(pass_i, image_max, applied_floor, pass_kwargs)

        # A supplied whole-volume intensity scale describes the VOLUME, not the
        # residual chain built from it. Pass 0 shares it (that is the point);
        # passes 1+ normalize their residual by its own extent, as they always
        # have — against the whole-volume range a residual worth several passes
        # sits below the absolute convergence tolerance and the pass ends at
        # its first evaluation.
        if pass_i > 0:
            pass_kwargs["norm_range"] = None
            # Residual-pass overrides (see module docstring for rationale):
            pass_kwargs["loss_type"] = "poisson"  # natural for sparse residuals
            pass_kwargs["lr"] = 0.03  # fine-detail splats converge faster

        # Relocation is OFF BY DEFAULT for progressive passes: seeds are already
        # placed at residual peaks, and a diverse benchmark showed no quality
        # benefit from relocation (quality iteration 30) while it adds
        # per-iteration overhead.  A caller may opt in via
        # ``enable_dynamic_ops=True``.
        pass_enable_dynamic = enable_dynamic_ops

        # Adaptive iteration count: later passes fit progressively smaller
        # residuals and converge faster. Scale iterations with pass index.
        # Pass 0: full iters.  Pass 1+: decreasing from 100% to 60%, floored
        # by residual_pass_min_iters (default 500 — see param docstring).
        if pass_i == 0:
            pass_iters = iters_per_pass
        else:
            # Gentle decay: pass 1=95%, pass 2=90%, ..., min 80%
            decay = max(0.8, 1.0 - 0.05 * pass_i)
            pass_iters = max(residual_pass_min_iters, int(iters_per_pass * decay))

        result = fit_gaussian_splats(
            target,
            seeds=seeds_this_pass,
            n_iters=pass_iters,
            asymmetric_penalty=pass_asymmetric_penalty,
            enable_dynamic_ops=pass_enable_dynamic,
            cull_retention=None,  # Disable per-pass; final cull at end of progressive
            seed_method=pass_seed_method,
            device=device,
            verbose=verbose,
            truncate=truncate,
            **pass_kwargs,
        )

        # --- Create LOD from result ---
        lod = AdditiveSubLOD(
            centers=result.centers,
            amplitudes=result.amplitudes,
            cholesky_factors=result.cholesky_factors,
            colors=result.colors,
            stats=dict(result.stats),
            truncation_radius=result.truncation_radius,
        )
        accumulated_lods.append(lod)

        # --- Inter-pass memory cleanup ---
        # gc.collect() + empty_cache() ensure the previous pass's model,
        # optimizer, and gradient tensors are freed before the next pass
        # allocates. Without this, peak GPU memory stacks both passes.
        from luxar.gsplats.models.gsplats.rendering_core import clear_grid_cache

        clear_grid_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Compute global PSNR (and cache render for next pass's residual) ---
        # Memory-efficient: render on GPU (CUDA backend is tiled and uses
        # minimal intermediates), compute PSNR in chunks (only ~200 MB of
        # V_original on GPU at a time), then move render to CPU.
        accumulated_data = GSplatData.from_additive_sublods(accumulated_lods)
        with torch.no_grad():
            rendered_gpu = render_to_volume_tensor(
                accumulated_data,
                shape=V.shape,
                device=device,
                truncate=truncate,
            )
            current_psnr = _compute_psnr_chunked(rendered_gpu, V_original)
            cached_rendered_np = rendered_gpu.cpu().numpy()
            del rendered_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        delta_psnr = current_psnr - prev_psnr
        pass_time = time.time() - pass_start

        # Store per-LOD metadata (dict is mutable despite frozen dataclass)
        lod.stats["cumulative_psnr_db"] = current_psnr
        lod.stats["delta_psnr_db"] = delta_psnr
        lod.stats["pass_index"] = pass_i
        lod.stats["pass_time_seconds"] = pass_time
        lod.stats["seeds_requested"] = seeds_this_pass
        lod.stats["splats_after_culling"] = lod.n_splats

        if verbose:
            total_splats = sum(lo.n_splats for lo in accumulated_lods)
            aprint(
                f"Pass {pass_i} complete: {lod.n_splats} splats, "
                f"PSNR={current_psnr:.2f} dB (Δ={delta_psnr:+.2f} dB), "
                f"total={total_splats:,} splats, {pass_time:.1f}s"
            )

        # --- Callback ---
        if on_pass_complete is not None:
            on_pass_complete(pass_i, lod, current_psnr)

        # --- PSNR patience check ---
        # Update prev_psnr BEFORE the break so overall_stats["psnr_db"]
        # reflects the last completed pass (not the one before it).
        prev_psnr = current_psnr

        if pass_i > 0 and delta_psnr < psnr_patience:
            if verbose:
                aprint(
                    f"\nPSNR patience: ΔPSNR={delta_psnr:.3f} dB < {psnr_patience} dB, stopping"
                )
            stop_reason = "psnr_patience"
            break
        pass_i += 1

    # Score the foreground before the last pass's render is freed.
    fg_psnr, fg_threshold, fg_fraction = _final_foreground_score(
        cached_rendered_np, V_original
    )

    # Free cached render (CPU numpy) from the last pass
    del cached_rendered_np
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Build result ---
    # Deferred: luxar.gsplats.io imports GSplatData from this package's
    # __init__, which is still executing when this module is first imported.
    from luxar.gsplats.io.save_gsplats import NORMALIZATION_STATS_KEYS

    total_time = time.time() - start_time
    total_splats = sum(lod.n_splats for lod in accumulated_lods)

    overall_stats: dict[str, Any] = {
        "fitter_name": "progressive",
        "n_passes": len(accumulated_lods),
        "n_splats": total_splats,
        "time_seconds": total_time,
        "psnr_db": prev_psnr,
        "foreground_psnr_db": fg_psnr,
        "foreground_threshold": fg_threshold,
        "foreground_fraction": fg_fraction,
        "stop_reason": stop_reason,
        "max_splats": max_splats,
        "max_splats_per_pass": max_splats_per_pass,
        "iters_per_pass": iters_per_pass,
        "psnr_patience": psnr_patience,
        "pass_psnrs": [
            float(lod.stats.get("cumulative_psnr_db", float("nan")))
            for lod in accumulated_lods
        ],
        "pass_splats": [int(lod.n_splats) for lod in accumulated_lods],
        # Full per-pass stats (one dict per pass) -- the per-pass LOD
        # intermediates are not preserved on the flattened return value,
        # so any caller that wants per-pass detail must read this list.
        #
        # MINUS the normalization block (#1175). This dict is not in
        # `_FITTING_INFO_KEYS`, so `split_fitting_info` routes it verbatim into
        # `pipeline/` — right beside the corrected top-level block. Every pass
        # ran with floor="none" on the ALREADY-subtracted array, so its own
        # `floor: null, image_min: 0.0` describes the fitter's input, not the
        # artifact, and shipping both put two contradicting answers in one group.
        "pass_stats": [
            {k: v for k, v in lod.stats.items() if k not in NORMALIZATION_STATS_KEYS}
            for lod in accumulated_lods
        ],
    }

    if verbose:
        aprint(f"\n{'=' * 60}")
        aprint(f"Progressive fitting complete: {len(accumulated_lods)} passes")
        aprint(f"Total splats: {total_splats:,}, PSNR: {prev_psnr:.2f} dB")
        aprint(f"Stop reason: {stop_reason}, total time: {total_time:.1f}s")

        # Culling efficiency hint
        total_requested = sum(
            lod.stats.get("seeds_requested", lod.n_splats) for lod in accumulated_lods
        )
        if total_requested > 0:
            culled_pct = 100.0 * (1.0 - total_splats / total_requested)
            if culled_pct > 30:
                aprint(
                    f"\nHint: {culled_pct:.0f}% of fitted splats were culled as negligible. "
                    f"Consider reducing max_splats_per_pass (e.g., {max_splats_per_pass // 2}) "
                    f"to avoid wasting compute on splats that get culled."
                )

    # Lift the source-grid stamps out of pass 1 and onto the whole result.
    #
    # Every pass sees the SAME volume (later ones fit its residual), so pass 1's
    # record of that volume describes the fit as a whole. Left where they are
    # they stay buried in `pass_stats`, never reach `_FITTING_INFO_KEYS`, and the
    # dataset ends up unable to say what it is a representation of — which is
    # how the progressive demos came to have no compression figure at all.
    #
    # `voxels_per_splat` is deliberately NOT copied: it is a ratio against one
    # pass's splat count, and the merged result has all of them — it is stamped
    # below instead, after the cull.
    lift_source_grid_stats(overall_stats, accumulated_lods)

    # Same treatment for the normalization block (#1175): the pedestal was
    # removed from V_original up front and every pass then ran with
    # floor="none", so without this a progressive fit ships no record at all of
    # the background it subtracted.
    lift_normalization_stats(
        overall_stats, accumulated_lods, applied_floor, floor_strategy=floor_strategy
    )
    final_result = GSplatData.from_additive_sublods(
        accumulated_lods, stats=overall_stats
    )

    # Convert to physical coordinates if the caller requested it
    if caller_output_space == "real" and caller_voxel_size is not None:
        vs = caller_voxel_size
        d = V.ndim
        # Scale centers and Cholesky factors for every LOD
        converted_lods: list[AdditiveSubLOD] = []
        for lod in final_result.additive_sublods:
            new_centers = lod.centers * vs  # (N, d) * (d,)
            tril_scales = np.concatenate([[vs[i]] * (i + 1) for i in range(d)])
            new_chol = lod.cholesky_factors * tril_scales  # (N, tril) * (tril,)
            converted_lods.append(
                AdditiveSubLOD(
                    centers=new_centers.astype(np.float32),
                    amplitudes=lod.amplitudes,
                    cholesky_factors=new_chol.astype(np.float32),
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            )
        final_result = GSplatData.from_additive_sublods(
            converted_lods, stats=overall_stats
        )
        if verbose:
            aprint(
                f"Converted output to physical coordinates (voxel_size={vs.tolist()})"
            )

    # Post-fit cumulative culling on the full accumulated result
    if (
        cull_retention is not None
        and 0 < cull_retention < 1.0
        and final_result.n_splats > 0
    ):
        from luxar.gsplats._data.filtering import (
            measured_stats_snapshot,
            restore_measured_stats,
        )

        n_before = final_result.n_splats
        # The per-pass and overall PSNRs were measured above, on the pre-trim
        # splats. `cull` drops inherited scores (#1600), but this trim is the
        # last step of the FIT — carried across rather than re-rendered, and
        # rather than shipping a progressive fit with no ladder scores at all.
        measured = measured_stats_snapshot(final_result)
        final_result = final_result.cull(method="cumulative", retention=cull_retention)
        restore_measured_stats(final_result, measured)
        n_removed = n_before - final_result.n_splats
        if verbose:
            aprint(
                f"Post-fit culling (cumulative, retention={cull_retention:.0%}): "
                f"{n_before} -> {final_result.n_splats} splats "
                f"(removed {n_removed}, {100.0 * n_removed / n_before:.1f}%)"
            )

    # Density is quoted against the splats actually DELIVERED, so it is stamped
    # here rather than beside the other source-grid stamps above: the post-fit
    # cull runs in between, and the pre-cull count would overstate how much of
    # the volume each surviving splat stands for.
    stamp_voxels_per_splat(final_result.stats, final_result.n_splats)

    # Collapse per-pass LODs into a single flattened LOD.  The pass-by-pass
    # accumulation is an internal implementation detail; callers that want
    # a streaming LOD ladder should run luxar.gsplats.lod.make_additive_lod
    # on the returned data.
    return final_result.flattened()
