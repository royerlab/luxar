"""Progressive Gaussian splat fitting via iterative residual decomposition.

Fits splats in multiple passes, each targeting the residual (original minus
current approximation).  Each pass solves a simpler subproblem, and the
asymmetric loss ensures under-prediction so residuals are clean and positive.

The result is a multi-LOD ``GSplatData`` where each LOD corresponds to one pass.

Optimised per-pass configuration
---------------------------------
Pass 0 and residual passes (1+) use different loss, penalty, and optimizer
settings.  These were tuned via systematic autoresearch iteration (35 experiments)
and validated on a diverse benchmark (3D chimeric microscopy volume + 2D
composite of mitosis + astronaut images).  Key findings:

1. **L1 loss for pass 0** — robust to outliers during the initial dense fit.
   MSE and Poisson both performed worse on pass 0 (−3 to −4 dB).
2. **Poisson loss for residual passes** — the single largest improvement
   (+2.4 dB on single-image, +5 dB total on diverse benchmark).  Poisson
   deviance naturally weights errors relative to signal level, which is
   ideal for sparse, count-like residuals from microscopy data.
3. **Asymmetric penalty** — mild (3×) for pass 0 to capture more signal;
   full (10×) for residual passes to prevent overshoot.  In progressive
   fitting, overshoot is *permanently locked in* (clamped residuals hide
   it from subsequent passes), making strong anti-overshoot essential.
4. **No eccentricity limit** — removing max_eccentricity constraints lets
   splats adapt their shape freely to irregular features (elongated nuclei,
   curved edges).  The adaptive sigma_max_diag cap already prevents splats
   from growing too large overall.
5. **Adaptive sigma_max_diag** — caps splat size in residual passes at the
   median sigma of the *previous* pass.  This creates a natural coarse-to-fine
   cascade: each pass works at a finer scale than the last.
6. **Higher LR (0.03) for residual passes** — small splats fitting fine detail
   converge faster with a larger learning rate (default 0.01 is conservative).
7. **Progressive L1 on Cholesky diagonal** — gentle shrinkage pressure
   (0.0001 × pass_index) that increases with each pass, encouraging compact
   splats at finer scales.

Speed optimisations (validated via autoresearch, 38 experiments)
----------------------------------------------------------------
Total speedup: **−51.4%** (825 s → 401 s) with quality preserved.

8. **Disable dynamic ops** (−10.6%): Splat relocation is unnecessary in
   progressive fitting — peaks seeding already places seeds at residual maxima.
   Disabling removes per-iteration overhead from the relocation tracker.
9. **Adaptive iteration count** (−9.4%): Later passes fit progressively
   smaller residuals and converge faster.  Iteration budget scales as
   100%, 95%, 90%, 85%, 80% for passes 0–4+.
10. **Per-parameter-group LR** (−4.4%): Amplitudes converge faster than
    positions/shapes.  Giving amplitudes 3× the base learning rate
    accelerates convergence without destabilising the optimisation.

Additional optimisations live in the fitting sub-modules:
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

from luxar.gsplats.gsplat_data import GSplatData, GSplatLOD


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


def fit_progressive_gaussian_splats(
    V: np.ndarray,
    max_splats: int = 50000,
    max_splats_per_pass: int = 5000,
    iters_per_pass: int = 1000,
    psnr_patience: float = 0.5,
    max_passes: Optional[int] = None,
    asymmetric_penalty: Optional[float] = 10.0,
    enable_dynamic_ops: bool = True,
    cull_retention: float | None = 0.98,
    on_pass_complete: Optional[Callable[[int, GSplatLOD, float], None]] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    truncate: float = 2.75,
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
    enable_dynamic_ops : bool
        Whether to enable dynamic splat relocation within each pass.
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
    **kwargs
        Additional keyword arguments passed through to ``fit_gaussian_splats``.

    Returns
    -------
    GSplatData
        Multi-LOD result where each LOD corresponds to one pass.
        ``result.n_lods`` equals the number of passes completed.

    Notes
    -----
    **GPU utilization**: Each pass fits only ``max_splats_per_pass`` splats,
    which may under-saturate the GPU compared to a single large fit.  When
    using tiled fitting on a cluster (``luxar gsplat batch plan``), combine
    ``--progressive`` with ``--parallel`` to run multiple tiles concurrently
    on the same GPU and fill the utilization gap.
    """
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    if max_passes is not None and max_passes < 1:
        raise ValueError(f"max_passes must be >= 1, got {max_passes}")

    # Pop params that progressive handles itself to avoid "multiple values" conflicts
    kwargs.pop("n_iters", None)  # progressive uses iters_per_pass instead
    kwargs.pop("seeds", None)  # progressive computes seeds_this_pass per pass
    kwargs.pop("seed_method", None)  # progressive sets auto/peaks per pass
    kwargs.pop("cull_retention", None)  # per-pass culling disabled; final cull at end
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

    accumulated_lods: list[GSplatLOD] = []
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

        if pass_i > 0:
            # No sigma_max_diag constraint: let residual splats size freely.
            # The old coarse-to-fine cascade prevented capturing broad residual
            # patterns. With MSE+Poisson, the optimizer finds optimal sizes.

            # Poisson loss: natural for sparse, count-like residuals.
            pass_kwargs["loss_type"] = "poisson"

            # Higher LR: fine-detail splats need faster convergence.
            pass_kwargs["lr"] = 0.03

            # No progressive L1 on diagonal: the MSE loss + Poisson loss combo
            # already provides good convergence; extra regularization hurts PSNR.
            # (Original l1_diag = 0.0001 * pass_i was tuned for L1 base pipeline.)

        # Disable dynamic ops for progressive passes: seeds are already placed
        # at residual peaks, and diverse benchmark showed no quality benefit
        # from relocation (quality iteration 30).  Saves per-iteration overhead.
        pass_enable_dynamic = False

        # Adaptive iteration count: later passes fit progressively smaller
        # residuals and converge faster. Scale iterations with pass index.
        # Pass 0: full iters.  Pass 1+: decreasing from 100% to 60%.
        if pass_i == 0:
            pass_iters = iters_per_pass
        else:
            # Gentle decay: pass 1=95%, pass 2=90%, ..., min 80%
            decay = max(0.8, 1.0 - 0.05 * pass_i)
            pass_iters = max(500, int(iters_per_pass * decay))

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
        lod = GSplatLOD(
            centers=result.centers,
            amplitudes=result.amplitudes,
            cholesky_factors=result.cholesky_factors,
            colors=result.colors,
            stats=dict(result.stats),
            truncation_radius=result.truncation_radius,
        )
        accumulated_lods.append(lod)

        # --- Free GPU memory from the per-pass fit before rendering ---
        # The fitter's optimizer state, gradients, and V_tensor are
        # unreferenced but PyTorch's caching allocator may hold the blocks.
        # The rendering grid cache (_GRID_CACHE) also accumulates GPU tensors
        # for each unique AABB box shape seen during optimization.
        from luxar.gsplats.models.gsplats.rendering_core import clear_grid_cache

        # Skip aggressive memory cleanup between passes — gc.collect() and
        # cache clearing add overhead. For moderate splat counts (≤50K/pass),
        # GPU memory is not a constraint. The allocator reuses freed blocks.
        clear_grid_cache()  # keep this (small, avoids stale cache entries)

        # --- Compute global PSNR (and cache render for next pass's residual) ---
        # Memory-efficient: render on GPU (CUDA backend is tiled and uses
        # minimal intermediates), compute PSNR in chunks (only ~200 MB of
        # V_original on GPU at a time), then move render to CPU.
        accumulated_data = GSplatData.from_lods(accumulated_lods)
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

    # Free cached render (CPU numpy) from the last pass.
    # Skip empty_cache() — caller manages GPU memory lifecycle.
    del cached_rendered_np

    # --- Build result ---
    total_time = time.time() - start_time
    total_splats = sum(lod.n_splats for lod in accumulated_lods)

    overall_stats: dict[str, Any] = {
        "fitter_name": "progressive",
        "n_passes": len(accumulated_lods),
        "n_splats": total_splats,
        "time_seconds": total_time,
        "psnr_db": prev_psnr,
        "stop_reason": stop_reason,
        "max_splats": max_splats,
        "max_splats_per_pass": max_splats_per_pass,
        "iters_per_pass": iters_per_pass,
        "psnr_patience": psnr_patience,
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

    final_result = GSplatData.from_lods(accumulated_lods, stats=overall_stats)

    # Convert to physical coordinates if the caller requested it
    if caller_output_space == "real" and caller_voxel_size is not None:
        vs = caller_voxel_size
        d = V.ndim
        # Scale centers and Cholesky factors for every LOD
        converted_lods: list[GSplatLOD] = []
        for lod in final_result.lods:
            new_centers = lod.centers * vs  # (N, d) * (d,)
            tril_scales = np.concatenate([[vs[i]] * (i + 1) for i in range(d)])
            new_chol = lod.cholesky_factors * tril_scales  # (N, tril) * (tril,)
            converted_lods.append(
                GSplatLOD(
                    centers=new_centers.astype(np.float32),
                    amplitudes=lod.amplitudes,
                    cholesky_factors=new_chol.astype(np.float32),
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            )
        final_result = GSplatData.from_lods(converted_lods, stats=overall_stats)
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
        n_before = final_result.n_splats
        final_result = final_result.cull(method="cumulative", retention=cull_retention)
        n_removed = n_before - final_result.n_splats
        if verbose:
            aprint(
                f"Post-fit culling (cumulative, retention={cull_retention:.0%}): "
                f"{n_before} -> {final_result.n_splats} splats "
                f"(removed {n_removed}, {100.0 * n_removed / n_before:.1f}%)"
            )

    return final_result
