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
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np
import torch
from arbol import aprint, asection

from luxar.gsplats.gsplat_data import GSplatData, GSplatLOD


def fit_progressive_gaussian_splats(
    V: np.ndarray,
    max_splats: int = 50000,
    max_splats_per_pass: int = 5000,
    iters_per_pass: int = 1000,
    psnr_patience: float = 0.5,
    max_passes: Optional[int] = None,
    asymmetric_penalty: Optional[float] = 10.0,
    enable_dynamic_ops: bool = True,
    cull_ratio: float = 0.0,
    on_pass_complete: Optional[Callable[[int, GSplatLOD, float], None]] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    truncate: float = 3.0,
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
    cull_ratio : float
        Post-fit culling ratio per pass (default 0.0, disabled).  Splats with
        amplitude below ``cull_ratio * max_abs_error`` are removed.  The
        residual-based approach of progressive fitting already prevents waste,
        so culling is disabled by default.  Set to 1.0 for aggressive culling.
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
    from luxar.gsplats.metrics import compute_quality_metrics
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    if max_passes is not None and max_passes < 1:
        raise ValueError(f"max_passes must be >= 1, got {max_passes}")

    # Pop params that progressive handles itself to avoid "multiple values" conflicts
    kwargs.pop("n_iters", None)  # progressive uses iters_per_pass instead
    kwargs.pop("seeds", None)  # progressive computes seeds_this_pass per pass
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
    # Cache the rendered tensor from PSNR computation to avoid re-rendering
    # at the start of the next pass (same accumulated splats).
    cached_rendered: Optional[torch.Tensor] = None

    if verbose:
        with asection("Progressive Gaussian splat fitting"):
            aprint(f"Volume shape: {V.shape}")
            aprint(f"Max splats: {max_splats:,}, splats/pass: {max_splats_per_pass:,}")
            aprint(f"Iters/pass: {iters_per_pass}, cull_ratio: {cull_ratio}")
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
            # Reuse the rendered tensor from the previous pass's PSNR computation
            # (same accumulated splats — avoids a redundant full render).
            with torch.no_grad():
                rendered = cached_rendered
                assert rendered is not None  # guaranteed after pass 0
                V_tensor = torch.from_numpy(V_original).to(rendered.device)
                residual_tensor = torch.clamp(V_tensor - rendered, min=0)
                target = residual_tensor.cpu().numpy()

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
            # Coarse-to-fine sigma cascade: cap at previous pass's median sigma.
            prev_lod = accumulated_lods[-1]
            if prev_lod.n_splats > 0:
                prev_sigmas = prev_lod.marginal_sigmas()  # (N, d)
                median_sigma = float(np.median(prev_sigmas))
                pass_kwargs["sigma_max_diag"] = max(2.0, median_sigma)
            else:
                pass_kwargs["sigma_max_diag"] = 4.0

            # Poisson loss: natural for sparse, count-like residuals.
            pass_kwargs["loss_type"] = "poisson"

            # LBFGS optimizer for residual passes: quasi-Newton method that
            # converges in far fewer iterations than Adam by using approximate
            # curvature.  Ideal for our full-batch, deterministic optimization.
            pass_kwargs["optimizer_type"] = "lbfgs"
            pass_kwargs["lr"] = 0.1  # Conservative LR for LBFGS with asymmetric loss
            pass_kwargs["gradient_clip"] = None  # LBFGS handles step size via line search

            # Progressive compactness: gentle shrinkage increasing each pass.
            pass_kwargs["l1_diag"] = 0.0001 * pass_i

        # Disable dynamic ops for progressive passes: seeds are already placed
        # at residual peaks, and diverse benchmark showed no quality benefit
        # from relocation (quality iteration 30).  Saves per-iteration overhead.
        pass_enable_dynamic = False

        result = fit_gaussian_splats(
            target,
            seeds=seeds_this_pass,
            n_iters=iters_per_pass,
            asymmetric_penalty=pass_asymmetric_penalty,
            enable_dynamic_ops=pass_enable_dynamic,
            cull_ratio=cull_ratio,
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
        )
        accumulated_lods.append(lod)

        # --- Compute global PSNR (and cache render for next pass's residual) ---
        accumulated_data = GSplatData.from_lods(accumulated_lods)
        with torch.no_grad():
            cached_rendered = render_to_volume_tensor(
                accumulated_data,
                shape=V.shape,
                device=device,
                truncate=truncate,
            )
            V_tensor = torch.from_numpy(V_original).to(cached_rendered.device)
            quality = compute_quality_metrics(cached_rendered, V_tensor)

        current_psnr = quality["psnr_db"]
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
            lod.stats.get("seeds_requested", lod.n_splats)
            for lod in accumulated_lods
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
                )
            )
        final_result = GSplatData.from_lods(converted_lods, stats=overall_stats)
        if verbose:
            aprint(f"Converted output to physical coordinates (voxel_size={vs.tolist()})")

    return final_result
