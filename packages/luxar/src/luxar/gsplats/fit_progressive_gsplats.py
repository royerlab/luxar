"""Progressive Gaussian splat fitting via iterative residual decomposition.

Fits splats in multiple passes, each targeting the residual (original minus
current approximation).  Each pass solves a simpler subproblem, and the
asymmetric loss ensures under-prediction so residuals are clean and positive.

The result is a multi-LOD ``GSplatData`` where each LOD corresponds to one pass.
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
    splats_per_pass: int = 5000,
    iters_per_pass: int = 1000,
    psnr_patience: float = 0.5,
    asymmetric_penalty: Optional[float] = 10.0,
    enable_dynamic_ops: bool = True,
    on_pass_complete: Optional[Callable[[int, GSplatLOD, float], None]] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    truncate: float = 3.0,
    **kwargs: Any,
) -> GSplatData:
    """Fit Gaussian splats progressively via iterative residual decomposition.

    Each pass fits ``splats_per_pass`` splats to the current residual
    (``clamp(V - render(accumulated), min=0)``).  Passes continue until
    ``max_splats`` is reached, PSNR improvement drops below ``psnr_patience``
    dB, or the residual becomes negligible.

    Parameters
    ----------
    V : np.ndarray
        Input volume to approximate (any dimensionality).
    max_splats : int
        Maximum total number of splats across all passes.
    splats_per_pass : int
        Number of splats to fit in each pass.
    iters_per_pass : int
        Optimization iterations per pass.
    psnr_patience : float
        Stop if ΔPSNR between consecutive passes < this value (in dB).
    asymmetric_penalty : float, optional
        Asymmetric loss penalty factor (default 10.0).
    enable_dynamic_ops : bool
        Whether to enable dynamic splat relocation within each pass.
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
    """
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.metrics import compute_quality_metrics
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    start_time = time.time()
    V_original = V.astype(np.float32)
    max_passes = max(1, max_splats // splats_per_pass)

    accumulated_lods: list[GSplatLOD] = []
    prev_psnr = 0.0
    stop_reason = "max_passes"

    if verbose:
        with asection("Progressive Gaussian splat fitting"):
            aprint(f"Volume shape: {V.shape}")
            aprint(f"Max splats: {max_splats:,}, splats/pass: {splats_per_pass:,}")
            aprint(f"Max passes: {max_passes}, iters/pass: {iters_per_pass}")
            aprint(f"PSNR patience: {psnr_patience} dB")

    for pass_i in range(max_passes):
        pass_start = time.time()

        # --- Compute target for this pass ---
        if pass_i == 0:
            target = V_original
        else:
            # Render accumulated splats and compute clamped residual
            accumulated_data = GSplatData.from_lods(accumulated_lods)
            with torch.no_grad():
                rendered = render_to_volume_tensor(
                    accumulated_data,
                    shape=V.shape,
                    device=device,
                    truncate=truncate,
                )
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

        # --- Determine seed count ---
        total_so_far = sum(lod.n_splats for lod in accumulated_lods)
        seeds_this_pass = min(splats_per_pass, max_splats - total_so_far)
        if seeds_this_pass <= 0:
            stop_reason = "max_splats"
            break

        # --- Fit splats to target ---
        if verbose:
            aprint(f"\n{'=' * 60}")
            aprint(
                f"Pass {pass_i}: fitting {seeds_this_pass} splats to {'volume' if pass_i == 0 else 'residual'}"
            )

        result = fit_gaussian_splats(
            target,
            seeds=seeds_this_pass,
            n_iters=iters_per_pass,
            asymmetric_penalty=asymmetric_penalty,
            enable_dynamic_ops=enable_dynamic_ops,
            device=device,
            verbose=verbose,
            truncate=truncate,
            **kwargs,
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

        # --- Compute global PSNR ---
        accumulated_data = GSplatData.from_lods(accumulated_lods)
        with torch.no_grad():
            rendered = render_to_volume_tensor(
                accumulated_data,
                shape=V.shape,
                device=device,
                truncate=truncate,
            )
            V_tensor = torch.from_numpy(V_original).to(rendered.device)
            quality = compute_quality_metrics(rendered, V_tensor)

        current_psnr = quality["psnr_db"]
        delta_psnr = current_psnr - prev_psnr
        pass_time = time.time() - pass_start

        # Store per-LOD metadata (dict is mutable despite frozen dataclass)
        lod.stats["cumulative_psnr_db"] = current_psnr
        lod.stats["delta_psnr_db"] = delta_psnr
        lod.stats["pass_index"] = pass_i
        lod.stats["pass_time_seconds"] = pass_time

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
        if pass_i > 0 and delta_psnr < psnr_patience:
            if verbose:
                aprint(
                    f"\nPSNR patience: ΔPSNR={delta_psnr:.3f} dB < {psnr_patience} dB, stopping"
                )
            stop_reason = "psnr_patience"
            break

        prev_psnr = current_psnr

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
        "splats_per_pass": splats_per_pass,
        "iters_per_pass": iters_per_pass,
        "psnr_patience": psnr_patience,
    }

    if verbose:
        aprint(f"\n{'=' * 60}")
        aprint(f"Progressive fitting complete: {len(accumulated_lods)} passes")
        aprint(f"Total splats: {total_splats:,}, PSNR: {prev_psnr:.2f} dB")
        aprint(f"Stop reason: {stop_reason}, total time: {total_time:.1f}s")

    return GSplatData.from_lods(accumulated_lods, stats=overall_stats)
