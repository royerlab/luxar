"""
Result finalization for Gaussian splat fitting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import torch

import numpy as np
from arbol import aprint, asection

from luxar.gsplats.fitting.config import (
    FitConfig,
    OptimizationResults,
    PreprocessedData,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.utils.trils import pack_tril

# ANSI 256-color codes: red → yellow → green → cyan gradient
_GRADIENT_CODES = [
    196,
    196,
    202,
    208,
    214,
    220,
    226,
    190,
    154,
    118,
    82,
    46,
    48,
    50,
    51,
    45,
]
_RESET = "\033[0m"
_DIM = "\033[2m"


def _print_amplitude_histogram(
    amps: "torch.Tensor",
    n_bins: int = 16,
) -> None:
    """Print a colorful ASCII histogram of amplitude distribution.

    Parameters
    ----------
    amps : torch.Tensor or np.ndarray
        Amplitude values (normalized scale).
    n_bins : int
        Number of histogram bins.
    """
    # Convert to numpy
    if hasattr(amps, "cpu"):
        amps_np = amps.detach().cpu().numpy().ravel()
    else:
        amps_np = np.asarray(amps).ravel()

    if len(amps_np) == 0:
        aprint("No amplitudes to display")
        return

    total = len(amps_np)
    min_val = float(amps_np.min())
    max_val = float(amps_np.max())
    median_val = float(np.median(amps_np))
    mean_val = float(np.mean(amps_np))

    # Build histogram bins (nudge upper edge so max value falls inside last bin)
    eps = max(abs(max_val) * 1e-8, 1e-15)
    bin_edges = np.linspace(min_val, max_val + eps, n_bins + 1)
    counts, _ = np.histogram(amps_np, bins=bin_edges)
    max_count = int(counts.max()) if counts.max() > 0 else 1
    count_width = len(str(max_count))
    bar_width = 30

    with asection("📊 Amplitude Distribution (normalized scale)"):
        aprint(
            f"n={total}  "
            f"range: [{min_val:.6f}, {max_val:.6f}]  "
            f"median: {median_val:.6f}  "
            f"mean: {mean_val:.6f}"
        )
        for i in range(n_bins):
            lo = bin_edges[i]
            count = int(counts[i])
            pct = 100.0 * count / total

            # Bar with color gradient
            bar_len = round(bar_width * count / max_count) if max_count > 0 else 0
            cidx = min(i * len(_GRADIENT_CODES) // n_bins, len(_GRADIENT_CODES) - 1)
            color = f"\033[38;5;{_GRADIENT_CODES[cidx]}m"
            bar = color + "█" * bar_len + _RESET
            pad = " " * (bar_width - bar_len)

            aprint(f"  {lo:9.6f} ┤{bar}{pad} {count:>{count_width}} ({pct:5.1f}%)")

        aprint(f"  {max_val:9.6f} ┘")


def _apply_voxel_footprint_correction(
    Ls: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Inflate covariances: Sigma_new = L @ L^T + sigma^2 * I_d

    Works for any dimension d:
    - np.eye(d) creates d-dimensional identity
    - Batch matmul handles any (N, d, d) shape
    - np.linalg.cholesky works for any dimension

    Parameters
    ----------
    Ls : np.ndarray, shape (N, d, d)
        Cholesky factors (lower triangular matrices)
    sigma : float
        Standard deviation in voxel units to add (internally squared to get variance)

    Returns
    -------
    np.ndarray, shape (N, d, d)
        New Cholesky factors L_new where L_new @ L_new^T = Sigma_new
    """
    N, d, _ = Ls.shape  # d detected automatically from input
    variance = sigma * sigma  # Convert sigma to variance
    Sigma = Ls @ Ls.transpose(0, 2, 1)  # (N, d, d) batch matmul
    Sigma_corrected = Sigma + variance * np.eye(d, dtype=Ls.dtype)  # d-dim identity
    return np.linalg.cholesky(Sigma_corrected)  # Works for any d


def _clip_to_bounds(
    centers: np.ndarray,
    Ls: np.ndarray,
    shape: Sequence[int],
    truncate: float,
) -> np.ndarray:
    """
    Scale down L rows so no splat extends beyond volume bounds.

    For each splat k in dimension i, ensures:
        truncate * sqrt(Sigma_ii) <= min(center_ki, shape_i - 1 - center_ki)

    where Sigma_ii = sum_j(L[k,i,j]^2) is the marginal variance along axis i.

    This preserves the splat's orientation (ratios within each row of L) but
    scales it down to fit within the volume bounds.

    Parameters
    ----------
    centers : np.ndarray, shape (N, d)
        Splat center positions in voxel coordinates.
    Ls : np.ndarray, shape (N, d, d)
        Lower-triangular Cholesky factors.
    shape : Sequence[int]
        Volume shape (d elements).
    truncate : float
        Truncation radius (same as used during rendering).

    Returns
    -------
    np.ndarray, shape (N, d, d)
        Clipped Cholesky factors.
    """
    shape_arr = np.array(shape, dtype=np.float32)  # (d,)
    dist_to_edge = np.maximum(
        np.minimum(centers, shape_arr - 1.0 - centers), 0.0
    )  # (N, d)

    max_sigma_sq = (dist_to_edge / truncate) ** 2  # (N, d)

    # Actual sigma_sq per dimension: Sigma_ii = sum_j(L[i,j]^2)
    actual_sigma_sq = np.sum(Ls * Ls, axis=2)  # (N, d)

    # Scale factor per row: min(1, sqrt(max_allowed / actual))
    eps = 1e-12
    ratio = max_sigma_sq / np.maximum(actual_sigma_sq, eps)
    scale = np.sqrt(np.minimum(ratio, 1.0))  # (N, d)

    # Scale each row of L: Ls_clipped[k,i,j] = Ls[k,i,j] * scale[k,i]
    clipped: np.ndarray = Ls * scale[:, :, np.newaxis]  # (N,d,d) * (N,d,1) -> broadcast
    return clipped


def finalize_results(
    optimization_results: OptimizationResults,
    config: FitConfig,
    preprocessed_data: PreprocessedData,
) -> GSplatData:
    """
    Finalize optimization results and return as GSplatData.

    Parameters
    ----------
    optimization_results : OptimizationResults
        Results from optimization loop
    config : FitConfig
        Configuration used for fitting
    preprocessed_data : PreprocessedData
        Preprocessed data with normalization metadata

    Returns
    -------
    GSplatData
        Dataclass containing centers, amplitudes, cholesky_factors, and stats
    """
    amps_dev = optimization_results.amps  # still on device, normalized scale
    centers_dev = optimization_results.centers
    Ls_dev = optimization_results.Ls

    if config.verbose:
        _print_amplitude_histogram(amps_dev)

    # Transfer to CPU + numpy
    centers_np = centers_dev.cpu().numpy()
    Ls_np = Ls_dev.cpu().numpy()
    amps_np = amps_dev.cpu().numpy()

    # Rescale amplitudes to original intensity range.
    # NOTE: image_min (incl. any subtracted background floor) is intentionally
    # NOT added back — output amplitudes are background-relative by design
    # (background -> 0), which is what the viewer wants. See PreprocessedData.floor.
    amps_np = amps_np * preprocessed_data.intensity_range
    if config.verbose:
        aprint(
            f"Rescaled amplitudes to original intensity range (factor: {preprocessed_data.intensity_range:.4f})"
        )

    # Clip splats to volume bounds if enabled (before voxel footprint correction)
    # When downscaling is active, splats are still in downscaled coords here,
    # so use the downscaled shape for clipping.
    if config.clip_to_bounds:
        clip_shape = config.V.shape
        if preprocessed_data.downscale_factors is not None:
            clip_shape = tuple(
                -(-s // f)  # ceil division: equivalent to math.ceil(s / f)
                for s, f in zip(config.V.shape, preprocessed_data.downscale_factors)
            )
        Ls_np = _clip_to_bounds(centers_np, Ls_np, clip_shape, config.truncate)
        if config.verbose:
            aprint(f"Clipped splats to volume bounds (truncate={config.truncate:.1f})")

    # Apply voxel footprint correction if enabled
    if config.voxel_footprint_correction:
        # Check for numeric types (int/float) but exclude bool (which is a subclass of int)
        if isinstance(
            config.voxel_footprint_correction, (int, float)
        ) and not isinstance(config.voxel_footprint_correction, bool):
            sigma = float(config.voxel_footprint_correction)
        else:
            # Default: 1-voxel box footprint has sigma = sqrt(1/12) ≈ 0.289 voxels
            sigma = np.sqrt(1.0 / 12.0)
        Ls_np = _apply_voxel_footprint_correction(Ls_np, sigma)
        if config.verbose:
            aprint(f"Applied voxel footprint correction (sigma={sigma:.4f} voxels)")

    # Pack Cholesky factors (without sharpness)
    cholesky_packed = pack_tril(Ls_np)

    # Rescale from downscaled coords to original coords (before voxel_size conversion)
    if preprocessed_data.downscale_factors is not None:
        from luxar.gsplats.fitting.downscale import (
            rescale_centers,
            rescale_cholesky_packed,
        )

        factors = preprocessed_data.downscale_factors
        centers_np = rescale_centers(centers_np, factors)
        cholesky_packed = rescale_cholesky_packed(cholesky_packed, factors)
        if config.verbose:
            aprint(
                f"Rescaled splats to original coordinates (downscale factors={factors})"
            )

    # Convert to physical coordinates if requested
    if config.output_space == "real" and config.voxel_size is not None:
        vs = config.voxel_size  # (d,)
        d = centers_np.shape[1] if len(centers_np) > 0 else config.V.ndim
        # Scale centers: voxel indices → physical coordinates
        centers_np = centers_np * vs  # (N, d) * (d,)
        # Scale packed Cholesky: row i has (i+1) elements, each scaled by vs[i]
        # L_phys[i,j] = voxel_size[i] * L_vox[i,j]
        tril_scales = np.concatenate([[vs[i]] * (i + 1) for i in range(d)])
        cholesky_packed = cholesky_packed * tril_scales  # (N, tril) * (tril,)
        if config.verbose:
            aprint(
                f"Converted output to physical coordinates (voxel_size={vs.tolist()})"
            )

    # Compute statistics reflecting best state (not final state)
    stats: dict[str, Any] = {
        "time_seconds": optimization_results.end_time - optimization_results.start_time,
        "iterations": optimization_results.actual_iters,
        "best_iteration": optimization_results.best_iteration,  # Iteration that achieved best quality
        "final_loss": optimization_results.best_loss,
        "final_max_abs_error": optimization_results.best_max_abs_error,
        "final_rel_l2": optimization_results.best_rel_l2,
        "converged": optimization_results.converged_early,
        "early_stopped": optimization_results.early_stopped,
        "n_splats": len(amps_np),
        # Normalization metadata (recorded for inspection/reproducibility).
        # `floor` is the background level subtracted before fitting (None if
        # floor suppression was disabled); it is NOT added back to amplitudes.
        "image_min": preprocessed_data.image_min,
        "image_max": preprocessed_data.image_max,
        "intensity_range": preprocessed_data.intensity_range,
        "floor": preprocessed_data.floor,
    }

    # Store movie frames in stats for later display (don't show here to avoid timing issues)
    if (
        config.napari_movie
        and optimization_results.movie_frames is not None
        and len(optimization_results.movie_frames["reconstruction"]) > 0
    ):
        stats["movie_frames"] = optimization_results.movie_frames
        stats["movie_shape"] = config.V.shape
    else:
        stats["movie_frames"] = None

    result = GSplatData(
        centers=centers_np.astype(np.float32),
        amplitudes=amps_np.astype(np.float32),
        cholesky_factors=cholesky_packed.astype(np.float32),
        stats=stats,
        truncation_radius=config.truncate,
    )

    # Compute round-trip quality metrics (PSNR, SSIM, MSE).
    # Skip when output_space="real" — the GSplatData is in physical coordinates
    # which don't match config.V.shape (voxel grid).
    is_voxel_space = not (
        config.output_space == "real" and config.voxel_size is not None
    )
    if is_voxel_space:
        try:
            import torch

            from luxar.gsplats.metrics import compute_quality_metrics
            from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

            with torch.no_grad():
                device = str(preprocessed_data.V_tensor.device)
                rendered = render_to_volume_tensor(
                    result,
                    shape=config.V.shape,
                    device=device,
                    truncate=config.truncate,
                )
                ref = torch.from_numpy(config.V.astype(np.float32)).to(rendered.device)
                quality = compute_quality_metrics(rendered, ref)
                del rendered, ref
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stats["mse"] = quality["mse"]
            stats["psnr_db"] = quality["psnr_db"]
            stats["ssim"] = quality["ssim"]
            if config.verbose:
                aprint(
                    f"Quality: PSNR={quality['psnr_db']:.1f} dB, "
                    f"SSIM={quality['ssim']:.4f}, MSE={quality['mse']:.2e}"
                )
        except Exception as exc:
            if config.verbose:
                aprint(f"Note: post-fit quality metrics skipped ({exc})")

    return result
