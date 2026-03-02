"""
Result finalization for Gaussian splat fitting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

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
    noise_floor: float | None = None,
    n_bins: int = 16,
) -> None:
    """Print a colorful ASCII histogram of amplitude distribution.

    Parameters
    ----------
    amps : torch.Tensor or np.ndarray
        Amplitude values (normalized scale).
    noise_floor : float, optional
        Culling threshold to mark on the histogram.
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
        if noise_floor is not None:
            aprint(f"culling threshold: {noise_floor:.6f}")

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            count = int(counts[i])
            pct = 100.0 * count / total

            # Bar with color gradient
            bar_len = round(bar_width * count / max_count) if max_count > 0 else 0
            cidx = min(i * len(_GRADIENT_CODES) // n_bins, len(_GRADIENT_CODES) - 1)
            color = f"\033[38;5;{_GRADIENT_CODES[cidx]}m"
            bar = color + "█" * bar_len + _RESET
            pad = " " * (bar_width - bar_len)

            # Mark the bin containing the culling threshold
            marker = ""
            if noise_floor is not None and lo <= noise_floor < hi:
                marker = f" {_DIM}◄ cull{_RESET}"

            aprint(
                f"  {lo:9.6f} ┤{bar}{pad} {count:>{count_width}} ({pct:5.1f}%){marker}"
            )

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
    sharpness: np.ndarray | None = None,
) -> np.ndarray:
    """
    Scale down L rows so no splat extends beyond volume bounds.

    For each splat k in dimension i, ensures:
        effective_truncate * sqrt(Sigma_ii) <= min(center_ki, shape_i - 1 - center_ki)

    where Sigma_ii = sum_j(L[k,i,j]^2) is the marginal variance along axis i,
    and effective_truncate = truncate^(2/s) accounts for per-splat sharpness s.

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
    sharpness : np.ndarray, shape (N,), optional
        Per-splat sharpness values. If None, assumes s=2.0 (standard Gaussian)
        where effective_truncate = truncate.

    Returns
    -------
    np.ndarray, shape (N, d, d)
        Clipped Cholesky factors.
    """
    shape_arr = np.array(shape, dtype=np.float32)  # (d,)
    dist_to_edge = np.maximum(
        np.minimum(centers, shape_arr - 1.0 - centers), 0.0
    )  # (N, d)

    # Sharpness-adjusted truncation (matches rendering_core.py AABB logic)
    # For generalized Gaussian exp(-0.5 * ||y||^s), effective radius scales
    # as truncate^(2/s) where s is sharpness (s=2 → truncate^1 = truncate)
    if sharpness is not None:
        sharpness = np.maximum(sharpness, 1e-6)  # Guard against near-zero exponent
        eff_truncate = truncate ** (2.0 / sharpness)  # (N,)
        max_sigma_sq = (dist_to_edge / eff_truncate[:, np.newaxis]) ** 2  # (N, d)
    else:
        max_sigma_sq = (dist_to_edge / truncate) ** 2  # (N, d)

    # Actual sigma_sq per dimension: Sigma_ii = sum_j(L[i,j]^2)
    actual_sigma_sq = np.sum(Ls * Ls, axis=2)  # (N, d)

    # Scale factor per row: min(1, sqrt(max_allowed / actual))
    eps = 1e-12
    ratio = max_sigma_sq / np.maximum(actual_sigma_sq, eps)
    scale = np.sqrt(np.minimum(ratio, 1.0))  # (N, d)

    # Scale each row of L: Ls_clipped[k,i,j] = Ls[k,i,j] * scale[k,i]
    return Ls * scale[:, :, np.newaxis]  # (N,d,d) * (N,d,1) -> broadcast


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
        Dataclass containing centers, amplitudes, cholesky_factors, sharpnesses, and stats
    """
    # --- Post-fit culling: remove splats below the noise floor ---
    # The culling threshold is a fraction (cull_ratio) of max_abs_error.
    # A splat's peak contribution to any voxel equals its amplitude (at the
    # center). If amplitude < threshold, the splat is negligible — completing
    # the job L1 regularization started.
    amps_dev = optimization_results.amps  # still on device, normalized scale
    n_before = amps_dev.shape[0]

    # Show amplitude distribution before culling
    noise_floor_val = (
        config.cull_ratio * preprocessed_data.max_abs_error
        if config.cull_ratio > 0
        else None
    )
    if config.verbose:
        _print_amplitude_histogram(amps_dev, noise_floor=noise_floor_val)

    if config.cull_ratio > 0:
        noise_floor = config.cull_ratio * preprocessed_data.max_abs_error
        keep_mask = amps_dev >= noise_floor  # GPU-accelerated boolean comparison
        n_keep = int(keep_mask.sum().item())
        n_culled = n_before - n_keep

        if n_culled > 0:
            # Apply mask on-device before CPU transfer (fast GPU index_select)
            keep_indices = keep_mask.nonzero(as_tuple=True)[0]
            centers_dev = optimization_results.centers[keep_indices]
            Ls_dev = optimization_results.Ls[keep_indices]
            amps_dev = amps_dev[keep_indices]
            sharpness_dev = optimization_results.sharpness[keep_indices]

            culled_amps = optimization_results.amps[~keep_mask]
            with asection("Post-fit culling"):
                aprint(
                    f"Removed {n_culled}/{n_before} splats "
                    f"({100 * n_culled / n_before:.1f}%) below noise floor"
                )
                aprint(
                    f"  Threshold: amplitude < {noise_floor:.6f} "
                    f"(= {config.cull_ratio} * max_abs_error)"
                )
                aprint(f"  Remaining: {n_keep} splats")
                aprint(
                    f"  Culled amplitude range: "
                    f"[{culled_amps.min().item():.6f}, {culled_amps.max().item():.6f}]"
                )
        else:
            centers_dev = optimization_results.centers
            Ls_dev = optimization_results.Ls
            sharpness_dev = optimization_results.sharpness
            n_culled = 0
            aprint(
                f"Post-fit culling: 0/{n_before} splats below noise floor "
                f"(threshold: {noise_floor:.6f})"
            )
    else:
        # cull_ratio == 0: culling disabled
        centers_dev = optimization_results.centers
        Ls_dev = optimization_results.Ls
        sharpness_dev = optimization_results.sharpness
        n_culled = 0

    # Transfer to CPU + numpy
    centers_np = centers_dev.cpu().numpy()
    Ls_np = Ls_dev.cpu().numpy()
    amps_np = amps_dev.cpu().numpy()
    sharpness_np = sharpness_dev.cpu().numpy()

    # Rescale amplitudes to original intensity range
    amps_np = amps_np * preprocessed_data.intensity_range
    if config.verbose:
        aprint(
            f"Rescaled amplitudes to original intensity range (factor: {preprocessed_data.intensity_range:.4f})"
        )

    # Clip splats to volume bounds if enabled (before voxel footprint correction)
    if config.clip_to_bounds:
        Ls_np = _clip_to_bounds(
            centers_np, Ls_np, config.V.shape, config.truncate, sharpness_np
        )
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

    # Compute sharpness statistics (handle empty array after culling)
    if len(sharpness_np) > 0:
        sharpness_stats = {
            "sharpness_min": float(np.min(sharpness_np)),
            "sharpness_max": float(np.max(sharpness_np)),
            "sharpness_mean": float(np.mean(sharpness_np)),
            "sharpness_std": float(np.std(sharpness_np)),
            "sharpness_median": float(np.median(sharpness_np)),
        }
    else:
        sharpness_stats = {
            "sharpness_min": float("nan"),
            "sharpness_max": float("nan"),
            "sharpness_mean": float("nan"),
            "sharpness_std": float("nan"),
            "sharpness_median": float("nan"),
        }

    # Compute statistics reflecting best state (not final state)
    stats = {
        "time_seconds": optimization_results.end_time - optimization_results.start_time,
        "iterations": optimization_results.actual_iters,
        "best_iteration": optimization_results.best_iteration,  # Iteration that achieved best quality
        "final_loss": optimization_results.best_loss,
        "final_max_abs_error": optimization_results.best_max_abs_error,
        "final_rel_l2": optimization_results.best_rel_l2,
        "converged": optimization_results.converged_early,
        "early_stopped": optimization_results.early_stopped,
        "n_splats": len(amps_np),  # Final splat count (after culling)
        "n_splats_before_culling": n_before,
        "n_culled": n_culled,
        **sharpness_stats,  # Include sharpness statistics
    }

    # Store movie frames in stats for later display (don't show here to avoid timing issues)
    if (
        config.napari_movie
        and optimization_results.movie_frames is not None
        and len(optimization_results.movie_frames["target"]) > 0
    ):
        stats["movie_frames"] = optimization_results.movie_frames
        stats["movie_shape"] = config.V.shape
    else:
        stats["movie_frames"] = None

    return GSplatData(
        centers=centers_np.astype(np.float32),
        amplitudes=amps_np.astype(np.float32),
        cholesky_factors=cholesky_packed.astype(np.float32),
        sharpnesses=sharpness_np.astype(np.float32),
        stats=stats,
    )
