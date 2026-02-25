"""
Result finalization for Gaussian splat fitting.
"""

from __future__ import annotations

import numpy as np
from arbol import aprint, asection

from luxar.gsplats.fitting.config import (
    FitConfig,
    OptimizationResults,
    PreprocessedData,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.utils.trils import pack_tril


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
