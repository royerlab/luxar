"""
Result finalization for Gaussian splat fitting.
"""

from __future__ import annotations

import numpy as np
from arbol import aprint

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
    # Extract parameters from optimization results
    centers_np = optimization_results.centers.cpu().numpy()
    Ls_np = optimization_results.Ls.cpu().numpy()
    amps_np = optimization_results.amps.cpu().numpy()
    sharpness_np = optimization_results.sharpness.cpu().numpy()

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

    # Compute sharpness statistics
    sharpness_stats = {
        "sharpness_min": float(np.min(sharpness_np)),
        "sharpness_max": float(np.max(sharpness_np)),
        "sharpness_mean": float(np.mean(sharpness_np)),
        "sharpness_std": float(np.std(sharpness_np)),
        "sharpness_median": float(np.median(sharpness_np)),
    }

    # Compute statistics reflecting best state (not final state)
    stats = {
        "time_seconds": optimization_results.end_time - optimization_results.start_time,
        "iterations": optimization_results.actual_iters,
        "best_iteration": optimization_results.best_iteration,  # Iteration that achieved best quality
        "final_loss": optimization_results.best_loss,
        "final_max_abs_error": optimization_results.best_max_abs_error,
        "converged": optimization_results.actual_iters < config.n_iters,
        "n_splats": len(amps_np),  # Final splat count from best state
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
