"""
Result finalization for Gaussian splat fitting.
"""

from __future__ import annotations

import numpy as np
from arbol import aprint

from luxar.gsplats.fit_result import GaussianSplatResult
from luxar.gsplats.fitting.config import (
    FitConfig,
    OptimizationResults,
    PreprocessedData,
)
from luxar.gsplats.utils.trils import pack_tril


def finalize_results(
    optimization_results: OptimizationResults,
    config: FitConfig,
    preprocessed_data: PreprocessedData,
) -> GaussianSplatResult:
    """
    Finalize optimization results and return as GaussianSplatResult.

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
    GaussianSplatResult
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

    return GaussianSplatResult(
        centers=centers_np.astype(np.float32),
        amplitudes=amps_np.astype(np.float32),
        cholesky_factors=cholesky_packed.astype(np.float32),
        sharpnesses=sharpness_np.astype(np.float32),
        stats=stats,
    )
