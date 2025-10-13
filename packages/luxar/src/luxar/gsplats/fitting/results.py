"""
Result finalization for Gaussian splat fitting.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from arbol import aprint

from luxar.gsplats.fitting.config import (
    FitConfig,
    OptimizationResults,
    PreprocessedData,
)
from luxar.gsplats.fitting.visualization import (
    display_compression_analysis,
    show_optimization_movie,
)
from luxar.gsplats.utils.trils import pack_tril


def finalize_results(
    optimization_results: OptimizationResults,
    config: FitConfig,
    preprocessed_data: PreprocessedData,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Finalize optimization results and return in expected format.

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
    params_full : np.ndarray
        Concatenated parameters [centers, packed_cholesky]
    amps : np.ndarray
        Amplitudes rescaled to original intensity range
    stats : dict
        Optimization statistics
    """
    # Extract parameters from optimization results
    centers_np = optimization_results.centers.cpu().numpy()
    Ls_np = optimization_results.Ls.cpu().numpy()
    amps_np = optimization_results.amps.cpu().numpy()

    # Rescale amplitudes to original intensity range
    amps_np = amps_np * preprocessed_data.intensity_range
    if config.verbose:
        aprint(
            f"Rescaled amplitudes to original intensity range (factor: {preprocessed_data.intensity_range:.4f})"
        )

    # Pack parameters
    params_full = np.concatenate([centers_np, pack_tril(Ls_np)], axis=1)

    # Compute statistics reflecting best state (not final state)
    stats = {
        "time_seconds": optimization_results.end_time - optimization_results.start_time,
        "iterations": optimization_results.actual_iters,
        "best_iteration": optimization_results.best_iteration,  # Iteration that achieved best quality
        "final_loss": optimization_results.best_loss,
        "final_max_abs_error": optimization_results.best_max_abs_error,
        "converged": optimization_results.actual_iters < config.n_iters,
        "n_splats": len(amps_np),  # Final splat count from best state
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

    return params_full.astype(np.float32), amps_np.astype(np.float32), stats
