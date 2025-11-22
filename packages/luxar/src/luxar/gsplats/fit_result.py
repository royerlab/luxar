"""Gaussian Splat fitting result dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class GaussianSplatResult:
    """Results from Gaussian splat fitting.

    Attributes
    ----------
    centers : np.ndarray, shape (N, d)
        Splat center positions in voxel coordinates.
    amplitudes : np.ndarray, shape (N,)
        Non-negative splat amplitudes, rescaled to original image intensity range.
    cholesky_factors : np.ndarray, shape (N, d*(d+1)//2)
        Packed lower-triangular Cholesky factors (L) where Σ = L @ L.T.
        The packing order follows: [L00, L10, L11, L20, L21, L22, ...]
    sharpnesses : np.ndarray, shape (N,)
        Per-splat sharpness values (s=2.0 is standard Gaussian).
        Lower values (s<2) create heavy-tailed Gaussians.
        Higher values (s>2) create sharper, more compact splats.
    stats : Dict[str, Any]
        Optimization statistics including:
        - time_seconds: Total optimization time
        - iterations: Number of iterations completed
        - converged: Whether convergence criteria were met
        - sharpness_min/max/mean/std/median: Sharpness statistics
        - best_iteration: Iteration where best state was found
        - final_max_abs_error: Maximum absolute error in best state
        - movie_frames: Optional optimization movie frames (if napari_movie=True)
    """

    centers: np.ndarray
    amplitudes: np.ndarray
    cholesky_factors: np.ndarray
    sharpnesses: np.ndarray
    stats: Dict[str, Any]
