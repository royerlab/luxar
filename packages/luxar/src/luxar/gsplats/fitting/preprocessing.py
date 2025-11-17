"""
Data preprocessing for Gaussian splat fitting.

Handles normalization, candidate generation, and gradient dilution compensation.
"""

from __future__ import annotations

from math import log1p

import numpy as np
import torch
from arbol import aprint

from luxar.gsplats.fitting.config import FitConfig, PreprocessedData


def preprocess_data(config: FitConfig) -> PreprocessedData:
    """
    Preprocess input data for optimization.

    Performs normalization, seed generation, and gradient dilution compensation.

    Parameters
    ----------
    config : FitConfig
        Configuration containing input data and parameters

    Returns
    -------
    PreprocessedData
        Preprocessed data ready for optimization
    """
    V = config.V.copy()  # Work with a copy
    seeds = config.seeds

    # Generate seed centers
    if seeds is None:
        # Auto-generate with default proportion
        seed_centers = _generate_candidates(V, None, config.verbose)
    elif isinstance(seeds, (int, float)):
        # User-specified proportion
        seed_centers = _generate_candidates(V, float(seeds), config.verbose)
    else:
        # User-provided array of seed centers
        seed_centers = seeds

    # Normalize input data
    V_normalized, image_min, image_max, intensity_range = _normalize_data(
        V, config.norm_percentile, config.verbose
    )

    # Set auto-convergence threshold
    max_abs_error = _set_convergence_threshold(config.max_abs_error, config.verbose)

    # Get dimensions
    d = V.ndim
    N = int(seed_centers.shape[0])

    # Set L1 regularization defaults based on parameter type learning rate multipliers
    # These scale with the learning rates used by the optimizer for each parameter type
    if config.l1_amp is None:
        # Amplitude learns at 2.0× base lr, so L1 should be ~5% of that
        config.l1_amp = 0.1 * config.lr  # 10% of base LR = 5% of amplitude LR (2.0×)
    if config.l1_diag is None:
        # Diagonal/variance learns at 1.0× base lr (with gradient dilution), so L1 should be ~1%
        config.l1_diag = 0.01 * config.lr  # 1% of base LR
    if config.l1_sharpness is None:
        # Sharpness learns at 0.5× base lr (no gradient dilution), so L1 should be ~10% of that
        config.l1_sharpness = (
            0.05 * config.lr
        )  # 5% of base LR = 10% of sharpness LR (0.5×)

    # Move to device
    V_tensor = torch.tensor(V_normalized, dtype=torch.float32, device=config.device)

    # Log L1 regularization settings
    if config.verbose:
        aprint("L1 regularization (as % of parameter type LR):")
        aprint(
            f"  Amplitude: {config.l1_amp:.4f} (5% of amp LR: {config.lr:.3f} × 2.0)"
        )
        aprint(
            f"  Diagonal: {config.l1_diag:.5f} (1% of diag LR: {config.lr:.3f} × 1.0)"
        )
        aprint(
            f"  Sharpness: {config.l1_sharpness:.5f} (10% of sharpness LR: {config.lr:.3f} × 0.5)"
        )

    return PreprocessedData(
        V_normalized=V_normalized,
        V_tensor=V_tensor,
        seed_centers=seed_centers,
        image_min=image_min,
        image_max=image_max,
        intensity_range=intensity_range,
        d=d,
        N=N,
        max_abs_error=max_abs_error,
    )


def _generate_candidates(
    V: np.ndarray, proportion: float | None, verbose: bool
) -> np.ndarray:
    """
    Generate seed centers using multiscale approach.

    Parameters
    ----------
    V : np.ndarray
        Input image/volume
    proportion : float | None
        Target proportion of voxels to use as seeds (0 < proportion <= 1.0).
        If None, uses default heuristic (~1% of voxels).
    verbose : bool
        Whether to print progress

    Returns
    -------
    np.ndarray
        Generated seed centers (N, ndim)
    """
    from luxar.gsplats.candidates import find_candidates_overcomplete_nd

    # Determine target number of seeds
    if proportion is None:
        # Default heuristic: volume-proportional (~0.1% of voxels)
        peaks_per_scale = max(50, int(V.size * 0.001))
    else:
        # User-specified proportion
        target_seeds = int(V.size * proportion)
        # Distribute across scales (assuming 6 scales)
        peaks_per_scale = max(10, target_seeds // 6)

    if verbose:
        if proportion is None:
            aprint(
                f"Auto-generating seeds: {peaks_per_scale} peaks/scale for {V.size:,} pixels"
            )
        else:
            aprint(
                f"Generating seeds: {proportion * 100:.2f}% of {V.size:,} pixels = ~{target_seeds} seeds"
            )

    seed_centers = find_candidates_overcomplete_nd(
        V,
        scales=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0),  # Universal scale series
        peaks_per_scale=peaks_per_scale,  # Volume-proportional
        percentile_thresh=10.0,  # Inclusive threshold
        min_dist=2.0,  # Standard spacing
        add_intensity_grid=False,  # Clean peak-based detection
    )

    if verbose:
        actual_proportion = len(seed_centers) / V.size * 100
        aprint(
            f"Generated {len(seed_centers)} seed centers ({actual_proportion:.3f}% of voxels)"
        )

    return seed_centers


def _normalize_data(
    V: np.ndarray, norm_percentile: float, verbose: bool
) -> tuple[np.ndarray, float, float, float]:
    """Normalize input data to [0, 1] range."""
    # Configurable normalization - store parameters for intensity rescaling
    if norm_percentile == 0.0:
        # Full range normalization
        image_min = np.min(V)
        image_max = np.max(V)
        if verbose:
            aprint("Normalization: full min-max range")
    else:
        # Percentile-based robust normalization
        image_min = np.percentile(V, norm_percentile)
        image_max = np.percentile(V, 100.0 - norm_percentile)
        if verbose:
            aprint(
                f"Normalization: {norm_percentile:.1f}%-{100.0 - norm_percentile:.1f}% percentile range"
            )

    intensity_range = image_max - image_min

    if np.abs(intensity_range) < 1e-12:
        V = np.full_like(V, 0.5, dtype=np.float32)
        intensity_range = 1.0  # Avoid division by zero in rescaling
        if verbose:
            aprint("Warning: Input image is nearly uniform")
    else:
        V = np.clip((V - image_min) / intensity_range, 0.0, 1.0)

    return V, image_min, image_max, intensity_range


def _set_convergence_threshold(max_abs_error: float | None, verbose: bool) -> float:
    """Set convergence threshold with sensible default."""
    # Auto-convergence threshold: set sensible default if not provided
    if max_abs_error is None:
        max_abs_error = 0.01  # 1% of normalized [0,1] dynamic range
        if verbose:
            aprint(
                f"Auto-convergence threshold: {max_abs_error:.3f} (1% of normalized range)"
            )
    else:
        if verbose:
            aprint(f"Convergence threshold: {max_abs_error:.6f} (user-specified)")

    return max_abs_error
