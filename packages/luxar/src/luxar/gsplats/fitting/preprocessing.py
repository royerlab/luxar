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
from luxar.gsplats.utils.trils import tril_size


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

    # Calculate gradient dilution compensation
    d = V.ndim
    N = int(seed_centers.shape[0])

    (
        effective_lr,
        gradient_dilution_factor,
        dimensional_complexity,
        parameter_complexity,
    ) = _calculate_gradient_dilution_compensation(d, config.lr, config.verbose)

    # Move to device
    V_tensor = torch.tensor(V_normalized, dtype=torch.float32, device=config.device)

    # Log L1 regularization setting
    if config.verbose:
        aprint(
            f"Proportional L1 regularization: {config.l1_amp:.4f} (10% of lr={config.lr:.3f})"
        )

    return PreprocessedData(
        V_normalized=V_normalized,
        V_tensor=V_tensor,
        seed_centers=seed_centers,
        image_min=image_min,
        image_max=image_max,
        intensity_range=intensity_range,
        effective_lr=effective_lr,
        gradient_dilution_factor=gradient_dilution_factor,
        dimensional_complexity=dimensional_complexity,
        parameter_complexity=parameter_complexity,
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
        # Default heuristic: volume-log-proportional (~1% of voxels)
        peaks_per_scale = max(50, int(log1p(V.size * 0.01)))
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
        scales=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0),  # Universal scale series
        peaks_per_scale=peaks_per_scale,  # Volume-proportional
        percentile_thresh=70.0,  # Inclusive threshold
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


def _calculate_gradient_dilution_compensation(
    d: int, lr: float, verbose: bool
) -> tuple[float, float, float | None, float]:
    """Calculate gradient dilution compensation for higher dimensions."""
    # Enhanced gradient dilution compensation: targeted for 4D+ challenges
    params_2d = 2 + tril_size(2)  # 5 parameters (baseline)
    params_current = d + tril_size(d)  # Current dimension parameters

    if d <= 3:
        # Conservative scaling for 2D/3D (maintain existing quality)
        gradient_dilution_factor = params_current / params_2d
        dimensional_complexity = None
        parameter_complexity = gradient_dilution_factor
    else:
        # More aggressive scaling for 4D+ (address empirical findings)
        dimensional_complexity = d**0.8  # Spatial complexity scaling
        parameter_complexity = params_current / params_2d  # Parameter dilution
        gradient_dilution_factor = dimensional_complexity * parameter_complexity

    effective_lr = lr * gradient_dilution_factor

    if verbose:
        if d <= 3:
            aprint(
                f"Gradient dilution compensation: {d}D uses {gradient_dilution_factor:.1f}× learning rate ({lr:.3f} → {effective_lr:.3f})"
            )
        else:
            aprint(
                f"Enhanced gradient dilution compensation: {d}D uses {gradient_dilution_factor:.1f}× learning rate ({lr:.3f} → {effective_lr:.3f})"
            )
            aprint(
                f"  Dimensional complexity: {dimensional_complexity:.1f}×, Parameter complexity: {parameter_complexity:.1f}×"
            )

    return (
        effective_lr,
        gradient_dilution_factor,
        dimensional_complexity,
        parameter_complexity,
    )
