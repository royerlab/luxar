"""
Data preprocessing for Gaussian splat fitting.

Handles normalization, seed generation, and gradient dilution compensation.
"""

from __future__ import annotations

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
    seed_kwargs = config.seed_kwargs or {}  # Default to empty dict if None

    # Generate seed centers
    if seeds is None:
        # Auto-generate using specified method
        seed_centers = _generate_seeds(
            V, None, None, config.seed_method, config.verbose, **seed_kwargs
        )
    elif isinstance(seeds, int):
        # User-specified exact count (seed_method still applies)
        seed_centers = _generate_seeds(
            V, None, seeds, config.seed_method, config.verbose, **seed_kwargs
        )
    elif isinstance(seeds, float):
        # User-specified proportion (seed_method still applies)
        seed_centers = _generate_seeds(
            V, seeds, None, config.seed_method, config.verbose, **seed_kwargs
        )
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
        # Diagonal learns at 1.0× base lr (with dilution), L1 ~1%
        config.l1_diag = 0.01 * config.lr  # 1% of base LR
    if config.l1_sharpness is None:
        # Sharpness learns at 0.5× base lr (no dilution), L1 ~2%
        config.l1_sharpness = (
            0.01 * config.lr
        )  # 1% of base LR = 2% of sharpness LR (0.5×)

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
            f"  Sharpness: {config.l1_sharpness:.5f} "
            f"(2% of sharpness LR: {config.lr:.3f} × 0.5)"
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


def _generate_seeds(
    V: np.ndarray,
    proportion: float | None,
    target_count: int | None,
    seed_method: str,
    verbose: bool,
    **seed_kwargs,
) -> np.ndarray:
    """
    Generate seed centers using specified detection method(s).

    Parameters
    ----------
    V : np.ndarray
        Input image/volume
    proportion : float | None
        Target proportion of voxels to use as seeds (0 < proportion <= 1.0).
        If None, uses default heuristic (~1% of voxels).
        Note: Currently not enforced, seed generation methods use their own heuristics.
    target_count : int | None
        Target number of seeds to generate. If specified:
        - If more seeds detected: subsample to exact count (keep highest intensity)
        - If fewer seeds detected: adaptively lower thresholds to find more
        - As last resort: add grid-based seeds to reach target
        If None, uses all generated seeds.
    seed_method : str
        Seed generation method: "gaussian", "decomposition", "both",
        "decomposition,gaussian", or "gaussian,decomposition"
    verbose : bool
        Whether to print progress
    **seed_kwargs
        Additional parameters routed to seed generation methods

    Returns
    -------
    np.ndarray
        Generated seed centers (N, ndim)
    """
    from luxar.gsplats.seeds import generate_seeds

    # Generate seeds using specified method
    seed_centers = generate_seeds(V, method=seed_method, **seed_kwargs)

    # Log initial generation
    if verbose:
        actual_proportion = len(seed_centers) / V.size * 100
        aprint(
            f"Generated {len(seed_centers)} seed centers using '{seed_method}' method "
            f"({actual_proportion:.3f}% of voxels)"
        )

    # Handle target count if specified
    if target_count is not None:
        if len(seed_centers) > target_count:
            # More than needed: subsample with spatial diversity + intensity weighting
            idx = np.clip(
                np.round(seed_centers).astype(int),
                0,
                np.array(V.shape) - 1,
            )
            intensities = V[tuple(idx.T)]
            seed_centers = _subsample_seeds_spatially_diverse(
                seed_centers, intensities, target_count, verbose
            )

            if verbose:
                actual_proportion = len(seed_centers) / V.size * 100
                aprint(
                    f"Subsampled to {len(seed_centers)} seeds "
                    f"(spatial diversity + intensity)"
                )

        elif len(seed_centers) < target_count:
            # Not enough: try adaptive threshold lowering
            seed_centers = _ensure_minimum_seeds(
                V, target_count, seed_centers, seed_method, verbose, **seed_kwargs
            )

    return seed_centers


def _subsample_seeds_spatially_diverse(
    seeds: np.ndarray,
    intensities: np.ndarray,
    target_count: int,
    verbose: bool,
) -> np.ndarray:
    """
    Subsample seeds to exact count with spatial diversity and intensity weighting.

    Uses farthest-first selection among high-quality candidates to ensure
    both good spatial coverage and high-intensity seeds.

    Algorithm:
    1. Filter to keep only seeds above intensity threshold (50th percentile)
    2. Start with highest intensity seed
    3. Iteratively select seed that is FARTHEST from already-selected seeds
    4. Repeat until target_count reached

    Parameters
    ----------
    seeds : np.ndarray, shape (N, ndim)
        Candidate seed positions
    intensities : np.ndarray, shape (N,)
        Intensity values at each seed location
    target_count : int
        Exact number of seeds to select
    verbose : bool
        Whether to print progress

    Returns
    -------
    np.ndarray, shape (target_count, ndim)
        Selected seeds with spatial diversity and high intensity
    """
    from scipy.spatial import cKDTree

    n_available = len(seeds)
    if n_available <= target_count:
        return seeds  # Return all if not enough

    # Determine intensity threshold (50th percentile = median)
    # This filters out lower-quality seeds while keeping good spatial coverage
    intensity_threshold = np.percentile(intensities, 50.0)

    # Filter to high-quality candidates
    valid_mask = intensities >= intensity_threshold
    valid_seeds = seeds[valid_mask]
    valid_intensities = intensities[valid_mask]

    # If filtering removed too many, relax threshold
    if len(valid_seeds) < target_count:
        # Fall back to top N by intensity if not enough high-quality seeds
        top_indices = np.argsort(intensities)[-target_count:]
        return seeds[top_indices]

    # Farthest-first selection with intensity priority
    # Start with highest intensity seed
    selected_indices = [np.argmax(valid_intensities)]
    selected = [valid_seeds[selected_indices[0]]]

    # Build remaining candidates
    remaining_indices = set(range(len(valid_seeds))) - {selected_indices[0]}

    # Iteratively select farthest seed
    while len(selected) < target_count and remaining_indices:
        # Build KD-tree of selected seeds for fast distance queries
        tree = cKDTree(np.array(selected))

        # Find distances to nearest selected seed for all remaining candidates
        remaining_list = list(remaining_indices)
        remaining_coords = valid_seeds[remaining_list]
        distances, _ = tree.query(remaining_coords, k=1)

        # Select the farthest one (maximum distance to nearest selected seed)
        farthest_idx_in_remaining = np.argmax(distances)
        farthest_idx_global = remaining_list[farthest_idx_in_remaining]

        # Add to selection
        selected.append(valid_seeds[farthest_idx_global])
        selected_indices.append(farthest_idx_global)
        remaining_indices.remove(farthest_idx_global)

    result = np.array(selected)

    if verbose and len(result) == target_count:
        # Calculate spatial distribution metric (average nearest-neighbor distance)
        if len(result) > 1:
            tree = cKDTree(result)
            distances, _ = tree.query(result, k=2)  # k=2 to get nearest neighbor
            avg_spacing = np.mean(distances[:, 1])  # distances[:,1] is nearest neighbor
            aprint(
                f"Spatial diversity: avg nearest-neighbor distance = {avg_spacing:.1f} voxels"
            )

    return result


def _ensure_minimum_seeds(
    V: np.ndarray,
    target_count: int,
    initial_seeds: np.ndarray,
    seed_method: str,
    verbose: bool,
    **seed_kwargs,
) -> np.ndarray:
    """
    Ensure minimum seed count through adaptive threshold lowering.

    Strategy:
    1. Progressively lower percentile_thresh to find more seeds
    2. If still not enough, add grid-based seeds as fallback

    Parameters
    ----------
    V : np.ndarray
        Input image/volume
    target_count : int
        Target number of seeds needed
    initial_seeds : np.ndarray
        Seeds already found
    seed_method : str
        Seed generation method
    verbose : bool
        Whether to print progress
    **seed_kwargs
        Seed generation parameters

    Returns
    -------
    np.ndarray
        Seed centers (at least target_count)
    """
    from luxar.gsplats.seeds import generate_seeds

    current_seeds = initial_seeds
    current_count = len(current_seeds)

    # Get current percentile threshold (default is 75 for gaussian)
    base_thresh = seed_kwargs.get("percentile_thresh", 75.0)

    # Try progressively lower thresholds
    thresholds = [70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
    for thresh in thresholds:
        if thresh >= base_thresh:
            continue  # Skip if not actually lower

        if current_count >= target_count:
            break

        if verbose:
            aprint(
                f"Need {target_count - current_count} more seeds, "
                f"retrying with percentile_thresh={thresh}"
            )

        # Try with lower threshold
        kwargs_adjusted = seed_kwargs.copy()
        kwargs_adjusted["percentile_thresh"] = thresh
        new_seeds = generate_seeds(V, method=seed_method, **kwargs_adjusted)

        if len(new_seeds) > current_count:
            current_seeds = new_seeds
            current_count = len(current_seeds)
            if verbose:
                aprint(f"Found {current_count} seeds with lower threshold")

    # If still not enough, add grid-based seeds as fallback
    if current_count < target_count:
        if verbose:
            aprint(
                f"Still need {target_count - current_count} seeds, "
                "adding grid-based fallback seeds"
            )
        current_seeds = _add_grid_fallback_seeds(
            V, target_count, current_seeds, verbose
        )
        current_count = len(current_seeds)

    # If we ended up with more than target (from adaptive threshold),
    # subsample to exact count with spatial diversity
    if current_count > target_count:
        idx = np.clip(
            np.round(current_seeds).astype(int),
            0,
            np.array(V.shape) - 1,
        )
        intensities = V[tuple(idx.T)]
        current_seeds = _subsample_seeds_spatially_diverse(
            current_seeds, intensities, target_count, verbose
        )

        if verbose:
            aprint(
                f"Subsampled to {target_count} seeds "
                f"(spatial diversity + intensity)"
            )

    return current_seeds


def _add_grid_fallback_seeds(
    V: np.ndarray,
    target_count: int,
    existing_seeds: np.ndarray,
    verbose: bool,
) -> np.ndarray:
    """
    Add grid-based seeds to reach target count.

    Places seeds on a uniform grid, avoiding regions near existing seeds.

    Parameters
    ----------
    V : np.ndarray
        Input image/volume
    target_count : int
        Target number of seeds
    existing_seeds : np.ndarray
        Existing seed locations
    verbose : bool
        Whether to print progress

    Returns
    -------
    np.ndarray
        Combined seeds (existing + grid-based)
    """
    needed = target_count - len(existing_seeds)
    if needed <= 0:
        return existing_seeds

    ndim = V.ndim
    shape = np.array(V.shape)

    # Calculate grid spacing to get approximately 'needed' points
    # Volume = ∏ shape[i], points = ∏ (shape[i] / spacing[i])
    # So spacing ≈ (Volume / needed)^(1/ndim)
    volume = np.prod(shape)
    spacing = int(np.ceil((volume / (needed * 2)) ** (1.0 / ndim)))
    spacing = max(spacing, 3)  # Minimum spacing of 3

    # Generate grid points
    grid_coords = []
    ranges = [np.arange(spacing // 2, s, spacing) for s in shape]

    import itertools
    for coords in itertools.product(*ranges):
        grid_coords.append(coords)

    grid_coords = np.array(grid_coords, dtype=float)

    # Remove grid points too close to existing seeds (if any exist)
    if len(existing_seeds) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(existing_seeds)
        distances, _ = tree.query(grid_coords, k=1)
        min_distance = spacing * 0.5  # Keep points at least half-spacing away
        grid_coords = grid_coords[distances > min_distance]

    # Sort by intensity and take top N
    if len(grid_coords) > 0:
        idx = np.clip(
            np.round(grid_coords).astype(int), 0, shape - 1
        )
        intensities = V[tuple(idx.T)]
        sorted_indices = np.argsort(intensities)[::-1]

        # Take enough to reach target
        take = min(needed, len(grid_coords))
        grid_coords = grid_coords[sorted_indices[:take]]

        if verbose:
            aprint(f"Added {len(grid_coords)} grid-based fallback seeds")

        # Combine with existing
        return np.vstack([existing_seeds, grid_coords])
    else:
        if verbose:
            aprint("Warning: Could not add grid seeds, using existing only")
        return existing_seeds


def _normalize_data(
    V: np.ndarray, norm_percentile: float, verbose: bool
) -> tuple[np.ndarray, float, float, float]:
    """Normalize input data to [0, 1] range."""
    # Configurable normalization - store parameters for intensity rescaling
    if norm_percentile == 0.0:
        # Full range normalization
        image_min: float = float(np.min(V))
        image_max: float = float(np.max(V))
        if verbose:
            aprint("Normalization: full min-max range")
    else:
        # Percentile-based robust normalization
        image_min = np.percentile(V, norm_percentile)
        image_max = np.percentile(V, 100.0 - norm_percentile)
        if verbose:
            aprint(
                f"Normalization: {norm_percentile:.1f}%-"
                f"{100.0 - norm_percentile:.1f}% percentile range"
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
                f"Auto-convergence threshold: {max_abs_error:.3f} "
                f"(1% of normalized range)"
            )
    else:
        if verbose:
            aprint(f"Convergence threshold: {max_abs_error:.6f} (user-specified)")

    return max_abs_error
