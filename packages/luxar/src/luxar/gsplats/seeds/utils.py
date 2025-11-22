# utils.py
"""
Shared utility functions for seed generation.

This module contains common utilities used by multiple seed generation methods,
including peak detection and spatial deduplication algorithms.
"""

from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def local_maxima(
    img: np.ndarray, radius: int, thresh: float, top_k: Optional[int]
) -> np.ndarray:
    """
    Find local maxima in n-dimensional image using L∞ (Chebyshev) neighborhood.

    This function identifies peaks by comparing each voxel with all neighbors within
    a hypercube of given radius. A voxel is considered a peak if it equals the
    maximum value in its neighborhood and exceeds the threshold.

    Parameters
    ----------
    img : np.ndarray
        Input n-dimensional image array.
    radius : int
        Half-width of the L∞ neighborhood (hypercube). Minimum value is 1.
    thresh : float
        Minimum intensity threshold for peak detection.
    top_k : int, optional
        Maximum number of strongest peaks to return. If None, returns all peaks.

    Returns
    -------
    np.ndarray
        Array of shape (N, ndim) containing integer coordinates of detected peaks,
        where N is the number of peaks found.
    """
    # Ensure minimum neighborhood size of 3x3x...x3 (radius=1)
    if radius < 1:
        radius = 1

    # Use size parameter instead of footprint for better memory efficiency
    # This avoids creating large boolean arrays: (2*radius+1)^ndim
    # For 3D with radius=5: footprint would be 11^3 = 1331 elements vs size=[11,11,11]
    kernel_size = [2 * radius + 1] * img.ndim

    # Apply maximum filter to find local maxima
    # Each voxel is compared with all neighbors in the hypercube
    max_f = ndi.maximum_filter(img, size=kernel_size, mode="nearest")

    # Peak criteria: voxel equals neighborhood maximum AND exceeds threshold
    peaks_mask = (img == max_f) & (img >= thresh)
    # Extract coordinates of detected peaks
    coords = np.argwhere(peaks_mask)

    # Handle empty result case
    if coords.size == 0:
        return coords

    # Optionally limit to top_k strongest peaks
    if top_k is not None and len(coords) > top_k:
        # Get intensity values at peak locations
        vals = img[tuple(coords.T)]
        # Select indices of top_k strongest peaks (highest intensities)
        keep = np.argsort(vals)[-top_k:]
        coords = coords[keep]
    return coords


def dedupe_farthest_first(
    coords: np.ndarray, min_distance: float, intensities: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Remove duplicate seeds using farthest-first selection for maximum spatial diversity.

    Uses KD-tree for O(N log N) performance instead of naive O(N³) approach.

    Algorithm:
    1. Sort seeds by intensity (if provided) or keep original order
    2. Start with strongest/first seed
    3. For all remaining seeds, find nearest distance to selected set
    4. Select seed with MAXIMUM nearest-distance (farthest-first)
    5. Repeat until no seeds satisfy min_distance constraint

    This ensures maximum spatial spread with quality priority.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, ndim) containing seed coordinates.
    min_distance : float
        Minimum Euclidean distance strictly enforced between kept seeds.
        Seeds closer than min_distance to any selected seed are excluded.
    intensities : np.ndarray, optional
        Array of shape (N,) containing intensity/quality values for each seed.
        If provided, seeds are sorted by intensity (highest first) before selection.

    Returns
    -------
    np.ndarray
        Array of deduplicated coordinates with maximum spatial diversity.
        Returns float coordinates for consistency with downstream processing.

    Notes
    -----
    - Time complexity: O(N * M * log M) where M is number of selected seeds
      (worst case O(N² log N) when most seeds are kept)
    - Space complexity: O(M) where M is the number of selected seeds
    - Uses KD-tree for O(log M) nearest-neighbor queries instead of O(M) naive
    - For small datasets (<50 seeds), uses simple O(N²) greedy fallback
    """
    # Handle empty input case
    if len(coords) == 0:
        return coords.astype(float)

    # For very small inputs, use simple greedy (overhead not worth it)
    if len(coords) < 50:
        return _dedupe_simple(coords, min_distance, intensities)

    # Sort by intensity if provided (highest first), otherwise keep original order
    if intensities is not None:
        sort_indices = np.argsort(intensities)[::-1]  # Descending order
        coords_sorted = coords[sort_indices].astype(float)
    else:
        coords_sorted = coords.astype(float)

    # Start with first (strongest) seed
    selected = [coords_sorted[0]]
    selected_array = np.array(selected)
    tree = cKDTree(selected_array)

    # Track which seeds have been used
    remaining_mask = np.ones(len(coords_sorted), dtype=bool)
    remaining_mask[0] = False  # First one is already selected

    # Farthest-first selection loop
    while True:
        # Get indices of remaining seeds
        remaining_indices = np.where(remaining_mask)[0]

        if len(remaining_indices) == 0:
            break  # No more seeds

        # Query KD-tree for all remaining seeds at once (vectorized)
        remaining_coords = coords_sorted[remaining_indices]
        distances, _ = tree.query(remaining_coords, k=1)

        # Find seeds that satisfy min_distance constraint
        valid_mask = distances >= min_distance

        if not np.any(valid_mask):
            break  # No more seeds satisfy constraint

        # Among valid seeds, pick the FARTHEST one (maximum distance)
        valid_distances = distances[valid_mask]
        valid_indices_in_remaining = np.where(valid_mask)[0]
        farthest_idx_in_valid = np.argmax(valid_distances)
        farthest_idx_in_remaining = valid_indices_in_remaining[farthest_idx_in_valid]
        farthest_idx_global = remaining_indices[farthest_idx_in_remaining]

        # Add the farthest seed
        selected.append(coords_sorted[farthest_idx_global])
        remaining_mask[farthest_idx_global] = False

        # Rebuild KD-tree with all selected points
        selected_array = np.array(selected)
        tree = cKDTree(selected_array)

    return np.array(selected, dtype=float)


def _dedupe_simple(
    coords: np.ndarray, min_distance: float, intensities: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Simple greedy deduplication for small datasets.

    O(N²) algorithm used for <50 seeds where KD-tree overhead isn't worth it.
    """
    if len(coords) == 0:
        return coords.astype(float)

    # Sort by intensity if provided
    if intensities is not None:
        sort_indices = np.argsort(intensities)[::-1]
        coords_sorted = coords[sort_indices]
    else:
        coords_sorted = coords

    # Greedy selection
    used = np.zeros(len(coords_sorted), dtype=bool)
    selected = []

    for i in range(len(coords_sorted)):
        if used[i]:
            continue

        c = coords_sorted[i]
        selected.append(c)

        # Mark nearby seeds as used
        diffs = coords_sorted - c
        dist2 = np.sum(diffs**2, axis=1) if coords_sorted.ndim > 1 else diffs**2
        used |= dist2 < (min_distance**2)

    return np.array(selected, dtype=float)


def combine_seeds(
    *candidate_arrays: np.ndarray,
    min_distance: Optional[float] = None,
    method: str = "union",
) -> np.ndarray:
    """
    Combine seeds from multiple detection methods.

    Merges seed arrays from different methods and optionally deduplicates
    spatially close seeds.

    Parameters
    ----------
    *candidate_arrays : np.ndarray
        Variable number of seed arrays, each of shape (N_i, ndim).
        Arrays with zero seeds are automatically filtered out.
    min_distance : float, optional
        If provided, deduplicate seeds using farthest-first selection.
        Seeds closer than min_distance are merged. If None, no deduplication
        is performed (simple concatenation).
    method : str, default="union"
        Combination method. Currently only "union" is supported.
        Future: could support "intersection", "weighted", etc.

    Returns
    -------
    np.ndarray
        Combined seeds of shape (M, ndim) where M is the total number
        of seeds after merging and optional deduplication.

    Examples
    --------
    >>> from luxar.gsplats.seeds import (
    ...     find_seeds_multiscale_gaussian,
    ...     find_seeds_multiscale_decomposition,
    ... )
    >>> from luxar.gsplats.seeds.utils import combine_seeds
    >>>
    >>> # Generate seeds from both methods
    >>> seeds1 = find_seeds_multiscale_gaussian(image)
    >>> seeds2 = find_seeds_multiscale_decomposition(image, scales=[1, 2, 4, 8])
    >>>
    >>> # Combine with deduplication
    >>> combined = combine_seeds(seeds1, seeds2, min_distance=3.0)
    >>> print(f"Combined: {len(seeds1)} + {len(seeds2)} → {len(combined)} seeds")

    Notes
    -----
    - Empty seed arrays are automatically filtered out
    - If min_distance is provided, uses farthest-first deduplication
    - Seeds are processed in the order provided (first array has priority)
    - For best results, pass seeds from coarse-scale methods first
    """
    if method != "union":
        raise ValueError(
            f"Unknown combination method: {method}. Only 'union' is supported."
        )

    # Filter out empty arrays
    non_empty = [arr for arr in candidate_arrays if len(arr) > 0]

    if len(non_empty) == 0:
        # All arrays are empty - return empty array with correct shape
        # Infer ndim from first non-None array, or default to 2
        for arr in candidate_arrays:
            if arr is not None:
                return np.zeros((0, arr.shape[1]), dtype=float)
        return np.zeros((0, 2), dtype=float)  # Default to 2D

    # Concatenate all non-empty arrays
    combined = np.vstack(non_empty)

    # Optionally deduplicate
    if min_distance is not None:
        combined = dedupe_farthest_first(combined, min_distance=min_distance)

    return combined
