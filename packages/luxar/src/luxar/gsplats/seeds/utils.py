# utils.py
"""
Shared utility functions for seed generation.

This module contains common utilities used by multiple seed generation methods,
including peak detection, spatial deduplication algorithms, and Cholesky factor
construction for Gaussian initialization.
"""

from typing import Optional, cast

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# Amplitude scaling factor for seed initialization.
# Multiplying by 0.9 (90%) helps avoid initial over-prediction when splats overlap,
# which can trigger the asymmetric over-prediction penalty and cause divergence.
# Starting slightly below the target intensity allows the optimizer to increase
# amplitudes as needed rather than fighting against penalty gradients.
SEED_AMPLITUDE_SCALE = 0.9


def sigmas_to_cholesky_isotropic(
    sigmas: np.ndarray,
    ndim: int,
) -> np.ndarray:
    """
    Convert per-seed isotropic sigmas to packed Cholesky factors.

    For isotropic Gaussians, the covariance matrix is Σ = σ²I,
    so the Cholesky factor is L = σI (diagonal matrix).

    Parameters
    ----------
    sigmas : np.ndarray, shape (N,)
        Isotropic sigma (standard deviation) per seed.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    cholesky_factors : np.ndarray, shape (N, ndim*(ndim+1)//2)
        Packed lower-triangular Cholesky factors for isotropic Gaussians.
        L = diag(sigma, sigma, ...) so L @ L.T = diag(sigma², ...)

    Notes
    -----
    The packed Cholesky format stores the lower triangular elements
    column by column: [L00, L10, L11, L20, L21, L22, ...].
    For isotropic (diagonal) matrices, only diagonal positions are non-zero.

    Diagonal indices in packed format: 0, 2, 5, 9, 14, ... = k*(k+3)//2

    Examples
    --------
    >>> sigmas = np.array([1.0, 2.0, 3.0])  # 3 seeds
    >>> L = sigmas_to_cholesky_isotropic(sigmas, ndim=3)
    >>> L.shape
    (3, 6)
    >>> # For first seed (sigma=1.0): L = [[1,0,0],[0,1,0],[0,0,1]]
    >>> # Packed: [1, 0, 1, 0, 0, 1] → diagonal at indices 0, 2, 5
    """
    sigmas = np.asarray(sigmas, dtype=np.float32)
    N = len(sigmas)
    tril_size = ndim * (ndim + 1) // 2
    cholesky = np.zeros((N, tril_size), dtype=np.float32)

    # For isotropic: diagonal elements are sigma, off-diagonal are 0
    # Packed order: [L00, L10, L11, L20, L21, L22, ...]
    # Diagonal indices for dimension k: k*(k+1)//2 + k = k*(k+3)//2
    for k in range(ndim):
        diag_idx = k * (k + 3) // 2
        cholesky[:, diag_idx] = sigmas

    return cholesky


def soft_blur_nd(img: np.ndarray) -> np.ndarray:
    """
    Apply a soft separable blur to reduce noise before peak detection.

    Uses a 3-point kernel [0.25, 0.5, 0.25] applied separably along each axis.
    This is equivalent to a tent filter that smooths high-frequency noise
    while preserving peak locations.

    Parameters
    ----------
    img : np.ndarray
        Input n-dimensional image array.

    Returns
    -------
    np.ndarray
        Blurred image with same shape as input.

    Notes
    -----
    The kernel [0.25, 0.5, 0.25] corresponds to [0.5, 1.0, 0.5] normalized.
    Separable application is O(3*ndim*N) instead of O(3^ndim * N) for direct.
    """
    # Tent kernel: [0.5, 1.0, 0.5] normalized → [0.25, 0.5, 0.25]
    kernel_1d = np.array([0.25, 0.5, 0.25], dtype=np.float32)

    result = img.astype(np.float32)
    for axis in range(img.ndim):
        result = ndi.convolve1d(result, kernel_1d, axis=axis, mode="nearest")

    return result


def count_local_maxima(
    img: np.ndarray,
    radius: int = 1,
    threshold_rel: float = 0.1,
    blur: bool = True,
) -> int:
    """
    Count local maxima in an n-dimensional image.

    Optionally applies soft blur to reduce noise-induced false peaks.

    Parameters
    ----------
    img : np.ndarray
        Input n-dimensional image array.
    radius : int, default=1
        Half-width of the L∞ neighborhood (hypercube).
    threshold_rel : float, default=0.1
        Relative threshold (fraction of image max) for peak detection.
    blur : bool, default=True
        Apply soft blur before counting to reduce noise.

    Returns
    -------
    int
        Number of local maxima detected.
    """
    if blur:
        img = soft_blur_nd(img)

    max_val = img.max()
    if max_val <= 0:
        return 0

    thresh = threshold_rel * max_val
    peaks = local_maxima(img, radius=radius, thresh=thresh, top_k=None)
    return len(peaks)


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
        return cast(np.ndarray, coords)

    # Optionally limit to top_k strongest peaks
    if top_k is not None and len(coords) > top_k:
        # Get intensity values at peak locations
        vals = img[tuple(coords.T)]
        # Select indices of top_k strongest peaks (highest intensities)
        keep = np.argsort(vals)[-top_k:]
        coords = coords[keep]
    return cast(np.ndarray, coords)


def dedupe_farthest_first(
    coords: np.ndarray, min_distance: float, intensities: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
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
    tuple[np.ndarray, np.ndarray]
        (deduped_coords, kept_indices) where:
        - deduped_coords: Array of deduplicated coordinates (M, ndim)
        - kept_indices: Indices into original coords array (M,)
        This allows O(1) indexing into other arrays instead of O(M×N) coordinate matching.

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
        return coords.astype(float), np.array([], dtype=np.intp)

    # For very small inputs, use simple greedy (overhead not worth it)
    if len(coords) < 50:
        return _dedupe_simple(coords, min_distance, intensities)

    # Sort by intensity if provided (highest first), otherwise keep original order
    if intensities is not None:
        sort_indices = np.argsort(intensities)[::-1]  # Descending order
        coords_sorted = coords[sort_indices].astype(np.float32)
    else:
        sort_indices = None
        coords_sorted = coords.astype(np.float32)

    # Pre-allocate arrays for selected seeds (avoids repeated list→array conversions)
    max_selections = len(coords_sorted)
    selected_array = np.empty((max_selections, coords_sorted.shape[1]), dtype=np.float32)
    selected_sorted_indices_array = np.empty(max_selections, dtype=np.intp)

    # Start with first (strongest) seed
    selected_array[0] = coords_sorted[0]
    selected_sorted_indices_array[0] = 0
    n_selected = 1

    tree = cKDTree(selected_array[:1])

    # Track which seeds have been used
    remaining_mask = np.ones(len(coords_sorted), dtype=bool)
    remaining_mask[0] = False  # First one is already selected

    # Farthest-first selection loop
    while True:
        # Get indices of remaining seeds
        remaining_indices = np.where(remaining_mask)[0]

        if len(remaining_indices) == 0:
            break  # No more seeds

        # Early termination optimization: for last few candidates, use simple O(N²) check
        # This avoids tree rebuild overhead when very few candidates remain
        if len(remaining_indices) <= 5:
            found_valid = False
            for idx in remaining_indices:
                coord = coords_sorted[idx]
                # Check distance to all selected seeds
                diffs = selected_array[:n_selected] - coord
                min_dist_sq = np.min(np.sum(diffs**2, axis=1))
                if min_dist_sq >= min_distance**2:
                    # Found valid seed, add it
                    selected_array[n_selected] = coord
                    selected_sorted_indices_array[n_selected] = idx
                    n_selected += 1
                    remaining_mask[idx] = False
                    found_valid = True
                    break
            if not found_valid:
                break
            continue

        # Query KD-tree for all remaining seeds at once (vectorized)
        remaining_coords = coords_sorted[remaining_indices]
        distances, _ = tree.query(remaining_coords, k=1)
        distances = np.atleast_1d(distances)

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

        # Add the farthest seed to pre-allocated array
        selected_array[n_selected] = coords_sorted[farthest_idx_global]
        selected_sorted_indices_array[n_selected] = farthest_idx_global
        n_selected += 1
        remaining_mask[farthest_idx_global] = False

        # Rebuild KD-tree with array slice (no list→array conversion overhead)
        tree = cKDTree(selected_array[:n_selected])

    # Trim to actual size and convert back to float64 for consistency
    deduped_coords = selected_array[:n_selected].astype(float)
    selected_sorted_indices = selected_sorted_indices_array[:n_selected]

    # Map back to original indices if we sorted by intensity
    if sort_indices is not None:
        kept_indices = sort_indices[selected_sorted_indices]
    else:
        kept_indices = selected_sorted_indices

    return deduped_coords, kept_indices


def _dedupe_simple(
    coords: np.ndarray, min_distance: float, intensities: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple greedy deduplication for small datasets.

    O(N²) algorithm used for <50 seeds where KD-tree overhead isn't worth it.

    Returns (deduped_coords, kept_indices) tuple.
    """
    if len(coords) == 0:
        return coords.astype(float), np.array([], dtype=np.intp)

    # Sort by intensity if provided
    if intensities is not None:
        sort_indices = np.argsort(intensities)[::-1]
        coords_sorted = coords[sort_indices]
    else:
        sort_indices = None
        coords_sorted = coords

    # Greedy selection
    used = np.zeros(len(coords_sorted), dtype=bool)
    selected = []
    selected_sorted_indices = []

    for i in range(len(coords_sorted)):
        if used[i]:
            continue

        c = coords_sorted[i]
        selected.append(c)
        selected_sorted_indices.append(i)

        # Mark nearby seeds as used
        diffs = coords_sorted - c
        dist2 = np.sum(diffs**2, axis=1) if coords_sorted.ndim > 1 else diffs**2
        used |= dist2 < (min_distance**2)

    # Convert to arrays
    deduped_coords = np.array(selected, dtype=float)
    selected_sorted_indices = np.array(selected_sorted_indices, dtype=np.intp)

    # Map back to original indices if we sorted by intensity
    if sort_indices is not None:
        kept_indices = sort_indices[selected_sorted_indices]
    else:
        kept_indices = selected_sorted_indices

    return deduped_coords, kept_indices


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
    ...     seed_from_grid,
    ...     seed_from_decomposition,
    ... )
    >>> from luxar.gsplats.seeds.utils import combine_seeds
    >>>
    >>> # Generate seeds from both methods (returns GSplatData)
    >>> result1 = seed_from_grid(image, spacing=5.0)
    >>> result2 = seed_from_decomposition(image, scales=[1, 2, 4, 8])
    >>>
    >>> # Combine centers with deduplication
    >>> combined = combine_seeds(result1.centers, result2.centers, min_distance=3.0)
    >>> n1, n2, nc = len(result1.centers), len(result2.centers), len(combined)
    >>> aprint(f"Combined: {n1} + {n2} → {nc} seeds")

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
                return cast(np.ndarray, np.zeros((0, arr.shape[1]), dtype=float))
        return cast(np.ndarray, np.zeros((0, 2), dtype=float))  # Default to 2D

    # Concatenate all non-empty arrays
    combined = np.vstack(non_empty)

    # Optionally deduplicate
    if min_distance is not None:
        combined, _ = dedupe_farthest_first(combined, min_distance=min_distance)

    return combined
