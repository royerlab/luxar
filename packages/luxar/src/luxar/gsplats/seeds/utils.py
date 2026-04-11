# utils.py
"""
Shared utility functions for seed generation.

This module contains common utilities used by multiple seed generation methods,
including peak detection, spatial deduplication algorithms, Cholesky factor
construction for Gaussian initialization, and spatial hash grid for fast
proximity queries.
"""

import itertools
import math
from typing import Optional, cast

import numpy as np
from scipy import ndimage as ndi

# Amplitude scaling factor for seed initialization.
# Multiplying by 0.9 (90%) helps avoid initial over-prediction when splats overlap,
# which can trigger the asymmetric over-prediction penalty and cause divergence.
# Starting slightly below the target intensity allows the optimizer to increase
# amplitudes as needed rather than fighting against penalty gradients.
SEED_AMPLITUDE_SCALE = 0.9


class SpatialHashGrid:
    """
    Spatial hash grid for O(1) amortized proximity queries on nD points.

    Points are hashed into cells of a given size. Proximity queries check
    only the 3^ndim neighboring cells, giving O(1) amortized cost per query
    (assuming bounded density per cell). This replaces KD-tree approaches
    that require O(M log M) rebuilds.

    Parameters
    ----------
    cell_size : float
        Size of each grid cell. Must be >= the query distance used in
        ``has_neighbor_within`` for correctness (the 3^ndim neighbor check
        only guarantees finding all points within ``cell_size``).
    ndim : int
        Number of spatial dimensions.

    Notes
    -----
    Correctness guarantee: if ``cell_size >= distance``, then for any query
    point p and stored point q with ||p - q|| < distance, q is guaranteed
    to be in the same cell or an adjacent cell (within 1 cell offset in
    each dimension). Proof: ||p - q|| < distance <= cell_size implies
    |p[d] - q[d]| < cell_size for each dimension d.
    """

    def __init__(self, cell_size: float, ndim: int):
        self._cell_size = max(cell_size, 1e-10)
        self._inv_cell_size = 1.0 / self._cell_size
        self._ndim = ndim
        self._grid: dict[tuple[int, ...], list[int]] = {}
        # Pre-allocate with doubling growth
        self._points = np.empty((64, ndim), dtype=np.float32)
        self._n_points = 0
        self._neighbor_offsets = list(
            itertools.product([-1, 0, 1], repeat=ndim)
        )

    def _cell_key(self, point: np.ndarray) -> tuple[int, ...]:
        """Compute the grid cell key for a point.

        Uses ``math.floor`` (not ``int()``) so that negative coordinates
        are handled correctly.  ``int()`` truncates toward zero, which
        would map e.g. -0.5 and +0.5 to the same cell 0.
        """
        return tuple(math.floor(x) for x in (point * self._inv_cell_size))

    def insert(self, point: np.ndarray) -> int:
        """
        Insert a point into the grid.

        Parameters
        ----------
        point : np.ndarray
            Point coordinates, shape (ndim,).

        Returns
        -------
        int
            Index of the inserted point.
        """
        idx = self._n_points
        # Grow backing array if needed (doubling strategy)
        if idx >= len(self._points):
            new_size = len(self._points) * 2
            new_arr = np.empty((new_size, self._ndim), dtype=np.float32)
            new_arr[:idx] = self._points[:idx]
            self._points = new_arr
        self._points[idx] = point
        self._n_points += 1

        key = self._cell_key(point)
        if key in self._grid:
            self._grid[key].append(idx)
        else:
            self._grid[key] = [idx]
        return idx

    def has_neighbor_within(self, point: np.ndarray, distance: float) -> bool:
        """
        Check if any stored point is within the given distance.

        Parameters
        ----------
        point : np.ndarray
            Query point coordinates, shape (ndim,).
        distance : float
            Maximum distance threshold. Must be <= cell_size for the
            3^ndim neighbor check to be correct.

        Returns
        -------
        bool
            True if any stored point is strictly closer than ``distance``.
        """
        dist_sq = distance * distance
        cell_key = self._cell_key(point)
        for offset in self._neighbor_offsets:
            key = tuple(cell_key[d] + offset[d] for d in range(self._ndim))
            bucket = self._grid.get(key)
            if bucket is not None:
                for j in bucket:
                    diff = self._points[j] - point
                    if np.dot(diff, diff) < dist_sq:
                        return True
        return False

    @property
    def points(self) -> np.ndarray:
        """Return a copy of all stored points, shape (n_points, ndim)."""
        result: np.ndarray = self._points[: self._n_points].copy()
        return result

    def __len__(self) -> int:
        """Return the number of stored points."""
        return self._n_points


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
    coords: np.ndarray,
    min_distance: float,
    intensities: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove duplicate seeds using greedy selection with min_distance constraint.

    Uses a spatial hash grid for O(1) amortized distance queries.

    Algorithm:
    1. Sort seeds by intensity (if provided) or keep original order
    2. Start with strongest/first seed
    3. Iterate through remaining seeds in order
    4. Keep seed if distance >= min_distance from all selected seeds
    5. Repeat until all seeds processed

    This simple greedy approach is ~50x faster than farthest-first selection
    and produces equivalent results for Gaussian splatting, since the optimizer
    will adjust positions during fitting anyway.

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
    device : str, optional
        Device parameter (ignored for deduplication).
        Deduplication always uses CPU as it's fastest for typical seed counts.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (deduped_coords, kept_indices) where:
        - deduped_coords: Array of deduplicated coordinates (M, ndim)
        - kept_indices: Indices into original coords array (M,)
        This allows O(1) indexing into other arrays instead of O(M×N) coordinate matching.

    Notes
    -----
    Implementation:
    - Time complexity: O(N) amortized via spatial hash grid
    - Space complexity: O(M) where M is the number of selected seeds
    - For small datasets (<50 seeds), uses simple O(N²) greedy fallback

    Examples
    --------
    >>> # Basic usage
    >>> deduped, indices = dedupe_farthest_first(coords, min_distance=2.0)

    >>> # With intensity prioritization
    >>> deduped, indices = dedupe_farthest_first(coords, min_distance=2.0, intensities=amps)
    """
    # Handle empty input case
    if len(coords) == 0:
        return coords.astype(float), np.array([], dtype=np.intp)

    # For very small inputs, use simple greedy (overhead not worth it)
    if len(coords) < 50:
        return _dedupe_simple(coords, min_distance, intensities)

    ndim = coords.shape[1]

    # Sort by intensity if provided (highest first), otherwise keep original order
    if intensities is not None:
        sort_indices = np.argsort(intensities)[::-1]  # Descending order
        coords_sorted = coords[sort_indices].astype(np.float32)
    else:
        sort_indices = None
        coords_sorted = coords.astype(np.float32)

    # Spatial hash grid for O(1) amortized distance checks
    grid = SpatialHashGrid(cell_size=min_distance, ndim=ndim)
    selected_sorted_indices_list: list[int] = []

    for idx in range(len(coords_sorted)):
        coord = coords_sorted[idx]

        if not grid.has_neighbor_within(coord, min_distance):
            grid.insert(coord)
            selected_sorted_indices_list.append(idx)

    # Build output arrays
    deduped_coords = grid.points.astype(float)
    selected_sorted_indices = np.array(selected_sorted_indices_list, dtype=np.intp)

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
    selected_sorted_indices_arr = np.array(selected_sorted_indices, dtype=np.intp)

    # Map back to original indices if we sorted by intensity
    if sort_indices is not None:
        kept_indices = sort_indices[selected_sorted_indices_arr]
    else:
        kept_indices = selected_sorted_indices_arr

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
