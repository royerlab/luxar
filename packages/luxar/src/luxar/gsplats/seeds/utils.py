# utils.py
"""
Shared utility functions for seed generation.

This module contains common utilities used by multiple seed generation methods,
including peak detection, spatial deduplication algorithms, and Cholesky
factor construction for Gaussian initialization. The spatial hash grid for
proximity queries lives in :mod:`luxar.gsplats.spatial_hash` and is re-exported
here for intra-package convenience.
"""

from typing import Optional, cast

import numpy as np
from scipy import ndimage as ndi

from luxar.gsplats.spatial_hash import SpatialHashGrid

__all__ = [
    "SpatialHashGrid",
    "SEED_AMPLITUDE_SCALE",
    "argsort_descending",
    "greedy_min_distance_select",
]

# Amplitude scaling factor for seed initialization.
# Multiplying by 0.9 (90%) helps avoid initial over-prediction when splats overlap,
# which can trigger the asymmetric over-prediction penalty and cause divergence.
# Starting slightly below the target intensity allows the optimizer to increase
# amplitudes as needed rather than fighting against penalty gradients.
SEED_AMPLITUDE_SCALE = 0.9


def argsort_descending(values: np.ndarray) -> np.ndarray:
    """Indices ordering ``values`` brightest first, ties kept in input order.

    NumPy's default ``argsort`` is an unstable introsort whose SIMD backends
    order equal keys differently on x86 and arm64, so ``argsort(v)[::-1]`` gave
    platform-dependent seed orders (and, through the order-dependent dedup and
    top-up steps, platform-dependent fits) whenever intensities tie. This is a
    stable O(N log N) sort that breaks every tie by input index. NaN sorts
    LAST (``argsort(v)[::-1]`` put it first): a NaN is never the brightest.

    Floats are sorted on their negation, which is exact. Integers and bools are
    not negated (``-int8(-128)`` and ``-int64.min`` wrap, and no float type
    holds every int64/uint64 exactly): the input is reversed, sorted ascending
    stably, and the order reversed back, which leaves ties in input order.
    """
    v = np.asarray(values)
    if v.dtype.kind == "f":
        return np.argsort(-v, kind="stable")
    order: np.ndarray = (len(v) - 1) - np.argsort(v[::-1], kind="stable")[::-1]
    return order


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
    row by row: [L00, L10, L11, L20, L21, L22, ...].
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
    threshold_abs: Optional[float] = None,
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
        Ignored when ``threshold_abs`` is given.
    blur : bool, default=True
        Apply soft blur before counting to reduce noise.
    threshold_abs : float, optional
        Absolute peak threshold on the (blurred) field. When given it overrides
        ``threshold_rel * max`` — used to count maxima at a *shared* level across
        crops so counts compose (a per-crop relative level does not).

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

    thresh = (
        float(threshold_abs) if threshold_abs is not None else threshold_rel * max_val
    )
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
        # Keep the top_k strongest peaks (ties by C-order index), returned in
        # the same C-order as the untruncated result.
        keep = np.sort(argsort_descending(vals)[:top_k])
        coords = coords[keep]
    return cast(np.ndarray, coords)


# Block sizing for :func:`greedy_min_distance_select`. A block's survivors are
# resolved from their candidate-candidate conflict pairs, whose count is
# survivors x degree / 2: degree ~26 at min_distance=2 on a dense lattice, but
# ~2,100 at min_distance=8, where a fixed 65,536 block built 70M pairs (6 GB).
# The degree is ESTIMATED per block from a small sample of the survivors
# (counts only) before any pair list exists, and the survivors handed to
# query_pairs are capped so the list stays near _GREEDY_PAIR_BUDGET pairs;
# sizing from the previous block is not enough, since an ordered lattice
# walks from sparse survivors straight into fresh, dense territory. Past
# _GREEDY_LOOP_DEGREE the per-point loop (which only ever checks KEPT points,
# sparse by construction) is cheaper and takes over.
_GREEDY_BLOCK_MIN = 1024
_GREEDY_BLOCK_MAX = 65536
_GREEDY_PAIR_BUDGET = 500_000
_GREEDY_LOOP_DEGREE = 512.0
_GREEDY_DEGREE_SAMPLE = 256


def greedy_min_distance_select(
    coords: np.ndarray,
    min_distance: float,
    max_count: Optional[int] = None,
) -> np.ndarray:
    """Greedy in-order selection under a strict minimum-distance constraint.

    Walks ``coords`` in the given (priority) order and keeps a point iff no
    previously KEPT point lies strictly closer than ``min_distance``, stopping
    once ``max_count`` points are kept. Returns the kept row indices, in order.
    ``max_count=0`` keeps nothing.

    This is exactly the result of the former per-point loop over a
    :class:`SpatialHashGrid` (``has_neighbor_within`` then ``insert``), computed
    block-wise instead of point-wise: each block of candidates is first tested
    against everything kept so far with one KD-tree range query, and only the
    survivors that conflict WITH EACH OTHER are resolved in a short sequential
    pass over their (precomputed) conflict lists. Each block's conflict degree
    is estimated from a small sample before any pair list is built, and only
    as many survivors as keep that list near ``_GREEDY_PAIR_BUDGET`` pairs are
    resolved per pass; when the estimated degree exceeds
    ``_GREEDY_LOOP_DEGREE`` (a large ``min_distance`` on dense candidates), the
    remaining candidates go through the per-point loop instead. The candidate order — and so the
    intensity priority callers sort by — is preserved, so the kept set is
    identical; the vectorised part evaluates distances in float64 rather than
    float32, which can only matter for a pair within float32 rounding of
    exactly ``min_distance``.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)
    limit = n if max_count is None else max(0, min(int(max_count), n))
    if n == 0 or limit == 0:
        return np.empty(0, dtype=np.intp)
    if not min_distance > 0:  # nothing can be "strictly closer" than <= 0
        return np.arange(limit, dtype=np.intp)

    # query_pairs keeps d <= r; the constraint is strict (d < min_distance).
    r_pairs = float(np.nextafter(min_distance, 0.0))
    kept: list[np.ndarray] = []
    n_kept = 0
    kept_tree = None
    block_size = 2048
    start = 0
    while start < n and n_kept < limit:
        end = min(start + block_size, n)
        block = np.arange(start, end, dtype=np.intp)
        if kept_tree is not None:
            d, _ = kept_tree.query(pts[block], k=1, distance_upper_bound=min_distance)
            block = block[~(d < min_distance)]
        if len(block) == 0:
            start = end
            continue
        # Estimate the survivors' conflict degree from a cheap sample BEFORE
        # building any pair list: the previous block says nothing reliable
        # about this one (an ordered lattice walks into fresh territory).
        degree = _sampled_conflict_degree(pts[block], r_pairs)
        if degree > _GREEDY_LOOP_DEGREE:
            loop_kept = _greedy_loop(
                coords, _concat(kept), start, min_distance, limit - n_kept
            )
            kept.append(loop_kept)
            break
        cap = max(int(2 * _GREEDY_PAIR_BUDGET / max(degree, 1.0)), 1)
        if len(block) > cap:
            block = block[:cap]
            end = int(block[-1]) + 1  # later candidates are re-tested next pass
        chosen, _ = _greedy_within_block(pts[block], r_pairs)
        sel = block[chosen][: limit - n_kept]
        kept.append(sel)
        n_kept += len(sel)
        start = end
        kept_tree = cKDTree(pts[_concat(kept)])
        block_size = int(
            np.clip(
                cap,
                _GREEDY_BLOCK_MIN,
                min(4 * block_size, _GREEDY_BLOCK_MAX),
            )
        )
    return _concat(kept)


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate index arrays (empty-safe)."""
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.intp)


def _sampled_conflict_degree(bpts: np.ndarray, r_pairs: float) -> float:
    """Mean number of OTHER points within ``r_pairs`` of a point of ``bpts``,
    estimated from up to ``_GREEDY_DEGREE_SAMPLE`` evenly spaced points (counts
    only; no neighbour lists are built)."""
    from scipy.spatial import cKDTree

    m = len(bpts)
    if m < 2:
        return 0.0
    probe = np.linspace(0, m - 1, num=min(m, _GREEDY_DEGREE_SAMPLE)).astype(np.intp)
    counts = cKDTree(bpts).query_ball_point(bpts[probe], r_pairs, return_length=True)
    return float(np.mean(counts)) - 1.0


def _greedy_loop(
    coords: np.ndarray,
    kept_so_far: np.ndarray,
    start: int,
    min_distance: float,
    budget: int,
) -> np.ndarray:
    """The original per-point greedy over ``coords[start:]``, seeded with the
    points kept so far; used where conflicts are too dense to vectorise."""
    pts32 = np.asarray(coords, dtype=np.float32)
    grid = SpatialHashGrid(cell_size=min_distance, ndim=pts32.shape[1])
    for i in kept_so_far.tolist():
        grid.insert(pts32[i])
    out: list[int] = []
    for i in range(start, len(pts32)):
        if not grid.has_neighbor_within(pts32[i], min_distance):
            grid.insert(pts32[i])
            out.append(i)
            if len(out) >= budget:
                break
    return np.asarray(out, dtype=np.intp)


def _greedy_within_block(bpts: np.ndarray, r_pairs: float) -> tuple[np.ndarray, int]:
    """Positions (in order) kept by the greedy rule among ``bpts`` alone, and
    the number of conflict pairs found.

    A point with no conflicting neighbour in the block is always kept; only the
    conflicted ones need the sequential pass, which visits them in order and
    discards the later neighbours of every point it keeps.
    """
    from scipy.spatial import cKDTree

    m = len(bpts)
    pairs = cKDTree(bpts).query_pairs(r_pairs, output_type="ndarray")
    keep = np.ones(m, dtype=bool)
    if len(pairs) == 0:
        return np.flatnonzero(keep), 0
    # Directed edges earlier -> later, as CSR over the earlier endpoint.
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    n_pairs = len(pairs)
    del pairs
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], hi[order]
    del order
    starts = np.searchsorted(lo, np.arange(m + 1))
    for i in np.unique(np.concatenate([lo, hi])).tolist():
        if keep[i]:
            keep[hi[starts[i] : starts[i + 1]]] = False
    return np.flatnonzero(keep), n_pairs


def dedupe_farthest_first(
    coords: np.ndarray,
    min_distance: float,
    intensities: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove duplicate seeds using greedy selection with min_distance constraint.

    Uses block-vectorised KD-tree range tests
    (:func:`greedy_min_distance_select`).

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
    - Time complexity: O(N log N), block-vectorised KD-tree range tests
      (:func:`greedy_min_distance_select`)
    - Space complexity: O(N) for the per-block conflict lists
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

    # Sort by intensity if provided (highest first), otherwise keep original order
    if intensities is not None:
        sort_indices = argsort_descending(intensities)
        coords_sorted = coords[sort_indices].astype(np.float32)
    else:
        sort_indices = None
        coords_sorted = coords.astype(np.float32)

    # Greedy in priority order, vectorised block-wise (same kept set as the
    # former per-seed SpatialHashGrid loop; see greedy_min_distance_select).
    selected_sorted_indices = greedy_min_distance_select(coords_sorted, min_distance)
    deduped_coords = coords_sorted[selected_sorted_indices].astype(float)

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
        sort_indices = argsort_descending(intensities)
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
