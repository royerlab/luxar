"""GSplat spatial ordering: compound sort + ellipsoidal chunk bounds."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np

from ...typing_utils.constants import DEFAULT_TRUNCATION_RADIUS
from .bounds import _BARRIER_BOUND_EPS
from .compound import _compound_sort


def sort_splats_spatial(
    centers: np.ndarray,
    method: Literal["morton", "hilbert"] = "hilbert",
    resolution: Optional[int] = None,
    slice_dims: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, dict]:
    """Sort GSplats using barrier-aware compound spatial ordering.

    When ``slice_dims`` names one or more categorical/barrier axes (e.g. time or
    channel), splats are grouped by those axes first (lexicographic) and a
    Morton/Hilbert curve orders spatially within each barrier value — so a chunk
    never straddles two timepoints. This mirrors :func:`sort_points_compound`
    and is what keeps per-timepoint reads local (see the ``io`` ordering README).

    With ``slice_dims`` empty/None this is pure spatial ordering over all center
    columns — the historical behavior, unchanged for 3D data.

    Args:
        centers: Splat centers, shape (N, d), float32
        method: Spatial curve method ("morton" or "hilbert")
        resolution: Ignored (kept for signature stability; the grid resolution
            is derived per-axis from the bit budget, as it always has been).
        slice_dims: Barrier/categorical column indices to order by first.

    Returns:
        sort_indices: Indices to reorder splats
        metadata: Dict with ordering metadata (incl. slice_dims / ordering_dims)
    """
    ndim = centers.shape[1]
    slice_set = set(int(d) for d in slice_dims) if slice_dims else set()
    ordering_dims = [d for d in range(ndim) if d not in slice_set]
    return _compound_sort(centers, sorted(slice_set), ordering_dims, method)


def compute_chunk_bounds_gsplats(
    centers: np.ndarray,
    cholesky_factors: np.ndarray,
    chunk_size: int,
    coverage_sigma: float = DEFAULT_TRUNCATION_RADIUS,
    slice_dims: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Compute chunk bounding boxes for GSplats (includes ellipsoidal extent).

    CRITICAL: the ellipsoidal (coverage_sigma·σ) extent is applied only to
    SPATIAL axes. ``slice_dims`` name categorical/barrier axes (time, channel):
    a splat at time=0 must not extend into time=1's bounds, so those axes get
    only a tight float-boundary epsilon (``_BARRIER_BOUND_EPS``). This mirrors
    :func:`compute_chunk_bounds_points` and keeps a chunk's barrier-axis footprint
    from straddling categories, which is what makes single-timepoint queries fetch
    only their own chunks.

    Args:
        centers: Splat centers (already sorted), shape (N, d)
        cholesky_factors: Packed Cholesky factors (already sorted), shape
            (N, k), or a single shared row (1, k) reused for every chunk when
            all splats have a uniform (identical) Cholesky factorization
        chunk_size: Number of splats per chunk
        coverage_sigma: Coverage radius in standard deviations. This is the
            gsplat ``truncation_radius`` under its spatial-ordering name; the
            compiler binds the two in ``gsplat_tree.py``.
        slice_dims: Barrier/categorical dimension indices (no σ expansion).
            Default None → expand all axes (historical behavior).

    Returns:
        chunk_bounds: Bounding boxes, shape (num_chunks, d, 2)
    """
    n_splats, ndim = centers.shape
    if n_splats == 0:
        return np.zeros((0, ndim, 2), dtype=np.float32)
    num_chunks = (n_splats + chunk_size - 1) // chunk_size
    discrete_dims = set(int(d) for d in slice_dims) if slice_dims else set()

    # Uniform-Cholesky convenience: a single packed row (shape (1, k)) is shared
    # by all splats and never expanded to (N, k). It must be used for every
    # chunk — positional slicing would yield an empty (0, k) array for any chunk
    # after the first and break broadcasting. The (1, k) row broadcasts cleanly
    # against the (chunk_len,) extents accumulator.
    uniform_cholesky = cholesky_factors.shape[0] == 1 and n_splats > 1

    chunk_bounds = np.zeros((num_chunks, ndim, 2), dtype=np.float32)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_splats)

        chunk_centers = centers[start_idx:end_idx]
        chunk_cholesky = (
            cholesky_factors
            if uniform_cholesky
            else cholesky_factors[start_idx:end_idx]
        )

        # Compute ellipsoidal extent (per spec: extent[d] = sqrt(covariance[d,d]) * 3σ)
        extents = np.zeros((end_idx - start_idx, ndim), dtype=np.float32)

        for d in range(ndim):
            if d in discrete_dims:
                # BARRIER dimension: no σ expansion (a category has no extent).
                continue
            # Covariance diagonal from Cholesky factors
            # For dimension d: start_idx = d*(d+1)//2
            start_chol_idx = d * (d + 1) // 2
            # covariance[d,d] = sum(L[start_chol_idx + i]^2 for i in 0..d)
            for i in range(d + 1):
                extents[:, d] += chunk_cholesky[:, start_chol_idx + i] ** 2

            # Extent = sqrt(covariance) * coverage_sigma
            extents[:, d] = np.sqrt(extents[:, d]) * coverage_sigma

        # Compute bounds including extent (barrier dims have zero extent).
        mins = (chunk_centers - extents).min(axis=0)
        maxs = (chunk_centers + extents).max(axis=0)

        # Barrier dims: tight bounds (exact category values) padded only by a
        # float-boundary epsilon — the reader's tolerance owns the query reach.
        # Mirrors compute_chunk_bounds_points.
        for d in discrete_dims:
            mins[d] = chunk_centers[:, d].min() - _BARRIER_BOUND_EPS
            maxs[d] = chunk_centers[:, d].max() + _BARRIER_BOUND_EPS

        chunk_bounds[chunk_idx, :, 0] = mins
        chunk_bounds[chunk_idx, :, 1] = maxs

    return chunk_bounds
