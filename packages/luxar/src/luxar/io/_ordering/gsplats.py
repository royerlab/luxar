"""GSplat spatial ordering: compound sort + ellipsoidal chunk bounds."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np

from ...typing_utils.constants import DEFAULT_TRUNCATION_RADIUS
from .bounds import (
    _BARRIER_BOUND_EPS,
    _normalise_slice_dims,
    _store_outward_f32_array,
)
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
        slice_dims: Barrier/categorical column indices to order by first. An
            index outside ``[0, d)`` raises ``ValueError`` (see
            :func:`_normalise_slice_dims`) — the SAME check
            :func:`compute_chunk_bounds_gsplats` applies, so the failure lands at
            the first door. ``apply_gsplat_spatial_ordering`` hands the same list
            to both, and a negative index is a genuine sort-side bug of its own
            (it lands in the barrier set AND in ``ordering_dims``) — see
            :func:`_normalise_slice_dims` for the full argument.

    Returns:
        sort_indices: Indices to reorder splats
        metadata: Dict with ordering metadata (incl. slice_dims / ordering_dims)

    Raises:
        ValueError: If a ``slice_dims`` entry is outside ``[0, d)``.
    """
    ndim = centers.shape[1]
    slice_set = _normalise_slice_dims(slice_dims, ndim)
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

    The extent is accumulated in float64 and narrowed to the float32 store with
    OUTWARD rounding (see :func:`_store_outward_f32_array`), so a stored bound is
    never tighter than the footprint at any coordinate magnitude — not only where
    a small σ happens to survive float32 arithmetic and a round-to-nearest store.

    That guarantee is against the AUTHORED centers. Centers are themselves stored
    as per-axis uint16 fixed point under the default AUTO encoding, so a decoded
    center could in principle sit half a quantum outside its chunk's bound —
    except that gsplats have a rail for exactly that:
    ``_compiler/gsplat_assembly.py::_axis_center_offender`` escalates the centers
    array to float32 when half an axis's grid step exceeds the per-splat marginal
    σ for more than 0.1% of the splats, and a gridded (stacked time/channel) axis
    is snapped to store exactly. Points and Lines have no equivalent rail — see
    the note on :func:`~luxar.io._ordering.points.compute_chunk_bounds_points`.

    Args:
        centers: Splat centers (already sorted), shape (N, d)
        cholesky_factors: Packed Cholesky factors (already sorted), shape
            (N, k), or a single shared row (1, k) reused for every chunk when
            all splats have a uniform (identical) Cholesky factorization
        chunk_size: Number of splats per chunk
        coverage_sigma: Coverage radius in standard deviations. This is the
            gsplat ``truncation_radius`` under its spatial-ordering name — the
            same quantity the LOD path spells ``truncation_sigmas``. Two compiler
            sites bind it to the dataset's own radius:
            ``io/_compiler/gsplat_tree.py::_write_single_splat_set`` (by keyword)
            and the scene path ``geometry_writers/gsplats.py`` →
            ``gsplat_assembly.py::apply_gsplat_spatial_ordering``, which passes
            ``truncation_radius`` positionally. The default here only applies to
            a direct call.
        slice_dims: Barrier/categorical dimension indices (no σ expansion).
            Default None → expand all axes (historical behavior). An index
            outside ``[0, d)`` raises ``ValueError`` (see
            :func:`_normalise_slice_dims`).

    Returns:
        chunk_bounds: Bounding boxes, shape (num_chunks, d, 2)
    """
    n_splats, ndim = centers.shape
    discrete_dims = _normalise_slice_dims(slice_dims, ndim)
    if n_splats == 0:
        return np.zeros((0, ndim, 2), dtype=np.float32)
    num_chunks = (n_splats + chunk_size - 1) // chunk_size

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

        # float64 throughout: a small σ added to a large center is lost outright
        # in float32 arithmetic, before the outward store below can rescue it.
        chunk_centers = centers[start_idx:end_idx].astype(np.float64, copy=False)
        chunk_cholesky = (
            cholesky_factors
            if uniform_cholesky
            else cholesky_factors[start_idx:end_idx]
        ).astype(np.float64, copy=False)

        # Ellipsoidal extent (per spec:
        # extent[d] = sqrt(covariance[d,d]) * coverage_sigma)
        extents = np.zeros((end_idx - start_idx, ndim), dtype=np.float64)

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

        lo32, hi32 = _store_outward_f32_array(mins, maxs)
        chunk_bounds[chunk_idx, :, 0] = lo32
        chunk_bounds[chunk_idx, :, 1] = hi32

    return chunk_bounds
