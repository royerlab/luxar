"""Compound-ordering core shared by every geometry.

``_compound_sort`` is the single primitive underneath the Points, GSplats, and
Lines sorts: lexsort categorical/barrier axes first, then a space-filling curve
within each barrier value. ``detect_barrier_dims`` heuristically infers the
barrier axes for a provenance-less standalone ``.gsplats.zarr``.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from .curves.hilbert import hilbert_encode_nd
from .curves.morton import morton_encode_nd
from .grid import normalize_coords_to_grid

_DEFAULT_BARRIER_MAX_CARDINALITY = 1024


def _rounded_barrier_values(values: np.ndarray) -> np.ndarray | None:
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-3):
        return None
    return rounded.astype(np.int64)


def _barrier_axis_qualifies(n_splats: int, n_unique: int, max_cardinality: int) -> bool:
    return n_unique <= max_cardinality and n_unique * 4 <= n_splats


def _compound_sort(
    coords: np.ndarray,
    slice_dims: Sequence[int],
    ordering_dims: Sequence[int],
    method: Literal["morton", "hilbert"] = "hilbert",
) -> tuple[np.ndarray, dict]:
    """Compound spatial ordering: lexsort by ``slice_dims`` (categorical/barrier
    axes) first, then a Morton/Hilbert space-filling curve over ``ordering_dims``
    (spatial axes) within each barrier value.

    This is the single source of truth shared by :func:`sort_points_compound`
    (barrier = discrete non-display dims) and :func:`sort_splats_spatial`
    (barrier = the LOD/categorical dims). Keeping the two geometries on one
    primitive guarantees a chunk never straddles a categorical value regardless
    of geometry type. With ``slice_dims == []`` this reduces to pure spatial
    ordering over all of ``ordering_dims`` (the historical GSplat behavior).

    Args:
        coords: Coordinates, shape (N, d).
        slice_dims: Barrier/categorical column indices (lexsorted first).
        ordering_dims: Spatial column indices (space-filling curve within).
        method: "morton" or "hilbert".

    Returns:
        (sort_indices, metadata) — the same metadata schema Points/Lines emit.
    """
    n = coords.shape[0]
    slice_dims = list(slice_dims)
    ordering_dims = list(ordering_dims)

    # Bits budget is split across ONLY the spatial (ordering) dims, so excluding
    # a barrier axis gives the spatial axes more resolution too.
    if ordering_dims:
        bits_per_dim = min(21, 64 // len(ordering_dims))
    else:
        bits_per_dim = 21  # Fallback (all-discrete)

    if ordering_dims:
        ordering_coords = coords[:, ordering_dims]
        ordering_min = ordering_coords.min(axis=0)
        ordering_max = ordering_coords.max(axis=0)

        grid_coords = normalize_coords_to_grid(
            ordering_coords, ordering_min, ordering_max, 2**bits_per_dim
        )

        if method == "morton":
            spatial_codes = morton_encode_nd(grid_coords, bits_per_dim)
        elif method == "hilbert":
            spatial_codes = hilbert_encode_nd(grid_coords, bits_per_dim)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # No ordering dimensions - all discrete
        spatial_codes = np.zeros(n, dtype=np.uint64)
        ordering_min = np.array([])
        ordering_max = np.array([])

    if slice_dims:
        # Lexicographic sort on barrier dims, then spatial code within each.
        slice_values = coords[:, slice_dims]
        sort_indices = np.lexsort(
            [spatial_codes]
            + [slice_values[:, i] for i in range(len(slice_dims) - 1, -1, -1)]
        )
    else:
        # Pure spatial ordering (no barrier dimensions)
        sort_indices = np.argsort(spatial_codes)

    metadata = {
        "ordering": method,
        "slice_dims": slice_dims,
        "ordering_dims": ordering_dims,
        "ordering_min": ordering_min.tolist() if len(ordering_min) > 0 else [],
        "ordering_max": ordering_max.tolist() if len(ordering_max) > 0 else [],
        "ordering_bits_per_dim": bits_per_dim,
    }

    return sort_indices, metadata


def detect_barrier_dims(
    centers: np.ndarray,
    max_cardinality: int = _DEFAULT_BARRIER_MAX_CARDINALITY,
) -> list[int]:
    """Heuristically identify categorical/barrier axes in a GSplat center array.

    A standalone ``.gsplats.zarr`` carries no per-dimension descriptors, so when
    neither an explicit ``barrier_dims`` nor persisted ``coarsen_dims`` provenance
    is available this conservatively infers which axes behave like a categorical
    stack (time, channel): an axis qualifies iff its values are integers AND take
    few distinct values (``<= max_cardinality`` and ``<< N``). Continuous spatial
    float coordinates never qualify.

    **Conservative in the safe direction.** A false NEGATIVE (missing a barrier)
    only causes over-fetch — no worse than pure spatial ordering. A false
    POSITIVE (flagging a spatial axis) gives it tight (epsilon-padded) chunk
    bounds with no σ expansion, so a spatially-extended splat can fall outside its chunk bounds
    and be *dropped* from a query — a correctness bug. So both guards err toward
    NOT flagging: the integer test is strict (``rtol=0``, absolute tolerance
    only — a large-magnitude continuous coordinate is never "close enough" to an
    integer), and the ``n_unique * 4 <= n`` guard rejects a fine integer spatial
    grid (many distinct values relative to N) that is not a true category.

    Subordinate by design: callers apply explicit ``barrier_dims`` and
    ``coarsen_dims`` complements first (the scene compiler passes scene
    ``Dimension.discrete`` dims; the batch merge passes the stacked-time axis),
    using this only as the last resort for provenance-less standalone files.

    Args:
        centers: Splat centers, shape (N, d).
        max_cardinality: Max distinct values for an axis to count as categorical.

    Returns:
        Sorted list of barrier column indices (possibly empty).
    """
    if centers.ndim != 2 or centers.shape[0] == 0:
        return []
    n, ndim = centers.shape
    barrier: list[int] = []
    for d in range(ndim):
        col = centers[:, d]
        # Must lie on an integer grid (categorical stacks are integer-labelled).
        # rtol=0: a large-magnitude continuous float must NOT count as integer
        # (np.allclose's default rtol=1e-5 makes |coord|>~5e4 always "integer",
        # which would misclassify a spatial axis → dropped splats).
        rounded = _rounded_barrier_values(col)
        if rounded is None:
            continue
        n_unique = int(np.unique(rounded).size)
        # Few distinct values, and materially fewer than N (so a genuinely
        # per-splat-varying axis — or a fine integer spatial grid — is never
        # mistaken for a category).
        if _barrier_axis_qualifies(n, n_unique, max_cardinality):
            barrier.append(d)
    return barrier
