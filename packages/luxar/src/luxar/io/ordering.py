"""Spatial ordering algorithms for Points, Lines, and GSplats.

This module provides Morton and Hilbert ordering for:
- Points: compound ordering for discrete dimensions
- Lines: dual ordering (vertices in D-space + segments in (2×D)-space)
- GSplats: simple spatial ordering
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import numpy as np

from luxar.core import Dimension

# Padding added to barrier/discrete-dimension chunk bounds. This is ONLY a
# float-boundary safety margin — the query "reach" (how far a slice query
# selects around a category) lives entirely in the reader's per-dimension
# tolerance (see luxar-viewer tolerance-computer.ts, discrete = 0.25 × step).
# It used to be 0.5 (half a step); combined with the reader's own half-step
# tolerance that summed to a full step and made a single-category query (e.g.
# one timepoint) pull in the entire neighbouring category. Keep this tiny.
_BARRIER_BOUND_EPS = 1e-3


def _get_morton_numba_kernel():  # type: ignore[no-untyped-def]
    """Lazy-compile the Numba Morton encoding kernel on first use."""
    import numba

    @numba.njit(cache=True)  # type: ignore[misc]
    def _morton_kernel(coords: np.ndarray, bits_per_dim: int, out: np.ndarray) -> None:
        n_points = coords.shape[0]
        n_dims = coords.shape[1]
        for idx in range(n_points):
            h = np.uint64(0)
            for bit in range(bits_per_dim):
                for d in range(n_dims):
                    h |= np.uint64((coords[idx, d] >> bit) & 1) << np.uint64(
                        bit * n_dims + d
                    )
            out[idx] = h

    return _morton_kernel


# None = not tried, False = tried and failed, callable = compiled kernel
_morton_numba_kernel: Any = None


def morton_encode_nd(coords: np.ndarray, bits_per_dim: int = 16) -> np.ndarray:
    """Encode nD integer coordinates to Morton codes via bit interleaving.

    Uses a Numba JIT-compiled kernel when available, falling back to
    vectorized NumPy.

    Args:
        coords: Integer coordinates, shape (N, d)
        bits_per_dim: Bits to use per dimension (default 16)

    Returns:
        Morton codes, shape (N,), dtype uint64
    """
    global _morton_numba_kernel  # noqa: PLW0603

    n_points, n_dims = coords.shape

    if _morton_numba_kernel is None:
        try:
            _morton_numba_kernel = _get_morton_numba_kernel()  # type: ignore[no-untyped-call]
        except (ImportError, Exception):
            _morton_numba_kernel = False

    if _morton_numba_kernel:
        out = np.empty(n_points, dtype=np.uint64)
        coords_i64 = np.ascontiguousarray(coords, dtype=np.int64)
        _morton_numba_kernel(coords_i64, bits_per_dim, out)
        return out

    # Fallback: vectorized NumPy
    morton = np.zeros(n_points, dtype=np.uint64)
    for bit in range(bits_per_dim):
        for dim in range(n_dims):
            coord_bit = (coords[:, dim] >> bit) & 1
            morton |= coord_bit.astype(np.uint64) << (bit * n_dims + dim)
    return morton


def _get_hilbert_numba_kernel():  # type: ignore[no-untyped-def]
    """Lazy-compile the Numba Hilbert encoding kernel on first use."""
    import numba

    @numba.njit(cache=True)  # type: ignore[misc]
    def _hilbert_kernel(coords: np.ndarray, bits_per_dim: int, out: np.ndarray) -> None:
        """Numba-accelerated Hilbert curve encoding.

        Implements the same algorithm as the hilbertcurve library
        (Skilling's "Programming the Hilbert curve") but compiled to
        native code and parallelised over points.
        """
        n_points = coords.shape[0]
        n_dims = coords.shape[1]
        m = np.int64(1) << np.int64(bits_per_dim - 1)

        for idx in range(n_points):
            # Copy point to local mutable array
            pt = np.empty(n_dims, dtype=np.int64)
            for d in range(n_dims):
                pt[d] = np.int64(coords[idx, d])

            # --- Inverse undo excess work ---
            q = m
            while q > 1:
                p = q - 1
                for i in range(n_dims):
                    if pt[i] & q:
                        pt[0] ^= p
                    else:
                        t = (pt[0] ^ pt[i]) & p
                        pt[0] ^= t
                        pt[i] ^= t
                q >>= 1

            # --- Gray encode ---
            for i in range(1, n_dims):
                pt[i] ^= pt[i - 1]

            t2 = np.int64(0)
            q = m
            while q > 1:
                if pt[n_dims - 1] & q:
                    t2 ^= q - 1
                q >>= 1

            for i in range(n_dims):
                pt[i] ^= t2

            # --- Transpose to Hilbert integer (MSB-first bit interleave) ---
            # Matches hilbertcurve library convention: MSB of dim 0 first.
            h = np.uint64(0)
            for bit in range(bits_per_dim - 1, -1, -1):
                for d in range(n_dims):
                    h = (h << np.uint64(1)) | np.uint64((pt[d] >> bit) & 1)

            out[idx] = h

    return _hilbert_kernel


# None = not tried, False = tried and failed, callable = compiled kernel
_hilbert_numba_kernel: Any = None


def hilbert_encode_nd(coords: np.ndarray, bits_per_dim: int = 16) -> np.ndarray:
    """Encode nD integer coordinates to Hilbert curve indices.

    Uses a Numba JIT-compiled kernel for fast parallel encoding.
    Falls back to the hilbertcurve library if Numba is unavailable.

    Args:
        coords: Integer coordinates, shape (N, d)
        bits_per_dim: Bits to use per dimension (default 16)

    Returns:
        Hilbert indices, shape (N,), dtype uint64
    """
    global _hilbert_numba_kernel  # noqa: PLW0603

    n_points, n_dims = coords.shape

    # Try Numba first (compiled, parallel, no memory overhead)
    if _hilbert_numba_kernel is None:
        try:
            _hilbert_numba_kernel = _get_hilbert_numba_kernel()  # type: ignore[no-untyped-call]
        except (ImportError, Exception):
            _hilbert_numba_kernel = False

    if _hilbert_numba_kernel:
        out = np.empty(n_points, dtype=np.uint64)
        _hilbert_numba_kernel(coords.astype(np.int64), bits_per_dim, out)
        return out

    # Fallback: hilbertcurve library (pure Python, slow for large N)
    try:
        from hilbertcurve.hilbertcurve import (  # type: ignore[import-untyped]
            HilbertCurve,
        )
    except ImportError:
        raise ImportError(
            "Either numba or hilbertcurve package is required for Hilbert ordering. "
            "Install with: pip install numba  (or: pip install hilbertcurve)"
        )

    hilbert = HilbertCurve(bits_per_dim, n_dims)
    hilbert_indices = np.array(
        [hilbert.distance_from_point(coords[i]) for i in range(n_points)],
        dtype=np.uint64,
    )
    return hilbert_indices


def normalize_coords_to_grid(
    coords: np.ndarray, min_coords: np.ndarray, max_coords: np.ndarray, resolution: int
) -> np.ndarray:
    """Normalize float coordinates to integer grid [0, resolution-1].

    Args:
        coords: Float coordinates, shape (N, d)
        min_coords: Minimum bounds, shape (d,)
        max_coords: Maximum bounds, shape (d,)
        resolution: Grid resolution (e.g., 2^16 = 65536)

    Returns:
        Integer coordinates, shape (N, d), dtype uint32
    """
    # Normalize to [0, 1]
    ranges = max_coords - min_coords
    # Handle degenerate dimensions (zero range)
    ranges = np.where(ranges > 0, ranges, 1.0)
    normalized = (coords - min_coords) / ranges

    # Clamp to [0, 1] (handle floating point errors)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Scale to [0, resolution-1]
    grid_coords = (normalized * (resolution - 1)).astype(np.uint32)

    return np.asarray(grid_coords)


def compute_auto_resolution(coords: np.ndarray, max_resolution: int = 2**16) -> int:
    """Compute appropriate resolution based on data spread.

    Args:
        coords: Float coordinates, shape (N, d)
        max_resolution: Maximum resolution (default 65536)

    Returns:
        Resolution as power of 2, capped at max_resolution
    """
    spread = coords.max(axis=0) - coords.min(axis=0)
    max_spread = spread.max()

    # Target ~10 grid cells per unit of spread
    target_resolution = int(max_spread * 10)

    # Clamp to [256, max_resolution]
    resolution = min(max_resolution, max(256, target_resolution))

    # Round to nearest power of 2
    resolution = int(2 ** int(np.log2(resolution)))

    return resolution


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
    max_cardinality: int = 1024,
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
        if not np.allclose(col, np.round(col), rtol=0.0, atol=1e-3):
            continue
        n_unique = int(np.unique(np.round(col).astype(np.int64)).size)
        # Few distinct values, and materially fewer than N (so a genuinely
        # per-splat-varying axis — or a fine integer spatial grid — is never
        # mistaken for a category).
        if n_unique <= max_cardinality and n_unique * 4 <= n:
            barrier.append(d)
    return barrier


def sort_points_compound(
    positions: np.ndarray,
    dimensions: list[Dimension],
    method: Literal["morton", "hilbert"] = "hilbert",
) -> tuple[np.ndarray, dict]:
    """Sort Points using compound ordering (discrete dims → spatial curve).

    This implements the compound ordering strategy from luxar.io spec:
    - Primary sort: Discrete dimensions (lexicographic)
    - Secondary sort: Morton/Hilbert code of spatial dimensions

    Args:
        positions: Point positions, shape (N, d), all dimensions
        dimensions: Dimension objects defining discrete/spatial properties
        method: Spatial curve method ("morton" or "hilbert"), default "hilbert"

    Returns:
        sort_indices: Indices to reorder points
        metadata: Dict with ordering metadata
    """
    # Identify dimension categories (per spec Section "Compound Ordering").
    slice_dims = [i for i, d in enumerate(dimensions) if d.discrete and not d.display]
    ordering_dims = [i for i, d in enumerate(dimensions) if not d.discrete or d.display]
    return _compound_sort(positions, slice_dims, ordering_dims, method)


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


def compute_chunk_bounds_points(
    positions: np.ndarray,
    radii: Optional[np.ndarray | float],
    chunk_size: int,
    slice_dims: Optional[list[int]] = None,
) -> np.ndarray:
    """Compute chunk bounding boxes for Points (includes radius extent).

    CRITICAL: Radius expansion is only applied to SPATIAL dimensions, not discrete
    dimensions. Discrete dimensions (slice_dims) represent categorical values like
    time steps, channels, or orbital indices. A point at orbital=0 should NOT
    extend into orbital=3 space - they are separate categories.

    Args:
        positions: Point positions (already sorted), shape (N, d)
        radii: Point radii (already sorted), shape (N,), or None
        chunk_size: Number of points per chunk
        slice_dims: Indices of discrete (non-spatial) dimensions where radius
                   expansion should NOT be applied. Default: None (apply to all dims)

    Returns:
        chunk_bounds: Bounding boxes, shape (num_chunks, d, 2)
    """
    n_points, ndim = positions.shape
    num_chunks = (n_points + chunk_size - 1) // chunk_size

    # Convert slice_dims to a set for fast lookup
    discrete_dims = set(slice_dims) if slice_dims else set()

    chunk_bounds = np.zeros((num_chunks, ndim, 2), dtype=np.float32)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_points)

        chunk_positions = positions[start_idx:end_idx]

        if radii is not None:
            radii_scalar: Optional[float] = None
            chunk_radii: Optional[np.ndarray] = None
            if isinstance(radii, np.ndarray):
                if radii.shape[0] == 1:
                    radii_scalar = float(radii.flat[0])
                else:
                    chunk_radii = radii[start_idx:end_idx]
            else:
                radii_scalar = float(radii)

            # Compute bounds for each dimension separately
            for d in range(ndim):
                if d in discrete_dims:
                    # DISCRETE dimension: No radius expansion!
                    # Just use exact min/max of coordinate values, padded only by a
                    # tiny float-boundary epsilon (the reader's tolerance owns the
                    # query reach — see _BARRIER_BOUND_EPS).
                    mins_d = chunk_positions[:, d].min() - _BARRIER_BOUND_EPS
                    maxs_d = chunk_positions[:, d].max() + _BARRIER_BOUND_EPS
                else:
                    # SPATIAL dimension: Include radius extent
                    if radii_scalar is not None:
                        mins_d = chunk_positions[:, d].min() - radii_scalar
                        maxs_d = chunk_positions[:, d].max() + radii_scalar
                    else:
                        assert chunk_radii is not None
                        mins_d = (chunk_positions[:, d] - chunk_radii).min()
                        maxs_d = (chunk_positions[:, d] + chunk_radii).max()

                chunk_bounds[chunk_idx, d, 0] = mins_d
                chunk_bounds[chunk_idx, d, 1] = maxs_d
        else:
            # No radii provided - add small safety margin to prevent missing points
            # at chunk boundaries when default radius is applied during rendering
            # Safety margin: 1% of coordinate range or 0.01, whichever is larger
            coord_range = chunk_positions.max(axis=0) - chunk_positions.min(axis=0)
            safety_margin = np.maximum(coord_range * 0.01, 0.01)

            # For discrete dims, use tighter bounds
            for d in range(ndim):
                if d in discrete_dims:
                    # Discrete: tight bounds with a float-boundary epsilon only.
                    chunk_bounds[chunk_idx, d, 0] = (
                        chunk_positions[:, d].min() - _BARRIER_BOUND_EPS
                    )
                    chunk_bounds[chunk_idx, d, 1] = (
                        chunk_positions[:, d].max() + _BARRIER_BOUND_EPS
                    )
                else:
                    # Spatial: include safety margin
                    chunk_bounds[chunk_idx, d, 0] = (
                        chunk_positions[:, d].min() - safety_margin[d]
                    )
                    chunk_bounds[chunk_idx, d, 1] = (
                        chunk_positions[:, d].max() + safety_margin[d]
                    )

    return chunk_bounds


def compute_chunk_bounds_gsplats(
    centers: np.ndarray,
    cholesky_factors: np.ndarray,
    chunk_size: int,
    coverage_sigma: float = 3.0,
    slice_dims: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Compute chunk bounding boxes for GSplats (includes ellipsoidal extent).

    CRITICAL: the ellipsoidal (coverage_sigma·σ) extent is applied only to
    SPATIAL axes. ``slice_dims`` name categorical/barrier axes (time, channel):
    a splat at time=0 must not extend into time=1's bounds, so those axes get
    only tight ±0.5 bounds. This mirrors :func:`compute_chunk_bounds_points`
    and keeps a chunk's barrier-axis footprint from straddling categories,
    which is what makes single-timepoint queries fetch only their own chunks.

    Args:
        centers: Splat centers (already sorted), shape (N, d)
        cholesky_factors: Packed Cholesky factors (already sorted), shape (N, k)
        chunk_size: Number of splats per chunk
        coverage_sigma: Coverage radius in standard deviations (default 3.0)
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

    chunk_bounds = np.zeros((num_chunks, ndim, 2), dtype=np.float32)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_splats)

        chunk_centers = centers[start_idx:end_idx]
        chunk_cholesky = cholesky_factors[start_idx:end_idx]

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


# ============================================================================
# Lines Spatial Ordering (Dual: vertices + segments)
# ============================================================================


def convert_to_indexed(
    n_vertices: int, line_type: str, indices: Optional[np.ndarray]
) -> np.ndarray:
    """Convert any line type to indexed segment pairs.

    All line types are internally converted to the unified indexed representation
    for efficient spatial ordering and storage.

    Args:
        n_vertices: Number of vertices in the lines
        line_type: One of "segments", "polyline", "loop", "indexed"
        indices: For "indexed" type, the user-provided index array

    Returns:
        segments: (S, 2) uint32 array of vertex index pairs

    Raises:
        ValueError: If line_type is invalid or indices missing for "indexed" type
    """
    if line_type == "indexed":
        if indices is None:
            raise ValueError("Indexed line type requires indices array")
        return indices.reshape(-1, 2).astype(np.uint32)
    elif line_type == "segments":
        # Each pair of consecutive vertices forms a segment
        return np.arange(n_vertices, dtype=np.uint32).reshape(-1, 2)
    elif line_type == "polyline":
        # Connect consecutive vertices: (0,1), (1,2), (2,3), ...
        return np.column_stack(
            [
                np.arange(n_vertices - 1, dtype=np.uint32),
                np.arange(1, n_vertices, dtype=np.uint32),
            ]
        )
    elif line_type == "loop":
        # Like polyline but connect last to first
        return np.column_stack(
            [
                np.arange(n_vertices, dtype=np.uint32),
                np.roll(np.arange(n_vertices, dtype=np.uint32), -1),
            ]
        )
    else:
        raise ValueError(
            f"Invalid line_type '{line_type}'. Must be one of: segments, polyline, loop, indexed"
        )


def morton_encode_128bit(
    coords: np.ndarray, bits_per_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Encode nD integer coordinates to 128-bit Morton codes as (high, low) pairs.

    For high-dimensional data (> 6 dims), 64-bit Morton codes have insufficient
    precision. This function produces 128-bit codes as paired uint64 values.

    Args:
        coords: Integer coordinates, shape (N, d)
        bits_per_dim: Bits to use per dimension

    Returns:
        high: Upper 64 bits of Morton codes, shape (N,), dtype uint64
        low: Lower 64 bits of Morton codes, shape (N,), dtype uint64
    """
    n_points, n_dims = coords.shape
    high = np.zeros(n_points, dtype=np.uint64)
    low = np.zeros(n_points, dtype=np.uint64)

    for bit in range(bits_per_dim):
        for dim in range(n_dims):
            coord_bit = (coords[:, dim] >> bit) & 1
            bit_pos = bit * n_dims + dim
            if bit_pos < 64:
                low |= coord_bit.astype(np.uint64) << bit_pos
            else:
                high |= coord_bit.astype(np.uint64) << (bit_pos - 64)

    return high, low


def sort_segments_compound(
    segment_coords_2d: np.ndarray,
    dimensions: list[Dimension],
    method: Literal["morton", "hilbert"] = "hilbert",
) -> tuple[np.ndarray, dict]:
    """Sort segments using compound ordering in (2×D)-space.

    Segments are represented as (2×D)-dimensional points by concatenating both
    endpoint coordinates. This captures the full geometric nature of segments
    (position, orientation, length) for spatial coherence.

    The compound ordering strategy is:
    - Primary sort: Discrete dimensions from both endpoints (lexicographic)
    - Secondary sort: Morton/Hilbert code of spatial dimensions from both endpoints

    Args:
        segment_coords_2d: Segment coordinates in (2×D)-space, shape (S, 2*d)
            First d columns are start point, second d columns are end point.
        dimensions: Original Dimension objects (may have more dims than data)
        method: Spatial curve method ("morton" or "hilbert"), default "hilbert"

    Returns:
        sort_indices: Indices to reorder segments
        metadata: Dict with ordering metadata in (2×D)-space
    """
    n_segments, n_dims_2d = segment_coords_2d.shape
    # Use actual data dimensionality (segment_coords_2d is 2×D, so D = n_dims_2d // 2)
    n_dims_original = n_dims_2d // 2
    # Use only the first n_dims_original dimensions from the dimensions list
    dimensions = dimensions[:n_dims_original]

    # Build (2×D) dimension classification
    # Discrete dims from both endpoints: if dim 3,4 are discrete in D-space,
    # then dims 3,4,d+3,d+4 are discrete in (2×D)-space
    slice_dims_2d = []
    ordering_dims_2d = []

    for i, d in enumerate(dimensions):
        if d.discrete and not d.display:
            slice_dims_2d.append(i)  # Start point discrete dim
            slice_dims_2d.append(n_dims_original + i)  # End point discrete dim
        else:
            ordering_dims_2d.append(i)  # Start point spatial dim
            ordering_dims_2d.append(n_dims_original + i)  # End point spatial dim

    # Compute bits per ordering dimension (auto-select 128-bit if needed)
    if ordering_dims_2d:
        bits_per_dim = min(21, 64 // len(ordering_dims_2d))
        use_128bit = bits_per_dim < 10
        if use_128bit:
            bits_per_dim = min(21, 128 // len(ordering_dims_2d))
    else:
        bits_per_dim = 21
        use_128bit = False

    # Extract ordering dimension coordinates
    spatial_codes: tuple[np.ndarray, np.ndarray] | np.ndarray
    if ordering_dims_2d:
        ordering_coords = segment_coords_2d[:, ordering_dims_2d]
        ordering_min = ordering_coords.min(axis=0)
        ordering_max = ordering_coords.max(axis=0)

        # Normalize to grid
        grid_coords = normalize_coords_to_grid(
            ordering_coords, ordering_min, ordering_max, 2**bits_per_dim
        )

        # Compute spatial curve codes
        if method == "morton":
            if use_128bit:
                high, low = morton_encode_128bit(grid_coords, bits_per_dim)
                spatial_codes = (high, low)  # Tuple for lexsort
            else:
                spatial_codes = morton_encode_nd(grid_coords, bits_per_dim)
        elif method == "hilbert":
            # Hilbert doesn't have 128-bit implementation - fall back to morton for high dims
            if use_128bit:
                high, low = morton_encode_128bit(grid_coords, bits_per_dim)
                spatial_codes = (high, low)
            else:
                spatial_codes = hilbert_encode_nd(grid_coords, bits_per_dim)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # No ordering dimensions - all discrete
        spatial_codes = np.zeros(n_segments, dtype=np.uint64)
        ordering_min = np.array([])
        ordering_max = np.array([])

    # Create compound sort key
    if slice_dims_2d:
        # Extract discrete dimension values
        slice_values = segment_coords_2d[:, slice_dims_2d]

        # Create sort keys: (discrete_tuple, spatial_code)
        if isinstance(spatial_codes, tuple):
            # 128-bit: lexsort by (low, high, discrete_dims...)
            high, low = spatial_codes
            sort_indices = np.lexsort(
                [low, high]
                + [slice_values[:, i] for i in range(len(slice_dims_2d) - 1, -1, -1)]
            )
        else:
            sort_indices = np.lexsort(
                [spatial_codes]
                + [slice_values[:, i] for i in range(len(slice_dims_2d) - 1, -1, -1)]
            )
    else:
        # Pure spatial ordering (no discrete dimensions)
        if isinstance(spatial_codes, tuple):
            high, low = spatial_codes
            sort_indices = np.lexsort([low, high])
        else:
            sort_indices = np.argsort(spatial_codes)

    # Build metadata
    metadata = {
        "ordering": method,
        "slice_dims": slice_dims_2d,
        "ordering_dims": ordering_dims_2d,
        "ordering_min": ordering_min.tolist() if len(ordering_min) > 0 else [],
        "ordering_max": ordering_max.tolist() if len(ordering_max) > 0 else [],
        "ordering_bits_per_dim": bits_per_dim,
    }

    return sort_indices, metadata


def order_lines_spatial(
    vertices: np.ndarray,
    segments: np.ndarray,
    dimensions: list[Dimension],
    method: Literal["morton", "hilbert"] = "hilbert",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply dual spatial ordering to lines using Morton or Hilbert curves.

    Lines have dual ordering:
    1. Vertices are ordered in D-space (like Points)
    2. Segments are ordered in (2×D)-space (concatenating both endpoints)

    This is the main entry point for Lines spatial ordering.

    Args:
        vertices: (V, D) float32 vertex positions
        segments: (S, 2) uint32 index pairs (from convert_to_indexed)
        dimensions: List of Dimension objects (may have more dims than data)
        method: Ordering method ("morton" or "hilbert")

    Returns:
        sorted_vertices: (V, D) float32 reordered vertex positions
        sorted_segments: (S, 2) uint32 reordered and remapped segment indices
        vertex_sort_indices: Indices to recover original vertex order
        segment_sort_indices: Indices to recover original segment order
        metadata: Dict with "vertex_ordering" and "segment_ordering" sub-dicts
    """
    _V, D = vertices.shape  # V unused but kept for reference

    # Use only the first D dimensions from the dimensions list
    # This handles cases where scene has more dimensions than data
    dimensions = dimensions[:D]

    # 1. Order vertices in D-space (with compound ordering for discrete dims)
    vertex_sort_indices, vertex_metadata = sort_points_compound(
        vertices, dimensions, method=method
    )
    sorted_vertices = vertices[vertex_sort_indices]

    # 2. Create inverse mapping for index remapping
    # After sorting, vertex at original index i is now at inverse_map[i]
    inverse_map = np.argsort(vertex_sort_indices).astype(np.uint32)
    remapped_segments = inverse_map[segments]

    # 3. Build (2×D) segment coordinates for ordering
    # Each segment becomes a 2D point: [x1, y1, z1, x2, y2, z2] for 3D
    segment_coords_2d = np.concatenate(
        [
            sorted_vertices[remapped_segments[:, 0]],  # Start points
            sorted_vertices[remapped_segments[:, 1]],  # End points
        ],
        axis=1,
    )  # Shape: (S, 2×D)

    # 4. Order segments in (2×D)-space (with compound ordering)
    segment_sort_indices, segment_metadata = sort_segments_compound(
        segment_coords_2d, dimensions, method=method
    )
    sorted_segments = remapped_segments[segment_sort_indices]

    return (
        sorted_vertices,
        sorted_segments,
        vertex_sort_indices,
        segment_sort_indices,
        {"vertex_ordering": vertex_metadata, "segment_ordering": segment_metadata},
    )


def compute_vertex_chunk_bounds(
    vertices: np.ndarray,
    chunk_size: int,
    slice_dims: Optional[list[int]] = None,
    dimensions: Optional[list] = None,
) -> np.ndarray:
    """Compute chunk bounding boxes for vertices (no radius/width expansion).

    Args:
        vertices: Vertex positions (already sorted), shape (V, D)
        chunk_size: Number of vertices per chunk
        slice_dims: Indices of discrete (non-spatial) dimensions
        dimensions: Retained for API stability; no longer drives padding width
            (discrete dims are padded by a float-boundary epsilon only — the
            reader's per-dimension tolerance owns the query reach).

    Returns:
        chunk_bounds: Bounding boxes, shape (num_chunks, D, 2)
    """
    n_vertices, n_dims = vertices.shape
    num_chunks = (n_vertices + chunk_size - 1) // chunk_size
    discrete_dims = set(slice_dims) if slice_dims else set()

    chunk_bounds = np.zeros((num_chunks, n_dims, 2), dtype=np.float32)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_vertices)
        chunk_verts = vertices[start_idx:end_idx]

        for d in range(n_dims):
            if d in discrete_dims:
                # Discrete: tight bounds padded by a float-boundary epsilon only
                # (the reader's per-dimension tolerance owns the query reach).
                chunk_bounds[chunk_idx, d, 0] = (
                    chunk_verts[:, d].min() - _BARRIER_BOUND_EPS
                )
                chunk_bounds[chunk_idx, d, 1] = (
                    chunk_verts[:, d].max() + _BARRIER_BOUND_EPS
                )
            else:
                # Spatial: exact bounds (no size expansion for vertices)
                chunk_bounds[chunk_idx, d, 0] = chunk_verts[:, d].min()
                chunk_bounds[chunk_idx, d, 1] = chunk_verts[:, d].max()

    return chunk_bounds


def compute_segment_chunk_bounds(
    vertices: np.ndarray,
    segments: np.ndarray,
    widths: np.ndarray,
    chunk_size: int,
    slice_dims: Optional[list[int]] = None,
    dimensions: Optional[list] = None,
) -> np.ndarray:
    """Compute chunk bounding boxes for segments (includes line width).

    Segment bounds are in D-dimensional space (not 2×D) for view frustum
    intersection tests. Each segment's bounds include the line width extent.

    IMPORTANT: widths must be a full (V,) array. Broadcast widths should be
    expanded with np.full(V, width_value) before calling this function.

    Args:
        vertices: Vertex positions (already sorted), shape (V, D)
        segments: Segment index pairs (already sorted), shape (S, 2)
        widths: Vertex widths (already sorted), shape (V,)
        chunk_size: Number of segments per chunk
        slice_dims: Indices of discrete (non-spatial) dimensions
        dimensions: Retained for API stability; no longer drives padding width
            (discrete dims are padded by a float-boundary epsilon only — the
            reader's per-dimension tolerance owns the query reach).

    Returns:
        chunk_bounds: Bounding boxes, shape (num_chunks, D, 2)
    """
    _V, D = vertices.shape
    S = segments.shape[0]
    num_chunks = (S + chunk_size - 1) // chunk_size
    discrete_dims = set(slice_dims) if slice_dims else set()

    chunk_bounds = np.zeros((num_chunks, D, 2), dtype=np.float32)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, S)
        chunk_segs = segments[start_idx:end_idx]

        # Get vertex positions and widths for this chunk's segments
        p1 = vertices[chunk_segs[:, 0]]  # Start vertex positions
        p2 = vertices[chunk_segs[:, 1]]  # End vertex positions
        w1 = widths[chunk_segs[:, 0]]  # Start vertex widths
        w2 = widths[chunk_segs[:, 1]]  # End vertex widths
        max_w = np.maximum(w1, w2)  # Conservative bound per segment

        for d in range(D):
            if d in discrete_dims:
                # Discrete: no width expansion, only a float-boundary epsilon.
                # The reader's per-dimension tolerance owns the query reach
                # (see _BARRIER_BOUND_EPS); the `dimensions` param is retained
                # for API stability but no longer drives the padding width.
                chunk_bounds[chunk_idx, d, 0] = (
                    min(p1[:, d].min(), p2[:, d].min()) - _BARRIER_BOUND_EPS
                )
                chunk_bounds[chunk_idx, d, 1] = (
                    max(p1[:, d].max(), p2[:, d].max()) + _BARRIER_BOUND_EPS
                )
            else:
                # Spatial: include width extent
                chunk_bounds[chunk_idx, d, 0] = min(
                    (p1[:, d] - max_w).min(), (p2[:, d] - max_w).min()
                )
                chunk_bounds[chunk_idx, d, 1] = max(
                    (p1[:, d] + max_w).max(), (p2[:, d] + max_w).max()
                )

    return chunk_bounds
