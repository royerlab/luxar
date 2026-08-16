"""Points spatial ordering: compound sort + radius-aware chunk bounds."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from luxar.core import Dimension
from luxar.typing_utils.constants import DEFAULT_POINT_RADIUS

from .bounds import _BARRIER_BOUND_EPS
from .compound import _compound_sort


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


def compute_chunk_bounds_points(
    positions: np.ndarray,
    radii: Optional[np.ndarray | float],
    chunk_size: int,
    slice_dims: Optional[list[int]] = None,
) -> np.ndarray:
    """Compute chunk bounding boxes for Points (includes radius extent).

    The pad IS the footprint: ``[min - r, max + r]`` is exactly the set of query
    positions for which some point in the chunk can still be visible, so the
    bound is right in both directions rather than merely wide enough. (With
    per-point uint8-encoded radii the encoder's rounding can move a stored radius
    by up to one quantum after these bounds are computed, so the "never tighter"
    half holds only up to that sub-quantum slack.)

    ``radii=None`` does not mean "no extent" — a points node that stores no radii
    array is drawn with the renderer's default radius, so the bounds are expanded
    by ``DEFAULT_POINT_RADIUS`` (:mod:`luxar.typing_utils.constants`), exactly as
    if a scalar radius of that value had been passed.

    Reader-side note: for a radii-less node the viewer runs NO per-point
    effective-radius cull (``data/points/effective-radius-calculator.ts`` and
    ``data/points/projection.ts`` both gate on ``radii``), and its hidden-dim
    query tolerance falls back to ``defaultMaxRadius = 0.1`` because no
    ``max_radius`` attr is stamped without radii. An honest (wider) bound
    therefore costs a few more fetched chunks, every point of which is drawn —
    the right trade against the old under-fetch. Shrinking that tolerance to a
    float-safety epsilon now that the bounds are honest is issue #1655 items 2
    and 3; nothing on the reader side changes here.

    CRITICAL: Radius expansion is only applied to SPATIAL dimensions, not discrete
    dimensions. Discrete dimensions (slice_dims) represent categorical values like
    time steps, channels, or orbital indices. A point at orbital=0 should NOT
    extend into orbital=3 space - they are separate categories. Those axes get
    only the tiny float-boundary epsilon ``_BARRIER_BOUND_EPS``, on every path.

    Args:
        positions: Point positions (already sorted), shape (N, d)
        radii: Point radii (already sorted), shape (N,), a broadcast scalar, or
               None (⇒ the renderer's ``DEFAULT_POINT_RADIUS``)
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

        # Absent radii == the renderer's default radius (see the docstring), so
        # the two cases share one code path.
        radii_scalar: Optional[float] = None
        chunk_radii: Optional[np.ndarray] = None
        if radii is None:
            radii_scalar = DEFAULT_POINT_RADIUS
        elif isinstance(radii, np.ndarray):
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

    return chunk_bounds
