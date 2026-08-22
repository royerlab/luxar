"""Points spatial ordering: compound sort + radius-aware chunk bounds."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from luxar.core import Dimension
from luxar.typing_utils.constants import DEFAULT_POINT_RADIUS

from .bounds import (
    _BARRIER_BOUND_EPS,
    _normalise_coord_slack,
    _normalise_scalar_slack,
    _normalise_slice_dims,
    _store_outward_f32,
)
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
    *,
    coord_slack: Optional[NDArray[np.float64]] = None,
    scalar_slack: Optional[float] = None,
) -> np.ndarray:
    """Compute chunk bounding boxes for Points (includes radius extent).

    The pad IS the footprint: ``[min - r, max + r]`` is exactly the set of query
    positions for which some point in the chunk can still be visible, so the
    bound is right in both directions rather than merely wide enough. The
    interval is computed in float64 and narrowed to the float32 store with
    OUTWARD rounding (see :func:`_store_outward_f32`), so the "never tighter"
    half holds against the padded radii at every coordinate magnitude — not
    only where a 0.5 pad happens to survive a round-to-nearest store.

    SCALAR QUANTISATION SLACK (``scalar_slack``): ``radii`` is itself a
    POSITIVE_SCALAR and may decode larger than the authored value. This single
    per-array pad is added to the radius on SPATIAL dimensions only; barrier
    dimensions still receive no footprint expansion. The compiler supplies it
    from
    :meth:`~luxar.encoding.encoder.ArrayEncoder.positive_scalar_round_trip_slack`;
    a direct caller that omits it gets authored-radius bounds.

    QUANTISATION SLACK (``coord_slack``): these bounds are computed from the
    positions AS HANDED IN, but under the default AUTO encoding the positions
    themselves are stored as per-axis uint16 fixed point (the COORDINATE path in
    ``luxar.encoding._encoders.perchannel``), so a DECODED position can land up
    to half a quantum (``extent/131070``) outside a bound derived from the
    authored one — 7.6e-3 at an extent of 1000, well above the float32 ULP the
    outward store closes, and enough for the reader to skip the chunk entirely.
    ``coord_slack`` is that displacement, per axis, and is added outward on
    EVERY dimension: on top of the radius on a spatial axis and on top of
    ``_BARRIER_BOUND_EPS`` on a barrier axis (the epsilon is float-boundary
    safety, the slack is a displacement — they add). The compiler supplies it
    from the encoder's own predicate
    (:meth:`~luxar.encoding.encoder.ArrayEncoder.coordinate_round_trip_slack`),
    which returns zero for an axis the encoder stores exactly (a gridded
    time/channel axis, a constant axis, a float32 fallback — and the whole array
    when the write will really store a LUT, which the caller declares with that
    predicate's ``allow_lut``). A direct caller that omits it gets bounds for the
    authored coordinates only (issue #1655).

    ``radii=None`` does not mean "no extent" — a points node that stores no radii
    array is drawn with the renderer's default radius, so the bounds are expanded
    by ``DEFAULT_POINT_RADIUS`` (:mod:`luxar.typing_utils.constants`), exactly as
    if a scalar radius of that value had been passed.

    Reader-side note: for a radii-less node the viewer runs NO per-point
    effective-radius cull (``data/points/effective-radius-calculator.ts`` and
    ``data/points/projection.ts`` both gate on ``radii``), and its hidden-dim
    query tolerance falls back to ``defaultMaxRadius = 0.1`` because no
    ``max_radius`` attr is stamped without radii. An honest (wider) bound is
    therefore paid in over-DRAW, not merely over-fetch: nothing culls the extra
    points, so every point of every chunk the window touches is rendered. And
    the cost is relative to the AXIS, not to the pad — the selection window on a
    hidden continuous axis grows from ``chunk_extent + 2*(0.01 + 0.1)`` to
    ``chunk_extent + 2*(0.5 + 0.1)``, a few percent on an axis spanning ~100
    units but EVERY chunk on an axis spanning ~1 unit. It is still the right
    trade against the old under-fetch, which dropped points with nothing to
    notice. Shrinking that tolerance to a float-safety epsilon now that the
    bounds are honest is issue #1655 items 2 and 3; nothing on the reader side
    changes here.

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
                   expansion should NOT be applied. Default: None (apply to all
                   dims). An index outside ``[0, d)`` raises ``ValueError``
                   (see :func:`_normalise_slice_dims`).
        coord_slack: Per-axis outward pad, shape ``(d,)``, covering how far the
                   STORE can move a coordinate from the value passed here (see
                   the QUANTISATION SLACK note above). Default None ⇒ zero on
                   every axis. Validated by :func:`_normalise_coord_slack`.
        scalar_slack: Outward pad covering how far the stored radius can exceed
                   the authored radius. Applied on spatial dimensions only.

    Returns:
        chunk_bounds: Bounding boxes, shape (num_chunks, d, 2)
    """
    n_points, ndim = positions.shape
    num_chunks = (n_points + chunk_size - 1) // chunk_size

    discrete_dims = _normalise_slice_dims(slice_dims, ndim)
    slack = _normalise_coord_slack(coord_slack, ndim)
    footprint_slack = _normalise_scalar_slack(scalar_slack)

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

        padded_chunk_radii: Optional[np.ndarray] = None
        if chunk_radii is not None:
            padded_chunk_radii = (
                np.asarray(chunk_radii).astype(np.float64, copy=False) + footprint_slack
            )

        # Compute bounds for each dimension separately. The pad is added in
        # float64 and stored with outward rounding — see _store_outward_f32.
        for d in range(ndim):
            coords = chunk_positions[:, d].astype(np.float64, copy=False)
            if d in discrete_dims:
                # DISCRETE dimension: No radius expansion!
                # Just use exact min/max of coordinate values, padded only by a
                # tiny float-boundary epsilon (the reader's tolerance owns the
                # query reach — see _BARRIER_BOUND_EPS).
                mins_d = coords.min() - _BARRIER_BOUND_EPS
                maxs_d = coords.max() + _BARRIER_BOUND_EPS
            else:
                # SPATIAL dimension: Include radius extent
                if radii_scalar is not None:
                    radius = radii_scalar + footprint_slack
                    mins_d = coords.min() - radius
                    maxs_d = coords.max() + radius
                else:
                    assert padded_chunk_radii is not None
                    mins_d = (coords - padded_chunk_radii).min()
                    maxs_d = (coords + padded_chunk_radii).max()

            # The quantisation displacement applies to EVERY dim — spatial and
            # barrier alike — and is added in float64, before the outward
            # float32 store (which cannot recover a pad already rounded away).
            lo32, hi32 = _store_outward_f32(
                float(mins_d) - slack[d], float(maxs_d) + slack[d]
            )
            chunk_bounds[chunk_idx, d, 0] = lo32
            chunk_bounds[chunk_idx, d, 1] = hi32

    return chunk_bounds
