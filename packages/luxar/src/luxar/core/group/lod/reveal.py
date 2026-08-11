"""luxar.core.group.lod.reveal – the element-side concentric-shell reveal.

Everything the ``method="radial"`` REVEAL ordering needs, for Points and Lines
alike: the ordering key itself (:func:`radial_element_score`), the two resolvers
that decide which columns are *spatial* enough to be shell dimensions, and the
spec-validation for the two reveal-only knobs.

Split out of :mod:`luxar.core.group.lod.group` — which is the geometry-agnostic
LOD hub and had grown past 1500 lines — because this is one cohesive concern with
one name. Nothing here imports ``group``, so the dependency runs one way only
(``group`` → ``reveal``).

Which methods count as a reveal is NOT decided here: it is written once in
:mod:`luxar.utils.lod_methods` and shared with the gsplat ladder, because being a
reveal is a property of the concept rather than of a geometry. See
MESH_NODE_SPEC §9.1, whose reasoning is geometry-agnostic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from ....utils.lod_methods import REVEAL_METHODS, is_reveal_method

#: Methods that order for a REVEAL rather than for approximation quality, and so
#: must not carry energy stamps — the viewer's ``1/e(k)`` brightness compensation
#: is gated on the blending mode, not on geometry type, and a reveal's prefix is a
#: partial object at FULL brightness rather than a dim version of the whole.
#:
#: ALIASES the shared registry rather than restating it. The element-side
#: :data:`~luxar.core.group.lod.group.ADDITIVE_METHODS` and the gsplat-side one
#: legitimately differ (they order different things), but which methods are a
#: reveal is a property of the *concept*, so it is written once in
#: :mod:`luxar.utils.lod_methods` and shared with the gsplat ladder. See
#: MESH_NODE_SPEC §9.1, whose reasoning is geometry-agnostic.
REVEAL_ADDITIVE_METHODS: frozenset[str] = REVEAL_METHODS
is_reveal_additive_method = is_reveal_method


def resolve_reveal_spatial_dims(
    spec: Dict[str, Any], scene: Any, n_cols: int
) -> Optional[List[int]]:
    """The ``spatial_dims`` a scene-aware adder should hand the reveal scorer.

    The caller's explicit value always wins. Otherwise, and only for a reveal
    ordering, fall back to :func:`default_reveal_spatial_dims` — the displayed
    dims — so a stacked time/channel column cannot become a shell dimension.
    ``None`` leaves :func:`radial_element_score` on its own extent rule.
    """
    explicit = spec.get("spatial_dims")
    if explicit is not None or not is_reveal_additive_method(str(spec.get("method"))):
        return explicit
    return default_reveal_spatial_dims(scene, n_cols)


def default_reveal_spatial_dims(scene: Any, n_cols: int) -> Optional[List[int]]:
    """The scene's DISPLAYED columns, when a reveal can be anchored to them.

    :func:`radial_element_score`'s own fallback — the columns with non-zero
    positional extent — drops a *constant* time/channel column (the shape a
    one-node-per-timepoint scene has), but it cannot drop a **stacked** one: a
    column holding several timepoints in one array varies across elements exactly
    the way a spatial axis does, so extent has nothing to separate them by. The
    scene does: a stacked axis is a NON-DISPLAYED dimension. Anchoring the shells
    to the displayed dims is the same Auto rule :func:`~luxar.core.group.lod.group.resolve_coarsen_dims` uses
    for coarsening barriers.

    Only the scene-aware callers (the ``add_points`` / ``add_lines`` adders) can
    apply this; ``make_additive_lod_points`` / ``_lines`` are handed bare arrays
    and keep the extent fallback.

    Returns ``None`` — meaning "keep the extent fallback" — when there is no
    dimension metadata, when the positions are not aligned with it (``dim_order``
    / ``extend_to_all`` reshaped the columns, so a scene-dim index is not a
    position column), or when every dimension is displayed (nothing to exclude).
    """
    dims = getattr(scene, "_dimensions", None) if scene is not None else None
    if dims is None or int(getattr(dims, "ndim", -1)) != int(n_cols):
        return None
    displayed = sorted({int(d) for d in dims.displayed if 0 <= int(d) < n_cols})
    if not displayed or len(displayed) == n_cols:
        return None
    return displayed


def _resolve_score_dims(pts_all: NDArray, spatial_dims: Optional[List[int]]) -> NDArray:
    """Validate an explicit ``spatial_dims``, or derive it from non-zero extent.

    Extracted from :func:`radial_element_score` to keep it under the C901
    ratchet; validate-or-derive is one concern and reads better named.

    The explicit branch is validated HERE and not only in
    :func:`~luxar.core.group.lod.group.resolve_additive_axis`, because
    ``compute_additive_order_points`` / ``_lines`` are public entry points that
    bypass the resolver entirely. Each
    rejected case silently produced a WRONG ordering rather than an error: a
    negative index ALIASES to another column via numpy indexing, a repeat
    DOUBLE-COUNTS that axis in the distance, an empty list scores every
    element 0.0 — degrading the ordering to input order with no indication —
    and a NESTED sequence makes ``pts_all[:, dims]`` 3-D, so the score comes
    back ``(N, k)`` and ``argsort`` returns a per-row permutation rather than
    an ordering of the elements.
    """
    if spatial_dims is None:
        mins_all = pts_all.min(axis=0)
        maxs_all = pts_all.max(axis=0)
        dims = np.flatnonzero(maxs_all - mins_all > 0.0).astype(np.intp)
        # A single-position input has no extent on any axis. Fall back to every
        # axis rather than to an empty selection, which would score every element
        # 0.0 and silently degrade the ordering to input order.
        if dims.size == 0:
            dims = np.arange(pts_all.shape[1], dtype=np.intp)
        return dims

    dims = np.asarray(spatial_dims, dtype=np.intp)
    if dims.ndim != 1:
        raise ValueError(
            f"spatial_dims must be a 1-D sequence of column indices (a nested "
            f"one makes the score 2-D and the permutation malformed); got shape "
            f"{dims.shape}"
        )
    if dims.size == 0:
        raise ValueError("spatial_dims must not be empty")
    if int(dims.min()) < 0:
        raise ValueError(
            f"spatial_dims must be non-negative (a negative index would alias "
            f"to another column); got {list(spatial_dims)}"
        )
    if np.unique(dims).size != dims.size:
        raise ValueError(
            f"spatial_dims must not repeat an axis (a repeat would count it "
            f"twice in the distance); got {list(spatial_dims)}"
        )
    if int(dims.max()) >= pts_all.shape[1]:
        raise ValueError(
            f"spatial_dims {list(spatial_dims)} out of range for coords with "
            f"{pts_all.shape[1]} columns"
        )
    return dims


def resolve_reveal_centre(
    centre: Optional[List[float]],
    coords: NDArray,
    scored: NDArray,
    spatial_dims: Optional[List[int]] = None,
) -> Optional[List[float]]:
    """An explicit shell origin, or the bbox centre of ``coords`` over the shell axes.

    ``scored`` is the array :func:`radial_element_score` will be handed — one
    representative coordinate per element — and decides WHICH columns are shell
    dimensions, exactly as the scorer would. ``coords`` is the geometry those
    shells should be centred on.

    The two are the same array for Points, whose representative IS its position,
    so that geometry can leave the centre to the scorer's own default. This helper
    exists for Lines: a polyline's representative is its own bbox centre, and the
    bounding box OF THOSE CENTRES is not the node's. One long polyline spanning
    x=0…100 plus a short one near x=0 give centres of 50 and 0 — an origin of 25,
    where the node's own vertex bbox centre is 50. Passing the vertices as
    ``coords`` keeps the documented contract ("the node's own bbox centre") true
    for both geometries.
    """
    if centre is not None:
        return centre
    dims = _resolve_score_dims(np.asarray(scored, dtype=np.float64), spatial_dims)
    pts = np.asarray(coords, dtype=np.float64)[:, dims]
    return [float(c) for c in 0.5 * (pts.min(axis=0) + pts.max(axis=0))]


def radial_element_score(
    coords: NDArray,
    centre: Optional[List[float]] = None,
    spatial_dims: Optional[List[int]] = None,
) -> NDArray:
    """Distance of each element from ``centre``, over the spatial axes only.

    The ordering key for ``method="radial"`` — the concentric-shell reveal —
    shared by Points and Lines so the two can't disagree about what "radial"
    means. ``coords`` is one representative coordinate per element: a point's
    position, or a polyline's own bbox centre.

    ``centre`` defaults to the **bounding-box centre of the spatial axes**, not
    the scene origin: a dataset sitting far from the origin would otherwise
    reveal from one corner instead of growing from its own middle.

    ``spatial_dims`` defaults to the columns with **non-zero extent** — the
    element-side stand-in for the gsplat path's ``_nondegenerate_axes`` (which is
    covariance-based and has no meaning here). Read its reach precisely: it drops
    a *constant* time/channel column, so a one-node-per-timepoint scene is
    handled, but a **stacked** column (several timepoints in one array) varies
    across elements exactly like a spatial axis and IS included — including it
    pushes the elements furthest in time to the end of the ladder, so an
    off-centre timepoint's slice paints last instead of growing outward. Extent
    alone cannot separate the two cases; the scene can, which is why the
    ``add_points`` / ``add_lines`` adders pass
    :func:`default_reveal_spatial_dims` (the displayed dims) when the caller
    named none. On a bare array — this function's own contract — pass
    ``spatial_dims`` explicitly for stacked data.

    Unlike ``spatial-uniform`` / ``poisson-disk``, this deliberately does NOT
    require ``d >= 3``: a distance is well defined in any dimension, and a 2D
    scene is a first-class authoring path.
    """
    pts_all = np.asarray(coords, dtype=np.float64)
    if pts_all.ndim != 2:
        raise ValueError(f"coords must be 2-D (N, d); got shape {pts_all.shape}")
    if pts_all.shape[0] == 0:
        # The bbox reductions below have no identity on an empty axis, and the
        # bare numpy message names neither the argument nor this function. Both
        # element callers return early at n == 0, so this only guards a direct
        # caller of this (public) helper.
        return np.empty(0, dtype=np.float64)
    if pts_all.shape[1] == 0:
        # No columns means no distance to measure; every score would be 0.0 and
        # the ordering would silently degrade to input order.
        raise ValueError("coords must have at least one column; got shape (N, 0)")
    dims = _resolve_score_dims(pts_all, spatial_dims)
    pts = pts_all[:, dims]
    if centre is None:
        origin = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    else:
        origin = np.asarray(centre, dtype=np.float64)
        if origin.shape != (len(dims),):
            raise ValueError(
                f"reveal_centre must have one coordinate per spatial axis "
                f"{[int(d) for d in dims]}; got {len(origin)}"
            )
        if not bool(np.all(np.isfinite(origin))):
            # Same class as an empty `spatial_dims`: every distance comes back
            # non-finite, they all compare equal under a stable argsort, and the
            # ladder silently degrades to input order instead of revealing.
            raise ValueError(
                f"reveal_centre must be finite (a NaN/inf coordinate makes every "
                f"distance non-finite, degrading the ordering to input order); "
                f"got {[float(c) for c in origin]}"
            )
    return np.asarray(np.linalg.norm(pts - origin, axis=1), dtype=np.float64)


def pop_reveal_knobs(
    kwargs: Dict[str, Any], method: str
) -> tuple[Optional[List[float]], Optional[List[int]]]:
    """Pop and validate the ``radial``-only shell-geometry keys from a spec dict.

    Extracted from :func:`~luxar.core.group.lod.group.resolve_additive_axis` to
    keep that function under the
    C901 ratchet — its validation is branchy (two optional keys, four shape rules,
    one cross-check) and self-contained.

    Validated at RESOLVE time rather than write time, for the same reason as
    ``counts``: under a substitutive ladder the wrapper group already exists on
    disk before its children are written, so a late raise leaves a partial group
    behind.

    Mutates ``kwargs`` (pops the two keys) so the caller's leftover-keys check
    still catches genuinely unknown names.
    """
    reveal_centre = kwargs.pop("reveal_centre", None)
    if reveal_centre is not None:
        reveal_centre = [float(c) for c in reveal_centre]
        if not reveal_centre:
            raise ValueError("reveal_centre must not be empty")
        if not all(math.isfinite(c) for c in reveal_centre):
            # Checked HERE and not only in the scorer for the same reason as the
            # rest of this function: under a substitutive ladder the wrapper group
            # is on disk before the scorer ever runs, so a late raise leaves a
            # partial group behind.
            raise ValueError(
                f"reveal_centre must be finite (a NaN/inf coordinate makes every "
                f"distance non-finite, degrading the ordering to input order); "
                f"got {reveal_centre}"
            )

    spatial_dims = kwargs.pop("spatial_dims", None)
    if spatial_dims is not None:
        spatial_dims = [int(d) for d in spatial_dims]
        if not spatial_dims:
            raise ValueError("spatial_dims must not be empty")
        if len(set(spatial_dims)) != len(spatial_dims):
            raise ValueError(
                f"spatial_dims must not repeat an axis; got {spatial_dims}"
            )
        if any(d < 0 for d in spatial_dims):
            raise ValueError(f"spatial_dims must be non-negative; got {spatial_dims}")

    if (reveal_centre is not None or spatial_dims is not None) and not (
        is_reveal_additive_method(method)
    ):
        # Silently ignoring these would look like the centre had been honoured.
        raise ValueError(
            "additive_lod: 'reveal_centre' / 'spatial_dims' apply only to a "
            f"reveal ordering ({' / '.join(sorted(REVEAL_ADDITIVE_METHODS))}); "
            f"got method={method!r}"
        )
    return reveal_centre, spatial_dims
