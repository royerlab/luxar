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
from ....validation.types import (
    validate_finite_reveal_coords,
    validate_integral_axis_indices,
)

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
    dimension metadata, when the column count does not match the scene's
    dimensionality, or when every dimension is displayed (nothing to exclude).

    Why a column-COUNT match is enough to treat a scene-dim index as a column
    index: ``dim_order`` does not leave a permutation for this function to trip
    over. Both adders call
    :func:`~luxar.core.group.dim_order.apply_dim_order_positions` *before* they
    read ``ndim`` or call this resolver, and
    :func:`~luxar.core.scene.dim_order.apply_dim_order` builds its output as
    ``np.zeros((N, scene_ndim))`` filled by iterating ``enumerate(scene_names)``
    — so the array it returns is already in scene-dimension order at the scene's
    full dimensionality, and column *i* IS scene dimension *i*. ``extend_to_all``
    never reshapes the array at all (it is an attr describing broadcast, resolved
    separately). So the count check is not a proxy for alignment: after that
    normalization, matching counts means the columns really are the scene's, and
    a MISmatch means no ``dim_order`` was applied to a differently-shaped array,
    which is exactly the case that must decline.
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
    a NESTED sequence makes ``pts_all[:, dims]`` 3-D, so the score comes
    back ``(N, k)`` and ``argsort`` returns a per-row permutation rather than
    an ordering of the elements, and a FRACTIONAL index is truncated to a
    different column than the one named.
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

    validate_integral_axis_indices(spatial_dims)
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


def _validated_score_dims(
    pts_all: NDArray, spatial_dims: Optional[List[int]], what: str
) -> NDArray:
    """:func:`_resolve_score_dims`, with the finite check at the right granularity.

    The granularity is the whole point, and it is not "check everything":

    * When ``spatial_dims`` is EXPLICIT, only those columns are checked. A NaN on
      an axis the distance does not span cannot affect the ordering, and refusing
      it would reject data a reveal can rank perfectly well.
    * When the dims are DERIVED from extent, the full array must be checked
      **first**, because ``max - min > 0.0`` is ``False`` for a NaN column — so the
      extent rule silently DROPS the axis the NaN sits on and the reveal then
      measures over fewer axes than the data has. Checking only the surviving
      columns cannot see that: the offending column has already been discarded
      *because* it was bad. Measured — this is why the first version of the guard
      passed the inf case and missed the NaN one.
    """
    if spatial_dims is None:
        # The whole-array check already covers every column the derived dims can
        # select, so this arm returns before the per-column one — re-checking a
        # slice of an array just proved finite costs an (N, k) copy for nothing.
        validate_finite_reveal_coords(pts_all, what)
        return _resolve_score_dims(pts_all, None)
    dims = _resolve_score_dims(pts_all, spatial_dims)
    validate_finite_reveal_coords(pts_all[:, dims], what)
    return dims


def resolve_reveal_center(
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
    # Checked BEFORE the bbox so bad data is reported as bad data: without this a
    # NaN vertex produced a NaN origin here, which then tripped the scorer's
    # `reveal_center must be finite` check and blamed a knob the caller never
    # passed — the derived default wearing the knob's name.
    dims = _validated_score_dims(
        np.asarray(scored, dtype=np.float64), spatial_dims, "vertices"
    )
    pts = np.asarray(coords, dtype=np.float64)[:, dims]
    # `coords` is a DIFFERENT array from `scored` here (vertices vs per-polyline
    # representatives), and it is the one the origin is measured from, so it needs
    # its own check — a NaN vertex inside an otherwise-finite polyline centre would
    # slip through a check on the representatives alone.
    validate_finite_reveal_coords(pts, "vertices")
    return [float(c) for c in 0.5 * (pts.min(axis=0) + pts.max(axis=0))]


def wants_reveal_center_preflight(spec: Optional[Dict[str, Any]]) -> bool:
    """Whether :func:`preflight_reveal_center` would check anything for ``spec``.

    Only an EXPLICIT ``reveal_center`` under a reveal ordering has a length to
    cross-check; every other spec makes the preflight a no-op. Exposed so a
    caller whose ``coords`` argument is expensive to build can skip building it:
    the Lines wrapper has to run :func:`~luxar.core.group.lod.lines.identify_polylines`
    plus :func:`~luxar.core.group.lod.lines.polyline_bbox_centers` to get the
    per-polyline representatives, which is a Python loop over polylines —
    measured at ~2.5 s for a 400k-vertex ``segments`` node, on every
    ``add_lines(substitutive_lod=…)`` call, since the composed ladder defaults to
    ON. Kept next to the preflight so the two cannot disagree about when there is
    work to do.
    """
    if not spec:
        return False
    return spec.get("reveal_center") is not None and is_reveal_additive_method(
        str(spec.get("method"))
    )


def preflight_reveal_center(
    spec: Optional[Dict[str, Any]], scene: Any, coords: NDArray, what: str
) -> None:
    """Cross-check an explicit ``reveal_center`` against the axes it will pair with.

    :func:`pop_reveal_knobs` already does this when the caller names BOTH knobs,
    but it cannot when ``spatial_dims`` is left to be DERIVED — from the scene's
    displayed dims, or from non-zero extent — because that needs the data. So the
    mismatch surfaced inside :func:`radial_element_score`, which under a
    substitutive ladder runs while writing the FINEST child, i.e. after the
    wrapper ``kind=lod`` group and every coarse child are already on disk.
    Measured: a 3D scene whose points are planar (one constant column) derives
    shell axes ``[0, 1]``, so the natural 3-coordinate centre raised only at
    ``child_2``, stranding ``child_0``/``child_1`` and making a corrected retry
    die on "duplicate child name".

    Called by the two substitutive wrappers BEFORE the lift, with the same array
    and the same resolver the finest child will use, so it cannot drift from the
    scorer: the derivation IS :func:`_validated_score_dims`. That also brings the
    finite-coordinate check forward, which was late for the same reason.
    """
    if spec is None or not wants_reveal_center_preflight(spec):
        return
    centre = spec["reveal_center"]
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        # Degenerate shapes have their own (better) messages downstream, and the
        # element callers return early at n == 0 rather than scoring at all.
        return
    dims = _validated_score_dims(
        arr, resolve_reveal_spatial_dims(spec, scene, int(arr.shape[1])), what
    )
    if len(centre) != len(dims):
        raise ValueError(
            f"reveal_center has {len(centre)} coordinates but the shell axes "
            f"resolve to {[int(d) for d in dims]} ({len(dims)} axes); they must "
            f"match (one coordinate per shell axis). Pass spatial_dims= to name "
            f"the axes explicitly."
        )


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

    For Lines the scored representative is each polyline's own bbox centre, whose
    bounding box is NOT the node's — so ``add_lines`` must not leave the default
    to this function. It resolves the origin up front with
    :func:`resolve_reveal_center` and passes a concrete ``centre``. Points can use
    the default, because its representative IS its position.

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
    dims = _validated_score_dims(pts_all, spatial_dims, "coords")
    pts = pts_all[:, dims]
    if centre is None:
        origin = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    else:
        origin = np.asarray(centre, dtype=np.float64)
        if origin.shape != (len(dims),):
            raise ValueError(
                f"reveal_center must have one coordinate per spatial axis "
                f"{[int(d) for d in dims]}; got {len(origin)}"
            )
        if not bool(np.all(np.isfinite(origin))):
            # Same class as an empty `spatial_dims`: every distance comes back
            # non-finite, they all compare equal under a stable argsort, and the
            # ladder silently degrades to input order instead of revealing.
            raise ValueError(
                f"reveal_center must be finite (a NaN/inf coordinate makes every "
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
    two cross-checks) and self-contained.

    Validated at RESOLVE time rather than write time, for the same reason as
    ``counts``: under a substitutive ladder the wrapper group already exists on
    disk before its children are written, so a late raise leaves a partial group
    behind.

    Mutates ``kwargs`` (pops the two keys) so the caller's leftover-keys check
    still catches genuinely unknown names.
    """
    reveal_center = kwargs.pop("reveal_center", None)
    if reveal_center is not None:
        reveal_center = [float(c) for c in reveal_center]
        if not reveal_center:
            raise ValueError("reveal_center must not be empty")
        if not all(math.isfinite(c) for c in reveal_center):
            # Checked HERE and not only in the scorer for the same reason as the
            # rest of this function: under a substitutive ladder the wrapper group
            # is on disk before the scorer ever runs, so a late raise leaves a
            # partial group behind.
            raise ValueError(
                f"reveal_center must be finite (a NaN/inf coordinate makes every "
                f"distance non-finite, degrading the ordering to input order); "
                f"got {reveal_center}"
            )

    spatial_dims = kwargs.pop("spatial_dims", None)
    if spatial_dims is not None:
        # Before the `int()` below, which truncates 1.9 to 1 without a word.
        validate_integral_axis_indices(spatial_dims)
        spatial_dims = [int(d) for d in spatial_dims]
        if not spatial_dims:
            raise ValueError("spatial_dims must not be empty")
        if len(set(spatial_dims)) != len(spatial_dims):
            raise ValueError(
                f"spatial_dims must not repeat an axis; got {spatial_dims}"
            )
        if any(d < 0 for d in spatial_dims):
            raise ValueError(f"spatial_dims must be non-negative; got {spatial_dims}")

    if (reveal_center is not None or spatial_dims is not None) and not (
        is_reveal_additive_method(method)
    ):
        # Silently ignoring these would look like the centre had been honoured.
        raise ValueError(
            "additive_lod: 'reveal_center' / 'spatial_dims' apply only to a "
            f"reveal ordering ({' / '.join(sorted(REVEAL_ADDITIVE_METHODS))}); "
            f"got method={method!r}"
        )

    # The centre carries one coordinate per axis the distance spans, so when the
    # caller names both they must agree. The scorer checks this too, but only
    # once it runs — which under a substitutive ladder is AFTER the wrapper
    # kind=lod group and its coarse children are on disk, leaving a partial group
    # that a corrected retry then trips over with "duplicate child name". Same
    # cross-check the CLI does in `cli/reveal_options.py::parse_reveal_knobs`, so
    # the two entry points agree — and since that module is now shared by
    # `gsplat lod` and `mesh lod`, all three surfaces agree by construction rather
    # than by three copies staying in step.
    # Only checkable when both are explicit: a derived `spatial_dims`
    # (displayed dims / non-zero extent) is not known until the data is in hand.
    if (
        reveal_center is not None
        and spatial_dims is not None
        and len(reveal_center) != len(spatial_dims)
    ):
        raise ValueError(
            f"reveal_center has {len(reveal_center)} coordinates but "
            f"spatial_dims lists {len(spatial_dims)} axes; they must match "
            f"(one coordinate per shell axis)"
        )
    return reveal_center, spatial_dims
