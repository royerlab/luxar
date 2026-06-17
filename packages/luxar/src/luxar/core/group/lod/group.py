"""luxar.core.group.lod.group – Geometry-agnostic helpers for the LOD-kind Group.

The LOD-kind ``Group`` selects one of N alternative children at runtime based
on a view-driven metric (currently: the projected bbox diagonal in pixels). It
is geometry-agnostic — children can be ``points``, ``lines``, ``gsplats``, or
themselves a specialized group (``kind=lod`` / ``kind=partition``).

Each child carries its own ``min_pixel_size`` attribute (strictly monotonic
increasing in coarsest→finest order; coarsest = 0.0). The viewer picks the
finest child whose threshold is satisfied by the current view. The standalone
builder ``add_lod_group()`` lets users assemble these by hand; the convenience
paths (e.g. ``Scene.add_gsplats_from_data(..., lod_group=...)``) auto-derive
thresholds via ``lod_thresholds``. The default ``"extent"`` method anchors each
threshold in physical element size (``T·W/r`` — self-calibrating); the legacy
``"count"`` method is the scene-relative √(N_finer / N_coarsest) heuristic
anchored at ``BASE_PIXEL_SIZE``.

Everything here is type-agnostic and shared across all leaf geometries
(Points, Lines, GSplats) and the Partition kind. The geometry-specific
``lod_group=`` / ``additive_lod=`` axis resolvers live next to their data
types (e.g. ``lod.gsplats`` for ``GSplatData``).

This module hosts:

* The ``lod_thresholds`` method selector (``"extent"`` default, ``"count"``
  fallback), ``extent_min_pixel_sizes`` (``T·W/r``), the ``derive_min_pixel_sizes``
  √N heuristic, and their shared ``_apply_monotonicity_guard``.
* The free-function validator ``validate_lod_group``, callable on any
  ``Group`` whose ``attrs["kind"] == "lod"``.
* The shared ``resolve_display_type`` helper used by both LOD and Partition
  kinds, which walks down through nested specialized groups to determine what
  geometry type the user sees this layer as, and ``compute_lod_display_type``
  which derives an LOD group's ``display_type`` from its finest child.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from ...node import Node


#: Base value for auto-derived ``min_pixel_size`` thresholds. Child *i* gets
#: ``BASE_PIXEL_SIZE * sqrt(n_elements[i] / n_elements[0])`` (coarsest =
#: child 0). Geometry can resolve roughly √N effective screen pixels of
#: detail, so the threshold scales linearly with that. 10 px is a reasonable
#: detail-floor below which a finer level isn't worth the cost.
BASE_PIXEL_SIZE: float = 10.0


# ────────────────────────────────────────────────────────────────────────
# Display-type resolution (shared with the Partition kind)
# ────────────────────────────────────────────────────────────────────────


def resolve_display_type(node: "Node") -> str:
    """Return the geometry type this node would appear as to the user.

    For plain leaves and plain groups, this is the node's own ``type`` attr
    (``"points"`` / ``"lines"`` / ``"gsplats"`` / ``"group"``). For specialized
    groups (``kind == "lod"`` or ``kind == "partition"``), this is the
    ``display_type`` attr the writer recorded on them — which is itself
    derived transitively when one specialized group wraps another.

    Used by:
        * ``validate_partition_group`` — to compare homogeneity across children
          even when some children are themselves specialized groups.
        * ``compute_lod_display_type`` — to walk a finest-child chain
          through nested LOD/Partition groups down to a real geometry leaf.
        * The compiler at finalize, to compute ``display_type`` for a
          freshly-assembled specialized group.
    """
    kind = node.attrs.get("kind")
    if kind in ("lod", "partition"):
        display = node.attrs.get("display_type")
        if isinstance(display, str):
            return display
    return str(node.attrs.get("type", "group"))


def compute_lod_display_type(children: List["Node"]) -> str:
    """Derive an LOD group's ``display_type`` from its finest child.

    Convention: children are stored in coarsest→finest order, so the
    finest is the last entry. If that child is itself a kind=lod / kind=partition
    group, recurse through its own ``display_type``.
    """
    if not children:
        raise ValueError(
            "compute_lod_display_type: cannot derive display_type from an "
            "empty children list"
        )
    return resolve_display_type(children[-1])


# ────────────────────────────────────────────────────────────────────────
# min_pixel_size monotonicity invariant
# ────────────────────────────────────────────────────────────────────────


def _assert_strict_ascending(thresholds: List[float], source: str) -> None:
    """Validate ``thresholds`` is strictly monotonic increasing (coarsest→finest).

    Shared between the geometry-specific ``lod_group=`` resolvers (explicit
    ``min_pixel_sizes=`` path) and :func:`derive_min_pixel_sizes` so both
    paths apply the same invariant — and so explicit lists fail at the
    resolver instead of deferring to a later :func:`validate_lod_group` call
    that the user may never make.
    """
    prev = float("-inf")
    for i, v in enumerate(thresholds):
        if v <= prev:
            raise ValueError(
                f"{source}: min_pixel_sizes must be strictly increasing in "
                f"coarsest→finest order; entry {i}={v} is not greater than "
                f"previous {prev}"
            )
        prev = v


def derive_min_pixel_sizes(
    element_counts: list[int],
    base_pixel_size: Optional[float] = None,
) -> list[float]:
    """Auto-derive monotonic ``min_pixel_size`` thresholds from element counts.

    Coarsest child (index 0) gets ``0.0``; each subsequent child *i* gets
    ``base_pixel_size * sqrt(n_elements[i] / n_elements[0])``. Geometry can
    resolve ~√N effective screen pixels of detail, so the threshold scales
    linearly with that. ``BASE_PIXEL_SIZE`` (~10 px) is the detail floor
    below which a finer level isn't worth the cost.

    **Heuristic caveat.** Element *count* is only a proxy for screen
    *coverage*: a level with 4× the elements does not necessarily resolve
    2× the linear detail — that depends on how they are distributed
    in space. The proxy is weakest for *substitutive* levels, where a
    coarser level has fewer but larger elements; there a count-driven
    threshold can switch a touch early. When a particular ladder switches
    at the wrong zoom, override the anchor via ``base_pixel_size``.

    Args:
        element_counts: One entry per child, in coarsest→finest order. Must
            be non-empty and the first entry must be > 0.
        base_pixel_size: Override for the global ``BASE_PIXEL_SIZE``
            constant. ``None`` (default) → use the module-level constant.
            Useful when a particular LOD ladder benefits from a higher
            switching threshold (e.g., very small elements where the
            default 10-px floor crosses too early).

    Returns:
        List of thresholds, same length as ``element_counts``. Strictly
        monotonic increasing for monotonic non-decreasing input.
    """
    if not element_counts:
        raise ValueError("element_counts must be non-empty")
    n0 = element_counts[0]
    if n0 <= 0:
        raise ValueError(
            "coarsest LOD level is empty (0 elements), so per-level pixel "
            "thresholds cannot be derived. This usually means a substitutive "
            "reduction culled every representative — e.g. the input splats all "
            f"have non-positive amplitude. Got element_counts={list(element_counts)}."
        )
    bps = BASE_PIXEL_SIZE if base_pixel_size is None else float(base_pixel_size)
    if bps <= 0:
        raise ValueError(f"base_pixel_size must be positive, got {bps}")
    thresholds: list[float] = [0.0]
    for n in element_counts[1:]:
        thresholds.append(bps * (n / n0) ** 0.5)
    return _apply_monotonicity_guard(thresholds, "derive_min_pixel_sizes")


def _apply_monotonicity_guard(thresholds: list[float], source: str) -> list[float]:
    """Enforce strict ascending thresholds, then assert the invariant.

    Defensive: guarantee strict monotonicity even when later children have the
    same (or fewer) elements / a larger extent than a previous level. Uses a
    *relative* bump (×1.1) rather than a fixed +1px so near-equal levels
    separate proportionally to their scale — a fixed pixel nudge places the
    switch threshold at a meaningless absolute value for large ladders.

    The relative bump is a no-op when the previous threshold is ``0`` (the
    coarsest entry is always the ``0.0`` floor, and a zero-count intermediate
    level derives to ``0``), which would leave two equal entries and trip the
    trailing assertion. So when the previous threshold is non-positive we fall
    back to a small absolute floor (``_MONOTONIC_FLOOR``); every later entry
    then bumps relative to a positive value. This makes the guard *total* — it
    never raises — so a degenerate ladder yields a usable strictly-ascending
    sequence instead of a crash. The trailing ``_assert_strict_ascending`` is
    the same invariant the explicit-``min_pixel_sizes`` path is checked against.
    """
    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            prev = thresholds[i - 1]
            thresholds[i] = prev * 1.1 if prev > 0 else _MONOTONIC_FLOOR
    _assert_strict_ascending(thresholds, source)
    return thresholds


#: Absolute floor (px) used by :func:`_apply_monotonicity_guard` when the
#: previous threshold is ``0`` and the relative ×1.1 bump would be a no-op.
#: Tiny on purpose: a degenerate level it separates effectively switches at the
#: smallest on-screen size, which is the least-surprising fallback.
_MONOTONIC_FLOOR: float = 1e-6


#: Target on-screen element size (px) for the ``extent`` method. A level is
#: adopted once its elements project to >= this many pixels — finer levels would
#: be sub-pixel waste, coarser ones visibly blocky. Unlike :data:`BASE_PIXEL_SIZE`
#: (a scene-*relative* √N anchor) this is a *physical* anchor: it does not need
#: per-dataset tuning, because it is expressed directly in screen pixels.
DEFAULT_TARGET_PIXEL_SIZE: float = 1.5

#: LOD threshold-derivation methods. ``"extent"`` (default) anchors the switch in
#: physical element size (W / r); ``"count"`` is the legacy scene-relative √N proxy.
LodThresholdMethod = Literal["extent", "count"]


def extent_min_pixel_sizes(
    element_extents: list[float],
    node_extent: float,
    base_pixel_size: Optional[float] = None,
) -> list[float]:
    """Derive ``min_pixel_size`` thresholds from per-level element *extents* (W/r).

    Mipmap-style resolvability: a level is adopted once its elements project to
    >= ``base_pixel_size`` (the target pixel size ``T``) on screen. With node
    world bbox diagonal ``W`` and level extent ``r_i`` (world units), an element
    projects to ``r_i * diagonalPx / W`` pixels at node-diagonal ``diagonalPx``,
    so the switch threshold is::

        threshold_0 = 0.0                # coarsest = always-eligible floor
        threshold_i = T * W / r_i        # i >= 1

    Because ``T`` is a physical pixel size (~1.5 px), it is **scene-independent**
    — no per-dataset anchor tuning (unlike :func:`derive_min_pixel_sizes`). And
    ``r_i`` is the *true* element size, so a substitutive level's fewer-but-larger
    coarse elements raise its threshold correctly (the count proxy cannot see
    this). Thresholds ascend because ``r_i`` descends coarsest→finest; the
    monotonicity guard covers any non-monotone extent input.

    Args:
        element_extents: per-level element radius (world units), coarsest→finest.
            Non-empty. The coarsest entry (index 0) is unused (its threshold is
            the 0.0 floor); entries 1.. should be > 0 (a tiny floor is applied).
        node_extent: the node's world bbox diagonal ``W`` (> 0).
        base_pixel_size: target element pixel size ``T``; ``None`` →
            :data:`DEFAULT_TARGET_PIXEL_SIZE`.
    """
    if not element_extents:
        raise ValueError("element_extents must be non-empty")
    if node_extent <= 0:
        raise ValueError(f"node_extent must be positive, got {node_extent}")
    t_px = (
        DEFAULT_TARGET_PIXEL_SIZE if base_pixel_size is None else float(base_pixel_size)
    )
    if t_px <= 0:
        raise ValueError(f"base_pixel_size (target px) must be positive, got {t_px}")
    eps = 1e-9
    thresholds: list[float] = [0.0]
    for r in element_extents[1:]:
        thresholds.append(t_px * node_extent / max(float(r), eps))
    return _apply_monotonicity_guard(thresholds, "extent_min_pixel_sizes")


def lod_thresholds(
    method: LodThresholdMethod = "extent",
    *,
    element_counts: List[int],
    element_extents: Optional[List[float]] = None,
    node_extent: Optional[float] = None,
    base_pixel_size: Optional[float] = None,
) -> List[float]:
    """Derive ``min_pixel_size`` thresholds by the selected method.

    ``"extent"`` (default) → :func:`extent_min_pixel_sizes` (physically-anchored
    W/r) when ``element_extents`` and a positive ``node_extent`` are available;
    otherwise it transparently falls back to ``"count"`` (e.g. a non-gsplat
    hand-built tree with no extent data). ``"count"`` → :func:`derive_min_pixel_sizes`
    (scene-relative √N). ``base_pixel_size`` is the per-method anchor (``None`` →
    the method's default: ~1.5 px target for ``extent``, ~10 px for ``count``).
    """
    if (
        method == "extent"
        and element_extents is not None
        and node_extent is not None
        and node_extent > 0
        # Every finer level must have a usable (positive) extent; a non-positive
        # one (degenerate zero-size or empty level) would otherwise produce a
        # spurious huge threshold via the eps-clamp. The coarsest extent (index 0)
        # is unused (its threshold is the 0.0 floor), so it is exempt.
        and all(r > 0 for r in element_extents[1:])
    ):
        return extent_min_pixel_sizes(element_extents, node_extent, base_pixel_size)
    return derive_min_pixel_sizes(element_counts, base_pixel_size)


# ────────────────────────────────────────────────────────────────────────
# Validator
# ────────────────────────────────────────────────────────────────────────


def validate_lod_group(group: "Node") -> None:
    """Check that a kind=lod ``Group`` is well-formed.

    Raises ``ValueError`` if:

    - the group has zero children;
    - ``default_level`` is out of range (``not 0 <= default_level <
      len(children)``);
    - any child is missing ``min_pixel_size`` in its attrs;
    - the per-child ``min_pixel_size`` values are not strictly monotonic
      increasing in insertion order.

    Call this manually before finalizing if you want eager validation;
    otherwise the viewer falls back to silently ignoring malformed
    children at load time.
    """
    if not group.children:
        raise ValueError(f"LOD group '{group.path or group.name}' has no children")
    n_children = len(group.children)
    default_level = int(group.attrs.get("default_level", 0))
    if not 0 <= default_level < n_children:
        raise ValueError(
            f"LOD group '{group.path or group.name}' has "
            f"default_level={default_level}, must be in [0, {n_children})"
        )
    prev = float("-inf")
    for i, child in enumerate(group.children):
        if "min_pixel_size" not in child.attrs:
            raise ValueError(
                f"LOD-group child {i} ({child.name!r}) is missing "
                "'min_pixel_size' in its attrs"
            )
        value = float(child.attrs["min_pixel_size"])
        if value <= prev:
            raise ValueError(
                f"LOD-group child {i} ({child.name!r}) has "
                f"min_pixel_size={value}, must be strictly greater than "
                f"previous child's {prev}"
            )
        prev = value


# ─────────────────────────────────────────────────────────────────────
# Shared substitutive-LOD axis resolver (Points + Lines)
# ─────────────────────────────────────────────────────────────────────
#
# Both geometries coarsen by lifting elements to gsplats and running the gsplat
# substitutive pipeline, so the ``substitutive_lod=`` kwarg vocabulary is
# identical. Keep ONE implementation here so the per-geometry resolvers
# (``resolve_substitutive_axis_points`` / ``resolve_substitutive_axis_lines``)
# can never drift.

#: Defaults for ``substitutive_lod=True`` / ``substitutive_lod=dict()`` — mirror
#: the gsplat substitutive defaults (compression_factor=4, levels=3, auto method).
DEFAULT_SUBSTITUTIVE_K: int = 4
DEFAULT_SUBSTITUTIVE_LEVELS: int = 3
DEFAULT_SUBSTITUTIVE_METHOD: str = "auto"
#: Accepted substitutive reduction methods (passed to make_substitutive_lod).
SUBSTITUTIVE_METHODS = frozenset(
    {"auto", "kmeans", "kmeans_lloyd", "greedy", "greedy_lloyd"}
)


def _validate_coarsen_dims_spec(value: Any) -> Any:
    """Shape/type-validate the raw ``coarsen_dims`` spec value (no scene yet).

    Accepts ``None``, the sentinels ``"display"`` / ``"all"``, or a non-empty
    list/tuple of dim names (str) and/or column indices (int). Names and the
    ``"display"`` default are resolved against the scene later by
    :func:`resolve_coarsen_dims`.
    """
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("display", "displayed"):
            return "display"
        if v in ("all", "*"):
            return "all"
        raise ValueError(
            f"coarsen_dims string must be 'display' or 'all'; got {value!r}"
        )
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("coarsen_dims must be non-empty")
        out: list[Any] = []
        for x in value:
            if isinstance(x, bool):
                raise ValueError("coarsen_dims entries must be int or str, not bool")
            if isinstance(x, int):
                out.append(int(x))
            elif isinstance(x, str):
                out.append(x)
            else:
                raise ValueError(
                    "coarsen_dims entries must be int (column index) or str "
                    f"(dimension name); got {type(x).__name__}"
                )
        return out
    raise TypeError(
        "coarsen_dims must be None, 'display'/'all', or a list of dim "
        f"names/indices; got {type(value).__name__}"
    )


def resolve_coarsen_dims(scene: Any, n_cols: int, raw: Any) -> Optional[tuple]:
    """Resolve a raw ``coarsen_dims`` spec into concrete center-column indices.

    ``scene`` supplies the dimension metadata; ``n_cols`` is the lifted gsplat
    dimensionality (== the position columns being coarsened). Returns a sorted
    tuple of allowed-coarsen column indices, or ``None`` meaning "coarsen over
    all dims" (no barrier — the historical behavior).

    Default (``raw is None``) is **Auto**: coarsen over the scene's *displayed*
    dims and group by the *non-displayed* dims. This only applies when the
    positions are aligned with the scene (``n_cols == scene.ndim``); otherwise
    (dim_order / extend_to_all reshaped the columns) Auto safely falls back to
    all-dims. ``"display"`` is the explicit form of Auto and errors if unaligned;
    ``"all"`` forces all-dims; a list resolves names via ``Dimensions.get_index``
    and ints as direct column indices.
    """
    dims = getattr(scene, "_dimensions", None) if scene is not None else None
    aligned = dims is not None and int(dims.ndim) == int(n_cols)

    def _finalize(idxs: Any) -> Optional[tuple]:
        norm = sorted({int(i) for i in idxs})
        if not norm:
            raise ValueError("coarsen_dims must be non-empty")
        for i in norm:
            if i < 0 or i >= n_cols:
                raise ValueError(
                    f"coarsen_dims index {i} out of range for {n_cols} dims"
                )
        return None if len(norm) == n_cols else tuple(norm)

    if raw == "all":
        return None
    if raw is None or raw == "display":
        if not aligned:
            if raw == "display":
                raise ValueError(
                    "coarsen_dims='display' requires positions aligned with the "
                    f"scene dims (got {n_cols} columns, scene ndim "
                    f"{getattr(dims, 'ndim', '?')}). Pass explicit indices."
                )
            return None  # Auto, unaligned -> safe all-dims fallback
        assert dims is not None  # implied by `aligned`
        displayed = [d for d in dims.displayed if 0 <= d < n_cols]
        non_displayed = [d for d in range(n_cols) if d not in set(displayed)]
        if not non_displayed:
            return None  # nothing to group by -> coarsen everything (no-op barrier)
        return _finalize(displayed)
    # Explicit list of names / indices.
    idxs: list[int] = []
    for x in raw:
        if isinstance(x, str):
            if not aligned:
                # Names resolve to scene-dim indices, which only equal center
                # columns when positions span the scene 1:1. Refuse rather than
                # silently mapping a name to the wrong column.
                raise ValueError(
                    "coarsen_dims by name requires positions aligned with the "
                    f"scene dims (got {n_cols} columns, scene ndim "
                    f"{getattr(dims, 'ndim', '?')}). Pass explicit column indices."
                )
            idxs.append(int(dims.get_index(x)))  # type: ignore[union-attr]
        else:
            idxs.append(int(x))
    return _finalize(idxs)


def resolve_substitutive_axis(spec: Any, geometry: str) -> Optional[Dict[str, Any]]:
    """Normalize the ``substitutive_lod=`` kwarg into a spec dict (or ``None``).

    Geometry-agnostic — ``geometry`` ("Points"/"Lines") only flavours the error
    message. Vocabulary:

    * ``None`` / ``False`` → no-op (caller writes a flat / additive node).
    * ``True`` / ``dict()`` → defaults (K=4, levels=3, method="auto").
    * ``dict(...)`` → keys ``compression_factor`` (alias ``K``), ``levels``
      (alias ``n_lods``), ``method`` (reduction algorithm), ``lod_method``
      (``"extent"`` default | ``"count"``), ``extent_percentile`` (default 90),
      ``extent_anisotropy`` (default True), ``base_pixel_size``,
      ``truncation_radius``, ``device``, ``seed``, ``min_pixel_sizes`` (explicit,
      strict-ascending). Unrecognized keys raise.
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise TypeError(
            f"substitutive_lod must be None, bool, or dict; got {type(spec).__name__}"
        )
    kwargs = dict(spec)

    compression_factor = int(
        kwargs.pop("compression_factor", kwargs.pop("K", DEFAULT_SUBSTITUTIVE_K))
    )
    if compression_factor < 2:
        raise ValueError(f"compression_factor must be >= 2, got {compression_factor}")

    levels = int(kwargs.pop("levels", kwargs.pop("n_lods", DEFAULT_SUBSTITUTIVE_LEVELS)))
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    method = str(kwargs.pop("method", DEFAULT_SUBSTITUTIVE_METHOD)).replace("-", "_")
    if method not in SUBSTITUTIVE_METHODS:
        raise ValueError(
            f"method must be one of {sorted(SUBSTITUTIVE_METHODS)}; got {method!r}"
        )

    base_pixel_size = kwargs.pop("base_pixel_size", None)
    if base_pixel_size is not None:
        base_pixel_size = float(base_pixel_size)
        if base_pixel_size <= 0:
            raise ValueError(f"base_pixel_size must be positive, got {base_pixel_size}")

    # LOD switching-threshold method + its extent-mode knobs (see lod_thresholds /
    # extent_min_pixel_sizes). ``lod_method`` is distinct from ``method`` above,
    # which selects the substitutive *reduction* algorithm.
    lod_method = str(kwargs.pop("lod_method", "extent"))
    if lod_method not in ("extent", "count"):
        raise ValueError(
            f"lod_method must be 'extent' or 'count', got {lod_method!r}"
        )
    extent_percentile = float(kwargs.pop("extent_percentile", 90.0))
    if not (0.0 < extent_percentile <= 100.0):
        raise ValueError(
            f"extent_percentile must lie in (0, 100], got {extent_percentile}"
        )
    extent_anisotropy = bool(kwargs.pop("extent_anisotropy", True))

    truncation_radius = float(kwargs.pop("truncation_radius", 3.0))
    if truncation_radius <= 0:
        raise ValueError(f"truncation_radius must be > 0, got {truncation_radius}")

    device = kwargs.pop("device", "auto")
    seed = kwargs.pop("seed", None)
    if seed is not None:
        seed = int(seed)

    min_pixel_sizes = kwargs.pop("min_pixel_sizes", None)
    if min_pixel_sizes is not None:
        min_pixel_sizes = [float(m) for m in min_pixel_sizes]
        _assert_strict_ascending(
            min_pixel_sizes, "substitutive_lod=dict(min_pixel_sizes=...)"
        )

    # Dims coarsening may cluster over; complement = hard grouping barriers.
    # Shape/type only here (scene dims aren't known yet); names + the "display"
    # default are resolved in the scene-aware adder via resolve_coarsen_dims().
    coarsen_dims = _validate_coarsen_dims_spec(kwargs.pop("coarsen_dims", None))

    if kwargs:
        raise ValueError(
            f"substitutive_lod for {geometry}: unrecognized keys {sorted(kwargs)}. "
            "Valid keys: compression_factor (K), levels (n_lods), method, "
            "lod_method, extent_percentile, extent_anisotropy, base_pixel_size, "
            "truncation_radius, device, seed, min_pixel_sizes, coarsen_dims."
        )

    return {
        "compression_factor": compression_factor,
        "levels": levels,
        "method": method,
        "lod_method": lod_method,
        "extent_percentile": extent_percentile,
        "extent_anisotropy": extent_anisotropy,
        "base_pixel_size": base_pixel_size,
        "truncation_radius": truncation_radius,
        "device": device,
        "seed": seed,
        "min_pixel_sizes": min_pixel_sizes,
        "coarsen_dims": coarsen_dims,
    }
