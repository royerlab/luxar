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
thresholds using a √(N_finer / N_coarsest) heuristic anchored at
``BASE_PIXEL_SIZE``.

Everything here is type-agnostic and shared across all leaf geometries
(Points, Lines, GSplats) and the Partition kind. The geometry-specific
``lod_group=`` / ``additive_lod=`` axis resolvers live next to their data
types (e.g. ``lod.gsplats`` for ``GSplatData``).

This module hosts:

* The ``derive_min_pixel_sizes`` heuristic and its monotonicity guard.
* The free-function validator ``validate_lod_group``, callable on any
  ``Group`` whose ``attrs["kind"] == "lod"``.
* The shared ``resolve_display_type`` helper used by both LOD and Partition
  kinds, which walks down through nested specialized groups to determine what
  geometry type the user sees this layer as, and ``compute_lod_display_type``
  which derives an LOD group's ``display_type`` from its finest child.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

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
        raise ValueError(f"coarsest child must have at least 1 element, got {n0}")
    bps = BASE_PIXEL_SIZE if base_pixel_size is None else float(base_pixel_size)
    if bps <= 0:
        raise ValueError(f"base_pixel_size must be positive, got {bps}")
    thresholds: list[float] = [0.0]
    for n in element_counts[1:]:
        thresholds.append(bps * (n / n0) ** 0.5)
    # Defensive: guarantee strict monotonicity even when later children have
    # the same (or fewer) elements as a previous level. Use a *relative*
    # bump (×1.1) rather than a fixed +1px so near-equal-count levels
    # separate proportionally to their scale — a fixed pixel nudge places
    # the switch threshold at a meaningless absolute value for large
    # ladders. ``thresholds[i-1]`` is always > 0 when this fires (i >= 2),
    # so the bump is strictly increasing.
    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1] * 1.1
    # Belt-and-braces: same invariant the explicit-list path is checked
    # against in the geometry-specific resolvers. Free now that the loop
    # above runs.
    _assert_strict_ascending(thresholds, "derive_min_pixel_sizes")
    return thresholds


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
