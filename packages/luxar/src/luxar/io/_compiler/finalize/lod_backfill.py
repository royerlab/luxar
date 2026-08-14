"""Finalize-time back-fill of position_bounds / display_type on kind=lod groups.

Also hosts :func:`warn_one_part_partition_anchors`, which reports (never rewrites)
the one anchor mistake no *producer* can catch — see its docstring.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import zarr
from arbol import aprint

from ....core.group.lod.group import MAX_COVERAGE_FRACTION
from ....typing_utils._format_contract import GEOMETRY_TYPES
from ....typing_utils.geometry_capabilities import require_lod_display_type


def finalize_lod_position_bounds(store: zarr.Group) -> None:
    """Back-fill missing ``position_bounds`` on kind=lod / kind=partition wrappers.

    Walks the zarr tree post-order and, for every ``kind == 'lod'`` or
    ``kind == 'partition'`` group without a ``position_bounds``, computes
    the union of its children's ``position_bounds`` (recursing into
    nested ``kind="lod"`` / ``kind="partition"`` wrappers and plain
    groups). Two producers leave a wrapper without aggregate bounds:

    - The convenience-builder path (``_add_gsplats_as_lod_group``) leaves
      the ``kind="lod"`` parent without bounds because each leaf carries
      its own, so the viewer's ``loadLodGroupNode`` saw empty bounds for a
      nested LOD-of-LOD construction and skipped that level in projection.
    - The **scene-graft** path (``graft_gsplat_node`` → ``add_partition_group``)
      composes a ``kind="partition"`` wrapper without computing the
      children-union that the standalone writer
      (``gsplat_tree.write_gsplat_node``) stamps at write time. A grafted
      partition — e.g. the ``multiscale`` recipe's fine branch, or a bare
      partition via ``add_gsplats_from_file`` / ``gsplat convert`` — would
      otherwise reach the viewer without wrapper-level bounds, losing
      partition-unit frustum culling and graft/standalone parity.

    **Never overwrites** an authored ``position_bounds`` — only
    fills missing values. Children with empty / mismatched bounds
    are skipped in the union (same convention as the viewer's
    registry projection).
    """

    def union(
        a: Optional[Dict[str, List[float]]], b: Optional[Dict[str, List[float]]]
    ) -> Optional[Dict[str, List[float]]]:
        if a is None:
            return b
        if b is None:
            return a
        a_min, a_max = a["min"], a["max"]
        b_min, b_max = b["min"], b["max"]
        if len(a_min) != len(b_min) or len(a_min) != len(a_max):
            # Mismatched dimensionality — skip b. Same defensive
            # fallback as the viewer's registry.
            return a
        return {
            "min": [min(a_min[i], b_min[i]) for i in range(len(a_min))],
            "max": [max(a_max[i], b_max[i]) for i in range(len(a_max))],
        }

    def resolve(group: "zarr.Group") -> Optional[Dict[str, List[float]]]:
        """Return the position_bounds of a group (leaf or wrapper).

        Returns None when the group has no leaves with bounds (e.g.
        empty group or all-mismatched children) so callers can skip.
        """
        attrs = dict(group.attrs)
        authored = attrs.get("position_bounds")
        if isinstance(authored, dict) and "min" in authored and "max" in authored:
            return {
                "min": list(authored["min"]),
                "max": list(authored["max"]),
            }
        # No authored bounds → recurse into children (groups only;
        # zarr arrays don't have descendants).
        acc: Optional[Dict[str, List[float]]] = None
        for child_name in group.group_keys():
            acc = union(acc, resolve(group[child_name]))
        return acc

    def walk(group: "zarr.Group") -> None:
        attrs = dict(group.attrs)
        kind = attrs.get("kind")
        if kind in ("lod", "partition") and "position_bounds" not in attrs:
            aggregated = resolve(group)
            if aggregated is not None:
                group.attrs["position_bounds"] = aggregated
                aprint(
                    f"  📐 Back-filled position_bounds on "
                    f"kind={kind} group {group.path or '/'}"
                )
        for child_name in group.group_keys():
            walk(group[child_name])

    walk(store)


def finalize_lod_display_types(store: zarr.Group) -> None:
    """Back-fill missing ``display_type`` on kind=lod groups.

    Walks the zarr tree and, for every group whose attrs declare
    ``kind == 'lod'`` without a ``display_type``, resolves one from
    the **finest** child's own type (recursing through nested
    kind=lod / kind=partition groups). The convenience-builder path
    (``_add_gsplats_as_lod_group``) already sets ``display_type``
    explicitly; this hook serves explicit-builder constructions
    where the user wrote ``add_lod_group(...)`` + a mix of leaf
    types and never set the parent's ``display_type`` themselves.

    **Never overwrites** an authored ``display_type`` — only fills
    missing values.
    """

    def resolve(group: "zarr.Group") -> str:
        """Return the display_type of a group (leaf or wrapper)."""
        attrs = dict(group.attrs)
        t = attrs.get("type")
        if t in GEOMETRY_TYPES:
            return str(t)  # leaf
        kind = attrs.get("kind")
        if kind in ("lod", "partition") and "display_type" in attrs:
            return str(attrs["display_type"])
        # Plain group or kind=lod / kind=partition without display_type
        # → recurse into children. Children of an lod_group are stored in
        # coarsest→finest order, so the finest is the LAST one. Order by the
        # child's ``child_index`` attr (the canonical insertion order) rather
        # than by name: name-sort puts ``child_10`` before ``child_2``, which
        # would pick the wrong "finest" for a >=10-level ladder. Fall back to
        # name order for any child missing ``child_index`` (legacy data).
        #
        # Consider child GROUPS only. ``keys()`` lists a group's arrays too, so
        # a node that reaches this branch while holding datasets — any leaf
        # whose ``type`` is outside ``GEOMETRY_TYPES``, or a wrapper that mixes
        # arrays with sub-groups — would otherwise pick a ``zarr.Array`` as its
        # "finest child" and recurse into it, raising a bare AttributeError on
        # ``Array.keys()``. Every child-iteration site in this module uses
        # ``group_keys()`` for the same reason.
        child_names = list(group.group_keys())
        if not child_names:
            return ""  # nothing to resolve

        def _order_key(name: str) -> tuple[float, str]:
            idx = dict(group[name].attrs).get("child_index")
            return (float(idx) if isinstance(idx, (int, float)) else float("inf"), name)

        finest_child = group[max(child_names, key=_order_key)]
        return resolve(finest_child)

    def walk(group: "zarr.Group") -> None:
        attrs = dict(group.attrs)
        if attrs.get("kind") == "lod" and "display_type" not in attrs:
            resolved = resolve(group)
            if resolved:
                # This is the route that actually runs for an explicit-builder
                # ladder, so the LOD-capability rule is enforced here and not
                # only at the add-time entry points. A geometry type with no LOD
                # ladder must not be stamped: the result loads nowhere, and once
                # written the store looks authored rather than back-filled.
                require_lod_display_type(
                    resolved, f"kind=lod group {group.path or '/'}"
                )
                group.attrs["display_type"] = resolved
                aprint(
                    f"  📐 Back-filled display_type={resolved!r} on "
                    f"kind=lod group {group.path or '/'}"
                )
        for child_name in group.group_keys():
            walk(group[child_name])

    walk(store)


#: How close a child's ``coverage_fraction`` must be to the fills-screen anchor
#: to read as the per-tile one. The derived value is an exact power-of-two
#: rescale, so this only absorbs JSON round-tripping.
_ANCHOR_TOLERANCE: float = 1e-9


def _tile_anchor(group_attrs: dict) -> float:
    """The fills-screen (per-tile) anchor in this group's OWN selector units.

    ``selector="screen-area"`` thresholds are literal screen-area fractions, so
    a tile anchors at area 1.0 (``PARTITION_FINEST_AREA``); the legacy
    ``"coverage"`` diagonal metric anchors at ``MAX_COVERAGE_FRACTION`` = 4.0.
    """
    from luxar.core.group.lod.group import PARTITION_FINEST_AREA

    if group_attrs.get("selector") == "screen-area":
        return PARTITION_FINEST_AREA
    return MAX_COVERAGE_FRACTION


def _is_tile_anchored(group: "zarr.Group") -> bool:
    """Does this kind=lod group's finest child sit on the fills-screen anchor?

    Reads every child group's ``coverage_fraction`` rather than trusting insertion
    order: the ladder is written coarsest→finest, but a hand-built group need not
    be, and the question here is only whether the ladder REACHES the ceiling —
    the ceiling being selector-dependent (see :func:`_tile_anchor`).
    """
    anchor = _tile_anchor(dict(group.attrs))
    for child_name in group.group_keys():
        raw = dict(group[child_name].attrs).get("coverage_fraction")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            if float(raw) >= anchor - _ANCHOR_TOLERANCE:
                return True
    return False


def warn_one_part_partition_anchors(store: zarr.Group) -> None:
    """Warn about a fills-screen ladder under a ONE-PART ``kind=partition``.

    The per-tile anchor (``coverage_fraction`` up to ``MAX_COVERAGE_FRACTION`` =
    4.0) is correct only when the partition is a real TILING: a tile's projected
    bbox diagonal is intrinsically a fraction of the whole object's, so a
    whole-object anchor would put every tile on its finest level while the object
    is merely full-frame. A ONE-part partition inverts that — its single part's
    bbox IS the whole object — and the ladder then holds its finest level back
    until the object *overfills* the screen.

    Every producer that can see the final sibling count already excludes that
    shape (``gsplats/lod/recipes.py::build_adaptive`` and both gsplat tree
    writers). The SCENE-ADDER path
    (``core/group/lod/group.py::derive_coverage_fractions``) cannot: it is handed
    only the insertion point, and part 0's ladder is derived before part 1 has
    been added. Finalize is the first moment the sibling count exists, which is
    why the check lives here.

    **Reports, never rewrites.** An authored ``coverage_fractions=[0, …, 4.0]``
    list and a derived one are indistinguishable on disk, so silently
    re-anchoring would override a deliberate choice. One warning per offending
    ``kind=lod`` group.

    The test mirrors the producers' rule exactly: a ladder is legitimately
    tile-anchored when ANY enclosing partition is a real tiling, because that is
    what both gsplat writers thread down (``under_partition or len(children) >
    1`` — an inner one-part wrapper ORs the outer binding in rather than clearing
    it). So a warning needs BOTH a one-part partition above the ladder AND no
    genuine tiling further up: ``partition(2 parts) → partition(1 part) → lod``
    is still inside one tile and is correct, while a genuine multi-part partition
    nested inside a one-part wrapper is likewise not blamed for its children's
    (correct) anchors. When it does fire, the offender named is the NEAREST
    enclosing partition.
    """

    def walk(
        group: "zarr.Group", lone_partition: Optional[str], under_tiling: bool
    ) -> None:
        attrs = dict(group.attrs)
        kind = attrs.get("kind")
        child_names = list(group.group_keys())
        if (
            kind == "lod"
            and lone_partition is not None
            and not under_tiling
            and _is_tile_anchored(group)
        ):
            aprint(
                f"  ⚠️  kind=lod group '{group.path or '/'}' is anchored at "
                f"coverage_fraction={_tile_anchor(attrs):g} — the per-TILE, "
                f"fills-screen anchor — but its enclosing kind=partition group "
                f"'{lone_partition}' holds only ONE part. A one-part partition is "
                "not a tiling: that part's bbox IS the whole object, so this "
                "ladder will hold its finest level back until the object "
                "OVERFILLS the viewport instead of showing it at a normal "
                "full-frame view. Drop the partition wrapper, or pass an "
                "explicit coverage_fractions=[...] ending at the whole-object "
                "anchor (0.5 under selector='screen-area'; 1.0 legacy)."
            )
        if kind == "partition":
            if len(child_names) > 1:
                under_tiling = True
            else:
                lone_partition = group.path or "/"
        for child_name in child_names:
            walk(group[child_name], lone_partition, under_tiling)

    walk(store, None, False)
