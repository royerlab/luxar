"""Finalize-time back-fill of position_bounds / display_type on kind=lod groups."""

from __future__ import annotations

from typing import Dict, List, Optional

import zarr
from arbol import aprint


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
        if t in ("points", "lines", "gsplats"):
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
        # whose ``type`` is not in the tuple above, or a wrapper that mixes
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
                group.attrs["display_type"] = resolved
                aprint(
                    f"  📐 Back-filled display_type={resolved!r} on "
                    f"kind=lod group {group.path or '/'}"
                )
        for child_name in group.group_keys():
            walk(group[child_name])

    walk(store)
