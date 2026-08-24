"""Finalize-time diagnostics for overlapping geometry blend states."""

from __future__ import annotations

from dataclasses import dataclass

import zarr
from arbol import aprint

from ....typing_utils.constants import DEFAULT_BLENDING_MODE_BY_GEOMETRY
from ..bounds import WorldBoundsLeaf, collect_world_bounds


@dataclass(frozen=True)
class _BlendLeaf:
    leaf: WorldBoundsLeaf
    mode: str
    opacity: float
    lod_branches: tuple[tuple[str, str], ...]


def _effective_leaf(store: zarr.Group, leaf: WorldBoundsLeaf) -> _BlendLeaf:
    mode: str | None = None
    opacity = 1.0
    lod_branches: list[tuple[str, str]] = []
    group = store
    parts = leaf.path.split("/") if leaf.path else []

    attrs = dict(group.attrs)
    if "blending_mode" in attrs:
        mode = str(attrs["blending_mode"])
    opacity *= float(attrs.get("opacity", 1.0))

    for part in parts:
        if dict(group.attrs).get("kind") == "lod":
            lod_branches.append((group.path, part))
        group = group[part]
        attrs = dict(group.attrs)
        if "blending_mode" in attrs:
            mode = str(attrs["blending_mode"])
        opacity *= float(attrs.get("opacity", 1.0))

    opacity = min(1.0, max(0.0, opacity))
    resolved = mode or DEFAULT_BLENDING_MODE_BY_GEOMETRY[leaf.geometry_type]
    if leaf.geometry_type == "mesh" and resolved == "volumetric":
        resolved = "opaque"
    return _BlendLeaf(leaf, resolved, opacity, tuple(lod_branches))


def _can_coexist(left: _BlendLeaf, right: _BlendLeaf) -> bool:
    left_branches = dict(left.lod_branches)
    return all(
        left_branches.get(path, branch) == branch for path, branch in right.lod_branches
    )


def _intersects(left: WorldBoundsLeaf, right: WorldBoundsLeaf) -> bool:
    dimensions = min(len(left.bounds["min"]), len(right.bounds["min"]))
    if dimensions == 0:
        return False
    return all(
        left.bounds["min"][axis] <= right.bounds["max"][axis]
        and right.bounds["min"][axis] <= left.bounds["max"][axis]
        for axis in range(dimensions)
    )


def _depth_tests(node: _BlendLeaf) -> bool:
    return node.mode != "additive"


def _depth_writes(node: _BlendLeaf) -> bool:
    if node.mode == "opaque":
        return True
    if node.mode != "normal" or node.opacity < 0.99:
        return False
    return node.leaf.geometry_type in {"lines", "mesh"}


def _internally_sorted(node: _BlendLeaf) -> bool:
    return node.mode in {"normal", "volumetric"}


def warn_overlapping_blending(store: zarr.Group) -> None:
    """Warn once per co-visible overlapping pair with unsafe blend semantics."""
    leaves = [_effective_leaf(store, leaf) for leaf in collect_world_bounds(store)]
    for index, left in enumerate(leaves):
        for right in leaves[index + 1 :]:
            if not _can_coexist(left, right) or not _intersects(left.leaf, right.leaf):
                continue

            left_writes = _depth_writes(left)
            right_writes = _depth_writes(right)
            if (left.mode == "additive" and right_writes) or (
                right.mode == "additive" and left_writes
            ):
                additive = left if left.mode == "additive" else right
                writer = right if additive is left else left
                aprint(
                    f"  ⚠️  overlapping nodes '{additive.leaf.path}' ({additive.mode}) and "
                    f"'{writer.leaf.path}' ({writer.mode}) mix depth-ignoring and "
                    "depth-writing geometry; use blending_mode='luminous' on the "
                    "additive node unless X-ray rendering is intentional."
                )
                continue

            if (
                _depth_tests(left)
                and _depth_tests(right)
                and not left_writes
                and not right_writes
                and _internally_sorted(left)
                and _internally_sorted(right)
            ):
                aprint(
                    f"  ⚠️  overlapping depth-sorted nodes '{left.leaf.path}' ({left.mode}) "
                    f"and '{right.leaf.path}' ({right.mode}) have view-dependent cross-node "
                    "order; make one node additive or separate their bounds."
                )
