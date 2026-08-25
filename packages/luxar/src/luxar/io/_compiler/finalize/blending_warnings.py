"""Finalize-time diagnostics for overlapping geometry blend states."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import zarr
from arbol import aprint

from ....core.dimensions import Dimensions
from ....typing_utils.constants import (
    DEFAULT_BLENDING_MODE,
    DEFAULT_BLENDING_MODE_BY_GEOMETRY,
)
from ....typing_utils.enums import BlendingMode
from ..bounds import WorldBoundsLeaf, collect_world_bounds

_BLENDING_MODES = frozenset(mode.value for mode in BlendingMode)
_MESH_SUPPORTED_BLENDING_MODES = frozenset(
    {"opaque", "normal", "additive", "luminous", "max"}
)


@dataclass(frozen=True)
class _BlendLeaf:
    leaf: WorldBoundsLeaf
    mode: str
    opacity: float
    lod_branches: tuple[tuple[str, str], ...]
    mode_explicit: bool = False

    @property
    def owner_path(self) -> str:
        return self.leaf.owner_path or self.leaf.path


def _effective_leaf(leaf: WorldBoundsLeaf) -> _BlendLeaf:
    mode = leaf.blending_mode
    resolved = (
        mode
        if mode is not None
        else DEFAULT_BLENDING_MODE_BY_GEOMETRY.get(
            leaf.geometry_type, DEFAULT_BLENDING_MODE
        )
    )
    if resolved not in _BLENDING_MODES:
        resolved = "normal"
    if leaf.geometry_type == "mesh" and resolved not in _MESH_SUPPORTED_BLENDING_MODES:
        resolved = "opaque"
    return _BlendLeaf(
        leaf,
        resolved,
        leaf.opacity,
        leaf.lod_branches,
        mode_explicit=mode is not None,
    )


def _can_coexist(left: _BlendLeaf, right: _BlendLeaf) -> bool:
    left_branches = dict(left.lod_branches)
    return all(
        left_branches.get(path, branch) == branch for path, branch in right.lod_branches
    )


def _intersects(
    left: WorldBoundsLeaf,
    right: WorldBoundsLeaf,
    displayed_dimensions: set[int],
) -> bool:
    dimensions = min(len(left.bounds["min"]), len(right.bounds["min"]))
    if dimensions == 0:
        return False
    for axis in range(dimensions):
        left_min = left.bounds["min"][axis]
        left_max = left.bounds["max"][axis]
        right_min = right.bounds["min"][axis]
        right_max = right.bounds["max"][axis]
        if axis in displayed_dimensions:
            if left_min >= right_max or right_min >= left_max:
                return False
        elif left_min > right_max or right_min > left_max:
            return False
    return True


def _contains(left: WorldBoundsLeaf, right: WorldBoundsLeaf) -> bool:
    dimensions = min(len(left.bounds["min"]), len(right.bounds["min"]))
    return dimensions > 0 and all(
        left.bounds["min"][axis] <= right.bounds["min"][axis]
        and left.bounds["max"][axis] >= right.bounds["max"][axis]
        for axis in range(dimensions)
    )


def _depth_writes(node: _BlendLeaf) -> bool:
    if node.mode == "opaque":
        return True
    if node.mode != "normal" or node.opacity < 0.99:
        return False
    return node.leaf.geometry_type in {"lines", "mesh"}


def _internally_sorted(node: _BlendLeaf) -> bool:
    return node.mode in {"normal", "volumetric"}


def _candidate_pairs(
    leaves: list[_BlendLeaf], displayed_dimensions: set[int]
) -> Iterator[tuple[_BlendLeaf, _BlendLeaf]]:
    """Yield spatial candidates with a sweep on one displayed dimension."""
    sweep_axis = next(
        (
            axis
            for axis in sorted(displayed_dimensions)
            if all(axis < len(node.leaf.bounds["min"]) for node in leaves)
        ),
        None,
    )
    if sweep_axis is None:
        for index, left in enumerate(leaves):
            for right in leaves[index + 1 :]:
                yield left, right
        return

    ordered = sorted(
        leaves,
        key=lambda node: (node.leaf.bounds["min"][sweep_axis], node.leaf.path),
    )
    for index, left in enumerate(ordered):
        left_max = left.leaf.bounds["max"][sweep_axis]
        for right in ordered[index + 1 :]:
            if right.leaf.bounds["min"][sweep_axis] >= left_max:
                break
            yield left, right


def warn_overlapping_blending(
    store: zarr.Group, world_leaves: list[WorldBoundsLeaf] | None = None
) -> None:
    """Warn once per co-visible node with unsafe overlapping blend semantics."""
    if "scene_dimensions" not in store.attrs:
        return
    dimensions = Dimensions.from_dict(store.attrs["scene_dimensions"])
    displayed_dimensions = set(dimensions.displayed[:3])
    if world_leaves is None:
        world_leaves = collect_world_bounds(store)
    leaves = [_effective_leaf(leaf) for leaf in world_leaves]
    warned_additive: set[str] = set()
    warned_sorted: set[str] = set()
    for left, right in _candidate_pairs(leaves, displayed_dimensions):
        if left.owner_path == right.owner_path:
            continue
        if not _can_coexist(left, right) or not _intersects(
            left.leaf, right.leaf, displayed_dimensions
        ):
            continue

        left_writes = _depth_writes(left)
        right_writes = _depth_writes(right)
        if (left.mode == "additive" and not left.mode_explicit and right_writes) or (
            right.mode == "additive" and not right.mode_explicit and left_writes
        ):
            additive = left if left.mode == "additive" else right
            if additive.owner_path in warned_additive:
                continue
            writer = right if additive is left else left
            warned_additive.add(additive.owner_path)
            aprint(
                f"  ⚠️  overlapping nodes '{additive.owner_path}' ({additive.mode}) and "
                f"'{writer.owner_path}' ({writer.mode}) mix depth-ignoring and "
                "depth-writing geometry; use blending_mode='luminous' on the "
                "additive node unless X-ray rendering is intentional."
            )
            continue

        if (
            not left_writes
            and not right_writes
            and _internally_sorted(left)
            and _internally_sorted(right)
            and (_contains(left.leaf, right.leaf) or _contains(right.leaf, left.leaf))
        ):
            if left.owner_path in warned_sorted or right.owner_path in warned_sorted:
                continue
            warned_sorted.update((left.owner_path, right.owner_path))
            aprint(
                f"  ⚠️  overlapping order-dependent nodes '{left.owner_path}' ({left.mode}) "
                f"and '{right.owner_path}' ({right.mode}) have view-dependent cross-node "
                "order; make one node additive or separate their bounds."
            )
