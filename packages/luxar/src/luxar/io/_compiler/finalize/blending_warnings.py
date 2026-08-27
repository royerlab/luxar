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
_MAX_CLUSTER_PARTICIPANTS = 5


@dataclass(frozen=True)
class _BlendLeaf:
    leaf: WorldBoundsLeaf
    mode: str
    opacity: float
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
        mode_explicit=mode is not None,
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


def _cluster_root(parents: dict[str, str], path: str) -> str:
    root = path
    while parents[root] != root:
        root = parents[root]
    while path != root:
        parent = parents[path]
        parents[path] = root
        path = parent
    return root


def _merge_sorted_overlap(
    parents: dict[str, str],
    modes: dict[str, set[str]],
    geometry_types: dict[str, set[str]],
    containers: set[str],
    left: _BlendLeaf,
    right: _BlendLeaf,
) -> None:
    for node in (left, right):
        parents.setdefault(node.owner_path, node.owner_path)
        modes.setdefault(node.owner_path, set()).add(node.mode)
        geometry_types.setdefault(node.owner_path, set()).add(node.leaf.geometry_type)
    if _contains(left.leaf, right.leaf):
        containers.add(left.owner_path)
    if _contains(right.leaf, left.leaf):
        containers.add(right.owner_path)
    left_root = _cluster_root(parents, left.owner_path)
    right_root = _cluster_root(parents, right.owner_path)
    if left_root != right_root:
        parents[max(left_root, right_root)] = min(left_root, right_root)


def _warn_sorted_overlap_clusters(
    parents: dict[str, str],
    modes: dict[str, set[str]],
    geometry_types: dict[str, set[str]],
    containers: set[str],
    displayed_dimensions: tuple[int, ...],
) -> None:
    clusters: dict[str, list[str]] = {}
    for path in sorted(parents):
        clusters.setdefault(_cluster_root(parents, path), []).append(path)
    for cluster in sorted(clusters.values()):
        ordered = sorted(cluster, key=lambda path: (path not in containers, path))
        shown = ordered[:_MAX_CLUSTER_PARTICIPANTS]
        participants = ", ".join(
            f"'{path}' ({' + '.join(sorted(modes[path]))})"
            f"{' [contains another node]' if path in containers else ''}"
            for path in shown
        )
        omitted = len(ordered) - len(shown)
        if omitted:
            participants += f", and {omitted} more"
        cluster_modes = {mode for path in cluster for mode in modes[path]}
        cluster_geometry_types = {
            geometry_type for path in cluster for geometry_type in geometry_types[path]
        }
        displayed_advice = ""
        expected_displayed_dimensions = tuple(range(len(displayed_dimensions)))
        if displayed_dimensions != expected_displayed_dimensions:
            displayed_advice = (
                f" Displayed dimensions are position columns {displayed_dimensions}, not "
                f"{expected_displayed_dimensions}. Put the displayed dimensions first to keep "
                "exact BSP ordering."
            )
        if len(cluster_geometry_types) == 1 and len(cluster_modes) == 1:
            remedy = (
                ' Merge them into one node and pass partition={"max_elements": N} to order '
                "disjoint BSP cells back-to-front. This preserves placement and appearance for "
                "same-type, same-mode Points, Lines, Mesh, and Gaussian Splats."
            )
        else:
            remedy = (
                " Merging cannot preserve this cluster because its geometry types or blending "
                "modes differ."
            )
        aprint(
            f"  ⚠️  overlapping order-dependent nodes {participants} have view-dependent "
            f"cross-node order.{displayed_advice}{remedy} For an emissive medium, additive "
            "blending is order-independent but changes surface appearance; otherwise separate "
            "their bounds."
        )


def warn_overlapping_blending(
    store: zarr.Group, world_leaves: list[WorldBoundsLeaf] | None = None
) -> None:
    """Warn about co-visible nodes with unsafe overlapping blend semantics."""
    if "scene_dimensions" not in store.attrs:
        return
    dimensions = Dimensions.from_dict(store.attrs["scene_dimensions"])
    displayed_dimensions = set(dimensions.displayed[:3])
    if world_leaves is None:
        world_leaves = collect_world_bounds(store)
    leaves = [_effective_leaf(leaf) for leaf in world_leaves]
    warned_additive: set[str] = set()
    sorted_parents: dict[str, str] = {}
    sorted_modes: dict[str, set[str]] = {}
    sorted_geometry_types: dict[str, set[str]] = {}
    sorted_containers: set[str] = set()
    for left, right in _candidate_pairs(leaves, displayed_dimensions):
        if left.owner_path == right.owner_path:
            continue
        if not _intersects(left.leaf, right.leaf, displayed_dimensions):
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
            _merge_sorted_overlap(
                sorted_parents,
                sorted_modes,
                sorted_geometry_types,
                sorted_containers,
                left,
                right,
            )

    _warn_sorted_overlap_clusters(
        sorted_parents,
        sorted_modes,
        sorted_geometry_types,
        sorted_containers,
        tuple(dimensions.displayed[:3]),
    )
