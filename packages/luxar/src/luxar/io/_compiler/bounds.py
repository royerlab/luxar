"""Scene-bounds computation, union, and world-space transform expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import zarr
from numpy.typing import NDArray

from ...core.dimensions import Dimensions
from ...core.transforms import read_transform_from_zarr, transform_bounding_box
from ...typing_utils._format_contract import GEOMETRY_TYPES
from ...typing_utils.aliases import (
    PositionArray,
)
from ...validation.nd_transforms import (
    apply_nd_transform_to_bounds,
    compose_nd_transforms,
)


@dataclass(frozen=True)
class WorldBoundsLeaf:
    """One geometry leaf, its author-facing owner, and world-space bounds."""

    path: str
    geometry_type: str
    bounds: dict[str, list[float]]
    blending_mode: str | None = None
    opacity: float = 1.0
    owner_path: str | None = None


@dataclass
class _WorldBoundsCollector:
    dimensions: Dimensions
    displayed: list[int]
    leaves: list[WorldBoundsLeaf]

    def _apply_matrix(
        self, bounds: dict[str, list[float]], matrix: NDArray[np.float64]
    ) -> dict[str, list[float]]:
        min_vals = list(bounds["min"])
        max_vals = list(bounds["max"])
        lo3 = [0.0, 0.0, 0.0]
        hi3 = [0.0, 0.0, 0.0]
        for axis, dimension in enumerate(self.displayed):
            if dimension < len(min_vals):
                lo3[axis] = min_vals[dimension]
                hi3[axis] = max_vals[dimension]
        new_lo, new_hi = transform_bounding_box(matrix, lo3, hi3)
        for axis, dimension in enumerate(self.displayed):
            if dimension < len(min_vals):
                min_vals[dimension] = float(new_lo[axis])
                max_vals[dimension] = float(new_hi[axis])
        return {"min": min_vals, "max": max_vals}

    def _append_geometry(
        self,
        group: zarr.Group,
        attrs: dict,
        nd_chain: list[dict],
        world_matrix: NDArray[np.float64],
        has_matrix: bool,
        blending_mode: str | None,
        opacity: float,
        owner_path: str | None,
    ) -> None:
        node_type = attrs.get("type")
        # Keep this keyed to the format contract: a newly authorable geometry
        # must reach scene bounds rather than being silently skipped here.
        if node_type not in GEOMETRY_TYPES:
            return
        local_bounds = attrs.get("position_bounds")
        if not local_bounds:
            return
        transformed = local_bounds
        if nd_chain:
            transformed = apply_nd_transform_to_bounds(
                transformed,
                compose_nd_transforms(*nd_chain),
                self.dimensions,
            )
        if has_matrix:
            transformed = self._apply_matrix(transformed, world_matrix)
        self.leaves.append(
            WorldBoundsLeaf(
                group.path,
                node_type,
                transformed,
                blending_mode,
                min(1.0, max(0.0, opacity)),
                owner_path or group.path,
            )
        )

    def walk(
        self,
        group: zarr.Group,
        nd_chain: list[dict],
        world_matrix: NDArray[np.float64],
        has_matrix: bool,
        blending_mode: str | None,
        opacity: float,
        owner_path: str | None,
        is_root: bool = False,
    ) -> None:
        attrs = dict(group.attrs)
        node_owner_path = owner_path
        if node_owner_path is None and (
            attrs.get("type") in GEOMETRY_TYPES
            or attrs.get("kind") in {"lod", "partition"}
        ):
            node_owner_path = group.path
        node_blending_mode = blending_mode
        node_opacity = opacity
        if not (is_root and attrs.get("type") == "scene"):
            if "blending_mode" in attrs:
                node_blending_mode = str(attrs["blending_mode"])
            node_opacity *= float(attrs.get("opacity", 1.0))

        chain = list(nd_chain)
        nd_transform = attrs.get("nd_transform")
        if nd_transform:
            chain.append(nd_transform)

        node_matrix = world_matrix
        node_has_matrix = has_matrix
        raw_transform = attrs.get("transform")
        if raw_transform is not None:
            # world = ancestors @ this (child transform applied first)
            node_matrix = world_matrix @ read_transform_from_zarr(list(raw_transform))
            node_has_matrix = True

        self._append_geometry(
            group,
            attrs,
            chain,
            node_matrix,
            node_has_matrix,
            node_blending_mode,
            node_opacity,
            node_owner_path,
        )
        for child_name in sorted(group.group_keys()):
            self.walk(
                group[child_name],
                chain,
                node_matrix,
                node_has_matrix,
                node_blending_mode,
                node_opacity,
                node_owner_path,
            )


def compute_position_bounds(
    positions: PositionArray,
) -> Dict[str, List[float]]:
    """Compute nD bounding box from positions array.

    Args:
        positions: Positions array of shape (N, D)

    Returns:
        Dictionary with 'min' and 'max' keys, each containing a list of D floats
    """
    # Compute min and max along each dimension
    if positions.shape[0] == 0:
        n_dims = positions.shape[1] if positions.ndim == 2 else 0
        return {"min": [0.0] * n_dims, "max": [0.0] * n_dims}
    min_vals = positions.min(axis=0).tolist()
    max_vals = positions.max(axis=0).tolist()

    return {"min": min_vals, "max": max_vals}


def update_scene_bounds(
    scene_bounds: Optional[Dict[str, List[float]]],
    node_bounds: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """Update scene-level bounds by taking union with node bounds.

    Returns the updated scene bounds (for the caller to store back). Pass the
    current ``scene_bounds`` (or None for the first node).

    Args:
        scene_bounds: Current scene-level bounds, or None for the first node.
        node_bounds: Dictionary with 'min' and 'max' keys from a node

    Returns:
        The updated scene-level bounds.
    """
    if scene_bounds is None:
        # First node - initialize scene bounds
        return {
            "min": list(node_bounds["min"]),
            "max": list(node_bounds["max"]),
        }

    # Expand scene bounds to include this node
    # Handle potentially different dimensionalities by extending with the node's values
    node_ndim = len(node_bounds["min"])
    scene_ndim = len(scene_bounds["min"])

    if node_ndim > scene_ndim:
        # Extend scene bounds with new dimensions from this node
        scene_bounds["min"].extend(node_bounds["min"][scene_ndim:])
        scene_bounds["max"].extend(node_bounds["max"][scene_ndim:])
        scene_ndim = node_ndim

    # Update min/max for each dimension
    for i in range(min(node_ndim, scene_ndim)):
        scene_bounds["min"][i] = min(scene_bounds["min"][i], node_bounds["min"][i])
        scene_bounds["max"][i] = max(scene_bounds["max"][i], node_bounds["max"][i])

    return scene_bounds


def collect_world_bounds(store: zarr.Group) -> list[WorldBoundsLeaf]:
    """Collect transform-expanded world-space bounds for every geometry leaf.

    Walks the zarr tree, composes the world-space transform chain for each leaf
    node, and applies it to the per-node (local) position bounds. The same pass
    carries effective compositing attrs and the author-facing owner for finalize
    consumers that need them.

    Two independent transform families are composed down the hierarchy
    and applied together:

    - The 4x4 spatial ``transform`` (translate / rotate / scale) moves
      the **displayed** dimensions (the ones the viewer maps to mesh
      x/y/z). The matrix is applied to the box by transforming all 8
      corners (see :func:`transform_bounding_box`), which is correct
      under rotation — only transforming the (min, max) corner pair
      would underestimate the rotated extent.
    - The per-dimension ``nd_transform`` (affine scale/offset) moves the
      **non-displayed** dimensions (slider axes).

    Getting the spatial 4x4 into the scene bounds is load-bearing for the
    viewer: per-frame dynamic clipping derives near/far from a bounding
    sphere built from this metadata. If a node is translated far from the
    origin but its transform is ignored here, the sphere is too small and
    that geometry gets clipped as the camera rotates.

    Args:
        store: The opened zarr store.

    Returns:
        Geometry leaves with world-space bounds, in deterministic path order.
    """
    if "scene_dimensions" not in store.attrs:
        return []

    dimensions = Dimensions.from_dict(store.attrs["scene_dimensions"])
    # The 4x4 transform's x/y/z axes map, in order, to the displayed
    # dimensions — matching how the viewer projects nD positions to the
    # mesh's x/y/z before applying the node transform.
    collector = _WorldBoundsCollector(dimensions, dimensions.displayed[:3], [])
    collector.walk(
        store,
        [],
        np.eye(4, dtype=np.float64),
        False,
        None,
        1.0,
        None,
        is_root=True,
    )
    return collector.leaves


def expand_bounds_with_transforms(
    store: zarr.Group,
    scene_bounds: Optional[Dict[str, List[float]]],
    leaves: list[WorldBoundsLeaf] | None = None,
) -> Optional[Dict[str, List[float]]]:
    """Expand scene-level position bounds into world space.

    Uses :func:`collect_world_bounds` as the single transform-aware leaf walk,
    then stores the union as the scene-level ``position_bounds``. A caller that
    has already collected the leaves may pass them to share the snapshot with
    other finalize consumers.

    Args:
        store: The opened zarr store (in r+ mode)
        scene_bounds: Current scene-level bounds, or None.
        leaves: Optional pre-collected world-space leaves.

    Returns:
        The updated scene bounds (world-space union), or the input
        ``scene_bounds`` unchanged when there is nothing to expand.
    """
    if scene_bounds is None:
        return scene_bounds

    if leaves is None:
        leaves = collect_world_bounds(store)

    # If no leaf nodes found, nothing to do
    if not leaves:
        return scene_bounds

    all_world_bounds = [leaf.bounds for leaf in leaves]

    # Union all world-space bounds (same logic as update_scene_bounds)
    world_scene_bounds: dict[str, list[float]] = {
        "min": list(all_world_bounds[0]["min"]),
        "max": list(all_world_bounds[0]["max"]),
    }
    for bounds in all_world_bounds[1:]:
        node_ndim = len(bounds["min"])
        scene_ndim = len(world_scene_bounds["min"])

        if node_ndim > scene_ndim:
            world_scene_bounds["min"].extend(bounds["min"][scene_ndim:])
            world_scene_bounds["max"].extend(bounds["max"][scene_ndim:])
            scene_ndim = node_ndim

        for i in range(min(node_ndim, scene_ndim)):
            world_scene_bounds["min"][i] = min(
                world_scene_bounds["min"][i], bounds["min"][i]
            )
            world_scene_bounds["max"][i] = max(
                world_scene_bounds["max"][i], bounds["max"][i]
            )

    # Overwrite scene-level bounds with world-space bounds
    store.attrs["position_bounds"] = world_scene_bounds
    return world_scene_bounds
