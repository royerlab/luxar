"""Scene-bounds computation, union, and world-space transform expansion."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import zarr
from numpy.typing import NDArray


def compute_position_bounds(
    positions: NDArray[np.float32],
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


def expand_bounds_with_transforms(
    store: zarr.Group,
    scene_bounds: Optional[Dict[str, List[float]]],
) -> Optional[Dict[str, List[float]]]:
    """Expand scene-level position bounds into world space.

    Walks the zarr tree, composes the world-space transform chain for
    each leaf node, applies it to the per-node (local) position bounds,
    and stores the union of all world-space bounds as the scene-level
    ``position_bounds``.

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
        store: The opened zarr store (in r+ mode)
        scene_bounds: Current scene-level bounds, or None.

    Returns:
        The updated scene bounds (world-space union), or the input
        ``scene_bounds`` unchanged when there is nothing to expand.
    """
    # Guard: need both scene_dimensions and scene_bounds
    if scene_bounds is None:
        return scene_bounds
    if "scene_dimensions" not in store.attrs:
        return scene_bounds

    from ...core.dimensions import Dimensions
    from ...core.transforms import (
        read_transform_from_zarr,
        transform_bounding_box,
    )
    from ...validation.nd_transforms import (
        apply_nd_transform_to_bounds,
        compose_nd_transforms,
    )

    dimensions = Dimensions.from_dict(store.attrs["scene_dimensions"])
    # The 4x4 transform's x/y/z axes map, in order, to the displayed
    # dimensions — matching how the viewer projects nD positions to the
    # mesh's x/y/z before applying the node transform.
    displayed = dimensions.displayed[:3]

    def apply_matrix_to_displayed_dims(
        bounds: dict[str, list[float]], matrix: "np.ndarray"
    ) -> dict[str, list[float]]:
        """Apply a 4x4 world matrix to the displayed dims of ``bounds``."""
        min_vals = list(bounds["min"])
        max_vals = list(bounds["max"])
        # Gather the displayed-dim sub-box into 3D (missing axes -> 0,
        # mirroring the viewer's zero-padding of < 3 displayed dims).
        lo3 = [0.0, 0.0, 0.0]
        hi3 = [0.0, 0.0, 0.0]
        for axis, dim in enumerate(displayed):
            if dim < len(min_vals):
                lo3[axis] = min_vals[dim]
                hi3[axis] = max_vals[dim]
        new_lo, new_hi = transform_bounding_box(matrix, lo3, hi3)
        for axis, dim in enumerate(displayed):
            if dim < len(min_vals):
                min_vals[dim] = float(new_lo[axis])
                max_vals[dim] = float(new_hi[axis])
        return {"min": min_vals, "max": max_vals}

    # Collect all world-space bounds from leaf nodes
    all_world_bounds: list[dict[str, list[float]]] = []

    def walk(
        group: zarr.Group,
        nd_chain: list[dict],
        world_matrix: "np.ndarray",
        has_matrix: bool,
    ) -> None:
        """Recursively walk zarr tree, composing both transform families."""
        attrs = dict(group.attrs)

        chain = list(nd_chain)
        nd_t = attrs.get("nd_transform", None)
        if nd_t:
            chain.append(nd_t)

        node_matrix = world_matrix
        node_has_matrix = has_matrix
        raw_transform = attrs.get("transform", None)
        if raw_transform is not None:
            local_matrix = read_transform_from_zarr(list(raw_transform))
            # world = ancestors @ this  (child transform applied first).
            node_matrix = world_matrix @ local_matrix
            node_has_matrix = True

        node_type = attrs.get("type", None)
        if node_type in ("points", "lines", "gsplats"):
            # Leaf node with geometry
            local_bounds = attrs.get("position_bounds", None)
            if local_bounds:
                transformed = local_bounds
                if chain:
                    world_nd_t = compose_nd_transforms(*chain)
                    transformed = apply_nd_transform_to_bounds(
                        transformed, world_nd_t, dimensions
                    )
                if node_has_matrix:
                    transformed = apply_matrix_to_displayed_dims(
                        transformed, node_matrix
                    )
                all_world_bounds.append(transformed)

        # Recurse into child groups
        for child_name in sorted(group.group_keys()):
            walk(group[child_name], chain, node_matrix, node_has_matrix)

    walk(store, [], np.eye(4, dtype=np.float64), False)

    # If no leaf nodes found, nothing to do
    if not all_world_bounds:
        return scene_bounds

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
