"""add_mesh body.

Pure function taking a ``group: Group`` parameter as the first arg. Called by
``Group.add_mesh`` (a thin signature + docstring + delegate) in
``core/group/group.py``.

Notably shorter than its siblings, and structurally so: mesh has no
``additive_lod`` / ``substitutive_lod`` / ``partition`` branches, because none of
those exist for a connected surface (MESH_NODE_SPEC.md §9). What remains is the
single-leaf write path the other adders reach after their LOD/partition
dispatch — plus one rejection those adders never need, since a specialized-group
parent is the one way a mesh could end up somewhere it cannot be rendered.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import numpy as np
from arbol import aprint

from ...mesh import Mesh
from ..compositing import sync_custom_colormap_attr
from ..dim_order import apply_dim_order_positions

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group


def _reject_specialized_parent(parent_node: "Node", name: str) -> None:
    """Refuse to write a mesh leaf under a ``kind=lod`` / ``kind=partition`` group.

    Mesh supports neither (spec §9), and the two group kinds fail differently if
    a mesh slips in:

    * ``kind=partition`` — ``add_partition_group`` already rejects
      ``display_type='mesh'``, but nothing stops a caller from creating a
      ``points`` partition and then adding a mesh child into it, which would make
      the group's declared display type a lie.
    * ``kind=lod`` — the ladder has no coarse stand-in for a surface, and the
      display type resolved from a mesh child is refused by the LOD guard. That
      guard fires at finalize, i.e. AFTER the mesh's arrays are on disk; catching
      it here keeps the failure fail-fast and leaves no partial node.

    Both are caller mistakes with no valid interpretation, so they raise rather
    than warn.
    """
    kind = parent_node.attrs.get("kind")
    if kind in ("lod", "partition"):
        raise ValueError(
            f"Cannot add mesh '{name}' to a kind={kind} group. Mesh supports "
            "neither LOD nor spatial partitioning yet: the additive/substitutive "
            "ladder reduces independent elements (a surface is connected), and a "
            "BSP cut needs vertex duplication at part boundaries. Add the mesh to "
            "a plain group instead."
        )


def add_mesh_impl(
    group: "Group",
    *,
    name: str,
    vertices: Any,
    faces: Any,
    normals: Any = None,
    normal_dims: Optional[Sequence[int]] = None,
    colors: Any = None,
    scalars: Any = None,
    shading: Optional[str] = None,
    double_sided: bool = True,
    labels: Optional[Union[List[str], Sequence[str]]] = None,
    image_labels: Optional[Any] = None,
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    **attrs: Any,
) -> Mesh:
    try:
        # Fail-fast pre-write gate: reject invalid names (empty/'/'/dot-prefixed —
        # an empty name resolves to the zarr ROOT group and would clobber the
        # scene root) and duplicate siblings BEFORE any zarr write. Node.__init__
        # re-checks both post-write (belt and braces).
        from ....validation.base import validate_node_name

        validate_node_name(name)
        (parent or group)._ensure_no_duplicate_child(name)
        _reject_specialized_parent(parent or group, name)

        scene = group._find_scene()

        vert_arr: np.ndarray = (
            vertices if isinstance(vertices, np.ndarray) else np.asarray(vertices)
        )
        if vert_arr.ndim != 2:
            raise ValueError(
                f"Vertices must have shape (V, D), got shape {vert_arr.shape}"
            )

        # Apply dim_order before validation. Vertices are coordinates and get
        # reordered like every other geometry type's positions; `faces` is INDEX
        # data addressing vertex ROWS, so it is deliberately NOT reordered —
        # permuting columns of the coordinate array leaves row indices valid.
        vert_arr, extend_to_all = apply_dim_order_positions(
            vert_arr, scene, dim_order, fill, extend_to_all
        )

        n_vertices = vert_arr.shape[0]
        ndim = vert_arr.shape[1]

        # Colormap / colors mutual exclusivity, matching the sibling adders.
        if colors is not None and attrs.get("colormap") is not None:
            raise ValueError(
                "Cannot specify both 'colors' and 'colormap'. Use one or the other."
            )
        if scalars is not None and attrs.get("colormap") is None:
            raise ValueError(
                "'scalars' requires a 'colormap' attribute to map values to colors."
            )

        faces_arr: np.ndarray = (
            faces if isinstance(faces, np.ndarray) else np.asarray(faces)
        )
        n_faces = int(faces_arr.size // 3)

        aprint(
            f"Adding mesh node '{name}' with {n_vertices:,} vertices and "
            f"{n_faces:,} faces in {ndim}D."
        )

        scene._validate_data_dimensions(vert_arr, name, data_type="vertices")

        final_extend_dims = scene._resolve_extend_to_all(
            extend_to_all, vert_arr, "mesh"
        )
        if final_extend_dims:
            attrs["extend_to_all"] = final_extend_dims
            aprint(f"  📡 Extending visibility across: {final_extend_dims}")

        parent_node = parent or group

        writer = group._require_scene_writer(scene)
        path = f"{parent_node.path}/{name}" if parent_node.path else name
        metadata = writer.write_mesh(
            path,
            vert_arr.astype(np.float32),
            faces_arr,
            normals=normals,
            normal_dims=normal_dims,
            colors=cast(Any, colors),
            scalars=scalars,
            shading=shading,
            double_sided=double_sided,
            labels=labels,
            image_labels=image_labels,
            **attrs,
        )

        # Sync colormap attr with what the compiler wrote to zarr
        sync_custom_colormap_attr(attrs)

        if labels is not None:
            scene._notify_labels_added()
        if image_labels is not None:
            scene._notify_image_labels_added()

        return Mesh(
            name,
            metadata=metadata,
            parent=cast(Any, parent_node),
            writer=writer,
            **attrs,
        )
    except (ValueError, TypeError) as e:
        aprint(f"Failed to add mesh node '{name}': {e}")
        raise ValueError(f"Could not add mesh '{name}': {e}") from e
