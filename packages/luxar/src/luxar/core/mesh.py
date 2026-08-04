"""luxar.mesh – Defines the Mesh node for triangle-surface data in Luxar scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, cast

from arbol import aprint

from ..typing_utils.enums import NodeType
from .datanode import DataNode

if TYPE_CHECKING:
    from ..io.writer import ZarrWriterProtocol

ShadingMode = Literal["smooth", "flat"]


class Mesh(DataNode):
    """Mesh node that holds metadata about triangle-surface data.

    This class is a lightweight metadata container. Actual mesh data is written
    immediately to Zarr via the writer interface and not kept in memory.

    Meshes are created internally by Scene.add_mesh() and should not be
    instantiated directly by users.

    Args:
        name: Name of the mesh node
        metadata: Metadata dictionary about the written mesh
        parent: Parent node in hierarchy
        writer: Writer interface for progressive writing
        **attrs: Additional attributes
    """

    def __init__(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent: Optional["DataNode"] = None,
        writer: Optional["ZarrWriterProtocol"] = None,
        **attrs: Any,
    ) -> None:
        """Initialize a Mesh node.

        Args:
            name: Name of the mesh node
            metadata: Metadata dictionary about the written mesh
            parent: Parent node in the scene graph
            writer: Writer interface for progressive writing
            **attrs: Additional attributes for the node
        """
        # Initialize parent DataNode (which handles metadata storage)
        node_type = NodeType.MESH.value
        super().__init__(
            name,
            parent=parent,
            writer=writer,
            type=node_type,
            metadata=metadata,
            **attrs,
        )

        # Log creation
        if metadata:
            n_vertices = metadata.get("n_vertices", 0)
            n_faces = metadata.get("n_faces", 0)
            n_dims = metadata.get("ndim", 3)
            aprint(
                f"✓ Mesh node '{name}' created with {n_vertices:,} vertices, "
                f"{n_faces:,} faces in {n_dims}D."
            )

    @property
    def n_elements(self) -> int:
        """Number of primary elements (vertices).

        Vertices, not faces — matching ``Lines.n_elements``, which also counts
        vertices rather than segments. The primary element is the thing the
        per-element attribute arrays (colors / scalars / normals / labels) are
        indexed by, and for a mesh that is the vertex.

        Returns:
            Number of vertices
        """
        return int(self._metadata.get("n_vertices", 0))

    @property
    def n_faces(self) -> int:
        """Get number of triangles."""
        return int(self._metadata.get("n_faces", 0))

    @property
    def has_normals(self) -> bool:
        """Check if the mesh has per-vertex normals."""
        return bool(self._metadata.get("has_normals", False))

    @property
    def normal_dims(self) -> Optional[List[int]]:
        """The three dimension indices the stored normals describe.

        ``None`` when the mesh has no normals. Never inferred: normals are a
        display-space quantity, and which three dimensions they belong to is
        recorded explicitly (an implicit "first three" is wrong for any mesh whose
        leading dimension is not spatial).
        """
        dims = self._metadata.get("normal_dims")
        return [int(d) for d in dims] if dims is not None else None

    @property
    def has_colors(self) -> bool:
        """Check if the mesh has per-vertex colors."""
        return bool(self._metadata.get("has_colors", False))

    @property
    def has_scalars(self) -> bool:
        """Check if the mesh has scalar values for colormap lookup."""
        return bool(self._metadata.get("has_scalars", False))

    @property
    def has_labels(self) -> bool:
        """Check if the mesh has per-vertex string labels for hover tooltips."""
        return bool(self._metadata.get("has_labels", False))

    @property
    def has_image_labels(self) -> bool:
        """Check if the mesh has per-vertex image labels for hover thumbnails."""
        return bool(self._metadata.get("has_image_labels", False))

    @property
    def shading(self) -> ShadingMode:
        """Get the shading mode ('smooth' or 'flat')."""
        return cast(ShadingMode, str(self._metadata.get("shading", "flat")))

    @property
    def double_sided(self) -> bool:
        """Whether back faces render."""
        return bool(self._metadata.get("double_sided", True))

    @property
    def ordering(self) -> str:
        """Get spatial ordering method.

        Always ``"none"`` for a mesh: v1 has no spatial index because the loader
        is whole-node, so there is nothing for a chunk index to skip. The property
        exists for symmetry with the sibling geometry types, and reads the stamped
        attr rather than returning a literal so it stays honest if that changes.
        """
        return str(self._metadata.get("ordering", "none"))
