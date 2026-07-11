"""luxar.lines – Defines the Lines node for line/curve data in Luxar scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, cast

from arbol import aprint

from ..typing_utils.enums import NodeType
from .datanode import DataNode

if TYPE_CHECKING:
    from ..io.writer import ZarrWriterProtocol

LineType = Literal["segments", "polyline", "loop", "indexed"]


class Lines(DataNode):
    """Lines node that holds metadata about line/curve data.

    This class is a lightweight metadata container. Actual line data is written
    immediately to Zarr via the writer interface and not kept in memory.

    Lines are created internally by Scene.add_lines() and should not be
    instantiated directly by users.

    Args:
        name: Name of the lines node
        metadata: Metadata dictionary about the written lines
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
        """Initialize a Lines node.

        Args:
            name: Name of the lines node
            metadata: Metadata dictionary about the written lines
            parent: Parent node in the scene graph
            writer: Writer interface for progressive writing
            **attrs: Additional attributes for the node
        """
        # Initialize parent DataNode (which handles metadata storage)
        node_type = NodeType.LINES.value
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
            n_segments = metadata.get("n_segments", 0)
            line_type = metadata.get("line_type", "unknown")
            n_dims = metadata.get("ndim", 3)
            aprint(
                f"✓ Lines node '{name}' created with {n_vertices:,} vertices, "
                f"{n_segments:,} segments ({line_type}) in {n_dims}D."
            )

    @property
    def n_elements(self) -> int:
        """Number of primary elements (vertices).

        Returns:
            Number of vertices
        """
        return int(self._metadata.get("n_vertices", 0))

    @property
    def n_segments(self) -> int:
        """Get number of line segments."""
        return int(self._metadata.get("n_segments", 0))

    @property
    def line_type(self) -> LineType:
        """Get original line type (user-specified)."""
        # Support both old 'line_type' and new 'original_line_type' keys
        result = self._metadata.get(
            "original_line_type", self._metadata.get("line_type", "polyline")
        )
        return cast(LineType, result)

    @property
    def has_colors(self) -> bool:
        """Check if lines have colors."""
        return bool(self._metadata.get("has_colors", False))

    @property
    def has_sharpness(self) -> bool:
        """Check if lines have sharpness."""
        return bool(self._metadata.get("has_sharpness", False))

    @property
    def has_scalars(self) -> bool:
        """Check if lines have scalar values for colormap lookup."""
        return bool(self._metadata.get("has_scalars", False))

    @property
    def has_labels(self) -> bool:
        """Check if lines have per-element string labels for hover tooltips."""
        return bool(self._metadata.get("has_labels", False))

    @property
    def has_image_labels(self) -> bool:
        """Check if lines have per-element image labels for hover thumbnails."""
        return bool(self._metadata.get("has_image_labels", False))

    @property
    def max_width(self) -> float:
        """Get maximum line width."""
        return float(self._metadata.get("max_width", 0.0))

    @property
    def has_spatial_index(self) -> bool:
        """Check if lines have spatial indexing enabled."""
        return bool(self._metadata.get("has_spatial_index", False))

    @property
    def ordering(self) -> str:
        """Get spatial ordering method (e.g., 'morton', 'hilbert', 'none')."""
        return str(self._metadata.get("ordering", "none"))
