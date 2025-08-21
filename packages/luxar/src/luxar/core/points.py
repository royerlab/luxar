"""luxar.points – Defines the Points node for point cloud data in Luxar scenes."""

from __future__ import annotations

from typing import Any, Dict, Optional

from arbol import aprint

from ..core.node import Node
from ..typing_utils.protocols import NodeType


class Points(Node):
    """Point-cloud node that holds metadata about point data.

    This class is a lightweight metadata container. Actual data is written
    immediately to Zarr via the writer interface and not kept in memory.
    This class is intended for internal use via Scene.add_points().

    Args:
        name: Name of the point cloud node
        metadata: Metadata dictionary about the written points
        parent: Parent node in hierarchy
        writer: Writer interface for progressive writing
        **attrs: Additional attributes
    """

    def __init__(
        self,
        name: str,
        positions: Optional[Any] = None,  # Ignored, for compatibility
        metadata: Optional[Dict[str, Any]] = None,
        parent: Optional[Node] = None,
        writer: Optional[Any] = None,  # ZarrWriterProtocol
        **attrs: Any,
    ) -> None:
        """Initialize a Points node.

        Args:
            name: Name of the point cloud node
            positions: Ignored, for compatibility only
            metadata: Metadata dictionary about the written points
            parent: Parent node in the scene graph
            writer: Writer interface for progressive writing
            **attrs: Additional attributes for the node
        """
        # Store metadata
        self._metadata = metadata or {}

        # Initialize parent Node
        node_type: NodeType = "points"
        super().__init__(
            name, group=None, parent=parent, writer=writer, type=node_type, **attrs
        )

        # Log creation
        if metadata:
            n_points = metadata.get("n_points", 0)
            n_dims = metadata.get("dims", 3)
            aprint(
                f"✓ Points node '{name}' created with {n_points:,} points in {n_dims}D."
            )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the points."""
        return self._metadata

    @property
    def n_points(self) -> int:
        """Get number of points."""
        return self._metadata.get("n_points", 0)

    @property
    def has_colors(self) -> bool:
        """Check if points have colors."""
        return self._metadata.get("has_colors", False)

    @property
    def has_radii(self) -> bool:
        """Check if points have radii."""
        return self._metadata.get("has_radii", False)

    @property
    def has_sharpness(self) -> bool:
        """Check if points have sharpness."""
        return self._metadata.get("has_sharpness", False)
