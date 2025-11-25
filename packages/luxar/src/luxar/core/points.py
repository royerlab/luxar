"""luxar.points – Defines the Points node for points data in Luxar scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from arbol import aprint

from ..core.node import Node
from ..typing_utils.enums import NodeType

if TYPE_CHECKING:
    from ..io.writer import ZarrWriterProtocol


class Points(Node):
    """Points node that holds metadata about points data.

    This class is a lightweight metadata container. Actual point data is written
    immediately to Zarr via the writer interface and not kept in memory.

    Points are created internally by Scene.add_points() and should not be
    instantiated directly by users.

    Args:
        name: Name of the points node
        metadata: Metadata dictionary about the written points
        parent: Parent node in hierarchy
        writer: Writer interface for progressive writing
        **attrs: Additional attributes
    """

    def __init__(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent: Optional[Node] = None,
        writer: Optional[ZarrWriterProtocol] = None,
        **attrs: Any,
    ) -> None:
        """Initialize a Points node.

        Args:
            name: Name of the points node
            metadata: Metadata dictionary about the written points
            parent: Parent node in the scene graph
            writer: Writer interface for progressive writing
            **attrs: Additional attributes for the node
        """
        # Initialize parent Node FIRST
        node_type = NodeType.POINTS.value
        super().__init__(name, parent=parent, writer=writer, type=node_type, **attrs)

        # Store metadata AFTER super().__init__() to avoid being overwritten
        # (Node.__init__() sets self._metadata = {}, which would overwrite ours)
        self._metadata = metadata or {}

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
        return int(self._metadata.get("n_points", 0))

    @property
    def has_colors(self) -> bool:
        """Check if points have colors."""
        return bool(self._metadata.get("has_colors", False))

    @property
    def has_radii(self) -> bool:
        """Check if points have radii."""
        return bool(self._metadata.get("has_radii", False))

    @property
    def has_sharpness(self) -> bool:
        """Check if points have sharpness."""
        return bool(self._metadata.get("has_sharpness", False))
