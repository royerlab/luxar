"""luxar.gsplats – Defines the GSplats node for Gaussian splat data in Luxar scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from arbol import aprint

from ..typing_utils.enums import NodeType
from .datanode import DataNode

if TYPE_CHECKING:
    from ..io.writer import ZarrWriterProtocol


class GSplats(DataNode):
    """GSplats node that holds metadata about Gaussian splat data.

    This class is a lightweight metadata container. Actual splat data is written
    immediately to Zarr via the writer interface and not kept in memory.

    GSplats are created internally by Scene.add_gsplats() and should not be
    instantiated directly by users.

    Args:
        name: Name of the gsplats node
        metadata: Metadata dictionary about the written gsplats
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
        """Initialize a GSplats node.

        Args:
            name: Name of the gsplats node
            metadata: Metadata dictionary about the written gsplats
            parent: Parent node in the scene graph
            writer: Writer interface for progressive writing
            **attrs: Additional attributes for the node
        """
        # Initialize parent DataNode (which handles metadata storage)
        node_type = NodeType.GSPLATS.value
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
            n_splats = metadata.get("n_splats", 0)
            n_dims = metadata.get("ndim", 3)
            msg = f"✓ GSplats node '{name}' created with {n_splats:,} splats"
            aprint(f"{msg} in {n_dims}D.")

    @property
    def n_elements(self) -> int:
        """Number of primary elements (splats).

        Returns:
            Number of splats
        """
        return int(self._metadata.get("n_splats", 0))

    @property
    def has_colors(self) -> bool:
        """Check if splats have colors."""
        return bool(self._metadata.get("has_colors", False))

    @property
    def has_labels(self) -> bool:
        """Check if gsplats have per-element string labels for hover tooltips."""
        return bool(self._metadata.get("has_labels", False))

    @property
    def has_image_labels(self) -> bool:
        """Check if gsplats have per-element image labels for hover thumbnails."""
        return bool(self._metadata.get("has_image_labels", False))

    @property
    def ordering(self) -> str:
        """Get spatial ordering type."""
        return str(self._metadata.get("ordering", "none"))

    @property
    def amplitude_range(self) -> Dict[str, float]:
        """Get amplitude range."""
        return self._metadata.get("amplitude_range", {"min": 0.0, "max": 1.0})  # type: ignore

    @property
    def center_bounds(self) -> Dict[str, list]:
        """Get center coordinate bounds."""
        return self._metadata.get("center_bounds", {"min": [], "max": []})  # type: ignore
