"""DataNode abstract base class for data-bearing nodes.

This module defines the abstract base class for all nodes that contain
visualization data (Points, Lines, GSplats, Mesh).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from .node import Node

if TYPE_CHECKING:
    from ..io.writer import ZarrWriterProtocol


class DataNode(Node, ABC):
    """Abstract base class for nodes that contain visualization data.

    All data nodes share:
    - Immediate writing to Zarr (no data kept in memory)
    - Type-specific metadata
    - Element count property
    - Semantic type mapping for encoding

    Subclasses must implement:
    - n_elements property (returns count of primary elements)
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Node] = None,
        writer: Optional["ZarrWriterProtocol"] = None,
        type: str = "datanode",
        metadata: Optional[Dict[str, Any]] = None,
        **attrs: Any,
    ) -> None:
        """Initialize DataNode.

        Args:
            name: Node name
            parent: Parent node
            writer: Writer for progressive writing
            type: Node type discriminator
            metadata: Type-specific metadata
            **attrs: Additional zarr attributes
        """
        # Call parent first (critical: see spec note on constructor order)
        super().__init__(name, parent=parent, writer=writer, type=type, **attrs)

        # Then set DataNode-specific attributes
        self._metadata = metadata or {}

    @property
    @abstractmethod
    def n_elements(self) -> int:
        """Number of primary elements.

        What this counts depends on node type:
        - Points: number of points (n_points)
        - Lines: number of vertices (n_vertices)
        - GSplats: number of splats (n_splats)

        Returns:
            Count of primary elements

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        ...

    @property
    def ndim(self) -> int:
        """Dimensionality of primary data.

        Returns:
            Number of dimensions (from metadata)

        Raises:
            ValueError: If the node has no 'ndim' key in its metadata
        """
        value = self._metadata.get("ndim")
        if value is None:
            raise ValueError(
                f"Node '{self.name}' has no dimensionality metadata "
                f"(expected 'ndim' key in metadata). "
                f"Available keys: {list(self._metadata.keys())}"
            )
        return int(value)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Type-specific metadata.

        Returns:
            Metadata dictionary
        """
        return self._metadata
