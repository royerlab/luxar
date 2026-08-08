"""luxar.points – Defines the Points node for points data in Luxar scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from arbol import aprint

from ..typing_utils.enums import NodeType
from .datanode import DataNode

if TYPE_CHECKING:
    from ..io.writer import ZarrWriterProtocol


class Points(DataNode):
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
        parent: Optional["DataNode"] = None,
        writer: Optional["ZarrWriterProtocol"] = None,
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
        # Initialize parent DataNode (which handles metadata storage)
        node_type = NodeType.POINTS.value
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
            n_points = metadata.get("n_points", 0)
            n_dims = metadata.get("ndim", 3)
            aprint(
                f"✓ Points node '{name}' created with {n_points:,} points in {n_dims}D."
            )

    @property
    def n_elements(self) -> int:
        """Number of primary elements (points).

        Returns:
            Number of points
        """
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

    @property
    def has_scalars(self) -> bool:
        """Check if points have scalar values for colormap lookup."""
        return bool(self._metadata.get("has_scalars", False))

    @property
    def has_labels(self) -> bool:
        """Check if points have per-element string labels for hover tooltips."""
        return bool(self._metadata.get("has_labels", False))

    @property
    def has_image_labels(self) -> bool:
        """Check if points have per-element image labels for hover thumbnails."""
        return bool(self._metadata.get("has_image_labels", False))

    @property
    def max_radius(self) -> float:
        """Get maximum point radius."""
        return float(self._metadata.get("max_radius", 0.0))

    @property
    def has_spatial_index(self) -> bool:
        """Check if points have spatial indexing enabled."""
        return bool(self._metadata.get("has_spatial_index", False))

    @property
    def ordering(self) -> str:
        """Get spatial ordering method (e.g., 'morton', 'hilbert', 'none')."""
        return str(self._metadata.get("ordering", "none"))

    @property
    def join(self) -> Optional[str]:
        """Get the line join style for this node.

        Re-declared (identically to :class:`Node`, ``None`` when unset rather than
        a substituted default) only so the setter below can be overridden — a
        property's getter and setter travel together.
        """
        value = self.attrs.get("join")
        return str(value) if value is not None else None

    @join.setter
    def join(self, value: Any) -> None:
        """Refuse ``join``, which is lines-only (issue #790).

        The adder refuses an explicitly-authored ``join=`` at add time; without
        this override the inherited :class:`Node` setter (and ``set_join``, which
        assigns through this property) would re-open the same door one line later,
        and the dead attr would persist to zarr.
        """
        from .group.compositing import reject_lines_only_join_assignment

        reject_lines_only_join_assignment("points", self.name, value)
