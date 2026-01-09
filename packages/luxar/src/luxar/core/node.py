"""luxar.node – Defines the Node class for Luxar scene graph nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from arbol import aprint

from ..typing_utils.aliases import GroupAttrs, SceneHierarchy, TransformMatrix

if TYPE_CHECKING:
    from ..io.writer import ZarrWriterProtocol


class Node:
    """A node in the Luxar scene graph.

    This class represents a single node in the hierarchical scene graph structure.
    Nodes are lightweight metadata containers that write data immediately through
    the writer interface without keeping Zarr groups in memory.

    Args:
        name: Name of the node
        parent: Parent node in the hierarchy
        writer: Writer interface for progressive writing
        **attrs: Additional attributes for the node
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Node] = None,
        writer: Optional[ZarrWriterProtocol] = None,
        **attrs: Any,
    ) -> None:
        """Initialize a scene graph node.

        Args:
            name: Name of the node
            parent: Parent node in the scene hierarchy
            writer: Writer interface for progressive data writing
            **attrs: Additional attributes to set on the node
        """
        self.name: str = name
        self._writer = writer
        self.parent: Optional[Node] = parent
        self.children: List[Node] = []
        self._metadata: Dict[str, Any] = {}  # Metadata storage
        self._attrs_cache: Dict[str, Any] = {}  # Attributes cache

        # Determine path in hierarchy
        if parent is not None:
            parent.children.append(self)
            parent_path = getattr(parent, "path", None)
            self.path = f"{parent_path}/{name}" if parent_path else name
        else:
            self.path = ""

        # Initialize or merge attributes
        if attrs:
            # Validate transform if present
            if "transform" in attrs:
                try:
                    from ..core.transforms import prepare_transform_for_zarr

                    # Use centralized function for consistent handling
                    attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])
                except Exception as e:
                    aprint(f"Invalid transform for node '{name}': {e}")
                    raise ValueError(f"Invalid transform: {e}") from e

            # Validate rendering attributes if present
            if "opacity" in attrs:
                from ..validation.types import validate_opacity

                attrs["opacity"] = validate_opacity(attrs["opacity"])

            if "gamma" in attrs:
                from ..validation.types import validate_gamma

                attrs["gamma"] = validate_gamma(attrs["gamma"])

            if "blending_mode" in attrs:
                from ..validation.types import validate_blending_mode

                attrs["blending_mode"] = validate_blending_mode(attrs["blending_mode"])

            # Store attributes
            if self._writer is not None:
                # Write via writer interface and cache
                self._writer.write_group(self.path, **attrs)
                self._attrs_cache.update(attrs)
            else:
                # Metadata-only mode (no writer available)
                self._attrs_cache.update(attrs)

    # --------------------------------------------------------------------- attrs
    @property
    def attrs(self) -> GroupAttrs:
        """Get node attributes.

        Returns:
            Dictionary of node attributes from cache
        """
        return self._attrs_cache

    # --------------------------------------------------------------- hierarchy
    def add_group(self, name: str, **attrs: Any) -> Node:
        """Create and add a child group node.

        Args:
            name: Name of the child group
            **attrs: Additional attributes for the group

        Returns:
            The created child node

        Raises:
            ValueError: If group creation fails
        """
        try:
            aprint(f"Adding child group '{name}' to node '{self.name}'.")

            # Process transform if present to convert to storage format
            if "transform" in attrs:
                from ..core.transforms import prepare_transform_for_zarr

                # Use centralized function for consistent handling
                attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

            if self._writer is not None:
                # Create via writer interface
                child_path = f"{self.path}/{name}" if self.path else name
                self._writer.write_group(child_path, type="group", **attrs)
                child_node = Node(name, parent=self, writer=self._writer, **attrs)
            else:
                # Metadata-only mode (no writer available)
                child_node = Node(name, parent=self, **attrs)

            aprint(f"✓ Child group '{name}' added successfully.")
            return child_node
        except Exception as e:
            aprint(f"Failed to add child group '{name}' to node '{self.name}': {e}")
            raise ValueError(f"Could not create child group '{name}': {e}") from e

    # --------------------------------------------------------------- traversal
    def walk(self, depth: int = 0) -> SceneHierarchy:
        """Walk the node hierarchy depth-first.

        Args:
            depth: Current depth in the hierarchy (used for indentation)

        Yields:
            Tuple of (depth, node) for each node in the hierarchy

        Raises:
            ValueError: If traversal encounters an error
        """
        try:
            aprint(f"Walking node hierarchy from '{self.name}' at depth {depth}.")
            # Cast self to NodeProtocol to satisfy type checker
            from typing import cast

            from ..typing_utils.protocols import NodeProtocol

            yield depth, cast(NodeProtocol, self)
            for child in self.children:
                yield from child.walk(depth + 1)
        except Exception as e:
            aprint(f"Failed to walk node hierarchy from '{self.name}': {e}")
            raise ValueError(
                f"Could not traverse hierarchy from '{self.name}': {e}"
            ) from e

    # --------------------------------------------------------------- properties
    @property
    def transform(self) -> Optional[TransformMatrix]:
        """Get the transformation matrix for this node.

        Returns:
            4x4 transformation matrix if set, None otherwise
        """
        if "transform" in self.attrs:
            from ..core.transforms import read_transform_from_zarr

            transform_list = self.attrs["transform"]
            return read_transform_from_zarr(transform_list)
        return None

    @transform.setter
    def transform(
        self, matrix: Optional[Union[TransformMatrix, np.ndarray, list]]
    ) -> None:
        """Set the transformation matrix for this node.

        Args:
            matrix: 4x4 transformation matrix, flat list of 16 values, or None to remove

        Raises:
            ValueError: If transform is invalid
        """
        if matrix is None:
            # Remove transform if it exists
            if "transform" in self.attrs:
                del self.attrs["transform"]
        else:
            from ..core.transforms import prepare_transform_for_zarr

            # Use centralized function for consistent handling
            self.attrs["transform"] = prepare_transform_for_zarr(matrix)

    @property
    def num_children(self) -> int:
        """Get the number of direct children of this node."""
        return len(self.children)

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Check if this node is the root (has no parent)."""
        return self.parent is None

    # --------------------------------------------------------------- rendering
    @property
    def opacity(self) -> float:
        """Get the opacity value for this node.

        Returns:
            Opacity value (0.0 to 1.0), defaults to 1.0 if not set
        """
        return float(self.attrs.get("opacity", 1.0))

    @opacity.setter
    def opacity(self, value: Any) -> None:
        """Set the opacity value for this node.

        Args:
            value: Opacity value (0.0 to 1.0)

        Raises:
            ValueError: If opacity is not in valid range
            TypeError: If opacity cannot be converted to float
        """
        from ..validation.types import validate_opacity

        self.attrs["opacity"] = validate_opacity(value)

    @property
    def gamma(self) -> float:
        """Get the gamma value for this node.

        Returns:
            Gamma value (0.1 to 10.0), defaults to 1.0 if not set
        """
        return float(self.attrs.get("gamma", 1.0))

    @gamma.setter
    def gamma(self, value: Any) -> None:
        """Set the gamma value for this node.

        Args:
            value: Gamma value (0.1 to 10.0)

        Raises:
            ValueError: If gamma is not in valid range
            TypeError: If gamma cannot be converted to float
        """
        from ..validation.types import validate_gamma

        self.attrs["gamma"] = validate_gamma(value)

    @property
    def blending_mode(self) -> str:
        """Get the blending mode for this node.

        Returns:
            Blending mode string, defaults to "additive" if not set.
            Valid modes: "normal", "additive", "max", "opaque", "luminous"
        """
        return str(self.attrs.get("blending_mode", "additive"))

    @blending_mode.setter
    def blending_mode(self, value: Any) -> None:
        """Set the blending mode for this node.

        Args:
            value: Blending mode string. Valid modes:
                - "normal": Standard alpha blending (semi-transparent)
                - "additive": Classic additive blending, ignores depth (renders on top)
                - "max": Maximum of source and destination (brightest wins)
                - "opaque": Solid rendering with depth write (closest wins)
                - "luminous": Same as additive visually, but respects depth occlusion

        Raises:
            ValueError: If blending mode is not valid
            TypeError: If blending mode is not a string
        """
        from ..validation.types import validate_blending_mode

        self.attrs["blending_mode"] = validate_blending_mode(value)

    def set_opacity(self, value: Any) -> "Node":
        """Set opacity and return self for chaining.

        Args:
            value: Opacity value (0.0 to 1.0)

        Returns:
            Self for method chaining
        """
        self.opacity = value
        return self

    def set_gamma(self, value: Any) -> "Node":
        """Set gamma and return self for chaining.

        Args:
            value: Gamma value (0.1 to 10.0)

        Returns:
            Self for method chaining
        """
        self.gamma = value
        return self

    def set_blending_mode(self, value: Any) -> "Node":
        """Set blending mode and return self for chaining.

        Args:
            value: Blending mode string. Valid modes:
                - "normal": Standard alpha blending (semi-transparent)
                - "additive": Classic additive blending, ignores depth (renders on top)
                - "max": Maximum of source and destination (brightest wins)
                - "opaque": Solid rendering with depth write (closest wins)
                - "luminous": Same as additive visually, but respects depth occlusion

        Returns:
            Self for method chaining
        """
        self.blending_mode = value
        return self

    # --------------------------------------------------------------- repr
    def __repr__(self) -> str:  # pragma: no cover
        """String representation of the node.

        Returns:
            Human-readable string representation of the node
        """
        node_type = self.attrs.get("type", "unknown")
        return f"<{self.__class__.__name__} '{self.name}' ({node_type}) with {len(self.children)} children>"
