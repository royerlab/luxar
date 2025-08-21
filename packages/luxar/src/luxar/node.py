"""luxar.node – Defines the Node class for Luxar scene graph nodes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import zarr
from arbol import aprint

from .types import (
    GroupAttrs,
    SceneHierarchy,
    TransformMatrix,
    ZarrGroupProtocol,
    validate_transform,
)


class Node:
    """A node in the Luxar scene graph.

    This class represents a single node in the hierarchical scene graph structure.
    In progressive mode with a writer, nodes are lightweight metadata containers.
    In legacy mode, they mirror Zarr groups directly.

    Args:
        name: Name of the node
        group: Backing Zarr group (optional in progressive mode)
        parent: Parent node in the hierarchy
        writer: Optional writer interface for progressive writing
        **attrs: Additional attributes for the node
    """

    def __init__(
        self,
        name: str,
        group: Optional[Union[zarr.Group, ZarrGroupProtocol]] = None,
        parent: Optional[Node] = None,
        writer: Optional[Any] = None,  # ZarrWriterProtocol
        **attrs: Any,
    ) -> None:
        """Initialize a scene graph node.

        Args:
            name: Name of the node
            group: Zarr group backing this node (optional in progressive mode)
            parent: Parent node in the scene hierarchy
            writer: Optional writer for progressive mode
            **attrs: Additional attributes to set on the node
        """
        self.name: str = name
        self._group: Optional[Union[zarr.Group, ZarrGroupProtocol]] = group
        self._writer = writer
        self.parent: Optional[Node] = parent
        self.children: List[Node] = []
        self._metadata: Dict[str, Any] = {}  # For progressive mode
        self._attrs_cache: Dict[str, Any] = {}  # Attributes cache for progressive mode

        # Determine path in hierarchy
        if parent is not None:
            parent.children.append(self)
            self.path = f"{parent.path}/{name}" if hasattr(parent, 'path') and parent.path else name
        else:
            self.path = ""

        # Initialize or merge attributes
        if attrs:
            # Validate transform if present
            if "transform" in attrs:
                try:
                    transform_value = attrs["transform"]
                    # Check if it's already a list (already transposed for THREE.js)
                    if isinstance(transform_value, list) and len(transform_value) == 16:
                        # Already in the correct format, just validate it
                        # Convert to matrix, validate, and store back as list
                        matrix = np.array(transform_value, dtype=np.float32).reshape(4, 4).T
                        validated = validate_transform(matrix)
                        # Store back as list in THREE.js format (transpose back)
                        attrs["transform"] = validated.T.ravel().tolist()
                    else:
                        # It's a numpy array or other format, needs conversion
                        transform_array = np.array(transform_value, dtype=np.float32)
                        if transform_array.size == 16:
                            transform_matrix = transform_array.reshape(4, 4)
                            validated = validate_transform(transform_matrix)
                            # Transpose for THREE.js (column-major order) before flattening
                            attrs["transform"] = validated.T.ravel().tolist()
                        else:
                            raise ValueError(
                                f"Transform must have 16 elements, got {transform_array.size}"
                            )
                except Exception as e:
                    aprint(f"Invalid transform for node '{name}': {e}")
                    raise ValueError(f"Invalid transform: {e}") from e

            # Validate rendering attributes if present
            if "opacity" in attrs:
                from .types import validate_opacity

                attrs["opacity"] = validate_opacity(attrs["opacity"])

            if "gamma" in attrs:
                from .types import validate_gamma

                attrs["gamma"] = validate_gamma(attrs["gamma"])

            if "blending_mode" in attrs:
                from .types import validate_blending_mode

                attrs["blending_mode"] = validate_blending_mode(attrs["blending_mode"])

            # Store attributes
            if self._group is not None:
                # Legacy mode: write to Zarr
                self._group.attrs.update(attrs)
            elif self._writer is not None:
                # Progressive mode: write via writer and cache
                self._writer.write_group(self.path, **attrs)
                self._attrs_cache.update(attrs)
            else:
                # Metadata-only mode
                self._attrs_cache.update(attrs)

    # --------------------------------------------------------------------- attrs
    @property
    def attrs(self) -> GroupAttrs:
        """Get node attributes.

        Returns:
            Dictionary of node attributes (from Zarr or cache)
        """
        if self._group is not None:
            # Legacy mode: return Zarr attrs
            return self._group.attrs
        else:
            # Progressive mode: return cached attrs
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

            # Process transform if present to convert numpy array to list
            if "transform" in attrs:
                transform_value = attrs["transform"]
                if not isinstance(transform_value, list):
                    # Convert numpy array to list for JSON serialization
                    transform_array = np.array(transform_value, dtype=np.float32)
                    if transform_array.size == 16:
                        transform_matrix = transform_array.reshape(4, 4)
                        # Transpose for THREE.js (column-major order) before flattening
                        attrs["transform"] = transform_matrix.T.ravel().tolist()
                    else:
                        raise ValueError(f"Transform must have 16 elements, got {transform_array.size}")

            if self._writer is not None:
                # Progressive mode: create via writer
                child_path = f"{self.path}/{name}" if self.path else name
                self._writer.write_group(child_path, type="group", **attrs)
                child_node = Node(name, group=None, parent=self, writer=self._writer, **attrs)
            elif self._group is not None:
                # Legacy mode: create Zarr group
                grp = self._group.require_group(name)
                child_node = Node(name, grp, parent=self, **attrs)
            else:
                # Metadata-only mode
                child_node = Node(name, group=None, parent=self, **attrs)

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

            from .types import NodeProtocol

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
            transform_list = self.attrs["transform"]
            # Transpose back from THREE.js format (column-major) to numpy format (row-major)
            return np.array(transform_list, dtype=np.float32).reshape(4, 4).T
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
            # Convert and validate
            if isinstance(matrix, list):
                matrix = np.array(matrix, dtype=np.float32)

            if matrix.size == 16:
                matrix = matrix.reshape(4, 4)

            validated = validate_transform(matrix)
            # Transpose for THREE.js (column-major order) before flattening
            self.attrs["transform"] = validated.T.ravel().tolist()

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
        from .types import validate_opacity

        self.attrs["opacity"] = validate_opacity(value)

    @property
    def gamma(self) -> float:
        """Get the gamma value for this node.

        Returns:
            Gamma value (0.2 to 2.0), defaults to 1.0 if not set
        """
        return float(self.attrs.get("gamma", 1.0))

    @gamma.setter
    def gamma(self, value: Any) -> None:
        """Set the gamma value for this node.

        Args:
            value: Gamma value (0.2 to 2.0)

        Raises:
            ValueError: If gamma is not in valid range
            TypeError: If gamma cannot be converted to float
        """
        from .types import validate_gamma

        self.attrs["gamma"] = validate_gamma(value)

    @property
    def blending_mode(self) -> str:
        """Get the blending mode for this node.

        Returns:
            Blending mode string, defaults to "additive" if not set
        """
        return str(self.attrs.get("blending_mode", "additive"))

    @blending_mode.setter
    def blending_mode(self, value: Any) -> None:
        """Set the blending mode for this node.

        Args:
            value: Blending mode ("normal", "additive", "subtractive", "minimum", "maximum")

        Raises:
            ValueError: If blending mode is not valid
            TypeError: If blending mode is not a string
        """
        from .types import validate_blending_mode

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
            value: Gamma value (0.2 to 2.0)

        Returns:
            Self for method chaining
        """
        self.gamma = value
        return self

    def set_blending_mode(self, value: Any) -> "Node":
        """Set blending mode and return self for chaining.

        Args:
            value: Blending mode string

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
