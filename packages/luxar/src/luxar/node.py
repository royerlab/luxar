"""
luxar.node – Defines the Node class for Luxar scene graph nodes.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np
import zarr
from arbol import aprint

from .types import GroupAttrs, SceneHierarchy, ZarrGroupProtocol, TransformMatrix, validate_transform


class Node:
    """
    A node in the Luxar scene graph (mirrors a Zarr group).

    This class represents a single node in the hierarchical scene graph structure.
    Each node corresponds to a Zarr group and can contain child nodes, forming
    a tree structure that mirrors the Zarr group hierarchy.

    Args:
        name: Name of the node
        group: Backing Zarr group
        parent: Parent node in the hierarchy
        **attrs: Additional attributes for the node
    """

    def __init__(
        self,
        name: str,
        group: Union[zarr.Group, ZarrGroupProtocol],
        parent: Optional[Node] = None,
        **attrs: Any,
    ) -> None:
        """
        Initialize a scene graph node.

        Args:
            name: Name of the node
            group: Zarr group backing this node
            parent: Parent node in the scene hierarchy
            **attrs: Additional attributes to set on the node
        """
        self.name: str = name
        self._group: Union[zarr.Group, ZarrGroupProtocol] = group
        self.parent: Optional[Node] = parent
        self.children: List[Node] = []

        # Add this node to parent's children list
        if parent is not None:
            parent.children.append(self)

        # Initialize or merge attributes
        if attrs:
            # Validate transform if present
            if 'transform' in attrs:
                try:
                    # Convert to numpy array and validate
                    transform_array = np.array(attrs['transform'], dtype=np.float32)
                    if transform_array.size == 16:
                        transform_matrix = transform_array.reshape(4, 4)
                        validated = validate_transform(transform_matrix)
                        attrs['transform'] = validated.ravel().tolist()
                    else:
                        raise ValueError(f"Transform must have 16 elements, got {transform_array.size}")
                except Exception as e:
                    aprint(f"Invalid transform for node '{name}': {e}")
                    raise ValueError(f"Invalid transform: {e}") from e
            
            self._group.attrs.update(attrs)

    # --------------------------------------------------------------------- attrs
    @property
    def attrs(self) -> GroupAttrs:
        """
        Live view of the Zarr group's attributes.

        Returns:
            Dictionary of Zarr group attributes that can be modified in place
        """
        return self._group.attrs

    # --------------------------------------------------------------- hierarchy
    def add_group(self, name: str, **attrs: Any) -> Node:
        """
        Create and add a child group node.

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
            grp = self._group.require_group(name)
            child_node = Node(name, grp, parent=self, **attrs)
            aprint(f"✓ Child group '{name}' added successfully.")
            return child_node
        except Exception as e:
            aprint(f"Failed to add child group '{name}' to node '{self.name}': {e}")
            raise ValueError(f"Could not create child group '{name}': {e}") from e

    # --------------------------------------------------------------- traversal
    def walk(self, depth: int = 0) -> SceneHierarchy:
        """
        Walk the node hierarchy depth-first.

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
        """
        Get the transformation matrix for this node.
        
        Returns:
            4x4 transformation matrix if set, None otherwise
        """
        if 'transform' in self.attrs:
            transform_list = self.attrs['transform']
            return np.array(transform_list, dtype=np.float32).reshape(4, 4)
        return None
    
    @transform.setter
    def transform(self, matrix: Optional[Union[TransformMatrix, np.ndarray, list]]) -> None:
        """
        Set the transformation matrix for this node.
        
        Args:
            matrix: 4x4 transformation matrix, flat list of 16 values, or None to remove
            
        Raises:
            ValueError: If transform is invalid
        """
        if matrix is None:
            # Remove transform if it exists
            if 'transform' in self.attrs:
                del self.attrs['transform']
        else:
            # Convert and validate
            if isinstance(matrix, list):
                matrix = np.array(matrix, dtype=np.float32)
            
            if matrix.size == 16:
                matrix = matrix.reshape(4, 4)
            
            validated = validate_transform(matrix)
            self.attrs['transform'] = validated.ravel().tolist()
    
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

    # --------------------------------------------------------------- repr
    def __repr__(self) -> str:  # pragma: no cover
        """
        String representation of the node.

        Returns:
            Human-readable string representation of the node
        """
        node_type = self.attrs.get("type", "unknown")
        return f"<{self.__class__.__name__} '{self.name}' ({node_type}) with {len(self.children)} children>"
