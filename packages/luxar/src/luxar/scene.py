from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
from arbol import aprint

from .array_utils import (
    broadcast_color_to_points,
    broadcast_radii_to_points,
    broadcast_sharpness_to_points,
)
from .dimensions import Dimensions
from .node import Node
from .points import Points
from .types import (
    ColorArray,
    DimensionMetadata,
    PositionArray,
    validate_dimension_metadata,
)
from .writer import ZarrWriterProtocol


class Scene(Node):
    """Scene root node representing the top level of a scene hierarchy.

    The Scene class is a pure scene graph node that must be created through
    LuxarZarrCompiler for progressive writing and memory-efficient handling
    of large datasets.

    Example:
        >>> from luxar import LuxarZarrCompiler, Dimensions
        >>> with LuxarZarrCompiler('output.zarr') as compiler:
        ...     scene = compiler.create_scene(dimensions=dims)
        ...     scene.add_points('points', huge_array)  # Written immediately

    Args:
        writer: Writer interface for progressive writing (required)
        dimensions: Scene-level dimension definitions
    """

    def __init__(
        self,
        writer: ZarrWriterProtocol,
        dimensions: Optional[Dimensions] = None,
    ) -> None:
        """Initialize a new Luxar scene.

        Args:
            writer: Writer interface for progressive writing (required)
            dimensions: Scene-level dimension definitions

        Raises:
            ValueError: If scene initialization fails or writer is None
        """
        try:
            if writer is None:
                raise ValueError("Writer is required. Use LuxarZarrCompiler to create scenes.")

            # Store writer interface
            self._writer = writer

            # Create lightweight root node (no Zarr group)
            super().__init__("Scene", group=None, writer=writer)

            # Store dimensions
            self._dimensions: Optional[Dimensions] = dimensions
            self._dimension_metadata: Optional[list[DimensionMetadata]] = None

            # Store dimensions in attributes if provided
            if dimensions is not None:
                writer.write_group("/", scene_dimensions=dimensions.to_dict())

            aprint("✓ Scene initialized successfully with progressive writer")

        except Exception as e:
            aprint(f"Failed to initialize Scene: {e}")
            raise ValueError(f"Could not initialize Scene: {e}") from e

    # ---------------------------------------------------------- builder helpers
    def add_group(self, name: str, **attrs: Any) -> Node:
        """Create and add a child group node to the scene.

        Args:
            name: Name of the group
            **attrs: Additional attributes for the group. Supports:
                opacity: float (0.0-1.0, default 1.0) - Node opacity
                gamma: float (0.2-2.0, default 1.0) - Gamma correction
                blending_mode: str ("normal", "additive", "subtractive", "minimum", "maximum",
                               default "additive") - Blending mode for rendering

        Returns:
            The created group node

        Raises:
            ValueError: If group creation fails or rendering attributes are invalid
        """
        try:
            aprint(f"Adding group node '{name}'.")
            return super().add_group(name, **attrs)
        except Exception as e:
            aprint(f"Failed to add group node '{name}': {e}")
            raise ValueError(f"Could not add group '{name}': {e}") from e

    def add_points(
        self,
        name: str,
        positions: Union[PositionArray, np.ndarray[Any, Any]],
        colors: Optional[Union[ColorArray, np.ndarray[Any, Any]]] = None,
        radii: Optional[
            Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]
        ] = None,
        sharpness: Optional[
            Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]
        ] = None,
        parent: Optional[Node] = None,
        dimension_metadata: Optional[list[DimensionMetadata]] = None,
        **attrs: Any,
    ) -> Points:
        """Add a point cloud node to the scene.

        Args:
            name: Name of the point cloud node
            positions: Array of shape (N, D) for point positions where D is dimensionality
            colors: Optional array of shape (N, 3) for point colors, or single RGB color
                as (R, G, B) tuple/list to apply to all points
            radii: Optional array of shape (N,) for point radii, or single radius value
                to apply to all points
            sharpness: Optional array of shape (N,) for point edge sharpness, or single
                sharpness value to apply to all points
            parent: Parent node, defaults to scene root
            dimension_metadata: Optional list of DimensionMetadata for each dimension
            **attrs: Additional attributes for the node. Supports:
                opacity: float (0.0-1.0, default 1.0) - Node opacity
                gamma: float (0.2-2.0, default 1.0) - Gamma correction
                blending_mode: str ("normal", "additive", "subtractive", "minimum", "maximum",
                               default "additive") - Blending mode for rendering

        Returns:
            The created Points node

        Raises:
            ValueError: If point cloud creation fails or rendering attributes are invalid
        """
        try:
            # Ensure positions is array-like
            if not hasattr(positions, "shape"):
                positions = np.asarray(positions)

            # Check shape
            if positions.ndim != 2:
                raise ValueError(
                    f"Positions must have shape (N, D), got shape {positions.shape}"
                )

            n_points = positions.shape[0]
            ndim = positions.shape[1]
            aprint(f"Adding points node '{name}' with {n_points:,} points in {ndim}D.")

            # Skip dimension validation in new API - allow flexible dimensions
            # self._validate_or_infer_dimensions(positions, dimension_metadata)

            # Skip dimension metadata application - handled at scene level if needed
            # self._apply_dimension_metadata(attrs, dimension_metadata, ndim)

            # Process colors, radii, and sharpness using helper functions
            processed_colors = broadcast_color_to_points(colors, n_points)
            processed_radii = broadcast_radii_to_points(radii, n_points)
            processed_sharpness = broadcast_sharpness_to_points(sharpness, n_points)

            parent_node = parent or self

            # Use writer to write points immediately
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = self._writer.write_points(
                path,
                positions.astype(np.float32),
                colors=processed_colors,
                radii=processed_radii,
                sharpness=processed_sharpness,
                **attrs
            )

            # Return lightweight Points node with only metadata
            return Points(
                name,
                positions=None,  # No data in memory
                metadata=metadata,
                parent=parent_node,
                writer=self._writer,
                **attrs,
            )
        except Exception as e:
            aprint(f"Failed to add points node '{name}': {e}")
            raise ValueError(f"Could not add points '{name}': {e}") from e

    def _validate_or_infer_dimensions(
        self,
        positions: np.ndarray,
        dimension_metadata: Optional[list[DimensionMetadata]],
    ) -> None:
        """Validate positions against scene dimensions or infer them if not set."""
        if self._dimensions is not None:
            self._dimensions.validate_positions(positions, "Points")
        else:
            # Infer dimensions from first point cloud if not set
            if dimension_metadata is None and not hasattr(self, "_inferred_dimensions"):
                aprint("No scene dimensions defined, inferring from first point cloud")
                self._dimensions = Dimensions.from_positions(positions)
                self._inferred_dimensions = True
                self.attrs["scene_dimensions"] = self._dimensions.to_dict()

    def _apply_dimension_metadata(
        self,
        attrs: Dict[str, Any],
        dimension_metadata: Optional[list[DimensionMetadata]],
        ndim: int,
    ) -> None:
        """Apply dimension metadata to node attributes."""
        if dimension_metadata is not None:
            aprint(
                "Warning: dimension_metadata parameter is deprecated, use scene-level dimensions"
            )
            # Validate dimension metadata matches dimensionality
            validated_metadata = validate_dimension_metadata(dimension_metadata, ndim)
            # Set scene-wide dimension metadata if not already set
            if self.dimension_metadata is None:
                self.dimension_metadata = validated_metadata
            # Store in node attributes
            attrs["dimension_metadata"] = [m.to_dict() for m in validated_metadata]
        elif self.dimension_metadata is not None:
            # Use scene's dimension metadata if available
            if len(self.dimension_metadata) == ndim:
                attrs["dimension_metadata"] = [
                    m.to_dict() for m in self.dimension_metadata
                ]
        elif self._dimensions is not None:
            # Convert new Dimensions to legacy format for compatibility
            legacy_metadata = []
            for dim in self._dimensions.dimensions:
                legacy_metadata.append(
                    DimensionMetadata(
                        name=dim.name,
                        unit=dim.unit,
                        scale=dim.scale,
                        range=dim.range,
                    )
                )
            attrs["dimension_metadata"] = [m.to_dict() for m in legacy_metadata]

    def finalize(self) -> None:
        """Finalize the scene.
        
        Note: Finalization is now handled automatically by LuxarZarrCompiler's
        context manager. This method is kept for compatibility but does nothing.
        """
        aprint("Note: Finalization is handled by LuxarZarrCompiler context manager")

    def get_store_path(self) -> str:
        """Get the path to the backing Zarr store.

        Returns:
            Path to the Zarr store backing this scene
        """
        if self._writer:
            return self._writer.store_path
        raise ValueError("No store path available without writer")

    @property
    def dimension_metadata(self) -> Optional[list[DimensionMetadata]]:
        """Get dimension metadata for the scene.

        Returns:
            List of DimensionMetadata objects if set, None otherwise
        """
        # Try to load from zarr attrs if not cached
        if self._dimension_metadata is None and "dimension_metadata" in self.attrs:
            metadata_dicts = self.attrs["dimension_metadata"]
            if metadata_dicts:
                ndim = len(metadata_dicts)
                self._dimension_metadata = validate_dimension_metadata(
                    metadata_dicts, ndim
                )
        return self._dimension_metadata

    @dimension_metadata.setter
    def dimension_metadata(self, metadata: Optional[list[DimensionMetadata]]) -> None:
        """Set dimension metadata for the scene.

        Args:
            metadata: List of DimensionMetadata objects, one per dimension

        Raises:
            ValueError: If metadata is invalid
        """
        if metadata is None:
            self._dimension_metadata = None
            if "dimension_metadata" in self.attrs:
                del self.attrs["dimension_metadata"]
        else:
            # Store in attrs for serialization
            self.attrs["dimension_metadata"] = [m.to_dict() for m in metadata]
            self._dimension_metadata = metadata

    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Get scene-level dimensions.

        Returns:
            Dimensions object if set, None otherwise
        """
        # Try to load from zarr attrs if not cached
        if self._dimensions is None and "scene_dimensions" in self.attrs:
            dims_dict = self.attrs["scene_dimensions"]
            self._dimensions = Dimensions.from_dict(dims_dict)
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims: Optional[Dimensions]) -> None:
        """Set scene-level dimensions.

        Args:
            dims: Dimensions object or None to clear
        """
        if dims is None:
            self._dimensions = None
            if "scene_dimensions" in self.attrs:
                del self.attrs["scene_dimensions"]
        else:
            self.attrs["scene_dimensions"] = dims.to_dict()
            self._dimensions = dims

    def to_zarr(self, path: PathLike) -> None:
        """Export scene to a new Zarr store location.

        Args:
            path: Destination path for the Zarr store

        Raises:
            NotImplementedError: Scene export is not yet implemented
        """
        aprint(f"Exporting scene to {path}")
        # This would require copying the entire Zarr store
        # Implementation depends on zarr library capabilities
        raise NotImplementedError("Scene export not yet implemented")
