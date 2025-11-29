"""Scene root node for Luxar hierarchical scene graphs.

This module provides the Scene class, which serves as the root node of the
scene hierarchy and provides convenient methods for building points scenes.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from arbol import aprint

from ..core.dimensions import Dimensions
from ..core.gsplats import GSplats
from ..core.lines import Lines
from ..core.node import Node
from ..core.points import Points
from ..io.writer import ZarrWriterProtocol
from ..typing_utils.protocols import (
    ColorArray,
    PositionArray,
)
from ..utils.array import (
    broadcast_color_to_points,
    broadcast_radii_to_points,
    broadcast_sharpness_to_points,
)


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
                raise ValueError(
                    "Writer is required. Use LuxarZarrCompiler to create scenes."
                )

            # Store writer interface
            self._writer = writer

            # Create lightweight root node
            super().__init__("Scene", writer=writer)

            # Store dimensions
            self._dimensions: Optional[Dimensions] = dimensions

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
                blending_mode: str ("normal", "additive", default "additive") - Blending mode for rendering

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
        broadcast_dims: Optional[Union[List[str], str]] = None,
        grid_shape: Optional[Tuple[int, ...]] = None,
        **attrs: Any,
    ) -> Points:
        """Add a points node to the scene.

        Important: Position arrays must ALWAYS include ALL scene dimensions, even when
        broadcasting. Broadcasting means "show these points at all values of specified
        dimensions", not "skip these dimensions from the position array".

        Example:
            For a 5D scene (X, Y, Z, Time, Channel), if you want points to appear
            at all times and channels:

            CORRECT:
                positions = [[x, y, z, 0, 0]]  # Include Time=0, Channel=0
                scene.add_points("pts", positions, broadcast_dims=["Time", "Channel"])

            INCORRECT:
                positions = [[x, y, z]]  # Missing Time and Channel dimensions!

        Args:
            name: Name of the points node
            positions: Array of shape (N, D) for point positions where D is dimensionality
            colors: Optional array of shape (N, 3) for point colors, or single RGB color
                as (R, G, B) tuple/list to apply to all points
            radii: Optional array of shape (N,) for point radii, or single radius value
                to apply to all points
            sharpness: Optional array of shape (N,) for point edge sharpness, or single
                sharpness value to apply to all points
            parent: Parent node, defaults to scene root
            broadcast_dims: Controls broadcasting behavior for non-displayed dimensions.
                - None (default): No broadcasting, points only appear at their defined values
                - List of dimension names: Broadcast to all values of specified dimensions
                  e.g., ["Time", "Channel"] makes points appear at all times and channels
                - "auto": Auto-detect broadcast dimensions (use with caution - can be ambiguous)
                - "all": Broadcast to all non-displayed dimensions
            grid_shape: Optional grid resolution for spatial index (if enabled in compiler).
                Tuple of integers specifying number of cells per dimension, e.g., (10, 10, 10, 5)
                for a 4D dataset. If None, grid shape is auto-determined.
            **attrs: Additional attributes for the node. Supports:
                opacity: float (0.0-1.0, default 1.0) - Node opacity
                gamma: float (0.2-2.0, default 1.0) - Gamma correction
                blending_mode: str ("normal", "additive", default "additive") - Blending mode for rendering

        Returns:
            The created Points node

        Raises:
            ValueError: If points creation fails or rendering attributes are invalid
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

            # Dimensions are managed only at scene level - no per-node validation needed

            # Handle broadcast dimensions based on user specification
            final_broadcast_dims = []

            if broadcast_dims is None:
                # Default: No broadcasting
                final_broadcast_dims = []
            elif broadcast_dims == "auto":
                # Auto-detect (use with caution)
                if self._dimensions is not None:
                    final_broadcast_dims = self._auto_detect_broadcast_dims(positions)
                    if final_broadcast_dims:
                        aprint(
                            f"  🔍 Auto-detected broadcast dimensions: {final_broadcast_dims}"
                        )
            elif broadcast_dims == "all":
                # Broadcast all non-displayed dimensions
                if self._dimensions is not None:
                    final_broadcast_dims = [
                        dim.name
                        for dim in self._dimensions.dimensions
                        if not dim.display and dim.name
                    ]
            elif isinstance(broadcast_dims, list):
                # Use explicit list
                final_broadcast_dims = broadcast_dims
            else:
                raise ValueError(f"Invalid broadcast_dims value: {broadcast_dims}")

            # Add broadcast_dims to attributes if we have any
            if final_broadcast_dims:
                attrs["broadcast_dims"] = final_broadcast_dims
                aprint(f"  📡 Broadcasting across dimensions: {final_broadcast_dims}")

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
                grid_shape=grid_shape,
                **attrs,
            )

            # Return lightweight Points node with only metadata
            return Points(
                name,
                metadata=metadata,
                parent=parent_node,
                writer=self._writer,
                **attrs,
            )
        except Exception as e:
            aprint(f"Failed to add points node '{name}': {e}")
            raise ValueError(f"Could not add points '{name}': {e}") from e

    def add_lines(
        self,
        name: str,
        vertices: Union[PositionArray, np.ndarray[Any, Any]],
        widths: Union[
            np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any], float
        ],
        colors: Optional[Union[ColorArray, np.ndarray[Any, Any]]] = None,
        sharpness: Optional[
            Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]
        ] = None,
        indices: Optional[np.ndarray[Any, Any]] = None,
        line_type: str = "polyline",
        parent: Optional[Node] = None,
        **attrs: Any,
    ) -> Lines:
        """Add a lines node to the scene.

        Args:
            name: Name of the lines node
            vertices: Array of shape (N, D) for vertex positions
            widths: Array of shape (N,) for line widths, or single width value
            colors: Optional array of shape (N, 3) for per-vertex colors
            sharpness: Optional array of shape (N,) for edge sharpness
            indices: Optional array of vertex indices for indexed line type
            line_type: Type of line connectivity ("segments", "polyline", "loop", "indexed")
            parent: Parent node, defaults to scene root
            **attrs: Additional attributes for the node

        Returns:
            The created Lines node

        Raises:
            ValueError: If lines creation fails or parameters are invalid
        """
        try:
            # Ensure vertices is array-like
            if not hasattr(vertices, "shape"):
                vertices = np.asarray(vertices)

            if vertices.ndim != 2:
                raise ValueError(
                    f"Vertices must have shape (N, D), got shape {vertices.shape}"
                )

            n_vertices = vertices.shape[0]
            ndim = vertices.shape[1]
            aprint(
                f"Adding lines node '{name}' with {n_vertices:,} vertices in {ndim}D."
            )

            # Process widths (similar to radii for points)
            from ..utils.array import broadcast_scalar_to_points

            processed_widths = broadcast_scalar_to_points(
                widths, n_vertices, name="widths"
            )

            # Process colors and sharpness
            processed_colors = broadcast_color_to_points(colors, n_vertices)
            processed_sharpness = broadcast_sharpness_to_points(sharpness, n_vertices)

            parent_node = parent or self

            # Use writer to write lines immediately
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = self._writer.write_lines(
                path,
                vertices.astype(np.float32),
                widths=processed_widths,
                colors=processed_colors,
                sharpness=processed_sharpness,
                indices=indices,
                line_type=line_type,
                **attrs,
            )

            # Return lightweight Lines node with only metadata
            return Lines(
                name,
                metadata=metadata,
                parent=parent_node,
                writer=self._writer,
                **attrs,
            )
        except Exception as e:
            aprint(f"Failed to add lines node '{name}': {e}")
            raise ValueError(f"Could not add lines '{name}': {e}") from e

    def add_gsplats(
        self,
        name: str,
        centers: Union[PositionArray, np.ndarray[Any, Any]],
        amplitudes: Union[
            np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any], float
        ],
        cholesky_factors: Union[
            np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]
        ],
        colors: Optional[Union[ColorArray, np.ndarray[Any, Any]]] = None,
        sharpness: Optional[
            Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]
        ] = None,
        parent: Optional[Node] = None,
        **attrs: Any,
    ) -> GSplats:
        """Add a Gaussian splats node to the scene.

        Args:
            name: Name of the gsplats node
            centers: Array of shape (N, D) for splat centers
            amplitudes: Array of shape (N,) for intensities, or single value
            cholesky_factors: Array of shape (N, k) for packed Cholesky factors, k=D*(D+1)/2
            colors: Optional array of shape (N, 3) for splat colors
            sharpness: Optional array of shape (N,) for generalized Gaussian exponent
            parent: Parent node, defaults to scene root
            **attrs: Additional attributes for the node

        Returns:
            The created GSplats node

        Raises:
            ValueError: If gsplats creation fails or parameters are invalid
        """
        try:
            # Ensure centers is array-like
            if not hasattr(centers, "shape"):
                centers = np.asarray(centers)

            if centers.ndim != 2:
                raise ValueError(
                    f"Centers must have shape (N, D), got shape {centers.shape}"
                )

            n_splats = centers.shape[0]
            ndim = centers.shape[1]
            aprint(f"Adding gsplats node '{name}' with {n_splats:,} splats in {ndim}D.")

            # Process amplitudes (similar to radii/widths, but can be zero)
            from ..utils.array import broadcast_scalar_to_points

            processed_amplitudes = broadcast_scalar_to_points(
                amplitudes, n_splats, name="amplitudes", require_positive=False
            )

            # Process colors and sharpness
            processed_colors = broadcast_color_to_points(colors, n_splats)
            processed_sharpness = broadcast_sharpness_to_points(sharpness, n_splats)

            parent_node = parent or self

            # Use writer to write gsplats immediately
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = self._writer.write_gsplats(
                path,
                centers.astype(np.float32),
                amplitudes=processed_amplitudes,
                cholesky_factors=cholesky_factors,
                colors=processed_colors,
                sharpness=processed_sharpness,
                **attrs,
            )

            # Return lightweight GSplats node with only metadata
            return GSplats(
                name,
                metadata=metadata,
                parent=parent_node,
                writer=self._writer,
                **attrs,
            )
        except Exception as e:
            aprint(f"Failed to add gsplats node '{name}': {e}")
            raise ValueError(f"Could not add gsplats '{name}': {e}") from e

    def _auto_detect_broadcast_dims(self, positions: np.ndarray) -> List[str]:
        """Auto-detect which dimensions should be broadcast based on data.

        A dimension should be broadcast if:
        1. It has only a single unique value across all points, AND
        2. The total number of points suggests incomplete coverage

        Args:
            positions: Position array to analyze

        Returns:
            List of dimension names that should be auto-broadcasted
        """
        broadcast_dims: list[str] = []

        if self._dimensions is None:
            return broadcast_dims

        data_ndim = positions.shape[1]
        n_points = positions.shape[0]

        # Calculate expected total points for full coverage
        expected_total = 1
        non_displayed_sizes = []

        for i, dim in enumerate(self._dimensions.dimensions):
            if not dim.display and i < data_ndim:
                if dim.range and dim.discrete:
                    # For discrete dimensions, use the range
                    dim_size = int(dim.range[1] - dim.range[0] + 1)
                else:
                    # For continuous dimensions, check unique values
                    dim_size = len(np.unique(positions[:, i]))
                non_displayed_sizes.append((i, dim, dim_size))
                expected_total *= dim_size

        # If we have full coverage or close to it, don't broadcast anything
        # Allow some tolerance for non-grid datasets
        if n_points >= expected_total * 0.8:
            return broadcast_dims

        # Check each non-displayed dimension
        for i, dim in enumerate(self._dimensions.dimensions):
            # Skip displayed dimensions
            if dim.display:
                continue

            # Skip if this dimension is beyond the data's dimensionality
            if i >= data_ndim:
                # Dimension not present in data - should be broadcast
                if dim.name:
                    broadcast_dims.append(dim.name)
                continue

            # Check if this dimension has only one unique value
            unique_values = np.unique(positions[:, i])
            if len(unique_values) == 1:
                # Single value AND incomplete coverage - good candidate for broadcasting
                if dim.name:
                    broadcast_dims.append(dim.name)

        return broadcast_dims

    def get_store_path(self) -> str:
        """Get the path to the backing Zarr store.

        Returns:
            Path to the Zarr store backing this scene
        """
        if self._writer:
            return self._writer.store_path
        raise ValueError("No store path available without writer")

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
