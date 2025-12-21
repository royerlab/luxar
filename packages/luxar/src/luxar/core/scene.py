"""Scene root node for Luxar hierarchical scene graphs.

This module provides the Scene class, which serves as the root node of the
scene hierarchy and provides convenient methods for building points scenes.
"""

from __future__ import annotations

import warnings
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
from ..typing_utils.protocols import ColorArray, PositionArray

# Default radius used when radii are not provided
DEFAULT_POINT_RADIUS = 0.5


class Scene(Node):
    """Scene root node representing the top level of a scene hierarchy.

    The Scene class is a pure scene graph node that must be created through
    LuxarZarrCompiler for progressive writing and memory-efficient handling
    of large datasets.

    Scene dimensions are REQUIRED and serve as the single source of truth for
    the coordinate system. All data nodes must conform to these dimensions.

    Example:
        >>> from luxar import LuxarZarrCompiler, Dimensions, Dimension
        >>> dims = Dimensions([
        ...     Dimension("X", display=True),
        ...     Dimension("Y", display=True),
        ...     Dimension("Z", display=True),
        ... ])
        >>> with LuxarZarrCompiler('output.zarr') as compiler:
        ...     scene = compiler.create_scene(dimensions=dims)
        ...     scene.add_points('points', huge_array)  # Written immediately

    Args:
        writer: Writer interface for progressive writing (required)
        dimensions: Scene-level dimension definitions (REQUIRED)
    """

    def __init__(
        self,
        writer: ZarrWriterProtocol,
        dimensions: Dimensions,
    ) -> None:
        """Initialize a new Luxar scene.

        Args:
            writer: Writer interface for progressive writing (required)
            dimensions: Scene-level dimension definitions (REQUIRED).
                Defines the coordinate system for all data in the scene.

        Raises:
            ValueError: If writer is None, dimensions is None, or initialization fails
        """
        try:
            if writer is None:
                raise ValueError(
                    "Writer is required. Use LuxarZarrCompiler to create scenes."
                )

            if dimensions is None:
                raise ValueError(
                    "dimensions is required. Scene dimensions define the coordinate "
                    "system and are the single source of truth for all data in the scene."
                )

            # Store writer interface
            self._writer = writer

            # Create lightweight root node
            super().__init__("Scene", writer=writer)

            # Store dimensions (REQUIRED)
            self._dimensions: Dimensions = dimensions

            # Store dimensions in attributes
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
                gamma: float (0.1-10.0, default 1.0) - Gamma correction
                blending_mode: str ("normal", "additive", "max", default "additive") - Blending mode for rendering

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
        extend_to_all: Optional[Union[List[str], str]] = None,
        grid_shape: Optional[Tuple[int, ...]] = None,
        **attrs: Any,
    ) -> Points:
        """Add a points node to the scene.

        Important: Position arrays must ALWAYS include ALL scene dimensions, even when
        using extend_to_all. Extension means "show these points at all values of
        specified dimensions", not "skip these dimensions from the position array".

        Example:
            For a 5D scene (X, Y, Z, Time, Channel), if you want points to appear
            at all times and channels:

            CORRECT:
                positions = [[x, y, z, 0, 0]]  # Include Time=0, Channel=0
                scene.add_points("pts", positions, extend_to_all=["Time", "Channel"])

            INCORRECT:
                positions = [[x, y, z]]  # Missing Time and Channel dimensions!

        Args:
            name: Name of the points node
            positions: Array of shape (N, D) for point positions where D is dimensionality
            colors: Optional array of shape (N, 3) for point colors, or single RGB color
                as (R, G, B) tuple/list to apply to all points
            radii: Optional array of shape (N,) for point radii, or single radius value
                to apply to all points. Defaults to 0.5 if not provided.
            sharpness: Optional array of shape (N,) for point edge sharpness, or single
                sharpness value to apply to all points
            parent: Parent node, defaults to scene root
            extend_to_all: Controls visibility across non-displayed dimensions.

              - None (default): Points only visible at their defined dimension values.
                If candidates for extension are detected, a warning will suggest
                setting this parameter explicitly.
              - List of dimension names: Extend visibility to all values of specified
                dimensions, e.g., ["Time", "Channel"] makes points visible at all
                times and channels regardless of the current slice position.
              - "all": Extend to all non-displayed dimensions (points always visible).
              - []: Explicitly no extension (silences the warning).
            grid_shape: Optional tuple specifying the grid shape for structured data
            **attrs: Additional attributes for the node. Supports:
                opacity: float (0.0-1.0, default 1.0) - Node opacity
                gamma: float (0.1-10.0, default 1.0) - Gamma correction
                blending_mode: str ("normal", "additive", "max", default "additive") - Blending mode for rendering

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

            # Handle extend_to_all based on user specification
            final_extend_dims: List[str] = []

            if extend_to_all is None:
                # Default: No extension, but warn if candidates detected
                if self._dimensions is not None:
                    candidates = self._analyze_extend_candidates(positions)
                    if candidates:
                        warnings.warn(
                            f"Dimension(s) {candidates} have single values but defined ranges.\n"
                            f"If these points should be visible at ALL values of these dimensions, use:\n"
                            f"    extend_to_all={candidates}\n"
                            f"If intentional (points only at these specific values), use:\n"
                            f"    extend_to_all=[]  # Explicit: no extension\n"
                            f"Set extend_to_all explicitly to silence this warning.",
                            UserWarning,
                            stacklevel=2,
                        )
                final_extend_dims = []
            elif extend_to_all == "all":
                # Extend to all non-displayed dimensions
                if self._dimensions is not None:
                    final_extend_dims = [
                        dim.name
                        for dim in self._dimensions.dimensions
                        if not dim.display and dim.name
                    ]
            elif isinstance(extend_to_all, list):
                # Use explicit list (including empty list to silence warning)
                final_extend_dims = extend_to_all
            else:
                raise ValueError(
                    f"Invalid extend_to_all value: {extend_to_all}. "
                    f"Expected None, list of dimension names, 'all', or []."
                )

            # Add extend_to_all to attributes if we have any
            if final_extend_dims:
                attrs["extend_to_all"] = final_extend_dims
                aprint(f"  📡 Extending visibility across: {final_extend_dims}")

            # Pass data directly - ArrayEncoder handles scalar/array conversion
            parent_node = parent or self

            # Apply default radius if not provided
            if radii is None:
                radii = DEFAULT_POINT_RADIUS  # Scalar will be broadcast to all points
                aprint(f"  📐 Using default radius: {DEFAULT_POINT_RADIUS}")

            # Use writer to write points immediately
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = self._writer.write_points(
                path,
                positions.astype(np.float32),
                colors=colors,  # Pass directly (scalar, tuple, or array)
                radii=radii,  # Pass directly (scalar or array)
                sharpness=sharpness,  # Pass directly (scalar or array)
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
        extend_to_all: Optional[Union[List[str], str]] = None,
        **attrs: Any,
    ) -> Lines:
        """Add a lines node to the scene.

        Important: Vertex arrays must ALWAYS include ALL scene dimensions, even when
        using extend_to_all. Extension means "show these lines at all values of
        specified dimensions", not "skip these dimensions from the vertex array".

        Example:
            For a 4D scene (X, Y, Z, Time), if you want lines to appear at all times:

            CORRECT:
                vertices = [[x1, y1, z1, 0], [x2, y2, z2, 0]]  # Include Time=0
                scene.add_lines("lines", vertices, widths=0.1, extend_to_all=["Time"])

            INCORRECT:
                vertices = [[x1, y1, z1], [x2, y2, z2]]  # Missing Time dimension!

        Args:
            name: Name of the lines node
            vertices: Array of shape (N, D) for vertex positions
            widths: Array of shape (N,) for line widths, or single width value
            colors: Optional array of shape (N, 3) for per-vertex colors
            sharpness: Optional array of shape (N,) for edge sharpness
            indices: Optional array of vertex indices for indexed line type
            line_type: Type of line connectivity ("segments", "polyline", "loop", "indexed")
            parent: Parent node, defaults to scene root
            extend_to_all: Controls visibility across non-displayed dimensions.

              - None (default): Lines only visible at their defined dimension values.
                If candidates for extension are detected, a warning will suggest
                setting this parameter explicitly.
              - List of dimension names: Extend visibility to all values of specified
                dimensions, e.g., ["Time"] makes lines visible at all times.
              - "all": Extend to all non-displayed dimensions (lines always visible).
              - []: Explicitly no extension (silences the warning).
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

            # Handle extend_to_all based on user specification
            final_extend_dims: List[str] = []

            if extend_to_all is None:
                # Default: No extension, but warn if candidates detected
                if self._dimensions is not None:
                    candidates = self._analyze_extend_candidates(vertices)
                    if candidates:
                        warnings.warn(
                            f"Dimension(s) {candidates} have single values but defined ranges.\n"
                            f"If these lines should be visible at ALL values of these dimensions, use:\n"
                            f"    extend_to_all={candidates}\n"
                            f"If intentional (lines only at these specific values), use:\n"
                            f"    extend_to_all=[]  # Explicit: no extension\n"
                            f"Set extend_to_all explicitly to silence this warning.",
                            UserWarning,
                            stacklevel=2,
                        )
                final_extend_dims = []
            elif extend_to_all == "all":
                # Extend to all non-displayed dimensions
                if self._dimensions is not None:
                    final_extend_dims = [
                        dim.name
                        for dim in self._dimensions.dimensions
                        if not dim.display and dim.name
                    ]
            elif isinstance(extend_to_all, list):
                # Use explicit list (including empty list to silence warning)
                final_extend_dims = extend_to_all
            else:
                raise ValueError(
                    f"Invalid extend_to_all value: {extend_to_all}. "
                    f"Expected None, list of dimension names, 'all', or []."
                )

            # Add extend_to_all to attributes if we have any
            if final_extend_dims:
                attrs["extend_to_all"] = final_extend_dims
                aprint(f"  📡 Extending visibility across: {final_extend_dims}")

            # Pass data directly - ArrayEncoder handles scalar/array conversion
            parent_node = parent or self

            # Use writer to write lines immediately
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = self._writer.write_lines(
                path,
                vertices.astype(np.float32),
                widths=widths,  # Pass directly (scalar or array)
                colors=colors,  # Pass directly (scalar, tuple, or array)
                sharpness=sharpness,  # Pass directly (scalar or array)
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

            # Pass data directly - ArrayEncoder handles scalar/array conversion
            parent_node = parent or self

            # Use writer to write gsplats immediately
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = self._writer.write_gsplats(
                path,
                centers.astype(np.float32),
                amplitudes=amplitudes,  # Pass directly (scalar or array)
                cholesky_factors=cholesky_factors,
                colors=colors,  # Pass directly (scalar, tuple, or array)
                sharpness=sharpness,  # Pass directly (scalar or array)
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

    def add_gsplats_from_data(
        self,
        name: str,
        result: "GSplatData",
        parent: Optional[Node] = None,
        **attrs: Any,
    ) -> GSplats:
        """Add Gaussian splats from a GSplatData object.

        This convenience method bridges the gap between fitting results and scene
        composition, allowing fitted splats to be added directly to a scene without
        manually unpacking arrays.

        Args:
            name: Name of the gsplats node
            result: GSplatData from fit_gaussian_splats()
            parent: Parent node, defaults to scene root
            **attrs: Additional attributes for the node

        Returns:
            The created GSplats node

        Raises:
            TypeError: If result is not a GSplatData instance
            ValueError: If gsplats creation fails

        Example:
            >>> result = fit_gaussian_splats(image, n_iters=1000)
            >>> with LuxarZarrCompiler("scene.zarr") as compiler:
            ...     scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            ...     gsplats = scene.add_gsplats_from_data("fitted", result)
        """
        from luxar.gsplats.fit_result import GSplatData

        if not isinstance(result, GSplatData):
            raise TypeError(
                f"Expected GSplatData, got {type(result).__name__}"
            )

        # Extract arrays from result
        return self.add_gsplats(
            name=name,
            centers=result.centers,
            amplitudes=result.amplitudes,
            cholesky_factors=result.cholesky_factors,
            colors=result.colors,
            sharpness=result.sharpnesses,
            parent=parent,
            **attrs,
        )

    def add_gsplats_from_file(
        self,
        name: str,
        path: Union[str, "Path"],
        parent: Optional[Node] = None,
        **attrs: Any,
    ) -> GSplats:
        """Add Gaussian splats by loading from a .gsplats.zarr file.

        This convenience method allows loading previously saved splat data
        (from fitting or other sources) directly into a scene.

        Args:
            name: Name of the gsplats node
            path: Path to .gsplats.zarr file
            parent: Parent node, defaults to scene root
            **attrs: Additional attributes for the node

        Returns:
            The created GSplats node

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If file format is invalid or gsplats creation fails

        Example:
            >>> # After saving: result.save("fitted.gsplats.zarr")
            >>> with LuxarZarrCompiler("scene.zarr") as compiler:
            ...     scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            ...     gsplats = scene.add_gsplats_from_file("fitted", "fitted.gsplats.zarr")
        """
        from pathlib import Path

        from luxar.gsplats.io.load_gsplats import load_gsplats

        # Load the gsplats file
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GSplats file not found: {path}")

        result = load_gsplats(path)

        # Use add_gsplats_from_data to add to scene
        return self.add_gsplats_from_data(
            name=name,
            result=result,
            parent=parent,
            **attrs,
        )

    def _analyze_extend_candidates(self, positions: np.ndarray) -> List[str]:
        """Analyze which dimensions might be candidates for extend_to_all.

        This method identifies dimensions where the user might want to extend
        visibility. It does NOT auto-apply extension - only suggests candidates
        for the warning message.

        A dimension is a candidate if:
        1. It is not displayed (non-spatial dimension)
        2. It has only ONE unique value in the data
        3. It has a defined range that is larger than just that single value

        This suggests the user may have data at a "placeholder" value and might
        want those points visible across all values of that dimension.

        Args:
            positions: Position array to analyze

        Returns:
            List of dimension names that are candidates for extension
        """
        candidates: List[str] = []

        if self._dimensions is None:
            return candidates

        data_ndim = positions.shape[1]

        for i, dim in enumerate(self._dimensions.dimensions):
            # Skip displayed dimensions (they're always "extended" in the spatial sense)
            if dim.display:
                continue

            # Skip if this dimension is beyond the data's dimensionality
            if i >= data_ndim:
                continue

            # Check if this dimension has only one unique value
            unique_values = np.unique(positions[:, i])
            if len(unique_values) != 1:
                # Multiple values - user clearly has data across this dimension
                continue

            # Single value - check if dimension has a larger range
            if dim.range is not None:
                value = unique_values[0]
                range_min, range_max = dim.range

                # If the range covers more than just this single value,
                # it's a candidate for extension
                if range_max > range_min and (
                    value >= range_min and value <= range_max
                ):
                    if dim.name:
                        candidates.append(dim.name)

        return candidates

    def get_store_path(self) -> str:
        """Get the path to the backing Zarr store.

        Returns:
            Path to the Zarr store backing this scene
        """
        if self._writer:
            return self._writer.store_path
        raise ValueError("No store path available without writer")

    @property
    def dimensions(self) -> Dimensions:
        """Get scene-level dimensions.

        Returns:
            Dimensions object (always present - required for scenes)
        """
        # Try to load from zarr attrs if not cached
        if self._dimensions is None and "scene_dimensions" in self.attrs:
            dims_dict = self.attrs["scene_dimensions"]
            self._dimensions = Dimensions.from_dict(dims_dict)
        if self._dimensions is None:
            raise ValueError(
                "Scene dimensions are not set. This should never happen - "
                "dimensions are required when creating a scene."
            )
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims: Dimensions) -> None:
        """Set scene-level dimensions.

        Args:
            dims: Dimensions object (REQUIRED - cannot be None)

        Raises:
            ValueError: If dims is None
        """
        if dims is None:
            raise ValueError(
                "dimensions cannot be None. Scene dimensions are required and "
                "define the coordinate system for all data in the scene."
            )
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
