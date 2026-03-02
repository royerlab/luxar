"""Scene root node for Luxar hierarchical scene graphs.

This module provides the Scene class, which serves as the root node of the
scene hierarchy. Data-adding methods (add_points, add_lines, add_gsplats)
are inherited from Group.
"""

from __future__ import annotations

import warnings
from os import PathLike
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from arbol import aprint

from ..core.dimensions import Dimensions
from ..core.group import Group
from ..core.viewer_config import ViewerConfig
from ..io.writer import ZarrWriterProtocol


class Scene(Group):
    """Scene root node representing the top level of a scene hierarchy.

    The Scene class is a Group that must be created through LuxarZarrCompiler
    for progressive writing and memory-efficient handling of large datasets.

    Scene dimensions are REQUIRED and serve as the single source of truth for
    the coordinate system. All data nodes must conform to these dimensions.

    Data-adding methods (add_points, add_lines, add_gsplats, etc.) are
    inherited from Group and work identically on Scene.

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
        ...
        ...     # Groups also support add_points, add_lines, add_gsplats:
        ...     group = scene.add_group("my_group")
        ...     group.add_points('nested_pts', more_data)

    Args:
        writer: Writer interface for progressive writing (required)
        dimensions: Scene-level dimension definitions (REQUIRED)
    """

    def __init__(
        self,
        writer: ZarrWriterProtocol,
        dimensions: Dimensions,
        viewer_config: Optional[ViewerConfig] = None,
    ) -> None:
        """Initialize a new Luxar scene.

        Args:
            writer: Writer interface for progressive writing (required)
            dimensions: Scene-level dimension definitions (REQUIRED).
                Defines the coordinate system for all data in the scene.
            viewer_config: Optional viewer configuration hints. Stored in
                the zarr file and used by the viewer as scene-specific defaults.

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

            # Create lightweight root node (sets self._writer)
            super().__init__("Scene", writer=writer)

            # Store dimensions (REQUIRED)
            self._dimensions: Dimensions = dimensions

            # Store dimensions in attributes
            writer.write_group("/", scene_dimensions=dimensions.to_dict())

            # Store viewer config if provided
            self._viewer_config: Optional[ViewerConfig] = viewer_config
            if viewer_config is not None:
                writer.write_group("/", viewer_config=viewer_config.to_dict())

            aprint("✓ Scene initialized successfully with progressive writer")

        except Exception as e:
            aprint(f"Failed to initialize Scene: {e}")
            raise ValueError(f"Could not initialize Scene: {e}") from e

    # ---------------------------------------------------------- hierarchy

    def add_group(self, name: str, **attrs: Any) -> Group:
        """Create and add a child group node to the scene.

        Args:
            name: Name of the group
            **attrs: Additional attributes for the group. Supports:
                opacity: float (0.0-1.0, default 1.0) - Node opacity
                gamma: float (0.1-10.0, default 1.0) - Gamma correction
                blending_mode: str ("normal", "additive", "max", default "additive")

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

    # ---------------------------------------------------------- scene overrides

    def _find_scene(self) -> Scene:
        """Scene is its own root — returns self."""
        return self

    # ---------------------------------------------------------- validation

    def _resolve_extend_to_all(
        self,
        extend_to_all: Optional[Union[List[str], str]],
        positions: np.ndarray,
        data_type: str,
        _stacklevel: int = 3,
    ) -> List[str]:
        """Resolve extend_to_all parameter into a final list of dimension names.

        Handles all extend_to_all modes:
        - None: No extension, but warn if candidates detected
        - "all": Extend to all non-displayed dimensions
        - List of names: Validate and use explicit list
        - []: Explicitly no extension (silences warning)

        Args:
            extend_to_all: User-specified extend_to_all value
            positions: Position/vertex/center array for candidate analysis
            data_type: Human-readable data type for warning messages
                ("points", "lines", "splats")

        Returns:
            List of dimension names to extend visibility across

        Raises:
            ValueError: If extend_to_all contains unknown dimensions or invalid value
        """
        if extend_to_all is None:
            # Default: No extension, but warn if candidates detected
            candidates = self._analyze_extend_candidates(positions)
            if candidates:
                warnings.warn(
                    f"Dimension(s) {candidates} have single values but defined ranges.\n"
                    f"If these {data_type} should be visible at ALL values of these dimensions, use:\n"
                    f"    extend_to_all={candidates}\n"
                    f"If intentional ({data_type} only at these specific values), use:\n"
                    f"    extend_to_all=[]  # Explicit: no extension\n"
                    f"Set extend_to_all explicitly to silence this warning.",
                    UserWarning,
                    stacklevel=_stacklevel,
                )
            return []
        elif extend_to_all == "all":
            # Extend to all non-displayed dimensions
            return [
                dim.name
                for dim in self._dimensions.dimensions
                if not dim.display and dim.name
            ]
        elif isinstance(extend_to_all, list):
            # Use explicit list (including empty list to silence warning)
            unknown_dims = [
                dim_name
                for dim_name in extend_to_all
                if dim_name not in self._dimensions.names
            ]
            if unknown_dims:
                raise ValueError(
                    f"Unknown dimension(s) in extend_to_all: {unknown_dims}. "
                    f"Valid dimensions: {self._dimensions.names}"
                )
            return extend_to_all
        else:
            raise ValueError(
                f"Invalid extend_to_all value: {extend_to_all}. "
                f"Expected None, list of dimension names, 'all', or []."
            )

    def _analyze_extend_candidates(self, positions: np.ndarray) -> List[str]:
        """Analyze which dimensions might be candidates for extend_to_all.

        A dimension is a candidate if:
        1. It is not displayed (non-spatial dimension)
        2. It has only ONE unique value in the data
        3. It has a defined range that is larger than just that single value

        Args:
            positions: Position array to analyze

        Returns:
            List of dimension names that are candidates for extension
        """
        candidates: List[str] = []
        data_ndim = positions.shape[1]

        for i, dim in enumerate(self._dimensions.dimensions):
            if dim.display:
                continue
            if i >= data_ndim:
                continue

            unique_values = np.unique(positions[:, i])
            if len(unique_values) != 1:
                continue

            if dim.range is not None:
                value = unique_values[0]
                range_min, range_max = dim.range

                if range_max > range_min and (
                    value >= range_min and value <= range_max
                ):
                    if dim.name:
                        candidates.append(dim.name)

        return candidates

    def _validate_data_dimensions(
        self,
        positions: np.ndarray,
        node_name: str,
        data_type: str = "positions",
        _stacklevel: int = 3,
    ) -> None:
        """Validate that data dimensions match scene dimensions.

        Performs two levels of validation:
        1. HARD ERROR: Dimensionality mismatch (data columns != scene dimensions)
        2. WARNING: Values outside declared dimension ranges

        Args:
            positions: Position/vertex/center array to validate (shape N x D)
            node_name: Name of the node being added (for error messages)
            data_type: Type of data ("positions", "vertices", "centers")

        Raises:
            ValueError: If dimensionality doesn't match scene dimensions
        """
        data_ndim = positions.shape[1]
        scene_ndim = self._dimensions.ndim

        if data_ndim != scene_ndim:
            dim_names = self._dimensions.names
            raise ValueError(
                f"Dimension mismatch for '{node_name}': {data_type} array has "
                f"{data_ndim} columns, but scene has {scene_ndim} dimensions "
                f"({dim_names}).\n"
                f"Expected {data_type} shape: (N, {scene_ndim})\n"
                f"Got {data_type} shape: {positions.shape}"
            )

        if positions.shape[0] == 0:
            return

        for i, dim in enumerate(self._dimensions.dimensions):
            if dim.range is not None:
                col = positions[:, i]
                min_val, max_val = float(col.min()), float(col.max())
                range_min, range_max = dim.range

                if min_val < range_min or max_val > range_max:
                    warnings.warn(
                        f"'{node_name}' {data_type}: dimension '{dim.name}' has values "
                        f"[{min_val:.4g}, {max_val:.4g}] outside declared range "
                        f"[{range_min}, {range_max}]. "
                        f"Consider adjusting the dimension range or data values.",
                        UserWarning,
                        stacklevel=_stacklevel,
                    )

    # ---------------------------------------------------------- dim_order

    def _apply_dim_order(
        self,
        positions: np.ndarray,
        dim_order: List[str],
        fill: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Reorder and pad position data to match scene dimensions.

        Maps data columns to scene dimensions by name, reordering and
        padding as needed. Returns the transformed array and a list of
        unmapped dimension names (candidates for extend_to_all).

        Args:
            positions: Data array of shape (N, d_data)
            dim_order: Scene dimension names for each data column.
                len(dim_order) must equal positions.shape[1].
            fill: Fixed values for unmapped dimensions (default 0.0)

        Returns:
            Tuple of (transformed_positions, unmapped_dim_names):
            - transformed_positions: shape (N, scene_ndim)
            - unmapped_dim_names: names of dims not covered by dim_order

        Raises:
            ValueError: If dim_order names are invalid or have wrong length
        """
        if fill is None:
            fill = {}

        scene_names = self._dimensions.names
        scene_ndim = self._dimensions.ndim
        data_ndim = positions.shape[1]

        # Validate dim_order length matches data columns
        if len(dim_order) != data_ndim:
            raise ValueError(
                f"dim_order has {len(dim_order)} names but data has "
                f"{data_ndim} columns. They must match."
            )

        # Validate names exist in scene dimensions and are unique
        if len(set(dim_order)) != len(dim_order):
            raise ValueError(f"dim_order has duplicate names: {dim_order}")
        for name in dim_order:
            if name not in scene_names:
                raise ValueError(
                    f"dim_order name '{name}' not found in scene dimensions "
                    f"{scene_names}"
                )

        # Validate fill keys are valid dim names and not in dim_order
        for name in fill:
            if name not in scene_names:
                raise ValueError(
                    f"fill key '{name}' not found in scene dimensions {scene_names}"
                )
            if name in dim_order:
                raise ValueError(
                    f"fill key '{name}' is already in dim_order — cannot "
                    f"both map a data column and fill a fixed value"
                )

        # Build the mapping: for each scene dim, which data column (or fill)
        N = positions.shape[0]
        result = np.zeros((N, scene_ndim), dtype=np.float32)
        unmapped: List[str] = []

        dim_order_set = set(dim_order)
        for scene_idx, scene_name in enumerate(scene_names):
            if scene_name in dim_order_set:
                # Find which data column maps to this scene dim
                data_col = dim_order.index(scene_name)
                result[:, scene_idx] = positions[:, data_col]
            else:
                # Unmapped — fill with fixed value
                result[:, scene_idx] = fill.get(scene_name, 0.0)
                unmapped.append(scene_name)

        return result, unmapped

    # ---------------------------------------------------------- properties

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
        if self._dimensions is None and "scene_dimensions" in self.attrs:
            dims_dict = self.attrs["scene_dimensions"]
            self._dimensions = Dimensions.from_dict(dims_dict)
        if self._dimensions is None:
            raise ValueError(
                "Scene dimensions are not set. This should never happen - "
                "dimensions are required when creating a scene."
            )
        return self._dimensions

    @property
    def viewer_config(self) -> Optional[ViewerConfig]:
        """Get viewer configuration hints.

        Returns:
            ViewerConfig if set, None otherwise.
        """
        if self._viewer_config is None and "viewer_config" in self.attrs:
            vc_dict = self.attrs["viewer_config"]
            self._viewer_config = ViewerConfig.from_dict(vc_dict)
        return self._viewer_config

    @viewer_config.setter
    def viewer_config(self, vc: Optional[ViewerConfig]) -> None:
        """Set viewer configuration hints.

        Args:
            vc: ViewerConfig object, or None to clear.
        """
        self._viewer_config = vc
        if vc is not None:
            vc.validate()
            self.attrs["viewer_config"] = vc.to_dict()
            if self._writer:
                self._writer.write_group("/", viewer_config=vc.to_dict())
        elif "viewer_config" in self.attrs:
            del self.attrs["viewer_config"]

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
        raise NotImplementedError("Scene export not yet implemented")
