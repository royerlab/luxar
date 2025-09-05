"""luxar.compiler – Progressive Zarr compiler for memory-efficient scene building.

This module provides the LuxarZarrCompiler class which implements progressive
writing to Zarr stores, enabling handling of arbitrarily large datasets without
memory constraints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from ..core.dimensions import Dimensions
from ..io.reader import DEFAULT_COMP
from ..io.writer import ZarrWriterProtocol
from ..typing_utils.aliases import ChunkSpec, MaxShape, NodePath, PointsMetadata
from ..typing_utils.config import DEFAULT_CHUNK_SIZE, DEFAULT_VERSION
from ..typing_utils.constants import SHARPNESS_MAX
from ..typing_utils.datatypes import (
    DEFAULT_CONFIG,
    DataTypeConfig,
    convert_array_dtype,
)
from ..typing_utils.protocols import (
    CompressorProtocol,
    PhysicalUnit,
    validate_physical_unit,
)

# Validation functions imported locally to avoid circular imports
from .point_spatial_index import (
    apply_sort_order,
    build_spatial_index,
    validate_spatial_index,
)


def _calculate_intelligent_chunks(
    shape: Tuple[int, ...],
    target_chunk_size: int = DEFAULT_CHUNK_SIZE,
    spatial_index_data: Optional[Dict[str, Any]] = None,
) -> Tuple[int, ...]:
    """Calculate optimal chunk shape for a dataset.

    When spatial index data is available, aligns chunks with spatial cells
    for better query performance during lazy loading.

    Args:
        shape: Shape of the dataset
        target_chunk_size: Target size for chunks in elements
        spatial_index_data: Optional spatial index data for optimization

    Returns:
        Optimized chunk shape
    """
    if len(shape) == 1:
        # 1D array - simple chunking
        return (min(shape[0], target_chunk_size),)

    if len(shape) == 2:
        # 2D array (e.g., positions) - chunk along first dimension
        n_points, n_dims = shape

        # If spatial index is available, align chunks with spatial cells
        if spatial_index_data and "grid_shape" in spatial_index_data:
            grid_shape = spatial_index_data["grid_shape"]
            total_cells = int(np.prod(grid_shape))

            if total_cells > 0:
                # Calculate average points per cell
                avg_points_per_cell = max(1, n_points // total_cells)

                # Try to make chunks contain 4-8 cells worth of points for balance
                # between I/O efficiency and spatial locality
                cells_per_chunk = max(1, min(8, total_cells // 10))
                chunk_points = avg_points_per_cell * cells_per_chunk

                # Clamp to reasonable bounds
                chunk_points = max(1024, min(target_chunk_size, chunk_points))
                return (chunk_points, n_dims)

        # Fallback to standard chunking
        chunk_points = min(n_points, target_chunk_size // n_dims)
        return (chunk_points, n_dims)

    # For higher dimensions, use reasonable defaults
    return tuple(min(s, target_chunk_size) for s in shape)


class LuxarZarrCompiler(ZarrWriterProtocol):
    """Progressive Zarr compiler with context manager support.

    This compiler writes data immediately to Zarr without keeping it in memory,
    enabling processing of datasets larger than available RAM.

    Args:
        store_path: Path where the Zarr store will be created
        compressor: Compression configuration for datasets
        units: Physical units for the scene (deprecated, use dimensions)
        version: Luxar format version

    Examples:
        Basic usage with context manager:
        >>> with LuxarZarrCompiler('output.zarr') as compiler:
        ...     scene = compiler.create_scene()
        ...     positions = np.random.randn(10000, 3).astype(np.float32)
        ...     scene.add_points('points', positions)

        With HDR colors and dimensions:
        >>> dims = Dimensions([
        ...     Dimension('x', unit='um'),
        ...     Dimension('y', unit='um'),
        ...     Dimension('z', unit='um')
        ... ])
        >>> with LuxarZarrCompiler('scene.zarr') as compiler:
        ...     scene = compiler.create_scene(dimensions=dims)
        ...     # HDR colors with values > 1.0
        ...     colors = np.random.rand(1000, 3).astype(np.float32) * 5.0
        ...     scene.add_points('bright_points', positions, colors=colors)

        Progressive writing for huge datasets:
        >>> with LuxarZarrCompiler('huge.zarr') as compiler:
        ...     scene = compiler.create_scene()
        ...     # Process data in chunks to avoid memory issues
        ...     for i in range(100):
        ...         chunk = load_chunk(i)  # Load chunk from disk
        ...         scene.add_points(f'chunk_{i}', chunk)
        ...         # Data written immediately, memory freed
    """

    def __init__(
        self,
        store_path: Optional[Union[str, Path]] = None,
        compressor: Optional[CompressorProtocol] = DEFAULT_COMP,
        units: Union[PhysicalUnit, str] = "metre",
        version: str = DEFAULT_VERSION,
        enable_spatial_index: bool = True,
        dtype_config: Optional[DataTypeConfig] = None,
    ) -> None:
        """Initialize the Zarr compiler.

        Args:
            store_path: Path for the Zarr store, or None for temporary
            compressor: Compressor for datasets
            units: Physical units for the scene
            version: Luxar format version
            enable_spatial_index: Whether to build spatial indices for points (default: True)
            dtype_config: Configuration for data types (default: auto-detection)
        """
        # Handle store path
        if store_path is None:
            self._tmpdir = tempfile.TemporaryDirectory()
            self._store_path = Path(self._tmpdir.name) / "scene.zarr"
            aprint(f"📁 Using temporary directory: {self._store_path}")
        else:
            self._tmpdir = None
            self._store_path = Path(store_path)
            aprint(f"📁 Creating scene at: {self._store_path}")

        # Validate units
        if isinstance(units, str):
            units = validate_physical_unit(units)

        # Store spatial index flag
        self.enable_spatial_index = enable_spatial_index

        # Store dtype configuration
        self.dtype_config = dtype_config or DEFAULT_CONFIG

        # Create root Zarr group
        self.store = zarr.open_group(self._store_path, mode="w")
        self.store.attrs.update(
            {
                "luxar_version": version,
                "units": units,
                "type": "scene",
            }
        )

        self.compressor = compressor
        self._metadata_cache: Dict[str, Any] = {}
        self._is_finalized = False

        aprint(f"✅ Zarr compiler initialized at {self._store_path}")

    def __enter__(self) -> LuxarZarrCompiler:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and finalize."""
        if not self._is_finalized:
            self.finalize()

        # Clean up temporary directory if used
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def create_scene(self, dimensions: Optional[Dimensions] = None) -> Any:
        """Create a scene with this compiler as writer.

        Args:
            dimensions: Optional dimension specification for the scene

        Returns:
            Scene object configured with this compiler as writer
        """
        # Import here to avoid circular dependency
        from ..core.scene import Scene

        # Store dimensions in root attributes if provided
        if dimensions is not None:
            self.store.attrs["scene_dimensions"] = dimensions.to_dict()

        # Create scene with writer injection
        scene = Scene(writer=self, dimensions=dimensions)
        aprint("✅ Scene created with progressive writer")

        return scene

    def write_group(self, path: NodePath, **attrs: Any) -> None:
        """Create a group in the Zarr store.

        Args:
            path: Path for the group within the store
            **attrs: Attributes to attach to the group
        """
        # Handle root path
        if path == "/" or path == "":
            group = self.store
        else:
            # Remove leading slash if present
            path = path.lstrip("/")
            group = self.store.require_group(path)

        # Update attributes - preserve existing ones
        if attrs:
            # Get existing attributes
            existing_attrs = dict(group.attrs)
            # Update with new attributes
            existing_attrs.update(attrs)
            # Set all attributes back
            group.attrs.update(existing_attrs)

        aprint(f"📝 Created group: {path or '/'}")

    def write_points(
        self,
        path: NodePath,
        positions: NDArray[np.float32],
        colors: Optional[NDArray[np.float32]] = None,
        radii: Optional[NDArray[np.float32]] = None,
        sharpness: Optional[NDArray[np.float32]] = None,
        grid_shape: Optional[Tuple[int, ...]] = None,
        **attrs: Any,
    ) -> PointsMetadata:
        """Write points data progressively to Zarr.

        Data is written immediately to disk without being kept in memory.
        If spatial indexing is enabled, points are reordered for spatial locality.

        Args:
            path: Path for the points within the store
            positions: Point positions of shape (N, D)
            colors: Optional HDR colors of shape (N, 3)
            radii: Optional radii of shape (N,)
            sharpness: Optional sharpness of shape (N,)
            grid_shape: Optional grid resolution for spatial index (auto if None)
            **attrs: Additional attributes

        Returns:
            Metadata dictionary about the written data
        """
        # Import validation functions locally to avoid circular imports
        from ..validation.base import (
            validate_colors_for_writing,
            validate_positions_for_writing,
            validate_radii_for_writing,
            validate_sharpness_for_writing,
        )

        # Remove leading slash and create group
        path = path.lstrip("/")
        group = self.store.require_group(path)

        # Validate positions and get dimensions
        n_points, n_dims = validate_positions_for_writing(positions)

        aprint(f"📝 Writing {n_points:,} points ({n_dims}D) to {path}")

        # Build spatial index if enabled
        spatial_index_data = None
        spatial_extend_dims = None  # Initialize here so it's always defined
        if self.enable_spatial_index and n_points > 0:
            # Extract displayed dimensions and spatial flags from scene metadata
            displayed_dims = None
            if "scene_dimensions" in self.store.attrs:
                scene_dims = self.store.attrs["scene_dimensions"]
                dims_list = (
                    scene_dims.get("dimensions", [])
                    if isinstance(scene_dims, dict)
                    else []
                )

                # Find which dimensions are displayed
                displayed_dims = []
                for i, dim in enumerate(dims_list):
                    if dim.get("display", True):  # Default to True if not specified
                        displayed_dims.append(i)

                # Limit to max 3 displayed dimensions
                displayed_dims = displayed_dims[:3]

                # Get spatial flags from each dimension
                spatial_extend_dims = [
                    dim.get("spatial", True if dim.get("display", True) else False)
                    for dim in dims_list
                ]

            # If no dimension metadata, default to first 3 dimensions as displayed
            if displayed_dims is None:
                displayed_dims = list(range(min(3, n_dims)))

            # Calculate non-displayed dimensions
            non_displayed_dims = [d for d in range(n_dims) if d not in displayed_dims]

            # Only build spatial index if there are non-displayed dimensions
            spatial_index_data = None
            if len(non_displayed_dims) == 0:
                aprint(
                    f"  ⚠️ All {n_dims} dimensions are displayed - skipping spatial index"
                )
            else:
                aprint("  🔍 Building spatial index...")
                aprint(f"    Displayed dimensions: {displayed_dims}")
                aprint(f"    Non-displayed dimensions to index: {non_displayed_dims}")

                # Auto-determine grid shape based on data characteristics
                grid_shape_array = None
                if grid_shape is not None:
                    # Use provided grid shape, but only for non-displayed dimensions
                    if len(grid_shape) == len(non_displayed_dims):
                        grid_shape_array = np.array(grid_shape, dtype=np.uint32)
                    else:
                        aprint(
                            f"    ⚠️ Provided grid_shape has {len(grid_shape)} dims but {len(non_displayed_dims)} non-displayed dims"
                        )
                        grid_shape_array = None

                if grid_shape_array is None and "scene_dimensions" in self.store.attrs:
                    scene_dims = self.store.attrs["scene_dimensions"]
                    grid_shape_list = []

                    # Calculate position ranges for non-displayed dimensions only
                    indexed_positions = positions[:, non_displayed_dims]
                    min_coords = np.min(indexed_positions, axis=0)
                    max_coords = np.max(indexed_positions, axis=0)

                    # Get dimension list if available
                    dims_list = (
                        scene_dims.get("dimensions", [])
                        if isinstance(scene_dims, dict)
                        else []
                    )

                    for idx, d in enumerate(non_displayed_dims):
                        # Check if this dimension is discrete
                        is_discrete = False
                        if d < len(dims_list) and dims_list[d].get("discrete", False):
                            is_discrete = True

                        if is_discrete:
                            # For discrete dimensions, use one cell per unique value
                            unique_vals = len(np.unique(indexed_positions[:, idx]))
                            # One cell per discrete value for precise indexing
                            # Only apply a very high cap to prevent memory issues
                            cells = min(
                                10000, unique_vals
                            )  # Very high cap only for safety
                            grid_shape_list.append(cells)
                            aprint(
                                f"    Non-displayed dim {d}: discrete with {unique_vals} unique values → {cells} cells"
                            )
                        else:
                            # For continuous non-displayed dimensions, use fewer cells
                            dim_range = max_coords[idx] - min_coords[idx]
                            if dim_range > 0:
                                # Use fewer cells for non-displayed dimensions (5-10 typically)
                                cells = min(10, max(3, int(np.sqrt(n_points / 10000))))
                            else:
                                cells = 2
                            grid_shape_list.append(cells)
                            aprint(
                                f"    Non-displayed dim {d}: continuous → {cells} cells"
                            )

                    grid_shape_array = (
                        np.array(grid_shape_list, dtype=np.uint32)
                        if grid_shape_list
                        else None
                    )
                    if grid_shape_array is not None:
                        aprint(
                            f"  📊 Grid shape for non-displayed dims: {grid_shape_array}"
                        )

                # Determine which dimensions are discrete
                discrete_dims = []
                if "scene_dimensions" in self.store.attrs:
                    scene_dims = self.store.attrs["scene_dimensions"]
                    dims_list = (
                        scene_dims.get("dimensions", [])
                        if isinstance(scene_dims, dict)
                        else []
                    )
                    for d in range(n_dims):
                        if d < len(dims_list) and dims_list[d].get("discrete", False):
                            discrete_dims.append(d)

                # Build the spatial index with displayed and discrete dimensions info
                # Get max_radius from attrs or default
                max_radius = attrs.get("max_radius", 0.1)
                if radii is not None:
                    # If radii provided, use the max as max_radius
                    max_radius = float(np.max(radii))

                spatial_index_data = build_spatial_index(
                    positions,
                    grid_shape_array,
                    displayed_dims=displayed_dims,
                    discrete_dims=discrete_dims,
                    spatial_extend_dims=spatial_extend_dims,
                    max_radius=max_radius,
                )

            # Only process spatial index if it was created
            if spatial_index_data is not None:
                # Validate the spatial index before using it
                validate_spatial_index(spatial_index_data, n_points, n_dims)

                # Reorder all arrays according to spatial index
                positions = spatial_index_data["sorted_positions"]
                colors = apply_sort_order(colors, spatial_index_data["sort_order"])
                radii = apply_sort_order(radii, spatial_index_data["sort_order"])
                sharpness = apply_sort_order(
                    sharpness, spatial_index_data["sort_order"]
                )

                aprint(
                    f"  ✓ Spatial index built: {len(spatial_index_data['occupied_cells'])} occupied cells"
                )

        # Determine and apply optimal dtype for positions
        position_dtype = self.dtype_config.get_position_dtype(positions)
        positions_converted = convert_array_dtype(positions, position_dtype)

        # Write positions with intelligent chunking (aligned with spatial index if available)
        chunks = _calculate_intelligent_chunks(
            positions_converted.shape,
            target_chunk_size=DEFAULT_CHUNK_SIZE,
            spatial_index_data=spatial_index_data,
        )
        group.create_dataset(
            "positions",
            data=positions_converted,
            chunks=chunks,
            compressor=self.compressor,
            dtype=position_dtype,
            overwrite=True,
        )

        # Store dtype metadata for client-side handling
        group.attrs["position_dtype"] = np.dtype(position_dtype).name

        # Write optional arrays
        metadata = {
            "n_points": n_points,
            "dims": n_dims,
            "path": path,
            "has_colors": False,
            "has_radii": False,
            "has_sharpness": False,
        }

        if colors is not None:
            # Validate colors
            validate_colors_for_writing(colors, n_points)

            # Determine and apply optimal dtype for colors
            color_dtype = self.dtype_config.get_color_dtype(colors)
            colors_converted = convert_array_dtype(
                colors, color_dtype, normalize=(color_dtype == np.uint8)
            )

            color_chunks = _calculate_intelligent_chunks(colors_converted.shape)
            group.create_dataset(
                "colors",
                data=colors_converted,
                chunks=color_chunks,
                compressor=self.compressor,
                dtype=color_dtype,
                overwrite=True,
            )
            metadata["has_colors"] = True
            group.attrs["color_dtype"] = np.dtype(color_dtype).name

            # Log appropriate message based on dtype
            if color_dtype == np.float32:
                aprint("  ✓ Wrote HDR colors (float32)")
            elif color_dtype == np.uint16:
                aprint("  ✓ Wrote colors (uint16)")
            else:
                aprint("  ✓ Wrote colors (uint8)")

        if radii is not None:
            # Validate radii
            validate_radii_for_writing(radii, n_points)

            # Calculate max radius for efficient lazy loading (before conversion)
            max_radius = float(np.max(radii))
            metadata["max_radius"] = max_radius
            aprint(f"  ✓ Max radius: {max_radius:.3f}")

            # Determine and apply optimal dtype for radii
            radius_dtype = self.dtype_config.get_radius_dtype(radii)
            radii_converted = convert_array_dtype(
                radii, radius_dtype, normalize=(radius_dtype == np.uint8)
            )

            radii_chunks = _calculate_intelligent_chunks(radii_converted.shape)
            group.create_dataset(
                "radii",
                data=radii_converted,
                chunks=radii_chunks,
                compressor=self.compressor,
                dtype=radius_dtype,
                overwrite=True,
            )
            metadata["has_radii"] = True
            group.attrs["radius_dtype"] = np.dtype(radius_dtype).name
            aprint(f"  ✓ Wrote radii ({radius_dtype})")

        if sharpness is not None:
            # Validate sharpness
            validate_sharpness_for_writing(sharpness, n_points)

            # Determine and apply optimal dtype for sharpness
            sharpness_dtype = self.dtype_config.get_sharpness_dtype(sharpness)
            # Always use [0, 15] range for uint8 sharpness (no more legacy [0,1] behavior)
            input_range = (
                (0.0, SHARPNESS_MAX) if sharpness_dtype == np.uint8 else (0.0, 1.0)
            )
            sharpness_converted = convert_array_dtype(
                sharpness,
                sharpness_dtype,
                normalize=(sharpness_dtype == np.uint8),
                input_range=input_range,
            )

            sharp_chunks = _calculate_intelligent_chunks(sharpness_converted.shape)
            group.create_dataset(
                "sharpness",
                data=sharpness_converted,
                chunks=sharp_chunks,
                compressor=self.compressor,
                dtype=sharpness_dtype,
                overwrite=True,
            )
            metadata["has_sharpness"] = True
            group.attrs["sharpness_dtype"] = np.dtype(sharpness_dtype).name
            aprint(f"  ✓ Wrote sharpness ({sharpness_dtype})")

        # Process transform if present using centralized conversion
        if "transform" in attrs:
            from ..core.transforms import prepare_transform_for_zarr

            attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

        # Set default rendering attributes if not provided
        if "opacity" not in attrs:
            attrs["opacity"] = 1.0
        if "gamma" not in attrs:
            attrs["gamma"] = 1.0
        if "blending_mode" not in attrs:
            attrs["blending_mode"] = "additive"

        # Store spatial extension dimensions if available
        if spatial_extend_dims is not None:
            attrs["spatial_extend_dims"] = spatial_extend_dims

        # Store attributes
        group.attrs.update(attrs)
        group.attrs["type"] = "points"
        group.attrs["n_points"] = n_points

        # Store max_radius if radii were provided
        if "max_radius" in metadata:
            group.attrs["max_radius"] = metadata["max_radius"]

        # Write spatial index if built
        if spatial_index_data is not None:
            aprint("  📝 Writing spatial index...")

            # Create spatial_index group
            index_group = group.require_group("spatial_index")

            # Store index metadata
            index_metadata = {
                "grid_shape": spatial_index_data["grid_shape"].tolist(),
                "grid_origin": spatial_index_data["grid_origin"].tolist(),
                "cell_size": spatial_index_data["cell_size"].tolist(),
                "num_occupied": len(spatial_index_data["occupied_cells"]),
                "total_cells": int(np.prod(spatial_index_data["grid_shape"]))
                if len(spatial_index_data["grid_shape"]) > 0
                else 0,
                "total_points": spatial_index_data[
                    "total_points"
                ],  # Total number of points in dataset
                "dimensions": len(
                    spatial_index_data["indexed_dimensions"]
                ),  # Number of indexed dimensions
                "full_dimensions": spatial_index_data[
                    "full_dimensions"
                ],  # Total dimensions
                "indexed_dimensions": spatial_index_data[
                    "indexed_dimensions"
                ],  # Which dimensions are indexed
                "displayed_dimensions": spatial_index_data[
                    "displayed_dimensions"
                ],  # Which dimensions are displayed
                "build_version": "0.1",
            }

            # Calculate max points per cell
            if len(spatial_index_data["cell_ranges"]) > 0:
                cell_sizes = np.diff(
                    spatial_index_data["cell_ranges"], axis=1
                ).flatten()
                index_metadata["max_points_per_cell"] = int(np.max(cell_sizes))
            else:
                index_metadata["max_points_per_cell"] = 0

            index_group.attrs.update(index_metadata)

            # Store occupied cells
            if len(spatial_index_data["occupied_cells"]) > 0:
                occupied_chunks = _calculate_intelligent_chunks(
                    spatial_index_data["occupied_cells"].shape, target_chunk_size=4096
                )
                index_group.create_dataset(
                    "occupied_cells",
                    data=spatial_index_data["occupied_cells"],
                    chunks=occupied_chunks,
                    compressor=self.compressor,
                    dtype=np.uint32,
                    overwrite=True,
                )

                # Store cell ranges
                ranges_chunks = _calculate_intelligent_chunks(
                    spatial_index_data["cell_ranges"].shape, target_chunk_size=4096
                )
                index_group.create_dataset(
                    "cell_ranges",
                    data=spatial_index_data["cell_ranges"],
                    chunks=ranges_chunks,
                    compressor=self.compressor,
                    dtype=np.uint64,
                    overwrite=True,
                )

            metadata["has_spatial_index"] = True
            aprint("  ✓ Spatial index written")

        # Cache metadata
        self._metadata_cache[path] = metadata

        aprint(f"✅ Points written to {path}")
        return metadata

    def create_resizable_dataset(
        self,
        path: NodePath,
        dtype: Any,
        shape: Tuple[int, ...],
        maxshape: MaxShape = None,
        chunks: ChunkSpec = True,
    ) -> Any:
        """Create a resizable dataset for streaming writes.

        Args:
            path: Path for the dataset
            dtype: Data type
            shape: Initial shape
            maxshape: Maximum shape (None for unlimited)
            chunks: Chunking configuration

        Returns:
            Zarr dataset handle
        """
        path = path.lstrip("/")

        # Parse parent group and dataset name
        parts = path.rsplit("/", 1)
        if len(parts) == 2:
            group_path, dataset_name = parts
            group = self.store.require_group(group_path)
        else:
            dataset_name = path
            group = self.store

        # Create resizable dataset
        dataset = group.create_dataset(
            dataset_name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=self.compressor,
            maxshape=maxshape,
            overwrite=True,
        )

        aprint(f"📝 Created resizable dataset: {path}")
        return dataset

    def finalize(self) -> None:
        """Finalize the Zarr store with metadata consolidation."""
        if self._is_finalized:
            return

        try:
            aprint("🔧 Finalizing Zarr store...")

            # Close the store to ensure all data is written
            if hasattr(self.store, "close"):
                self.store.close()

            # Re-open the store to consolidate metadata
            # This ensures all groups and datasets are properly written to disk
            store = zarr.open_group(self._store_path, mode="r+")

            # Now consolidate metadata with all data present
            zarr.consolidate_metadata(store.store)

            # Close the store again
            if hasattr(store, "close"):
                store.close()

            self._is_finalized = True
            aprint(f"✅ Zarr store finalized at {self._store_path}")

        except Exception as e:
            aprint(f"⚠️ Failed to finalize: {e}")
            raise ValueError(f"Could not finalize Zarr store: {e}") from e

    @property
    def store_path(self) -> str:
        """Get the path to the Zarr store."""
        return str(self._store_path)
