"""luxar.compiler – Progressive Zarr compiler for memory-efficient scene building.

This module provides the LuxarZarrCompiler class which implements progressive
writing to Zarr stores, enabling handling of arbitrarily large datasets without
memory constraints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from ..core.scene import Scene
import zarr
from arbol import aprint
from numpy.typing import NDArray

from ..core.dimensions import Dimensions
from ..encoding import (
    ArrayEncoder,
    EncodingMode,
    SemanticType,
)
from ..io.reader import DEFAULT_COMP
from ..io.writer import ZarrWriterProtocol
from ..typing_utils.aliases import ChunkSpec, MaxShape, NodePath, PointsMetadata
from ..typing_utils.config import DEFAULT_CHUNK_SIZE, DEFAULT_VERSION
from ..typing_utils.constants import (
    SHARPNESS_MAX,
    SPATIAL_INDEX_CHUNK_SIZE,
    SPATIAL_INDEX_FALLBACK_CELLS,
    SPATIAL_INDEX_MAX_CELLS_CONTINUOUS,
    SPATIAL_INDEX_MAX_CELLS_DISCRETE,
    SPATIAL_INDEX_MIN_CELLS,
    SPATIAL_INDEX_TARGET_POINTS_PER_CELL,
)
from ..typing_utils.protocols import CompressorProtocol

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
        version: Luxar format version
        enable_spatial_index: Whether to build spatial indices for points
        encoding_mode: Encoding mode for array storage

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
        version: str = DEFAULT_VERSION,
        enable_spatial_index: bool = True,
        encoding_mode: EncodingMode = EncodingMode.AUTO,
    ) -> None:
        """Initialize the Zarr compiler.

        Args:
            store_path: Path for the Zarr store, or None for temporary
            compressor: Compressor for datasets
            version: Luxar format version
            enable_spatial_index: Whether to build spatial indices for points (default: True)
            encoding_mode: Encoding mode for array storage (AUTO/PRECISION/MEMORY)

        Note:
            Physical units should be specified per-dimension using the Dimensions
            system when calling create_scene(dimensions=...). This provides more
            flexibility for multi-dimensional data.
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

        # Store spatial index flag
        self.enable_spatial_index = enable_spatial_index

        # Create array encoder with specified encoding mode
        self._encoder = ArrayEncoder()
        self._encoding_mode = encoding_mode

        # Create root Zarr group
        self.store = zarr.open_group(self._store_path, mode="w")
        self.store.attrs.update(
            {
                "luxar_version": version,
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

    def create_scene(self, dimensions: Optional[Dimensions] = None) -> "Scene":
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

        # 1. Setup: Create group and validate positions
        path = path.lstrip("/")
        group = self.store.require_group(path)
        n_points, n_dims = validate_positions_for_writing(positions)

        aprint(f"📝 Writing {n_points:,} points ({n_dims}D) to {path}")

        # 2. Build spatial index if enabled (also reorders arrays)
        spatial_index_data, spatial_extend_dims = self._build_spatial_index_if_enabled(
            positions, n_points, n_dims, grid_shape, attrs, radii
        )

        # Apply spatial reordering if index was built
        if spatial_index_data is not None:
            positions = spatial_index_data["sorted_positions"]
            colors = apply_sort_order(colors, spatial_index_data["sort_order"])
            radii = apply_sort_order(radii, spatial_index_data["sort_order"])
            sharpness = apply_sort_order(sharpness, spatial_index_data["sort_order"])

        # 3. Write positions dataset
        self._write_positions_dataset(group, positions, spatial_index_data)

        # 4. Initialize metadata
        metadata: PointsMetadata = {
            "n_points": n_points,
            "dims": n_dims,
            "path": path,
            "has_colors": False,
            "has_radii": False,
            "has_sharpness": False,
        }

        # 5. Write optional datasets
        if colors is not None:
            validate_colors_for_writing(colors, n_points)
            self._write_colors_dataset(group, colors, spatial_index_data)
            metadata["has_colors"] = True

        if radii is not None:
            validate_radii_for_writing(radii, n_points)
            max_radius = self._write_radii_dataset(group, radii, spatial_index_data)
            metadata["max_radius"] = max_radius
            metadata["has_radii"] = True
            group.attrs["max_radius"] = max_radius

        if sharpness is not None:
            validate_sharpness_for_writing(sharpness, n_points)
            self._write_sharpness_dataset(group, sharpness, spatial_index_data)
            metadata["has_sharpness"] = True

        # 6. Process transform if present using centralized conversion
        if "transform" in attrs:
            from ..core.transforms import prepare_transform_for_zarr

            attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

        # 7. Set default rendering attributes if not provided
        if "opacity" not in attrs:
            attrs["opacity"] = 1.0
        if "gamma" not in attrs:
            attrs["gamma"] = 1.0
        if "blending_mode" not in attrs:
            attrs["blending_mode"] = "additive"

        # 8. Store spatial extension dimensions if available
        if spatial_extend_dims is not None:
            attrs["spatial_extend_dims"] = spatial_extend_dims

        # 9. Store attributes
        group.attrs.update(attrs)
        group.attrs["type"] = "points"
        group.attrs["n_points"] = n_points

        # 10. Write spatial index if built
        if spatial_index_data is not None:
            self._write_spatial_index_to_zarr(group, spatial_index_data)
            metadata["has_spatial_index"] = True

        # 11. Cache metadata and finish
        self._metadata_cache[path] = metadata
        aprint(f"✅ Points written to {path}")

        return metadata

    def write_lines(
        self,
        path: NodePath,
        vertices: NDArray[np.float32],
        widths: NDArray[np.float32],
        colors: Optional[NDArray[np.float32]] = None,
        sharpness: Optional[NDArray[np.float32]] = None,
        indices: Optional[NDArray[np.uint32]] = None,
        line_type: str = "polyline",
        **attrs: Any,
    ) -> dict[str, Any]:
        """Write lines data to Zarr.

        Lines do NOT support spatial indexing (per spec) - arrays stored in original order.

        Args:
            path: Path for the lines within the store
            vertices: Vertex positions of shape (N, D)
            widths: Line widths of shape (N,)
            colors: Optional colors of shape (N, 3)
            sharpness: Optional sharpness of shape (N,)
            indices: Optional vertex indices for indexed line type
            line_type: Type of line connectivity
            **attrs: Additional attributes

        Returns:
            Metadata dictionary about the written lines
        """
        from ..validation.base import (
            validate_colors_for_writing,
            validate_positions_for_writing,
        )

        # Setup and validation
        path = path.lstrip("/")
        group = self.store.require_group(path)
        n_vertices, n_dims = validate_positions_for_writing(vertices)

        aprint(f"📝 Writing {n_vertices:,} line vertices ({n_dims}D) to {path}")

        # Validate line type
        valid_line_types = ("segments", "polyline", "loop", "indexed")
        if line_type not in valid_line_types:
            raise ValueError(
                f"Invalid line_type '{line_type}'. Must be one of {valid_line_types}"
            )

        # Validate type-specific requirements
        if line_type == "segments" and n_vertices % 2 != 0:
            raise ValueError(
                f"Segments require even number of vertices, got {n_vertices}"
            )
        if line_type == "polyline" and n_vertices < 2:
            raise ValueError(f"Polyline requires at least 2 vertices, got {n_vertices}")
        if line_type == "loop" and n_vertices < 3:
            raise ValueError(f"Loop requires at least 3 vertices, got {n_vertices}")
        if line_type == "indexed":
            if indices is None:
                raise ValueError("Indexed line type requires indices array")
            if len(indices) < 2:
                raise ValueError("Indexed requires at least 2 indices")
            if len(indices) % 2 != 0:
                raise ValueError("Indices must have even length (pairs)")
            if np.max(indices) >= n_vertices:
                raise ValueError(f"Index {np.max(indices)} >= n_vertices {n_vertices}")

        # Validate widths (must be positive like radii)
        if widths.shape[0] != n_vertices:
            raise ValueError(
                f"Widths shape {widths.shape} doesn't match n_vertices {n_vertices}"
            )
        if np.any(widths <= 0):
            min_val = float(np.min(widths))
            raise ValueError(
                f"Widths must be positive (> 0). Found minimum value: {min_val:.3f}"
            )

        # Calculate n_segments
        if line_type == "segments":
            n_segments = n_vertices // 2
        elif line_type == "polyline":
            n_segments = n_vertices - 1
        elif line_type == "loop":
            n_segments = n_vertices
        elif line_type == "indexed":
            n_segments = len(indices) // 2

        # Write vertices using ArrayEncoder (COORDINATE)
        chunks_2d = _calculate_intelligent_chunks((n_vertices, n_dims))
        self._encoder.encode(
            data=vertices,
            zarr_group=group,
            name="vertices",
            semantic_type=SemanticType.COORDINATE,
            mode=self._encoding_mode,
            chunks=chunks_2d,
            compressor=self.compressor,
        )

        # Write widths using ArrayEncoder (POSITIVE_SCALAR)
        max_width = float(np.max(widths))
        chunks_1d = _calculate_intelligent_chunks((n_vertices,))
        self._encoder.encode(
            data=widths,
            zarr_group=group,
            name="widths",
            semantic_type=SemanticType.POSITIVE_SCALAR,
            mode=self._encoding_mode,
            chunks=chunks_1d,
            compressor=self.compressor,
        )

        # Initialize metadata
        metadata: dict[str, Any] = {
            "n_vertices": n_vertices,
            "n_segments": n_segments,
            "ndim": n_dims,
            "line_type": line_type,
            "has_colors": False,
            "has_sharpness": False,
            "max_width": max_width,
        }

        # Write optional datasets
        if colors is not None:
            validate_colors_for_writing(colors, n_vertices)
            chunks_colors = _calculate_intelligent_chunks(colors.shape)

            # Detect color_mode
            color_mode = None
            if np.issubdtype(colors.dtype, np.floating):
                color_mode = "hdr" if np.any(colors > 1.0) else "sdr"

            self._encoder.encode(
                data=colors,
                zarr_group=group,
                name="colors",
                semantic_type=SemanticType.COLOR,
                mode=self._encoding_mode,
                color_mode=color_mode,
                chunks=chunks_colors,
                compressor=self.compressor,
            )
            metadata["has_colors"] = True

        if sharpness is not None:
            from ..validation.base import validate_sharpness_for_writing

            validate_sharpness_for_writing(sharpness, n_vertices)
            self._encoder.encode(
                data=sharpness,
                zarr_group=group,
                name="sharpness",
                semantic_type=SemanticType.BOUNDED_SCALAR,
                mode=self._encoding_mode,
                bounds=(0.0, SHARPNESS_MAX),
                chunks=chunks_1d,
                compressor=self.compressor,
            )
            metadata["has_sharpness"] = True

        # Write indices if provided (INDEX semantic type)
        if indices is not None:
            chunks_indices = _calculate_intelligent_chunks((len(indices),))
            self._encoder.encode(
                data=indices,
                zarr_group=group,
                name="indices",
                semantic_type=SemanticType.INDEX,
                mode=self._encoding_mode,
                chunks=chunks_indices,
                compressor=self.compressor,
            )

        # Process transform if present
        if "transform" in attrs:
            from ..core.transforms import prepare_transform_for_zarr

            attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

        # Set attributes
        group.attrs.update(attrs)
        group.attrs["type"] = "lines"
        group.attrs["line_type"] = line_type
        group.attrs["n_vertices"] = n_vertices
        group.attrs["n_segments"] = n_segments
        group.attrs["max_width"] = max_width

        self._metadata_cache[path] = metadata
        aprint(f"✅ Lines written to {path}")

        return metadata

    def write_gsplats(
        self,
        path: NodePath,
        centers: NDArray[np.float32],
        amplitudes: NDArray[np.float32],
        cholesky_factors: NDArray[np.float32],
        colors: Optional[NDArray[np.float32]] = None,
        sharpness: Optional[NDArray[np.float32]] = None,
        **attrs: Any,
    ) -> dict[str, Any]:
        """Write Gaussian splats data to Zarr.

        Args:
            path: Path for the gsplats within the store
            centers: Splat centers of shape (N, D)
            amplitudes: Amplitudes of shape (N,)
            cholesky_factors: Packed Cholesky factors of shape (N, k)
            colors: Optional colors of shape (N, 3)
            sharpness: Optional sharpness of shape (N,)
            **attrs: Additional attributes

        Returns:
            Metadata dictionary about the written gsplats
        """
        from ..validation.base import (
            validate_colors_for_writing,
            validate_positions_for_writing,
        )

        # Setup and validation
        path = path.lstrip("/")
        group = self.store.require_group(path)
        n_splats, n_dims = validate_positions_for_writing(centers)

        aprint(f"📝 Writing {n_splats:,} gsplats ({n_dims}D) to {path}")

        # Validate amplitudes (must be non-negative: >= 0, zero is valid but invisible)
        if amplitudes.shape[0] != n_splats:
            raise ValueError(
                f"Amplitudes shape {amplitudes.shape} doesn't match n_splats {n_splats}"
            )
        if np.any(amplitudes < 0):
            min_val = float(np.min(amplitudes))
            raise ValueError(
                f"Amplitudes must be non-negative (>= 0). Found minimum value: {min_val:.3f}"
            )

        # Validate cholesky_factors shape
        expected_k = n_dims * (n_dims + 1) // 2
        if cholesky_factors.ndim == 1:
            if cholesky_factors.shape[0] != expected_k:
                raise ValueError(
                    f"Cholesky factors shape mismatch: expected ({expected_k},), "
                    f"got {cholesky_factors.shape}"
                )
            # Broadcast to all splats
            cholesky_factors = np.broadcast_to(
                cholesky_factors, (n_splats, expected_k)
            ).copy()
        elif cholesky_factors.shape != (n_splats, expected_k):
            raise ValueError(
                f"Cholesky factors shape mismatch: expected ({n_splats}, {expected_k}), "
                f"got {cholesky_factors.shape}"
            )

        # Write centers using ArrayEncoder (COORDINATE)
        chunks_centers = _calculate_intelligent_chunks(centers.shape)
        self._encoder.encode(
            data=centers,
            zarr_group=group,
            name="centers",
            semantic_type=SemanticType.COORDINATE,
            mode=self._encoding_mode,
            chunks=chunks_centers,
            compressor=self.compressor,
        )

        # Write amplitudes using ArrayEncoder (POSITIVE_SCALAR, can be zero)
        amplitude_min, amplitude_max = (
            float(np.min(amplitudes)),
            float(np.max(amplitudes)),
        )
        chunks_1d = _calculate_intelligent_chunks((n_splats,))
        self._encoder.encode(
            data=amplitudes,
            zarr_group=group,
            name="amplitudes",
            semantic_type=SemanticType.POSITIVE_SCALAR,
            mode=self._encoding_mode,
            chunks=chunks_1d,
            compressor=self.compressor,
        )

        # Write cholesky_factors using ArrayEncoder (CHOLESKY)
        chunks_cholesky = _calculate_intelligent_chunks(cholesky_factors.shape)
        self._encoder.encode(
            data=cholesky_factors,
            zarr_group=group,
            name="cholesky_factors",
            semantic_type=SemanticType.CHOLESKY,
            mode=self._encoding_mode,
            chunks=chunks_cholesky,
            compressor=self.compressor,
        )

        # Initialize metadata
        center_min = centers.min(axis=0).tolist()
        center_max = centers.max(axis=0).tolist()
        metadata: dict[str, Any] = {
            "n_splats": n_splats,
            "ndim": n_dims,
            "has_colors": False,
            "has_sharpness": False,
            "ordering": "none",  # TODO: Implement Morton ordering
            "amplitude_range": {"min": amplitude_min, "max": amplitude_max},
            "center_bounds": {"min": center_min, "max": center_max},
        }

        # Write optional datasets
        if colors is not None:
            validate_colors_for_writing(colors, n_splats)
            chunks_colors = _calculate_intelligent_chunks(colors.shape)

            # Detect color_mode
            color_mode = None
            if np.issubdtype(colors.dtype, np.floating):
                color_mode = "hdr" if np.any(colors > 1.0) else "sdr"

            self._encoder.encode(
                data=colors,
                zarr_group=group,
                name="colors",
                semantic_type=SemanticType.COLOR,
                mode=self._encoding_mode,
                color_mode=color_mode,
                chunks=chunks_colors,
                compressor=self.compressor,
            )
            metadata["has_colors"] = True

        if sharpness is not None:
            from ..validation.base import validate_sharpness_for_writing

            validate_sharpness_for_writing(sharpness, n_splats)
            self._encoder.encode(
                data=sharpness,
                zarr_group=group,
                name="sharpness",
                semantic_type=SemanticType.BOUNDED_SCALAR,
                mode=self._encoding_mode,
                bounds=(0.0, SHARPNESS_MAX),
                chunks=chunks_1d,
                compressor=self.compressor,
            )
            metadata["has_sharpness"] = True

        # Process transform if present
        if "transform" in attrs:
            from ..core.transforms import prepare_transform_for_zarr

            attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

        # Set attributes
        group.attrs.update(attrs)
        group.attrs["type"] = "gsplats"
        group.attrs["n_splats"] = n_splats
        group.attrs["amplitude_range"] = metadata["amplitude_range"]
        group.attrs["center_bounds"] = metadata["center_bounds"]

        self._metadata_cache[path] = metadata
        aprint(f"✅ GSplats written to {path}")

        return metadata

    def create_resizable_dataset(
        self,
        path: NodePath,
        dtype: np.dtype,
        shape: Tuple[int, ...],
        maxshape: MaxShape = None,
        chunks: ChunkSpec = True,
    ) -> zarr.Array:
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

    def _determine_displayed_and_indexed_dims(
        self, n_dims: int
    ) -> Tuple[List[int], List[int], Optional[List[bool]]]:
        """Determine which dimensions are displayed vs indexed from scene metadata.

        Args:
            n_dims: Total number of dimensions in the data

        Returns:
            Tuple of (displayed_dims, non_displayed_dims, spatial_extend_dims)
        """
        displayed_dims = None
        spatial_extend_dims = None

        if "scene_dimensions" in self.store.attrs:
            scene_dims = self.store.attrs["scene_dimensions"]
            dims_list = (
                scene_dims.get("dimensions", []) if isinstance(scene_dims, dict) else []
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

        return displayed_dims, non_displayed_dims, spatial_extend_dims

    def _calculate_spatial_index_grid_shape(
        self,
        positions: NDArray[np.float32],
        non_displayed_dims: List[int],
        grid_shape: Optional[Tuple[int, ...]],
    ) -> Optional[NDArray[np.uint32]]:
        """Calculate grid shape for spatial index based on data characteristics.

        Args:
            positions: Point positions array
            non_displayed_dims: Indices of non-displayed dimensions to index
            grid_shape: User-provided grid shape (if any)

        Returns:
            Grid shape array or None
        """
        # Use provided grid shape if available
        if grid_shape is not None:
            if len(grid_shape) == len(non_displayed_dims):
                return np.array(grid_shape, dtype=np.uint32)
            else:
                aprint(
                    f"    ⚠️ Provided grid_shape has {len(grid_shape)} dims "
                    f"but {len(non_displayed_dims)} non-displayed dims"
                )

        # Auto-determine grid shape
        if "scene_dimensions" not in self.store.attrs:
            return None

        scene_dims = self.store.attrs["scene_dimensions"]
        grid_shape_list = []

        # Calculate position ranges for non-displayed dimensions only
        indexed_positions = positions[:, non_displayed_dims]
        min_coords = np.min(indexed_positions, axis=0)
        max_coords = np.max(indexed_positions, axis=0)
        n_points = positions.shape[0]

        # Get dimension list if available
        dims_list = (
            scene_dims.get("dimensions", []) if isinstance(scene_dims, dict) else []
        )

        for idx, d in enumerate(non_displayed_dims):
            # Determine if this dimension contains discrete values (categorical/time)
            is_discrete = False
            if d < len(dims_list) and dims_list[d].get("discrete", False):
                is_discrete = True

            if is_discrete:
                # For discrete dimensions (time, channel, category):
                # Use one grid cell per unique value for precise indexing
                # This enables exact lookups (e.g., "give me all points at time=5")
                unique_vals = len(np.unique(indexed_positions[:, idx]))
                cells = min(
                    SPATIAL_INDEX_MAX_CELLS_DISCRETE, unique_vals
                )  # Safety cap to prevent memory issues
                grid_shape_list.append(cells)
                aprint(
                    f"    Non-displayed dim {d}: discrete with {unique_vals} "
                    f"unique values → {cells} cells"
                )
            else:
                # For continuous dimensions (depth, intensity):
                # Use fewer cells with adaptive sizing based on data density
                # Goal: Balance between index size and query precision
                dim_range = max_coords[idx] - min_coords[idx]
                if dim_range > 0:
                    # Heuristic: sqrt(n_points / TARGET) gives reasonable cell count
                    # More points → more cells (up to max)
                    # Ensures each cell has roughly 100-1000 points
                    cells = min(
                        SPATIAL_INDEX_MAX_CELLS_CONTINUOUS,
                        max(
                            SPATIAL_INDEX_MIN_CELLS,
                            int(
                                np.sqrt(n_points / SPATIAL_INDEX_TARGET_POINTS_PER_CELL)
                            ),
                        ),
                    )
                else:
                    # Degenerate case: all points at same value
                    cells = SPATIAL_INDEX_FALLBACK_CELLS  # Minimal grid
                grid_shape_list.append(cells)
                aprint(f"    Non-displayed dim {d}: continuous → {cells} cells")

        if grid_shape_list:
            result = np.array(grid_shape_list, dtype=np.uint32)
            aprint(f"  📊 Grid shape for non-displayed dims: {result}")
            return result

        return None

    def _build_spatial_index_if_enabled(
        self,
        positions: NDArray[np.float32],
        n_points: int,
        n_dims: int,
        grid_shape: Optional[Tuple[int, ...]],
        attrs: Dict[str, Any],
        radii: Optional[NDArray[np.float32]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[bool]]]:
        """Build spatial index for points if enabled.

        Args:
            positions: Point positions
            n_points: Number of points
            n_dims: Number of dimensions
            grid_shape: Optional user-provided grid shape
            attrs: Point attributes (may contain max_radius)
            radii: Optional radii array

        Returns:
            Tuple of (spatial_index_data, spatial_extend_dims)
        """
        if not self.enable_spatial_index or n_points == 0:
            return None, None

        # Determine displayed and indexed dimensions
        displayed_dims, non_displayed_dims, spatial_extend_dims = (
            self._determine_displayed_and_indexed_dims(n_dims)
        )

        # Skip if all dimensions are displayed
        if len(non_displayed_dims) == 0:
            aprint(
                f"  ⚠️ All {n_dims} dimensions are displayed - skipping spatial index"
            )
            return None, spatial_extend_dims

        aprint("  🔍 Building spatial index...")
        aprint(f"    Displayed dimensions: {displayed_dims}")
        aprint(f"    Non-displayed dimensions to index: {non_displayed_dims}")

        # Calculate grid shape
        grid_shape_array = self._calculate_spatial_index_grid_shape(
            positions, non_displayed_dims, grid_shape
        )

        # Determine discrete dimensions
        discrete_dims = []
        if "scene_dimensions" in self.store.attrs:
            scene_dims = self.store.attrs["scene_dimensions"]
            dims_list = (
                scene_dims.get("dimensions", []) if isinstance(scene_dims, dict) else []
            )
            for d in range(n_dims):
                if d < len(dims_list) and dims_list[d].get("discrete", False):
                    discrete_dims.append(d)

        # Get max_radius for spatial dimensions
        max_radius = attrs.get("max_radius", 0.1)
        if radii is not None:
            max_radius = float(np.max(radii))

        # Build the spatial index
        spatial_index_data = build_spatial_index(
            positions,
            grid_shape_array,
            displayed_dims=displayed_dims,
            discrete_dims=discrete_dims,
            spatial_extend_dims=spatial_extend_dims,
            max_radius=max_radius,
        )

        # Validate the index
        if spatial_index_data is not None:
            validate_spatial_index(spatial_index_data, n_points, n_dims)
            aprint(
                f"  ✓ Spatial index built: {len(spatial_index_data['occupied_cells'])} occupied cells"
            )

        return spatial_index_data, spatial_extend_dims

    def _write_positions_dataset(
        self,
        group: zarr.Group,
        positions: NDArray[np.float32],
        spatial_index_data: Optional[Dict[str, Any]],
    ) -> None:
        """Write positions dataset to Zarr using ArrayEncoder.

        Args:
            group: Zarr group to write to
            positions: Positions array (may be reordered by spatial index)
            spatial_index_data: Optional spatial index data for chunk optimization
        """
        # Calculate intelligent chunks (aligned with spatial index if available)
        chunks = _calculate_intelligent_chunks(
            positions.shape,
            target_chunk_size=DEFAULT_CHUNK_SIZE,
            spatial_index_data=spatial_index_data,
        )

        # Use ArrayEncoder for positions (COORDINATE semantic type)
        self._encoder.encode(
            data=positions,
            zarr_group=group,
            name="positions",
            semantic_type=SemanticType.COORDINATE,
            mode=self._encoding_mode,
            chunks=chunks,
            compressor=self.compressor,
        )

        # Log encoding result
        enc = group["positions"].attrs.get("encoding", {})
        enc_name = enc.get("name", "unknown")
        if enc_name == "float16":
            aprint("  ✓ Wrote positions (float16 - MEMORY mode)")
        else:
            aprint("  ✓ Wrote positions (float32)")

    def _write_colors_dataset(
        self,
        group: zarr.Group,
        colors: NDArray[np.float32],
        spatial_index_data: Optional[Dict[str, Any]],
    ) -> None:
        """Write colors dataset to Zarr using ArrayEncoder.

        Args:
            group: Zarr group to write to
            colors: Colors array
            spatial_index_data: Optional spatial index for chunk optimization
        """
        # Calculate chunks
        color_chunks = _calculate_intelligent_chunks(colors.shape)

        # Detect color_mode for float arrays
        color_mode = None
        if np.issubdtype(colors.dtype, np.floating):
            # Float colors require explicit color_mode
            if np.any(colors > 1.0):
                color_mode = "hdr"
                aprint("  ✓ Detected HDR colors (values > 1.0)")
            else:
                color_mode = "sdr"

        # Use ArrayEncoder with all optimizations
        self._encoder.encode(
            data=colors,
            zarr_group=group,
            name="colors",
            semantic_type=SemanticType.COLOR,
            mode=self._encoding_mode,
            color_mode=color_mode,
            chunks=color_chunks,
            compressor=self.compressor,
        )

        # Log encoding result
        enc = group["colors"].attrs.get("encoding", {})
        enc_name = enc.get("name", "unknown")
        if enc_name == "broadcasted":
            aprint("  ✓ Wrote colors (broadcasted - uniform)")
        elif enc_name == "array_ref":
            aprint(f"  ✓ Wrote colors (reference to {enc['target']})")
        elif enc_name == "lut_uint8":
            aprint(f"  ✓ Wrote colors (LUT with {len(enc['lut'])} unique values)")
        elif enc_name in ("rgb_uint8", "uint8"):
            aprint("  ✓ Wrote colors (uint8)")
        elif enc_name in ("rgb_uint16", "uint16"):
            aprint("  ✓ Wrote colors (uint16)")
        elif enc_name == "float32":
            aprint("  ✓ Wrote HDR colors (float32)")
        else:
            aprint(f"  ✓ Wrote colors ({enc_name})")

    def _write_radii_dataset(
        self,
        group: zarr.Group,
        radii: NDArray[np.float32],
        spatial_index_data: Optional[Dict[str, Any]],
    ) -> float:
        """Write radii dataset to Zarr using ArrayEncoder.

        Args:
            group: Zarr group to write to
            radii: Radii array
            spatial_index_data: Optional spatial index for chunk optimization

        Returns:
            Maximum radius value
        """
        # Calculate max radius before encoding
        max_radius = float(np.max(radii))
        aprint(f"  ✓ Max radius: {max_radius:.3f}")

        # Calculate chunks
        radii_chunks = _calculate_intelligent_chunks(radii.shape)

        # Use ArrayEncoder for radii (POSITIVE_SCALAR semantic type)
        self._encoder.encode(
            data=radii,
            zarr_group=group,
            name="radii",
            semantic_type=SemanticType.POSITIVE_SCALAR,
            mode=self._encoding_mode,
            chunks=radii_chunks,
            compressor=self.compressor,
        )

        # Log encoding result
        enc = group["radii"].attrs.get("encoding", {})
        enc_name = enc.get("name", "unknown")
        if enc_name == "broadcasted":
            aprint("  ✓ Wrote radii (broadcasted - uniform)")
        elif enc_name == "array_ref":
            aprint(f"  ✓ Wrote radii (reference to {enc['target']})")
        elif enc_name.startswith("log_scalar"):
            aprint(f"  ✓ Wrote radii ({enc_name} - log encoding)")
        else:
            aprint(f"  ✓ Wrote radii ({enc_name})")

        return max_radius

    def _write_sharpness_dataset(
        self,
        group: zarr.Group,
        sharpness: NDArray[np.float32],
        spatial_index_data: Optional[Dict[str, Any]],
    ) -> None:
        """Write sharpness dataset to Zarr using ArrayEncoder.

        Args:
            group: Zarr group to write to
            sharpness: Sharpness array
            spatial_index_data: Optional spatial index for chunk optimization
        """
        # Calculate chunks
        sharp_chunks = _calculate_intelligent_chunks(sharpness.shape)

        # Use ArrayEncoder for sharpness (BOUNDED_SCALAR with [0, 31] range)
        self._encoder.encode(
            data=sharpness,
            zarr_group=group,
            name="sharpness",
            semantic_type=SemanticType.BOUNDED_SCALAR,
            mode=self._encoding_mode,
            bounds=(0.0, SHARPNESS_MAX),
            chunks=sharp_chunks,
            compressor=self.compressor,
        )

        # Log encoding result
        enc = group["sharpness"].attrs.get("encoding", {})
        enc_name = enc.get("name", "unknown")
        if enc_name == "broadcasted":
            aprint("  ✓ Wrote sharpness (broadcasted - uniform)")
        elif enc_name == "array_ref":
            aprint(f"  ✓ Wrote sharpness (reference to {enc['target']})")
        elif enc_name == "bounded_scalar_uint8":
            aprint("  ✓ Wrote sharpness (uint8, quantized to [0, 31])")
        else:
            aprint(f"  ✓ Wrote sharpness ({enc_name})")

    def _write_spatial_index_to_zarr(
        self, group: zarr.Group, spatial_index_data: Dict[str, Any]
    ) -> None:
        """Write spatial index metadata and data to Zarr.

        Args:
            group: Parent Zarr group
            spatial_index_data: Spatial index data to write
        """
        aprint("  📝 Writing spatial index...")

        # Create spatial_index group
        index_group = group.require_group("spatial_index")

        # Store index metadata
        index_metadata = {
            "grid_shape": spatial_index_data["grid_shape"].tolist(),
            "grid_origin": spatial_index_data["grid_origin"].tolist(),
            "cell_size": spatial_index_data["cell_size"].tolist(),
            "num_occupied": len(spatial_index_data["occupied_cells"]),
            "total_cells": (
                int(np.prod(spatial_index_data["grid_shape"]))
                if len(spatial_index_data["grid_shape"]) > 0
                else 0
            ),
            "total_points": spatial_index_data["total_points"],
            "dimensions": len(spatial_index_data["indexed_dimensions"]),
            "full_dimensions": spatial_index_data["full_dimensions"],
            "indexed_dimensions": spatial_index_data["indexed_dimensions"],
            "displayed_dimensions": spatial_index_data["displayed_dimensions"],
            "build_version": "0.1",
        }

        # Calculate max points per cell
        if len(spatial_index_data["cell_ranges"]) > 0:
            cell_sizes = np.diff(spatial_index_data["cell_ranges"], axis=1).flatten()
            index_metadata["max_points_per_cell"] = int(np.max(cell_sizes))
        else:
            index_metadata["max_points_per_cell"] = 0

        index_group.attrs.update(index_metadata)

        # Store occupied cells
        if len(spatial_index_data["occupied_cells"]) > 0:
            occupied_chunks = _calculate_intelligent_chunks(
                spatial_index_data["occupied_cells"].shape,
                target_chunk_size=SPATIAL_INDEX_CHUNK_SIZE,
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
                spatial_index_data["cell_ranges"].shape,
                target_chunk_size=SPATIAL_INDEX_CHUNK_SIZE,
            )
            index_group.create_dataset(
                "cell_ranges",
                data=spatial_index_data["cell_ranges"],
                chunks=ranges_chunks,
                compressor=self.compressor,
                dtype=np.uint64,
                overwrite=True,
            )

        aprint("  ✓ Spatial index written")

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
