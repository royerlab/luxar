"""luxar.compiler – Progressive Zarr compiler for memory-efficient scene building.

This module provides the LuxarZarrCompiler class which implements progressive
writing to Zarr stores, enabling handling of arbitrarily large datasets without
memory constraints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

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
from ..typing_utils.constants import SHARPNESS_MAX
from ..typing_utils.protocols import CompressorProtocol

# Ordering functions will be imported locally where needed to avoid circular imports


def _calculate_intelligent_chunks(
    shape: Tuple[int, ...],
    target_chunk_size: int = DEFAULT_CHUNK_SIZE,
    spatial_index_data: Optional[Dict[str, Any]] = None,
) -> Tuple[int, ...]:
    """Calculate optimal chunk shape for a dataset.

    When spatial ordering data is available, uses chunk_size from ordering.

    Args:
        shape: Shape of the dataset
        target_chunk_size: Target size for chunks in elements
        spatial_index_data: Optional ordering data (with chunk_size)

    Returns:
        Optimized chunk shape
    """
    if len(shape) == 1:
        # 1D array - simple chunking
        return (min(shape[0], target_chunk_size),)

    if len(shape) == 2:
        # 2D array (e.g., positions) - chunk along first dimension
        n_points, n_dims = shape

        # If ordering data is available, use its chunk_size
        if spatial_index_data and "chunk_size" in spatial_index_data:
            chunk_points = spatial_index_data["chunk_size"]
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

        Large datasets (split into multiple nodes):
        >>> with LuxarZarrCompiler('huge.zarr', ordering_method="hilbert") as compiler:
        ...     scene = compiler.create_scene()
        ...     # Process chunks one at a time, each becomes a separate node
        ...     for i in range(100):
        ...         chunk_positions, chunk_colors = load_chunk(i)  # 10M points
        ...         scene.add_points(f'chunk_{i}', chunk_positions, colors=chunk_colors)
        ...         # Each node: sorted, has chunk_bounds, memory freed after write
    """

    def __init__(
        self,
        store_path: Optional[Union[str, Path]] = None,
        compressor: Optional[CompressorProtocol] = DEFAULT_COMP,
        version: str = DEFAULT_VERSION,
        enable_spatial_index: bool = True,
        encoding_mode: EncodingMode = EncodingMode.AUTO,
        ordering_method: Literal["morton", "hilbert"] = "morton",
    ) -> None:
        """Initialize the Zarr compiler.

        Args:
            store_path: Path for the Zarr store, or None for temporary
            compressor: Compressor for datasets
            version: Luxar format version
            enable_spatial_index: Whether to use spatial ordering for points/gsplats (default: True)
            encoding_mode: Encoding mode for array storage (AUTO/PRECISION/MEMORY)
            ordering_method: Spatial ordering method ("morton" or "hilbert", default: "morton")

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

        # Store spatial ordering configuration
        self.enable_spatial_index = enable_spatial_index
        self.ordering_method = ordering_method

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
        colors: Optional[
            Union[NDArray[np.float32], List[float], Tuple[float, ...]]
        ] = None,
        radii: Optional[Union[NDArray[np.float32], float]] = None,
        sharpness: Optional[Union[NDArray[np.float32], float]] = None,
        **attrs: Any,
    ) -> PointsMetadata:
        """Write points data progressively to Zarr.

        Data is written immediately to disk without being kept in memory.
        If spatial ordering is enabled, points are reordered using Morton/Hilbert curves.

        Scalar convenience: radii, sharpness, and colors accept scalars:
        - radii=0.5 → all points get radius 0.5 (broadcasted)
        - colors=[1.0, 0, 0] → all points red (broadcasted)
        - sharpness=2.0 → all points standard Gaussian (broadcasted)

        Args:
            path: Path for the points within the store
            positions: Point positions of shape (N, D)
            colors: Colors - array (N, 3), RGB tuple/list, or None
            radii: Radii - array (N,), scalar float, or None
            sharpness: Sharpness - array (N,), scalar float, or None
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

        # 2. Log scalar inputs (no expansion - passed to encoder)
        if radii is not None and isinstance(radii, (int, float)):
            aprint(f"  → Uniform radius {radii:.3f} for all points")
        if sharpness is not None and isinstance(sharpness, (int, float)):
            aprint(f"  → Uniform sharpness {sharpness:.1f} for all points")
        if colors is not None and isinstance(colors, (list, tuple)):
            aprint(f"  → Uniform color RGB{list(colors)} for all points")

        # 3. Apply spatial ordering if enabled (reorders arrays only)
        # Note: Spatial ordering requires radii to compute chunk_bounds
        # If radii is scalar, create temp array just for ordering
        radii_for_ordering = radii
        if radii is not None and isinstance(radii, (int, float)):
            radii_for_ordering = np.full(n_points, float(radii), dtype=np.float32)

        ordering_data = self._build_spatial_ordering_if_enabled(
            positions, n_points, n_dims, radii_for_ordering
        )

        # Apply spatial reordering to arrays only (skip scalars)
        if ordering_data is not None:
            positions = ordering_data["sorted_positions"]
            # Apply sort order only to array attributes
            if colors is not None and isinstance(colors, np.ndarray):
                colors = colors[ordering_data["sort_order"]]
            if radii is not None and isinstance(radii, np.ndarray):
                radii = radii[ordering_data["sort_order"]]
            if sharpness is not None and isinstance(sharpness, np.ndarray):
                sharpness = sharpness[ordering_data["sort_order"]]
            # Scalars stay as-is (they're uniform, order doesn't matter)

        # 3. Write positions dataset
        self._write_positions_dataset(group, positions, ordering_data)

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
            # Validate arrays only (scalars validated by encoder)
            if isinstance(colors, np.ndarray):
                validate_colors_for_writing(colors, n_points)
            self._write_colors_dataset(group, colors, ordering_data)
            metadata["has_colors"] = True

        if radii is not None:
            # Validate arrays only (scalars validated by encoder)
            if isinstance(radii, np.ndarray):
                validate_radii_for_writing(radii, n_points)
            max_radius = self._write_radii_dataset(group, radii, ordering_data)
            metadata["max_radius"] = max_radius
            metadata["has_radii"] = True
            group.attrs["max_radius"] = max_radius

        if sharpness is not None:
            # Validate arrays only (scalars validated by encoder)
            if isinstance(sharpness, np.ndarray):
                validate_sharpness_for_writing(sharpness, n_points)
            self._write_sharpness_dataset(group, sharpness, ordering_data)
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

        # 8. Store attributes
        group.attrs.update(attrs)
        group.attrs["type"] = "points"
        group.attrs["n_points"] = n_points

        # 9. Write spatial ordering metadata if built
        if ordering_data is not None:
            self._write_spatial_ordering_to_zarr(group, ordering_data)
            metadata["has_spatial_index"] = True

        # 10. Cache metadata and finish
        self._metadata_cache[path] = metadata
        aprint(f"✅ Points written to {path}")

        return metadata

    def write_lines(
        self,
        path: NodePath,
        vertices: NDArray[np.float32],
        widths: Union[NDArray[np.float32], float],
        colors: Optional[
            Union[NDArray[np.float32], List[float], Tuple[float, ...]]
        ] = None,
        sharpness: Optional[Union[NDArray[np.float32], float]] = None,
        indices: Optional[NDArray[np.uint32]] = None,
        line_type: str = "polyline",
        **attrs: Any,
    ) -> dict[str, Any]:
        """Write lines data to Zarr.

        Lines do NOT support spatial indexing (per spec) - arrays stored in original order.

        Scalar convenience: widths, colors, and sharpness accept scalars:
        - widths=0.1 → all vertices get width 0.1
        - colors=[1.0, 0, 0] → all vertices red
        - sharpness=2.0 → uniform sharpness

        Args:
            path: Path for the lines within the store
            vertices: Vertex positions of shape (N, D)
            widths: Line widths - array (N,) or scalar float
            colors: Colors - array (N, 3), RGB tuple/list, or None
            sharpness: Sharpness - array (N,), scalar float, or None
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

        # Scalars are now passed directly to encoder - no expansion needed
        # Just log what we're receiving
        if isinstance(widths, (int, float)):
            aprint(f"  → Uniform width {widths:.3f} for all vertices")
        if sharpness is not None and isinstance(sharpness, (int, float)):
            aprint(f"  → Uniform sharpness {sharpness:.1f} for all vertices")
        if colors is not None and isinstance(colors, (list, tuple)):
            aprint(f"  → Uniform color RGB{list(colors)} for all vertices")

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
        # Skip array-specific validation for scalars (handled by encoder)
        if isinstance(widths, np.ndarray):
            if widths.shape[0] != n_vertices:
                raise ValueError(
                    f"Widths shape {widths.shape} doesn't match n_vertices {n_vertices}"
                )
            if np.any(widths <= 0):
                min_val = float(np.min(widths))
                raise ValueError(
                    f"Widths must be positive (> 0). Found minimum value: {min_val:.3f}"
                )
        elif isinstance(widths, (int, float)) and widths <= 0:
            raise ValueError(f"Width must be positive (> 0). Got {widths}")

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
        # Handle both scalar and array inputs
        if isinstance(widths, (int, float)):
            max_width = float(widths)
            n_elems = n_vertices
            chunks_1d = None  # Scalar doesn't need chunks
        else:
            max_width = float(np.max(widths))
            n_elems = None  # Array already has correct size
            chunks_1d = _calculate_intelligent_chunks((n_vertices,))

        self._encoder.encode(
            data=widths,
            zarr_group=group,
            name="widths",
            semantic_type=SemanticType.POSITIVE_SCALAR,
            mode=self._encoding_mode,
            n_elements=n_elems,
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
            # Handle both scalar/tuple and array inputs
            if isinstance(colors, (tuple, list)):
                # Scalar color input - detect HDR vs SDR
                max_val = max(colors)
                color_mode = "hdr" if max_val > 1.0 else "sdr"
                n_elems_color = n_vertices
                chunks_colors = None
            elif isinstance(colors, np.ndarray):
                validate_colors_for_writing(colors, n_vertices)
                chunks_colors = _calculate_intelligent_chunks(colors.shape)
                n_elems_color = None

                # Detect color_mode for arrays
                color_mode = None
                if np.issubdtype(colors.dtype, np.floating):
                    color_mode = "hdr" if np.any(colors > 1.0) else "sdr"
            else:
                raise ValueError(f"Unsupported colors type: {type(colors)}")

            self._encoder.encode(
                data=colors,
                zarr_group=group,
                name="colors",
                semantic_type=SemanticType.COLOR,
                mode=self._encoding_mode,
                color_mode=color_mode,
                n_elements=n_elems_color,
                chunks=chunks_colors,
                compressor=self.compressor,
            )
            metadata["has_colors"] = True

        if sharpness is not None:
            from ..validation.base import validate_sharpness_for_writing

            # Handle both scalar and array inputs
            if isinstance(sharpness, (int, float)):
                n_elems_sharp = n_vertices
                chunks_sharp = None
            else:
                validate_sharpness_for_writing(sharpness, n_vertices)
                n_elems_sharp = None
                chunks_sharp = _calculate_intelligent_chunks((n_vertices,))

            self._encoder.encode(
                data=sharpness,
                zarr_group=group,
                name="sharpness",
                semantic_type=SemanticType.BOUNDED_SCALAR,
                mode=self._encoding_mode,
                bounds=(0.0, SHARPNESS_MAX),
                n_elements=n_elems_sharp,
                chunks=chunks_sharp,
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
        amplitudes: Union[NDArray[np.float32], float],
        cholesky_factors: NDArray[np.float32],
        colors: Optional[
            Union[NDArray[np.float32], List[float], Tuple[float, ...]]
        ] = None,
        sharpness: Optional[Union[NDArray[np.float32], float]] = None,
        **attrs: Any,
    ) -> dict[str, Any]:
        """Write Gaussian splats data to Zarr.

        Scalar convenience: amplitudes, colors, and sharpness accept scalars:
        - amplitudes=1.0 → all splats get amplitude 1.0
        - colors=[1.0, 0, 0] → all splats red
        - sharpness=2.0 → standard Gaussian for all

        Args:
            path: Path for the gsplats within the store
            centers: Splat centers of shape (N, D)
            amplitudes: Amplitudes - array (N,) or scalar float
            cholesky_factors: Packed Cholesky factors of shape (N, k)
            colors: Colors - array (N, 3), RGB tuple/list, or None
            sharpness: Sharpness - array (N,), scalar float, or None
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

        # Log scalar inputs (no expansion - passed to encoder)
        if isinstance(amplitudes, (int, float)):
            aprint(f"  → Uniform amplitude {amplitudes:.3f} for all splats")
        if sharpness is not None and isinstance(sharpness, (int, float)):
            aprint(f"  → Uniform sharpness {sharpness:.1f} for all splats")
        if colors is not None and isinstance(colors, (list, tuple)):
            aprint(f"  → Uniform color RGB{list(colors)} for all splats")

        # Validate cholesky_factors shape FIRST (before spatial ordering)
        expected_k = n_dims * (n_dims + 1) // 2
        cholesky_is_uniform = False  # Track if cholesky is uniform (for encoder)

        if cholesky_factors.ndim == 1:
            # Shape (k,) - uniform cholesky for all splats
            if cholesky_factors.shape[0] != expected_k:
                raise ValueError(
                    f"Cholesky factors shape mismatch: expected ({expected_k},), "
                    f"got {cholesky_factors.shape}"
                )
            # Reshape to (1, k) for encoder passthrough (no intermediate array)
            cholesky_factors = cholesky_factors.reshape(1, expected_k)
            cholesky_is_uniform = True
            aprint(f"  → Uniform Cholesky factors (shape {expected_k}) for all splats")

        elif cholesky_factors.shape != (n_splats, expected_k):
            raise ValueError(
                f"Cholesky factors shape mismatch: expected ({n_splats}, {expected_k}), "
                f"got {cholesky_factors.shape}"
            )

        # Apply spatial ordering if enabled
        ordering_data = None
        if self.enable_spatial_index and n_splats > 0:
            from .ordering import compute_chunk_bounds_gsplats, sort_splats_spatial

            aprint(f"  🔍 Applying {self.ordering_method} ordering to gsplats...")
            sort_indices, ordering_metadata = sort_splats_spatial(
                centers, method=self.ordering_method
            )

            # Reorder arrays only (skip scalars and uniform cholesky)
            centers = centers[sort_indices]
            if not cholesky_is_uniform:
                cholesky_factors = cholesky_factors[sort_indices]
            # Scalars stay as-is (uniform, order doesn't matter)
            if isinstance(amplitudes, np.ndarray):
                amplitudes = amplitudes[sort_indices]
            if colors is not None and isinstance(colors, np.ndarray):
                colors = colors[sort_indices]
            if sharpness is not None and isinstance(sharpness, np.ndarray):
                sharpness = sharpness[sort_indices]

            # Compute chunk size
            from ..typing_utils import TARGET_CHUNK_BYTES

            bytes_per_splat = n_dims * 4 + 4 + expected_k * 4 + 16
            chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_splat)
            chunk_size = min(chunk_size, n_splats)

            # Compute chunk bounds
            chunk_bounds = compute_chunk_bounds_gsplats(
                centers, cholesky_factors, chunk_size
            )

            ordering_data = {
                "chunk_bounds": chunk_bounds,
                "chunk_size": chunk_size,
                **ordering_metadata,
            }

            aprint(
                f"  ✓ Spatial ordering complete: {ordering_metadata['ordering']} with {len(chunk_bounds)} chunks"
            )

        # Validate amplitudes (must be non-negative: >= 0, zero is valid but invisible)
        if isinstance(amplitudes, np.ndarray):
            if amplitudes.shape[0] != n_splats:
                raise ValueError(
                    f"Amplitudes shape {amplitudes.shape} doesn't match n_splats {n_splats}"
                )
            if np.any(amplitudes < 0):
                min_val = float(np.min(amplitudes))
                raise ValueError(
                    f"Amplitudes must be non-negative (>= 0). Found minimum value: {min_val:.3f}"
                )
        elif isinstance(amplitudes, (int, float)) and amplitudes < 0:
            raise ValueError(f"Amplitude must be non-negative (>= 0). Got {amplitudes}")

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
        if isinstance(amplitudes, (int, float)):
            amplitude_min = amplitude_max = float(amplitudes)
            n_elems_amp = n_splats
            chunks_amp = None
        else:
            amplitude_min, amplitude_max = (
                float(np.min(amplitudes)),
                float(np.max(amplitudes)),
            )
            n_elems_amp = None
            chunks_amp = _calculate_intelligent_chunks((n_splats,))

        self._encoder.encode(
            data=amplitudes,
            zarr_group=group,
            name="amplitudes",
            semantic_type=SemanticType.POSITIVE_SCALAR,
            mode=self._encoding_mode,
            n_elements=n_elems_amp,
            chunks=chunks_amp,
            compressor=self.compressor,
        )

        # Write cholesky_factors using ArrayEncoder (CHOLESKY)
        if cholesky_is_uniform:
            # Pass (1, k) array with n_elements for broadcasting
            n_elems_chol = n_splats
            chunks_cholesky = None
        else:
            n_elems_chol = None
            chunks_cholesky = _calculate_intelligent_chunks(cholesky_factors.shape)

        self._encoder.encode(
            data=cholesky_factors,
            zarr_group=group,
            name="cholesky_factors",
            semantic_type=SemanticType.CHOLESKY,
            mode=self._encoding_mode,
            n_elements=n_elems_chol,
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
            "amplitude_range": {"min": amplitude_min, "max": amplitude_max},
            "center_bounds": {"min": center_min, "max": center_max},
        }

        # Add ordering metadata if spatial ordering was applied
        if ordering_data is not None:
            metadata.update(
                {
                    "ordering": ordering_data["ordering"],
                    "morton_min": ordering_data["morton_min"],
                    "morton_max": ordering_data["morton_max"],
                    "morton_bits_per_dim": ordering_data["morton_bits_per_dim"],
                    "chunk_size": ordering_data["chunk_size"],
                }
            )
        else:
            metadata["ordering"] = "none"

        # Write optional datasets
        if colors is not None:
            # Handle scalar/tuple vs array
            if isinstance(colors, (tuple, list)):
                # Detect HDR vs SDR from values
                max_val = max(colors)
                color_mode = "hdr" if max_val > 1.0 else "sdr"
                n_elems_color = n_splats
                chunks_colors = None
            elif isinstance(colors, np.ndarray):
                validate_colors_for_writing(colors, n_splats)
                chunks_colors = _calculate_intelligent_chunks(colors.shape)
                n_elems_color = None

                # Detect color_mode
                color_mode = None
                if np.issubdtype(colors.dtype, np.floating):
                    color_mode = "hdr" if np.any(colors > 1.0) else "sdr"
            else:
                raise ValueError(f"Unsupported colors type: {type(colors)}")

            self._encoder.encode(
                data=colors,
                zarr_group=group,
                name="colors",
                semantic_type=SemanticType.COLOR,
                mode=self._encoding_mode,
                color_mode=color_mode,
                n_elements=n_elems_color,
                chunks=chunks_colors,
                compressor=self.compressor,
            )
            metadata["has_colors"] = True

        if sharpness is not None:
            from ..validation.base import validate_sharpness_for_writing

            # Handle scalar vs array
            if isinstance(sharpness, (int, float)):
                n_elems_sharp = n_splats
                chunks_sharp = None
            else:
                validate_sharpness_for_writing(sharpness, n_splats)
                n_elems_sharp = None
                chunks_sharp = _calculate_intelligent_chunks((n_splats,))

            self._encoder.encode(
                data=sharpness,
                zarr_group=group,
                name="sharpness",
                semantic_type=SemanticType.BOUNDED_SCALAR,
                mode=self._encoding_mode,
                bounds=(0.0, SHARPNESS_MAX),
                n_elements=n_elems_sharp,
                chunks=chunks_sharp,
                compressor=self.compressor,
            )
            metadata["has_sharpness"] = True

        # Write chunk_bounds if ordering was applied
        if ordering_data is not None:
            chunk_bounds = ordering_data["chunk_bounds"]
            if len(chunk_bounds) > 0:
                group.create_dataset(
                    "chunk_bounds",
                    data=chunk_bounds,
                    chunks=(chunk_bounds.shape[0], n_dims, 2),
                    dtype=np.float32,
                )
                aprint(f"  ✓ Chunk bounds written: {len(chunk_bounds)} chunks")

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

        # Add ordering metadata to attrs if present
        if ordering_data is not None:
            for key in [
                "ordering",
                "morton_min",
                "morton_max",
                "morton_bits_per_dim",
                "chunk_size",
            ]:
                if key in metadata:
                    group.attrs[key] = metadata[key]

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

    def _build_spatial_ordering_if_enabled(
        self,
        positions: NDArray[np.float32],
        n_points: int,
        n_dims: int,
        radii: Optional[NDArray[np.float32]],
    ) -> Optional[Dict[str, Any]]:
        """Apply spatial ordering using Morton/Hilbert curves.

        Args:
            positions: Point positions
            n_points: Number of points
            n_dims: Number of dimensions
            radii: Optional radii array

        Returns:
            Dict with:
            - sorted_positions: Reordered positions
            - sort_order: Indices to apply to other arrays
            - chunk_bounds: (num_chunks, n_dims, 2) array
            - ordering_metadata: Dict from sort_points_compound
            Or None if ordering disabled/not applicable
        """
        if not self.enable_spatial_index or n_points == 0:
            return None

        # Get scene dimensions from attrs
        if "scene_dimensions" not in self.store.attrs:
            aprint("  ⚠️ No scene dimensions - skipping spatial ordering")
            return None

        from ..core.dimensions import Dimensions

        scene_dims_dict = self.store.attrs["scene_dimensions"]
        dimensions = Dimensions.from_dict(scene_dims_dict)

        aprint(f"  🔍 Applying {self.ordering_method} ordering...")

        # Apply compound ordering
        from .ordering import compute_chunk_bounds_points, sort_points_compound

        sort_indices, ordering_metadata = sort_points_compound(
            positions,
            dimensions.dimensions,  # List of Dimension objects
            method=self.ordering_method,
        )

        # Reorder positions
        sorted_positions = positions[sort_indices]

        # Compute chunk size (from TARGET_CHUNK_BYTES)
        from ..typing_utils import TARGET_CHUNK_BYTES

        bytes_per_point = n_dims * 4 + 16  # Conservative estimate
        chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_point)
        chunk_size = min(chunk_size, n_points)

        # Compute chunk bounds
        # Handle scalar radii vs array radii
        if radii is not None:
            if isinstance(radii, np.ndarray):
                sorted_radii = radii[sort_indices]
            else:
                # Scalar radii - no reordering needed
                sorted_radii = np.full(n_points, float(radii), dtype=np.float32)
        else:
            sorted_radii = None
        chunk_bounds = compute_chunk_bounds_points(
            sorted_positions, sorted_radii, chunk_size
        )

        aprint(
            f"  ✓ Ordering complete: {len(ordering_metadata['slice_dims'])} discrete dims, "
            f"{len(ordering_metadata['morton_dims'])} spatial dims"
        )

        return {
            "sorted_positions": sorted_positions,
            "sort_order": sort_indices,
            "chunk_bounds": chunk_bounds,
            "chunk_size": chunk_size,
            **ordering_metadata,  # ordering, slice_dims, morton_dims, etc.
        }

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
        colors: Union[NDArray[np.float32], tuple, list],
        spatial_index_data: Optional[Dict[str, Any]],
    ) -> None:
        """Write colors dataset to Zarr using ArrayEncoder.

        Args:
            group: Zarr group to write to
            colors: Colors array or tuple/list
            spatial_index_data: Optional spatial index for chunk optimization
        """
        # Handle scalar vs array
        if isinstance(colors, (tuple, list)):
            # Detect HDR vs SDR from values
            max_val = max(colors)
            color_mode = "hdr" if max_val > 1.0 else "sdr"
            if color_mode == "hdr":
                aprint("  ✓ Detected HDR colors (values > 1.0)")
            n_elems = group["positions"].shape[0]
            color_chunks = None
        else:
            n_elems = None
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
            n_elements=n_elems,
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
        radii: Union[NDArray[np.float32], float, int],
        spatial_index_data: Optional[Dict[str, Any]],
    ) -> float:
        """Write radii dataset to Zarr using ArrayEncoder.

        Args:
            group: Zarr group to write to
            radii: Radii array or scalar value
            spatial_index_data: Optional spatial index for chunk optimization

        Returns:
            Maximum radius value
        """
        # Handle scalar vs array
        if isinstance(radii, (int, float)):
            max_radius = float(radii)
            n_elems = group["positions"].shape[0]  # Get from positions
            radii_chunks = None
        else:
            max_radius = float(np.max(radii))
            n_elems = None
            radii_chunks = _calculate_intelligent_chunks(radii.shape)

        aprint(f"  ✓ Max radius: {max_radius:.3f}")

        # Use ArrayEncoder for radii (POSITIVE_SCALAR semantic type)
        self._encoder.encode(
            data=radii,
            zarr_group=group,
            name="radii",
            semantic_type=SemanticType.POSITIVE_SCALAR,
            mode=self._encoding_mode,
            n_elements=n_elems,
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
        sharpness: Union[NDArray[np.float32], float, int],
        spatial_index_data: Optional[Dict[str, Any]],
    ) -> None:
        """Write sharpness dataset to Zarr using ArrayEncoder.

        Args:
            group: Zarr group to write to
            sharpness: Sharpness array or scalar value
            spatial_index_data: Optional spatial index for chunk optimization
        """
        # Handle scalar vs array
        if isinstance(sharpness, (int, float)):
            n_elems = group["positions"].shape[0]
            sharp_chunks = None
        else:
            n_elems = None
            sharp_chunks = _calculate_intelligent_chunks(sharpness.shape)

        # Use ArrayEncoder for sharpness (BOUNDED_SCALAR with [0, 31] range)
        self._encoder.encode(
            data=sharpness,
            zarr_group=group,
            name="sharpness",
            semantic_type=SemanticType.BOUNDED_SCALAR,
            mode=self._encoding_mode,
            bounds=(0.0, SHARPNESS_MAX),
            n_elements=n_elems,
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

    def _write_spatial_ordering_to_zarr(
        self, group: zarr.Group, ordering_data: Dict[str, Any]
    ) -> None:
        """Write spatial ordering metadata and chunk bounds to Zarr.

        Args:
            group: Parent Zarr group
            ordering_data: Ordering data with chunk_bounds and metadata
        """
        aprint("  📝 Writing spatial ordering metadata...")

        # Write ordering metadata directly to group attrs (simple, clean)
        ordering_metadata = {
            "ordering": ordering_data["ordering"],
            "slice_dims": ordering_data["slice_dims"],
            "morton_dims": ordering_data["morton_dims"],
            "morton_min": ordering_data["morton_min"],
            "morton_max": ordering_data["morton_max"],
            "morton_bits_per_dim": ordering_data["morton_bits_per_dim"],
            "chunk_size": ordering_data["chunk_size"],
        }
        group.attrs.update(ordering_metadata)

        # Write chunk_bounds array directly to group
        chunk_bounds = ordering_data["chunk_bounds"]
        if len(chunk_bounds) > 0:
            group.create_dataset(
                "chunk_bounds",
                data=chunk_bounds,
                shape=chunk_bounds.shape,
                dtype=np.float32,
                chunks=(chunk_bounds.shape[0], chunk_bounds.shape[1], 2),
                compressor=self.compressor,
            )

        aprint(
            f"  ✓ Spatial ordering written: {ordering_data['ordering']} with {len(chunk_bounds)} chunks"
        )

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
