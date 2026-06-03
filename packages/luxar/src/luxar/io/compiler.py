"""luxar.compiler – Progressive Zarr compiler for memory-efficient scene building.

This module provides the LuxarZarrCompiler class which implements progressive
writing to Zarr stores, enabling handling of arbitrarily large datasets without
memory constraints.
"""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import xxhash

if TYPE_CHECKING:
    from ..core.scene import Scene
    from ..core.viewer_config import ViewerConfig
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
from ..typing_utils.config import DEFAULT_VERSION
from ..typing_utils.constants import SHARPNESS_MAX, TARGET_CHUNK_BYTES
from ..typing_utils.protocols import CompressorProtocol

# Ordering functions will be imported locally where needed to avoid circular imports


def _calculate_intelligent_chunks(
    shape: Tuple[int, ...],
    target_chunk_bytes: int = TARGET_CHUNK_BYTES,
    spatial_index_data: Optional[Dict[str, Any]] = None,
    *,
    dtype: np.dtype = np.dtype(np.float32),
) -> Tuple[int, ...]:
    """Calculate optimal chunk shape for a dataset.

    When spatial ordering data is available, uses chunk_size from ordering.

    The byte-target heuristic depends on dtype itemsize: a uint8 colors
    array of shape (N, 3) yields different optimal chunks than a float32
    positions array of the same shape. CC-1-r: previous versions of this
    helper accepted ``itemsize: int = 4`` which silently under-chunked any
    non-float32 caller that forgot to thread the parameter through. The
    ``dtype=`` form makes the contract explicit at every call site.

    Args:
        shape: Shape of the dataset.
        target_chunk_bytes: Target chunk payload size in bytes.
        spatial_index_data: Optional ordering data (with chunk_size).
        dtype: NumPy dtype of the array being chunked. Defaults to float32
            for backwards compatibility but every call site should pass the
            actual array dtype explicitly.

    Returns:
        Optimized chunk shape.
    """
    element_size = max(1, int(dtype.itemsize))
    target_elements = max(1, int(target_chunk_bytes) // element_size)

    if len(shape) == 1:
        # 1D array - use spatial index chunk_size if available for alignment
        if spatial_index_data and "chunk_size" in spatial_index_data:
            return (min(shape[0], spatial_index_data["chunk_size"]),)
        return (min(shape[0], target_elements),)

    if len(shape) == 2:
        # 2D array (e.g., positions) - chunk along first dimension
        n_points, n_dims = shape

        # If ordering data is available, use its chunk_size
        if spatial_index_data and "chunk_size" in spatial_index_data:
            chunk_points = spatial_index_data["chunk_size"]
            return (chunk_points, n_dims)

        # Fallback to standard byte-based chunking
        chunk_points = min(n_points, max(1, target_elements // n_dims))
        return (chunk_points, n_dims)

    # For higher dimensions, use reasonable byte-based defaults
    return tuple(min(s, target_elements) for s in shape)


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
        >>> dims = Dimensions.default_3d()
        >>> with LuxarZarrCompiler('output.zarr') as compiler:
        ...     scene = compiler.create_scene(dimensions=dims)
        ...     positions = np.random.randn(10000, 3).astype(np.float32)
        ...     scene.add_points('points', positions)

        With HDR colors and custom dimensions:
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

        Large datasets (partitioned into multiple nodes):
        >>> dims = Dimensions.default_3d()
        >>> with LuxarZarrCompiler('huge.zarr', ordering_method="hilbert") as compiler:
        ...     scene = compiler.create_scene(dimensions=dims)
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
        ordering_method: Literal["morton", "hilbert"] = "hilbert",
        float16_allowed: bool = False,
        auto_partition_max_elements: Optional[int] = None,
    ) -> None:
        """Initialize the Zarr compiler.

        Args:
            store_path: Path for the Zarr store, or None for temporary
            compressor: Compressor for datasets
            version: Luxar format version
            enable_spatial_index: Whether to use spatial ordering for points/gsplats (default: True)
            encoding_mode: Encoding mode for array storage (AUTO/PRECISION/MEMORY)
            ordering_method: Spatial ordering method ("morton" or "hilbert", default: "hilbert")
            float16_allowed: Allow float16 encoding in MEMORY mode (default: False for TypeScript compatibility)
            auto_partition_max_elements: If set, ``add_points`` and ``add_gsplats``
                automatically apply ``partition=dict(max_elements=N)`` when the
                input element count exceeds N. User-explicit ``partition=`` at
                the call site always wins. Default ``None`` (opt-in, no
                auto-partition). Useful for large datasets where you want a
                per-part frustum-cull benefit without explicit per-call
                boilerplate.

        Note:
            Physical units should be specified per-dimension using the Dimensions
            system when calling create_scene(dimensions=...). This provides more
            flexibility for multi-dimensional data.
        """
        # Handle store path
        self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None
        if store_path is None:
            self._tmpdir = tempfile.TemporaryDirectory()
            self._store_path = Path(self._tmpdir.name) / "scene.zarr"
            aprint(f"📁 Using temporary directory: {self._store_path}")
        else:
            self._store_path = Path(store_path)
            aprint(f"📁 Creating scene at: {self._store_path}")

        # Store spatial ordering configuration
        self.enable_spatial_index = enable_spatial_index
        self.ordering_method = ordering_method

        # Opt-in auto-partition threshold. Read at add_points / add_gsplats
        # time via resolve_auto_partition(). None disables.
        if (
            auto_partition_max_elements is not None
            and auto_partition_max_elements <= 0
        ):
            raise ValueError(
                "auto_partition_max_elements must be positive; "
                f"got {auto_partition_max_elements}"
            )
        self.auto_partition_max_elements: Optional[int] = auto_partition_max_elements

        # Emit the ACES-vs-LUT tone-mapping warning at most once per compile,
        # the first time a colormap LUT is written (see _write_colormap_lut_if_needed).
        self._lut_tone_mapping_warned: bool = False

        # Create array encoder with specified encoding mode and float16 control
        self._encoder = ArrayEncoder(float16_allowed=float16_allowed)
        self._encoding_mode = encoding_mode
        self._float16_allowed = float16_allowed

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

        # Scene-level bounds tracking (union of all node bounds)
        # Each entry is [min_per_dim, max_per_dim] where each is a list of floats
        self._scene_bounds: Optional[Dict[str, List[float]]] = None

        # Scene reference for finalize-time hover overlay auto-injection
        self._scene: Optional["Scene"] = None

        aprint(f"✅ Zarr compiler initialized at {self._store_path}")

    def __enter__(self) -> LuxarZarrCompiler:
        """Enter context manager."""
        return self

    def _check_not_finalized(self, op: str) -> None:
        """Refuse mutating operations after finalize() has run.

        CL-2: ``finalize()`` writes the consolidated zarr metadata; any
        subsequent ``write_*`` / ``create_*`` call would silently produce a
        store with stale ``.zmetadata`` (the chunk would land but the
        consolidated index would not see it). Failing fast surfaces the
        misuse at the call site rather than as a downstream "missing data"
        symptom.
        """
        if self._is_finalized:
            raise RuntimeError(
                f"Cannot {op} after the writer has been finalized. "
                "All mutating calls must run inside the active "
                "LuxarZarrCompiler context, before context exit or before "
                "Scene.to_zarr() finalizes the store."
            )

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and finalize."""
        if not self._is_finalized:
            self.finalize()

        # Clean up temporary directory if used
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def create_scene(
        self,
        dimensions: Dimensions,
        viewer_config: Optional["ViewerConfig"] = None,
    ) -> "Scene":
        """Create a scene with this compiler as writer.

        Args:
            dimensions: Dimension specification for the scene (REQUIRED).
                Scene dimensions are the single source of truth for the
                coordinate system and must always be specified.
            viewer_config: Optional viewer configuration hints. Stored in
                the zarr file and read by the viewer at load time as
                scene-specific defaults.

        Returns:
            Scene object configured with this compiler as writer

        Raises:
            ValueError: If dimensions is None or invalid
        """
        self._check_not_finalized("create_scene")

        # Import here to avoid circular dependency
        from ..core.scene import Scene

        # Validate dimensions (REQUIRED)
        if dimensions is None:
            raise ValueError(
                "dimensions is required. Scene dimensions define the coordinate system "
                "and are the single source of truth for all data in the scene. "
                "Use Dimensions([...]) to specify the coordinate axes."
            )

        # Store dimensions in root attributes
        self.store.attrs["scene_dimensions"] = dimensions.to_dict()

        # Create scene with writer injection and optional viewer config
        scene = Scene(writer=self, dimensions=dimensions, viewer_config=viewer_config)
        self._scene = scene  # Store reference for finalize-time hover overlay injection
        aprint("✅ Scene created with progressive writer")

        return scene

    def write_group(self, path: NodePath, **attrs: Any) -> None:
        """Create a group in the Zarr store.

        Args:
            path: Path for the group within the store
            **attrs: Attributes to attach to the group
        """
        self._check_not_finalized("write_group")
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

    def delete_group_attr(self, path: NodePath, key: str) -> None:
        """Remove an attribute from a group in the Zarr store.

        Args:
            path: Path for the group within the store
            key: Attribute key to remove
        """
        if path == "/" or path == "":
            group = self.store
        else:
            path = path.lstrip("/")
            try:
                group = self.store[path]
            except KeyError:
                return
            if not isinstance(group, zarr.Group):
                return

        attrs = dict(group.attrs)
        if key in attrs:
            del attrs[key]
            group.attrs.clear()
            group.attrs.update(attrs)

    def write_points(  # type: ignore[override]
        self,
        path: NodePath,
        positions: NDArray[np.float32],
        colors: Optional[
            Union[NDArray[np.float32], List[float], Tuple[float, ...]]
        ] = None,
        radii: Optional[Union[NDArray[np.float32], float]] = None,
        sharpness: Optional[Union[NDArray[np.float32], float]] = None,
        scalars: Optional[Union[NDArray[np.float32], float]] = None,
        labels: Optional["Sequence[str]"] = None,
        image_labels: Optional[Any] = None,
        **attrs: Any,
    ) -> PointsMetadata:
        """Write points data progressively to Zarr.

        Data is written immediately to disk without being kept in memory.
        If spatial ordering is enabled, points are reordered using Morton/Hilbert curves.

        Scalar convenience: radii, sharpness, scalars, and colors accept scalars:
        - radii=0.5 → all points get radius 0.5 (broadcasted)
        - colors=[1.0, 0, 0] → all points red (broadcasted)
        - sharpness=2.0 → all points standard Gaussian (broadcasted)
        - scalars=0.5 → all points get scalar 0.5 (broadcasted)

        Args:
            path: Path for the points within the store
            positions: Point positions of shape (N, D)
            colors: Colors - array (N, 3), RGB tuple/list, or None
            radii: Radii - array (N,), scalar float, or None
            sharpness: Sharpness - array (N,), scalar float, or None
            scalars: Scalars for colormap lookup - array (N,), scalar float, or None
            labels: Optional list of strings, one per point. Stored as CSR-encoded
                label_offsets + label_bytes arrays for hover tooltips.
            image_labels: Optional per-element images for hover thumbnails.
                Accepts List[bytes], List[PIL.Image], List[ndarray], List[Path],
                or Dict[int, Any] for sparse assignment.
            **attrs: Additional attributes

        Returns:
            Metadata dictionary about the written data
        """
        self._check_not_finalized("write_points")

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
        # Note: Spatial ordering uses radii to compute chunk_bounds.
        # Scalar/broadcasted radii are handled without expanding to full arrays.
        radii_for_ordering = radii

        ordering_data = self._build_spatial_ordering_if_enabled(
            positions, n_points, n_dims, radii_for_ordering
        )

        # Apply spatial reordering to arrays only (skip scalars and broadcasted arrays)
        if ordering_data is not None:
            positions = ordering_data["sorted_positions"]
            # Apply sort order only to non-broadcasted array attributes
            # Broadcasted arrays (shape[0] == 1) should NOT be reordered
            if colors is not None and isinstance(colors, np.ndarray):
                if colors.shape[0] > 1:  # Not broadcasted
                    colors = colors[ordering_data["sort_order"]]
                # else: broadcasted, skip reordering
            if radii is not None and isinstance(radii, np.ndarray):
                if radii.shape[0] > 1:  # Not broadcasted
                    radii = radii[ordering_data["sort_order"]]
                # else: broadcasted, skip reordering
            if sharpness is not None and isinstance(sharpness, np.ndarray):
                if sharpness.shape[0] > 1:  # Not broadcasted
                    sharpness = sharpness[ordering_data["sort_order"]]
                # else: broadcasted, skip reordering
            if scalars is not None and isinstance(scalars, np.ndarray):
                if scalars.shape[0] > 1:  # Not broadcasted
                    scalars = scalars[ordering_data["sort_order"]]
                # else: broadcasted, skip reordering

        # 3. Write positions dataset
        self._write_positions_dataset(group, positions, ordering_data)

        # 4. Initialize metadata
        metadata: PointsMetadata = {
            "n_points": n_points,
            "ndim": n_dims,
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
            self._write_colors_dataset(group, colors, ordering_data, n_points)
            metadata["has_colors"] = True

        if radii is not None:
            # Validate arrays only (scalars validated by encoder)
            if isinstance(radii, np.ndarray):
                validate_radii_for_writing(radii, n_points)
            max_radius = self._write_radii_dataset(
                group, radii, ordering_data, n_points
            )
            metadata["max_radius"] = max_radius
            metadata["has_radii"] = True
            group.attrs["max_radius"] = max_radius

        if sharpness is not None:
            # Validate arrays only (scalars validated by encoder)
            if isinstance(sharpness, np.ndarray):
                validate_sharpness_for_writing(sharpness, n_points)
            max_sharpness = self._write_sharpness_dataset(
                group, sharpness, ordering_data, n_points
            )
            metadata["max_sharpness"] = max_sharpness
            metadata["has_sharpness"] = True
            group.attrs["max_sharpness"] = max_sharpness

        if scalars is not None:
            self._write_scalars_dataset(group, scalars, ordering_data, n_points)
            metadata["has_scalars"] = True
            group.attrs["has_scalars"] = True

        # 5b. Write colormap LUT if colormap is a custom array
        self._write_colormap_lut_if_needed(group, attrs)

        # 6. Process transform if present using centralized conversion
        if "transform" in attrs:
            from ..core.transforms import prepare_transform_for_zarr

            attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

        # 6b. Validate nd_transform if present
        if "nd_transform" in attrs:
            from ..validation.nd_transforms import validate_nd_transform

            dims = None
            if "scene_dimensions" in self.store.attrs:
                dims = Dimensions.from_dict(self.store.attrs["scene_dimensions"])
            attrs["nd_transform"] = validate_nd_transform(attrs["nd_transform"], dims)

        # 7. Set default rendering attributes if not provided
        if "opacity" not in attrs:
            attrs["opacity"] = 1.0
        if "gamma" not in attrs:
            attrs["gamma"] = 1.0
        if "intensity" not in attrs:
            attrs["intensity"] = 1.0
        if "offset" not in attrs:
            attrs["offset"] = 0.0
        if "blending_mode" not in attrs:
            attrs["blending_mode"] = "additive"

        # 8. Store attributes
        group.attrs.update(attrs)
        group.attrs["type"] = "points"
        group.attrs["n_points"] = n_points

        # 9. Compute and store position bounds (nD bounding box)
        # This is computed from the final positions (potentially reordered)
        position_bounds = self._compute_position_bounds(positions)
        group.attrs["position_bounds"] = position_bounds
        metadata["position_bounds"] = position_bounds

        # Update scene-level bounds (union of all node bounds). Skipped
        # when ``write_points_multi_lod`` is the caller — the parent
        # multi-LOD writer aggregates the global bounds once instead of
        # accumulating each subgroup's contribution separately.
        if not attrs.pop("_skip_scene_bounds", False):
            self._update_scene_bounds(position_bounds)

        # 10. Write spatial ordering metadata if built
        if ordering_data is not None:
            self._write_spatial_ordering_to_zarr(group, ordering_data)
            metadata["has_spatial_index"] = True

        # 11. Write labels if provided (CSR-style: label_offsets + label_bytes)
        if labels is not None:
            sort_order = (
                ordering_data["sort_order"] if ordering_data is not None else None
            )
            self._write_labels_csr(group, labels, n_points, sort_order)
            metadata["has_labels"] = True

        # 12. Write image labels if provided (CSR-style, no compression on blobs)
        if image_labels is not None:
            sort_order = (
                ordering_data["sort_order"] if ordering_data is not None else None
            )
            self._write_image_labels_csr(group, image_labels, n_points, sort_order)
            metadata["has_image_labels"] = True

        # 13. Cache metadata and finish
        self._metadata_cache[path] = metadata
        aprint(f"✅ Points written to {path}")

        return metadata

    def write_lines(  # type: ignore[override]
        self,
        path: NodePath,
        vertices: NDArray[np.float32],
        widths: Union[NDArray[np.float32], float],
        colors: Optional[
            Union[NDArray[np.float32], List[float], Tuple[float, ...]]
        ] = None,
        sharpness: Optional[Union[NDArray[np.float32], float]] = None,
        scalars: Optional[Union[NDArray[np.float32], float]] = None,
        indices: Optional[NDArray[np.uint32]] = None,
        line_type: str = "polyline",
        labels: Optional["Sequence[str]"] = None,
        image_labels: Optional[Any] = None,
        **attrs: Any,
    ) -> dict[str, Any]:
        """Write lines data to Zarr with dual spatial indexing.

        Lines support dual spatial indexing for efficient lazy loading:
        - Vertices ordered in D-space (like Points)
        - Segments ordered in (2×D)-space (concatenating both endpoints)

        All line types are internally converted to indexed representation.

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
            scalars: Scalars for colormap lookup - array (N,), scalar float, or None
            indices: Optional vertex indices for indexed line type
            line_type: Type of line connectivity
            labels: Optional list of strings, one per vertex. Stored as CSR-encoded
                label_offsets + label_bytes arrays for hover tooltips.
            image_labels: Optional per-element images for hover thumbnails.
            **attrs: Additional attributes

        Returns:
            Metadata dictionary about the written lines
        """
        self._check_not_finalized("write_lines")

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

        # Convert line type to indexed representation (unified internal format)
        from .ordering import convert_to_indexed

        segments = convert_to_indexed(n_vertices, line_type, indices)
        n_segments = segments.shape[0]

        aprint(f"  → Converted {line_type} to {n_segments:,} indexed segments")

        # Get max_width before spatial ordering
        if isinstance(widths, (int, float)):
            max_width = float(widths)
        else:
            max_width = float(np.max(widths))

        # Apply dual spatial ordering if enabled
        ordering_data = self._build_lines_spatial_ordering_if_enabled(
            vertices, segments, widths, n_vertices, n_dims, n_segments
        )

        # Apply reordering if spatial ordering was applied
        if ordering_data is not None:
            vertices = ordering_data["sorted_vertices"]
            segments = ordering_data["sorted_segments"]
            vertex_sort_order = ordering_data["vertex_sort_indices"]

            # Reorder per-vertex arrays (skip scalars and broadcasted)
            if isinstance(widths, np.ndarray) and widths.shape[0] > 1:
                widths = widths[vertex_sort_order]
            if (
                colors is not None
                and isinstance(colors, np.ndarray)
                and colors.shape[0] > 1
            ):
                colors = colors[vertex_sort_order]
            if (
                sharpness is not None
                and isinstance(sharpness, np.ndarray)
                and sharpness.shape[0] > 1
            ):
                sharpness = sharpness[vertex_sort_order]
            if (
                scalars is not None
                and isinstance(scalars, np.ndarray)
                and scalars.shape[0] > 1
            ):
                scalars = scalars[vertex_sort_order]

        # Write vertices using ArrayEncoder (COORDINATE)
        chunks_2d = _calculate_intelligent_chunks(
            (n_vertices, n_dims),
            spatial_index_data=ordering_data.get("vertex_ordering")
            if ordering_data
            else None,
        )
        self._encoder.encode(
            data=vertices,
            zarr_group=group,
            name="vertices",
            semantic_type=SemanticType.COORDINATE,
            mode=self._encoding_mode,
            chunks=chunks_2d,
            compressor=self.compressor,
        )

        # Write segments array (always, not just for indexed type)
        segment_chunk_size = (
            ordering_data["segment_ordering"]["chunk_size"] if ordering_data else 2048
        )
        self._encoder.encode(
            data=segments,
            zarr_group=group,
            name="segments",
            semantic_type=SemanticType.INDEX,
            mode=self._encoding_mode,
            chunks=(segment_chunk_size, 2),
            compressor=self.compressor,
        )
        aprint(f"  ✓ Wrote segments ({n_segments:,} pairs)")

        # Write widths via the canonical POSITIVE_SCALAR helper (shared
        # with Points "radii" and GSplats "amplitudes"). The helper
        # picks the same default precision for every geometry.
        self._write_positive_scalar_dataset(
            group=group,
            data=widths,
            name="widths",
            spatial_index_data=ordering_data.get("vertex_ordering")
            if ordering_data
            else None,
            n_elements=n_vertices,
            log_label_singular="width",
        )

        # Initialize metadata
        metadata: dict[str, Any] = {
            "n_vertices": n_vertices,
            "n_segments": n_segments,
            "ndim": n_dims,
            "original_line_type": line_type,  # Store original user-specified type
            "has_colors": False,
            "has_sharpness": False,
            "max_width": max_width,
        }

        # Write optional datasets
        if colors is not None:
            if isinstance(colors, np.ndarray):
                validate_colors_for_writing(colors, n_vertices)
            # Use the canonical COLOR helper (shared with Points / GSplats)
            # so the default-precision and color_mode-detection logic is
            # symmetric across all three geometry types.
            self._write_colors_dataset(
                group=group,
                colors=colors,
                spatial_index_data=ordering_data.get("vertex_ordering")
                if ordering_data
                else None,
                n_elements=n_vertices,
            )
            metadata["has_colors"] = True

        if sharpness is not None:
            from ..validation.base import validate_sharpness_for_writing

            if isinstance(sharpness, np.ndarray):
                validate_sharpness_for_writing(sharpness, n_vertices)
            # Canonical BOUNDED_SCALAR helper (shared with Points
            # "sharpnesses"). Same bounds tuple as Points so the
            # encoder's Uint8-quantization step produces matching disk
            # layouts.
            self._write_bounded_scalar_dataset(
                group=group,
                data=sharpness,
                name="sharpnesses",
                bounds=(0.0, SHARPNESS_MAX),
                spatial_index_data=ordering_data.get("vertex_ordering")
                if ordering_data
                else None,
                n_elements=n_vertices,
                log_label_singular="sharpness",
            )
            metadata["has_sharpness"] = True

        if scalars is not None:
            self._write_scalars_dataset(group, scalars, ordering_data, n_vertices)
            metadata["has_scalars"] = True
            group.attrs["has_scalars"] = True

        # Write colormap LUT if colormap is a custom array
        self._write_colormap_lut_if_needed(group, attrs)

        # Write spatial ordering data (chunk bounds and metadata)
        if ordering_data is not None:
            self._write_lines_spatial_ordering_to_zarr(group, ordering_data)
            metadata["has_spatial_index"] = True
            metadata["ordering"] = ordering_data["ordering"]
            metadata["vertex_ordering"] = ordering_data["vertex_ordering"]
            metadata["segment_ordering"] = ordering_data["segment_ordering"]
        else:
            metadata["ordering"] = "none"

        # Process transform if present
        if "transform" in attrs:
            from ..core.transforms import prepare_transform_for_zarr

            attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

        # Validate nd_transform if present
        if "nd_transform" in attrs:
            from ..validation.nd_transforms import validate_nd_transform

            dims = None
            if "scene_dimensions" in self.store.attrs:
                dims = Dimensions.from_dict(self.store.attrs["scene_dimensions"])
            attrs["nd_transform"] = validate_nd_transform(attrs["nd_transform"], dims)

        # Set default rendering attributes if not provided (must match write_points/write_gsplats)
        if "opacity" not in attrs:
            attrs["opacity"] = 1.0
        if "gamma" not in attrs:
            attrs["gamma"] = 1.0
        if "intensity" not in attrs:
            attrs["intensity"] = 1.0
        if "offset" not in attrs:
            attrs["offset"] = 0.0
        if "blending_mode" not in attrs:
            attrs["blending_mode"] = "additive"

        # Set attributes (all core metadata per spec Section 6.6)
        group.attrs.update(attrs)
        group.attrs["type"] = "lines"
        group.attrs["n_vertices"] = n_vertices
        group.attrs["n_segments"] = n_segments
        group.attrs["ndim"] = n_dims
        group.attrs["original_line_type"] = line_type
        group.attrs["has_colors"] = metadata["has_colors"]
        group.attrs["has_sharpness"] = metadata["has_sharpness"]
        group.attrs["max_width"] = max_width

        # Add ordering metadata to attrs if present
        if ordering_data is not None:
            group.attrs["ordering"] = ordering_data["ordering"]
            group.attrs["vertex_ordering"] = ordering_data["vertex_ordering"]
            group.attrs["segment_ordering"] = ordering_data["segment_ordering"]
        else:
            group.attrs["ordering"] = "none"

        # Compute and store position bounds (nD bounding box) for dynamic clipping
        position_bounds = self._compute_position_bounds(vertices)
        group.attrs["position_bounds"] = position_bounds
        metadata["position_bounds"] = position_bounds

        # Update scene-level bounds (union of all node bounds). Skipped
        # when ``write_lines_multi_lod`` is the caller — the parent
        # writer aggregates global bounds once.
        if not attrs.pop("_skip_scene_bounds", False):
            self._update_scene_bounds(position_bounds)

        # Write labels if provided (CSR-style: label_offsets + label_bytes)
        # For lines, labels are per-vertex (n_vertices)
        if labels is not None:
            sort_order = (
                ordering_data["vertex_sort_indices"]
                if ordering_data is not None
                else None
            )
            self._write_labels_csr(group, labels, n_vertices, sort_order)
            metadata["has_labels"] = True

        # Write image labels if provided (CSR-style, no compression on blobs)
        if image_labels is not None:
            sort_order = (
                ordering_data["vertex_sort_indices"]
                if ordering_data is not None
                else None
            )
            self._write_image_labels_csr(group, image_labels, n_vertices, sort_order)
            metadata["has_image_labels"] = True

        self._metadata_cache[path] = metadata
        aprint(f"✅ Lines written to {path}")

        return metadata

    # ── GSplats helpers (composable building blocks) ─────────────

    def _validate_gsplat_inputs(
        self,
        centers: NDArray[np.float32],
        amplitudes: Union[NDArray[np.float32], float],
        cholesky_factors: NDArray[np.float32],
        colors: Optional[
            Union[NDArray[np.float32], List[float], Tuple[float, ...]]
        ] = None,
    ) -> Tuple[
        NDArray[np.float32],
        Union[NDArray[np.float32], float],
        NDArray[np.float32],
        Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
        int,
        int,
        bool,
    ]:
        """Validate and normalize gsplat inputs.

        Returns:
            (centers, amplitudes, cholesky_factors, colors,
             n_splats, n_dims, cholesky_is_uniform)
        """
        from ..validation.base import validate_positions_for_writing

        n_splats, n_dims = validate_positions_for_writing(centers)
        expected_k = n_dims * (n_dims + 1) // 2
        cholesky_is_uniform = False

        if cholesky_factors.ndim == 1:
            if cholesky_factors.shape[0] != expected_k:
                raise ValueError(
                    f"Cholesky factors shape mismatch: expected ({expected_k},), "
                    f"got {cholesky_factors.shape}"
                )
            cholesky_factors = cholesky_factors.reshape(1, expected_k)
            cholesky_is_uniform = True
        elif cholesky_factors.shape != (n_splats, expected_k):
            raise ValueError(
                f"Cholesky factors shape mismatch: expected ({n_splats}, {expected_k}), "
                f"got {cholesky_factors.shape}"
            )

        # Validate amplitudes
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

        return (
            centers,
            amplitudes,
            cholesky_factors,
            colors,
            n_splats,
            n_dims,
            cholesky_is_uniform,
        )

    def _apply_gsplat_spatial_ordering(
        self,
        centers: NDArray[np.float32],
        amplitudes: Union[NDArray[np.float32], float],
        cholesky_factors: NDArray[np.float32],
        colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
        n_splats: int,
        n_dims: int,
        cholesky_is_uniform: bool,
        coverage_sigma: float = 3.0,
    ) -> Tuple[
        NDArray[np.float32],
        Union[NDArray[np.float32], float],
        NDArray[np.float32],
        Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
        Optional[Dict[str, Any]],
    ]:
        """Apply spatial ordering to gsplat arrays.

        Returns:
            (centers, amplitudes, cholesky_factors, colors, ordering_data)
            where ordering_data is None if ordering was not applied.
        """
        ordering_data = None
        if self.enable_spatial_index and n_splats > 0:
            from .ordering import compute_chunk_bounds_gsplats, sort_splats_spatial

            aprint(f"  🔍 Applying {self.ordering_method} ordering to gsplats...")
            sort_indices, ordering_metadata = sort_splats_spatial(
                centers, method=self.ordering_method
            )

            centers = centers[sort_indices]
            if not cholesky_is_uniform:
                cholesky_factors = cholesky_factors[sort_indices]
            if isinstance(amplitudes, np.ndarray):
                amplitudes = amplitudes[sort_indices]
            if colors is not None and isinstance(colors, np.ndarray):
                if colors.shape[0] > 1:
                    colors = colors[sort_indices]

            from ..typing_utils import TARGET_CHUNK_BYTES

            expected_k = n_dims * (n_dims + 1) // 2
            bytes_per_splat = n_dims * 4 + 4 + expected_k * 4 + 16
            chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_splat)
            chunk_size = min(chunk_size, n_splats)

            chunk_bounds = compute_chunk_bounds_gsplats(
                centers,
                cholesky_factors,
                chunk_size,
                coverage_sigma=coverage_sigma,
            )

            ordering_data = {
                "sort_order": sort_indices,
                "chunk_bounds": chunk_bounds,
                "chunk_size": chunk_size,
                **ordering_metadata,
            }

            aprint(
                f"  ✓ Spatial ordering complete: {ordering_metadata['ordering']} "
                f"with {len(chunk_bounds)} chunks"
            )

        return centers, amplitudes, cholesky_factors, colors, ordering_data

    def _write_gsplat_arrays(
        self,
        group: zarr.Group,
        centers: NDArray[np.float32],
        amplitudes: Union[NDArray[np.float32], float],
        cholesky_factors: NDArray[np.float32],
        colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
        n_splats: int,
        n_dims: int,
        cholesky_is_uniform: bool,
        ordering_data: Optional[Dict[str, Any]],
    ) -> dict[str, Any]:
        """Write gsplat arrays to a zarr group and return metadata.

        This is the core array-writing routine used by both single-LOD
        and multi-LOD writers.

        Returns:
            Metadata dict with n_splats, ndim, has_colors, amplitude_range,
            center_bounds, and ordering info.
        """
        from ..validation.base import validate_colors_for_writing

        # Write centers
        chunks_centers = _calculate_intelligent_chunks(
            centers.shape, spatial_index_data=ordering_data
        )
        self._encoder.encode(
            data=centers,
            zarr_group=group,
            name="centers",
            semantic_type=SemanticType.COORDINATE,
            mode=self._encoding_mode,
            chunks=chunks_centers,
            compressor=self.compressor,
        )

        # Compute amplitude range up-front for the layer-control
        # metadata block below. The helper recomputes max internally;
        # we surface min here because the GSplat metadata dict needs
        # both bounds.
        if isinstance(amplitudes, (int, float)):
            amplitude_min = amplitude_max = float(amplitudes)
        else:
            amplitude_min, amplitude_max = (
                float(np.min(amplitudes)),
                float(np.max(amplitudes)),
            )

        # Canonical POSITIVE_SCALAR helper — shared with Points "radii"
        # and Lines "widths" so the default-precision selection is
        # symmetric across all three geometries.
        self._write_positive_scalar_dataset(
            group=group,
            data=amplitudes,
            name="amplitudes",
            spatial_index_data=ordering_data,
            n_elements=n_splats,
            log_label_singular="amplitude",
        )

        if isinstance(amplitudes, np.ndarray) and amplitudes.size > 0:
            group.attrs["amplitude_data_range"] = [
                float(amplitudes.min()),
                float(amplitudes.max()),
            ]

        # Write cholesky_factors
        if cholesky_is_uniform:
            n_elems_chol = n_splats
            chunks_cholesky = None
        else:
            n_elems_chol = None
            chunks_cholesky = _calculate_intelligent_chunks(
                cholesky_factors.shape, spatial_index_data=ordering_data
            )

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

        # Compute metadata
        if n_splats > 0:
            center_min = centers.min(axis=0).tolist()
            center_max = centers.max(axis=0).tolist()
        else:
            center_min = [0.0] * n_dims
            center_max = [0.0] * n_dims

        metadata: dict[str, Any] = {
            "n_splats": n_splats,
            "ndim": n_dims,
            "has_colors": False,
            "amplitude_range": {"min": amplitude_min, "max": amplitude_max},
            "center_bounds": {"min": center_min, "max": center_max},
        }

        if ordering_data is not None:
            metadata.update(
                {
                    "ordering": ordering_data["ordering"],
                    "ordering_min": ordering_data["ordering_min"],
                    "ordering_max": ordering_data["ordering_max"],
                    "ordering_bits_per_dim": ordering_data["ordering_bits_per_dim"],
                    "chunk_size": ordering_data["chunk_size"],
                }
            )
        else:
            metadata["ordering"] = "none"

        # Write colors via the canonical COLOR helper (shared with
        # Points + Lines). The helper handles the tuple/list/ndarray
        # branch, picks the right `color_mode`, and writes
        # `color_data_range` attrs identically across geometries.
        if colors is not None:
            if isinstance(colors, np.ndarray):
                validate_colors_for_writing(colors, n_splats)
            self._write_colors_dataset(
                group=group,
                colors=colors,
                spatial_index_data=ordering_data,
                n_elements=n_splats,
            )
            metadata["has_colors"] = True

        # Write chunk_bounds
        if ordering_data is not None:
            chunk_bounds = ordering_data["chunk_bounds"]
            if len(chunk_bounds) > 0:
                group.create_dataset(
                    "chunk_bounds",
                    data=chunk_bounds,
                    chunks=(chunk_bounds.shape[0], n_dims, 2),
                    dtype=np.float32,
                    compressor=self.compressor,
                    overwrite=True,
                )
                aprint(f"  ✓ Chunk bounds written: {len(chunk_bounds)} chunks")

        return metadata

    def _apply_gsplat_group_attrs(
        self,
        group: zarr.Group,
        metadata: dict[str, Any],
        attrs: dict[str, Any],
    ) -> None:
        """Set standard gsplats group attributes and rendering defaults.

        Mutates both group.attrs and attrs dict in-place.
        """
        # Default colormap if no colors and no colormap
        if not metadata.get("has_colors") and "colormap" not in attrs:
            attrs["colormap"] = "gray"

        # Write colormap LUT if colormap is a custom array
        self._write_colormap_lut_if_needed(group, attrs)

        # Process transform if present
        if "transform" in attrs:
            from ..core.transforms import prepare_transform_for_zarr

            attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

        # Validate nd_transform if present
        if "nd_transform" in attrs:
            from ..validation.nd_transforms import validate_nd_transform

            dims = None
            if "scene_dimensions" in self.store.attrs:
                dims = Dimensions.from_dict(self.store.attrs["scene_dimensions"])
            attrs["nd_transform"] = validate_nd_transform(attrs["nd_transform"], dims)

        # Set rendering defaults
        for key, default in [
            ("opacity", 1.0),
            ("gamma", 1.0),
            ("intensity", 1.0),
            ("offset", 0.0),
            ("blending_mode", "additive"),
            ("truncation_radius", 3.0),
        ]:
            if key not in attrs:
                attrs[key] = default

        # Write all attrs, then override with authoritative metadata
        group.attrs.update(attrs)
        group.attrs["type"] = "gsplats"
        group.attrs["n_splats"] = metadata["n_splats"]
        group.attrs["ndim"] = metadata["ndim"]
        group.attrs["has_colors"] = metadata["has_colors"]
        group.attrs["amplitude_range"] = metadata["amplitude_range"]
        group.attrs["center_bounds"] = metadata["center_bounds"]
        group.attrs["ordering"] = metadata["ordering"]

        if metadata["ordering"] != "none":
            for key in [
                "ordering_min",
                "ordering_max",
                "ordering_bits_per_dim",
                "chunk_size",
            ]:
                if key in metadata:
                    group.attrs[key] = metadata[key]
        else:
            default_chunk_size = min(1024, max(64, metadata["n_splats"]))
            group.attrs["chunk_size"] = default_chunk_size

        # Position bounds
        center_bounds = metadata["center_bounds"]
        position_bounds = {"min": center_bounds["min"], "max": center_bounds["max"]}
        group.attrs["position_bounds"] = position_bounds
        metadata["position_bounds"] = position_bounds

    # ── Multi-additive-LOD write helpers for Points and Lines ──

    def write_points_multi_lod(
        self,
        path: NodePath,
        levels: List[Dict[str, Any]],
        *,
        method: str = "random",
        grid_shape: Optional[Tuple[int, ...]] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        **attrs: Any,
    ) -> Dict[str, Any]:
        """Write multi-additive-LOD Points: parent node + ``additive_<i>/`` subgroups.

        Each ``additive_<i>/`` subgroup is a fully-formed Points node
        (written via :meth:`write_points`) carrying that level's data
        arrays + its own spatial index. The parent group carries the
        global ``position_bounds``, ``n_additive_sublods``, the
        compositing attrs (``opacity`` / ``gamma`` / ``colormap`` /
        ``blending_mode`` / ``transform`` / ``nd_transform`` / ``layer``
        / ``visible``) — these inherit down to subgroups via the
        viewer's scene-graph composition at render time.

        Args:
            path: Path for the points node within the store.
            levels: List of per-level dicts with keys ``positions`` /
                ``colors`` / ``radii`` / ``sharpness`` / ``scalars`` /
                ``labels``. ``positions`` is required; others may be
                ``None``.
            method: Ordering method used (``random`` / ``salience`` /
                ``spatial-uniform``). Recorded as an attr on the parent
                node.
            grid_shape: Forwarded to each per-level ``write_points``.
            extend_to_all: Forwarded to each per-level write.
            **attrs: Additional parent-node attrs (compositing,
                colormap, etc.).

        Returns:
            Aggregate metadata dict with ``type`` / ``n_points`` /
            ``n_additive_sublods`` / ``position_bounds`` / ``levels``.
        """
        self._check_not_finalized("write_points_multi_lod")

        if not levels:
            raise ValueError("levels must contain at least one LOD level")

        path = path.lstrip("/")
        group = self.store.require_group(path)
        n_levels = len(levels)

        # Compute global bounds + total count from all levels' positions.
        all_positions = np.concatenate([L["positions"] for L in levels], axis=0)
        n_points_total = int(all_positions.shape[0])
        global_bounds = self._compute_position_bounds(all_positions)

        # Per-level writes. Skip scene-bounds update so we aggregate
        # once at the parent. Each subgroup gets its own attrs from
        # the writer's defaults (opacity=1.0, etc.) — the parent's
        # compositing wins via scene-graph composition.
        level_metas: List[Dict[str, Any]] = []
        for i, lvl in enumerate(levels):
            level_path = f"{path}/additive_{i}"
            level_meta = self.write_points(
                level_path,
                lvl["positions"],
                colors=lvl.get("colors"),
                radii=lvl.get("radii"),
                sharpness=lvl.get("sharpness"),
                scalars=lvl.get("scalars"),
                labels=lvl.get("labels"),
                grid_shape=grid_shape,
                **({"extend_to_all": extend_to_all} if extend_to_all else {}),
                _skip_scene_bounds=True,
            )
            level_metas.append(level_meta)

        # Parent-node attrs. Persist after subgroup writes so they
        # don't get clobbered by side effects.
        group.attrs.update(attrs)
        group.attrs["type"] = "points"
        group.attrs["n_points"] = n_points_total
        group.attrs["n_additive_sublods"] = n_levels
        group.attrs["position_bounds"] = global_bounds
        group.attrs["additive_lod_method"] = method
        if extend_to_all:
            group.attrs["extend_to_all"] = extend_to_all

        # Aggregate the parent's bbox into scene-bounds once.
        self._update_scene_bounds(global_bounds)

        metadata: Dict[str, Any] = {
            "type": "points",
            "n_points": n_points_total,
            "n_additive_sublods": n_levels,
            "position_bounds": global_bounds,
            "levels": level_metas,
        }
        self._metadata_cache[path] = metadata
        aprint(
            f"✅ Multi-LOD Points written to {path} ({n_levels} levels, "
            f"{n_points_total:,} points total)"
        )
        return metadata

    def write_lines_multi_lod(
        self,
        path: NodePath,
        levels: List[Dict[str, Any]],
        *,
        method: str = "random",
        extend_to_all: Optional[Union[List[str], str]] = None,
        **attrs: Any,
    ) -> Dict[str, Any]:
        """Write multi-additive-LOD Lines.

        Mirrors :meth:`write_points_multi_lod`. Each level dict carries
        ``vertices`` + ``widths`` + ``colors`` / ``sharpness`` /
        ``scalars`` / ``labels`` + ``segments`` (local index pairs into
        that level's vertices) + ``n_polylines``. Each subgroup is
        written via :meth:`write_lines` with ``line_type='indexed'``
        and the local segment indices.
        """
        self._check_not_finalized("write_lines_multi_lod")

        if not levels:
            raise ValueError("levels must contain at least one LOD level")

        path = path.lstrip("/")
        group = self.store.require_group(path)
        n_levels = len(levels)

        all_vertices = np.concatenate([L["vertices"] for L in levels], axis=0)
        n_vertices_total = int(all_vertices.shape[0])
        n_polylines_total = sum(int(L.get("n_polylines", 0)) for L in levels)
        global_bounds = self._compute_position_bounds(all_vertices)

        level_metas: List[Dict[str, Any]] = []
        for i, lvl in enumerate(levels):
            level_path = f"{path}/additive_{i}"
            # write_lines expects a flat ``indices`` array (length 2M)
            # for ``line_type='indexed'``; our local segments are (M, 2).
            segments_arr = lvl.get("segments")
            flat_indices = (
                np.asarray(segments_arr, dtype=np.uint32).reshape(-1)
                if segments_arr is not None and len(np.asarray(segments_arr)) > 0
                else None
            )
            level_meta = self.write_lines(
                level_path,
                lvl["vertices"],
                widths=cast(Any, lvl.get("widths")),
                colors=lvl.get("colors"),
                sharpness=lvl.get("sharpness"),
                scalars=lvl.get("scalars"),
                labels=lvl.get("labels"),
                image_labels=None,
                indices=flat_indices,
                line_type="indexed" if flat_indices is not None else "polyline",
                **({"extend_to_all": extend_to_all} if extend_to_all else {}),
                _skip_scene_bounds=True,
            )
            level_metas.append(level_meta)

        # Sum per-level segment counts so the parent advertises a segment
        # total — parity with flat `write_lines` (which writes `n_segments`),
        # progressive points (`n_points`), and progressive gsplats
        # (`n_splats`). Without this the viewer's scene-graph converter reads
        # `n_segments` as undefined → "0% of 0 total" in the data monitor.
        n_segments_total = sum(int(m.get("n_segments", 0)) for m in level_metas)

        group.attrs.update(attrs)
        group.attrs["type"] = "lines"
        group.attrs["n_vertices"] = n_vertices_total
        group.attrs["n_segments"] = n_segments_total
        group.attrs["n_polylines"] = n_polylines_total
        group.attrs["n_additive_sublods"] = n_levels
        group.attrs["position_bounds"] = global_bounds
        group.attrs["additive_lod_method"] = method
        if extend_to_all:
            group.attrs["extend_to_all"] = extend_to_all

        self._update_scene_bounds(global_bounds)

        metadata: Dict[str, Any] = {
            "type": "lines",
            "n_vertices": n_vertices_total,
            "n_segments": n_segments_total,
            "n_polylines": n_polylines_total,
            "n_additive_sublods": n_levels,
            "position_bounds": global_bounds,
            "levels": level_metas,
        }
        self._metadata_cache[path] = metadata
        aprint(
            f"✅ Multi-LOD Lines written to {path} ({n_levels} levels, "
            f"{n_vertices_total:,} vertices in {n_polylines_total:,} "
            f"polylines)"
        )
        return metadata

    # ── GSplats public write methods ───────────────────────────

    def write_gsplats(  # type: ignore[override]
        self,
        path: NodePath,
        centers: NDArray[np.float32],
        amplitudes: Union[NDArray[np.float32], float],
        cholesky_factors: NDArray[np.float32],
        colors: Optional[
            Union[NDArray[np.float32], List[float], Tuple[float, ...]]
        ] = None,
        labels: Optional["Sequence[str]"] = None,
        image_labels: Optional[Any] = None,
        **attrs: Any,
    ) -> dict[str, Any]:
        """Write Gaussian splats data to Zarr (single-LOD, flat layout).

        Scalar convenience: amplitudes and colors accept scalars:
        - amplitudes=1.0 → all splats get amplitude 1.0
        - colors=[1.0, 0, 0] → all splats red

        Args:
            path: Path for the gsplats within the store
            centers: Splat centers of shape (N, D)
            amplitudes: Amplitudes - array (N,) or scalar float
            cholesky_factors: Packed Cholesky factors of shape (N, k)
            colors: Colors - array (N, 3), RGB tuple/list, or None
            labels: Optional list of strings, one per splat. Stored as CSR-encoded
                label_offsets + label_bytes arrays for hover tooltips.
            image_labels: Optional per-element images for hover thumbnails.
            **attrs: Additional attributes

        Returns:
            Metadata dictionary about the written gsplats
        """
        self._check_not_finalized("write_gsplats")

        path = path.lstrip("/")
        group = self.store.require_group(path)

        # Validate
        (
            centers,
            amplitudes,
            cholesky_factors,
            colors,
            n_splats,
            n_dims,
            cholesky_is_uniform,
        ) = self._validate_gsplat_inputs(centers, amplitudes, cholesky_factors, colors)

        # Extract truncation_radius for spatial ordering (default 3.0)
        truncation_radius = float(attrs.get("truncation_radius", 3.0))

        aprint(f"📝 Writing {n_splats:,} gsplats ({n_dims}D) to {path}")
        if isinstance(amplitudes, (int, float)):
            aprint(f"  → Uniform amplitude {amplitudes:.3f} for all splats")
        if colors is not None and isinstance(colors, (list, tuple)):
            aprint(f"  → Uniform color RGB{list(colors)} for all splats")
        if cholesky_is_uniform:
            aprint(
                f"  → Uniform Cholesky factors (shape {n_dims * (n_dims + 1) // 2}) "
                f"for all splats"
            )

        # Spatial ordering
        (
            centers,
            amplitudes,
            cholesky_factors,
            colors,
            ordering_data,
        ) = self._apply_gsplat_spatial_ordering(
            centers,
            amplitudes,
            cholesky_factors,
            colors,
            n_splats,
            n_dims,
            cholesky_is_uniform,
            coverage_sigma=truncation_radius,
        )

        # Write arrays
        metadata = self._write_gsplat_arrays(
            group,
            centers,
            amplitudes,
            cholesky_factors,
            colors,
            n_splats,
            n_dims,
            cholesky_is_uniform,
            ordering_data,
        )

        # Set group attrs
        self._apply_gsplat_group_attrs(group, metadata, attrs)

        # Update scene-level bounds
        self._update_scene_bounds(metadata["position_bounds"])

        # Write labels if provided (CSR-style: label_offsets + label_bytes)
        if labels is not None:
            sort_order = (
                ordering_data["sort_order"] if ordering_data is not None else None
            )
            self._write_labels_csr(group, labels, n_splats, sort_order)
            metadata["has_labels"] = True

        # Write image labels if provided (CSR-style, no compression on blobs)
        if image_labels is not None:
            sort_order = (
                ordering_data["sort_order"] if ordering_data is not None else None
            )
            self._write_image_labels_csr(group, image_labels, n_splats, sort_order)
            metadata["has_image_labels"] = True

        self._metadata_cache[path] = metadata
        aprint(f"✅ GSplats written to {path}")

        return metadata

    def write_gsplats_multi_lod(
        self,
        path: NodePath,
        lods: list[
            Tuple[
                NDArray[np.float32],
                Union[NDArray[np.float32], float],
                NDArray[np.float32],
                Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
            ]
        ],
        lod_stats: Optional[list[dict[str, Any]]] = None,
        **attrs: Any,
    ) -> dict[str, Any]:
        """Write multi-additive-LOD Gaussian splats to Zarr.

        Writes additive sub-LOD subgroups directly under the gsplats
        node path:

            <path>/additive_<i>/{centers, amplitudes, ...}

        with ``n_additive_sublods=<n>`` on the splats group attrs. The
        viewer streams these progressively (prefix-sum LODs).

        Substitutive LODs (alternatives between which the viewer picks
        one) are not represented in the scene format; assemble those at
        scene-build time via the ``lod_group`` node type.

        Args:
            path: Path for the gsplats node within the store
            lods: List of (centers, amplitudes, cholesky_factors, colors)
                tuples, one per additive sub-LOD (index 0 = coarsest).
            lod_stats: Optional per-additive-sub-LOD statistics dicts.
            **attrs: Additional node attributes (opacity, blending_mode, etc.)

        Returns:
            Metadata dictionary about the written gsplats (aggregate).
        """
        self._check_not_finalized("write_gsplats_multi_lod")

        if not lods:
            raise ValueError("lods must contain at least one LOD")

        path = path.lstrip("/")
        group = self.store.require_group(path)
        n_lods = len(lods)

        # Extract truncation_radius for spatial ordering (default 3.0)
        truncation_radius = float(attrs.get("truncation_radius", 3.0))

        # Validate all LODs and collect metadata
        total_splats = 0
        all_center_mins: list[list[float]] = []
        all_center_maxs: list[list[float]] = []
        all_amp_mins: list[float] = []
        all_amp_maxs: list[float] = []
        has_any_colors = False
        n_dims: Optional[int] = None

        for i, (ctr, amp, chol, col) in enumerate(lods):
            (
                ctr,
                amp,
                chol,
                col,
                ns,
                nd,
                chol_uniform,
            ) = self._validate_gsplat_inputs(ctr, amp, chol, col)

            if n_dims is None:
                n_dims = nd
            elif nd != n_dims:
                raise ValueError(f"LOD {i} has {nd}D data but LOD 0 has {n_dims}D")

            lod_group = group.require_group(f"additive_{i}")
            aprint(f"📝 Writing additive sub-LOD {i}: {ns:,} gsplats ({nd}D)")

            (
                ctr,
                amp,
                chol,
                col,
                ordering_data,
            ) = self._apply_gsplat_spatial_ordering(
                ctr,
                amp,
                chol,
                col,
                ns,
                nd,
                chol_uniform,
                coverage_sigma=truncation_radius,
            )

            lod_meta = self._write_gsplat_arrays(
                lod_group,
                ctr,
                amp,
                chol,
                col,
                ns,
                nd,
                chol_uniform,
                ordering_data,
            )

            # Write per-LOD group attrs (lightweight — no rendering defaults)
            lod_group.attrs["type"] = "gsplats"
            lod_group.attrs["n_splats"] = ns
            lod_group.attrs["ndim"] = nd
            lod_group.attrs["has_colors"] = lod_meta["has_colors"]
            lod_group.attrs["amplitude_range"] = lod_meta["amplitude_range"]
            lod_group.attrs["center_bounds"] = lod_meta["center_bounds"]
            lod_group.attrs["ordering"] = lod_meta["ordering"]
            if lod_meta["ordering"] != "none":
                for key in [
                    "ordering_min",
                    "ordering_max",
                    "ordering_bits_per_dim",
                    "chunk_size",
                ]:
                    if key in lod_meta:
                        lod_group.attrs[key] = lod_meta[key]
            else:
                lod_group.attrs["chunk_size"] = min(1024, max(64, ns))

            if lod_stats and i < len(lod_stats):
                lod_group.attrs["lod_stats"] = lod_stats[i]

            # Accumulate aggregate info
            total_splats += ns
            if lod_meta["has_colors"]:
                has_any_colors = True
            bounds = lod_meta["center_bounds"]
            all_center_mins.append(bounds["min"])
            all_center_maxs.append(bounds["max"])
            amp_range = lod_meta["amplitude_range"]
            all_amp_mins.append(amp_range["min"])
            all_amp_maxs.append(amp_range["max"])

        assert n_dims is not None  # guaranteed by non-empty lods

        # Compute aggregate bounds
        if all_center_mins:
            agg_min = [min(m[d] for m in all_center_mins) for d in range(n_dims)]
            agg_max = [max(m[d] for m in all_center_maxs) for d in range(n_dims)]
        else:
            agg_min = [0.0] * n_dims
            agg_max = [0.0] * n_dims

        metadata: dict[str, Any] = {
            "n_splats": total_splats,
            "ndim": n_dims,
            "has_colors": has_any_colors,
            "amplitude_range": {"min": min(all_amp_mins), "max": max(all_amp_maxs)},
            "center_bounds": {"min": agg_min, "max": agg_max},
            "ordering": "none",  # aggregate has no single ordering
            "n_additive_sublods": n_lods,
        }

        # Apply group attrs (rendering defaults, transforms, etc.)
        self._apply_gsplat_group_attrs(group, metadata, attrs)

        group.attrs["n_additive_sublods"] = n_lods

        # Update scene-level bounds
        self._update_scene_bounds(metadata["position_bounds"])

        self._metadata_cache[path] = metadata
        aprint(
            f"✅ Multi-LOD GSplats written to {path} ({n_lods} additive sub-LODs, "
            f"{total_splats:,} total)"
        )

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
        self._check_not_finalized("create_resizable_dataset")

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
        radii: Optional[Union[NDArray[np.float32], float]],
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
        # Handle scalar radii vs array radii vs broadcasted radii
        if radii is not None:
            if isinstance(radii, np.ndarray):
                # Check if radii are broadcasted (shape (1,) or (1, k))
                if radii.shape[0] == 1:
                    # Broadcasted radii - keep scalar to avoid large allocations
                    sorted_radii = float(radii.flat[0])
                else:
                    # Regular array radii - apply reordering
                    sorted_radii = radii[sort_indices]
            else:
                # Scalar radii - no reordering needed
                sorted_radii = float(radii)
        else:
            sorted_radii = None
        chunk_bounds = compute_chunk_bounds_points(
            sorted_positions,
            sorted_radii,
            chunk_size,
            slice_dims=ordering_metadata["slice_dims"],
        )

        aprint(
            f"  ✓ Ordering complete: {len(ordering_metadata['slice_dims'])} discrete dims, "
            f"{len(ordering_metadata['ordering_dims'])} spatial dims"
        )

        return {
            "sorted_positions": sorted_positions,
            "sort_order": sort_indices,
            "chunk_bounds": chunk_bounds,
            "chunk_size": chunk_size,
            **ordering_metadata,  # ordering, slice_dims, ordering_dims, etc.
        }

    def _compute_position_bounds(
        self, positions: NDArray[np.float32]
    ) -> Dict[str, List[float]]:
        """Compute nD bounding box from positions array.

        Args:
            positions: Positions array of shape (N, D)

        Returns:
            Dictionary with 'min' and 'max' keys, each containing a list of D floats
        """
        # Compute min and max along each dimension
        if positions.shape[0] == 0:
            n_dims = positions.shape[1] if positions.ndim == 2 else 0
            return {"min": [0.0] * n_dims, "max": [0.0] * n_dims}
        min_vals = positions.min(axis=0).tolist()
        max_vals = positions.max(axis=0).tolist()

        return {"min": min_vals, "max": max_vals}

    def _update_scene_bounds(self, node_bounds: Dict[str, List[float]]) -> None:
        """Update scene-level bounds by taking union with node bounds.

        Args:
            node_bounds: Dictionary with 'min' and 'max' keys from a node
        """
        if self._scene_bounds is None:
            # First node - initialize scene bounds
            self._scene_bounds = {
                "min": list(node_bounds["min"]),
                "max": list(node_bounds["max"]),
            }
        else:
            # Expand scene bounds to include this node
            # Handle potentially different dimensionalities by extending with the node's values
            node_ndim = len(node_bounds["min"])
            scene_ndim = len(self._scene_bounds["min"])

            if node_ndim > scene_ndim:
                # Extend scene bounds with new dimensions from this node
                self._scene_bounds["min"].extend(node_bounds["min"][scene_ndim:])
                self._scene_bounds["max"].extend(node_bounds["max"][scene_ndim:])
                scene_ndim = node_ndim

            # Update min/max for each dimension
            for i in range(min(node_ndim, scene_ndim)):
                self._scene_bounds["min"][i] = min(
                    self._scene_bounds["min"][i], node_bounds["min"][i]
                )
                self._scene_bounds["max"][i] = max(
                    self._scene_bounds["max"][i], node_bounds["max"][i]
                )

    def _validate_discrete_dimension_ranges(self, store: zarr.Group) -> None:
        """Validate that discrete dimension ranges align with actual data.

        For discrete dimensions (like time frames), the declared range should
        correspond to actual data positions. If range.min < data_min or
        range.max > data_max, the viewer may initialize to a position with no data.

        This validation helps catch cases where:
        - Range starts at 0 but data starts at frame 1
        - Range extends beyond actual data extent

        Args:
            store: The opened zarr store to read dimensions from
        """
        # Check if we have the necessary data
        if self._scene_bounds is None:
            return
        if "scene_dimensions" not in store.attrs:
            return

        # Read dimensions from the store
        from ..core.dimensions import Dimensions

        scene_dims_dict = store.attrs["scene_dimensions"]
        dimensions = Dimensions.from_dict(scene_dims_dict)
        dims = dimensions.dimensions
        ndim = len(dims)

        # Only check dimensions that we have bounds for
        bounds_ndim = len(self._scene_bounds["min"])
        check_ndim = min(ndim, bounds_ndim)

        for i in range(check_ndim):
            dim = dims[i]

            # Only validate discrete, non-displayed dimensions with defined ranges
            if not dim.discrete or dim.display or dim.range is None:
                continue

            declared_min, declared_max = dim.range
            data_min = self._scene_bounds["min"][i]
            data_max = self._scene_bounds["max"][i]

            # Check for range/data misalignment
            tolerance = (dim.step / 2) if dim.step else 0.5

            if declared_min < data_min - tolerance:
                warnings.warn(
                    f"Dimension '{dim.name}' has range starting at {declared_min}, "
                    f"but actual data starts at {data_min:.4f}. "
                    f"The viewer will initialize at {declared_min} where no data exists. "
                    f"Consider setting range=({data_min}, {declared_max}) to match data extent.",
                    UserWarning,
                    stacklevel=3,
                )

            if declared_max > data_max + tolerance:
                warnings.warn(
                    f"Dimension '{dim.name}' has range ending at {declared_max}, "
                    f"but actual data ends at {data_max:.4f}. "
                    f"Navigation beyond {data_max} will show no data. "
                    f"Consider setting range=({declared_min}, {data_max}) to match data extent.",
                    UserWarning,
                    stacklevel=3,
                )

    def _finalize_lod_position_bounds(self, store: zarr.Group) -> None:
        """Back-fill missing ``position_bounds`` on kind=lod groups.

        Walks the zarr tree post-order and, for every group whose attrs
        declare ``kind == 'lod'`` without a ``position_bounds``, computes
        the union of its children's ``position_bounds`` (recursing into
        nested ``kind="lod"`` / ``kind="partition"`` wrappers and plain
        groups). The convenience-builder path
        (``_add_gsplats_as_lod_group``) leaves the parent without bounds
        because each leaf carries its own; ``kind="partition"`` wrappers
        already persist their union at write time
        (``add_*_partition_wrapper_impl``); only ``kind="lod"`` wrappers
        were left without aggregate bounds, so the viewer's
        ``loadLodGroupNode`` saw empty bounds for a nested LOD-of-LOD
        construction and skipped that level in projection.

        **Never overwrites** an authored ``position_bounds`` — only
        fills missing values. Children with empty / mismatched bounds
        are skipped in the union (same convention as the viewer's
        registry projection).
        """

        def union(
            a: Optional[Dict[str, List[float]]], b: Optional[Dict[str, List[float]]]
        ) -> Optional[Dict[str, List[float]]]:
            if a is None:
                return b
            if b is None:
                return a
            a_min, a_max = a["min"], a["max"]
            b_min, b_max = b["min"], b["max"]
            if len(a_min) != len(b_min) or len(a_min) != len(a_max):
                # Mismatched dimensionality — skip b. Same defensive
                # fallback as the viewer's registry.
                return a
            return {
                "min": [min(a_min[i], b_min[i]) for i in range(len(a_min))],
                "max": [max(a_max[i], b_max[i]) for i in range(len(a_max))],
            }

        def resolve(group: "zarr.Group") -> Optional[Dict[str, List[float]]]:
            """Return the position_bounds of a group (leaf or wrapper).

            Returns None when the group has no leaves with bounds (e.g.
            empty group or all-mismatched children) so callers can skip.
            """
            attrs = dict(group.attrs)
            authored = attrs.get("position_bounds")
            if isinstance(authored, dict) and "min" in authored and "max" in authored:
                return {
                    "min": list(authored["min"]),
                    "max": list(authored["max"]),
                }
            # No authored bounds → recurse into children (groups only;
            # zarr arrays don't have descendants).
            acc: Optional[Dict[str, List[float]]] = None
            for child_name in group.keys():
                child = group[child_name]
                if not hasattr(child, "keys"):
                    continue
                acc = union(acc, resolve(child))
            return acc

        def walk(group: "zarr.Group") -> None:
            attrs = dict(group.attrs)
            if attrs.get("kind") == "lod" and "position_bounds" not in attrs:
                aggregated = resolve(group)
                if aggregated is not None:
                    group.attrs["position_bounds"] = aggregated
                    aprint(
                        f"  📐 Back-filled position_bounds on "
                        f"kind=lod group {group.path or '/'}"
                    )
            for child_name in group.keys():
                child = group[child_name]
                if hasattr(child, "keys"):
                    walk(child)

        walk(store)

    def _finalize_lod_display_types(self, store: zarr.Group) -> None:
        """Back-fill missing ``display_type`` on kind=lod groups.

        Walks the zarr tree and, for every group whose attrs declare
        ``kind == 'lod'`` without a ``display_type``, resolves one from
        the **finest** child's own type (recursing through nested
        kind=lod / kind=partition groups). The convenience-builder path
        (``_add_gsplats_as_lod_group``) already sets ``display_type``
        explicitly; this hook serves explicit-builder constructions
        where the user wrote ``add_lod_group(...)`` + a mix of leaf
        types and never set the parent's ``display_type`` themselves.

        **Never overwrites** an authored ``display_type`` — only fills
        missing values.
        """

        def resolve(group: "zarr.Group") -> str:
            """Return the display_type of a group (leaf or wrapper)."""
            attrs = dict(group.attrs)
            t = attrs.get("type")
            if t in ("points", "lines", "gsplats"):
                return str(t)  # leaf
            kind = attrs.get("kind")
            if kind in ("lod", "partition") and "display_type" in attrs:
                return str(attrs["display_type"])
            # Plain group or kind=lod / kind=partition without display_type
            # → recurse into children. Children of an lod_group are
            # stored in coarsest→finest order, so the finest is the
            # last one — that's the one to read.
            child_names = sorted(group.keys())
            if not child_names:
                return ""  # nothing to resolve
            finest_child = group[child_names[-1]]
            return resolve(finest_child)

        def walk(group: "zarr.Group") -> None:
            attrs = dict(group.attrs)
            if attrs.get("kind") == "lod" and "display_type" not in attrs:
                resolved = resolve(group)
                if resolved:
                    group.attrs["display_type"] = resolved
                    aprint(
                        f"  📐 Back-filled display_type={resolved!r} on "
                        f"kind=lod group {group.path or '/'}"
                    )
            for child_name in group.keys():
                child = group[child_name]
                # Only recurse into groups (not arrays).
                if hasattr(child, "keys"):
                    walk(child)

        walk(store)

    def _expand_bounds_with_transforms(self, store: zarr.Group) -> None:
        """Expand scene-level position bounds into world space.

        Walks the zarr tree, composes the world-space transform chain for
        each leaf node, applies it to the per-node (local) position bounds,
        and stores the union of all world-space bounds as the scene-level
        ``position_bounds``.

        Two independent transform families are composed down the hierarchy
        and applied together:

        - The 4x4 spatial ``transform`` (translate / rotate / scale) moves
          the **displayed** dimensions (the ones the viewer maps to mesh
          x/y/z). The matrix is applied to the box by transforming all 8
          corners (see :func:`transform_bounding_box`), which is correct
          under rotation — only transforming the (min, max) corner pair
          would underestimate the rotated extent.
        - The per-dimension ``nd_transform`` (affine scale/offset) moves the
          **non-displayed** dimensions (slider axes).

        Getting the spatial 4x4 into the scene bounds is load-bearing for the
        viewer: per-frame dynamic clipping derives near/far from a bounding
        sphere built from this metadata. If a node is translated far from the
        origin but its transform is ignored here, the sphere is too small and
        that geometry gets clipped as the camera rotates.

        Args:
            store: The opened zarr store (in r+ mode)
        """
        # Guard: need both scene_dimensions and scene_bounds
        if self._scene_bounds is None:
            return
        if "scene_dimensions" not in store.attrs:
            return

        from ..core.dimensions import Dimensions
        from ..core.transforms import (
            read_transform_from_zarr,
            transform_bounding_box,
        )
        from ..validation.nd_transforms import (
            apply_nd_transform_to_bounds,
            compose_nd_transforms,
        )

        dimensions = Dimensions.from_dict(store.attrs["scene_dimensions"])
        # The 4x4 transform's x/y/z axes map, in order, to the displayed
        # dimensions — matching how the viewer projects nD positions to the
        # mesh's x/y/z before applying the node transform.
        displayed = dimensions.displayed[:3]

        def apply_matrix_to_displayed_dims(
            bounds: dict[str, list[float]], matrix: "np.ndarray"
        ) -> dict[str, list[float]]:
            """Apply a 4x4 world matrix to the displayed dims of ``bounds``."""
            min_vals = list(bounds["min"])
            max_vals = list(bounds["max"])
            # Gather the displayed-dim sub-box into 3D (missing axes -> 0,
            # mirroring the viewer's zero-padding of < 3 displayed dims).
            lo3 = [0.0, 0.0, 0.0]
            hi3 = [0.0, 0.0, 0.0]
            for axis, dim in enumerate(displayed):
                if dim < len(min_vals):
                    lo3[axis] = min_vals[dim]
                    hi3[axis] = max_vals[dim]
            new_lo, new_hi = transform_bounding_box(matrix, lo3, hi3)
            for axis, dim in enumerate(displayed):
                if dim < len(min_vals):
                    min_vals[dim] = float(new_lo[axis])
                    max_vals[dim] = float(new_hi[axis])
            return {"min": min_vals, "max": max_vals}

        # Collect all world-space bounds from leaf nodes
        all_world_bounds: list[dict[str, list[float]]] = []

        def walk(
            group: zarr.Group,
            nd_chain: list[dict],
            world_matrix: "np.ndarray",
            has_matrix: bool,
        ) -> None:
            """Recursively walk zarr tree, composing both transform families."""
            attrs = dict(group.attrs)

            chain = list(nd_chain)
            nd_t = attrs.get("nd_transform", None)
            if nd_t:
                chain.append(nd_t)

            node_matrix = world_matrix
            node_has_matrix = has_matrix
            raw_transform = attrs.get("transform", None)
            if raw_transform is not None:
                local_matrix = read_transform_from_zarr(list(raw_transform))
                # world = ancestors @ this  (child transform applied first).
                node_matrix = world_matrix @ local_matrix
                node_has_matrix = True

            node_type = attrs.get("type", None)
            if node_type in ("points", "lines", "gsplats"):
                # Leaf node with geometry
                local_bounds = attrs.get("position_bounds", None)
                if local_bounds:
                    transformed = local_bounds
                    if chain:
                        world_nd_t = compose_nd_transforms(*chain)
                        transformed = apply_nd_transform_to_bounds(
                            transformed, world_nd_t, dimensions
                        )
                    if node_has_matrix:
                        transformed = apply_matrix_to_displayed_dims(
                            transformed, node_matrix
                        )
                    all_world_bounds.append(transformed)

            # Recurse into child groups
            for child_name in sorted(group.group_keys()):
                walk(group[child_name], chain, node_matrix, node_has_matrix)

        walk(store, [], np.eye(4, dtype=np.float64), False)

        # If no leaf nodes found, nothing to do
        if not all_world_bounds:
            return

        # Union all world-space bounds (same logic as _update_scene_bounds)
        world_scene_bounds: dict[str, list[float]] = {
            "min": list(all_world_bounds[0]["min"]),
            "max": list(all_world_bounds[0]["max"]),
        }
        for bounds in all_world_bounds[1:]:
            node_ndim = len(bounds["min"])
            scene_ndim = len(world_scene_bounds["min"])

            if node_ndim > scene_ndim:
                world_scene_bounds["min"].extend(bounds["min"][scene_ndim:])
                world_scene_bounds["max"].extend(bounds["max"][scene_ndim:])
                scene_ndim = node_ndim

            for i in range(min(node_ndim, scene_ndim)):
                world_scene_bounds["min"][i] = min(
                    world_scene_bounds["min"][i], bounds["min"][i]
                )
                world_scene_bounds["max"][i] = max(
                    world_scene_bounds["max"][i], bounds["max"][i]
                )

        # Overwrite scene-level bounds with world-space bounds
        store.attrs["position_bounds"] = world_scene_bounds
        self._scene_bounds = world_scene_bounds

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
            spatial_index_data=spatial_index_data,
            dtype=positions.dtype,
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
        n_elements: int,
    ) -> None:
        """Write colors dataset to Zarr using ArrayEncoder.

        Canonical writer for the ``COLOR`` semantic type across all three
        geometry types (Points, Lines, GSplats). Lines previously
        duplicated this logic inline in ``write_lines``; the shared helper
        keeps the default-precision selection identical across geometries.

        Args:
            group: Zarr group to write to
            colors: Colors array or tuple/list
            spatial_index_data: Optional spatial index for chunk optimization
            n_elements: Logical element count (points / vertices / splats).
                This must not be inferred from the
                positions zarr array because duplicate positions may be stored as
                an array_ref with physical shape ``(0, D)``.
        """
        n_points = n_elements  # local alias keeps the rest of the body unchanged
        # Handle scalar vs array
        color_mode: Optional[Literal["sdr", "hdr"]] = None
        if isinstance(colors, (tuple, list)):
            # Detect HDR vs SDR from values
            max_val = max(colors)
            color_mode = "hdr" if max_val > 1.0 else "sdr"
            if color_mode == "hdr":
                aprint("  ✓ Detected HDR colors (values > 1.0)")
            n_elems = n_points
            color_chunks = None
        else:
            if colors.shape[0] == 1:
                n_elems = n_points
                color_chunks = None
            else:
                n_elems = None
                color_chunks = _calculate_intelligent_chunks(
                    colors.shape,
                    spatial_index_data=spatial_index_data,
                    dtype=colors.dtype,
                )

            # Detect color_mode for float arrays
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

        # Store color data range for layer controls (min/max of original data)
        if isinstance(colors, np.ndarray) and colors.size > 0:
            group.attrs["color_data_range"] = [
                float(colors.min()),
                float(colors.max()),
            ]
        elif isinstance(colors, (tuple, list)):
            group.attrs["color_data_range"] = [
                float(min(colors)),
                float(max(colors)),
            ]

    def _write_positive_scalar_dataset(
        self,
        group: zarr.Group,
        data: Union[NDArray[np.float32], float, int],
        name: str,
        spatial_index_data: Optional[Dict[str, Any]],
        n_elements: int,
        log_label_singular: Optional[str] = None,
    ) -> float:
        """Canonical writer for the ``POSITIVE_SCALAR`` semantic type.

        Used by Points (``radii``), Lines (``widths``), and GSplats
        (``amplitudes``). Returns the maximum value so callers that
        cache it for layer-controls (e.g. radius/amplitude range
        metadata) don't have to recompute.

        ``log_label_singular`` controls the aprint output —
        ``"radius"`` / ``"width"`` / ``"amplitude"``. Defaults to
        ``name`` if not provided.

        Args:
            group: Zarr group to write to.
            data: Array of positive scalars or a single broadcast value.
            name: Dataset name in the zarr group (e.g. ``"radii"``,
                ``"widths"``, ``"amplitudes"``).
            spatial_index_data: Optional spatial-ordering metadata used
                by ``_calculate_intelligent_chunks`` to pick chunk
                boundaries that line up with the spatial index.
            n_elements: Logical element count; must not be inferred from
                the positions array because deduplicated positions can be
                stored as an array_ref with physical shape ``(0, D)``.
            log_label_singular: Optional singular form for the aprint
                log line ("radius" / "width" / "amplitude"). Defaults to
                the dataset name.

        Returns:
            Maximum value across ``data``.
        """
        label = log_label_singular or name

        if isinstance(data, (int, float)):
            max_value = float(data)
            n_elems = n_elements
            chunks = None
        else:
            max_value = float(np.max(data))
            if data.shape[0] == 1:
                n_elems = n_elements
                chunks = None
            else:
                n_elems = None
                chunks = _calculate_intelligent_chunks(
                    data.shape, spatial_index_data=spatial_index_data
                )

        aprint(f"  ✓ Max {label}: {max_value:.3f}")

        self._encoder.encode(
            data=data,
            zarr_group=group,
            name=name,
            semantic_type=SemanticType.POSITIVE_SCALAR,
            mode=self._encoding_mode,
            n_elements=n_elems,
            chunks=chunks,
            compressor=self.compressor,
        )

        enc = group[name].attrs.get("encoding", {})
        enc_name = enc.get("name", "unknown")
        if enc_name == "broadcasted":
            aprint(f"  ✓ Wrote {name} (broadcasted - uniform)")
        elif enc_name == "array_ref":
            aprint(f"  ✓ Wrote {name} (reference to {enc['target']})")
        elif enc_name.startswith("log_scalar"):
            aprint(f"  ✓ Wrote {name} ({enc_name} - log encoding)")
        else:
            aprint(f"  ✓ Wrote {name} ({enc_name})")

        return max_value

    def _write_bounded_scalar_dataset(
        self,
        group: zarr.Group,
        data: Union[NDArray[np.float32], float, int],
        name: str,
        bounds: Tuple[float, float],
        spatial_index_data: Optional[Dict[str, Any]],
        n_elements: int,
        log_label_singular: Optional[str] = None,
    ) -> float:
        """Canonical writer for the ``BOUNDED_SCALAR`` semantic type.

        Used by Points + Lines (``sharpnesses``). The ``bounds`` tuple
        is forwarded to the encoder, which quantizes the data into
        Uint8 normalised to that range when the encoding mode allows.

        Returns the maximum value for callers that surface it on
        layer-control metadata.
        """
        label = log_label_singular or name

        if isinstance(data, (int, float)):
            max_value = float(data)
            n_elems = n_elements
            chunks = None
        else:
            max_value = float(np.max(data))
            if data.shape[0] == 1:
                n_elems = n_elements
                chunks = None
            else:
                n_elems = None
                chunks = _calculate_intelligent_chunks(
                    data.shape, spatial_index_data=spatial_index_data
                )

        aprint(f"  ✓ Max {label}: {max_value:.3f}")

        self._encoder.encode(
            data=data,
            zarr_group=group,
            name=name,
            semantic_type=SemanticType.BOUNDED_SCALAR,
            mode=self._encoding_mode,
            bounds=bounds,
            n_elements=n_elems,
            chunks=chunks,
            compressor=self.compressor,
        )

        enc = group[name].attrs.get("encoding", {})
        enc_name = enc.get("name", "unknown")
        if enc_name == "broadcasted":
            aprint(f"  ✓ Wrote {name} (broadcasted - uniform)")
        elif enc_name == "array_ref":
            aprint(f"  ✓ Wrote {name} (reference to {enc['target']})")
        elif enc_name == "bounded_scalar_uint8":
            aprint(
                f"  ✓ Wrote {name} (uint8, quantized to [{bounds[0]:g}, {bounds[1]:g}])"
            )
        else:
            aprint(f"  ✓ Wrote {name} ({enc_name})")

        return max_value

    def _write_radii_dataset(
        self,
        group: zarr.Group,
        radii: Union[NDArray[np.float32], float, int],
        spatial_index_data: Optional[Dict[str, Any]],
        n_points: int,
    ) -> float:
        """Thin Points-specific wrapper over ``_write_positive_scalar_dataset``.

        Kept as a named helper because ``write_points`` reads more
        clearly with the geometry-specific name. New geometries should
        call ``_write_positive_scalar_dataset`` directly with their own
        dataset name.
        """
        return self._write_positive_scalar_dataset(
            group=group,
            data=radii,
            name="radii",
            spatial_index_data=spatial_index_data,
            n_elements=n_points,
            log_label_singular="radius",
        )

    def _write_sharpness_dataset(
        self,
        group: zarr.Group,
        sharpness: Union[NDArray[np.float32], float, int],
        spatial_index_data: Optional[Dict[str, Any]],
        n_points: int,
    ) -> float:
        """Thin Points-specific wrapper over ``_write_bounded_scalar_dataset``.

        Like ``_write_radii_dataset``, kept for readability in
        ``write_points``. New geometries call
        ``_write_bounded_scalar_dataset`` directly with the appropriate
        bounds tuple.
        """
        return self._write_bounded_scalar_dataset(
            group=group,
            data=sharpness,
            name="sharpnesses",
            bounds=(0.0, SHARPNESS_MAX),
            spatial_index_data=spatial_index_data,
            n_elements=n_points,
            log_label_singular="sharpness",
        )

    def _write_scalars_dataset(
        self,
        group: zarr.Group,
        scalars: Union[NDArray[np.float32], float, int],
        spatial_index_data: Optional[Dict[str, Any]],
        n_elements: int,
    ) -> None:
        """Write scalars dataset to Zarr for colormap lookup.

        Args:
            group: Zarr group to write to
            scalars: Scalar array or uniform value
            spatial_index_data: Optional spatial index for chunk optimization
            n_elements: Logical element count. This must not be inferred from
                the position zarr array because duplicate positions/vertices may
                be stored as an array_ref with physical shape ``(0, D)``.
        """
        # Validate that this is a geometry group. The logical element count is
        # passed by the caller; physical zarr shape can be zero for array_ref.
        pos_key = next(
            (k for k in ("positions", "vertices", "centers") if k in group),
            None,
        )
        if pos_key is None:
            raise RuntimeError(
                f"No position data found in group '{group.path}' "
                f"(expected 'positions', 'vertices', or 'centers')"
            )

        if isinstance(scalars, (int, float)):
            n_elems = n_elements
            chunks = None
            scalar_min = float(scalars)
            scalar_max = float(scalars)
        else:
            scalars = np.asarray(scalars, dtype=np.float32)
            scalar_min = float(np.min(scalars))
            scalar_max = float(np.max(scalars))
            if scalars.shape[0] == 1:
                n_elems = n_elements
                chunks = None
            else:
                n_elems = None
                chunks = _calculate_intelligent_chunks(
                    scalars.shape, spatial_index_data=spatial_index_data
                )

        self._encoder.encode(
            data=scalars,
            zarr_group=group,
            name="scalars",
            semantic_type=SemanticType.POSITIVE_SCALAR,
            mode=self._encoding_mode,
            n_elements=n_elems,
            chunks=chunks,
            compressor=self.compressor,
        )

        # Store scalar data range for layer controls
        group.attrs["scalar_data_range"] = [scalar_min, scalar_max]
        aprint(f"  ✓ Wrote scalars (range [{scalar_min:.4f}, {scalar_max:.4f}])")

    def _write_labels_csr(
        self,
        group: zarr.Group,
        labels: "Sequence[str]",
        n_elements: int,
        sort_order: Optional[np.ndarray] = None,
    ) -> None:
        """Write per-element string labels using CSR-style encoding.

        Stores two zarr arrays:
        - ``label_offsets``: uint64 of shape (N+1,) — byte offset of each label
        - ``label_bytes``: uint8 — concatenated UTF-8 encoded label strings

        Label ``i`` is decoded as ``label_bytes[offsets[i]:offsets[i+1]]``.
        Empty strings (null labels) have ``offsets[i] == offsets[i+1]``.

        Args:
            group: Zarr group to write to
            labels: Sequence of strings, one per element. Length must equal n_elements.
            n_elements: Expected element count (for validation)
            sort_order: Optional index array to reorder labels (e.g. from spatial ordering).
                For points: ``ordering_data["sort_order"]``
                For lines: ``ordering_data["vertex_sort_indices"]``
                For gsplats: ``ordering_data["sort_order"]``
        """
        if len(labels) != n_elements:
            raise ValueError(
                f"Labels length ({len(labels)}) must match element count ({n_elements})"
            )

        # Apply spatial reordering if present
        ordered_labels: "Sequence[str]" = labels
        if sort_order is not None:
            ordered_labels = [labels[i] for i in sort_order]

        # Build CSR arrays
        offsets = np.zeros(n_elements + 1, dtype=np.uint64)
        encoded_parts: list[bytes] = []
        for i, label in enumerate(ordered_labels):
            encoded = label.encode("utf-8") if label else b""
            encoded_parts.append(encoded)
            offsets[i + 1] = offsets[i] + len(encoded)

        total_bytes = int(offsets[-1])
        label_bytes = np.zeros(max(total_bytes, 1), dtype=np.uint8)
        pos = 0
        for encoded in encoded_parts:
            if encoded:
                label_bytes[pos : pos + len(encoded)] = np.frombuffer(
                    encoded, dtype=np.uint8
                )
                pos += len(encoded)

        # Write to zarr
        group.create_dataset(
            "label_offsets",
            data=offsets,
            chunks=(min(n_elements + 1, 65536),),
            compressor=self.compressor,
            overwrite=True,
        )
        group.create_dataset(
            "label_bytes",
            data=label_bytes,
            chunks=(min(total_bytes, 65536) if total_bytes > 0 else 1,),
            compressor=self.compressor,
            overwrite=True,
        )
        group.attrs["has_labels"] = True
        n_nonempty = sum(1 for lbl in ordered_labels if lbl)
        aprint(
            f"  ✓ Wrote labels ({n_nonempty}/{n_elements} non-empty, {total_bytes:,} bytes)"
        )

    # ------------------------------------------------------------------
    # Image labels (per-element image blobs, CSR-encoded)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_image_label(item: Any) -> bytes:
        """Convert a single image label input to encoded bytes.

        Accepts:
        - ``bytes`` / ``bytearray`` — used as-is (pre-encoded JPEG/WebP/PNG)
        - ``PIL.Image.Image`` — encoded to WebP (quality 85)
        - ``numpy.ndarray`` (H, W, C) uint8 — converted to PIL, then WebP
        - ``pathlib.Path`` / ``str`` — file read as raw bytes

        Returns:
            Encoded image bytes, or ``b""`` for None / empty inputs.
        """
        if item is None:
            return b""
        if isinstance(item, (bytes, bytearray)):
            return bytes(item)
        if isinstance(item, Path):
            return item.read_bytes()
        if isinstance(item, str):
            return Path(item).read_bytes()

        # PIL Image
        try:
            from PIL import Image as PILImage

            if isinstance(item, PILImage.Image):
                import io

                buf = io.BytesIO()
                item.save(buf, format="webp", quality=85)
                return buf.getvalue()
        except ImportError:
            raise ImportError(
                "Pillow is required to encode PIL Image objects as image labels. "
                "Install it with: pip install Pillow"
            )

        # numpy array (H, W, C) uint8
        if isinstance(item, np.ndarray):
            try:
                import io

                from PIL import Image as PILImage

                if item.ndim == 2:
                    pil_img = PILImage.fromarray(item, mode="L")
                elif item.ndim == 3 and item.shape[2] == 3:
                    pil_img = PILImage.fromarray(item, mode="RGB")
                elif item.ndim == 3 and item.shape[2] == 4:
                    pil_img = PILImage.fromarray(item, mode="RGBA")
                else:
                    raise ValueError(
                        f"Unsupported ndarray shape for image label: {item.shape}. "
                        f"Expected (H, W), (H, W, 3), or (H, W, 4)."
                    )
                buf = io.BytesIO()
                pil_img.save(buf, format="webp", quality=85)
                return buf.getvalue()
            except ImportError:
                raise ImportError(
                    "Pillow is required to encode numpy arrays as image labels. "
                    "Install it with: pip install Pillow"
                )

        raise TypeError(
            f"Unsupported image label type: {type(item).__name__}. "
            f"Expected bytes, PIL.Image, numpy.ndarray, or file path."
        )

    def _write_image_labels_csr(
        self,
        group: zarr.Group,
        image_labels: Any,
        n_elements: int,
        sort_order: Optional[np.ndarray] = None,
    ) -> None:
        """Write per-element image labels using CSR-style encoding.

        Stores two zarr arrays:
        - ``image_label_offsets``: uint64 of shape (N+1,) — byte offset of each image
        - ``image_label_bytes``: uint8 — concatenated encoded image blobs

        Image ``i`` is decoded as ``image_label_bytes[offsets[i]:offsets[i+1]]``.
        Empty entries (no image) have ``offsets[i] == offsets[i+1]``.

        The ``image_label_bytes`` array uses **no compression** (``compressor=None``)
        because the image blobs are already compressed (JPEG/WebP/PNG). The offsets
        array uses the scene's default compressor since it is small.

        Args:
            group: Zarr group to write to.
            image_labels: Per-element images. Accepted types:
                - ``List[bytes]``: pre-encoded blobs
                - ``List[PIL.Image.Image]``: auto-encoded to WebP
                - ``List[numpy.ndarray]``: (H,W,C) uint8, auto-encoded to WebP
                - ``List[Path]`` or ``List[str]``: file paths, read as bytes
                - ``Dict[int, Any]``: sparse — missing indices get empty blobs
            n_elements: Expected element count (for validation).
            sort_order: Optional index array to reorder (from spatial ordering).
        """
        # Normalize dict (sparse) to list
        if isinstance(image_labels, dict):
            normalized: List[bytes] = [b""] * n_elements
            for idx, item in image_labels.items():
                if idx < 0 or idx >= n_elements:
                    raise ValueError(
                        f"Image label index {idx} out of range [0, {n_elements})"
                    )
                normalized[idx] = self._normalize_image_label(item)
            blob_list = normalized
        else:
            if len(image_labels) != n_elements:
                raise ValueError(
                    f"Image labels length ({len(image_labels)}) must match "
                    f"element count ({n_elements})"
                )
            blob_list = [self._normalize_image_label(item) for item in image_labels]

        # Apply spatial reordering if present
        if sort_order is not None:
            blob_list = [blob_list[i] for i in sort_order]

        # Build CSR arrays
        offsets = np.zeros(n_elements + 1, dtype=np.uint64)
        for i, blob in enumerate(blob_list):
            offsets[i + 1] = offsets[i] + len(blob)

        total_bytes = int(offsets[-1])
        image_bytes = np.zeros(max(total_bytes, 1), dtype=np.uint8)
        pos = 0
        for blob in blob_list:
            if blob:
                image_bytes[pos : pos + len(blob)] = np.frombuffer(blob, dtype=np.uint8)
                pos += len(blob)

        # Write offsets (small, compressible)
        group.create_dataset(
            "image_label_offsets",
            data=offsets,
            chunks=(min(n_elements + 1, 65536),),
            compressor=self.compressor,
            overwrite=True,
        )
        # Write image bytes — NO compression (already compressed blobs), 1MB chunks
        group.create_dataset(
            "image_label_bytes",
            data=image_bytes,
            chunks=(min(total_bytes, 1_048_576) if total_bytes > 0 else 1,),
            compressor=None,
            overwrite=True,
        )
        group.attrs["has_image_labels"] = True
        n_nonempty = sum(1 for b in blob_list if b)
        avg_size = total_bytes / n_nonempty if n_nonempty > 0 else 0
        aprint(
            f"  ✓ Wrote image labels ({n_nonempty}/{n_elements} non-empty, "
            f"{total_bytes:,} bytes, avg {avg_size:.0f} bytes/image)"
        )

    def _write_colormap_lut_if_needed(
        self,
        group: zarr.Group,
        attrs: Dict[str, Any],
    ) -> None:
        """Write custom colormap LUT to zarr if colormap is an array.

        If ``attrs["colormap"]`` is a numpy array, resolve it to a (256, 3) uint8
        LUT, write it as a dataset, and replace the attr value with ``"custom"``.
        String colormaps are left as-is.

        Args:
            group: Zarr group to write to
            attrs: Node attributes dict (modified in-place)
        """
        colormap = attrs.get("colormap")
        if colormap is None:
            return

        # The viewer defaults to ACES filmic tone-mapping, which intentionally
        # shifts hues for a pleasing HDR look. That hue shift distorts the exact
        # colors of a colormap LUT, so warn authors who rely on LUTs that they
        # may want to pin tone_mapping="Neutral" in the scene's viewer_config.
        # Skip the warning when:
        #   - the author has already chosen "Neutral", or
        #   - the colormap is the implicit grayscale default ("gray"), which has
        #     no hue for ACES to distort and is not a deliberate LUT choice.
        scene_tone_mapping = None
        if self._scene is not None and self._scene.viewer_config is not None:
            scene_tone_mapping = self._scene.viewer_config.tone_mapping
        is_grayscale_default = isinstance(colormap, str) and colormap == "gray"
        if (
            not self._lut_tone_mapping_warned
            and scene_tone_mapping != "Neutral"
            and not is_grayscale_default
        ):
            warnings.warn(
                "This scene uses a colormap LUT, but the viewer's default HDR "
                "tone-mapping is 'ACES', which intentionally shifts hues and can "
                "distort LUT colors. If exact colormap fidelity matters (e.g. for "
                "scientific color encoding), set tone_mapping='Neutral' in the "
                "scene's viewer_config.",
                UserWarning,
                stacklevel=3,
            )
            self._lut_tone_mapping_warned = True

        from ..colormaps import resolve_colormap
        from ..colormaps.builtins import BUILTIN_COLORMAP_NAMES

        if isinstance(colormap, str):
            if colormap in BUILTIN_COLORMAP_NAMES:
                # Built-in name — viewer resolves it directly, no LUT needed
                return

            # Non-built-in name (matplotlib/colorcet) — resolve to LUT and
            # store as "custom" so the viewer can render it without needing
            # matplotlib/colorcet at display time.
            lut = resolve_colormap(colormap)  # Raises ValueError if unknown
            group.create_dataset(
                "colormap_lut",
                data=lut,
                chunks=(256, 3),
                dtype=np.uint8,
            )
            attrs["colormap"] = "custom"
            aprint(
                f"  ✓ Resolved '{colormap}' to LUT and wrote as custom (256x3 uint8)"
            )
            return

        # Array colormap — resolve and write as dataset
        lut = resolve_colormap(colormap)  # (256, 3) uint8
        group.create_dataset(
            "colormap_lut",
            data=lut,
            chunks=(256, 3),
            dtype=np.uint8,
        )
        attrs["colormap"] = "custom"
        aprint("  ✓ Wrote custom colormap LUT (256x3 uint8)")

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
            "ordering_dims": ordering_data["ordering_dims"],
            "ordering_min": ordering_data["ordering_min"],
            "ordering_max": ordering_data["ordering_max"],
            "ordering_bits_per_dim": ordering_data["ordering_bits_per_dim"],
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

    def _build_lines_spatial_ordering_if_enabled(
        self,
        vertices: NDArray[np.float32],
        segments: NDArray[np.uint32],
        widths: Union[NDArray[np.float32], float],
        n_vertices: int,
        n_dims: int,
        n_segments: int,
    ) -> Optional[Dict[str, Any]]:
        """Build dual spatial ordering for Lines (vertices + segments).

        Args:
            vertices: Vertex positions
            segments: Segment index pairs
            widths: Vertex widths (array or scalar)
            n_vertices: Number of vertices
            n_dims: Number of dimensions
            n_segments: Number of segments

        Returns:
            Dict with sorted arrays, sort indices, chunk bounds, and ordering metadata.
            Or None if spatial ordering is disabled.
        """
        if not self.enable_spatial_index or n_vertices == 0:
            return None

        # Get scene dimensions from attrs
        if "scene_dimensions" not in self.store.attrs:
            aprint("  ⚠️ No scene dimensions - skipping spatial ordering")
            return None

        from ..core.dimensions import Dimensions

        scene_dims_dict = self.store.attrs["scene_dimensions"]
        dimensions = Dimensions.from_dict(scene_dims_dict)

        aprint(f"  🔍 Applying dual {self.ordering_method} ordering...")

        # Import ordering functions
        from .ordering import (
            compute_segment_chunk_bounds,
            compute_vertex_chunk_bounds,
            order_lines_spatial,
        )

        # Apply dual spatial ordering
        (
            sorted_vertices,
            sorted_segments,
            vertex_sort_indices,
            segment_sort_indices,
            ordering_metadata,
        ) = order_lines_spatial(
            vertices,
            segments,
            dimensions.dimensions,
            method=self.ordering_method,
        )

        # Compute chunk sizes (from TARGET_CHUNK_BYTES)
        from ..typing_utils import TARGET_CHUNK_BYTES

        # Vertex chunk size
        bytes_per_vertex = n_dims * 4 + 8  # Position + width + overhead
        vertex_chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_vertex)
        vertex_chunk_size = min(vertex_chunk_size, n_vertices)

        # Segment chunk size
        bytes_per_segment = 8 + 8  # 2 uint32 indices + overhead
        segment_chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_segment)
        segment_chunk_size = min(segment_chunk_size, n_segments)

        # Add chunk_size to metadata
        ordering_metadata["vertex_ordering"]["chunk_size"] = vertex_chunk_size
        ordering_metadata["segment_ordering"]["chunk_size"] = segment_chunk_size

        # Compute vertex chunk bounds
        vertex_chunk_bounds = compute_vertex_chunk_bounds(
            sorted_vertices,
            vertex_chunk_size,
            slice_dims=ordering_metadata["vertex_ordering"]["slice_dims"],
            dimensions=dimensions.dimensions,
        )

        # Compute segment chunk bounds
        # Need to expand widths if scalar or broadcasted
        if isinstance(widths, (int, float)):
            widths_expanded = np.full(n_vertices, float(widths), dtype=np.float32)
        elif isinstance(widths, np.ndarray) and widths.shape[0] == 1:
            # Broadcasted
            widths_expanded = np.full(n_vertices, widths[0], dtype=np.float32)
        else:
            widths_expanded = widths[vertex_sort_indices]  # Apply same reordering

        segment_chunk_bounds = compute_segment_chunk_bounds(
            sorted_vertices,
            sorted_segments,
            widths_expanded,
            segment_chunk_size,
            slice_dims=ordering_metadata["vertex_ordering"][
                "slice_dims"
            ],  # Use D-space dims
            dimensions=dimensions.dimensions,
        )

        aprint(
            f"  ✓ Dual ordering complete: {len(vertex_chunk_bounds)} vertex chunks, "
            f"{len(segment_chunk_bounds)} segment chunks"
        )

        return {
            "sorted_vertices": sorted_vertices,
            "sorted_segments": sorted_segments,
            "vertex_sort_indices": vertex_sort_indices,
            "segment_sort_indices": segment_sort_indices,
            "vertex_chunk_bounds": vertex_chunk_bounds,
            "segment_chunk_bounds": segment_chunk_bounds,
            "ordering": self.ordering_method,
            **ordering_metadata,
        }

    def _write_lines_spatial_ordering_to_zarr(
        self, group: zarr.Group, ordering_data: Dict[str, Any]
    ) -> None:
        """Write Lines spatial ordering metadata and dual chunk bounds to Zarr.

        Args:
            group: Parent Zarr group
            ordering_data: Ordering data with chunk bounds and metadata
        """
        aprint("  📝 Writing Lines spatial ordering metadata...")

        # Write vertex_chunk_bounds
        vertex_chunk_bounds = ordering_data["vertex_chunk_bounds"]
        if len(vertex_chunk_bounds) > 0:
            n_dims = vertex_chunk_bounds.shape[1]
            group.create_dataset(
                "vertex_chunk_bounds",
                data=vertex_chunk_bounds,
                shape=vertex_chunk_bounds.shape,
                dtype=np.float32,
                chunks=(vertex_chunk_bounds.shape[0], n_dims, 2),
                compressor=self.compressor,
            )

        # Write segment_chunk_bounds
        segment_chunk_bounds = ordering_data["segment_chunk_bounds"]
        if len(segment_chunk_bounds) > 0:
            n_dims = segment_chunk_bounds.shape[1]
            group.create_dataset(
                "segment_chunk_bounds",
                data=segment_chunk_bounds,
                shape=segment_chunk_bounds.shape,
                dtype=np.float32,
                chunks=(segment_chunk_bounds.shape[0], n_dims, 2),
                compressor=self.compressor,
            )

        aprint(
            f"  ✓ Dual spatial ordering written: {len(vertex_chunk_bounds)} vertex chunks, "
            f"{len(segment_chunk_bounds)} segment chunks"
        )

    def _compute_content_hashes(self, store: zarr.Group) -> str:
        """
        Compute content hashes for all nodes using post-order traversal.

        Called during finalize() after all nodes have been written.
        Uses xxhash64 for speed.

        Args:
            store: Root zarr group

        Returns:
            Root content hash
        """

        def compute_hash_recursive(group_path: str) -> str:
            """Recursively compute hash for a group and its children."""
            group = store[group_path] if group_path else store

            hasher = xxhash.xxh64()

            # 1. Hash this node's own datasets (positions, colors, etc.)
            for dataset_name in sorted(group.array_keys()):
                dataset = group[dataset_name]
                hasher.update(dataset[:].tobytes())

            # 2. Hash metadata (excluding content_hash to avoid recursion)
            attrs = {k: v for k, v in dict(group.attrs).items() if k != "content_hash"}
            hasher.update(json.dumps(attrs, sort_keys=True, default=str).encode())

            # 3. Hash child groups (recursively, sorted for determinism)
            for child_name in sorted(group.group_keys()):
                child_path = f"{group_path}/{child_name}" if group_path else child_name
                child_hash = compute_hash_recursive(child_path)
                hasher.update(child_hash.encode())

            # Store hash in this node's attrs
            content_hash = hasher.hexdigest()
            group.attrs["content_hash"] = content_hash

            return content_hash

        # Start from root (empty path)
        root_hash = compute_hash_recursive("")
        aprint(f"Scene content hash: {root_hash[:16]}...")
        return root_hash

    def finalize(self) -> None:
        """Finalize the Zarr store with metadata consolidation."""
        if self._is_finalized:
            return

        # Auto-inject default hover overlay if labels exist but no hover overlay defined
        if self._scene is not None:
            self._scene._auto_inject_hover_overlay()

        try:
            aprint("🔧 Finalizing Zarr store...")

            # Close the store to ensure all data is written
            if hasattr(self.store, "close"):
                self.store.close()

            # Re-open the store to consolidate metadata
            # This ensures all groups and datasets are properly written to disk
            store = zarr.open_group(self._store_path, mode="r+")

            # Store scene-level position bounds (union of all node bounds)
            if self._scene_bounds is not None:
                store.attrs["position_bounds"] = self._scene_bounds
                aprint(
                    f"📦 Scene bounds (local): min={self._scene_bounds['min']}, max={self._scene_bounds['max']}"
                )

            # Expand bounds into world space: 4x4 spatial transforms on the
            # displayed dims + nd_transforms on the non-displayed dims.
            self._expand_bounds_with_transforms(store)
            if self._scene_bounds is not None:
                aprint(
                    f"🌍 Scene bounds (world): min={self._scene_bounds['min']}, "
                    f"max={self._scene_bounds['max']}"
                )

            # Validate discrete dimension ranges against actual data
            self._validate_discrete_dimension_ranges(store)

            # Back-fill missing ``display_type`` on kind=lod groups by
            # recursing through their finest child (see
            # ``_finalize_lod_display_types`` for the rule). Never
            # overwrites a value the user already authored.
            self._finalize_lod_display_types(store)

            # Back-fill missing ``position_bounds`` on kind=lod groups
            # by unioning descendant bounds. Without this, the viewer's
            # registry projection skips nested kind=lod children
            # (which never carried their own bounds) and the LOD
            # selector can't see them.
            self._finalize_lod_position_bounds(store)

            # Now consolidate metadata with all data present
            zarr.consolidate_metadata(store.store)

            # Compute content hashes (post-order: children before parents)
            self._compute_content_hashes(store)

            # Re-consolidate to include hashes in .zmetadata
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
