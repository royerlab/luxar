"""luxar.compiler – Progressive Zarr compiler for memory-efficient scene building.

This module provides the LuxarZarrCompiler class which implements progressive
writing to Zarr stores, enabling handling of arbitrarily large datasets without
memory constraints.
"""

from __future__ import annotations

import tempfile
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
from ..typing_utils.constants import SHARPNESS_MAX
from ..typing_utils.protocols import CompressorProtocol
from ._compiler.bounds import (
    compute_position_bounds,
    expand_bounds_with_transforms,
    update_scene_bounds,
)
from ._compiler.chunking import calculate_intelligent_chunks
from ._compiler.colormap import write_colormap_lut_if_needed
from ._compiler.context import DatasetCtx, OrderingCtx
from ._compiler.dataset_writers.colors import write_colors
from ._compiler.dataset_writers.positions import write_positions
from ._compiler.dataset_writers.scalars import (
    write_bounded_scalar,
    write_positive_scalar,
    write_radii,
    write_scalars,
    write_sharpness,
)
from ._compiler.finalize.hashing import compute_content_hashes
from ._compiler.finalize.lod_backfill import (
    finalize_lod_display_types,
    finalize_lod_position_bounds,
)
from ._compiler.finalize.validation import validate_discrete_dimension_ranges
from ._compiler.gsplat_assembly import (
    apply_gsplat_group_attrs,
    apply_gsplat_spatial_ordering,
    validate_gsplat_inputs,
    write_gsplat_arrays,
)
from ._compiler.labels.image_labels import write_image_labels_csr
from ._compiler.labels.text_labels import write_labels_csr
from ._compiler.spatial_ordering.lines import (
    build_lines_ordering,
    write_lines_ordering_to_zarr,
)
from ._compiler.spatial_ordering.points import (
    build_points_ordering,
    write_points_ordering_to_zarr,
)

# Ordering functions will be imported locally where needed to avoid circular imports


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
        chunks_2d = calculate_intelligent_chunks(
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

    # ------------------------------------------------------------------
    # GSplat assembly — bodies live in _compiler/gsplat_assembly.py
    # ------------------------------------------------------------------

    def _make_ordering_ctx(self) -> OrderingCtx:
        """Build the narrow spatial-ordering context."""
        return OrderingCtx(
            enable_spatial_index=self.enable_spatial_index,
            ordering_method=self.ordering_method,
        )

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
        return validate_gsplat_inputs(centers, amplitudes, cholesky_factors, colors)

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
        return apply_gsplat_spatial_ordering(
            centers,
            amplitudes,
            cholesky_factors,
            colors,
            n_splats,
            n_dims,
            cholesky_is_uniform,
            self._make_ordering_ctx(),
            coverage_sigma,
        )

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
        return write_gsplat_arrays(
            group,
            centers,
            amplitudes,
            cholesky_factors,
            colors,
            n_splats,
            n_dims,
            cholesky_is_uniform,
            ordering_data,
            self._make_dataset_ctx(),
        )

    def _apply_gsplat_group_attrs(
        self,
        group: zarr.Group,
        metadata: dict[str, Any],
        attrs: dict[str, Any],
    ) -> None:
        scene_tone_mapping = None
        if self._scene is not None and self._scene.viewer_config is not None:
            scene_tone_mapping = self._scene.viewer_config.tone_mapping
        self._lut_tone_mapping_warned = apply_gsplat_group_attrs(
            group,
            metadata,
            attrs,
            self.store,
            scene_tone_mapping,
            self._lut_tone_mapping_warned,
        )

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
        return build_points_ordering(
            positions, n_points, n_dims, radii, self._make_ordering_ctx(), self.store
        )

    # ------------------------------------------------------------------
    # Scene bounds — bodies live in _compiler/bounds.py
    # ------------------------------------------------------------------

    def _compute_position_bounds(
        self, positions: NDArray[np.float32]
    ) -> Dict[str, List[float]]:
        return compute_position_bounds(positions)

    def _update_scene_bounds(self, node_bounds: Dict[str, List[float]]) -> None:
        self._scene_bounds = update_scene_bounds(self._scene_bounds, node_bounds)

    # ------------------------------------------------------------------
    # Finalize-time tree passes — bodies live in _compiler/finalize/
    # ------------------------------------------------------------------

    def _validate_discrete_dimension_ranges(self, store: zarr.Group) -> None:
        validate_discrete_dimension_ranges(store, self._scene_bounds)

    def _finalize_lod_position_bounds(self, store: zarr.Group) -> None:
        finalize_lod_position_bounds(store)

    def _finalize_lod_display_types(self, store: zarr.Group) -> None:
        finalize_lod_display_types(store)

    def _expand_bounds_with_transforms(self, store: zarr.Group) -> None:
        self._scene_bounds = expand_bounds_with_transforms(store, self._scene_bounds)

    # ------------------------------------------------------------------
    # Per-attribute dataset serializers — bodies live in _compiler/datasets/
    # ------------------------------------------------------------------

    def _make_dataset_ctx(self) -> DatasetCtx:
        """Build the narrow encoder-config context for dataset serializers."""
        return DatasetCtx(
            encoder=self._encoder,
            encoding_mode=self._encoding_mode,
            compressor=self.compressor,
        )

    def _write_positions_dataset(
        self,
        group: zarr.Group,
        positions: NDArray[np.float32],
        spatial_index_data: Optional[Dict[str, Any]],
    ) -> None:
        write_positions(
            group, positions, spatial_index_data, self._make_dataset_ctx()
        )

    def _write_colors_dataset(
        self,
        group: zarr.Group,
        colors: Union[NDArray[np.float32], tuple, list],
        spatial_index_data: Optional[Dict[str, Any]],
        n_elements: int,
    ) -> None:
        write_colors(
            group, colors, spatial_index_data, n_elements, self._make_dataset_ctx()
        )

    def _write_positive_scalar_dataset(
        self,
        group: zarr.Group,
        data: Union[NDArray[np.float32], float, int],
        name: str,
        spatial_index_data: Optional[Dict[str, Any]],
        n_elements: int,
        log_label_singular: Optional[str] = None,
    ) -> float:
        return write_positive_scalar(
            group,
            data,
            name,
            spatial_index_data,
            n_elements,
            self._make_dataset_ctx(),
            log_label_singular,
        )

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
        return write_bounded_scalar(
            group,
            data,
            name,
            bounds,
            spatial_index_data,
            n_elements,
            self._make_dataset_ctx(),
            log_label_singular,
        )

    def _write_radii_dataset(
        self,
        group: zarr.Group,
        radii: Union[NDArray[np.float32], float, int],
        spatial_index_data: Optional[Dict[str, Any]],
        n_points: int,
    ) -> float:
        return write_radii(
            group, radii, spatial_index_data, n_points, self._make_dataset_ctx()
        )

    def _write_sharpness_dataset(
        self,
        group: zarr.Group,
        sharpness: Union[NDArray[np.float32], float, int],
        spatial_index_data: Optional[Dict[str, Any]],
        n_points: int,
    ) -> float:
        return write_sharpness(
            group, sharpness, spatial_index_data, n_points, self._make_dataset_ctx()
        )

    def _write_scalars_dataset(
        self,
        group: zarr.Group,
        scalars: Union[NDArray[np.float32], float, int],
        spatial_index_data: Optional[Dict[str, Any]],
        n_elements: int,
    ) -> None:
        write_scalars(
            group, scalars, spatial_index_data, n_elements, self._make_dataset_ctx()
        )

    # ------------------------------------------------------------------
    # Labels (per-element string + image blobs, CSR-encoded) —
    # bodies live in _compiler/labels/
    # ------------------------------------------------------------------

    def _write_labels_csr(
        self,
        group: zarr.Group,
        labels: "Sequence[str]",
        n_elements: int,
        sort_order: Optional[np.ndarray] = None,
    ) -> None:
        write_labels_csr(group, labels, n_elements, self.compressor, sort_order)

    def _write_image_labels_csr(
        self,
        group: zarr.Group,
        image_labels: Any,
        n_elements: int,
        sort_order: Optional[np.ndarray] = None,
    ) -> None:
        write_image_labels_csr(
            group, image_labels, n_elements, self.compressor, sort_order
        )

    def _write_colormap_lut_if_needed(
        self,
        group: zarr.Group,
        attrs: Dict[str, Any],
    ) -> None:
        scene_tone_mapping = None
        if self._scene is not None and self._scene.viewer_config is not None:
            scene_tone_mapping = self._scene.viewer_config.tone_mapping
        self._lut_tone_mapping_warned = write_colormap_lut_if_needed(
            group, attrs, scene_tone_mapping, self._lut_tone_mapping_warned
        )

    def _write_spatial_ordering_to_zarr(
        self, group: zarr.Group, ordering_data: Dict[str, Any]
    ) -> None:
        """Write spatial ordering metadata and chunk bounds to Zarr.

        Args:
            group: Parent Zarr group
            ordering_data: Ordering data with chunk_bounds and metadata
        """
        write_points_ordering_to_zarr(group, ordering_data, self.compressor)

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
        return build_lines_ordering(
            vertices,
            segments,
            widths,
            n_vertices,
            n_dims,
            n_segments,
            self._make_ordering_ctx(),
            self.store,
        )

    def _write_lines_spatial_ordering_to_zarr(
        self, group: zarr.Group, ordering_data: Dict[str, Any]
    ) -> None:
        """Write Lines spatial ordering metadata and dual chunk bounds to Zarr.

        Args:
            group: Parent Zarr group
            ordering_data: Ordering data with chunk bounds and metadata
        """
        write_lines_ordering_to_zarr(group, ordering_data, self.compressor)

    def _compute_content_hashes(self, store: zarr.Group) -> str:
        return compute_content_hashes(store)

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
