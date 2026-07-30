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

from ..encoding.compression import CompressorLike, resolve_compressor

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
)
from ..io.reader import DEFAULT_COMP
from ..io.writer import ZarrWriterProtocol
from ..typing_utils.aliases import ChunkSpec, MaxShape, NodePath, PointsMetadata
from ..typing_utils.config import DEFAULT_VERSION
from ._compiler.bounds import (
    compute_position_bounds,
    expand_bounds_with_transforms,
    update_scene_bounds,
)
from ._compiler.colormap import write_colormap_lut_if_needed
from ._compiler.context import (
    DatasetCtx,
    GeometryWriteCtx,
    GSplatsWriteCtx,
    OrderingCtx,
)
from ._compiler.finalize.hashing import compute_content_hashes
from ._compiler.finalize.lod_backfill import (
    finalize_lod_display_types,
    finalize_lod_position_bounds,
)
from ._compiler.finalize.validation import validate_discrete_dimension_ranges
from ._compiler.geometry_writers.gsplats import (
    write_gsplat_leaf_subtree as _write_gsplat_leaf_subtree_impl,
)
from ._compiler.geometry_writers.gsplats import write_gsplats as _write_gsplats_impl
from ._compiler.geometry_writers.lines import write_lines as _write_lines_impl
from ._compiler.geometry_writers.points import write_points as _write_points_impl
from ._compiler.gsplat_assembly import apply_gsplat_group_attrs
from ._compiler.node_common import validate_node_path as _validate_node_path
from ._compiler.node_common import validate_render_attrs as _validate_render_attrs

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
        >>> with LuxarZarrCompiler('output.luxar.zarr') as compiler:
        ...     scene = compiler.create_scene(dimensions=dims)
        ...     positions = np.random.randn(10000, 3).astype(np.float32)
        ...     scene.add_points('points', positions)

        With HDR colors and custom dimensions:
        >>> dims = Dimensions([
        ...     Dimension('x', unit='um'),
        ...     Dimension('y', unit='um'),
        ...     Dimension('z', unit='um')
        ... ])
        >>> with LuxarZarrCompiler('scene.luxar.zarr') as compiler:
        ...     scene = compiler.create_scene(dimensions=dims)
        ...     # HDR colors with values > 1.0
        ...     colors = np.random.rand(1000, 3).astype(np.float32) * 5.0
        ...     scene.add_points('bright_points', positions, colors=colors)

        Large datasets (partitioned into multiple nodes):
        >>> dims = Dimensions.default_3d()
        >>> with LuxarZarrCompiler('huge.luxar.zarr', ordering_method="hilbert") as compiler:
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
        compressor: "CompressorLike" = DEFAULT_COMP,
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
        # Handle store path. Full scenes use the canonical ``.luxar.zarr``
        # extension; the path is normalized so callers that pass a bare name or
        # a plain ``.zarr`` still produce a canonically-named scene. Callers
        # should read back the final path via the ``store_path`` property.
        # Local import to avoid a module-load cycle
        # (luxar.utils.__init__ → demos → io.compiler).
        from ..utils.paths import normalize_zarr_path

        self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None
        if store_path is None:
            self._tmpdir = tempfile.TemporaryDirectory()
            self._store_path = Path(self._tmpdir.name) / "scene.luxar.zarr"
            aprint(f"📁 Using temporary directory: {self._store_path}")
        else:
            requested = Path(store_path)
            self._store_path = normalize_zarr_path(requested, ".luxar.zarr")
            if self._store_path.name != requested.name:
                aprint(
                    f"📁 Normalized scene path to canonical extension: "
                    f"{requested.name} → {self._store_path.name}"
                )
            aprint(f"📁 Creating scene at: {self._store_path}")

        # Store spatial ordering configuration
        self.enable_spatial_index = enable_spatial_index
        self.ordering_method = ordering_method

        # Opt-in auto-partition threshold. Read at add_points / add_gsplats
        # time via resolve_auto_partition(). None disables.
        if auto_partition_max_elements is not None and auto_partition_max_elements <= 0:
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
        # Fail fast on invalid render attrs BEFORE creating the group (same
        # contract as the geometry writers; the Node path validates earlier,
        # this covers the raw compiler API). The scene root ("/") and actual
        # ``overlays/<name>`` entries carry their own internal attr schemas
        # (scene dimensions / viewer config / overlay styling), so the
        # unknown-key guard is scoped to real geometry/group nodes. A bare
        # top-level group named ``overlays`` is still a user node at this point
        # and must not bypass typo detection. A non-str path is
        # treated as not-internal so the clean ValidationError from
        # ``_validate_node_path`` below (not an AttributeError here) surfaces.
        is_internal_namespace = False
        if isinstance(path, str):
            normalized_path = path.lstrip("/")
            if normalized_path == "overlays":
                raise ValueError(
                    "Top-level node path 'overlays' is reserved for screen-space "
                    "overlay metadata. Write internal overlays below 'overlays/<name>' "
                    "or choose a different user node name."
                )
            is_internal_namespace = path in ("/", "") or normalized_path.startswith(
                "overlays/"
            )
        _validate_render_attrs(attrs, reject_unknown=not is_internal_namespace)
        # Handle root path
        if path == "/" or path == "":
            group = self.store
        else:
            # Validate every path segment (rejects empty/dot-prefixed/
            # control-char names — the F1/F5 chokepoint) + strip the
            # leading slash.
            path = _validate_node_path(path)
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
        - sharpness=0.5 → all points standard Gaussian (broadcasted; normalized [0, 1] knob)
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
        metadata = _write_points_impl(
            self._make_geometry_ctx(),
            path,
            positions,
            colors,
            radii,
            sharpness,
            scalars,
            labels,
            image_labels,
            **attrs,
        )
        self._metadata_cache[metadata["path"]] = metadata
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
        - sharpness=0.5 → uniform sharpness (normalized [0, 1] knob; 0.5 = Gaussian)

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
        metadata = _write_lines_impl(
            self._make_geometry_ctx(),
            path,
            vertices,
            widths,
            colors,
            sharpness,
            scalars,
            indices,
            line_type,
            labels,
            image_labels,
            **attrs,
        )
        self._metadata_cache[path.lstrip("/")] = metadata
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

        # Fail fast on invalid render attrs BEFORE creating the parent group.
        _validate_render_attrs(attrs)

        # Validate every path segment (rejects empty/dot-prefixed names —
        # the F1/F5 chokepoint) + strip the leading slash.
        path = _validate_node_path(path)
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
                **({"lod_stats": lvl["lod_stats"]} if lvl.get("lod_stats") else {}),
                **({"extend_to_all": extend_to_all} if extend_to_all else {}),
                _skip_scene_bounds=True,
            )
            level_metas.append(level_meta)

        # Parent-node attrs. Persist after subgroup writes so they
        # don't get clobbered by side effects. A custom colormap (ndarray /
        # matplotlib name) must be resolved to a colormap_lut dataset +
        # colormap='custom' BEFORE the attrs land in JSON — same contract as
        # the flat writer (an ndarray in attrs is not JSON serializable).
        self._write_colormap_lut_if_needed(group, attrs)
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

        # Fail fast on invalid render attrs BEFORE creating the parent group.
        _validate_render_attrs(attrs)

        # Validate every path segment (rejects empty/dot-prefixed names —
        # the F1/F5 chokepoint) + strip the leading slash.
        path = _validate_node_path(path)
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
                **({"lod_stats": lvl["lod_stats"]} if lvl.get("lod_stats") else {}),
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

        # Resolve a custom colormap (ndarray / matplotlib name) to a
        # colormap_lut dataset + colormap='custom' before the JSON attr dump —
        # same contract as the flat writer.
        self._write_colormap_lut_if_needed(group, attrs)
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
        metadata = _write_gsplats_impl(
            self._make_gsplats_ctx(),
            path,
            centers,
            amplitudes,
            cholesky_factors,
            colors,
            labels,
            image_labels,
            **attrs,
        )
        self._metadata_cache[path.lstrip("/")] = metadata
        return metadata

    def write_gsplat_leaf_subtree(
        self,
        path: NodePath,
        leaf: Any,  # luxar.gsplats.tree.GSplatLeaf
        **attrs: Any,
    ) -> dict[str, Any]:
        """Write a ``GSplatLeaf`` (single set or additive ladder) into the scene.

        This is the scene-side seam onto :func:`~luxar.io._compiler.gsplat_tree.\
        write_gsplat_leaf` — the single authoring path also used by the
        standalone ``.gsplats.zarr`` writer. There is **no** parallel
        additive-ladder writer: the additive ``additive_<i>/`` subgroups, their
        attrs, spatial ordering, chunking and ``position_bounds`` come from the
        exact same code a standalone file uses, so a scene additive ladder is
        byte-identical to a standalone one by construction.

        Used for ``GSplatData`` embeds, whose arrays are always full per-splat
        (no uniform-Cholesky / scalar-amplitude / labels — those leaf-only
        scene features stay on :meth:`write_gsplats`).

        Args:
            path: Path for the gsplats node within the store.
            leaf: A :class:`~luxar.gsplats.tree.GSplatLeaf` (1 sub-LOD → flat
                leaf; >1 → additive ladder).
            **attrs: Node attributes (opacity, blending_mode, colormap,
                extend_to_all, truncation_radius, transform, ...).

        Returns:
            Aggregate metadata dict (incl. ``position_bounds``).
        """
        self._check_not_finalized("write_gsplat_leaf_subtree")
        # Validate every path segment (rejects empty/dot-prefixed names —
        # the F1/F5 chokepoint) + strip the leading slash.
        path = _validate_node_path(path)
        metadata = _write_gsplat_leaf_subtree_impl(
            self._make_gsplats_ctx(), path, leaf, **attrs
        )
        self._metadata_cache[path.lstrip("/")] = metadata
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
            compressor=resolve_compressor(self.compressor, dtype),
            maxshape=maxshape,
            overwrite=True,
        )

        aprint(f"📝 Created resizable dataset: {path}")
        return dataset

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

    def _make_geometry_ctx(self) -> GeometryWriteCtx:
        """Build the narrow context for the extracted geometry write pipelines.

        Bundles the dataset/ordering configs + compressor with two bound-method
        hooks for the orchestrator state a write mutates: the scene-bounds
        accumulator and the warn-once colormap-LUT flag.
        """
        return GeometryWriteCtx(
            store=self.store,
            dataset_ctx=self._make_dataset_ctx(),
            ordering_ctx=self._make_ordering_ctx(),
            compressor=self.compressor,
            update_scene_bounds=self._update_scene_bounds,
            write_colormap_lut=self._write_colormap_lut_if_needed,
        )

    def _make_gsplats_ctx(self) -> GSplatsWriteCtx:
        """Build the narrow context for the extracted GSplats write pipelines."""
        scene_tone_mapping = None
        if self._scene is not None and self._scene.viewer_config is not None:
            scene_tone_mapping = self._scene.viewer_config.tone_mapping
        return GSplatsWriteCtx(
            store=self.store,
            dataset_ctx=self._make_dataset_ctx(),
            ordering_ctx=self._make_ordering_ctx(),
            compressor=self.compressor,
            scene_tone_mapping=scene_tone_mapping,
            update_scene_bounds=self._update_scene_bounds,
            apply_gsplat_group_attrs=self._apply_gsplat_group_attrs,
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
