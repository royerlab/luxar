"""luxar.io.compiler – Progressive Zarr compiler for memory-efficient scene building.

This module provides the LuxarZarrCompiler class which implements progressive
writing to Zarr stores, enabling handling of arbitrarily large datasets without
memory constraints.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
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

from luxar._zarr_compat import close as zarr_close
from luxar._zarr_compat import consolidate, create_array
from luxar._zarr_compat import open_group as zarr_open_group

from ..core.dimensions import Dimensions
from ..encoding import (
    ArrayEncoder,
    EncodingMode,
)
from ..io.reader import DEFAULT_COMP
from ..io.writer import RollbackState, ZarrWriterProtocol
from ..typing_utils.aliases import ChunkSpec, MaxShape, NodePath, PointsMetadata
from ..typing_utils.config import DEFAULT_VERSION
from ..utils.arbol_warnings import arbol_warnings
from ..validation.writing import validate_render_attrs as _validate_render_attrs
from ._compiler.bounds import (
    WorldBoundsLeaf,
    collect_world_bounds,
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
from ._compiler.finalize.amplitude_window import harmonize_gsplat_amplitude_windows
from ._compiler.finalize.blending_warnings import warn_overlapping_blending
from ._compiler.finalize.hashing import compute_content_hashes
from ._compiler.finalize.lod_backfill import (
    finalize_lod_display_types,
    finalize_lod_position_bounds,
    warn_one_part_partition_anchors,
)
from ._compiler.finalize.validation import (
    prune_childless_wrappers,
    validate_discrete_dimension_ranges,
)
from ._compiler.geometry_writers.gsplats import (
    write_gsplat_leaf_subtree as _write_gsplat_leaf_subtree_impl,
)
from ._compiler.geometry_writers.gsplats import write_gsplats as _write_gsplats_impl
from ._compiler.geometry_writers.lines import write_lines as _write_lines_impl
from ._compiler.geometry_writers.mesh import write_mesh as _write_mesh_impl
from ._compiler.geometry_writers.points import write_points as _write_points_impl
from ._compiler.gsplat_assembly import apply_gsplat_group_attrs
from ._compiler.labels.text_labels import (
    validate_ladder_labels,
    write_ladder_union_labels_csr,
)
from ._compiler.node_common import prepare_transform_attrs as _prepare_transform_attrs
from ._compiler.node_common import validate_node_path as _validate_node_path
from ._compiler.node_common import warn_if_over_element_cap

# Ordering functions will be imported locally where needed to avoid circular imports


class LuxarZarrCompiler(ZarrWriterProtocol):
    """Progressive Zarr compiler with context manager support.

    This compiler writes data immediately to Zarr without keeping it in memory,
    enabling processing of datasets larger than available RAM.

    Args:
        store_path: Path where the Zarr store will be created. A ``.zip`` path
            normalizes the inner store name and publishes a single-file archive
            after finalization.
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

        Compile directly to a single-file archive:
        >>> with LuxarZarrCompiler('output.luxar.zarr.zip') as compiler:
        ...     scene = compiler.create_scene(dimensions=Dimensions.default_3d())
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
            store_path: Path for the Zarr store, or None for temporary. A
                ``.zip`` path normalizes the inner store name and publishes a
                single-file archive after finalization.
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
        # should read back the final path via the ``store_path`` property. A
        # ``.zip`` request writes to a hidden directory while active and
        # publishes the archive only after finalization succeeds.
        # Local import to avoid a module-load cycle
        # (luxar.utils.__init__ → demos → io.compiler).
        from ..utils.paths import normalize_zarr_path

        self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._archive_path: Optional[Path] = None
        self._archive_artifact_path: Optional[Path] = None
        if store_path is None:
            self._tmpdir = tempfile.TemporaryDirectory()
            self._store_path = Path(self._tmpdir.name) / "scene.luxar.zarr"
            aprint(f"📁 Using temporary directory: {self._store_path}")
        else:
            requested = Path(store_path)
            if requested.name.endswith(".zip"):
                inner_name = requested.name[:-4]
                if not inner_name:
                    raise ValueError(
                        "Archive output path must include a name before .zip"
                    )
                inner_path = requested.with_name(inner_name)
                normalized = normalize_zarr_path(inner_path, ".luxar.zarr")
                self._archive_path = Path(f"{normalized}.zip")
                if self._archive_path.is_symlink():
                    raise ValueError(
                        f"LuxarZarrCompiler refuses to write to {self._archive_path}: "
                        f"it is a symlink (→ {os.readlink(self._archive_path)}). The "
                        "archive is renamed into place, which would replace the link "
                        "rather than what it points at. Give the link's target as the "
                        "output path, or remove the link first."
                    )
                self._archive_path.parent.mkdir(parents=True, exist_ok=True)
                self._store_path = self._archive_path.parent / (
                    f".{self._archive_path.name}.compile-"
                    f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
                )
                self._archive_artifact_path = Path(f"{self._store_path}.zip")
                final_path = self._archive_path
            else:
                self._store_path = normalize_zarr_path(requested, ".luxar.zarr")
                final_path = self._store_path
            if final_path.name != requested.name:
                aprint(
                    f"📁 Normalized scene path to canonical extension: "
                    f"{requested.name} → {final_path.name}"
                )
            aprint(f"📁 Creating scene at: {final_path}")

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
        # Partition wrappers write one geometry leaf per part. Keep each
        # heuristic authoring warning once per logical node rather than once per
        # leaf. Keyed by (geometry kind, path) so the per-type heuristics share
        # one registry without their paths colliding — a new geometry type with
        # an authoring lint needs no new field here.
        self._authoring_warnings: set[tuple[str, str]] = set()

        # Create array encoder with specified encoding mode and float16 control
        self._encoder = ArrayEncoder(float16_allowed=float16_allowed)
        self._encoding_mode = encoding_mode
        self._float16_allowed = float16_allowed

        # Create root Zarr group
        self.store = zarr_open_group(self._store_path, mode="w")
        self.store.attrs.update(
            {
                "luxar_version": version,
                "type": "scene",
            }
        )

        self.compressor = compressor
        self._metadata_cache: Dict[str, Any] = {}
        self._is_finalized = False
        self._archive_finalize_failed = False
        # Transactions are re-entrant, not thread-local: scene authoring through
        # one compiler instance is single-threaded, like the writer itself.
        self._transaction_depth = 0

        # Scene-level bounds tracking (union of all node bounds)
        # Each entry is [min_per_dim, max_per_dim] where each is a list of floats
        self._scene_bounds: Optional[Dict[str, List[float]]] = None

        # Scene reference for finalize-time hover overlay auto-injection
        self._scene: Optional["Scene"] = None

        output_path = self._archive_path or self._store_path
        aprint(f"✅ Zarr compiler initialized at {output_path}")

    def __enter__(self) -> LuxarZarrCompiler:
        """Enter context manager."""
        return self

    def _check_not_finalized(self, op: str) -> None:
        """Refuse mutations after finalization or archive staging discard.

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
        if self._archive_finalize_failed:
            raise RuntimeError(
                f"Cannot {op} after archive staging was discarded. "
                "Create a new LuxarZarrCompiler."
            )

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, finalizing only on a clean exit.

        If an exception is propagating out of the ``with`` block (a write
        error, or Ctrl-C / KeyboardInterrupt) and the store was NOT already
        finalized, the store is left UNfinalized and a root ``incomplete``
        marker is stamped so a directory artifact is detectable; archive
        staging is discarded instead and the compiler becomes unusable.
        Finalizing here would seal a partial store as a valid, hash-stamped
        scene (the root ``type='scene'`` attr is written up front), silently
        corrupting downstream consumers. The marker is stamped only when the
        store was not already finalized — if the body finalized explicitly
        (e.g. ``Scene.to_zarr()``) and an unrelated exception then raised, the
        complete store is left untouched. The marker write is best-effort so it
        can never mask the original exception. The temporary directory (if any)
        is cleaned up on every path.
        """
        try:
            if exc_type is not None and not self._is_finalized:
                # An exception is propagating and the store is only partially
                # written: do NOT finalize, mark it incomplete instead.
                if self._archive_path is not None:
                    self._archive_finalize_failed = True
                try:
                    self.store.attrs["incomplete"] = True
                    if self._archive_path is not None:
                        aprint(
                            "⚠️ Build errored — discarding incomplete archive "
                            f"staging for {self._archive_path}"
                        )
                    else:
                        aprint(
                            "⚠️ Build errored — leaving the store unfinalized "
                            "and marked incomplete"
                        )
                except BaseException:
                    # Best-effort marker: never raise a new exception that
                    # would mask the one already propagating out of the with
                    # block (a second Ctrl-C is a BaseException, not Exception).
                    pass
            elif not self._is_finalized and not self._archive_finalize_failed:
                # Clean exit: finalize. If finalize() itself fails it leaves a
                # half-finalized store; mark it incomplete so it is rejected,
                # then re-raise the original finalize error.
                try:
                    self.finalize()
                except BaseException:
                    try:
                        self.store.attrs["incomplete"] = True
                    except BaseException:
                        pass
                    raise
        finally:
            self._cleanup_archive_staging()
            # Clean up temporary directory if used (every path).
            if self._tmpdir is not None:
                self._tmpdir.cleanup()

    def _cleanup_archive_staging(self) -> None:
        """Remove compiler-owned archive staging paths, if any."""
        if self._archive_artifact_path is not None:
            try:
                self._archive_artifact_path.unlink(missing_ok=True)
            except OSError:
                pass
        if self._archive_path is not None:
            shutil.rmtree(self._store_path, ignore_errors=True)

    def _publish_archive(self) -> None:
        """Package the finalized directory store and atomically publish it."""
        if self._archive_path is None or self._archive_artifact_path is None:
            return
        from .optimise import _package

        try:
            aprint(f"📦 Packaging scene archive at {self._archive_path}")
            _package(self._store_path, self._archive_artifact_path)
            os.replace(self._archive_artifact_path, self._archive_path)
        finally:
            try:
                self._archive_artifact_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _check_archive_can_finalize(self) -> None:
        """Refuse finalization after archive staging has been discarded."""
        if self._archive_finalize_failed:
            raise ValueError(
                "Cannot finalize archive after its staging was discarded; "
                "create a new LuxarZarrCompiler"
            )

    def _discard_failed_archive_staging(self) -> None:
        """Poison an archive compiler and remove its failed staging store."""
        if self._archive_path is None:
            return
        self._archive_finalize_failed = True
        self._cleanup_archive_staging()

    @arbol_warnings()
    def create_scene(
        self,
        dimensions: Dimensions,
        viewer_config: Optional["ViewerConfig"] = None,
        citation: Optional[Mapping[str, str]] = None,
    ) -> "Scene":
        """Create a scene with this compiler as writer.

        Args:
            dimensions: Dimension specification for the scene (REQUIRED).
                Scene dimensions are the single source of truth for the
                coordinate system and must always be specified.
            viewer_config: Optional viewer configuration hints. Stored in
                the zarr file and read by the viewer at load time as
                scene-specific defaults.
            citation: Optional credit for whoever produced the underlying
                dataset -- ``{"short", "ref"?, "doi"?, "license"?, "url"?}``.
                Stored in the root attributes so it travels with the data.
                ``None`` means there is no external dataset to credit.

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
        scene = Scene(
            writer=self,
            dimensions=dimensions,
            viewer_config=viewer_config,
            citation=citation,
        )
        self._scene = scene  # Store reference for finalize-time hover overlay injection
        aprint("✅ Scene created with progressive writer")

        return scene

    def write_group(
        self, path: NodePath, *, _transform_normalized: bool = False, **attrs: Any
    ) -> None:
        """Create a group in the Zarr store.

        Args:
            path: Path for the group within the store
            _transform_normalized: Internal control parameter (part of the
                writer protocol, never persisted). The Node/Scene API
                pre-normalizes ``transform``/``nd_transform`` (its attrs cache
                must hold the column-major form for the ``transform`` getter)
                and passes True so an already column-major matrix is not
                transposed a second time. Raw compiler-API callers leave the
                default False and go through the normalization gate below
                (issue #678).
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
        # Normalize transform (NumPy→THREE.js column-major, reject NaN/Inf/
        # non-affine) and validate nd_transform BEFORE the group is created,
        # so a bad transform fails fast without leaving a partial node. Only
        # the incoming attrs are normalized; already-stored group attrs are
        # merged below and stay untouched (no double-transpose). Skipped when
        # the Node/Scene API already normalized (see ``_transform_normalized``).
        if not _transform_normalized:
            _prepare_transform_attrs(attrs, self.store)
        # Handle root path
        if path == "/" or path == "":
            group = self.store
        else:
            # Validate every path segment (rejects empty/dot-prefixed/
            # control-char names — the F1/F5 chokepoint) + strip the
            # leading slash.
            path = _validate_node_path(path)
            group = self.store.require_group(path)

        # Resolve a custom colormap (ndarray LUT, or a matplotlib/colorcet
        # name) into a sibling ``colormap_lut`` array on THIS node, exactly as
        # the leaf writers do. A GROUP is now a legitimate place to author a
        # colormap — the viewer composes it root→leaf (#1600) — and an
        # unresolved ndarray would not even serialize into the group's attrs,
        # while an unresolved non-builtin NAME would reach the viewer, which
        # only knows the builtins, and silently fall back to viridis. Leaves
        # come through here too (``Node.__init__`` writes every node's attrs
        # via this method), but their colormap has already been resolved to the
        # ``"custom"`` sentinel by then, which the helper passes through.
        if attrs.get("colormap") is not None:
            self._write_colormap_lut_if_needed(group, attrs)

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
        self._check_not_finalized("delete_group_attr")
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

    def node_exists(self, path: NodePath) -> bool:
        """Return whether a node path exists for an internal rollback guard."""
        self._check_not_finalized("node_exists")
        normalized_path = path.lstrip("/")
        return not normalized_path or normalized_path in self.store

    def delete_node(self, path: NodePath) -> None:
        """Delete a subtree during rollback, without editing the scene graph."""
        self._check_not_finalized("delete_node")
        normalized_path = path.lstrip("/")
        if not normalized_path:
            raise ValueError("Cannot delete the scene root")
        if normalized_path in self.store:
            del self.store[normalized_path]

    def snapshot_rollback_state(self) -> RollbackState:
        """Capture mutable authoring state changed by geometry writes.

        The write-only metadata cache is intentionally excluded: it has no
        readers and cannot affect later output after its subtree is deleted.
        """
        self._check_not_finalized("snapshot_rollback_state")
        scene_bounds = (
            None
            if self._scene_bounds is None
            else {key: list(values) for key, values in self._scene_bounds.items()}
        )
        return (
            scene_bounds,
            frozenset(self._authoring_warnings),
            self._lut_tone_mapping_warned,
            self._encoder.snapshot(),
            (
                (self._scene._has_labels, self._scene._has_image_labels)
                if self._scene is not None
                else None
            ),
        )

    def restore_rollback_state(self, state: RollbackState) -> None:
        """Restore compiler state captured before a rolled-back write."""
        self._check_not_finalized("restore_rollback_state")
        (
            scene_bounds,
            authoring_warnings,
            lut_tone_mapping_warned,
            encoder_state,
            scene_label_state,
        ) = state
        self._scene_bounds = (
            None
            if scene_bounds is None
            else {key: list(values) for key, values in scene_bounds.items()}
        )
        self._authoring_warnings = set(authoring_warnings)
        self._lut_tone_mapping_warned = lut_tone_mapping_warned
        self._encoder.restore(encoder_state)
        if self._scene is not None and scene_label_state is not None:
            self._scene._has_labels, self._scene._has_image_labels = scene_label_state

    @contextmanager
    def transaction(self, path: NodePath) -> Iterator[None]:
        """Roll back writes below ``path`` while preserving the original error."""
        if self._transaction_depth:
            self._transaction_depth += 1
            try:
                yield
            finally:
                self._transaction_depth -= 1
            return

        rollback_state: Optional[RollbackState]
        try:
            rollback_state = self.snapshot_rollback_state()
        except Exception:
            rollback_state = None

        try:
            path_existed = self.node_exists(path)
        except Exception:
            path_existed = True

        self._transaction_depth = 1
        try:
            yield
        except BaseException as error:
            if rollback_state is not None:
                try:
                    self.restore_rollback_state(rollback_state)
                except BaseException as rollback_error:
                    error.add_note(
                        f"Writer state rollback also failed: {rollback_error}"
                    )
            if not path_existed:
                try:
                    self.delete_node(path)
                except BaseException as rollback_error:
                    error.add_note(
                        f"Writer store rollback also failed: {rollback_error}"
                    )
            raise
        finally:
            self._transaction_depth = 0

    @arbol_warnings()
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
        keys: Optional[Sequence[str]] = None,
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
            keys: Optional list of machine-readable strings, one per point.
                Stored as CSR-encoded key_offsets + key_bytes arrays for
                ``link`` / ``copy`` templates to substitute as
                ``{hover_key}``. Independent of ``labels``.
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
            keys=keys,
            **attrs,
        )
        self._metadata_cache[metadata["path"]] = metadata
        return metadata

    @arbol_warnings()
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
        keys: Optional[Sequence[str]] = None,
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
            indices: Vertex-index pairs for indexed lines, as flat ``(2E,)``
                elements or an ``(E, 2)`` pair array. Joint continuity requires
                connected edges to reference the same vertex row.
            line_type: Type of line connectivity. Use ``polyline`` for one chain
                or ``indexed`` for multiple chains / graph topology.
            labels: Optional list of strings, one per vertex. Stored as CSR-encoded
                label_offsets + label_bytes arrays for hover tooltips.
            image_labels: Optional per-element images for hover thumbnails.
            keys: Optional list of machine-readable strings, one per vertex.
                Stored as CSR-encoded key_offsets + key_bytes arrays for
                ``link`` / ``copy`` templates to substitute as
                ``{hover_key}``. Independent of ``labels``.
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
            keys=keys,
            **attrs,
        )
        self._metadata_cache[path.lstrip("/")] = metadata
        return metadata

    def write_mesh(  # type: ignore[override]
        self,
        path: NodePath,
        vertices: NDArray[np.float32],
        faces: NDArray[np.uint32],
        normals: Optional[NDArray[np.float32]] = None,
        normal_dims: Optional[Sequence[int]] = None,
        colors: Optional[
            Union[NDArray[np.float32], List[float], Tuple[float, ...]]
        ] = None,
        scalars: Optional[Union[NDArray[np.float32], float]] = None,
        uvs: Optional[NDArray[np.float32]] = None,
        texture: Optional[NDArray[Any]] = None,
        texture_encoding: str = "raw",
        texture_width: Optional[int] = None,
        texture_height: Optional[int] = None,
        texture_channels: Optional[int] = None,
        texture_color_space: str = "srgb",
        texture_ktx2_mode: str = "uastc",
        texture_ktx2_quality: Optional[int] = None,
        texture_ktx2_rdo_l: Optional[float] = None,
        texture_ktx2_zcmp: Optional[int] = None,
        shading: Optional[str] = None,
        double_sided: bool = True,
        labels: Optional["Sequence[str]"] = None,
        image_labels: Optional[Any] = None,
        keys: Optional[Sequence[str]] = None,
        **attrs: Any,
    ) -> dict[str, Any]:
        """Write a triangle mesh to Zarr.

        Mesh is the surface geometry type: ``vertices`` in nD plus a ``faces``
        triangle-index array. Unlike Points / Lines / GSplats it carries no
        per-element size (a triangle's extent comes from its own vertices), and
        it has no spatial index — the loader is whole-node. This call writes one
        leaf; LOD is layered on by the caller, not by this method — see
        :meth:`write_mesh_multi_lod` (additive reveal levels) below.

        Scalar convenience: ``colors`` accepts a broadcast RGB(A) tuple/list, and
        ``scalars`` a single value, exactly as the sibling writers do.

        Args:
            path: Path for the mesh within the store.
            vertices: Vertex positions of shape ``(V, D)``.
            faces: Triangle vertex indices, ``(F, 3)`` or flat ``(3F,)``. Wound
                counter-clockwise as seen with the mesh's authored spatial triple
                in ascending index order.
            normals: Optional per-vertex normals, shape ``(V, 3)``. Requires
                ``normal_dims``.
            normal_dims: The three dimension indices the normals describe —
                required with ``normals`` and rejected without them. Normals are a
                display-space quantity, so the store must say which three
                dimensions they belong to; an implicit "first three" is wrong for
                any mesh whose leading dimension is not spatial.
            colors: Colors — array ``(V, 3|4)``, RGB(A) tuple/list, or None. A 4th
                component is per-vertex opacity.
            scalars: Scalars for colormap lookup — array ``(V,)``, scalar, or None.
            shading: ``"smooth"``, ``"flat"``, or unlit ``"none"``. Defaults to
                ``"smooth"`` when normals are supplied, else ``"flat"``. An explicit
                value is stored as given: ``"flat"`` renders a faceted surface even
                with normals present, ``"smooth"`` without normals falls back to
                derived flat normals at render time, and ``"none"`` computes no
                lighting normal.
            double_sided: Whether back faces render. ``True`` by default.
            labels: Optional per-vertex strings for hover tooltips (CSR-encoded).
            image_labels: Optional per-vertex images for hover thumbnails.
            keys: Optional list of machine-readable strings, one per vertex.
                Stored as CSR-encoded key_offsets + key_bytes arrays for
                ``link`` / ``copy`` templates to substitute as
                ``{hover_key}``. Independent of ``labels``.
            uvs: Optional ``(V, 2)`` per-vertex texture coordinates. Required
                with ``texture`` and refused without it. Values outside
                ``[0, 1]`` are legal and tile under ``texture_wrap="repeat"``.
            texture: Optional base-colour image. ``(H, W, C)`` array under
                ``texture_encoding="raw"`` or ``"ktx2"``, else a 1-D ``uint8``
                array of encoded bytes. Mutually exclusive with ``colors`` and
                ``colormap`` — a mesh has one base-colour source.
            texture_encoding: ``raw`` | ``png`` | ``webp`` | ``jpeg`` | ``ktx2``.
                KTX2 accepts uint8 RGB/RGBA input and requires the Khronos
                ``toktx`` executable, version 4.1.0 or newer. HDR requires
                ``raw``.
            texture_ktx2_mode: ``uastc`` (default) or ``etc1s``.
            texture_ktx2_quality: Codec quality; defaults to 2 for UASTC and 128
                for ETC1S.
            texture_ktx2_rdo_l: UASTC RDO lambda in [0.001, 10.0]; defaults to
                0.25. Lower values preserve more quality and produce larger files;
                0 disables RDO while retaining zstd compression.
            texture_ktx2_zcmp: UASTC zstd level in [1, 22]; defaults to 9.
            texture_width: Declared width. Required for encoded payloads, where
                it cannot be read without decoding; read off the array for
                ``raw`` or ``ktx2``, and refused if it disagrees.
            texture_height: Declared height. Same contract as ``texture_width``.
            texture_channels: Declared channels — 1, 3 or 4. Same contract.
            texture_color_space: ``srgb`` (default) or ``linear``. An ordinary
                PNG/JPEG is sRGB-encoded; declaring it wrong gives a subtly
                over-dark or washed-out surface rather than an obvious failure.
                HDR raw textures must be declared ``linear``.
            **attrs: Additional attributes.

        Returns:
            Metadata dictionary about the written mesh.
        """
        self._check_not_finalized("write_mesh")
        # Keyword-forwarded, deliberately. This used to pass positionally, and
        # inserting a parameter into `_write_mesh_impl`'s signature silently
        # shifted every argument after it — mypy caught it, but only because the
        # shifted types happened to disagree. Keywords make the forward
        # insertion-order-proof.
        metadata = _write_mesh_impl(
            self._make_geometry_ctx(),
            path,
            vertices,
            faces,
            normals=normals,
            normal_dims=normal_dims,
            colors=colors,
            scalars=scalars,
            uvs=uvs,
            texture=texture,
            texture_encoding=texture_encoding,
            texture_width=texture_width,
            texture_height=texture_height,
            texture_channels=texture_channels,
            texture_color_space=texture_color_space,
            texture_ktx2_mode=texture_ktx2_mode,
            texture_ktx2_quality=texture_ktx2_quality,
            texture_ktx2_rdo_l=texture_ktx2_rdo_l,
            texture_ktx2_zcmp=texture_ktx2_zcmp,
            shading=shading,
            double_sided=double_sided,
            labels=labels,
            image_labels=image_labels,
            keys=keys,
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

    @arbol_warnings()
    def write_points_multi_lod(
        self,
        path: NodePath,
        levels: List[Dict[str, Any]],
        *,
        extend_to_all: Optional[List[str]] = None,
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

        **String channels**: per-element ``labels`` and ``keys`` are each written
        as one CSR pair on the parent (stamping ``has_labels`` / ``has_keys``);
        the ``additive_<i>`` subgroups carry neither. Each parent CSR's index space is
        the committed union — the concatenation of the levels in
        ``additive_0 … additive_{n-1}`` order, each level in its own stored
        (spatially reordered) order — because the viewer's progressive loader
        concatenates loaded levels into one buffer. Each channel is independently
        all-or-nothing across the ladder.

        Args:
            path: Path for the points node within the store.
            levels: List of per-level dicts with keys ``positions`` /
                ``colors`` / ``radii`` / ``sharpness`` / ``scalars`` /
                ``labels`` / ``keys``. ``positions`` is required; others may be
                ``None``. Each string channel must be present on every level or
                on none.
            extend_to_all: Forwarded to each per-level write. Already-resolved
                dimension NAMES — the caller expands the ``"all"`` sentinel,
                which must never reach disk (it is stamped verbatim here, onto
                the parent group and every sub-LOD).
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
        # Normalize/validate the parent transform + nd_transform before the
        # group is created. The per-level subgroups get their own default
        # attrs (via write_points), NOT these parent attrs, so the transform is
        # normalized exactly once here — safe despite prepare_transform_attrs
        # being non-idempotent.
        _prepare_transform_attrs(attrs, self.store)

        # Labels ride on the PARENT as one union CSR, so the ladder's label
        # situation is resolved here — inside the fail-fast block, ABOVE
        # require_group, because this gate is pure and a rejected ladder must
        # not leave an empty node behind.
        labelled = validate_ladder_labels(levels, "positions")
        # Keys obey the same all-or-nothing rule across the ladder, and land
        # in the same union CSR on the parent (#1917).
        keyed = validate_ladder_labels(levels, "positions", channel="keys")

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
        level_sort_orders: List[Optional[np.ndarray]] = []
        for i, lvl in enumerate(levels):
            level_path = f"{path}/additive_{i}"
            level_meta = self.write_points(
                level_path,
                lvl["positions"],
                colors=lvl.get("colors"),
                radii=lvl.get("radii"),
                sharpness=lvl.get("sharpness"),
                scalars=lvl.get("scalars"),
                # Labels live on the parent as one union CSR — never per level.
                labels=None,
                **({"lod_stats": lvl["lod_stats"]} if lvl.get("lod_stats") else {}),
                **({"extend_to_all": extend_to_all} if extend_to_all else {}),
                _skip_scene_bounds=True,
                # Plain keyword rather than a conditional `**{...}` unpack:
                # `record_forwarded_sort_order` treats False exactly as absent,
                # and a `dict[str, bool]` unpack is checked against every typed
                # parameter it could bind to — which now includes `keys`.
                _skip_element_cap_warning=True,
                _return_sort_order=labelled or keyed,
            )
            # POP, not read: the permutation is only needed to build the union
            # CSR, and the metadata dict lands in ``self._metadata_cache``.
            if labelled or keyed:
                level_sort_orders.append(level_meta.pop("sort_order", None))
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
        warn_if_over_element_cap("points", n_points_total, group.name)
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

        # The ladder's union label CSR, written AFTER the attr stamps above so
        # no attr update can clobber the ``has_labels`` flag
        # ``write_labels_csr`` stamps on the parent.
        if labelled:
            write_ladder_union_labels_csr(
                group,
                [lvl["labels"] for lvl in levels],
                level_sort_orders,
                n_points_total,
                self.compressor,
            )
            metadata["has_labels"] = True
        if keyed:
            write_ladder_union_labels_csr(
                group,
                [lvl["keys"] for lvl in levels],
                level_sort_orders,
                n_points_total,
                self.compressor,
                channel="keys",
            )
            metadata["has_keys"] = True

        self._metadata_cache[path] = metadata
        aprint(
            f"✅ Multi-LOD Points written to {path} ({n_levels} levels, "
            f"{n_points_total:,} points total)"
        )
        return metadata

    @arbol_warnings()
    def write_lines_multi_lod(
        self,
        path: NodePath,
        levels: List[Dict[str, Any]],
        *,
        extend_to_all: Optional[List[str]] = None,
        **attrs: Any,
    ) -> Dict[str, Any]:
        """Write multi-additive-LOD Lines.

        Mirrors :meth:`write_points_multi_lod`. Each level dict carries
        ``vertices`` + ``widths`` + ``colors`` / ``sharpness`` /
        ``scalars`` / ``labels`` / ``keys`` + ``segments`` (local index pairs into
        that level's vertices) + ``n_polylines``. Each subgroup is
        written via :meth:`write_lines` with ``line_type='indexed'``
        and the local segment indices.

        **String channels** (per-VERTEX for Lines) are written as one CSR pair
        per present ``labels`` / ``keys`` channel on the parent, which stamps
        ``has_labels`` / ``has_keys``. Each pair describes the committed union:
        the concatenation of the levels in
        ``additive_0 … additive_{n-1}`` order, each level in its own stored
        (spatially reordered) order. The ``additive_<i>`` subgroups carry neither.
        Each channel is independently all-or-nothing across the ladder.
        """
        self._check_not_finalized("write_lines_multi_lod")

        if not levels:
            raise ValueError("levels must contain at least one LOD level")

        # Fail fast on invalid render attrs BEFORE creating the parent group.
        _validate_render_attrs(attrs)
        # Normalize/validate the parent transform + nd_transform before the
        # group is created. The per-level subgroups get their own default
        # attrs (via write_lines), NOT these parent attrs, so the transform is
        # normalized exactly once here — safe despite prepare_transform_attrs
        # being non-idempotent.
        _prepare_transform_attrs(attrs, self.store)

        # Labels ride on the PARENT as one per-vertex union CSR, so the ladder's
        # label situation is resolved here — inside the fail-fast block, ABOVE
        # require_group, because this gate is pure and a rejected ladder must
        # not leave an empty node behind.
        labelled = validate_ladder_labels(levels, "vertices")
        # Keys obey the same all-or-nothing rule across the ladder, and land
        # in the same union CSR on the parent (#1917).
        keyed = validate_ladder_labels(levels, "vertices", channel="keys")

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
        level_sort_orders: List[Optional[np.ndarray]] = []
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
                # Labels live on the parent as one union CSR — never per level.
                labels=None,
                image_labels=None,
                indices=flat_indices,
                line_type="indexed" if flat_indices is not None else "polyline",
                **({"lod_stats": lvl["lod_stats"]} if lvl.get("lod_stats") else {}),
                **({"extend_to_all": extend_to_all} if extend_to_all else {}),
                _skip_scene_bounds=True,
                # Plain keyword rather than a conditional `**{...}` unpack:
                # `record_forwarded_sort_order` treats False exactly as absent,
                # and a `dict[str, bool]` unpack is checked against every typed
                # parameter it could bind to — which now includes `keys`.
                _skip_element_cap_warning=True,
                _return_sort_order=labelled or keyed,
            )
            # POP, not read: the permutation is only needed to build the union
            # CSR, and the metadata dict lands in ``self._metadata_cache``.
            if labelled or keyed:
                level_sort_orders.append(level_meta.pop("sort_order", None))
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
        group.attrs["n_additive_sublods"] = n_levels
        group.attrs["position_bounds"] = global_bounds
        warn_if_over_element_cap("lines", n_segments_total, group.name)
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

        # The ladder's union label CSR (per-VERTEX), written AFTER the attr
        # stamps above so no attr update can clobber the ``has_labels`` flag
        # ``write_labels_csr`` stamps on the parent.
        if labelled:
            write_ladder_union_labels_csr(
                group,
                [lvl["labels"] for lvl in levels],
                level_sort_orders,
                n_vertices_total,
                self.compressor,
            )
            metadata["has_labels"] = True
        if keyed:
            write_ladder_union_labels_csr(
                group,
                [lvl["keys"] for lvl in levels],
                level_sort_orders,
                n_vertices_total,
                self.compressor,
                channel="keys",
            )
            metadata["has_keys"] = True

        self._metadata_cache[path] = metadata
        aprint(
            f"✅ Multi-LOD Lines written to {path} ({n_levels} levels, "
            f"{n_vertices_total:,} vertices in {n_polylines_total:,} "
            f"polylines)"
        )
        return metadata

    @arbol_warnings()
    def write_mesh_multi_lod(
        self,
        path: NodePath,
        levels: List[Dict[str, Any]],
        *,
        extend_to_all: Optional[List[str]] = None,
        **attrs: Any,
    ) -> Dict[str, Any]:
        """Write multi-additive-LOD Mesh: parent node + ``additive_<i>/`` subgroups.

        Mirrors :meth:`write_points_multi_lod` in shape — a parent carrying the
        totals, the global ``position_bounds``, ``n_additive_sublods`` and the
        compositing attrs, with one fully-formed leaf per level underneath. Each
        level dict carries what :meth:`write_mesh` needs: ``vertices`` + ``faces``
        (already re-indexed into that level's OWN vertex table by
        :func:`luxar.mesh.split.split_mesh_by_faces`) plus optional ``normals`` /
        ``normal_dims`` / ``colors`` / ``scalars`` / ``shading`` /
        ``double_sided`` / ``_scalar_data_range``, and ``lod_stats``.

        A mesh ladder is a REVEAL: the levels are a partition of the faces,
        coarsest (innermost shell) first, and the viewer forms level *i* by
        concatenating levels 0..i. Level counts are therefore FACE counts, and the
        parent's ``n_vertices`` / ``n_faces`` are the sums over levels — the
        vertex sum exceeds the source vertex count by the shell-boundary
        duplication the re-indexing costs (see
        :func:`luxar.mesh.split.duplication_factor`).

        **No labels, and deliberately no union label CSR.** The three sibling
        ladders write ONE CSR pair on the parent spanning the levels, because
        their elements are independent rows and the concatenation of the levels is
        a well-defined index space. A mesh level RE-INDEXES its own vertices, so a
        single source vertex on a shell boundary maps to a slot in several levels
        (and a vertex no level's faces reference maps to none): the union index
        space is ill-defined, and any CSR written over it would pair labels with
        the wrong vertices. So the parent carries no ``has_labels`` / ``has_keys``
        and a level carrying either is refused outright rather than silently
        dropped.

        Args:
            path: Path for the mesh node within the store.
            levels: One dict per additive level, coarsest first. ``vertices`` and
                ``faces`` are required; every other key is optional.
            extend_to_all: Forwarded to each per-level write. Already-resolved
                dimension NAMES — the caller expands the ``"all"`` sentinel, which
                must never reach disk (it is stamped verbatim here, onto the parent
                group and every sub-LOD).
            **attrs: Additional parent-node attrs (compositing, colormap, ...).

        Returns:
            Aggregate metadata dict with ``type`` / ``n_vertices`` / ``n_faces`` /
            ``n_additive_sublods`` / ``position_bounds`` / ``levels``.

        Raises:
            ValueError: If ``levels`` is empty, an attr is invalid, or any level
                carries ``labels`` or ``keys``.
        """
        self._check_not_finalized("write_mesh_multi_lod")

        if not levels:
            raise ValueError("levels must contain at least one LOD level")

        # Fail fast on invalid render attrs BEFORE creating the parent group.
        _validate_render_attrs(attrs)
        # Normalize/validate the parent transform + nd_transform before the group
        # is created. The per-level subgroups get their own default attrs (via
        # write_mesh), NOT these parent attrs, so the transform is normalized
        # exactly once here — safe despite prepare_transform_attrs being
        # non-idempotent.
        _prepare_transform_attrs(attrs, self.store)

        # The labels refusal sits in the fail-fast block, ABOVE require_group,
        # for the same reason the sibling writers resolve their label situation
        # there: the gate is pure, and a rejected ladder must not leave an empty
        # node behind. See the docstring for why a union CSR is impossible here.
        # Both per-element string channels, for the same structural reason: the
        # refusal is about the union INDEX SPACE, not about what the strings
        # mean, so `keys` cannot be stored here any more than `labels` can.
        for channel in ("labels", "keys"):
            offending = [
                i for i, lvl in enumerate(levels) if lvl.get(channel) is not None
            ]
            if offending:
                raise ValueError(
                    f"write_mesh_multi_lod: level(s) {offending} carry '{channel}', "
                    "which a mesh reveal ladder cannot store. The sibling ladders put "
                    f"one union {channel} CSR on the parent spanning the levels, but "
                    "each mesh level re-indexes its own vertices — one source vertex "
                    "maps to a slot in several levels — so that union index space is "
                    "ill-defined and a CSR over it would pair strings with the wrong "
                    "vertices. Use substitutive_lod= (whose finest child is the "
                    "original surface and carries the channel) or partition= (which "
                    "splits the CSR per part), or write a plain leaf."
                )

        # Validate every path segment (rejects empty/dot-prefixed names — the
        # F1/F5 chokepoint) + strip the leading slash.
        path = _validate_node_path(path)
        group = self.store.require_group(path)
        n_levels = len(levels)

        # Global bounds + totals over every level's vertices. Concatenated rather
        # than taken from the source array, because the parent must describe what
        # is actually on disk: the levels' gathered tables together hold the
        # boundary duplicates, and a vertex referenced by no face at all is not in
        # any of them.
        all_vertices = np.concatenate([L["vertices"] for L in levels], axis=0)
        n_vertices_total = int(all_vertices.shape[0])
        global_bounds = self._compute_position_bounds(all_vertices)

        level_metas: List[Dict[str, Any]] = []
        for i, lvl in enumerate(levels):
            level_path = f"{path}/additive_{i}"
            level_meta = self.write_mesh(
                level_path,
                lvl["vertices"],
                lvl["faces"],
                normals=lvl.get("normals"),
                normal_dims=lvl.get("normal_dims"),
                colors=lvl.get("colors"),
                scalars=lvl.get("scalars"),
                shading=lvl.get("shading"),
                double_sided=bool(lvl.get("double_sided", True)),
                # Neither label channel rides a mesh ladder: text labels are
                # refused above, and image_labels is a whole-node mapping with no
                # per-level meaning (the adder degrades to a flat leaf for both).
                labels=None,
                image_labels=None,
                **(
                    {"_scalar_data_range": lvl["_scalar_data_range"]}
                    if lvl.get("_scalar_data_range") is not None
                    else {}
                ),
                **({"lod_stats": lvl["lod_stats"]} if lvl.get("lod_stats") else {}),
                # `Dict[str, Any]`, not the inferred `dict[str, list[str]]`: a
                # splat is checked against every parameter it could bind to,
                # and `write_mesh` now has typed optional params (the texture
                # ones) that a narrowly-typed splat cannot satisfy. The dict
                # really is heterogeneous attrs — the annotation says so, rather
                # than widening a signature to suit an inference artefact.
                **(
                    cast(Dict[str, Any], {"extend_to_all": extend_to_all})
                    if extend_to_all
                    else {}
                ),
                _skip_scene_bounds=True,
            )
            level_metas.append(level_meta)

        # Summed from what the levels actually wrote, so the parent advertises a
        # face total — parity with flat `write_mesh` and with the sibling ladders'
        # `n_points` / `n_segments`. Without it the viewer's scene-graph converter
        # reads `n_faces` as undefined.
        n_faces_total = sum(int(m.get("n_faces", 0)) for m in level_metas)

        # Resolve a custom colormap (ndarray / matplotlib name) to a colormap_lut
        # dataset + colormap='custom' before the JSON attr dump — same contract as
        # the flat writer.
        self._write_colormap_lut_if_needed(group, attrs)
        group.attrs.update(attrs)
        group.attrs["type"] = "mesh"
        group.attrs["n_vertices"] = n_vertices_total
        group.attrs["n_faces"] = n_faces_total
        group.attrs["n_additive_sublods"] = n_levels
        group.attrs["position_bounds"] = global_bounds
        if extend_to_all:
            group.attrs["extend_to_all"] = extend_to_all

        # The parent must DESCRIBE the surface, not merely count it.
        #
        # A ladder's parent IS the node: it is what the scene graph lists, what the
        # viewer constructs the drawable from, and what the Layers panel reads —
        # the `additive_<i>` subgroups are pruned from the graph entirely. So it
        # needs the same descriptive attrs a flat `write_mesh` stamps. Every level
        # is a face-partition of ONE source mesh written through that same
        # function, so these are level-invariant and level 0 is authoritative.
        #
        # This is stamped from the level metadata rather than left to the caller
        # because omitting it is invisible to a round-trip test and loud on screen.
        # The viewer fixes a mesh geometry's ATTRIBUTE SET once, at node creation,
        # from `has_normals` / `has_scalars` on the parent — it may never add an
        # attribute to a live geometry, because the WebGPU backend bakes the vertex
        # layout into its pipeline at first draw. A parent without `has_normals`
        # therefore binds no `normal` attribute, the levels' normals can never
        # reach the GPU however many arrive, and the node renders faceted and
        # forced double-sided (no winding frame to decide against) right beside an
        # identical unladdered mesh that renders smooth and single-sided. Measured
        # in the viewer, not reasoned about.
        for key in (
            "ndim",
            "has_normals",
            "normal_dims",
            "has_colors",
            "has_scalars",
            # Both always False on this route — `texture=` is refused with
            # `additive_lod=` (the image would be stored once per reveal shell)
            # and `uvs=` is refused without a texture. They are carried anyway,
            # because the ratchet this list exists for compares the parent's key
            # SET against a flat write's: a descriptive attr the flat path stamps
            # and the parent omits is exactly the class of bug the viewer's
            # fixed-attribute-set rule turns from invisible into on-screen.
            "has_uvs",
            "has_texture",
            "shading",
            "double_sided",
            "ordering",
        ):
            if key in level_metas[0]:
                group.attrs[key] = level_metas[0][key]

        # The colour/scalar windows, taken as the UNION over the levels.
        #
        # LOAD-BEARING for colours, a no-op for scalars — the two windows are not
        # alike, and it is worth saying which is which. `scalar_data_range` is the
        # AUTHORED window, passed down to every level unchanged, so its union
        # equals any one level's. `color_data_range` is MEASURED by the encoder
        # from each level's own vertex subset, so the levels genuinely disagree: a
        # radius ramp over three levels gives [0.00, 4.22], [2.98, 5.73],
        # [4.71, 8.00], and only their union [0.00, 8.00] matches what a flat
        # write of the same mesh stamps. Copying level 0's would set the node's
        # colour window from whatever landed in the innermost shell, and the
        # surface would recolour as the reveal completed. Costs one attrs read per
        # level, once, at authoring time.
        for key in ("color_data_range", "scalar_data_range"):
            ranges = [
                self.store.require_group(f"{path}/additive_{i}").attrs.get(key)
                for i in range(n_levels)
            ]
            present = [r for r in ranges if r is not None]
            if present:
                group.attrs[key] = [
                    min(float(r[0]) for r in present),
                    max(float(r[1]) for r in present),
                ]

        # Aggregate the parent's bbox into scene-bounds once (every level write
        # above was told to skip it).
        self._update_scene_bounds(global_bounds)

        metadata: Dict[str, Any] = {
            "type": "mesh",
            "n_vertices": n_vertices_total,
            "n_faces": n_faces_total,
            "n_additive_sublods": n_levels,
            "position_bounds": global_bounds,
            "levels": level_metas,
        }
        # No `has_labels` and no `write_ladder_union_labels_csr` call: see the
        # docstring — the ladder's union vertex index space does not exist.

        self._metadata_cache[path] = metadata
        aprint(
            f"✅ Multi-LOD Mesh written to {path} ({n_levels} levels, "
            f"{n_vertices_total:,} vertices / {n_faces_total:,} faces total)"
        )
        return metadata

    # ── GSplats public write methods ───────────────────────────

    @arbol_warnings()
    def write_gsplats(  # type: ignore[override]
        self,
        path: NodePath,
        centers: NDArray[np.float32],
        amplitudes: Union[NDArray[np.float32], float],
        cholesky_factors: NDArray[np.float32],
        colors: Optional[
            Union[NDArray[np.float32], List[float], Tuple[float, ...]]
        ] = None,
        label_ids: Optional[Union[np.ndarray[Any, Any], Sequence[int]]] = None,
        label_vocabulary: Optional[dict[int, str]] = None,
        labels: Optional["Sequence[str]"] = None,
        image_labels: Optional[Any] = None,
        keys: Optional[Sequence[str]] = None,
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
            label_ids: Optional non-negative integer class id per splat.
            label_vocabulary: Explicit mapping from stored class ids to names.
            labels: Optional list of strings, one per splat. Stored as CSR-encoded
                label_offsets + label_bytes arrays for hover tooltips.
            image_labels: Optional per-element images for hover thumbnails.
            keys: Optional list of machine-readable strings, one per splat.
                Stored as CSR-encoded key_offsets + key_bytes arrays for
                ``link`` / ``copy`` templates to substitute as
                ``{hover_key}``. Independent of ``labels``.
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
            colors=colors,
            label_ids=label_ids,
            label_vocabulary=label_vocabulary,
            labels=labels,
            image_labels=image_labels,
            keys=keys,
            **attrs,
        )
        self._metadata_cache[path.lstrip("/")] = metadata
        return metadata

    @arbol_warnings()
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
        (no uniform-Cholesky / scalar-amplitude / labels / keys — those leaf-only
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
            maxshape: ACCEPTED AND IGNORED. An h5py-compatibility argument that
                zarr has never enforced — a zarr array has no pre-declared
                ceiling, `.resize()` simply works — so this has always been
                inert. zarr 2 accepted and dropped it; zarr 3 removed the kwarg
                outright, so it is no longer forwarded at all. Kept on the
                signature because callers pass it and removing it would be a
                gratuitous break, but it constrains nothing.
            shape: Initial shape
            chunks: Chunking configuration

        Returns:
            Zarr dataset handle
        """
        self._check_not_finalized("create_resizable_dataset")

        # Validate every path segment (rejects empty/dot-prefixed names + the
        # reserved 'overlays' root — the F1/F5 chokepoint) + strip the leading
        # slash. Returns the stripped path, so the rsplit below is unchanged.
        path = _validate_node_path(path)

        # Parse parent group and dataset name
        parts = path.rsplit("/", 1)
        if len(parts) == 2:
            group_path, dataset_name = parts
            group = self.store.require_group(group_path)
        else:
            dataset_name = path
            group = self.store

        # Create resizable dataset.
        #
        # `maxshape` is deliberately NOT forwarded. It was an h5py-compatibility
        # kwarg on zarr 2's `create_dataset` that zarr ignored — zarr arrays are
        # always resizable via `.resize()`, with no pre-declared ceiling — and
        # zarr 3 removed it along with `create_dataset` itself. It stays on THIS
        # function's signature because it is part of Luxar's own documented API
        # and callers pass it, but it has never constrained anything.
        dataset = create_array(
            group,
            dataset_name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=resolve_compressor(self.compressor, dtype),
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

    def _prune_childless_wrappers(self, store: zarr.Group) -> None:
        prune_childless_wrappers(store)

    def _finalize_lod_position_bounds(self, store: zarr.Group) -> None:
        finalize_lod_position_bounds(store)

    def _finalize_lod_display_types(self, store: zarr.Group) -> None:
        finalize_lod_display_types(store)

    def _harmonize_gsplat_amplitude_windows(self, store: zarr.Group) -> None:
        harmonize_gsplat_amplitude_windows(store)

    def _warn_one_part_partition_anchors(self, store: zarr.Group) -> None:
        """Report a fills-screen LOD ladder under a ONE-part kind=partition.

        Read-only: finalize is simply the first place the final sibling count is
        visible. See ``warn_one_part_partition_anchors`` for why it warns rather
        than re-anchors.
        """
        warn_one_part_partition_anchors(store)

    def _warn_overlapping_blending(
        self, store: zarr.Group, world_leaves: list[WorldBoundsLeaf]
    ) -> None:
        warn_overlapping_blending(store, world_leaves)

    def _expand_bounds_with_transforms(
        self, store: zarr.Group, world_leaves: list[WorldBoundsLeaf]
    ) -> None:
        self._scene_bounds = expand_bounds_with_transforms(
            store, self._scene_bounds, world_leaves
        )

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

    def _claim_authoring_warning(self, kind: str, path: str) -> bool:
        """Return True once for each (geometry kind, logical node path) pair.

        Keyed by kind as well as path so two geometry types' lints on the same
        logical node stay independent — one type claiming the path must not
        silence another's.
        """
        key = (kind, path)
        if key in self._authoring_warnings:
            return False
        self._authoring_warnings.add(key)
        return True

    def _make_geometry_ctx(self) -> GeometryWriteCtx:
        """Build the narrow context for the extracted geometry write pipelines.

        Bundles the dataset/ordering configs + compressor with bound-method hooks
        for scene bounds, colormap warnings, and per-type authoring warnings.
        """
        return GeometryWriteCtx(
            store=self.store,
            dataset_ctx=self._make_dataset_ctx(),
            ordering_ctx=self._make_ordering_ctx(),
            compressor=self.compressor,
            update_scene_bounds=self._update_scene_bounds,
            write_colormap_lut=self._write_colormap_lut_if_needed,
            claim_authoring_warning=self._claim_authoring_warning,
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

    @arbol_warnings()
    def finalize(self) -> None:
        """Finalize the Zarr store with metadata consolidation."""
        if self._is_finalized:
            return
        self._check_archive_can_finalize()

        try:
            # A prior aborted attempt may have marked the store incomplete; a
            # real finalize supersedes that. Clearing it lives INSIDE the outer
            # boundary, and a failed clear must FAIL the finalize (the handler
            # re-stamps and raises): a "successful" finalize that left the
            # marker behind would set _is_finalized and produce a valid store
            # that LuxarScene.load permanently rejects, with no way to retry.
            if "incomplete" in self.store.attrs:
                del self.store.attrs["incomplete"]

            # Auto-inject default hover overlay if labels exist but no hover
            # overlay defined. Inside the boundary: if it raises, the store is
            # already partial and must be marked incomplete.
            if self._scene is not None:
                self._scene._auto_inject_hover_overlay()

            aprint("🔧 Finalizing Zarr store...")

            # Close the store to ensure all data is written
            zarr_close(self.store)

            # Re-open the store to consolidate metadata
            # This ensures all groups and datasets are properly written to disk
            store = zarr_open_group(self._store_path, mode="r+")

            # Store scene-level position bounds (union of all node bounds)
            if self._scene_bounds is not None:
                store.attrs["position_bounds"] = self._scene_bounds
                aprint(
                    f"📦 Scene bounds (local): min={self._scene_bounds['min']}, max={self._scene_bounds['max']}"
                )

            # Expand bounds into world space: 4x4 spatial transforms on the
            # displayed dims + nd_transforms on the non-displayed dims.
            world_leaves = collect_world_bounds(store)
            self._expand_bounds_with_transforms(store, world_leaves)
            if self._scene_bounds is not None:
                aprint(
                    f"🌍 Scene bounds (world): min={self._scene_bounds['min']}, "
                    f"max={self._scene_bounds['max']}"
                )

            # Validate discrete dimension ranges against actual data
            self._validate_discrete_dimension_ranges(store)

            # A caller may deliberately catch a child-add refusal after
            # creating its wrapper in a separate successful call. Remove that
            # now-empty wrapper, including empty wrapper chains, rather than
            # making the otherwise recoverable compile impossible to publish.
            self._prune_childless_wrappers(store)

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

            # Put every node of each gsplat structure on ONE colormap window
            # (per-LOD-level scaled by the mass-weighted amplitude ratio,
            # verbatim across partition parts). Runs before the content hashes
            # so the stamped hashes cover the corrected attrs.
            self._harmonize_gsplat_amplitude_windows(store)

            # Report — never rewrite — a per-TILE (fills-screen) LOD ladder
            # sitting under a ONE-part kind=partition. The scene adders derive
            # part 0's anchor before part 1 exists, so this is the first point
            # where the final sibling count is visible.
            self._warn_one_part_partition_anchors(store)

            # Report blend-state combinations whose overlapping world boxes
            # cannot be rendered unambiguously. Read-only: author intent and
            # viewer defaults remain unchanged.
            self._warn_overlapping_blending(store, world_leaves)

            # Now consolidate metadata with all data present
            consolidate(store)

            # Compute content hashes (post-order: children before parents)
            self._compute_content_hashes(store)

            # Re-consolidate to include hashes in .zmetadata
            consolidate(store)

            # Close the store again
            zarr_close(store)

            self._publish_archive()

        except BaseException as e:
            # ANY failure (marker clearing, hover injection, or a finalize
            # phase) leaves the store half-finalized; mark it so
            # LuxarScene.load rejects it. A direct finalize() call, e.g. via
            # Scene.to_zarr(), does not go through __exit__'s failure handling,
            # so this is the only safeguard. Covers BaseException too so a
            # KeyboardInterrupt mid-consolidation still stamps the marker.
            # Stamp FIRST — before the informational aprint, which can itself
            # raise (e.g. BrokenPipeError on a closed stdout) and would
            # otherwise skip the stamp and mask the original error.
            try:
                self.store.attrs["incomplete"] = True
            except BaseException:
                pass
            try:
                aprint(f"⚠️ Failed to finalize: {e}")
            except BaseException:
                pass
            # Preserve the historical wrapping for ordinary Exceptions, but let
            # a KeyboardInterrupt / SystemExit propagate unchanged.
            self._discard_failed_archive_staging()
            if isinstance(e, Exception):
                raise ValueError(f"Could not finalize Zarr store: {e}") from e
            raise

        # Finalization is complete. The flag flips OUTSIDE the failure
        # boundary: nothing past this point may re-enter the handler above,
        # which would stamp `incomplete` on a complete store that a repeat
        # finalize() (early return) could never un-mark.
        self._is_finalized = True
        self._cleanup_archive_staging()
        try:
            aprint(f"✅ Zarr store finalized at {self.store_path}")
        except Exception:
            # Purely informational — a broken stdout (e.g. BrokenPipeError)
            # must not fail an already-complete finalization. A
            # KeyboardInterrupt here still propagates; the store stays valid.
            # The bare `Exception` is deliberately broad: any stdout failure mode
            # qualifies and there is nothing to recover, the store being durable
            # by this point. (Bandit reports try/except/pass as a LOW finding,
            # which this project waives — see the thresholds in
            # .pre-commit-config.yaml.)
            pass

    @property
    def store_path(self) -> str:
        """Get the live directory store, or the finalized archive path."""
        if self._archive_path is not None and self._is_finalized:
            return str(self._archive_path)
        return str(self._store_path)

    @property
    def final_store_path(self) -> str:
        """Get the directory store or archive path produced by finalization."""
        return str(self._archive_path or self._store_path)
