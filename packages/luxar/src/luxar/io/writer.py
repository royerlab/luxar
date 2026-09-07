"""luxar.writer – Writer interface for progressive Zarr writing.

This module defines the protocol for writing data progressively to Zarr stores,
enabling memory-efficient handling of arbitrarily large datasets.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ContextManager,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from ..typing_utils.aliases import (
    ChunkSpec,
    ColorArray,
    GSplatsMetadata,
    LinesMetadata,
    MaxShape,
    MeshMetadata,
    NodePath,
    PointsMetadata,
    PositionArray,
    ScalarArray,
)

if TYPE_CHECKING:
    import zarr

# `PositionArray` / `ColorArray` / `ScalarArray` are imported above rather than
# defined here. This module used to redefine the first two with DIFFERENT,
# WIDER bodies than the same names carry in `typing_utils.aliases` — see the
# comment there for why that made this protocol unenforceable.
RollbackState = Tuple[
    Optional[dict[str, List[float]]],
    frozenset[Tuple[str, str]],
    bool,
    Tuple[dict[tuple, tuple[str, str]], dict[str, str]],
    Optional[Tuple[bool, bool]],
]


class ZarrWriterProtocol(Protocol):
    """Protocol for progressive Zarr writing.

    This interface enables different implementations of Zarr writers
    while maintaining a consistent API for the scene graph nodes.
    Writers implementing this protocol handle immediate data persistence
    without keeping data in memory.
    """

    store: zarr.Group

    def write_group(
        self, path: NodePath, *, _transform_normalized: bool = False, **attrs: Any
    ) -> None:
        """Create a group structure in the Zarr store.

        Args:
            path: Path within the Zarr store for the group
            _transform_normalized: Internal control parameter (NOT a group
                attribute — never persisted). Set to True by the Node/Scene
                API to declare that ``transform`` / ``nd_transform`` in
                ``attrs`` are already in the on-disk normalized form
                (column-major flat list / validated dict), so the writer must
                not normalize them a second time (the conversion is not
                idempotent). All other callers leave the default False and
                get full normalization + validation.
            **attrs: Attributes to attach to the group
        """
        ...

    def write_points(
        self,
        path: NodePath,
        positions: PositionArray,
        colors: Optional[Union[ColorArray, tuple, list]] = None,
        radii: Optional[Union[ScalarArray, float]] = None,
        sharpness: Optional[Union[ScalarArray, float]] = None,
        scalars: Optional[Union[ScalarArray, float]] = None,
        labels: Optional[Sequence[str]] = None,
        image_labels: Optional[Any] = None,
        keys: Optional[Sequence[str]] = None,
        **attrs: Any,
    ) -> PointsMetadata:
        """Write points data immediately to Zarr.

        Data is written directly to disk without being kept in memory.
        Only metadata about the written data is returned.

        The actual dtypes used for storage depend on EncodingMode:
        - AUTO mode: Analyzes data and selects optimal encoding
        - PRECISION mode: Uses float32 for maximum precision
        - MEMORY mode: Aggressively quantizes (float16/uint8)

        Scalar Convenience (v1.4.0): Optional attributes accept scalars:
        - radii=0.5 instead of np.full(N, 0.5)
        - colors=(1.0, 0, 0) instead of np.full((N, 3), [1,0,0])
        - sharpness=0.5 instead of np.full(N, 0.5)

        Args:
            path: Path within the Zarr store for this points
            positions: Point positions array of shape (N, D) - float32 or float16
                       (NOT scalar - positions must be full arrays)
            colors: Optional - array of shape (N, 3), tuple/list (R,G,B), or None
            radii: Optional - array of shape (N,), scalar float, or None
            sharpness: Optional - array of shape (N,), scalar float, or None
            scalars: Optional - array of shape (N,), scalar float, or None.
                Used for colormap lookup when a colormap is applied.
            labels: Optional list of strings, one per point, for hover tooltips
            image_labels: Optional per-element images for hover thumbnails
            keys: Optional machine-readable strings, one per point, for link and
                copy templates.
            **attrs: Additional attributes for the points

        Returns:
            Dictionary containing only metadata about the written data:
            - n_points: Number of points written
            - ndim: Dimensionality of the points
            - path: Path where data was written
            - has_colors: Whether colors were written
            - has_radii: Whether radii were written
            - has_sharpness: Whether sharpness was written
        """
        ...

    def write_lines(
        self,
        path: NodePath,
        vertices: PositionArray,
        widths: Union[ScalarArray, float],
        colors: Optional[Union[ColorArray, tuple, list]] = None,
        sharpness: Optional[Union[ScalarArray, float]] = None,
        scalars: Optional[Union[ScalarArray, float]] = None,
        indices: Optional[NDArray[np.uint32]] = None,
        line_type: str = "polyline",
        labels: Optional[Sequence[str]] = None,
        image_labels: Optional[Any] = None,
        keys: Optional[Sequence[str]] = None,
        **attrs: Any,
    ) -> LinesMetadata:
        """Write lines data immediately to Zarr.

        Scalar Convenience (v1.4.0): Uniform attributes accept scalars:
        - widths=0.1 instead of np.full(N, 0.1)
        - colors=(1.0, 0, 0) instead of np.full((N, 3), [1,0,0])
        - sharpness=0.5 instead of np.full(N, 0.5)

        Args:
            path: Path within the Zarr store for this lines node
            vertices: Vertex positions array of shape (N, D)
            widths: Line widths - array of shape (N,) or scalar float
            colors: Optional - array of shape (N, 3), tuple/list (R,G,B), or None
            sharpness: Optional - array of shape (N,), scalar float, or None
            scalars: Optional - array of shape (N,), scalar float, or None.
                Used for colormap lookup when a colormap is applied.
            indices: Optional vertex indices for indexed line type
            line_type: Type of line connectivity
            labels: Optional list of strings, one per vertex, for hover tooltips
            image_labels: Optional per-element images for hover thumbnails
            keys: Optional machine-readable strings, one per vertex, for link and
                copy templates.
            **attrs: Additional attributes for the lines

        Returns:
            Dictionary containing metadata about the written lines
        """
        ...

    def write_mesh(
        self,
        path: NodePath,
        vertices: PositionArray,
        faces: NDArray[np.uint32],
        normals: Optional[NDArray[np.float32]] = None,
        normal_dims: Optional[Sequence[int]] = None,
        colors: Optional[Union[ColorArray, tuple, list]] = None,
        scalars: Optional[Union[ScalarArray, float]] = None,
        # The texture block below must stay HERE, between `scalars` and
        # `shading`, and in this order. It is where `LuxarZarrCompiler.write_mesh`
        # puts it, and these are positional-or-keyword parameters: omitting them
        # from the protocol (as it did until now) does not merely under-declare
        # the interface, it renumbers every parameter after them. A caller typed
        # against the protocol writing
        # `writer.write_mesh(path, v, f, n, dims, colors, scalars, "flat")`
        # would land "flat" in `uvs`, not `shading`. The same insertion mistake
        # was made once already inside `_write_mesh_impl` — see the note at that
        # forwarding call, which now passes keywords for exactly this reason.
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
        labels: Optional[Sequence[str]] = None,
        image_labels: Optional[Any] = None,
        keys: Optional[Sequence[str]] = None,
        **attrs: Any,
    ) -> MeshMetadata:
        """Write a triangle mesh immediately to Zarr.

        The surface geometry type: nD ``vertices`` plus a ``faces`` triangle-index
        array. It carries no per-element size — a triangle's extent comes from its
        own vertices — and has no spatial index: the loader is whole-node. This call
        writes one leaf; LOD is layered on by the caller, not by this method — see
        ``write_mesh_multi_lod`` (additive reveal levels) below.

        Args:
            path: Path within the Zarr store for this mesh node
            vertices: Vertex positions of shape (V, D)
            faces: Triangle vertex indices, (F, 3) or flat (3F,)
            normals: Optional per-vertex normals of shape (V, 3); requires
                ``normal_dims``
            normal_dims: The three dimension indices ``normals`` describes —
                required with ``normals``, rejected without
            colors: Optional - array of shape (V, 3|4), tuple/list (R,G,B[,A]), or
                None
            scalars: Optional - array of shape (V,), scalar float, or None. Used
                for colormap lookup when a colormap is applied.
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
            shading: ``"smooth"`` / ``"flat"`` / unlit ``"none"``; defaults by
                normal presence
            double_sided: Whether back faces render (default True)
            labels: Optional list of strings, one per vertex, for hover tooltips
            image_labels: Optional per-element images for hover thumbnails
            keys: Optional machine-readable strings, one per vertex, for link and
                copy templates.
            **attrs: Additional attributes for the mesh

        Returns:
            Dictionary containing metadata about the written mesh
        """
        ...

    def write_gsplats(
        self,
        path: NodePath,
        centers: PositionArray,
        amplitudes: Union[ScalarArray, float],
        cholesky_factors: NDArray[np.float32],
        colors: Optional[Union[ColorArray, tuple, list]] = None,
        label_ids: Optional[Union[np.ndarray[Any, Any], Sequence[int]]] = None,
        label_vocabulary: Optional[dict[int, str]] = None,
        labels: Optional[Sequence[str]] = None,
        image_labels: Optional[Any] = None,
        keys: Optional[Sequence[str]] = None,
        **attrs: Any,
    ) -> GSplatsMetadata:
        """Write Gaussian splats data immediately to Zarr.

        Scalar Convenience (v1.4.0): Uniform attributes accept scalars:
        - amplitudes=1.0 instead of np.full(N, 1.0)
        - colors=(1.0, 0, 0) instead of np.full((N, 3), [1,0,0])

        Args:
            path: Path within the Zarr store for this gsplats node
            centers: Splat center positions array of shape (N, D)
            amplitudes: Amplitude values - array of shape (N,) or scalar float
            cholesky_factors: Packed Cholesky factors array of shape (N, k)
            colors: Optional - array of shape (N, 3), tuple/list (R,G,B), or None
            label_ids: Optional non-negative integer class id per splat
            label_vocabulary: Explicit mapping from stored class ids to names
            labels: Optional list of strings, one per splat, for hover tooltips
            image_labels: Optional per-element images for hover thumbnails
            keys: Optional machine-readable strings, one per splat, for link and
                copy templates.
            **attrs: Additional attributes for the gsplats

        Returns:
            Dictionary containing metadata about the written gsplats
        """
        ...

    def write_points_multi_lod(
        self,
        path: NodePath,
        levels: list,
        *,
        extend_to_all: Optional[List[str]] = None,
        **attrs: Any,
    ) -> dict:
        """Write multi-additive-LOD Points: parent node + ``additive_<i>/`` subgroups.

        Each level is a dict with ``positions`` plus optional
        ``colors`` / ``radii`` / ``sharpness`` / ``scalars`` / ``labels`` /
        ``keys``.
        See :func:`luxar.core.group.lod.points.make_additive_lod_points`
        for the level-construction helper that produces the input.

        ``labels`` and ``keys`` are independently all-or-nothing across the
        ladder and are NOT written per level. Each present channel gets one CSR
        pair on the parent, spanning the levels in stored order; the subgroups
        carry neither channel.
        """
        ...

    def write_lines_multi_lod(
        self,
        path: NodePath,
        levels: list,
        *,
        extend_to_all: Optional[List[str]] = None,
        **attrs: Any,
    ) -> dict:
        """Write multi-additive-LOD Lines (polyline-level granularity).

        Each level is a dict with ``vertices`` + ``widths`` + ``segments``
        plus optional ``colors`` / ``sharpness`` / ``scalars`` / ``labels`` /
        ``keys``
        and ``n_polylines`` (summed into the returned metadata only — the
        parent group does not stamp it). See
        :func:`luxar.core.group.lod.lines.make_additive_lod_lines` for the
        helper that produces the input.

        ``labels`` and ``keys`` are independently all-or-nothing across the
        ladder and are NOT written per level. Each present channel gets one
        per-vertex CSR pair on the parent, spanning the levels in stored order;
        the subgroups carry neither channel.
        """
        ...

    def write_sound(
        self,
        path: NodePath,
        payload: bytes,
        fmt: str,
        positions: Optional[NDArray[np.float32]],
        *,
        sound_attrs: dict[str, Any],
        **attrs: Any,
    ) -> dict[str, Any]:
        """Write a sound node: an opaque MP3/AAC clip + optional positions.

        ``fmt`` is ``"mp3"`` or ``"aac"`` (already sniffed by the adder);
        ``sound_attrs`` carries the validated playback / spatial / licence knobs
        and ``**attrs`` the compositing pass-throughs. Returns the node metadata.
        """
        ...

    def write_mesh_multi_lod(
        self,
        path: NodePath,
        levels: list,
        *,
        extend_to_all: Optional[List[str]] = None,
        **attrs: Any,
    ) -> dict:
        """Write multi-additive-LOD Mesh (a reveal ladder, face granularity).

        Each level is a dict with ``vertices`` + ``faces`` — the faces already
        re-indexed into that level's own gathered vertex table — plus optional
        ``normals`` / ``normal_dims`` / ``colors`` / ``scalars`` / ``shading`` /
        ``double_sided`` / ``_scalar_data_range`` and ``lod_stats``. See
        :func:`luxar.core.group.lod.mesh.make_additive_lod_mesh` for the helper
        that produces the face groups and
        :func:`luxar.mesh.split.split_mesh_by_faces` for the re-indexing.

        ``labels`` is REFUSED, not carried — the one place this diverges from its
        three siblings. They write ONE union CSR on the parent spanning the levels;
        a mesh level re-indexes its own vertices, so a single source vertex maps to
        a slot in several levels and the union index space is ill-defined. Labels
        survive on the ``substitutive_lod=`` and ``partition=`` paths instead.
        """
        ...

    def create_resizable_dataset(
        self,
        path: NodePath,
        dtype: np.dtype,
        shape: Tuple[int, ...],
        maxshape: MaxShape = None,
        chunks: ChunkSpec = True,
    ) -> "zarr.Array":
        """Create a resizable dataset.

        Part of writer protocol for potential future extensions.

        Args:
            path: Path for the dataset within the Zarr store
            dtype: Data type for the dataset (numpy dtype)
            shape: Initial shape of the dataset
            maxshape: Accepted and ignored — see
                :meth:`LuxarZarrCompiler.create_resizable_dataset`. zarr has
                never enforced a maximum shape; the argument is h5py heritage.
            chunks: Chunk configuration for the dataset

        Returns:
            Zarr array handle that supports resizing and slicing
        """
        ...

    def delete_group_attr(self, path: NodePath, key: str) -> None:
        """Remove an attribute from a group in the Zarr store.

        Args:
            path: Path within the Zarr store for the group
            key: Attribute key to remove
        """
        ...

    def node_exists(self, path: NodePath) -> bool:
        """Return whether a node path exists for an internal rollback guard."""
        ...

    def delete_node(self, path: NodePath) -> None:
        """Delete a subtree during rollback, without editing the scene graph."""
        ...

    def snapshot_rollback_state(self) -> RollbackState:
        """Capture compiler state that deleted geometry writes may have changed."""
        ...

    def restore_rollback_state(self, state: RollbackState) -> None:
        """Restore compiler state captured before a rolled-back write."""
        ...

    def transaction(self, path: NodePath) -> ContextManager[None]:
        """Roll back a newly-created subtree and compiler state on failure."""
        ...

    def finalize(self) -> None:
        """Finalize the Zarr store.

        Performs any necessary cleanup, metadata consolidation,
        or optimization steps before closing the store.
        """
        ...

    @property
    def store_path(self) -> str:
        """Get the current path to the underlying Zarr store.

        Writers may use a staging directory while active and return a different
        published path after ``finalize()``.

        Returns:
            Current staging or finalized store path.
        """
        ...

    @property
    def final_store_path(self) -> str:
        """Get the path where the finalized Zarr store will be published."""
        ...
