"""luxar.writer – Writer interface for progressive Zarr writing.

This module defines the protocol for writing data progressively to Zarr stores,
enabling memory-efficient handling of arbitrarily large datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..typing_utils.aliases import (
    ChunkSpec,
    GSplatsMetadata,
    LinesMetadata,
    MaxShape,
    MeshMetadata,
    NodePath,
    PointsMetadata,
)

if TYPE_CHECKING:
    import zarr

# Type aliases for arrays that can be written with different dtypes
# Actual dtype is selected by ArrayEncoder based on semantic type and mode
PositionArray = Union[NDArray[np.float32], NDArray[np.float16]]
ColorArray = Union[NDArray[np.float32], NDArray[np.uint8], NDArray[np.uint16]]
ScalarArray = Union[NDArray[np.float32], NDArray[np.float16], NDArray[np.uint8]]


class ZarrWriterProtocol(Protocol):
    """Protocol for progressive Zarr writing.

    This interface enables different implementations of Zarr writers
    while maintaining a consistent API for the scene graph nodes.
    Writers implementing this protocol handle immediate data persistence
    without keeping data in memory.
    """

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
        shading: Optional[str] = None,
        double_sided: bool = True,
        labels: Optional[Sequence[str]] = None,
        image_labels: Optional[Any] = None,
        **attrs: Any,
    ) -> MeshMetadata:
        """Write a triangle mesh immediately to Zarr.

        The surface geometry type: nD ``vertices`` plus a ``faces`` triangle-index
        array. It carries no per-element size — a triangle's extent comes from its
        own vertices — and has no spatial index or LOD in v1.

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
            shading: ``"smooth"`` / ``"flat"``; defaults by normal presence
            double_sided: Whether back faces render (default True)
            labels: Optional list of strings, one per vertex, for hover tooltips
            image_labels: Optional per-element images for hover thumbnails
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
        labels: Optional[Sequence[str]] = None,
        image_labels: Optional[Any] = None,
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
            labels: Optional list of strings, one per splat, for hover tooltips
            image_labels: Optional per-element images for hover thumbnails
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
        extend_to_all: Optional[Union[list, str]] = None,
        **attrs: Any,
    ) -> dict:
        """Write multi-additive-LOD Points: parent node + ``additive_<i>/`` subgroups.

        Each level is a dict with ``positions`` plus optional
        ``colors`` / ``radii`` / ``sharpness`` / ``scalars`` / ``labels``.
        See :func:`luxar.core.group.lod.points.make_additive_lod_points`
        for the level-construction helper that produces the input.
        """
        ...

    def write_lines_multi_lod(
        self,
        path: NodePath,
        levels: list,
        *,
        extend_to_all: Optional[Union[list, str]] = None,
        **attrs: Any,
    ) -> dict:
        """Write multi-additive-LOD Lines (polyline-level granularity).

        Each level is a dict with ``vertices`` + ``widths`` + ``segments``
        plus optional ``colors`` / ``sharpness`` / ``scalars`` / ``labels``
        and ``n_polylines`` (summed into the returned metadata only — the
        parent group does not stamp it). See
        :func:`luxar.core.group.lod.lines.make_additive_lod_lines` for the
        helper that produces the input.
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
            maxshape: Maximum shape (None for unlimited dimensions)
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

    def finalize(self) -> None:
        """Finalize the Zarr store.

        Performs any necessary cleanup, metadata consolidation,
        or optimization steps before closing the store.
        """
        ...

    @property
    def store_path(self) -> str:
        """Get the path to the underlying Zarr store.

        Returns:
            Path to the Zarr store being written to
        """
        ...
