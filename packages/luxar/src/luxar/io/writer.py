"""luxar.writer – Writer interface for progressive Zarr writing.

This module defines the protocol for writing data progressively to Zarr stores,
enabling memory-efficient handling of arbitrarily large datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..typing_utils.aliases import (
    ChunkSpec,
    GSplatsMetadata,
    LinesMetadata,
    MaxShape,
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

    def write_group(self, path: NodePath, **attrs: Any) -> None:
        """Create a group structure in the Zarr store.

        Args:
            path: Path within the Zarr store for the group
            **attrs: Attributes to attach to the group
        """
        ...

    def write_points(
        self,
        path: NodePath,
        positions: PositionArray,
        colors: Optional[ColorArray] = None,
        radii: Optional[ScalarArray] = None,
        sharpness: Optional[ScalarArray] = None,
        **attrs: Any,
    ) -> PointsMetadata:
        """Write points data immediately to Zarr.

        Data is written directly to disk without being kept in memory.
        Only metadata about the written data is returned.

        The actual dtypes used for storage depend on EncodingMode:
        - AUTO mode: Analyzes data and selects optimal encoding
        - PRECISION mode: Uses float32 for maximum precision
        - MEMORY mode: Aggressively quantizes (float16/uint8)

        Args:
            path: Path within the Zarr store for this points
            positions: Point positions array of shape (N, D) - float32 or float16
            colors: Optional colors array of shape (N, 3) - float32 (HDR), uint8/uint16 (SDR)
            radii: Optional radii array of shape (N,) - float32, float16, or uint8
            sharpness: Optional sharpness array of shape (N,) - float32, float16, or uint8
            **attrs: Additional attributes for the points

        Returns:
            Dictionary containing only metadata about the written data:
            - n_points: Number of points written
            - dims: Dimensionality of the points
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
        widths: ScalarArray,
        colors: Optional[ColorArray] = None,
        sharpness: Optional[ScalarArray] = None,
        indices: Optional[NDArray[np.uint32]] = None,
        line_type: str = "polyline",
        **attrs: Any,
    ) -> LinesMetadata:
        """Write lines data immediately to Zarr.

        Args:
            path: Path within the Zarr store for this lines node
            vertices: Vertex positions array of shape (N, D)
            widths: Line widths array of shape (N,)
            colors: Optional colors array of shape (N, 3)
            sharpness: Optional sharpness array of shape (N,)
            indices: Optional vertex indices for indexed line type
            line_type: Type of line connectivity
            **attrs: Additional attributes for the lines

        Returns:
            Dictionary containing metadata about the written lines
        """
        ...

    def write_gsplats(
        self,
        path: NodePath,
        centers: PositionArray,
        amplitudes: ScalarArray,
        cholesky_factors: NDArray[np.float32],
        colors: Optional[ColorArray] = None,
        sharpness: Optional[ScalarArray] = None,
        **attrs: Any,
    ) -> GSplatsMetadata:
        """Write Gaussian splats data immediately to Zarr.

        Args:
            path: Path within the Zarr store for this gsplats node
            centers: Splat center positions array of shape (N, D)
            amplitudes: Amplitude values array of shape (N,)
            cholesky_factors: Packed Cholesky factors array of shape (N, k)
            colors: Optional colors array of shape (N, 3)
            sharpness: Optional generalized Gaussian exponent array of shape (N,)
            **attrs: Additional attributes for the gsplats

        Returns:
            Dictionary containing metadata about the written gsplats
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
