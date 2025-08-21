"""luxar.writer – Writer interface for progressive Zarr writing.

This module defines the protocol for writing data progressively to Zarr stores,
enabling memory-efficient handling of arbitrarily large datasets.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from .type_aliases import ChunkSpec, MaxShape, NodePath, PointsMetadata


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
        positions: NDArray[np.float32],
        colors: Optional[NDArray[np.float32]] = None,
        radii: Optional[NDArray[np.float32]] = None,
        sharpness: Optional[NDArray[np.float32]] = None,
        **attrs: Any,
    ) -> PointsMetadata:
        """Write point cloud data immediately to Zarr.
        
        Data is written directly to disk without being kept in memory.
        Only metadata about the written data is returned.
        
        Args:
            path: Path within the Zarr store for this point cloud
            positions: Point positions array of shape (N, D)
            colors: Optional HDR colors array of shape (N, 3) 
            radii: Optional radii array of shape (N,)
            sharpness: Optional sharpness array of shape (N,)
            **attrs: Additional attributes for the point cloud
            
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

    def create_resizable_dataset(
        self,
        path: NodePath,
        dtype: Any,
        shape: Tuple[int, ...],
        maxshape: MaxShape = None,
        chunks: ChunkSpec = True,
    ) -> Any:
        """Create a resizable dataset for streaming writes.
        
        This enables appending data in batches without loading existing data.
        
        Args:
            path: Path for the dataset within the Zarr store
            dtype: Data type for the dataset
            shape: Initial shape of the dataset
            maxshape: Maximum shape (None for unlimited dimensions)
            chunks: Chunk configuration for the dataset
            
        Returns:
            Handle to the created dataset that supports resizing and slicing
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
