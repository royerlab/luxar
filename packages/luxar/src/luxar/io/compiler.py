"""luxar.compiler – Progressive Zarr compiler for memory-efficient scene building.

This module provides the LuxarZarrCompiler class which implements progressive
writing to Zarr stores, enabling handling of arbitrarily large datasets without
memory constraints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from ..core.dimensions import Dimensions
from ..io.reader import DEFAULT_COMP
from ..io.writer import ZarrWriterProtocol
from ..typing_utils.aliases import ChunkSpec, MaxShape, NodePath, PointsMetadata
from ..typing_utils.config import DEFAULT_CHUNK_SIZE, DEFAULT_VERSION
from ..typing_utils.protocols import (
    CompressorProtocol,
    PhysicalUnit,
    validate_physical_unit,
)
from ..validation.base import (
    validate_colors_for_writing,
    validate_positions_for_writing,
    validate_radii_for_writing,
    validate_sharpness_for_writing,
)


def _calculate_intelligent_chunks(
    shape: Tuple[int, ...],
    target_chunk_size: int = DEFAULT_CHUNK_SIZE,
    dimension_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[int, ...]:
    """Calculate optimal chunk shape for a dataset.

    Args:
        shape: Shape of the dataset
        target_chunk_size: Target size for chunks in elements
        dimension_metadata: Optional metadata about dimensions for optimization

    Returns:
        Optimized chunk shape
    """
    if len(shape) == 1:
        # 1D array - simple chunking
        return (min(shape[0], target_chunk_size),)

    if len(shape) == 2:
        # 2D array (e.g., positions) - chunk along first dimension
        n_points, n_dims = shape
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
        units: Physical units for the scene (deprecated, use dimensions)
        version: Luxar format version

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
        units: Union[PhysicalUnit, str] = "metre",
        version: str = DEFAULT_VERSION,
    ) -> None:
        """Initialize the Zarr compiler.

        Args:
            store_path: Path for the Zarr store, or None for temporary
            compressor: Compressor for datasets
            units: Physical units for the scene
            version: Luxar format version
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

        # Validate units
        if isinstance(units, str):
            units = validate_physical_unit(units)

        # Create root Zarr group
        self.store = zarr.open_group(self._store_path, mode="w")
        self.store.attrs.update(
            {
                "luxar_version": version,
                "units": units,
                "type": "scene",
                "node_type": "scene",  # Also add node_type for consistency
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

    def create_scene(self, dimensions: Optional[Dimensions] = None) -> Any:
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
        **attrs: Any,
    ) -> PointsMetadata:
        """Write point cloud data progressively to Zarr.

        Data is written immediately to disk without being kept in memory.

        Args:
            path: Path for the point cloud within the store
            positions: Point positions of shape (N, D)
            colors: Optional HDR colors of shape (N, 3)
            radii: Optional radii of shape (N,)
            sharpness: Optional sharpness of shape (N,)
            **attrs: Additional attributes

        Returns:
            Metadata dictionary about the written data
        """
        # Remove leading slash and create group
        path = path.lstrip("/")
        group = self.store.require_group(path)

        # Validate positions and get dimensions
        n_points, n_dims = validate_positions_for_writing(positions)

        aprint(f"📝 Writing {n_points:,} points ({n_dims}D) to {path}")

        # Write positions with intelligent chunking
        chunks = _calculate_intelligent_chunks(
            positions.shape, target_chunk_size=DEFAULT_CHUNK_SIZE
        )
        group.create_dataset(
            "positions",
            data=positions,
            chunks=chunks,
            compressor=self.compressor,
            dtype=np.float32,
            overwrite=True,
        )

        # Write optional arrays
        metadata = {
            "n_points": n_points,
            "dims": n_dims,
            "path": path,
            "has_colors": False,
            "has_radii": False,
            "has_sharpness": False,
        }

        if colors is not None:
            # Validate colors
            validate_colors_for_writing(colors, n_points)

            color_chunks = _calculate_intelligent_chunks(colors.shape)
            group.create_dataset(
                "colors",
                data=colors,
                chunks=color_chunks,
                compressor=self.compressor,
                dtype=np.float32,
                overwrite=True,
            )
            metadata["has_colors"] = True
            aprint("  ✓ Wrote HDR colors")

        if radii is not None:
            # Validate radii
            validate_radii_for_writing(radii, n_points)

            radii_chunks = _calculate_intelligent_chunks(radii.shape)
            group.create_dataset(
                "radii",
                data=radii,
                chunks=radii_chunks,
                compressor=self.compressor,
                dtype=np.float32,
                overwrite=True,
            )
            metadata["has_radii"] = True
            aprint("  ✓ Wrote radii")

        if sharpness is not None:
            # Validate sharpness
            validate_sharpness_for_writing(sharpness, n_points)

            sharp_chunks = _calculate_intelligent_chunks(sharpness.shape)
            group.create_dataset(
                "sharpness",
                data=sharpness,
                chunks=sharp_chunks,
                compressor=self.compressor,
                dtype=np.float32,
                overwrite=True,
            )
            metadata["has_sharpness"] = True
            aprint("  ✓ Wrote sharpness")

        # Process transform if present to convert numpy array to list
        if "transform" in attrs:
            transform_value = attrs["transform"]
            if not isinstance(transform_value, list):
                # Convert numpy array to list for JSON serialization
                transform_array = np.array(transform_value, dtype=np.float32)
                if transform_array.size == 16:
                    transform_matrix = transform_array.reshape(4, 4)
                    # Transpose for THREE.js (column-major order) before flattening
                    attrs["transform"] = transform_matrix.T.ravel().tolist()
                else:
                    raise ValueError(
                        f"Transform must have 16 elements, got {transform_array.size}"
                    )

        # Set default rendering attributes if not provided
        if "opacity" not in attrs:
            attrs["opacity"] = 1.0
        if "gamma" not in attrs:
            attrs["gamma"] = 1.0
        if "blending_mode" not in attrs:
            attrs["blending_mode"] = "additive"

        # Store attributes
        group.attrs.update(attrs)
        group.attrs["type"] = "points"
        group.attrs["n_points"] = n_points

        # Cache metadata
        self._metadata_cache[path] = metadata

        aprint(f"✅ Points written to {path}")
        return metadata

    def create_resizable_dataset(
        self,
        path: NodePath,
        dtype: Any,
        shape: Tuple[int, ...],
        maxshape: MaxShape = None,
        chunks: ChunkSpec = True,
    ) -> Any:
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
