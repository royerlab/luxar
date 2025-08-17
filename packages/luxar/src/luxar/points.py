"""luxar.points – Defines the Points node for point cloud data in Luxar scenes."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import zarr
from arbol import aprint

from ._io import DEFAULT_COMP
from .config import DEFAULT_CHUNK_SIZE, check_dataset_size_warning
from .node import Node
from .types import (
    ColorArray,
    CompressorProtocol,
    NodeType,
    PositionArray,
    ZarrGroupProtocol,
    validate_colors,
    validate_positions,
    validate_radii,
    validate_sharpness,
)


def _calculate_dimension_aware_chunks(
    shape: tuple[int, ...],
    chunk_size: int,
    dimension_metadata: list[Any],
) -> tuple[int, ...]:
    """Calculate chunk shape optimized for dimension-aware loading.
    
    For nD data where some dimensions are not displayed,
    we want to chunk along those dimensions to enable efficient
    lazy loading of individual slices.
    
    Args:
        shape: Data array shape (n_points, n_dims)
        chunk_size: Target chunk size in elements
        dimension_metadata: Metadata about each dimension (list of dicts)
        
    Returns:
        Optimized chunk shape tuple
    """
    n_points, n_dims = shape

    # Count non-displayed dimensions
    non_displayed_count = 0
    total_non_displayed_range = 1

    for i, dim_meta in enumerate(dimension_metadata):
        # Handle dict format from to_dict()
        if isinstance(dim_meta, dict):
            is_displayed = dim_meta.get('display', True)
            dim_range = dim_meta.get('range', [0, 0])
        elif hasattr(dim_meta, 'display'):
            is_displayed = dim_meta.display
            dim_range = getattr(dim_meta, 'range', [0, 0])
        else:
            is_displayed = True
            dim_range = [0, 0]

        if not is_displayed:
            non_displayed_count += 1
            # Calculate the size of this non-displayed dimension
            dim_size = int(dim_range[1] - dim_range[0] + 1) if dim_range else 1
            total_non_displayed_range *= dim_size

    # If we have non-displayed dimensions and nD data
    if non_displayed_count > 0 and n_dims > 3:
        # Calculate points per slice in non-displayed dimensions
        # This assumes data is organized with all points for one slice together
        if total_non_displayed_range > 0:
            points_per_slice = n_points // total_non_displayed_range
            if points_per_slice > 0:
                # Create chunks aligned with slices in non-displayed dimensions
                # This allows loading individual slices efficiently
                return (points_per_slice, n_dims)

    # Default chunking for 3D data or when dimension-aware chunking isn't applicable
    return (chunk_size, n_dims)


def _create_array(
    group: Union[zarr.Group, ZarrGroupProtocol],
    name: str,
    data: Union[PositionArray, ColorArray, np.ndarray[Any, np.dtype[np.float32]]],
    chunk_size: int,
    compressor: Optional[CompressorProtocol],
    dtype: Union[type[np.float32], type[np.uint8]],
    dimension_metadata: Optional[list[Any]] = None,
) -> None:
    """Write data into ``group/<name>`` with sensible defaults.

    Args:
        group: Zarr group to write to
        name: Dataset name
        data: Data to write (positions or colors)
        chunk_size: Chunk size for Zarr dataset
        compressor: Compressor for Zarr dataset
        dtype: Data type for Zarr dataset
        dimension_metadata: Optional dimension metadata for intelligent chunking

    Raises:
        ValueError: If data cannot be written to Zarr
    """
    try:
        aprint(f"Creating dataset '{name}' with shape {data.shape} and dtype {dtype}.")

        # Calculate appropriate chunk shape
        chunks: Union[int, tuple[int, ...]]
        if data.ndim == 2:
            # For nD position data, use dimension-aware chunking
            if name == "positions" and dimension_metadata:
                chunks = _calculate_dimension_aware_chunks(
                    data.shape, chunk_size, dimension_metadata
                )
            else:
                chunks = (chunk_size, data.shape[1])
        else:
            chunks = chunk_size

        group.create_dataset(
            name,
            data=data,
            chunks=chunks,
            compressor=compressor,
            dtype=dtype,
            overwrite=True,
        )

        aprint(f"✓ Dataset '{name}' created successfully.")

    except Exception as e:
        aprint(f"Failed to create dataset '{name}': {e}")
        raise ValueError(f"Could not create Zarr dataset '{name}': {e}") from e


class Points(Node):
    """Point-cloud node (writes immediately to the backing store).
    This class is intended for internal use via Scene.add_points().

    Args:
        name (str): Name of the point cloud node.
        positions (NDArray[np.float32]): Array of shape (N, D) for point positions where D is dimensionality.
        colors (NDArray[np.float32] | tuple | list, optional): Array of shape (N, 3) for HDR point colors,
            or single RGB color as (R, G, B) tuple/list to apply to all points.
            Values can exceed 1.0 for HDR emission (e.g., 10.0 for very bright points).
        radii (NDArray[np.float32] | float, optional): Array of shape (N,) for point radii,
            or single radius value to apply to all points.
        sharpness (NDArray[np.float32] | float, optional): Array of shape (N,) for point edge sharpness,
            or single sharpness value to apply to all points.
        parent (Node, optional): Parent node. Defaults to None.
        chunk_size (int, optional): Chunk size for Zarr dataset. Defaults to 32,768.
        compressor: Compressor for Zarr dataset. Defaults to DEFAULT_COMP.
        **attrs: Additional attributes for the node.

    Raises:
        ValueError: If positions, colors, radii, or sharpness are not valid shapes or types.
    """

    def __init__(
        self,
        name: str,
        positions: Union[PositionArray, np.ndarray[Any, Any]],
        colors: Optional[Union[ColorArray, np.ndarray[Any, Any]]] = None,
        radii: Optional[
            Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]
        ] = None,
        sharpness: Optional[
            Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]
        ] = None,
        parent: Optional[Node] = None,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        compressor: Optional[CompressorProtocol] = DEFAULT_COMP,
        **attrs: Any,
    ) -> None:
        """Initialize a Points node with position, color, radius, and sharpness data.

        Args:
            name: Name of the point cloud node
            positions: Array of shape (N, D) for point positions where D is dimensionality
            colors: Optional array of shape (N, 3) for point colors, or single RGB color
                as (R, G, B) tuple/list to apply to all points
            radii: Optional array of shape (N,) for point radii, or single radius value
                to apply to all points
            sharpness: Optional array of shape (N,) for point edge sharpness, or single
                sharpness value to apply to all points
            parent: Parent node in the scene graph
            chunk_size: Chunk size for Zarr dataset storage
            compressor: Compressor for Zarr dataset
            **attrs: Additional attributes for the node

        Raises:
            ValueError: If positions, colors, radii, or sharpness have invalid shapes or types
        """
        try:
            # Validate and convert positions
            validated_positions = validate_positions(positions)
            n_points = validated_positions.shape[0]

            # Check for performance warnings
            warning = check_dataset_size_warning(n_points)
            if warning:
                aprint(f"Performance warning: {warning}")

            # Validate and convert colors if provided
            validated_colors: Optional[ColorArray] = None
            if colors is not None:
                # Handle single color value (broadcast to all points)
                if isinstance(colors, (list, tuple)) and len(colors) == 3:  # type: ignore[unreachable]
                    # Single RGB color as list/tuple - HDR float32
                    color_array = np.array(colors, dtype=np.float32)  # type: ignore[unreachable]
                    validated_colors = np.tile(color_array, (n_points, 1))
                elif isinstance(colors, np.ndarray) and colors.shape == (3,):
                    # Single RGB color as numpy array - HDR float32
                    validated_colors = np.tile(colors.astype(np.float32), (n_points, 1))
                else:
                    # Full color array
                    validated_colors = validate_colors(colors, n_points)

            # Validate and convert radii if provided
            validated_radii: Optional[np.ndarray[Any, np.dtype[np.float32]]] = None
            if radii is not None:
                # Handle single radius value (broadcast to all points)
                if np.isscalar(radii) or (
                    isinstance(radii, np.ndarray) and radii.shape == ()
                ):
                    # Single radius value
                    radius_value = float(radii)  # type: ignore[arg-type]
                    if radius_value <= 0:
                        raise ValueError("Radius must be positive")
                    validated_radii = np.full(n_points, radius_value, dtype=np.float32)
                else:
                    # Full radius array
                    validated_radii = validate_radii(radii, n_points)

            # Validate and convert sharpness if provided
            validated_sharpness: Optional[np.ndarray[Any, np.dtype[np.float32]]] = None
            if sharpness is not None:
                # Handle single sharpness value (broadcast to all points)
                if np.isscalar(sharpness) or (
                    isinstance(sharpness, np.ndarray) and sharpness.shape == ()
                ):
                    # Single sharpness value
                    sharpness_value = float(sharpness)  # type: ignore[arg-type]
                    if sharpness_value <= 0:
                        raise ValueError("Sharpness must be positive")
                    validated_sharpness = np.full(
                        n_points, sharpness_value, dtype=np.float32
                    )
                else:
                    # Full sharpness array
                    validated_sharpness = validate_sharpness(sharpness, n_points)

            # Create or get Zarr group
            if parent is not None:
                grp = parent._group.require_group(name)
            else:
                # Standalone case (rarely used)
                grp = zarr.group().require_group(name)

            # Initialize parent Node with type annotation
            node_type: NodeType = "points"
            super().__init__(name, grp, parent=parent, type=node_type, **attrs)

            # Set node attributes
            grp.attrs.setdefault("num_points", n_points)

            # Get dimension metadata from parent scene for chunking
            dimension_metadata = None
            if parent is not None:
                # Try to get scene dimensions for intelligent chunking
                scene = parent
                while hasattr(scene, 'parent') and scene.parent is not None:
                    scene = scene.parent
                if hasattr(scene, '_dimensions') and scene._dimensions is not None:
                    dimension_metadata = [d.to_dict() for d in scene._dimensions.dimensions]

            # Create datasets with validated data
            _create_array(
                grp,
                "positions",
                validated_positions,
                chunk_size,
                compressor,
                np.float32,
                dimension_metadata,
            )
            if validated_colors is not None:
                _create_array(
                    grp, "colors", validated_colors, chunk_size, compressor, np.float32
                )
            if validated_radii is not None:
                _create_array(
                    grp, "radii", validated_radii, chunk_size, compressor, np.float32
                )
            if validated_sharpness is not None:
                _create_array(
                    grp,
                    "sharpness",
                    validated_sharpness,
                    chunk_size,
                    compressor,
                    np.float32,
                )

            aprint(f"✓ Points node '{name}' created with {n_points:,} points.")

        except Exception as e:
            aprint(f"Failed to create Points node '{name}': {e}")
            raise ValueError(f"Could not create Points node '{name}': {e}") from e
