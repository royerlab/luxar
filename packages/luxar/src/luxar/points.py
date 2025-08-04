"""
luxar.points – Defines the Points node for point cloud data in Luxar scenes.
"""

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


def _create_array(
    group: Union[zarr.Group, ZarrGroupProtocol],
    name: str,
    data: Union[PositionArray, ColorArray, np.ndarray[Any, np.dtype[np.float32]]],
    chunk_size: int,
    compressor: Optional[CompressorProtocol],
    dtype: Union[type[np.float32], type[np.uint8]],
) -> None:
    """
    Write data into ``group/<name>`` with sensible defaults.

    Args:
        group: Zarr group to write to
        name: Dataset name
        data: Data to write (positions or colors)
        chunk_size: Chunk size for Zarr dataset
        compressor: Compressor for Zarr dataset
        dtype: Data type for Zarr dataset

    Raises:
        ValueError: If data cannot be written to Zarr
    """
    try:
        aprint(f"Creating dataset '{name}' with shape {data.shape} and dtype {dtype}.")

        # Calculate appropriate chunk shape
        chunks: Union[int, tuple[int, ...]]
        if data.ndim == 2:
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
    """
    Point-cloud node (writes immediately to the backing store).
    This class is intended for internal use via Scene.add_points().

    Args:
        name (str): Name of the point cloud node.
        positions (NDArray[np.float32]): Array of shape (N, 3) for point positions.
        colors (NDArray[np.uint8], optional): Array of shape (N, 3) for point colors.
        radii (NDArray[np.float32], optional): Array of shape (N,) for point radii.
        sharpness (NDArray[np.float32], optional): Array of shape (N,) for point edge sharpness.
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
        radii: Optional[Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]] = None,
        sharpness: Optional[Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]] = None,
        parent: Optional[Node] = None,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        compressor: Optional[CompressorProtocol] = DEFAULT_COMP,
        **attrs: Any,
    ) -> None:
        """
        Initialize a Points node with position, color, radius, and sharpness data.

        Args:
            name: Name of the point cloud node
            positions: Array of shape (N, 3) for point positions
            colors: Optional array of shape (N, 3) for point colors
            radii: Optional array of shape (N,) for point radii
            sharpness: Optional array of shape (N,) for point edge sharpness
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
                validated_colors = validate_colors(colors, n_points)

            # Validate and convert radii if provided
            validated_radii: Optional[np.ndarray[Any, np.dtype[np.float32]]] = None
            if radii is not None:
                validated_radii = validate_radii(radii, n_points)

            # Validate and convert sharpness if provided
            validated_sharpness: Optional[np.ndarray[Any, np.dtype[np.float32]]] = None
            if sharpness is not None:
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

            # Create datasets with validated data
            _create_array(
                grp,
                "positions",
                validated_positions,
                chunk_size,
                compressor,
                np.float32,
            )
            if validated_colors is not None:
                _create_array(
                    grp, "colors", validated_colors, chunk_size, compressor, np.uint8
                )
            if validated_radii is not None:
                _create_array(
                    grp, "radii", validated_radii, chunk_size, compressor, np.float32
                )
            if validated_sharpness is not None:
                _create_array(
                    grp, "sharpness", validated_sharpness, chunk_size, compressor, np.float32
                )

            aprint(f"✓ Points node '{name}' created with {n_points:,} points.")

        except Exception as e:
            aprint(f"Failed to create Points node '{name}': {e}")
            raise ValueError(f"Could not create Points node '{name}': {e}") from e
