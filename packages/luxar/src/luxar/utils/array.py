"""Array utility functions for Luxar processing.

This module provides efficient utilities for array processing and validation.

Note: As of v1.4.0, scalar broadcasting is handled by ArrayEncoder in luxar.encoding.
The broadcast_*_to_points() functions have been removed (obsolete).

Key Functions:
    - ensure_float32: Convert arrays to float32 dtype for GPU compatibility
    - validate_array_shape: Validate array dimensions with helpful errors

Example:
    >>> # Ensure array is float32
    >>> data = ensure_float32(my_array)

    >>> # Validate shape
    >>> validate_array_shape(colors, (1000, 3), name="colors")
"""

from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray


def ensure_float32(array: NDArray) -> NDArray[np.float32]:
    """Ensure array is float32 dtype, converting if necessary.

    Args:
        array: Input array

    Returns:
        Array with float32 dtype
    """
    if array.dtype != np.float32:
        return array.astype(np.float32)
    return array


def validate_array_shape(
    array: NDArray,
    expected_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]],
    name: str = "array",
) -> None:
    """Validate that an array has the expected shape.

    Args:
        array: Array to validate
        expected_shape: Expected shape or list of acceptable shapes
        name: Name for error messages

    Raises:
        ValueError: If array doesn't match expected shape(s)
    """
    if isinstance(expected_shape, list):
        if array.shape not in expected_shape:
            shapes_str = " or ".join(str(s) for s in expected_shape)
            raise ValueError(
                f"{name.capitalize()} must have shape {shapes_str}, "
                f"got shape {array.shape}"
            )
    else:
        if array.shape != expected_shape:
            raise ValueError(
                f"{name.capitalize()} must have shape {expected_shape}, "
                f"got shape {array.shape}"
            )
