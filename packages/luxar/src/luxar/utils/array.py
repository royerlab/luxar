"""Array utility functions for Luxar point cloud processing.

This module provides efficient utilities for processing and broadcasting
arrays commonly used in point cloud operations. The functions handle
broadcasting single values to arrays, validation, and type conversions.

Key Functions:
    - broadcast_color_to_points: Handle color array broadcasting (RGB to n_points)
    - broadcast_radii_to_points: Handle radius array broadcasting
    - broadcast_sharpness_to_points: Handle sharpness array broadcasting
    - ensure_float32: Convert arrays to float32 dtype for GPU compatibility
    - validate_array_shape: Validate array dimensions with helpful errors

Broadcasting Rules:
    - Scalars are efficiently broadcast to all points
    - Single RGB colors are tiled to create per-point colors
    - Full arrays are validated for correct shape
    - All outputs are float32 for WebGL compatibility

Example:
    >>> # Broadcast single color to 1000 points
    >>> colors = broadcast_color_to_points([1.0, 0.5, 0.0], n_points=1000)
    >>> assert colors.shape == (1000, 3)

    >>> # Broadcast single radius to all points
    >>> radii = broadcast_radii_to_points(0.1, n_points=1000)
    >>> assert radii.shape == (1000,)

These utilities are used throughout Luxar to ensure consistent array
processing and reduce code duplication.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..typing_utils.aliases import ColorArray, RadiusArray, SharpnessArray
from ..typing_utils.constants import SHARPNESS_TYPICAL_MAX, SHARPNESS_TYPICAL_MIN


def broadcast_color_to_points(
    colors: Union[ColorArray, List, Tuple, None], n_points: int
) -> Optional[NDArray[np.float32]]:
    """Broadcast a color specification to all points.

    Args:
        colors: Can be:
            - None: No colors
            - Single RGB tuple/list: (r, g, b) broadcast to all points
            - Single RGB array shape (3,): broadcast to all points
            - Full array shape (n_points, 3): one color per point
        n_points: Number of points

    Returns:
        Array of shape (n_points, 3) with float32 dtype, or None

    Raises:
        ValueError: If colors have invalid shape
    """
    if colors is None:
        return None

    if isinstance(colors, (list, tuple)) and len(colors) == 3:
        # Single RGB color - broadcast to all points
        color_array = np.array(colors, dtype=np.float32)
        return np.tile(color_array, (n_points, 1))

    colors_array = np.asarray(colors, dtype=np.float32)

    if colors_array.shape == (3,):
        # Single RGB color as numpy array
        return np.tile(colors_array, (n_points, 1))
    elif colors_array.shape == (n_points, 3):
        # Full color array
        return colors_array
    else:
        raise ValueError(
            f"Colors must have shape (3,) for single color or ({n_points}, 3) "
            f"for per-point colors, got shape {colors_array.shape}"
        )


def broadcast_scalar_to_points(
    values: Union[float, NDArray[np.float32], None],
    n_points: int,
    name: str = "values",
    require_positive: bool = True,
) -> Optional[NDArray[np.float32]]:
    """Broadcast a scalar value or array to all points.

    Args:
        values: Can be:
            - None: No values
            - Scalar: Single value broadcast to all points
            - Array shape (n_points,): one value per point
        n_points: Number of points
        name: Name of the values for error messages
        require_positive: Whether to require all values > 0

    Returns:
        Array of shape (n_points,) with float32 dtype, or None

    Raises:
        ValueError: If values have invalid shape or negative values
    """
    if values is None:
        return None

    if np.isscalar(values):
        # Single value - broadcast to all points
        value = float(values)  # type: ignore[arg-type]
        if require_positive and value <= 0:
            raise ValueError(f"{name.capitalize()} must be positive, got {value}")
        return np.full(n_points, value, dtype=np.float32)

    values_array = np.asarray(values, dtype=np.float32)

    if values_array.shape != (n_points,):
        raise ValueError(
            f"{name.capitalize()} must be scalar or have shape ({n_points},), "
            f"got shape {values_array.shape}"
        )

    if require_positive and np.any(values_array <= 0):
        min_val = np.min(values_array)
        raise ValueError(
            f"All {name} values must be positive, found minimum value: {min_val}"
        )

    return values_array


def broadcast_radii_to_points(
    radii: Union[float, RadiusArray, None], n_points: int
) -> Optional[RadiusArray]:
    """Broadcast radii to all points.

    Args:
        radii: Single radius or array of radii
        n_points: Number of points

    Returns:
        Array of shape (n_points,) with radii, or None
    """
    return broadcast_scalar_to_points(radii, n_points, "radii", require_positive=True)


def broadcast_sharpness_to_points(
    sharpness: Union[float, SharpnessArray, None],
    n_points: int,
    warn_on_out_of_range: bool = True,
) -> Optional[SharpnessArray]:
    """Broadcast sharpness values to all points.

    Args:
        sharpness: Single sharpness or array of sharpness values
        n_points: Number of points
        warn_on_out_of_range: Whether to warn if values are outside typical range

    Returns:
        Array of shape (n_points,) with sharpness values, or None
    """
    result = broadcast_scalar_to_points(
        sharpness, n_points, "sharpness", require_positive=True
    )

    if warn_on_out_of_range and result is not None:
        out_of_range = np.any(
            (result < SHARPNESS_TYPICAL_MIN) | (result > SHARPNESS_TYPICAL_MAX)
        )
        if out_of_range:
            import warnings

            min_val = np.min(result)
            max_val = np.max(result)
            warnings.warn(
                f"Sharpness values outside typical range "
                f"[{SHARPNESS_TYPICAL_MIN}, {SHARPNESS_TYPICAL_MAX}]: "
                f"min={min_val:.2f}, max={max_val:.2f}",
                UserWarning,
            )

    return result


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
