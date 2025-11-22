"""Type validation functions for runtime type checking.

This module contains basic validation functions that are used for type guards,
property validation, and runtime type checking. For detailed write-time validation
with helpful error messages, see validation/base.py.

These validators are simpler and focus on type conversion and basic checks,
while the _for_writing validators provide comprehensive error messages for
users writing data.
"""

from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np

from ..typing_utils.aliases import PositionArray, TransformMatrix
from ..typing_utils.constants import GAMMA_MAX, GAMMA_MIN, OPACITY_MAX, OPACITY_MIN
from ..typing_utils.enums import BlendingMode, NodeType, PhysicalUnit


def validate_positions(positions: Any, ndim: Optional[int] = None) -> PositionArray:
    """Validate and convert positions array to correct type.

    Args:
        positions: Input array to validate
        ndim: Expected number of dimensions (optional). If None, any dimensionality is accepted.

    Returns:
        Validated positions array with shape (N, D)

    Raises:
        ValueError: If positions are invalid shape or type
    """
    if not isinstance(positions, np.ndarray):
        raise ValueError("Positions must be a numpy array")

    if positions.ndim != 2:
        raise ValueError(
            f"Positions must have shape (N, D), got shape {positions.shape}"
        )

    if positions.shape[1] < 1:
        raise ValueError(
            f"Positions must have at least 1 dimension, got {positions.shape[1]}"
        )

    if ndim is not None and positions.shape[1] != ndim:
        raise ValueError(f"Expected {ndim} dimensions, got {positions.shape[1]}")

    return positions.astype(np.float32, copy=False)


def validate_colors(
    colors: Any, n_points: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Validate and convert colors array to HDR float32 format.

    Args:
        colors: Input array to validate
        n_points: Expected number of points

    Returns:
        Validated colors array in HDR float32 format

    Raises:
        ValueError: If colors are invalid shape or type
    """
    if not isinstance(colors, np.ndarray):
        raise ValueError("Colors must be a numpy array")

    if colors.shape != (n_points, 3):
        raise ValueError(f"Colors must have shape ({n_points}, 3)")

    # Support HDR colors - use float32 for full HDR range
    # Colors can be any positive value (0.0 to infinity) for HDR emission
    return colors.astype(np.float32, copy=False)


def validate_radii(radii: Any, n_points: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Validate and convert radii array to correct type.

    Args:
        radii: Input array to validate
        n_points: Expected number of points

    Returns:
        Validated radii array

    Raises:
        ValueError: If radii are invalid shape or type
    """
    if not isinstance(radii, np.ndarray):
        raise ValueError("Radii must be a numpy array")
    if radii.ndim != 1:
        raise ValueError("Radii must have shape (N,)")
    if radii.shape[0] != n_points:
        raise ValueError(f"Radii shape {radii.shape} doesn't match positions")
    # Ensure all radii are positive
    if np.any(radii <= 0):
        raise ValueError("All radii must be positive values")
    return radii.astype(np.float32, copy=False)


def validate_sharpness(
    sharpness: Any, n_points: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Validate and convert sharpness array to correct type.

    Sharpness controls the falloff profile of points, from soft (low values) to sharp (high values).
    Must be positive float32 values with shape (N,) where N is the number of points.
    Typical range is 0.5 to 10.0.

    Args:
        sharpness: Input sharpness array to validate
        n_points: Expected number of points

    Returns:
        Validated sharpness array as float32

    Raises:
        ValueError: If sharpness values are invalid
    """
    if not isinstance(sharpness, np.ndarray):
        raise ValueError("Sharpness must be a numpy array")

    if sharpness.ndim != 1:
        raise ValueError("Sharpness must have shape (N,)")

    if sharpness.shape[0] != n_points:
        raise ValueError(f"Sharpness shape {sharpness.shape} doesn't match positions")

    if np.any(sharpness <= 0):
        raise ValueError("All sharpness values must be positive")

    # Warn if values are outside typical range
    if np.any(sharpness < 0.5) or np.any(sharpness > 10.0):
        import warnings

        warnings.warn(
            "Sharpness values outside typical range [0.5, 10.0] detected. "
            "Very low values (<0.5) create uniform disks, very high values (>10) create hard edges."
        )

    return sharpness.astype(np.float32, copy=False)


def validate_transform(transform: Any) -> TransformMatrix:
    """Validate and convert transform matrix to correct type.

    Args:
        transform: Input transform matrix

    Returns:
        Validated 4x4 transform matrix

    Raises:
        ValueError: If transform is invalid shape or type
    """
    if not isinstance(transform, np.ndarray):
        raise ValueError("Transform must be a numpy array")

    if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix")

    return transform.astype(np.float32, copy=False)


def validate_node_type(node_type: str) -> NodeType:
    """Validate node type string.

    Args:
        node_type: Input node type string

    Returns:
        Validated node type

    Raises:
        ValueError: If node type is invalid
    """
    valid_types = (NodeType.POINTS.value, NodeType.GROUP.value, NodeType.SCENE.value)
    if node_type not in valid_types:
        raise ValueError(
            f"Invalid node type '{node_type}'. Must be one of {valid_types}"
        )
    return cast(NodeType, node_type)


def validate_physical_unit(unit: str) -> PhysicalUnit:
    """Validate physical unit string.

    Args:
        unit: Input unit string

    Returns:
        Validated physical unit

    Raises:
        ValueError: If unit is invalid
    """
    valid_units = (
        PhysicalUnit.NANOMETER.value,
        PhysicalUnit.MICROMETER.value,
        PhysicalUnit.MILLIMETER.value,
        PhysicalUnit.CENTIMETER.value,
        PhysicalUnit.METER.value,
        PhysicalUnit.METRE.value,
        "meter",  # Not in enum but needed for compatibility
        PhysicalUnit.KILOMETER.value,
        PhysicalUnit.INCH.value,
        PhysicalUnit.FOOT.value,
        PhysicalUnit.PIXEL.value,
        PhysicalUnit.ASTRONOMICAL_UNIT.value,
    )
    if unit not in valid_units:
        raise ValueError(f"Invalid unit '{unit}'. Must be one of {valid_units}")
    return cast(PhysicalUnit, unit)


def validate_opacity(opacity: Any) -> float:
    """Validate and convert opacity value.

    Args:
        opacity: Value to validate as opacity (0.0 to 1.0)

    Returns:
        Valid opacity as float

    Raises:
        ValueError: If opacity is not a valid float between 0 and 1
        TypeError: If opacity cannot be converted to float
    """
    try:
        opacity_float = float(opacity)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Opacity must be convertible to float, got {type(opacity).__name__}"
        ) from e

    if not OPACITY_MIN <= opacity_float <= OPACITY_MAX:
        raise ValueError(
            f"Opacity must be between {OPACITY_MIN} and {OPACITY_MAX}, got {opacity_float}"
        )

    return opacity_float


def validate_gamma(gamma: Any) -> float:
    """Validate and convert gamma value.

    Args:
        gamma: Value to validate as gamma (0.2 to 2.0)

    Returns:
        Valid gamma as float

    Raises:
        ValueError: If gamma is not within valid range
        TypeError: If gamma cannot be converted to float
    """
    try:
        gamma_float = float(gamma)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Gamma must be convertible to float, got {type(gamma).__name__}"
        ) from e

    if not GAMMA_MIN <= gamma_float <= GAMMA_MAX:
        raise ValueError(
            f"Gamma must be between {GAMMA_MIN} and {GAMMA_MAX}, got {gamma_float}"
        )

    return gamma_float


def validate_blending_mode(mode: Any) -> BlendingMode:
    """Validate blending mode string.

    Args:
        mode: Blending mode to validate

    Returns:
        Valid blending mode

    Raises:
        ValueError: If mode is not a valid blending mode
        TypeError: If mode is not a string
    """
    if not isinstance(mode, str):
        raise TypeError(f"Blending mode must be a string, got {type(mode).__name__}")

    valid_modes = {"normal", "additive"}
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid blending mode '{mode}'. Must be one of: {', '.join(sorted(valid_modes))}"
        )

    return cast(BlendingMode, mode)


# Type guards (return bool for conditional type narrowing)
def is_position_array(obj: Any) -> bool:
    """Check if object is a valid position array."""
    try:
        validate_positions(obj)
        return True
    except (ValueError, TypeError):
        return False


def is_color_array(obj: Any, n_points: int) -> bool:
    """Check if object is a valid color array."""
    try:
        validate_colors(obj, n_points)
        return True
    except (ValueError, TypeError):
        return False


def is_transform_matrix(obj: Any) -> bool:
    """Check if object is a valid transform matrix."""
    try:
        validate_transform(obj)
        return True
    except (ValueError, TypeError):
        return False
