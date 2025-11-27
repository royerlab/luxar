"""Validation utilities with helpful error messages.

This module provides validation functions with detailed, user-friendly error
messages that help users understand and fix issues quickly.
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..typing_utils.constants import (
    SHARPNESS_MAX,
    SHARPNESS_MIN,
)


class ValidationError(ValueError):
    """Custom validation error with helpful suggestions."""

    def __init__(self, message: str, suggestion: Optional[str] = None) -> None:
        """Initialize validation error.

        Args:
            message: The error message
            suggestion: Optional suggestion for fixing the error
        """
        full_message = message
        if suggestion:
            full_message += f"\n💡 Suggestion: {suggestion}"
        super().__init__(full_message)


def validate_positions_for_writing(
    positions: NDArray[np.float32], context: str = "positions"
) -> Tuple[int, int]:
    """Validate positions array for writing to Zarr.

    Args:
        positions: Positions array to validate
        context: Context for error messages

    Returns:
        Tuple of (n_points, n_dims)

    Raises:
        ValidationError: If positions are invalid
    """
    if not isinstance(positions, np.ndarray):
        raise ValidationError(
            f"{context}: Expected numpy array, got {type(positions).__name__}",
            "Convert your data to a numpy array using np.array(data)",
        )

    if positions.ndim != 2:
        if positions.ndim == 1:
            raise ValidationError(
                f"{context}: Got 1D array with {len(positions)} elements. "
                f"Positions must be a 2D array with shape (n_points, n_dimensions).",
                f"If you have {len(positions)} 1D points, reshape with: positions.reshape(-1, 1)",
            )
        elif positions.ndim == 3:
            raise ValidationError(
                f"{context}: Got 3D array with shape {positions.shape}. "
                "Positions must be a 2D array with shape (n_points, n_dimensions).",
                "If you have multiple time steps, flatten them or use StreamingPoints",
            )
        else:
            raise ValidationError(
                f"{context}: Expected 2D array, got {positions.ndim}D array with shape {positions.shape}"
            )

    n_points, n_dims = positions.shape

    if n_points == 0:
        raise ValidationError(
            f"{context}: Cannot write empty points (0 points)",
            "Ensure your positions array contains at least one point",
        )

    if n_dims == 0:
        raise ValidationError(
            f"{context}: Points have 0 dimensions",
            "Each point must have at least 1 dimension (e.g., 1D, 2D, 3D)",
        )

    if n_dims > 10:
        import warnings

        warnings.warn(
            f"{context}: Writing {n_dims}D points. "
            "Visualization may be limited to the first 3 dimensions.",
            UserWarning,
        )

    return n_points, n_dims


def validate_colors_for_writing(
    colors: NDArray, n_points: int, context: str = "colors"
) -> None:
    """Validate colors array for writing.

    Args:
        colors: Colors array to validate
        n_points: Expected number of points
        context: Context for error messages

    Raises:
        ValidationError: If colors are invalid
    """
    if not isinstance(colors, np.ndarray):
        raise ValidationError(
            f"{context}: Expected numpy array, got {type(colors).__name__}",
            "Convert colors to numpy array: np.array(colors)",
        )

    expected_shape = (n_points, 3)

    if colors.shape != expected_shape:
        if colors.ndim == 1 and len(colors) == 3:
            raise ValidationError(
                f"{context}: Got single RGB color {colors.shape}. "
                f"Expected shape {expected_shape} for {n_points} points.",
                "For a single color, use Scene.add_points() which handles broadcasting",
            )
        elif colors.ndim == 2 and colors.shape[1] != 3:
            raise ValidationError(
                f"{context}: Colors must have 3 channels (RGB), got {colors.shape[1]} channels",
                "Ensure colors have shape (n_points, 3) for RGB values",
            )
        elif colors.shape[0] != n_points:
            raise ValidationError(
                f"{context}: Number of colors ({colors.shape[0]}) doesn't match "
                f"number of points ({n_points})",
                f"Provide exactly {n_points} colors or use a single color for all points",
            )
        else:
            raise ValidationError(
                f"{context}: Expected shape {expected_shape}, got {colors.shape}"
            )

    # Check for invalid values
    if np.any(colors < 0):
        min_val: float = float(np.min(colors))
        raise ValidationError(
            f"{context}: Colors cannot be negative. Found minimum value: {min_val:.3f}",
            "Ensure all color values are >= 0. Use np.clip(colors, 0, None) to fix",
        )

    # Warn about extreme HDR values
    max_val: float = float(np.max(colors))
    if max_val > 10.0:
        import warnings

        warnings.warn(
            f"{context}: HDR colors with maximum value {max_val:.1f} detected. "
            "Values > 10.0 may cause display issues.",
            UserWarning,
        )


def validate_radii_for_writing(
    radii: NDArray, n_points: int, context: str = "radii"
) -> None:
    """Validate radii array for writing.

    Args:
        radii: Radii array to validate
        n_points: Expected number of points
        context: Context for error messages

    Raises:
        ValidationError: If radii are invalid
    """
    if not isinstance(radii, np.ndarray):
        raise ValidationError(
            f"{context}: Expected numpy array, got {type(radii).__name__}",
            "Convert to numpy array: np.array(radii)",
        )

    if radii.ndim != 1:
        if radii.ndim == 2 and radii.shape[1] == 1:
            raise ValidationError(
                f"{context}: Got 2D array with shape {radii.shape}. Radii must be 1D.",
                "Flatten with: radii.ravel() or radii[:, 0]",
            )
        else:
            raise ValidationError(
                f"{context}: Expected 1D array, got {radii.ndim}D array with shape {radii.shape}",
                "Radii must be a 1D array with one value per point",
            )

    if len(radii) != n_points:
        raise ValidationError(
            f"{context}: Number of radii ({len(radii)}) doesn't match number of points ({n_points})",
            f"Provide exactly {n_points} radii values or use a single radius for all points",
        )

    # Check for invalid values
    if np.any(radii <= 0):
        min_val: float = float(np.min(radii))
        if min_val == 0:
            raise ValidationError(
                f"{context}: Radii must be positive (> 0). Found zero values.",
                "Replace zeros with small positive values: radii[radii == 0] = 0.01",
            )
        else:
            raise ValidationError(
                f"{context}: Radii must be positive. Found minimum value: {min_val:.3f}",
                "Use np.abs(radii) or np.clip(radii, 0.01, None) to ensure positive values",
            )


def validate_sharpness_for_writing(
    sharpness: NDArray, n_points: int, context: str = "sharpness"
) -> None:
    """Validate sharpness array for writing.

    Args:
        sharpness: Sharpness array to validate
        n_points: Expected number of points
        context: Context for error messages

    Raises:
        ValidationError: If sharpness values are invalid
    """
    if not isinstance(sharpness, np.ndarray):
        raise ValidationError(
            f"{context}: Expected numpy array, got {type(sharpness).__name__}",
            "Convert to numpy array: np.array(sharpness)",
        )

    if sharpness.ndim != 1:
        if sharpness.ndim == 2 and sharpness.shape[1] == 1:
            raise ValidationError(
                f"{context}: Got 2D array with shape {sharpness.shape}. Sharpness must be 1D.",
                "Flatten with: sharpness.ravel() or sharpness[:, 0]",
            )
        else:
            raise ValidationError(
                f"{context}: Expected 1D array, got {sharpness.ndim}D array with shape {sharpness.shape}",
                "Sharpness must be a 1D array with one value per point",
            )

    if len(sharpness) != n_points:
        raise ValidationError(
            f"{context}: Number of sharpness values ({len(sharpness)}) doesn't match "
            f"number of points ({n_points})",
            f"Provide exactly {n_points} sharpness values or use a single value for all points",
        )

    # Check for invalid values
    if np.any(sharpness <= 0):
        min_val: float = float(np.min(sharpness))
        raise ValidationError(
            f"{context}: Sharpness must be positive. Found minimum value: {min_val:.3f}",
            f"Use values between {SHARPNESS_MIN} and {SHARPNESS_MAX} for valid range",
        )

    # Check for values exceeding the maximum allowed (for uint8 mapping)
    if np.any(sharpness > SHARPNESS_MAX):
        max_val: float = float(np.max(sharpness))
        raise ValidationError(
            f"{context}: Sharpness values exceed maximum allowed value ({SHARPNESS_MAX}). "
            f"Found maximum: {max_val:.3f}",
            f"Clip values to valid range: np.clip(sharpness, 0, {SHARPNESS_MAX})",
        )

    # Optional: Warn about extreme values that might not look good
    if np.any(sharpness < 0.5) or np.any(sharpness > 10.0):
        import warnings

        min_val = np.min(sharpness)
        max_val = np.max(sharpness)
        warnings.warn(
            f"{context}: Using extreme sharpness values.\n"
            f"  Your range: [{min_val:.2f}, {max_val:.2f}]\n"
            f"  Values < 0.5 create very soft, blurry points\n"
            f"  Values > 10.0 create very hard-edged points\n"
            f"  Valid range: [{SHARPNESS_MIN}, {SHARPNESS_MAX}]",
            UserWarning,
        )


def validate_zarr_attributes(attrs: dict, is_root: bool = False) -> None:
    """Validate that all required Zarr attributes are present.

    Ensures that zarr groups have the required metadata attributes according
    to the Luxar Zarr format specification.

    Args:
        attrs: Dictionary of zarr attributes
        is_root: Whether this is the root scene group

    Raises:
        ValidationError: If required attributes are missing or invalid
    """
    if is_root:
        # Root scene requires additional attributes
        required = {"type", "luxar_version"}
        recommended = {"scene_dimensions"}
    else:
        # Child nodes only require type
        required = {"type"}
        recommended = set()

    # Check for missing required attributes
    missing_required = required - set(attrs.keys())
    if missing_required:
        raise ValidationError(
            f"Missing required zarr attributes: {missing_required}",
            f"Add these attributes: {', '.join(missing_required)}",
        )

    # Check for missing recommended attributes
    missing_recommended = recommended - set(attrs.keys())
    if missing_recommended:
        import warnings

        warnings.warn(
            f"Missing recommended zarr attributes: {missing_recommended}. "
            f"Consider adding these for better compatibility.",
            UserWarning,
        )

    # Validate type attribute
    if "type" in attrs:
        valid_types = {"scene", "group", "points"}
        if attrs["type"] not in valid_types:
            raise ValidationError(
                f"Invalid node type: '{attrs['type']}'",
                f"Use one of: {', '.join(valid_types)}",
            )

    # Validate version if present
    if "luxar_version" in attrs:
        from ..typing_utils.config import SUPPORTED_VERSIONS

        if attrs["luxar_version"] not in SUPPORTED_VERSIONS:
            raise ValidationError(
                f"Unsupported Luxar version: '{attrs['luxar_version']}'",
                f"Supported versions: {', '.join(SUPPORTED_VERSIONS)}",
            )
