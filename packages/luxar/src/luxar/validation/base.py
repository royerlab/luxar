"""Validation utilities with helpful error messages.

This module provides validation functions with detailed, user-friendly error
messages that help users understand and fix issues quickly.
"""

from typing import Any, Optional, Tuple

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


def _validate_numeric_finite_values(array: NDArray[Any], context: str) -> None:
    """Validate that an array has numeric dtype and contains only finite values."""
    if not np.issubdtype(array.dtype, np.number):
        raise ValidationError(
            f"{context}: Expected numeric array, got dtype {array.dtype}",
            "Convert your data to a numeric dtype such as np.float32",
        )

    try:
        finite_mask = np.isfinite(array)
    except TypeError as exc:
        raise ValidationError(
            f"{context}: Expected numeric finite values, got dtype {array.dtype}",
            "Convert your data to a numeric dtype and remove invalid values",
        ) from exc

    if not bool(np.all(finite_mask)):
        invalid_count = int(np.size(array) - np.count_nonzero(finite_mask))
        raise ValidationError(
            f"{context}: Contains {invalid_count} NaN or Inf value(s)",
            "Remove or replace invalid values before writing, e.g. np.nan_to_num(data)",
        )


def validate_positions_for_writing(
    positions: NDArray[Any], context: str = "positions"
) -> Tuple[int, int]:
    """Validate positions array for writing to Zarr.

    Args:
        positions: Positions array to validate
        context: Context for error messages

    Returns:
        Tuple of (n_points, n_dims). NOTE the array itself is NOT
        returned and is NOT dtype-converted: the validator deliberately
        accepts any numeric dtype (float32/float64/int) and verifies
        SHAPE + FINITENESS only. Callers that need float32 storage MUST
        convert via ``ensure_float32`` (or equivalent) AFTER calling
        this function — see the ``test_validate_positions_dtype_not_checked``
        regression-lock test for the pinned contract.

    Raises:
        ValidationError: If positions are invalid (wrong shape, empty,
            zero-dimensional, or containing NaN / ±Inf).
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
                "If you have multiple time steps, split them into separate nodes (e.g., points_t0, points_t1, ...)",
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

    _validate_numeric_finite_values(positions, context)

    if n_dims > 10:
        import warnings

        warnings.warn(
            f"{context}: Writing {n_dims}D points. "
            "Visualization may be limited to the first 3 dimensions.",
            UserWarning,
        )

    return n_points, n_dims


def validate_colors_for_writing(
    colors: NDArray[Any], n_points: int, context: str = "colors"
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

    # Allow broadcast shape (1, 3) or full shape (n_points, 3)
    broadcast_shape = (1, 3)
    expected_shape = (n_points, 3)

    if colors.shape != expected_shape and colors.shape != broadcast_shape:
        if colors.ndim == 1 and len(colors) == 3:
            raise ValidationError(
                f"{context}: Got single RGB color {colors.shape}. "
                f"Expected shape {expected_shape} for {n_points} points or {broadcast_shape} for broadcasting.",
                "For a single color, use shape (1, 3) for broadcasting",
            )
        elif colors.ndim == 2 and colors.shape[1] != 3:
            raise ValidationError(
                f"{context}: Colors must have 3 channels (RGB), got {colors.shape[1]} channels",
                "Ensure colors have shape (n_points, 3) or (1, 3) for broadcasting",
            )
        elif colors.shape[0] != n_points and colors.shape[0] != 1:
            raise ValidationError(
                f"{context}: Number of colors ({colors.shape[0]}) doesn't match "
                f"number of points ({n_points}) and is not 1 (broadcast)",
                f"Provide exactly {n_points} colors or (1, 3) for broadcasting",
            )
        else:
            raise ValidationError(
                f"{context}: Expected shape {expected_shape} or {broadcast_shape}, got {colors.shape}"
            )

    _validate_numeric_finite_values(colors, context)

    if colors.size == 0:
        if n_points == 0 and colors.shape == expected_shape:
            return
        raise ValidationError(
            f"{context}: Empty colors array is only valid when n_points=0 "
            f"and shape is {expected_shape}. Got shape {colors.shape}.",
            "Provide one broadcast color with shape (1, 3) or a full colors array",
        )

    # Check for invalid values
    if np.any(colors < 0):
        min_val: float = float(np.min(colors))
        raise ValidationError(
            f"{context}: Colors cannot be negative. Found minimum value: {min_val:.3f}",
            "Ensure all color values are >= 0. Use np.clip(colors, 0, None) to fix",
        )

    # Warn about extreme HDR values for floating-point HDR/SDR colors.
    # Integer color arrays are SDR storage in their native integer range.
    max_val: float = float(np.max(colors))
    if np.issubdtype(colors.dtype, np.floating) and max_val > 10.0:
        import warnings

        warnings.warn(
            f"{context}: HDR colors with maximum value {max_val:.1f} detected. "
            "Values > 10.0 may cause display issues.",
            UserWarning,
        )


def validate_radii_for_writing(
    radii: NDArray[Any], n_points: int, context: str = "radii"
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

    # Allow broadcast shape (1,) or full shape (n_points,)
    if len(radii) != n_points and len(radii) != 1:
        raise ValidationError(
            f"{context}: Number of radii ({len(radii)}) doesn't match "
            f"number of points ({n_points}) and is not 1 (broadcast)",
            f"Provide exactly {n_points} radii values or use shape (1,) for broadcasting",
        )

    _validate_numeric_finite_values(radii, context)

    # Check for invalid values
    if np.any(radii <= 0):
        min_val: float = float(np.min(radii))
        raise ValidationError(
            f"{context}: Radii must be positive (> 0). "
            f"Found zero or negative values (minimum: {min_val:.3f}).",
            "Use np.clip(radii, 0.01, None) to ensure positive values",
        )


def validate_widths_for_writing(
    widths: Any, n_vertices: int, context: str = "widths"
) -> None:
    """Validate line widths for writing.

    Accepts a per-vertex 1D array of shape ``(n_vertices,)``, a broadcast
    array of shape ``(1,)``, or a scalar width. The Lines sibling of
    :func:`validate_radii_for_writing` (Points), per the three-geometry
    symmetry rule.

    Args:
        widths: Per-vertex widths array, broadcast ``(1,)`` array, or scalar
        n_vertices: Expected number of vertices
        context: Context for error messages

    Raises:
        ValidationError: If widths are invalid
    """
    if isinstance(widths, np.ndarray):
        if widths.ndim != 1:
            raise ValidationError(
                f"{context}: Expected 1D array, got {widths.ndim}D array "
                f"with shape {widths.shape}",
                "Widths must be a 1D array with one value per vertex",
            )

        # Allow broadcast shape (1,) or full shape (n_vertices,) — mirrors radii
        if widths.shape[0] != n_vertices and widths.shape[0] != 1:
            raise ValidationError(
                f"{context}: Widths shape {widths.shape} doesn't match "
                f"n_vertices {n_vertices} and is not 1 (broadcast)",
                f"Provide exactly {n_vertices} width values or use shape (1,) "
                f"for broadcasting",
            )

        _validate_numeric_finite_values(widths, context)

        if np.any(widths <= 0):
            min_val: float = float(np.min(widths))
            raise ValidationError(
                f"{context}: Widths must be positive (> 0). "
                f"Found minimum value: {min_val:.3f}",
                "Use np.clip(widths, 0.01, None) to ensure positive values",
            )
    elif isinstance(widths, (int, float)):
        if not np.isfinite(widths):
            raise ValidationError(
                f"{context}: Width must be finite. Got {widths}",
                "Provide a finite positive width value",
            )
        if widths <= 0:
            raise ValidationError(
                f"{context}: Width must be positive (> 0). Got {widths}",
                "Provide a positive width value",
            )


def validate_sharpness_for_writing(
    sharpness: NDArray[Any], n_points: int, context: str = "sharpness"
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

    # Allow broadcast shape (1,) or full shape (n_points,)
    if len(sharpness) != n_points and len(sharpness) != 1:
        raise ValidationError(
            f"{context}: Number of sharpness values ({len(sharpness)}) doesn't match "
            f"number of points ({n_points}) and is not 1 (broadcast)",
            f"Provide exactly {n_points} sharpness values or use shape (1,) for broadcasting",
        )

    _validate_numeric_finite_values(sharpness, context)

    # Sharpness is a normalised [0, 1] knob mapped in the viewer to the
    # super-Gaussian falloff exponent beta = 2^(6s - 2): s=0.5 -> beta=2 (a
    # true Gaussian), higher s -> harder edge, lower s -> peakier cusp.
    if np.any(sharpness < SHARPNESS_MIN):
        min_val: float = float(np.min(sharpness))
        raise ValidationError(
            f"{context}: Sharpness must be >= {SHARPNESS_MIN}. Found minimum value: {min_val:.3f}",
            f"Use values between {SHARPNESS_MIN} and {SHARPNESS_MAX} for valid range",
        )

    if np.any(sharpness > SHARPNESS_MAX):
        max_val: float = float(np.max(sharpness))
        raise ValidationError(
            f"{context}: Sharpness values exceed maximum allowed value ({SHARPNESS_MAX}). "
            f"Found maximum: {max_val:.3f}",
            f"Clip values to valid range: np.clip(sharpness, {SHARPNESS_MIN}, {SHARPNESS_MAX})",
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
        valid_types = {"scene", "group", "points", "lines", "gsplats"}
        if attrs["type"] not in valid_types:
            raise ValidationError(
                f"Invalid node type: '{attrs['type']}'",
                f"Use one of: {', '.join(valid_types)}",
            )

    # Validate version if present. Keep this import local: typing_utils.config
    # itself imports io.reader.DEFAULT_COMP via a lazy/inline path, and io.reader
    # transitively imports core.dimensions which imports validation.category_validation.
    # validation.base is imported early enough that an unconditional top-level
    # import here would risk re-entering this module via that chain.
    if "luxar_version" in attrs:
        from ..typing_utils.config import SUPPORTED_VERSIONS

        if attrs["luxar_version"] not in SUPPORTED_VERSIONS:
            raise ValidationError(
                f"Unsupported Luxar version: '{attrs['luxar_version']}'",
                f"Supported versions: {', '.join(SUPPORTED_VERSIONS)}",
            )
