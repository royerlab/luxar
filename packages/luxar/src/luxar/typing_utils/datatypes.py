"""Data type configuration and utilities for Luxar.

This module provides data type mappings, conversions, and validation
for efficient storage of points attributes in zarr format.
"""

from enum import Enum
from typing import Any, Literal, Optional, Union, get_args

import numpy as np
from numpy.typing import DTypeLike, NDArray

# Type aliases for supported numpy dtypes
PositionDType = Union[np.float32, np.float16]
ColorDType = Union[np.float32, np.uint8, np.uint16]
ScalarDType = Union[np.float32, np.float16, np.uint8]

# String literals for dtype specifications
PositionDTypeStr = Literal["float32", "float16"]
ColorDTypeStr = Literal["float32", "uint8", "uint16"]
ScalarDTypeStr = Literal["float32", "float16", "uint8"]


class DataTypeMode(str, Enum):
    """Data type selection modes."""

    AUTO = "auto"  # Automatically select based on data range
    PRECISION = "precision"  # Prioritize precision (use float32)
    MEMORY = "memory"  # Prioritize memory efficiency
    CUSTOM = "custom"  # Use explicitly specified dtypes


class DataTypeConfig:
    """Configuration for data types used in zarr storage."""

    def __init__(
        self,
        mode: DataTypeMode = DataTypeMode.AUTO,
        position_dtype: Optional[PositionDTypeStr] = None,
        color_dtype: Optional[ColorDTypeStr] = None,
        radius_dtype: Optional[ScalarDTypeStr] = None,
        sharpness_dtype: Optional[ScalarDTypeStr] = None,
    ) -> None:
        """Initialize data type configuration.

        Args:
            mode: Data type selection mode
            position_dtype: Explicit dtype for positions (used in CUSTOM mode)
            color_dtype: Explicit dtype for colors (used in CUSTOM mode)
            radius_dtype: Explicit dtype for radii (used in CUSTOM mode)
            sharpness_dtype: Explicit dtype for sharpness (used in CUSTOM mode)
        """
        self.mode = mode
        self._position_dtype = position_dtype
        self._color_dtype = color_dtype
        self._radius_dtype = radius_dtype
        self._sharpness_dtype = sharpness_dtype

        # Validate custom mode has required dtypes
        if mode == DataTypeMode.CUSTOM:
            if not all([position_dtype, color_dtype, radius_dtype, sharpness_dtype]):
                raise ValueError(
                    "CUSTOM mode requires all dtypes to be explicitly specified"
                )

    def get_position_dtype(self, data: Optional[NDArray] = None) -> DTypeLike:
        """Get the dtype to use for positions.

        Args:
            data: Optional data array to analyze for AUTO mode

        Returns:
            Numpy dtype to use for positions
        """
        if self.mode == DataTypeMode.CUSTOM:
            return np.dtype(self._position_dtype)
        elif self.mode == DataTypeMode.PRECISION:
            return np.float32
        elif self.mode == DataTypeMode.MEMORY:
            return np.float16
        else:  # AUTO
            # For positions, default to float32 for accuracy
            # Could analyze coordinate range in future versions
            return np.float32

    def get_color_dtype(self, data: Optional[NDArray] = None) -> DTypeLike:
        """Get the dtype to use for colors.

        Args:
            data: Optional data array to analyze for AUTO mode

        Returns:
            Numpy dtype to use for colors
        """
        if self.mode == DataTypeMode.CUSTOM:
            return np.dtype(self._color_dtype)
        elif self.mode == DataTypeMode.PRECISION:
            return np.float32
        elif self.mode == DataTypeMode.MEMORY:
            return np.uint8
        else:  # AUTO
            if data is not None:
                # Check if HDR (values > 1.0)
                if np.any(data > 1.0):
                    return np.float32  # HDR colors
                elif np.all((data >= 0) & (data <= 1.0)):
                    return np.uint8  # Standard colors, can be normalized
                else:
                    return np.float32  # Safe default
            return np.uint8  # Default to uint8 for memory efficiency

    def get_radius_dtype(self, data: Optional[NDArray] = None) -> DTypeLike:
        """Get the dtype to use for radii.

        Args:
            data: Optional data array to analyze for AUTO mode

        Returns:
            Numpy dtype to use for radii
        """
        if self.mode == DataTypeMode.CUSTOM:
            return np.dtype(self._radius_dtype)
        elif self.mode == DataTypeMode.PRECISION:
            return np.float32
        elif self.mode == DataTypeMode.MEMORY:
            return np.uint8  # Will be normalized to 0-1 range
        else:  # AUTO
            if data is not None:
                # Check data range for appropriate dtype
                max_val: float = float(np.max(np.abs(data)))
                if max_val <= 1.0:
                    return np.uint8  # Can use normalized uint8
                elif max_val < 1000:
                    return np.float16  # Float16 has sufficient range
                else:
                    return np.float32  # Need full float32 range
            return np.float32  # Safe default

    def get_sharpness_dtype(self, data: Optional[NDArray] = None) -> DTypeLike:
        """Get the dtype to use for sharpness.

        Args:
            data: Optional data array to analyze for AUTO mode

        Returns:
            Numpy dtype to use for sharpness
        """
        if self.mode == DataTypeMode.CUSTOM:
            return np.dtype(self._sharpness_dtype)
        elif self.mode == DataTypeMode.PRECISION:
            return np.float32
        elif self.mode == DataTypeMode.MEMORY:
            return np.uint8  # Will be mapped to [0, 15] range for efficiency
        else:  # AUTO
            # Sharpness values in range [0, 15] are efficiently stored as uint8
            if data is not None:
                from .constants import SHARPNESS_MAX

                max_val: float = float(np.max(np.abs(data)))
                if max_val <= SHARPNESS_MAX:
                    return np.uint8  # Use uint8 with [0, 15] → [0, 255] mapping
                elif max_val < 1000:
                    return np.float16  # Float16 has sufficient range
                else:
                    return np.float32  # Need full float32 range
            return np.float32  # Safe default


def convert_array_dtype(
    array: NDArray,
    target_dtype: DTypeLike,
    normalize: bool = False,
    input_range: tuple[float, float] = (0.0, 1.0),
) -> NDArray:
    """Convert array to target dtype with optional normalization.

    Args:
        array: Input array to convert
        target_dtype: Target numpy dtype
        normalize: Whether to normalize when converting to/from integer types
        input_range: Range of input values for normalization (min, max)

    Returns:
        Converted array with target dtype
    """
    target_dtype = np.dtype(target_dtype)
    source_dtype = array.dtype

    # No conversion needed
    if source_dtype == target_dtype:
        return array

    # Float to uint8 conversion (normalize to 0-255)
    if source_dtype.kind == "f" and target_dtype == np.uint8:
        if normalize:
            # Map input range to 0-255 range
            input_min, input_max = input_range
            input_span = input_max - input_min
            if input_span == 0:
                # Handle degenerate case
                return np.full_like(array, 0, dtype=np.uint8)
            # Normalize input to 0-1, then scale to 0-255
            normalized = (array - input_min) / input_span
            return np.clip(normalized * 255, 0, 255).astype(np.uint8)
        else:
            # Direct conversion, clip to valid range
            return np.clip(array, 0, 255).astype(np.uint8)

    # Float to uint16 conversion
    if source_dtype.kind == "f" and target_dtype == np.uint16:
        if normalize:
            # Assume input is 0-1 range, scale to 0-65535
            return np.clip(array * 65535, 0, 65535).astype(np.uint16)
        else:
            # Direct conversion, clip to valid range
            return np.clip(array, 0, 65535).astype(np.uint16)

    # Uint8 to float conversion
    if source_dtype == np.uint8 and target_dtype.kind == "f":
        if normalize:
            # Convert 0-255 to specified output range
            input_min, input_max = input_range
            # First convert 0-255 to 0-1, then scale to output range
            normalized = array.astype(target_dtype) / 255.0
            return normalized * (input_max - input_min) + input_min
        else:
            return array.astype(target_dtype)

    # Uint16 to float conversion
    if source_dtype == np.uint16 and target_dtype.kind == "f":
        if normalize:
            # Convert 0-65535 to 0-1 range
            return array.astype(target_dtype) / 65535.0
        else:
            return array.astype(target_dtype)

    # Float16 <-> Float32 conversion
    if source_dtype.kind == "f" and target_dtype.kind == "f":
        return array.astype(target_dtype)

    # Default: direct conversion
    return array.astype(target_dtype)


def infer_optimal_dtype(
    array: NDArray, attribute_type: Literal["position", "color", "radius", "sharpness"]
) -> DTypeLike:
    """Infer the optimal dtype for an array based on its values and attribute type.

    Args:
        array: Input array to analyze
        attribute_type: Type of attribute (affects dtype selection logic)

    Returns:
        Optimal numpy dtype for the array
    """
    if attribute_type == "position":
        # Positions typically need good precision
        # Could use float16 for small-scale data, but float32 is safer
        return np.float32

    elif attribute_type == "color":
        # Check if HDR (values > 1.0)
        if np.any(array > 1.0):
            return np.float32  # HDR colors need float32
        elif np.all((array >= 0) & (array <= 1.0)):
            return np.uint8  # Standard colors can use uint8
        else:
            return np.float32  # Safe default for unusual ranges

    elif attribute_type in ["radius", "sharpness"]:
        # Check data range
        max_val: float = float(np.max(np.abs(array)))
        min_val: float = float(np.min(array))

        if min_val >= 0 and max_val <= 1.0:
            return np.uint8  # Can use normalized uint8
        elif max_val < 65536 and min_val > -65536:
            return np.float16  # Float16 has sufficient range
        else:
            return np.float32  # Need full float32 range

    return np.float32  # Safe default


def get_dtype_info(dtype: DTypeLike) -> dict[str, Any]:
    """Get information about a numpy dtype.

    Args:
        dtype: Numpy dtype to analyze

    Returns:
        Dictionary with dtype information including:
        - name: String name of the dtype
        - bytes: Number of bytes per element
        - kind: Kind of data (f=float, u=unsigned, i=signed)
        - range: Min and max representable values
        - normalized: Whether this dtype uses normalization in WebGL
    """
    dtype = np.dtype(dtype)

    info = {
        "name": dtype.name,
        "bytes": dtype.itemsize,
        "kind": dtype.kind,
        "normalized": False,  # Will be set for integer types
    }

    # Calculate range based on dtype
    if dtype == np.float32:
        info["range"] = (-3.4e38, 3.4e38)
    elif dtype == np.float16:
        info["range"] = (-65504.0, 65504.0)
    elif dtype == np.uint8:
        info["range"] = (0, 255)
        info["normalized"] = True  # WebGL can normalize to 0-1
    elif dtype == np.uint16:
        info["range"] = (0, 65535)
        info["normalized"] = True  # WebGL can normalize to 0-1
    elif dtype == np.int8:
        info["range"] = (-128, 127)
    elif dtype == np.int16:
        info["range"] = (-32768, 32767)
    else:
        info["range"] = (None, None)

    return info


# Default configurations for common use cases
DEFAULT_CONFIG = DataTypeConfig(mode=DataTypeMode.AUTO)
PRECISION_CONFIG = DataTypeConfig(mode=DataTypeMode.PRECISION)
MEMORY_CONFIG = DataTypeConfig(mode=DataTypeMode.MEMORY)


def validate_dtype_string(dtype_str: str, attribute_type: str) -> bool:
    """Validate that a dtype string is valid for a given attribute type.

    Args:
        dtype_str: String representation of the dtype
        attribute_type: Type of attribute ("position", "color", "radius", "sharpness")

    Returns:
        True if the dtype is valid for the attribute type

    Raises:
        ValueError: If the dtype is not valid for the attribute type
    """
    valid_dtypes = {
        "position": get_args(PositionDTypeStr),
        "color": get_args(ColorDTypeStr),
        "radius": get_args(ScalarDTypeStr),
        "sharpness": get_args(ScalarDTypeStr),
    }

    if attribute_type not in valid_dtypes:
        raise ValueError(f"Unknown attribute type: {attribute_type}")

    if dtype_str not in valid_dtypes[attribute_type]:
        raise ValueError(
            f"Invalid dtype '{dtype_str}' for {attribute_type}. "
            f"Valid options: {valid_dtypes[attribute_type]}"
        )

    return True
