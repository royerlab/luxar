"""Constants used throughout Luxar.

This module centralizes all magic numbers and constants to improve
maintainability and provide clear documentation of their purposes.
"""

from typing import Final

# Version constants
LUXAR_VERSION_CURRENT: Final[str] = "0.2"
LUXAR_VERSION_LEGACY: Final[str] = "0.1"
LUXAR_VERSION_FUTURE: Final[str] = "0.3"

# Rendering constants
OPACITY_MIN: Final[float] = 0.0
OPACITY_MAX: Final[float] = 1.0
OPACITY_DEFAULT: Final[float] = 1.0

GAMMA_MIN: Final[float] = 0.2
GAMMA_MAX: Final[float] = 2.0
GAMMA_DEFAULT: Final[float] = 1.0

# Sharpness constants
SHARPNESS_MIN: Final[float] = 0.0  # Technical minimum (must be positive)
SHARPNESS_TYPICAL_MIN: Final[float] = 0.5  # Typical range minimum
SHARPNESS_TYPICAL_MAX: Final[float] = 10.0  # Typical range maximum
SHARPNESS_DEFAULT: Final[float] = 2.0

# HDR color constants
COLOR_SDR_MIN: Final[float] = 0.0  # Standard dynamic range minimum
COLOR_SDR_MAX: Final[float] = 1.0  # Standard dynamic range maximum
COLOR_HDR_TYPICAL_MAX: Final[float] = 10.0  # Typical HDR maximum
COLOR_HDR_THEORETICAL_MAX: Final[float] = float("inf")  # No theoretical limit

# Chunk size constants
CHUNK_SIZE_MIN: Final[int] = 1_024  # Minimum chunk size in elements (1KB)
CHUNK_SIZE_DEFAULT: Final[int] = 32_768  # Default chunk size in elements (32KB)
CHUNK_SIZE_MAX: Final[int] = 1_048_576  # Maximum chunk size in elements (1MB)

# Memory constants
KB_TO_BYTES: Final[int] = 1024
MB_TO_BYTES: Final[int] = 1024 * 1024
GB_TO_BYTES: Final[int] = 1024 * 1024 * 1024

# Array size constants
MAX_POINTS_RECOMMENDED: Final[int] = 10_000_000  # 10M points
MAX_POINTS_WARNING: Final[int] = 100_000_000  # 100M points

# Compression constants
COMPRESSION_LEVEL_MIN: Final[int] = 0  # No compression
COMPRESSION_LEVEL_DEFAULT: Final[int] = 3
COMPRESSION_LEVEL_MAX: Final[int] = 9  # Maximum compression

# Transform matrix constants
TRANSFORM_MATRIX_SIZE: Final[int] = 4  # 4x4 matrices
TRANSFORM_MATRIX_ELEMENTS: Final[int] = 16  # Total elements when flattened

# Dimension constants
MAX_DISPLAYED_DIMENSIONS: Final[int] = 3  # Maximum dimensions shown in viewer
MIN_DISPLAYED_DIMENSIONS: Final[int] = 1  # Minimum dimensions shown in viewer
DEFAULT_DIMENSION_STEP_PERCENT: Final[float] = 0.01  # 1% of range for navigation

# Decimal precision for display
POSITION_DISPLAY_DECIMALS: Final[int] = 3
RADIUS_DISPLAY_DECIMALS: Final[int] = 3
COLOR_DISPLAY_DECIMALS: Final[int] = 2

# Physical units
PHYSICAL_UNIT_DEFAULT: Final[str] = "au"  # Arbitrary units

# Zarr metadata keys
ZARR_METADATA_FILENAME: Final[str] = ".zmetadata"
ZARR_ATTRS_KEY: Final[str] = ".zattrs"

# Node type identifiers
NODE_TYPE_SCENE: Final[str] = "scene"
NODE_TYPE_GROUP: Final[str] = "group"
NODE_TYPE_POINTS: Final[str] = "points"

# Validation messages
VALIDATION_POSITIVE_REQUIRED: Final[str] = "Value must be positive"
VALIDATION_SHAPE_MISMATCH: Final[str] = (
    "Shape mismatch: expected {expected}, got {actual}"
)
VALIDATION_OUT_OF_RANGE: Final[str] = (
    "Value {value} is outside valid range [{min}, {max}]"
)
