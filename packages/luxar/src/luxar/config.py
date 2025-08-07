"""luxar.config – Centralized configuration and settings for Luxar core."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal, Optional

from .types import (
    CompressionType,
    CompressorProtocol,
    LuxarVersion,
    PhysicalUnit,
)

# =============================================================================
# Core Configuration Constants
# =============================================================================

# Default Zarr chunk size for point clouds (32KB)
DEFAULT_CHUNK_SIZE: Final[int] = 32_768

# Minimum and maximum chunk sizes for validation
MIN_CHUNK_SIZE: Final[int] = 1_024  # 1KB
MAX_CHUNK_SIZE: Final[int] = 1_048_576  # 1MB

# Default units for scenes
DEFAULT_UNITS: Final[PhysicalUnit] = "metre"

# Default version for Luxar scenes
DEFAULT_VERSION: Final[LuxarVersion] = "0.2"

# Supported Luxar versions for backwards compatibility
SUPPORTED_VERSIONS: Final[tuple[LuxarVersion, ...]] = ("0.1", "0.2", "0.3")

# Default output directory for temporary scenes
DEFAULT_TMP_DIR: Final[Optional[Path]] = None

# =============================================================================
# Logging Configuration
# =============================================================================

# Available log levels
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Default logging level for arbol
DEFAULT_LOG_LEVEL: Final[LogLevel] = "INFO"

# =============================================================================
# Data Type Configuration
# =============================================================================

# Supported compression algorithms
SUPPORTED_COMPRESSION: Final[tuple[CompressionType, ...]] = (
    "blosc",
    "zstd",
    "lz4",
    "gzip",
    "bz2",
    "lzma",
)

# Default compression settings
DEFAULT_COMPRESSION: Final[CompressionType] = "blosc"
DEFAULT_COMPRESSION_LEVEL: Final[int] = 3

# Supported physical units
SUPPORTED_UNITS: Final[tuple[PhysicalUnit, ...]] = (
    "nm",    # nanometer
    "um",    # micrometer
    "mm",    # millimeter
    "cm",    # centimeter
    "m",     # meter (short form)
    "metre",  # meter (British spelling)
    "meter",  # meter (American spelling)
    "km",    # kilometer
    "inch",  # inch
    "foot",  # foot
    "px",    # pixel
    "au",    # arbitrary units
)

# =============================================================================
# Performance Configuration
# =============================================================================

# Maximum recommended points per scene for good performance
MAX_RECOMMENDED_POINTS: Final[int] = 10_000_000

# Warning threshold for large datasets
LARGE_DATASET_WARNING: Final[int] = 1_000_000

# Memory usage estimates (bytes per point)
MEMORY_PER_POINT_POSITIONS: Final[int] = 12  # 3 * float32
MEMORY_PER_POINT_COLORS: Final[int] = 3  # 3 * uint8
MEMORY_PER_POINT_TOTAL: Final[int] = (
    MEMORY_PER_POINT_POSITIONS + MEMORY_PER_POINT_COLORS
)

# =============================================================================
# Validation Configuration
# =============================================================================

# Array shape validation
POSITION_SHAPE_DIMS: Final[int] = 2
POSITION_SHAPE_CHANNELS: Final[int] = 3
COLOR_SHAPE_CHANNELS: Final[int] = 3
TRANSFORM_MATRIX_SIZE: Final[tuple[int, int]] = (4, 4)

# Data type validation
POSITION_DTYPE: Final[str] = "float32"
COLOR_DTYPE: Final[str] = "uint8"
TRANSFORM_DTYPE: Final[str] = "float32"

# =============================================================================
# Import Default Compressor
# =============================================================================

# Default compressor (imported from _io.py)
DEFAULT_COMP: Optional[CompressorProtocol]
try:
    from ._io import DEFAULT_COMP
except ImportError:
    DEFAULT_COMP = None

# =============================================================================
# Configuration Validation Functions
# =============================================================================


def validate_chunk_size(chunk_size: int) -> int:
    """Validate chunk size is within acceptable bounds.

    Args:
        chunk_size: Chunk size to validate

    Returns:
        Validated chunk size

    Raises:
        ValueError: If chunk size is invalid
    """
    if not isinstance(chunk_size, int):
        raise ValueError("Chunk size must be an integer")

    if chunk_size < MIN_CHUNK_SIZE:
        raise ValueError(
            f"Chunk size {chunk_size} is too small (min: {MIN_CHUNK_SIZE})"
        )

    if chunk_size > MAX_CHUNK_SIZE:
        raise ValueError(
            f"Chunk size {chunk_size} is too large (max: {MAX_CHUNK_SIZE})"
        )

    return chunk_size


def validate_compression_level(level: int) -> int:
    """Validate compression level is within acceptable bounds.

    Args:
        level: Compression level to validate (typically 1-9)

    Returns:
        Validated compression level

    Raises:
        ValueError: If compression level is invalid
    """
    if not isinstance(level, int):
        raise ValueError("Compression level must be an integer")

    if level < 1 or level > 9:
        raise ValueError(f"Compression level {level} must be between 1 and 9")

    return level


def estimate_memory_usage(n_points: int, has_colors: bool = True) -> int:
    """Estimate memory usage for a point cloud dataset.

    Args:
        n_points: Number of points
        has_colors: Whether the dataset includes colors

    Returns:
        Estimated memory usage in bytes
    """
    if not isinstance(n_points, int) or n_points < 0:
        raise ValueError("Number of points must be a non-negative integer")

    memory_per_point = MEMORY_PER_POINT_POSITIONS
    if has_colors:
        memory_per_point += MEMORY_PER_POINT_COLORS

    return n_points * memory_per_point


def check_dataset_size_warning(n_points: int) -> Optional[str]:
    """Check if dataset size warrants a performance warning.

    Args:
        n_points: Number of points in dataset

    Returns:
        Warning message if applicable, None otherwise
    """
    if n_points > MAX_RECOMMENDED_POINTS:
        return (
            f"Dataset with {n_points:,} points exceeds recommended maximum "
            f"of {MAX_RECOMMENDED_POINTS:,} points. Performance may be degraded."
        )
    elif n_points > LARGE_DATASET_WARNING:
        memory_mb = estimate_memory_usage(n_points) / (1024 * 1024)
        return (
            f"Large dataset with {n_points:,} points (~{memory_mb:.1f}MB). "
            f"Consider using compression and chunking for better performance."
        )

    return None
