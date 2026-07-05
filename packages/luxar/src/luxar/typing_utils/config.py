"""luxar.config – Centralized configuration and settings for Luxar core."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Literal, Optional

from ..typing_utils.constants import (
    LUXAR_VERSION_CURRENT,
    MAX_CHUNK_BYTES,
    MIN_CHUNK_BYTES,
    TARGET_CHUNK_BYTES,
)
from ..typing_utils.enums import PhysicalUnit
from ..typing_utils.protocols import CompressorProtocol

# Define literal types locally
CompressionType = Literal["blosc", "zstd", "lz4", "gzip", "bz2", "lzma"]
LuxarVersion = Literal["0.1", "0.2", "0.3"]

# =============================================================================
# Core Configuration Constants
# =============================================================================

# Zarr chunk target/bounds in bytes. Consumers convert to element counts using
# the array's dtype itemsize.
DEFAULT_CHUNK_BYTES: Final[int] = TARGET_CHUNK_BYTES

# Default version for Luxar scenes
DEFAULT_VERSION: Final[str] = LUXAR_VERSION_CURRENT

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
# Informational only — the REAL default is the width-aware per-dtype policy
# (zstd level 9 inside Blosc; see luxar.encoding.compression). These constants
# describe the container/codec family for metadata and docs.
DEFAULT_COMPRESSION: Final[CompressionType] = "blosc"
DEFAULT_COMPRESSION_LEVEL: Final[int] = 9

# Supported physical units
SUPPORTED_UNITS: Final[tuple[str, ...]] = (
    PhysicalUnit.NANOMETER.value,
    PhysicalUnit.MICROMETER.value,
    PhysicalUnit.MILLIMETER.value,
    PhysicalUnit.CENTIMETER.value,
    PhysicalUnit.METER.value,
    PhysicalUnit.METRE.value,
    "meter",  # American spelling - kept for compatibility
    PhysicalUnit.KILOMETER.value,
    PhysicalUnit.INCH.value,
    PhysicalUnit.FOOT.value,
    PhysicalUnit.PIXEL.value,
    PhysicalUnit.ASTRONOMICAL_UNIT.value,
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

# Data type defaults - Encoding system handles actual dtype selection
POSITION_DTYPE: Final[str] = "float32"  # Default for COORDINATE semantic type
COLOR_DTYPE: Final[str] = "float32"  # Changed to float32 for HDR, supports uint8/uint16
TRANSFORM_DTYPE: Final[str] = "float32"  # Always float32 for accuracy

# Supported data types for each attribute
SUPPORTED_POSITION_DTYPES: Final[tuple[str, ...]] = ("float32", "float16")
SUPPORTED_COLOR_DTYPES: Final[tuple[str, ...]] = ("float32", "uint8", "uint16")
SUPPORTED_SCALAR_DTYPES: Final[tuple[str, ...]] = ("float32", "float16", "uint8")

# =============================================================================
# Import Default Compressor
# =============================================================================

# Default compressor (imported from _io.py)
DEFAULT_COMP: Optional[CompressorProtocol]
try:
    from ..io.reader import DEFAULT_COMP
except ImportError:
    DEFAULT_COMP = None

# =============================================================================
# Configuration Validation Functions
# =============================================================================


def validate_chunk_bytes(chunk_bytes: Any) -> int:
    """Validate that a chunk size in **bytes** is within configured bounds.

    The bounds are :data:`~luxar.typing_utils.constants.MIN_CHUNK_BYTES`
    (16 KiB) and :data:`~luxar.typing_utils.constants.MAX_CHUNK_BYTES`
    (256 KiB). Returns the value unchanged if valid; raises ``ValueError``
    otherwise.

    .. note::
        Earlier versions of this function were named ``validate_chunk_size``
        and operated on element counts; the units flipped to bytes when the
        chunking heuristic became byte-targeted. Callers passing element
        counts (e.g. 1024) will now hit the "too small" branch. The older
        function name is preserved as a deprecated alias that emits
        ``DeprecationWarning`` — update call sites at your earliest
        convenience.

    Args:
        chunk_bytes: Chunk size in bytes.

    Returns:
        Validated chunk size, in bytes.

    Raises:
        ValueError: If ``chunk_bytes`` is not an int or is outside the
            configured byte band.
    """
    if not isinstance(chunk_bytes, int):
        raise ValueError("Chunk size must be an integer (bytes)")

    if chunk_bytes < MIN_CHUNK_BYTES:
        raise ValueError(
            f"Chunk size {chunk_bytes} bytes is too small "
            f"(min: {MIN_CHUNK_BYTES} bytes)"
        )

    if chunk_bytes > MAX_CHUNK_BYTES:
        raise ValueError(
            f"Chunk size {chunk_bytes} bytes is too large "
            f"(max: {MAX_CHUNK_BYTES} bytes)"
        )

    return chunk_bytes


def validate_compression_level(level: Any) -> int:
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
    """Estimate memory usage for a points dataset.

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
