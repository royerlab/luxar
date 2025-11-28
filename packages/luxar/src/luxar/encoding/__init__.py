"""Encoding package for semantic types and array encoding/decoding.

This package handles:
- Semantic type definitions (COORDINATE, COLOR, POSITIVE_SCALAR, etc.)
- Data type configuration and selection
- Array dtype conversion and quantization
- Encoding metadata
"""

from .datatypes import (
    DEFAULT_CONFIG,
    MEMORY_CONFIG,
    PRECISION_CONFIG,
    ColorDType,
    ColorDTypeStr,
    DataTypeConfig,
    DataTypeMode,
    PositionDType,
    PositionDTypeStr,
    ScalarDType,
    ScalarDTypeStr,
    convert_array_dtype,
    get_dtype_info,
    infer_optimal_dtype,
    validate_dtype_string,
)

__all__ = [
    # Enums
    "DataTypeMode",
    # Classes
    "DataTypeConfig",
    # Functions
    "convert_array_dtype",
    "infer_optimal_dtype",
    "get_dtype_info",
    "validate_dtype_string",
    # Type aliases
    "PositionDType",
    "PositionDTypeStr",
    "ColorDType",
    "ColorDTypeStr",
    "ScalarDType",
    "ScalarDTypeStr",
    # Config instances
    "DEFAULT_CONFIG",
    "PRECISION_CONFIG",
    "MEMORY_CONFIG",
]
