"""Type definitions for Luxar.

This package is organized as follows:
- aliases.py: Simple type aliases for readability
- protocols.py: Protocols, dataclasses, validation functions, and type guards
- enums.py: Enumeration types
- constants.py: Constant values
- config.py: Configuration classes
"""

import warnings

# Re-export everything for backward compatibility
# Import validation functions from validation.types module
from ..validation.types import (
    is_color_array,
    is_position_array,
    is_transform_matrix,
    validate_blending_mode,
    validate_colors,
    validate_gamma,
    validate_node_type,
    validate_opacity,
    validate_physical_unit,
    validate_positions,
    validate_radii,
    validate_sharpness,
    validate_transform,
)
from .aliases import (
    ArrayLike,
    CategoryList,
    ChunkSpec,
    ColorArray,
    ColorRGB,
    ColorRGBA,
    ColorValue,
    DimensionIndex,
    DimensionIndices,
    DimensionRange,
    Float32Array,
    GroupAttrs,
    GSplatsMetadata,
    LinesMetadata,
    MaxShape,
    NodeAttributes,
    NodePath,
    PathLike,
    PointsMetadata,
    PositionArray,
    RadiusArray,
    SceneHierarchy,
    SceneMetadata,
    SharpnessArray,
    TransformList,
    TransformMatrix,
    Uint8Array,
    ValidationResult,
    ZarrAttrs,
)
from .constants import (
    CATEGORICAL_STEP,
    DEFAULT_BLENDING_MODE,
    DEFAULT_COMPRESSOR,
    DEFAULT_GAMMA,
    DEFAULT_OPACITY,
    DEFAULT_ZARR_VERSION,
    GAMMA_MAX,
    GAMMA_MIN,
    MAX_CATEGORY_LABEL_LENGTH,
    MAX_CHUNK_BYTES,
    MAX_POINT_RADIUS,
    MIN_CATEGORIES,
    MIN_CHUNK_BYTES,
    MIN_POINT_RADIUS,
    NODE_TYPE_GSPLATS,
    NODE_TYPE_LINES,
    OPACITY_MAX,
    OPACITY_MIN,
    SHARPNESS_MAX,
    SHARPNESS_MIN,
    TARGET_CHUNK_BYTES,
)
from .enums import BlendingMode, Defaults, NodeType, PhysicalUnit, RenderingLimits

# Encoding system now in luxar.encoding package
from .protocols import (
    CompressorProtocol,
    NodeProtocol,
    NodeT,
    NumericT,
    PointsProtocol,
    SceneProtocol,
    ZarrDataT,
)

_DEPRECATED_REEXPORTS: dict[str, tuple[int, str]] = {
    "DEFAULT_CHUNK_SIZE": (32_768, "TARGET_CHUNK_BYTES"),
}


def __getattr__(name: str) -> int:
    if name in _DEPRECATED_REEXPORTS:
        value, replacement = _DEPRECATED_REEXPORTS[name]
        warnings.warn(
            f"luxar.typing_utils.{name} is deprecated and will be removed in a "
            f"future release; use {replacement} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # From aliases
    "ArrayLike",
    "CategoryList",
    "ChunkSpec",
    "ColorArray",
    "ColorRGB",
    "ColorRGBA",
    "ColorValue",
    "DimensionIndex",
    "DimensionIndices",
    "DimensionRange",
    "Float32Array",
    "GroupAttrs",
    "GSplatsMetadata",
    "LinesMetadata",
    "MaxShape",
    "NodeAttributes",
    "NodePath",
    "PathLike",
    "PointsMetadata",
    "PositionArray",
    "RadiusArray",
    "SceneHierarchy",
    "SceneMetadata",
    "SharpnessArray",
    "TransformList",
    "TransformMatrix",
    "Uint8Array",
    "ValidationResult",
    "ZarrAttrs",
    # From protocols
    "CompressorProtocol",
    "NodeProtocol",
    "NodeT",
    "NumericT",
    "PointsProtocol",
    "SceneProtocol",
    "ZarrDataT",
    "is_color_array",
    "is_position_array",
    "is_transform_matrix",
    "validate_blending_mode",
    "validate_colors",
    "validate_gamma",
    "validate_node_type",
    "validate_opacity",
    "validate_physical_unit",
    "validate_positions",
    "validate_radii",
    "validate_sharpness",
    "validate_transform",
    # From enums
    "BlendingMode",
    "NodeType",
    "PhysicalUnit",
    # From constants
    "CATEGORICAL_STEP",
    "DEFAULT_BLENDING_MODE",
    "DEFAULT_COMPRESSOR",
    "DEFAULT_GAMMA",
    "DEFAULT_OPACITY",
    "DEFAULT_ZARR_VERSION",
    "GAMMA_MAX",
    "GAMMA_MIN",
    "MAX_CATEGORY_LABEL_LENGTH",
    "MAX_CHUNK_BYTES",
    "MAX_POINT_RADIUS",
    "MIN_CATEGORIES",
    "MIN_CHUNK_BYTES",
    "MIN_POINT_RADIUS",
    "NODE_TYPE_GSPLATS",
    "NODE_TYPE_LINES",
    "OPACITY_MAX",
    "OPACITY_MIN",
    "SHARPNESS_MAX",
    "SHARPNESS_MIN",
    "TARGET_CHUNK_BYTES",
    # From enums
    "Defaults",
    "RenderingLimits",
]
