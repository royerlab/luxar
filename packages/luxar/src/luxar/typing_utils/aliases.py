"""Type aliases for improved code readability and maintainability.

This module defines simple type aliases used throughout the Luxar codebase.
For protocols, dataclasses, and validation functions, see protocols.py.
For enums and literal types, see enums.py.
For constants, see constants.py.
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Array type aliases
ArrayLike = Union[np.ndarray, List, Tuple]
Float32Array = NDArray[np.float32]
Uint8Array = NDArray[np.uint8]
ColorArray = Union[Float32Array, Uint8Array]
PositionArray = Float32Array
RadiusArray = Float32Array
SharpnessArray = Float32Array

# Transform type aliases
TransformMatrix = Float32Array  # 4x4 matrix
TransformList = List[float]  # 16-element list

# nD Transform type aliases (per-dimension transforms for non-displayed dimensions)
NdTransformEntry = Dict[
    str, Any
]  # {"scale": float, "offset": float} or {"permutation": [int]}
NdTransform = Dict[str, NdTransformEntry]  # Maps dim name → transform entry

# Path-like types
PathLike = Union[str, Path]

# Zarr-related type aliases
ChunkSpec = Optional[Union[bool, int, Tuple[int, ...]]]
MaxShape = Optional[Tuple[Optional[int], ...]]
ZarrAttrs = Dict[str, Any]

# Scene hierarchy type aliases
NodePath = str
NodeAttributes = Dict[str, Any]

# Dimension type aliases
DimensionRange = Tuple[float, float]
DimensionIndex = int
DimensionIndices = List[int]

# Categorical dimension type aliases
CategoryList = Optional[List[str]]  # Category labels for categorical dimensions

# Metadata type aliases
PointsMetadata = Dict[str, Any]
LinesMetadata = Dict[str, Any]
GSplatsMetadata = Dict[str, Any]
MeshMetadata = Dict[str, Any]
SceneMetadata = Dict[str, Any]

# Color value type aliases
ColorValue = Union[float, int]  # Single color component
ColorRGB = Tuple[ColorValue, ColorValue, ColorValue]
ColorRGBA = Tuple[ColorValue, ColorValue, ColorValue, ColorValue]

# Scene hierarchy types
SceneHierarchy = Generator[Tuple[int, Any], None, None]  # (depth, node) pairs
GroupAttrs = Dict[str, Any]  # Zarr group attributes

# Validation type aliases
ValidationResult = Tuple[bool, Optional[str]]  # (is_valid, error_message)
