"""Type aliases for improved code readability and maintainability.

This module defines simple type aliases used throughout the Luxar codebase.
For protocols, dataclasses, and validation functions, see protocols.py.
For enums and literal types, see enums.py.
For constants, see constants.py.
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, MutableMapping, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Array type aliases.
#
# The three writable-attribute aliases below name what the WRITING PATH accepts,
# not what it stores: `ArrayEncoder` picks the on-disk dtype from the semantic
# type and the `EncodingMode`, so the input dtype is a caller convenience, not a
# storage decision. `SUPPORTED_POSITION_DTYPES` and friends are the same list.
#
# They used to be narrower here (`PositionArray = Float32Array`, `ColorArray`
# without uint16) while `io/writer.py` redefined the SAME TWO NAMES wider, in a
# module that also imports from this one. That collision is what made
# `ZarrWriterProtocol` un-checkable against its only implementation: the
# protocol was annotated with the wide pair and `LuxarZarrCompiler` with the
# narrow one, so all four write methods carried `# type: ignore[override]` and
# mypy checked nothing about them. One definition, widest-accepted, here.
ArrayLike = Union[np.ndarray, List, Tuple]
Float32Array = NDArray[np.float32]
Float16Array = NDArray[np.float16]
Uint8Array = NDArray[np.uint8]
Uint16Array = NDArray[np.uint16]
ColorArray = Union[Float32Array, Uint8Array, Uint16Array]
PositionArray = Union[Float32Array, Float16Array]
ScalarArray = Union[Float32Array, Float16Array, Uint8Array]
RadiusArray = ScalarArray
SharpnessArray = ScalarArray

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
GroupAttrs = MutableMapping[str, Any]  # Write-through Zarr group attributes

# Validation type aliases
ValidationResult = Tuple[bool, Optional[str]]  # (is_valid, error_message)
