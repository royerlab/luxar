"""Type aliases for improved code readability and maintainability.

This module defines type aliases used throughout the Luxar codebase to
simplify complex type annotations and improve code clarity.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

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

# Metadata type aliases
PointsMetadata = Dict[str, Any]
SceneMetadata = Dict[str, Any]

# Color value type aliases
ColorValue = Union[float, int]  # Single color component
ColorRGB = Tuple[ColorValue, ColorValue, ColorValue]
ColorRGBA = Tuple[ColorValue, ColorValue, ColorValue, ColorValue]

# Physical units
PhysicalUnitStr = str  # One of the supported unit strings

# Validation type aliases
ValidationResult = Tuple[bool, Optional[str]]  # (is_valid, error_message)
