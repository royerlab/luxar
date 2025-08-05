"""luxar.__init__ – Package exports for Luxar core."""

# Export transform utilities
from . import transforms
from .dimensions import Dimension, Dimensions
from .node import Node
from .points import Points
from .scene import Scene

# Export commonly used types for external users
from .types import (
    ColorArray,
    DimensionMetadata,  # Keep for backward compatibility
    LuxarVersion,
    NodeType,
    PathLike,
    PhysicalUnit,
    PositionArray,
    TransformMatrix,
)

__version__ = "2025.08.03"

__all__: list[str] = [
    # Core classes
    "Scene",
    "Points",
    "Node",
    # Dimension system
    "Dimensions",
    "Dimension",
    "DimensionMetadata",  # Legacy support
    # Type definitions
    "PositionArray",
    "ColorArray",
    "TransformMatrix",
    "NodeType",
    "PhysicalUnit",
    "LuxarVersion",
    "PathLike",
    # Transform utilities
    "transforms",
    # Version
    "__version__",
]
