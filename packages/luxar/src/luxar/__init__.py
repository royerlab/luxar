"""
luxar.__init__ – Package exports for Luxar core.
"""

from .node import Node
from .points import Points
from .scene import Scene

# Export commonly used types for external users
from .types import (
    ColorArray,
    LuxarVersion,
    NodeType,
    PathLike,
    PhysicalUnit,
    PositionArray,
    TransformMatrix,
)

# Export transform utilities
from . import transforms

__version__ = "2025.08.03"

__all__: list[str] = [
    # Core classes
    "Scene",
    "Points",
    "Node",
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
