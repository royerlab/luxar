"""Luxar - High-dimensional point cloud visualization and analysis toolkit.

Luxar provides a powerful Python API for creating, manipulating, and visualizing
massive point cloud datasets with arbitrary dimensions. The system is designed
for memory-efficient progressive writing, enabling processing of TB-scale datasets
on GB-scale machines.

Key Features:
    - Progressive writing to Zarr for memory-efficient processing
    - Support for nD point clouds (not limited to 3D)
    - HDR color support with float32 precision
    - Hierarchical scene graphs with transforms
    - Streaming API for datasets larger than RAM
    - Type-safe API with comprehensive validation

Basic Usage:
    >>> import luxar
    >>> with luxar.LuxarZarrCompiler('output.zarr') as compiler:
    ...     scene = compiler.create_scene()
    ...     scene.add_points('my_points', positions)

Streaming Large Datasets:
    >>> streaming = luxar.StreamingPoints('huge_cloud', compiler)
    >>> for batch in data_generator():
    ...     streaming.append_batch(batch)
    >>> streaming.finalize()

For detailed documentation, see: https://github.com/royerlab/luxar
"""

from __future__ import annotations

# Export transform utilities
from . import transforms

# Core classes
from .compiler import LuxarZarrCompiler
from .dimensions import Dimension, Dimensions

# Enumerations
from .enums import BlendingMode, Defaults, NodeType, PhysicalUnit, RenderingLimits
from .node import Node
from .points import Points
from .scene import Scene
from .streaming import StreamingPoints

# Export commonly used types for external users
from .types import (
    ColorArray,
    DimensionMetadata,  # Keep for backward compatibility
    LuxarVersion,
    PathLike,
    PositionArray,
    TransformMatrix,
)

__version__ = "2025.08.03"

__all__: list[str] = [
    # Core classes
    "Scene",
    "Points",
    "Node",
    # Progressive writing
    "LuxarZarrCompiler",
    "StreamingPoints",
    # Dimension system
    "Dimensions",
    "Dimension",
    "DimensionMetadata",  # Legacy support
    # Enumerations
    "BlendingMode",
    "NodeType",
    "PhysicalUnit",
    "RenderingLimits",
    "Defaults",
    # Type definitions
    "PositionArray",
    "ColorArray",
    "TransformMatrix",
    "LuxarVersion",
    "PathLike",
    # Transform utilities
    "transforms",
    # Version
    "__version__",
]
