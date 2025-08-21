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

# Core classes and functions
from .core import transforms
from .core.dimensions import Dimension, Dimensions
from .core.node import Node
from .core.points import Points
from .core.scene import Scene
from .core.transforms import (
    compose,
    from_list,
    identity,
    inverse,
    look_at,
    rotate,
    rotate_x,
    rotate_y,
    rotate_z,
    rotation,
    scale,
    scaling,
    to_list,
    translate,
    translation,
)

# I/O classes
from .io.compiler import LuxarZarrCompiler
from .io.streaming import StreamingPoints

# Type definitions
from .typing_utils.aliases import (
    ColorArray,
    PathLike,
    PositionArray,
    TransformMatrix,
)
from .typing_utils.enums import (
    BlendingMode,
    Defaults,
    NodeType,
    PhysicalUnit,
    RenderingLimits,
)
from .typing_utils.protocols import DimensionMetadata

__version__ = "2025.08.03"

# Backward compatibility - import submodules for direct access
# This allows "from luxar.array_utils import ..." to work
import sys

from . import core, io, typing_utils, utils
from . import validation as validation_module
from .core import dimensions, node, points, scene
from .io import compiler, streaming, writer
from .utils import array as array_utils
from .utils import demos
from .validation import base as validation

# Add module aliases for backward compatibility
sys.modules["luxar.array_utils"] = array_utils
sys.modules["luxar.dimensions"] = dimensions
sys.modules["luxar.node"] = node
sys.modules["luxar.points"] = points
sys.modules["luxar.scene"] = scene
sys.modules["luxar.compiler"] = compiler
sys.modules["luxar.streaming"] = streaming
sys.modules["luxar.writer"] = writer
sys.modules["luxar.demos"] = demos
sys.modules["luxar.validation"] = validation
sys.modules["luxar.transforms"] = transforms
sys.modules["luxar.types"] = typing_utils
sys.modules["luxar.config"] = typing_utils.config
sys.modules["luxar._io"] = io.reader

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
    "PathLike",
    # Transform utilities
    "transforms",
    "identity",
    "translate",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "rotation",
    "scale",
    "scaling",
    "to_list",
    "translate",
    "translation",
    "compose",
    "from_list",
    "inverse",
    "look_at",
    # Version
    "__version__",
]
