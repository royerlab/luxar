"""Luxar - High-dimensional scientific visualization and analysis toolkit.

Luxar provides a powerful Python API for creating, manipulating, and visualizing
massive nD datasets (points, lines, Gaussian splats) with arbitrary dimensions. The system is designed
for memory-efficient progressive writing, enabling processing of TB-scale datasets
on GB-scale machines.

Key Features:
    - Progressive writing to Zarr for memory-efficient processing
    - Support for nD data (not limited to 3D)
    - Morton/Hilbert spatial ordering for better compression
    - HDR color support with float32 precision
    - Hierarchical scene graphs with transforms
    - Type-safe API with comprehensive validation
    - Semantic type-based encoding (quantization, broadcasting, LUT)

Basic Usage:
    >>> import luxar
    >>> dims = luxar.Dimensions.default_3d()
    >>> with luxar.LuxarZarrCompiler('output.luxar.zarr') as compiler:
    ...     scene = compiler.create_scene(dimensions=dims)
    ...     scene.add_points('my_points', positions, colors, radii)

Large Datasets (process in chunks):
    >>> with luxar.LuxarZarrCompiler('huge.luxar.zarr') as compiler:
    ...     scene = compiler.create_scene(dimensions=dims)
    ...     for i in range(100):
    ...         chunk = load_chunk(i)  # 10M points each
    ...         scene.add_points(f'chunk_{i}', chunk)

For detailed documentation, see: https://github.com/royerlab/luxar
"""

from __future__ import annotations

# Core classes and functions
from .core import transforms
from .core.dimensions import Dimension, Dimensions
from .core.group import Group
from .core.gsplats import GSplats
from .core.lines import Lines
from .core.node import Node
from .core.overlay import Overlay
from .core.points import Points
from .core.scene import Scene
from .core.transforms import (
    compose,
    from_list,
    identity,
    inverse,
    look_at,
    prepare_transform_for_zarr,
    read_transform_from_zarr,
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
from .core.viewer_config import (
    AnimationConfig,
    CameraConfig,
    DimensionsConfig,
    UIConfig,
    ViewerConfig,
)

# I/O classes
from .io.compiler import LuxarZarrCompiler
from .io.reader import LuxarScene

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
from .validation.nd_transforms import (
    apply_nd_transform_to_bounds,
    compose_nd_transforms,
    validate_nd_transform,
)

__version__ = "2026.06.05"

# Backward compatibility and optional re-exports.
# gsplats always provides names (real or stub that raises on use if torch is missing).
import sys

from . import colormaps, core, io, typing_utils, utils
from . import validation as validation_module
from .core import dimensions, node, points, scene
from .gsplats import GSplatData, fit_gaussian_splats
from .io import compiler, writer
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
sys.modules["luxar.writer"] = writer
sys.modules["luxar.demos"] = demos
# sys.modules["luxar.validation"] = validation  # Removed to allow proper package resolution
sys.modules["luxar.transforms"] = transforms
sys.modules["luxar.types"] = typing_utils
sys.modules["luxar.config"] = typing_utils.config
sys.modules["luxar._io"] = io.reader

__all__: list[str] = [
    # Core classes
    "Scene",
    "Group",
    "Points",
    "Lines",
    "GSplats",
    "Node",
    "Overlay",
    # Progressive writing
    "LuxarZarrCompiler",
    # Reading
    "LuxarScene",
    # Dimension system
    "Dimensions",
    "Dimension",
    # Viewer configuration
    "ViewerConfig",
    "CameraConfig",
    "UIConfig",
    "DimensionsConfig",
    "AnimationConfig",
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
    "translation",
    "rotate",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "rotation",
    "scale",
    "scaling",
    "compose",
    "inverse",
    "look_at",
    "to_list",
    "from_list",
    "prepare_transform_for_zarr",
    "read_transform_from_zarr",
    # nD Transforms
    "validate_nd_transform",
    "compose_nd_transforms",
    "apply_nd_transform_to_bounds",
    # Gaussian Splatting (optional)
    "GSplatData",
    "fit_gaussian_splats",
    # Version
    "__version__",
]
