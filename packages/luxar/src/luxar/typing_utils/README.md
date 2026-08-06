# Typing Utils Package

The `typing_utils` package provides comprehensive type definitions, protocols, and validation functions for type safety throughout Luxar.

## Overview

This package centralizes all type-related code to ensure consistency and type safety. It follows a clear separation of concerns between different aspects of typing.

## Quick Start

Import type aliases, protocols, enums, and validation helpers from the package
root (not the submodules):

```python
import numpy as np

from luxar.typing_utils import PositionArray, BlendingMode, validate_positions

raw = np.random.rand(100, 3)  # any (N, D) numpy array
positions = validate_positions(raw)  # runtime check -> Float32Array
mode = BlendingMode.ADDITIVE  # str-backed enum member (mode == "additive")
```

See [Usage Examples](#usage-examples) for protocols, enums, and constants.

## Modules

### `protocols.py`
Protocol definitions for type checking.

**Key Components:**
- **Protocols**: `CompressorProtocol`, `NodeProtocol`, `PointsProtocol`, `SceneProtocol`
- **Generic Type Variables**: `NodeT`, `NumericT`, `ArrayT`, `ZarrDataT`

**Purpose**: Define contracts for duck typing

**Note**: Validation functions and type guards (e.g., `validate_positions()`, `is_position_array()`) are defined in `validation/types.py` and re-exported from the `typing_utils` package `__init__.py` for convenience

### `aliases.py`
Simple type aliases for improved readability.

**Key Aliases:**
- **Arrays**: `Float32Array`, `Uint8Array`, `ColorArray`, `PositionArray`, `RadiusArray`, `SharpnessArray`, `ArrayLike`
- **Transforms**: `TransformMatrix`, `TransformList`, `NdTransform`, `NdTransformEntry` (the latter two are defined here but not re-exported from the package `__init__.py`; import them from `luxar.typing_utils.aliases`)
- **Paths**: `PathLike`
- **Zarr**: `ChunkSpec`, `MaxShape`, `ZarrAttrs`
- **Scene**: `NodePath`, `NodeAttributes`, `SceneHierarchy`, `GroupAttrs`
- **Dimensions**: `DimensionRange`, `DimensionIndex`, `DimensionIndices`, `CategoryList`
- **Colors**: `ColorValue`, `ColorRGB`, `ColorRGBA`
- **Metadata**: `PointsMetadata`, `LinesMetadata`, `GSplatsMetadata`, `SceneMetadata`
- **Validation**: `ValidationResult`

**Purpose**: Simplify complex type annotations

### `enums.py`
Enumeration types and configuration classes.

**Key Enums:**
- `BlendingMode`: Rendering blend modes (normal, additive, max, opaque, luminous, volumetric)
- `NodeType`: Scene graph node types (scene, group, points, lines, gsplats)
- `PhysicalUnit`: Supported physical units (nm, um, mm, etc.)

**Key Classes:**
- `RenderingLimits`: Maximum values for rendering parameters
- `Defaults`: Default values for various settings

**Purpose**: Define valid values and configurations

### `constants.py`
Constant values used throughout Luxar.

**Categories:**
- **Version**: `LUXAR_VERSION_CURRENT`, `DEFAULT_ZARR_VERSION`
- **Rendering**: `OPACITY_MIN/MAX`, `ABSORPTION_MIN`/`DEFAULT_ABSORPTION`, `GAMMA_MIN/MAX`, `DEFAULT_BLENDING_MODE`, `SHARPNESS_MIN/MAX`
- **Chunks**: `TARGET_CHUNK_BYTES`, `MIN_CHUNK_BYTES`, `MAX_CHUNK_BYTES` (byte-based single source of truth). Legacy element-count constants (`CHUNK_SIZE_*`, `DEFAULT_CHUNK_SIZE`) have been removed; use the byte-based names directly.
- **Memory**: `KB_TO_BYTES`, `MB_TO_BYTES`, `GB_TO_BYTES`
- **Limits**: `MAX_POINTS_RECOMMENDED`, `MAX_POINTS_WARNING`, `MIN_POINT_RADIUS`, `MAX_POINT_RADIUS`
- **Categorical**: `MIN_CATEGORIES`, `MAX_CATEGORY_LABEL_LENGTH`, `CATEGORICAL_STEP`
- **Node Types**: `NODE_TYPE_SCENE`, `NODE_TYPE_POINTS`, `NODE_TYPE_LINES`, `NODE_TYPE_GSPLATS`
- **Validation**: Error message templates

**Purpose**: Centralize magic numbers and limits

### `config.py`
Configuration settings, defaults, and validation functions.

**Key Constants:**
- `DEFAULT_CHUNK_BYTES` - Byte-based chunk target. Bounds (`MIN_CHUNK_BYTES`, `MAX_CHUNK_BYTES`) live in `constants.py`.
- `DEFAULT_VERSION`, `SUPPORTED_VERSIONS` - Luxar version management
- `SUPPORTED_COMPRESSION`, `SUPPORTED_UNITS` - Supported values
- `MAX_RECOMMENDED_POINTS`, `LARGE_DATASET_WARNING` - Performance thresholds

**Key Functions:**
- `validate_chunk_bytes()` - Validate a chunk size **in bytes** against `MIN_CHUNK_BYTES`/`MAX_CHUNK_BYTES`
- `validate_compression_level()` - Validate compression level (1-9)
- `estimate_memory_usage()` - Estimate memory for a points dataset
- `check_dataset_size_warning()` - Check if dataset size warrants a warning

**Purpose**: Centralize configuration management and validation

## Design Philosophy

### Separation of Concerns

1. **protocols.py**: Runtime validation and contracts
   - Protocols for duck typing
   - Validation with helpful error messages
   - Type guards for conditional narrowing

2. **aliases.py**: Readability improvements
   - Short names for complex types
   - Composite type definitions
   - No logic, just aliases

3. **enums.py**: Valid value sets
   - Literal types for string constants
   - Enum classes for related values
   - Configuration dataclasses

4. **constants.py**: Magic number elimination
   - All numeric limits in one place
   - Clear documentation of purposes
   - Easy to adjust limits

5. **config.py**: Application settings
   - User-configurable options
   - Environment-specific settings
   - Default behaviors

## Usage Examples

### Using Protocols
```python
from luxar.typing_utils import NodeProtocol, validate_positions


def process_node(node: NodeProtocol) -> None:
    """Process any node that follows the protocol."""
    for depth, child in node.walk():
        print(f"{'  ' * depth}{child.name}")


# Validate data at runtime
positions = np.random.randn(100, 3)
validated = validate_positions(positions)  # Returns Float32Array
```

### Using Type Aliases
```python
from luxar.typing_utils import ColorArray, TransformMatrix


def apply_transform(points: PositionArray, transform: TransformMatrix) -> PositionArray:
    """Apply 4x4 transform to points."""
    # Clear type signatures without numpy.typing verbosity
    ...
```

### Using Enums
```python
from luxar.typing_utils import BlendingMode, PhysicalUnit

# str-backed enum members compare equal to their string value
mode = BlendingMode.ADDITIVE  # mode == "additive"
mode = BlendingMode("additive")  # look up a member from its string

# Validation: convert a string, rejecting unknown values
unit = PhysicalUnit.validate("um")  # -> PhysicalUnit.MICROMETER
if unit in (PhysicalUnit.NANOMETER, PhysicalUnit.MICROMETER, PhysicalUnit.MILLIMETER):
    # Handle metric units
    ...
```

### Using Constants
```python
from luxar.typing_utils import (
    MAX_POINTS_WARNING,
    TARGET_CHUNK_BYTES,
    OPACITY_MIN,
    OPACITY_MAX,
)

if n_points > MAX_POINTS_WARNING:
    warnings.warn(f"Large dataset: {n_points} points")

opacity = np.clip(value, OPACITY_MIN, OPACITY_MAX)
```

## Type Safety Strategy

### Static Checking
- Use `mypy` or `pyright` for static analysis
- Protocols enable duck typing with safety
- Type aliases improve readability

### Runtime Validation
- Validation functions catch errors early
- Helpful error messages guide fixes
- Type guards enable safe narrowing

### Progressive Enhancement
```python
# Start simple
data = load_data()  # type: Any

# Add validation
data = validate_positions(data)  # type: PositionArray

# Now type-safe
process_points(data)  # Knows data is PositionArray
```

## Best Practices

1. **Import from typing_utils**: Don't import from submodules directly
2. **Validate early**: Use validation functions at API boundaries
3. **Use protocols**: Define interfaces, not implementations
4. **Keep aliases simple**: Don't add logic to type aliases
5. **Document constants**: Explain why each limit exists

## Dependencies

Internal:
- Cross-references between modules (imports managed carefully)

External:
- `numpy`: Array type definitions
- `typing`/`typing_extensions`: Type system features
- Python 3.10+ compatibility layer
