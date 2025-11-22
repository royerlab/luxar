# Typing Utils Package

The `typing_utils` package provides comprehensive type definitions, protocols, and validation functions for type safety throughout Luxar.

## Overview

This package centralizes all type-related code to ensure consistency and type safety. It follows a clear separation of concerns between different aspects of typing.

## Modules

### `protocols.py`
Protocol definitions for type checking and validation function imports.

**Key Components:**
- **Protocols**: `CompressorProtocol`, `NodeProtocol`, `PointsProtocol`, `SceneProtocol`
- **Validation Functions**: Imported from `validation/types.py` and re-exported for convenience
- **Type Guards**: `is_position_array()`, `is_color_array()`, `is_transform_matrix()`
- **Generic Type Variables**: `NodeT`, `NumericT`, `ArrayT`, `ZarrDataT`

**Purpose**: Define contracts for duck typing and centralize validation imports

**Note**: Validation functions are now defined in `validation/types.py` to avoid duplication

### `aliases.py`
Simple type aliases for improved readability.

**Key Aliases:**
- **Arrays**: `Float32Array`, `Uint8Array`, `ColorArray`, `PositionArray`
- **Transforms**: `TransformMatrix`, `TransformList`
- **Paths**: `PathLike`
- **Zarr**: `ChunkSpec`, `MaxShape`, `ZarrAttrs`
- **Scene**: `NodePath`, `NodeAttributes`, `SceneHierarchy`
- **Colors**: `ColorValue`, `ColorRGB`, `ColorRGBA`

**Purpose**: Simplify complex type annotations

### `enums.py`
Enumeration types and configuration classes.

**Key Enums:**
- `BlendingMode`: Rendering blend modes (normal, additive, etc.)
- `NodeType`: Scene graph node types (points, group, scene)
- `PhysicalUnit`: Supported physical units (nm, um, mm, etc.)

**Key Classes:**
- `RenderingLimits`: Maximum values for rendering parameters
- `Defaults`: Default values for various settings

**Purpose**: Define valid values and configurations

### `constants.py`
Constant values used throughout Luxar.

**Categories:**
- **Version**: `LUXAR_VERSION_CURRENT`, `DEFAULT_ZARR_VERSION`
- **Rendering**: `OPACITY_MIN/MAX`, `GAMMA_MIN/MAX`, `DEFAULT_BLENDING_MODE`
- **Chunks**: `CHUNK_SIZE_MIN/DEFAULT/MAX`
- **Memory**: `KB_TO_BYTES`, `MB_TO_BYTES`, `GB_TO_BYTES`
- **Limits**: `MAX_POINTS_RECOMMENDED`, `MAX_POINTS_WARNING`
- **Validation**: Error message templates

**Purpose**: Centralize magic numbers and limits

### `config.py`
Configuration settings and defaults.

**Key Components:**
- Compression settings and defaults
- Supported versions and units
- Logging configuration
- File path defaults

**Purpose**: Centralize configuration management

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

def apply_transform(
    points: PositionArray,
    transform: TransformMatrix
) -> PositionArray:
    """Apply 4x4 transform to points."""
    # Clear type signatures without numpy.typing verbosity
    ...
```

### Using Enums
```python
from luxar.typing_utils import BlendingMode, PhysicalUnit

# Type-safe string literals
mode: BlendingMode = "additive"  # OK
mode: BlendingMode = "invalid"   # Type error

# Validation
unit: PhysicalUnit = "um"
if unit in ["nm", "um", "mm"]:
    # Handle metric units
    ...
```

### Using Constants
```python
from luxar.typing_utils import (
    MAX_POINTS_WARNING,
    DEFAULT_CHUNK_SIZE,
    OPACITY_MIN, OPACITY_MAX
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