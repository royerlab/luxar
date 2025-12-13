# Utils Package

The `utils` package provides utility functions for common operations in Luxar, including array manipulation and demo data generation.

## Overview

This package contains helper functions that simplify common tasks and provide convenient demo data generators for testing and examples.

**Note**: As of Luxar v1.4.0, scalar broadcasting (e.g., `colors=(1,0,0)`, `radii=0.5`) is handled by `ArrayEncoder` in `luxar.encoding`. The previous `broadcast_*_to_points()` functions have been removed.

## Modules

### `array.py`
Array manipulation utilities.

**Key Functions:**
- `ensure_float32()`: Convert arrays to float32
- `validate_array_shape()`: Check array dimensions

**Features:**
- Automatic type conversion with validation
- Shape validation with helpful error messages

### `demos.py`
Demo scene generators for examples and testing.

**Key Functions:**
- `create_lorenz_attractor()`: Generate Lorenz attractor visualization
- `create_random_spheres()`: Create random spherical points
- `create_time_series_demo()`: Generate time-varying data

**Features:**
- Ready-to-use demo scenes
- Configurable parameters
- Educational examples of Luxar features

## Usage Examples

### Creating Demo Scenes

```python
from luxar.utils import create_lorenz_attractor

# Generate Lorenz attractor
create_lorenz_attractor(
    store_path='lorenz.zarr',
    n_points=50000,
    seed=42
)

# Creates a beautiful 3D visualization
# with time-based coloring
```

### Array Utilities

```python
from luxar.utils import ensure_float32, validate_array_shape
import numpy as np

# Ensure float32 dtype
arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
arr_f32 = ensure_float32(arr)  # Now float32

# Validate array shape
positions = np.random.rand(100, 3).astype(np.float32)
validate_array_shape(positions, (100, 3), name="positions")  # OK

# Multiple acceptable shapes
validate_array_shape(colors, [(100, 3), (100, 4)], name="colors")
```

**Note**: For scalar broadcasting (e.g., `radii=0.5` for all points), use the encoding system via `LuxarZarrCompiler.write_points()` which handles this automatically.

## Demo Generators

### Lorenz Attractor
```python
def create_lorenz_attractor(
    store_path: str,
    n_points: int = 10_000,
    seed: Optional[int] = None
) -> None:
```
Generates the famous Lorenz attractor with:
- Smooth trajectory through phase space
- Time-based coloring
- Configurable density

### Random Spheres
```python
def create_random_spheres(
    store_path: str,
    n_spheres: int = 100,
    points_per_sphere: int = 1000,
    seed: Optional[int] = None
) -> None:
```
Creates multiple spherical points:
- Random positions and sizes
- Different colors per sphere
- Hierarchical organization

### Time Series Demo
```python
def create_time_series_demo(
    store_path: PathLike,
    n_timepoints: int = 10,
    n_points_per_time: int = 1000,
    seed: Optional[int] = None
) -> None:
```
Generates time-varying data:
- 4D data (x, y, z, time)
- Animated trajectories
- Useful for testing nD features

## Best Practices

### Broadcasting
1. **Be explicit**: Specify expected shapes
2. **Validate early**: Check inputs before processing
3. **Preserve precision**: Maintain float32 for GPU compatibility
4. **Handle None**: Gracefully handle optional parameters

### Demo Data
1. **Use seeds**: Make demos reproducible
2. **Document parameters**: Explain what each demo shows
3. **Educational value**: Demos should teach Luxar features
4. **Performance**: Keep demos fast for testing

## Utility Patterns

### Safe Type Conversion
```python
def ensure_type(data, dtype=np.float32):
    """Safely convert to target dtype."""
    if data is None:
        return None
    return np.asarray(data, dtype=dtype)
```

### Shape Validation
```python
def check_compatible(a, b, axis=0):
    """Check if arrays are compatible along axis."""
    if a.shape[axis] != b.shape[axis]:
        raise ValueError(
            f"Incompatible shapes: {a.shape} vs {b.shape} "
            f"along axis {axis}"
        )
```

### Optional Processing
```python
def process_optional(data, processor, default=None):
    """Process data if provided, otherwise return default."""
    if data is None:
        return default
    return processor(data)
```

## Dependencies

Internal:
- `io.compiler`: For demo scene creation
- `typing_utils`: Type definitions and constants
- `core`: Scene graph components

External:
- `numpy`: Array operations
- `arbol`: Progress display in demos