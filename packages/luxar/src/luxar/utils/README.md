# Utils Package

The `utils` package provides utility functions for common operations in Luxar, including array manipulation and demo data generation.

## Overview

This package contains helper functions that simplify common tasks and provide convenient demo data generators for testing and examples.

## Modules

### `array.py`
Array manipulation and broadcasting utilities.

**Key Functions:**
- `broadcast_color_to_points()`: Broadcast color specification to all points
- `broadcast_radii_to_points()`: Broadcast radius values to points
- `broadcast_sharpness_to_points()`: Broadcast sharpness values
- `broadcast_scalar_to_points()`: Generic scalar broadcasting
- `ensure_float32()`: Convert arrays to float32
- `validate_array_shape()`: Check array dimensions

**Features:**
- Smart broadcasting for scalar, per-point, and per-component values
- Automatic type conversion with validation
- Support for various input formats (lists, tuples, arrays)

### `demos.py`
Demo scene generators for examples and testing.

**Key Functions:**
- `create_lorenz_attractor()`: Generate Lorenz attractor visualization
- `create_random_spheres()`: Create random spherical point clouds
- `create_time_series_demo()`: Generate time-varying data

**Features:**
- Ready-to-use demo scenes
- Configurable parameters
- Educational examples of Luxar features

## Usage Examples

### Array Broadcasting

```python
from luxar.utils import broadcast_color_to_points

# Single color for all points
color1 = broadcast_color_to_points([1, 0, 0], n_points=1000)
# Result: (1000, 3) array of red

# Per-point grayscale
grays = np.random.rand(1000)
color2 = broadcast_color_to_points(grays, n_points=1000)
# Result: (1000, 3) with R=G=B=gray value

# Already correct shape
colors = np.random.rand(1000, 3)
color3 = broadcast_color_to_points(colors, n_points=1000)
# Result: Same array, validated
```

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

### Smart Broadcasting Patterns

```python
from luxar.utils import broadcast_scalar_to_points

# Broadcast single value
radii = broadcast_scalar_to_points(0.5, n_points=100)
# Result: [0.5, 0.5, ..., 0.5]

# Pass through array
radii = np.random.rand(100)
result = broadcast_scalar_to_points(radii, n_points=100)
# Result: Same array if shape matches

# Error on mismatch
radii = np.random.rand(50)
result = broadcast_scalar_to_points(radii, n_points=100)
# Raises: ValueError with helpful message
```

## Broadcasting Rules

### Color Broadcasting

Input Shape | Output Shape | Description
------------|--------------|-------------
`(3,)` | `(N, 3)` | Single RGB color to all points
`(N,)` | `(N, 3)` | Grayscale values to RGB
`(N, 3)` | `(N, 3)` | Already correct, validated
`float` | `(N, 3)` | Single gray value to all

### Scalar Broadcasting

Input | Output | Description
------|--------|-------------
`float` | `(N,)` | Broadcast to all points
`(N,)` | `(N,)` | Validate and pass through
`(M,)` where M≠N | Error | Shape mismatch

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
    n_spheres: int = 10,
    points_per_sphere: int = 1000
) -> None:
```
Creates multiple spherical point clouds:
- Random positions and sizes
- Different colors per sphere
- Hierarchical organization

### Time Series Demo
```python
def create_time_series_demo(
    store_path: str,
    n_timesteps: int = 100,
    n_points: int = 1000
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