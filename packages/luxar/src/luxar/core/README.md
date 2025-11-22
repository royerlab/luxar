# luxar.core

The `core` module contains the fundamental data structures and classes that form the foundation of Luxar's scene graph system. This includes scene nodes, point containers, dimensional specifications, and transformation utilities.

## Overview

The core module implements Luxar's hierarchical scene graph architecture, enabling organization of large-scale point cloud data with transforms, metadata, and nD dimensional support.

## Key Components

### 1. Scene (`scene.py`)

The root node of the scene hierarchy. Provides builder methods for constructing complex scenes.

**Key Features:**
- Progressive writing through `LuxarZarrCompiler`
- Scene-level dimension definitions
- Broadcasting support for nD data
- Hierarchical organization with groups

**Usage Example:**
```python
from luxar import LuxarZarrCompiler, Dimensions, Dimension
import numpy as np

# Define 4D scene dimensions
dims = Dimensions([
    Dimension('x', unit='um', display=True),
    Dimension('y', unit='um', display=True),
    Dimension('z', unit='um', display=True),
    Dimension('time', unit='s', display=False, discrete=True, range=(0, 99))
])

# Create scene with progressive writer
with LuxarZarrCompiler('output.zarr') as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Add points with all dimensions
    positions = np.random.randn(10000, 4).astype(np.float32)
    scene.add_points('my_points', positions)
```

**Key Methods:**
- `add_group(name, **attrs)` - Create child group node
- `add_points(name, positions, ...)` - Add points with attributes
- `dimensions` (property) - Get/set scene-level dimensions

### 2. Node (`node.py`)

Base class for all scene graph nodes. Represents groups in the hierarchy.

**Key Features:**
- Hierarchical parent-child relationships
- Transform support (4x4 matrices)
- Rendering properties (opacity, gamma, blending mode)
- Progressive writing without keeping Zarr groups in memory

**Usage Example:**
```python
# Nodes are typically created via Scene.add_group()
group = scene.add_group('my_group',
                        opacity=0.8,
                        gamma=1.2,
                        blending_mode='additive')

# Transforms can be set directly
group.transform = luxar.translate(5, 0, 0)

# Or chained
group.set_opacity(0.5).set_gamma(1.0)
```

**Key Properties:**
- `transform` - 4x4 transformation matrix
- `opacity` - Rendering opacity (0.0-1.0)
- `gamma` - Gamma correction (0.2-2.0)
- `blending_mode` - Blending mode ('normal', 'additive')
- `children` - List of child nodes
- `parent` - Parent node reference

**Important Notes:**
- Transforms are automatically transposed for THREE.js compatibility when stored
- Nodes use writer interface for progressive writing without keeping data in memory
- All rendering attributes are validated on assignment

### 3. Points (`points.py`)

Specialized node for point cloud data. Lightweight metadata container in progressive mode.

**Key Features:**
- Metadata-only in progressive writing (data written immediately to Zarr)
- Tracks data characteristics (n_points, has_colors, has_radii, etc.)
- Inherits all Node capabilities

**Usage Example:**
```python
# Points created via Scene.add_points()
points = scene.add_points('cloud',
                         positions=positions,
                         colors=colors,
                         radii=radii,
                         sharpness=sharpness)

# Query metadata
print(f"Points: {points.n_points:,}")
print(f"Has colors: {points.has_colors}")
```

**Key Properties:**
- `n_points` - Number of points
- `has_colors` - Whether colors are present
- `has_radii` - Whether radii are present
- `has_sharpness` - Whether sharpness is present
- `metadata` - Full metadata dictionary

### 4. Dimensions (`dimensions.py`)

Scene-level coordinate system definitions.

**Key Classes:**
- `Dimension` - Single dimension specification
- `Dimensions` - Complete scene dimension system

**Key Features:**
- Support for arbitrary dimensionality (not limited to 3D)
- Displayed vs non-displayed dimensions
- Discrete vs continuous dimensions
- Spatial extension flags for point coverage
- Navigation properties (step sizes, ranges)

**Usage Example:**
```python
from luxar.core.dimensions import Dimension, Dimensions

# Define 5D space (XYZ + Time + Channel)
dims = Dimensions([
    Dimension('x', unit='um', display=True),
    Dimension('y', unit='um', display=True),
    Dimension('z', unit='um', display=True),
    Dimension('time', unit='s', display=False, discrete=True,
              range=(0, 99), step=1.0),
    Dimension('channel', unit='ch', display=False, discrete=True,
              range=(0, 2), step=1.0)
])

# Query properties
print(f"Total dimensions: {dims.ndim}")
print(f"Displayed: {dims.displayed}")
print(f"Non-displayed: {dims.non_displayed}")
```

**Dimension Properties:**
- `name` - Dimension identifier
- `unit` - Physical unit
- `display` - Whether dimension is displayed (max 3)
- `discrete` - Whether values are discrete
- `spatial` - Whether points extend through this dimension
- `range` - Optional (min, max) bounds
- `step` - Navigation step size
- `cyclic` - Whether dimension wraps around

**Automatic Behaviors:**
- `spatial` flag auto-determined from `display` if not specified
- Non-spatial, non-displayed dimensions automatically marked discrete
- Step sizes auto-calculated if not provided

### 5. Transforms (`transforms.py`)

Utilities for creating and manipulating 4x4 transformation matrices.

**Key Functions:**
- `identity()` - Create identity matrix
- `translate(x, y, z)` - Translation matrix
- `rotate_x/y/z(degrees)` - Axis-aligned rotations
- `rotate(degrees, axis)` - Arbitrary axis rotation
- `scale(x, y, z, uniform)` - Scaling matrix
- `compose(*transforms)` - Combine multiple transforms
- `inverse(transform)` - Compute inverse
- `look_at(eye, target, up)` - Camera-style transform
- `to_list(transform)` - Convert to storage format
- `from_list(values)` - Convert from storage format

**Usage Example:**
```python
import luxar

# Create transformation
t1 = luxar.translate(10, 0, 0)
t2 = luxar.rotate_z(45)
t3 = luxar.scale(2, 2, 2)

# Compose (applied in order: translate, then rotate, then scale)
combined = luxar.compose(t1, t2, t3)

# Apply to node
node.transform = combined
```

**Important Notes:**
- All matrices are 4x4 homogeneous transforms (float32)
- Matrices are automatically transposed for THREE.js when stored
- Use `to_list()` and `from_list()` for serialization (handles transpose)
- Composition order: `compose(A, B, C)` applies A first, then B, then C

## Architecture

### Progressive Writing Design

The core module is designed to work with Luxar's progressive writing system:

1. **Scene Creation**: Scene created with a writer (LuxarZarrCompiler)
2. **Node Creation**: Nodes are lightweight metadata containers
3. **Data Writing**: Point data written immediately to Zarr via writer
4. **Memory Efficiency**: Data never kept in memory after writing

### Scene Graph Structure

```
Scene (root)
├── Group "cells"
│   ├── Points "cell_1"
│   └── Points "cell_2"
└── Group "markers"
    └── Points "marker_points"
```

### Transform Hierarchy

Transforms compose hierarchically:
- Each node can have a local transform
- Final transform = parent_transform @ local_transform
- Transforms are automatically applied by the viewer

## Dependencies

**Internal:**
- `luxar.typing_utils` - Type definitions and validation
- `luxar.io.writer` - Writer protocol for progressive writing
- `luxar.utils.array` - Array broadcasting helpers

**External:**
- `numpy` - Array operations
- `zarr` - Data storage (indirect, through writer)
- `arbol` - Structured logging

## Testing

Tests are located in `core/tests/`:
- `test_dimensions.py` - Dimension system tests
- `test_node_rendering.py` - Node rendering properties
- `test_scene_methods.py` - Scene builder methods
- `test_scene_structure.py` - Scene graph structure
- `test_transforms.py` - Transform utilities
- `test_spatial_dimensions.py` - Spatial dimension handling

Run tests:
```bash
hatch run pytest packages/luxar/src/luxar/core/tests/
```

## Implementation Notes

### Transform Storage

Transforms are stored in THREE.js-compatible format (column-major):
```python
# NumPy (row-major) → Storage (column-major)
numpy_matrix = np.array([[...], [...], [...], [...]])  # 4x4
storage_list = numpy_matrix.T.ravel().tolist()  # Transpose for THREE.js

# Storage → NumPy
storage_list = [...]  # 16 elements
numpy_matrix = np.array(storage_list).reshape(4, 4).T  # Transpose back
```

This ensures correct interpretation by the THREE.js viewer.

### Dimension Validation

Scene dimensions are validated to ensure:
- Maximum 3 displayed dimensions
- At least 1 displayed dimension if any exist
- Unique dimension names
- Valid ranges (min < max)
- Consistent spatial extension flags

### Node Attribute Storage

Nodes cache attributes and write them immediately:
- Attributes cached in `_attrs_cache` for fast access
- Written immediately to Zarr via writer interface
- Rendering attributes validated on assignment
- No Zarr groups kept in memory (memory-efficient)

## Best Practices

### 1. Always Use Context Manager
```python
with LuxarZarrCompiler('output.zarr') as compiler:
    scene = compiler.create_scene()
    # ... build scene
# Automatically finalized
```

### 2. Define Dimensions Early
```python
# Define dimensions before creating scene
dims = Dimensions([...])
scene = compiler.create_scene(dimensions=dims)
```

### 3. Use Explicit Broadcasting
```python
# Explicit is better than implicit
scene.add_points('pts', positions,
                broadcast_dims=['time', 'channel'])
```

### 4. Validate Data Before Writing
```python
# Positions must match scene dimensions
if scene.dimensions:
    scene.dimensions.validate_positions(positions)
```

### 5. Use Transform Utilities
```python
# Use provided functions instead of manual matrix creation
transform = luxar.compose(
    luxar.translate(5, 0, 0),
    luxar.rotate_z(45)
)
```

## See Also

- [io/README.md](../io/README.md) - I/O operations and writers
- [typing_utils/README.md](../typing_utils/README.md) - Type system
- [validation/README.md](../validation/README.md) - Validation utilities
- [Main README](../../../../README.md) - Project overview
