# Core Package

The `core` package contains the fundamental data structures and scene graph implementation for Luxar.

## Overview

This package provides the core building blocks for creating and manipulating 3D/nD point cloud scenes with hierarchical structure and transformations.

## Modules

### `node.py`
Base class for all scene graph nodes.

**Key Classes:**
- `Node`: Base node class with transform support and hierarchical structure

**Key Features:**
- Hierarchical parent-child relationships
- 4x4 transform matrices with automatic validation
- Attribute storage for metadata
- Rendering properties (opacity, gamma, blending mode)
- Tree traversal utilities

### `scene.py`
Root node of the scene graph.

**Key Classes:**
- `Scene`: Top-level container for all scene content

**Key Features:**
- Dimension system integration for nD data
- Point cloud creation with automatic validation
- Group node creation for organization
- Progressive writing support via writer injection
- Convenient helper methods for common point cloud patterns

### `points.py`
Point cloud node implementation.

**Key Classes:**
- `Points`: Lightweight metadata container for point cloud data

**Key Features:**
- Metadata-only storage (actual data written to Zarr)
- Automatic shape tracking
- Integration with progressive writing system

### `dimensions.py`
Dimension system for nD point clouds.

**Key Classes:**
- `Dimension`: Single dimension specification with name, unit, range, and step
- `Dimensions`: Collection of dimensions defining a coordinate system

**Key Features:**
- Support for arbitrary number of dimensions
- Physical units for each dimension
- Display control (which dimensions are visible)
- Step sizes for keyboard navigation
- Serialization to/from dictionaries

### `transforms.py`
Transform utilities for 3D transformations.

**Key Functions:**
- `identity()`: Create identity matrix
- `translate(x, y, z)`: Create translation matrix
- `scale(x, y, z)`: Create scale matrix
- `rotate_x/y/z(degrees)`: Create rotation matrices
- `compose(*transforms)`: Compose multiple transforms
- `inverse(transform)`: Compute inverse transform
- `look_at(eye, target, up)`: Create look-at matrix

**Key Features:**
- All transforms are 4x4 matrices (float32)
- Automatic validation
- Column-major storage for THREE.js compatibility
- Convenient aliases for common operations

## Usage Examples

### Creating a Scene with Points

```python
from luxar.core import Scene, Dimensions, Dimension
from luxar import LuxarZarrCompiler
import numpy as np

# Create compiler for progressive writing
with LuxarZarrCompiler('output.zarr') as compiler:
    # Define dimensions for 4D data
    dims = Dimensions([
        Dimension("x", "um", (-100, 100), 1.0),
        Dimension("y", "um", (-100, 100), 1.0),
        Dimension("z", "um", (-50, 50), 0.5),
        Dimension("time", "ms", (0, 1000), 10.0),
    ])
    
    # Create scene with dimensions
    scene = compiler.create_scene(dimensions=dims)
    
    # Add points
    positions = np.random.randn(10000, 4).astype(np.float32)
    colors = np.random.rand(10000, 3).astype(np.float32)
    scene.add_points("my_points", positions, colors)
```

### Working with Transforms

```python
from luxar.core import transforms
import numpy as np

# Create a transform chain
t1 = transforms.translate(10, 0, 0)
t2 = transforms.rotate_z(45)
t3 = transforms.scale(2, 2, 2)

# Compose transforms (applied right to left)
combined = transforms.compose(t3, t2, t1)

# Apply to a node
node.transform = combined
```

### Building Hierarchical Scenes

```python
# Create groups for organization
group1 = scene.add_group("molecules")
group2 = scene.add_group("cells", parent=group1)

# Add points to specific groups
scene.add_points("proteins", positions1, parent=group1)
scene.add_points("organelles", positions2, parent=group2)

# Set rendering properties
group1.opacity = 0.8
group1.blending_mode = "additive"
```

## Architecture Notes

### Memory Efficiency
- Point data is immediately written to Zarr, not kept in memory
- Nodes only store metadata and array shapes
- Progressive writing enables TB-scale datasets

### Transform System
- Transforms compose hierarchically (parent → child)
- Stored as column-major for THREE.js compatibility
- Automatic validation ensures matrices are valid

### Dimension System
- Supports unlimited dimensions (not just 3D)
- Non-displayed dimensions are "sliced" in viewer
- Step sizes enable keyboard navigation in viewer

## Dependencies

Internal:
- `typing_utils`: Type definitions and protocols
- `io.writer`: Progressive writing interface
- `utils.array`: Array manipulation helpers
- `validation`: Input validation

External:
- `numpy`: Array operations
- `zarr`: Storage backend (via writer)
- `arbol`: Logging and progress display