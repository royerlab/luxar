# luxar.core

The `core` module contains the fundamental data structures and classes that form the foundation of Luxar's scene graph system. This includes scene nodes, data containers (Points, Lines, GSplats), dimensional specifications, and transformation utilities.

## Overview

The core module implements Luxar's hierarchical scene graph architecture, enabling organization of large-scale visualization data with transforms, metadata, and nD dimensional support.

## Getting Started

### Quick Example - Create Your First Scene

```python
import numpy as np
from luxar import LuxarZarrCompiler, Dimensions, Dimension

# Create sample data
positions = np.random.randn(1000, 3).astype(np.float32)
colors = np.random.rand(1000, 3).astype(np.float32)

# Create scene
with LuxarZarrCompiler('my_scene.zarr') as compiler:
    # Define 3D dimensions
    dims = Dimensions([
        Dimension("x", unit="um", display=True),
        Dimension("y", unit="um", display=True),
        Dimension("z", unit="um", display=True),
    ])

    compiler.create_scene(dimensions=dims)

    # Add points
    compiler.write_points(
        "MyPoints",
        positions=positions,
        colors=colors,
        radii=1.0,  # Scalar broadcasts to all points
        opacity=0.8,
        blending_mode="additive"
    )

# View with: luxar serve my_scene.zarr --viewer
```

**Key Concepts:**
- **Scene**: Root container defining dimensions
- **Nodes**: Hierarchical organization (groups can contain groups/data)
- **DataNodes**: Points, Lines, GSplats - the actual renderable data
- **Transforms**: 4x4 matrices for positioning/rotation/scaling
- **Dimensions**: Support nD data with keyboard navigation

## Key Components

### 1. Scene (`scene.py`)

The root node of the scene hierarchy. Provides builder methods for constructing complex scenes with multiple data types.

**Key Features:**
- Progressive writing through `LuxarZarrCompiler`
- Scene-level dimension definitions
- Broadcasting support for nD data
- Hierarchical organization with groups
- Support for Points, Lines, and GSplats data

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

    # Add different data types
    positions = np.random.randn(10000, 4).astype(np.float32)
    scene.add_points('my_points', positions)
    
    # Add lines
    vertices = np.random.randn(100, 4).astype(np.float32)
    scene.add_lines('my_lines', vertices, widths=0.1)
    
    # Add Gaussian splats
    centers = np.random.randn(500, 4).astype(np.float32)
    amplitudes = np.ones(500)
    cholesky = np.random.randn(500, 10).astype(np.float32)
    scene.add_gsplats('my_splats', centers, amplitudes, cholesky)
```

**Key Methods:**
- `add_group(name, **attrs)` - Create child group node
- `add_points(name, positions, ...)` - Add points with attributes (radii defaults to 0.5 if not provided)
- `add_lines(name, vertices, widths, ...)` - Add lines/curves
- `add_gsplats(name, centers, amplitudes, ...)` - Add Gaussian splats from arrays
- `add_gsplats_from_data(name, result, ...)` - Add Gaussian splats from GSplatData object
- `add_gsplats_from_file(name, path, ...)` - Add Gaussian splats by loading from .gsplats.zarr file
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
- `gamma` - Gamma correction (0.1-10.0)
- `blending_mode` - Blending mode ('normal', 'additive')
- `children` - List of child nodes
- `parent` - Parent node reference

**Important Notes:**
- Transforms are automatically transposed for THREE.js compatibility when stored
- Nodes use writer interface for progressive writing without keeping data in memory
- All rendering attributes are validated on assignment

### 3. DataNode (`datanode.py`)

Abstract base class for all data-bearing nodes (Points, Lines, GSplats).

**Purpose:**
Provides common interface and behavior for all nodes that contain visualization data.

**Key Features:**
- Immediate writing to Zarr (no data kept in memory)
- Type-specific metadata storage
- Unified `n_elements` property
- Support for semantic type mapping for encoding

**Subclasses Must Implement:**
- `n_elements` property - Returns count of primary elements

**Usage Example:**
```python
# DataNode is not instantiated directly, but used through subclasses
# All data nodes share common interface:
print(f"Elements: {data_node.n_elements}")
print(f"Dimensions: {data_node.ndim}")
print(f"Metadata: {data_node.metadata}")
```

**Key Properties:**
- `n_elements` - Number of primary elements (abstract, implemented by subclasses)
- `ndim` - Dimensionality of data
- `metadata` - Type-specific metadata dictionary

**Inheritance Hierarchy:**
```
Node
 └── DataNode (abstract)
      ├── Points
      ├── Lines
      └── GSplats
```

### 4. Points (`points.py`)

Specialized node for point cloud data. Lightweight metadata container in progressive mode.

**Key Features:**
- Metadata-only in progressive writing (data written immediately to Zarr)
- Tracks data characteristics (n_points, has_colors, has_radii, etc.)
- Inherits all Node and DataNode capabilities

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
print(f"Elements: {points.n_elements}")  # Alias for n_points
```

**Key Properties:**
- `n_points` - Number of points
- `n_elements` - Alias for n_points (DataNode protocol)
- `has_colors` - Whether colors are present
- `has_radii` - Whether radii are present
- `has_sharpness` - Whether sharpness is present
- `metadata` - Full metadata dictionary

### 5. Lines (`lines.py`)

Node for line and curve data (polylines, segments, loops).

**Purpose:**
Represents 1D structures like trajectories, fiber tracts, network edges, or arbitrary curves.

**Key Features:**
- Supports multiple line types (segments, polyline, loop, indexed)
- Per-vertex colors and sharpness
- Variable line widths
- Progressive writing (data written immediately to Zarr)

**Line Types:**
- `"segments"` - Independent line segments (pairs of vertices)
- `"polyline"` - Connected line strip
- `"loop"` - Closed loop (last connects to first)
- `"indexed"` - Custom connectivity via indices array

**Usage Example:**
```python
# Create polyline from trajectory
trajectory = np.random.randn(1000, 3).astype(np.float32)
widths = np.linspace(0.1, 0.5, 1000)
colors = np.random.rand(1000, 3).astype(np.float32)

lines = scene.add_lines('trajectory',
                       vertices=trajectory,
                       widths=widths,
                       colors=colors,
                       line_type='polyline')

# Query metadata
print(f"Vertices: {lines.n_vertices:,}")
print(f"Segments: {lines.n_segments:,}")
print(f"Line type: {lines.line_type}")
print(f"Max width: {lines.max_width}")
```

**Key Properties:**
- `n_vertices` - Number of vertices
- `n_elements` - Alias for n_vertices (DataNode protocol)
- `n_segments` - Number of line segments
- `line_type` - Type of line connectivity
- `has_colors` - Whether per-vertex colors are present
- `has_sharpness` - Whether per-vertex sharpness is present
- `max_width` - Maximum line width

**Arrays:**
- `vertices` - Shape (N, D) vertex positions
- `widths` - Shape (N,) line widths (or scalar broadcast)
- `colors` - Shape (N, 3) per-vertex colors (optional)
- `sharpness` - Shape (N,) edge sharpness (optional)
- `indices` - Vertex indices for indexed line type (optional)

### 6. GSplats (`gsplats.py`)

Node for Gaussian splat data (oriented anisotropic Gaussians).

**Purpose:**
Represents data as generalized Gaussian distributions, useful for 3D Gaussian Splatting, uncertainty visualization, or smooth field representations.

**Key Features:**
- Generalized Gaussian kernels (not limited to standard Gaussians)
- Anisotropic covariances via Cholesky factorization
- Variable amplitudes (intensities)
- Sharpness parameter for generalized Gaussian exponent
- Progressive writing (data written immediately to Zarr)

**Mathematical Representation:**
Each splat is defined by:
- Center position: μ ∈ ℝᴰ
- Cholesky factor: L (lower triangular)
- Amplitude: α (intensity/weight)
- Sharpness: β (generalized Gaussian exponent, default 2.0)

The splat function: `f(x) = α * exp(-||L(x - μ)||^β)`

**Usage Example:**
```python
# Create Gaussian splats
n_splats = 1000
centers = np.random.randn(n_splats, 3).astype(np.float32)
amplitudes = np.abs(np.random.randn(n_splats))

# Cholesky factors for 3D: 6 values per splat (packed lower triangle)
# For D dimensions: D*(D+1)/2 values per splat
cholesky = np.random.randn(n_splats, 6).astype(np.float32)

# Optional: colors and sharpness
colors = np.random.rand(n_splats, 3).astype(np.float32)
sharpness = np.full(n_splats, 2.0)  # Standard Gaussian

splats = scene.add_gsplats('gaussians',
                          centers=centers,
                          amplitudes=amplitudes,
                          cholesky_factors=cholesky,
                          colors=colors,
                          sharpness=sharpness)

# Query metadata
print(f"Splats: {splats.n_splats:,}")
print(f"Amplitude range: {splats.amplitude_range}")
print(f"Center bounds: {splats.center_bounds}")
```

**Key Properties:**
- `n_splats` - Number of splats
- `n_elements` - Alias for n_splats (DataNode protocol)
- `has_colors` - Whether splat colors are present
- `has_sharpness` - Whether sharpness values are present
- `ordering` - Spatial ordering type (e.g., 'morton', 'none')
- `amplitude_range` - Min/max amplitude values
- `center_bounds` - Bounding box of centers

**Arrays:**
- `centers` - Shape (N, D) splat centers
- `amplitudes` - Shape (N,) intensities (or scalar broadcast)
- `cholesky_factors` - Shape (N, k) where k=D*(D+1)/2 (packed lower triangle)
- `colors` - Shape (N, 3) splat colors (optional)
- `sharpness` - Shape (N,) generalized Gaussian exponent (optional, default 2.0)

**Convenience Methods for GSplats:**

In addition to `add_gsplats()` which requires explicit arrays, Scene provides convenience methods for common workflows:

**From GSplatData:**
```python
from luxar.gsplats import fit_gaussian_splats

# Fit Gaussian splats to image
image = np.random.rand(100, 100).astype(np.float32)
result = fit_gaussian_splats(image, n_iters=1000)

# Add directly to scene (no intermediate save)
with LuxarZarrCompiler('scene.zarr') as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    gsplats = scene.add_gsplats_from_data('fitted', result)
    print(f"Added {gsplats.n_splats} splats with colors={gsplats.has_colors}")
```

**From .gsplats.zarr File:**
```python
# Load previously saved Gaussian splats
with LuxarZarrCompiler('scene.zarr') as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    gsplats = scene.add_gsplats_from_file('loaded', 'path/to/fitted.gsplats.zarr')
    print(f"Loaded {gsplats.n_splats} splats")
```

These methods automatically handle:
- Extracting arrays from GSplatData objects
- Loading data from .gsplats.zarr archives
- Passing all data (centers, amplitudes, cholesky_factors, colors, sharpness) to add_gsplats()
- Preserving optional attributes (colors, sharpness) when present

### 7. Dimensions (`dimensions.py`)

Scene-level coordinate system definitions with support for categorical dimensions.

**Key Classes:**
- `Dimension` - Single dimension specification
- `Dimensions` - Complete scene dimension system

**Key Features:**
- Support for arbitrary dimensionality (not limited to 3D)
- Displayed vs non-displayed dimensions
- Discrete vs continuous dimensions
- **Categorical dimensions** with string labels
- Spatial extension flags for point coverage
- Navigation properties (step sizes, ranges)

**Usage Example:**
```python
from luxar.core.dimensions import Dimension, Dimensions

# Define 5D space with categorical channel dimension
dims = Dimensions([
    Dimension('x', unit='um', display=True),
    Dimension('y', unit='um', display=True),
    Dimension('z', unit='um', display=True),
    Dimension('time', unit='s', display=False, discrete=True,
              range=(0, 99), step=1.0),
    Dimension('channel', unit='ch', display=False, discrete=True,
              categories=['DAPI', 'GFP', 'mCherry'])  # Categorical!
])

# Query properties
print(f"Total dimensions: {dims.ndim}")
print(f"Displayed: {dims.displayed}")
print(f"Non-displayed: {dims.non_displayed}")

# Check categorical dimensions
channel_dim = dims.get_dimension('channel')
print(f"Is categorical: {channel_dim.is_categorical}")
print(f"Categories: {channel_dim.categories}")
```

**Dimension Properties:**
- `name` - Dimension identifier
- `unit` - Physical unit
- `display` - Whether dimension is displayed (max 3)
- `discrete` - Whether values are discrete
- `cyclic` - Whether dimension wraps around
- `scale` - Physical scale factor
- `spatial` - Whether points extend through this dimension
- `categories` - **NEW:** Optional list of category labels for categorical dimensions
- `description` - Human-readable description
- `range` - Optional (min, max) bounds
- `step` - Navigation step size

**Categorical Dimensions:**

Categorical dimensions allow string labels instead of numeric coordinates:

```python
# Define categorical dimension
channel_dim = Dimension(
    'channel',
    categories=['DAPI', 'GFP', 'mCherry', 'Cy5'],
    display=False
)

# Categorical dimensions are automatically:
# - discrete = True (enforced)
# - range = (0, len(categories)-1) (auto-set if not provided)
# - step = 1.0 (auto-set if not provided)

# In data, use integer indices (0-based):
# 0 = 'DAPI', 1 = 'GFP', 2 = 'mCherry', 3 = 'Cy5'
positions = np.array([
    [10.0, 20.0, 5.0, 0.0],  # Channel 0 (DAPI)
    [11.0, 21.0, 5.5, 1.0],  # Channel 1 (GFP)
    [12.0, 22.0, 6.0, 2.0],  # Channel 2 (mCherry)
])
```

**Categorical Dimension Features:**
- String labels for human-readable dimension values
- Automatic validation of category indices
- Preserved in zarr metadata for viewer display
- Useful for: channels, cell types, experimental conditions, time-lapse phases
- Categories must be unique, non-empty strings

**Automatic Behaviors:**
- `spatial` flag auto-determined from `display` if not specified
- Non-spatial, non-displayed dimensions automatically marked discrete
- Step sizes auto-calculated if not provided
- Categorical dimensions auto-set discrete=True, range, and step

### 8. Transforms (`transforms.py`)

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

**Visual Composition Example:**
```
Point → [Translate] → [Rotate] → [Scale] → Final Position
        ↑             ↑           ↑
     compose(Translate, Rotate, Scale)
        ↑             ↑           ↑
      First        Second       Last
     Applied      Applied      Applied

Example: compose(translate(5,0,0), rotate_z(90°), scale(2,2,2))
Point (1,0,0) → Translate → (6,0,0) → Rotate → (0,6,0) → Scale → (0,12,0)
```

**Important Notes:**
- All matrices are 4x4 homogeneous transforms (float32)
- Matrices are automatically transposed for THREE.js when stored
- Use `to_list()` and `from_list()` for serialization (handles transpose)
- Composition order: `compose(A, B, C)` applies A first, then B, then C (LEFT-to-RIGHT)
- In matrix math: `result = result @ A @ B @ C` (right-multiplication)

## Architecture

### Progressive Writing Design

The core module is designed to work with Luxar's progressive writing system:

1. **Scene Creation**: Scene created with a writer (LuxarZarrCompiler)
2. **Node Creation**: Nodes are lightweight metadata containers
3. **Data Writing**: Data written immediately to Zarr via writer
4. **Memory Efficiency**: Data never kept in memory after writing

### Scene Graph Structure

```
Scene (root)
├── Group "cells"
│   ├── Points "cell_1"
│   ├── Lines "cell_edges"
│   └── GSplats "cell_uncertainty"
└── Group "markers"
    ├── Points "marker_points"
    └── Lines "marker_connections"
```

### Transform Hierarchy

Transforms compose hierarchically:
- Each node can have a local transform
- Final transform = parent_transform @ local_transform
- Transforms are automatically applied by the viewer

### Data Node Hierarchy

All data-bearing nodes inherit from DataNode:

```
Node (base class)
 └── DataNode (abstract base for data nodes)
      ├── Points (point cloud data)
      ├── Lines (curve/line data)
      └── GSplats (Gaussian splat data)
```

## Dependencies

**Internal:**
- `luxar.typing_utils` - Type definitions and validation
- `luxar.io.writer` - Writer protocol for progressive writing
- `luxar.utils.array` - Array broadcasting helpers
- `luxar.validation` - Data validation utilities
- `luxar.encoding` - Data type encoding/decoding

**External:**
- `numpy` - Array operations
- `zarr` - Data storage (indirect, through writer)
- `arbol` - Structured logging

## Testing

Tests are located in `core/tests/`:
- `test_dimensions.py` - Dimension system tests (including categorical)
- `test_node_rendering.py` - Node rendering properties
- `test_scene_methods.py` - Scene builder methods (all data types)
- `test_scene_structure.py` - Scene graph structure
- `test_transforms.py` - Transform utilities
- `test_spatial_dimensions.py` - Spatial dimension handling
- `test_datanode.py` - DataNode base class tests
- `test_lines.py` - Lines node tests
- `test_gsplats.py` - GSplats node tests

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
- **Valid categories** (unique, non-empty strings)
- **Valid category indices** in data

### Node Attribute Storage

Nodes cache attributes and write them immediately:
- Attributes cached in `_attrs_cache` for fast access
- Written immediately to Zarr via writer interface
- Rendering attributes validated on assignment
- No Zarr groups kept in memory (memory-efficient)

### Categorical Dimension Validation

Categorical dimensions undergo additional validation:
- Categories must be list of strings
- Each category must be unique
- Empty strings not allowed
- Category labels limited to 1024 characters each
- Position values must be integer indices (0-based)
- Indices must be in range [0, len(categories)-1]

See `luxar.validation.validate_categories()` and `luxar.validation.validate_category_indices()`.

## Best Practices

### 1. Always Use Context Manager
```python
with LuxarZarrCompiler('output.zarr') as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    # ... build scene
# Automatically finalized
```

### 2. Define Dimensions Early
```python
# Define dimensions before creating scene
dims = Dimensions([...])
scene = compiler.create_scene(dimensions=dims)
```

### 3. Use Explicit Dimension Extension
```python
# Explicit is better than implicit - works for both points and lines
scene.add_points('pts', positions,
                extend_to_all=['time', 'channel'])

# Lines can also extend across dimensions (e.g., static detector geometry)
scene.add_lines('detector', vertices, widths=0.1,
               extend_to_all=['time'])  # Visible at all time values
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

### 6. Choose Appropriate Data Types
```python
# Points for discrete particles
scene.add_points('particles', positions)

# Lines for trajectories/networks
scene.add_lines('trajectories', vertices, widths=0.1, line_type='polyline')

# GSplats for smooth fields/uncertainty
scene.add_gsplats('uncertainty', centers, amplitudes, cholesky)
```

### 7. Use Categorical Dimensions for Discrete Labels
```python
# Better than numeric channel indices
Dimension('channel', categories=['DAPI', 'GFP', 'mCherry'])

# Instead of
Dimension('channel', range=(0, 2), discrete=True)  # What does 0 mean?
```

## See Also

- [io/README.md](../io/README.md) - I/O operations and writers
- [typing_utils/README.md](../typing_utils/README.md) - Type system
- [validation/README.md](../validation/README.md) - Validation utilities
- [encoding/README.md](../encoding/README.md) - Data encoding and semantic types
- [Main README](../../../../README.md) - Project overview
