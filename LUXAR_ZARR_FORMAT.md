# Luxar Zarr Format Specification

## Version: 0.3

This document specifies the Zarr-based storage format used by Luxar for high-performance 3D and nD point cloud visualization.

## Overview

The Luxar Zarr format is a hierarchical data structure designed for efficient storage and streaming of large-scale point cloud data with support for arbitrary dimensionality, transformations, and rendering attributes.

## Format Structure

```
scene.zarr/
├── .zattrs                  # Scene-level metadata
├── .zgroup                  # Zarr group marker
├── .zmetadata              # Consolidated metadata (optional, created by finalize())
└── <node_name>/            # Scene nodes (groups or point clouds)
    ├── .zattrs             # Node-level metadata
    ├── .zgroup             # Zarr group marker
    ├── positions/          # Point positions (required for point clouds)
    ├── colors/             # Point colors (optional)
    ├── radii/              # Point radii (optional)
    ├── sharpness/          # Point sharpness (optional)
    └── <child_nodes>/      # Nested child nodes (recursive structure)
```

## Scene-Level Metadata (.zattrs)

The root `.zattrs` file contains scene-wide configuration:

```json
{
  "luxar_version": "0.3",
  "type": "scene",
  "units": "um",  // Physical units (nm, um, mm, cm, m, meter, metre, km, inch, foot, px, au)
  "scene_dimensions": {  // Optional: Scene-level dimension specification
    "dimensions": [
      {
        "name": "x",
        "unit": "um",
        "range": [-100.0, 100.0],  // Optional bounds
        "step": 1.0,                // Optional navigation step size
        "display": true,             // Whether dimension is displayed (max 3)
        "discrete": false,           // Whether dimension has discrete values
        "cyclic": false,            // Whether dimension wraps around
        "scale": 1.0,               // Physical scale factor
        "description": "X axis"     // Optional description
      },
      // ... more dimensions
    ]
  },
  "dimension_metadata": [  // Legacy: Deprecated, use scene_dimensions
    {
      "name": "x",
      "unit": "um",
      "scale": 1.0,
      "range": [-100.0, 100.0]
    },
    // ... more dimensions
  ]
}
```

## Node Types

### 1. Group Nodes

Group nodes organize the scene hierarchy and can contain child nodes.

**Attributes (.zattrs):**
```json
{
  "type": "group",
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],  // 4x4 matrix as 16-element array
  "opacity": 1.0,           // 0.0-1.0, inherited by children
  "gamma": 1.0,            // 0.2-2.0, gamma correction
  "blending_mode": "additive"  // normal, additive, multiply, minimum, maximum
}
```

### 2. Point Cloud Nodes

Point cloud nodes contain the actual point data.

**Attributes (.zattrs):**
```json
{
  "type": "points",
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  "opacity": 1.0,
  "gamma": 1.0,
  "blending_mode": "additive",
  "dimension_metadata": [  // Optional: Per-node dimension metadata
    {"name": "x", "unit": "um", "scale": 1.0},
    // ... for each dimension
  ]
}
```

**Data Arrays:**

#### positions/ (Required)
- **Shape:** `(N, D)` where N = number of points, D = dimensionality
- **Dtype:** `float32`
- **Chunks:** `(min(N, 32768), D)` for 2D chunking
- **Compression:** Blosc with zstd, level 3, bit-shuffle
- **Description:** Point positions in D-dimensional space

#### colors/ (Optional)
- **Shape:** `(N, 3)` for RGB
- **Dtype:** `uint8`
- **Chunks:** `(min(N, 32768), 3)`
- **Compression:** Blosc with zstd, level 3, bit-shuffle
- **Description:** RGB colors, range 0-255
- **Default:** White (255, 255, 255) if not provided

#### radii/ (Optional)
- **Shape:** `(N,)`
- **Dtype:** `float32`
- **Chunks:** `(min(N, 32768),)` for 1D chunking
- **Compression:** Blosc with zstd, level 3, bit-shuffle
- **Description:** Point radii in scene units
- **Default:** 0.1 if not provided
- **Validation:** All values must be positive

#### sharpness/ (Optional)
- **Shape:** `(N,)`
- **Dtype:** `float32`
- **Chunks:** `(min(N, 32768),)` for 1D chunking
- **Compression:** Blosc with zstd, level 3, bit-shuffle
- **Description:** Point edge sharpness (0.5-10.0 typical range)
- **Default:** 2.0 if not provided
- **Validation:** All values must be positive

## Transform System

Transforms are stored as 16-element arrays representing 4x4 homogeneous transformation matrices in row-major order:

```
[m00, m01, m02, tx,
 m10, m11, m12, ty,
 m20, m21, m22, tz,
 0,   0,   0,   1]
```

Where:
- `m00-m22`: 3x3 rotation/scale matrix
- `tx, ty, tz`: Translation vector
- Bottom row is always `[0, 0, 0, 1]`

## Dimension System

### Scene Dimensions
Scene-level dimensions define the coordinate system for all objects:
- Maximum 3 dimensions can be displayed simultaneously
- Non-displayed dimensions are used for slicing/navigation
- Dimensions include metadata for units, ranges, and navigation

### nD Point Cloud Support
- Points can have arbitrary dimensionality (not limited to 3D)
- Viewer performs slicing for dimensions > 3
- Radius-based visibility: points are visible if their nD hypersphere intersects the current slice

## Chunking Strategy

Optimal chunk sizes balance memory usage and access patterns:
- **Default chunk size:** 32,768 elements
- **Minimum chunk size:** 1,024 elements
- **Maximum chunk size:** 262,144 elements
- **2D arrays (positions, colors):** Chunk along first dimension only
- **1D arrays (radii, sharpness):** Simple 1D chunking

## Compression

Default compression uses Blosc with:
- **Codec:** zstd (balanced speed/ratio)
- **Level:** 3 (moderate compression)
- **Shuffle:** bit-shuffle (optimized for scientific data)

Supported alternatives:
- Blosc with lz4, blosclz, snappy, zlib
- Native: gzip, bz2, lzma
- External: zstd, lz4

## Metadata Consolidation

After scene construction, call `scene.finalize()` to:
- Consolidate all metadata into `.zmetadata` file
- Improve load performance by reducing metadata requests
- Enable efficient streaming from remote stores

## Version History

- **0.3** (Current): Added scene dimensions, improved nD support
- **0.2**: Added sharpness attribute, rendering parameters
- **0.1**: Initial format with positions, colors, radii

## Best Practices

1. **Always finalize scenes** after construction for optimal performance
2. **Use appropriate chunk sizes** based on expected access patterns
3. **Store transforms at group level** for hierarchical transformations
4. **Define scene dimensions** for consistent coordinate systems
5. **Use consolidated metadata** for remote data access
6. **Validate all data** before writing to ensure consistency

## Example Creation (Python)

```python
import numpy as np
from luxar import Scene, Dimensions, Dimension

# Create scene with dimensions
dims = Dimensions([
    Dimension("x", unit="um", display=True),
    Dimension("y", unit="um", display=True),
    Dimension("z", unit="um", display=True),
    Dimension("time", unit="ms", display=False, discrete=True)
])

scene = Scene("output.zarr", dimensions=dims)

# Add point cloud
positions = np.random.randn(10000, 4).astype(np.float32)  # 4D points
colors = np.random.randint(0, 255, (10000, 3), dtype=np.uint8)
radii = np.ones(10000, dtype=np.float32) * 0.5

scene.add_points("my_points", positions, colors, radii=radii)

# Finalize for optimal loading
scene.finalize()
```

## Compatibility Notes

- **Zarr Version:** Format 2 (for JavaScript compatibility)
- **NumCodecs:** Required for compression support
- **Consolidated Metadata:** Recommended for web streaming
- **Browser Support:** Via zarrita.js library

## Future Extensions (Planned)

- Multiple blending modes per layer
- Support for meshes, lines, volumes
- Material system with shading models
- Temporal interpolation for smooth animations
- Hierarchical level-of-detail (LOD) support