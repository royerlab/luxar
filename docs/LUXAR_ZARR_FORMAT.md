# Luxar Zarr Format Specification

## Version: 0.1

**Features in v0.1:**
- Chunk-based spatial index for efficient nD point queries
- Points are reordered using Morton/Hilbert space-filling curves for spatial locality
- Compound ordering: discrete dimensions (time/channel) + spatial dimensions
- HDR color support with float32
- Transform system with matrix transposition for THREE.js compatibility

This document specifies the Zarr-based storage format used by Luxar for high-performance 3D and nD points visualization.

## Overview

The Luxar Zarr format is a hierarchical data structure designed for efficient storage and streaming of large-scale points data with support for arbitrary dimensionality, transformations, and rendering attributes.

## Format Structure

```
scene.zarr/
├── .zattrs                  # Scene-level metadata
├── .zgroup                  # Zarr group marker
├── .zmetadata              # Consolidated metadata (optional, created by finalize())
└── <node_name>/            # Scene nodes (groups or points)
    ├── .zattrs             # Node-level metadata (includes spatial index metadata)
    ├── .zgroup             # Zarr group marker
    ├── positions/          # Point positions (required for points, Morton-sorted)
    ├── colors/             # Point colors (optional, same order as positions)
    ├── radii/              # Point radii (optional, same order as positions)
    ├── sharpness/          # Point sharpness (optional, same order as positions)
    ├── chunk_bounds/       # Chunk bounding boxes for spatial queries (optional)
    └── <child_nodes>/      # Nested child nodes (recursive structure)
```

## Scene-Level Metadata (.zattrs)

The root `.zattrs` file contains scene-wide configuration:

```json
{
  "luxar_version": "0.1",
  "type": "scene",
  "units": "um",  // Physical units (nm, um, mm, cm, m, meter, metre, km, inch, foot, px, au)
  "scene_dimensions": {  // Scene-level dimension specification (REQUIRED for nD data)
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
        "spatial": true,            // Whether points extend through this dimension
        "description": "X axis"     // Optional description
      },
      // ... more dimensions
    ]
  }
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
  "blending_mode": "additive"  // normal, additive
}
```

### 2. Points Nodes

Points nodes contain the actual point data.

**Attributes (.zattrs):**
```json
{
  "type": "points",
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  "opacity": 1.0,
  "gamma": 1.0,
  "blending_mode": "additive",
  "n_points": 10000,
  "max_radius": 2.5,
  "broadcast_dims": ["Time", "Channel"]  // Optional: broadcasting configuration
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
- **Dtype:** `float32` (HDR colors)
- **Chunks:** `(min(N, 32768), 3)`
- **Compression:** Blosc with zstd, level 3, bit-shuffle
- **Description:** HDR RGB colors in normalized range
  - **SDR Range:** 0.0-1.0 (standard dynamic range)
  - **HDR Range:** Values > 1.0 represent HDR brightness
  - **Typical HDR:** 0.0-10.0 (extreme brightness)
  - **Note:** Values are NOT in 0-255 range; use 0.0-1.0 for normal colors
- **Default:** White (1.0, 1.0, 1.0) if not provided

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

## Point Spatial Index

The point spatial index enables efficient nD range queries for point visibility determination during slicing operations. Points are reordered during compilation using **Morton or Hilbert space-filling curves** to ensure spatial locality, with **chunk-based bounding boxes** for fast queries.

### Design Philosophy

The spatial index uses a simple but effective approach:
1. **Space-filling curves** (Morton/Hilbert) order points by spatial locality
2. **Compound ordering** handles mixed discrete/spatial dimensions
3. **Chunk bounding boxes** enable fast intersection queries
4. **No grid discretization** - queries use actual data bounds

### Index Structure

The spatial index stores metadata in the points group `.zattrs` and chunk bounds as a separate array:

#### Points Group .zattrs (Spatial Index Metadata)
```json
{
  "type": "points",
  "n_points": 100000,
  "ordering": "hilbert",          // or "morton" - space-filling curve algorithm (default: hilbert)
  "ordering_dims": [0, 1, 2],     // Indices of spatial dimensions (curve-ordered)
  "slice_dims": [3],              // Indices of discrete dimensions (lexicographic)
  "ordering_min": [-100.0, -100.0, -100.0],  // Bounds for curve normalization
  "ordering_max": [100.0, 100.0, 100.0],     // Bounds for curve normalization
  "ordering_bits_per_dim": 21,    // Bits per dimension (max 21 for uint64)
  "chunk_size": 10000,            // Points per chunk
  "max_radius": 2.5               // Maximum point radius in dataset
}
```

#### chunk_bounds/ Array
- **Shape:** `(num_chunks, D, 2)` where D = number of dimensions
- **Dtype:** `float32`
- **Chunks:** `(num_chunks, D, 2)` - stored as single chunk
- **Compression:** Blosc with zstd, level 3
- **Description:** Bounding box [min, max] for each dimension of each chunk
- **Example:** For chunk 5 in a 4D dataset: `chunk_bounds[5, :, :]` = `[[x_min, x_max], [y_min, y_max], [z_min, z_max], [t_min, t_max]]`
- **Note:** Bounds include point radii extent to ensure hyperspheres are found

### Compound Ordering

For datasets with both discrete (time, channel) and spatial (x, y, z) dimensions:

1. **Primary sort**: Lexicographic on discrete dimensions
2. **Secondary sort**: Morton/Hilbert code on spatial dimensions

This ensures:
- All points for time=0 come before time=1
- Within each time slice, points are spatially ordered
- Contiguous chunks contain spatially nearby points

```
Points ordered by: (time, channel) → Hilbert/Morton(x, y, z)

time=0, channel=0: [spatially-ordered xyz points]
time=0, channel=1: [spatially-ordered xyz points]
time=1, channel=0: [spatially-ordered xyz points]
...
```

### Space-Filling Curve Encoding

**Hilbert curves** (default) provide better spatial locality with continuous traversal.
**Morton codes** (Z-order) are simpler, computed by bit-interleaving normalized coordinates:

```python
def morton_encode_nd(coords: np.ndarray, bits_per_dim: int = 16) -> np.ndarray:
    """Encode nD integer coordinates to Morton codes via bit interleaving."""
    n_points, n_dims = coords.shape
    morton = np.zeros(n_points, dtype=np.uint64)

    for bit in range(bits_per_dim):
        for dim in range(n_dims):
            coord_bit = (coords[:, dim] >> bit) & 1
            morton |= coord_bit.astype(np.uint64) << (bit * n_dims + dim)

    return morton
```

### Query Algorithm

To find points visible at slice position with tolerance:

```typescript
function queryChunksForView(
  index: ChunkSpatialIndex,
  slicePosition: number[],
  tolerance: number[]
): number[] {
  const matchingChunks: number[] = [];

  for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
    let intersects = true;

    for (let d = 0; d < ndim; d++) {
      // Chunk bounding box (row-major layout)
      const offset = chunkIdx * ndim * 2 + d * 2;
      const chunkMin = chunkBounds[offset];
      const chunkMax = chunkBounds[offset + 1];

      // Query region
      const queryMin = slicePosition[d] - tolerance[d];
      const queryMax = slicePosition[d] + tolerance[d];

      // Non-intersection test
      if (chunkMax < queryMin || chunkMin > queryMax) {
        intersects = false;
        break;
      }
    }

    if (intersects) matchingChunks.push(chunkIdx);
  }

  return matchingChunks;
}
```

### Chunk to Point Range Conversion

```typescript
// Convert chunk indices to point ranges
function chunkIndicesToRanges(
  chunkIndices: number[],
  chunkSize: number,
  totalPoints: number
): PointRange[] {
  return chunkIndices.map(chunkIdx => ({
    start: chunkIdx * chunkSize,
    end: Math.min((chunkIdx + 1) * chunkSize, totalPoints)
  }));
}

// Merge adjacent ranges for efficient loading
// [0-100], [100-200], [300-400] → [0-200], [300-400]
function mergePointRanges(ranges: PointRange[]): PointRange[] {
  // Sort and merge overlapping/adjacent ranges
  ...
}
```

### Performance Characteristics

- **Query Time:** O(num_chunks × ndim) - linear scan of ~100-1000 chunks
- **Memory:** O(num_chunks × ndim × 2) - just bounding boxes
- **Build Time:** O(N log N) for sorting points by curve
- **Cache Efficiency:** Spatially close points are contiguous in memory

### Benefits of Chunk-Based Spatial Indexing

1. **Simple Implementation**: No complex tree structures, just bounding boxes
2. **Efficient nD Slicing**: Only load chunks intersecting the view hyperplane
3. **Radius-Aware Bounds**: Chunk bounds include point radii for hypersphere queries
4. **Memory Efficient**: Only store bounds, not per-point metadata
5. **Accurate Queries**: Uses actual data bounds, no grid discretization errors

### Backward Compatibility

Point clouds without `chunk_bounds` will load all points. The viewer detects the presence of `chunk_bounds/` and uses spatial queries when available.

### Implementation in LuxarZarrCompiler

When building points with spatial index (`enable_spatial_index=True`):

1. **Identify Dimension Types**: Classify dimensions as discrete vs spatial
2. **Compute Sort Order**: Lexsort on discrete dims, then Morton/Hilbert code
3. **Reorder All Arrays**: Apply same sort order to positions, colors, radii, sharpness
4. **Compute Chunk Bounds**: Calculate bounding boxes including radius extent
5. **Store Metadata**: Write ordering info to group attributes
6. **Store Bounds Array**: Write chunk_bounds array to group

## Transform System

Transforms are stored as 16-element arrays representing 4x4 homogeneous transformation matrices in **column-major order** (THREE.js format):

```
[m00, m10, m20, 0,
 m01, m11, m21, 0,
 m02, m12, m22, 0,
 tx,  ty,  tz,  1]
```

Where:
- `m00-m22`: 3x3 rotation/scale matrix (transposed)
- `tx, ty, tz`: Translation vector (at indices 12, 13, 14)
- Bottom row is always `[0, 0, 0, 1]`

**CRITICAL**: Python transposes matrices from NumPy row-major to THREE.js column-major format before storage. TypeScript consumes them directly using `Matrix4.fromArray()`.

## Dimension System

### Scene Dimensions
Scene-level dimensions define the coordinate system for all objects:
- Maximum 3 dimensions can be displayed simultaneously
- Non-displayed dimensions are used for slicing/navigation
- Dimensions include metadata for units, ranges, and navigation

#### Dimension Types
Dimensions are categorized by two key properties:

1. **Spatial vs Non-Spatial:**
   - **Spatial dimensions**: Points extend through these dimensions as hyperspheres
   - **Non-spatial dimensions**: Points exist at specific values only (always discrete)
   - Displayed dimensions are always spatial by default
   - Non-displayed, non-spatial dimensions must be discrete (enforced automatically)

2. **Continuous vs Discrete:**
   - **Continuous dimensions**: Can take any value in their range
   - **Discrete dimensions**: Represent categorical data or specific values (e.g., channels, time frames)
   - Discrete dimensions use exact matching (tolerance = 0) during queries

#### Dimension Attributes
Each dimension in `scene_dimensions` contains:
- `name`: Dimension identifier
- `unit`: Physical unit
- `range`: [min, max] values
- `display`: Whether shown in 3D view
- `discrete`: Whether dimension represents discrete values
- `spatial`: Whether points extend through this dimension (auto-determined if not specified)
- `step`: Navigation step size

### nD Point Cloud Support
- Points can have arbitrary dimensionality (not limited to 3D)
- Viewer performs slicing for dimensions > 3
- Radius-based visibility for spatial dimensions: points visible if their nD hypersphere intersects the current slice
- Exact matching for discrete dimensions: only points at the exact value are shown

## Chunking Strategy

Optimal chunk sizes balance memory usage and access patterns:
- **Default chunk size:** 32,768 elements
- **Minimum chunk size:** 1,024 elements
- **Maximum chunk size:** 262,144 elements
- **2D arrays (positions, colors):** Chunk along first dimension only
- **1D arrays (radii, sharpness):** Simple 1D chunking

### Chunking with Spatial Index

When using spatial indices:
- **Chunk Alignment**: Zarr chunks are automatically aligned with spatial index chunks
- **Typical Strategy**: `chunk_size` is computed based on target memory per chunk (~32KB)
- **Benefits**: Loading a chunk index range loads exactly that zarr chunk
- **Morton Ordering**: Points within a chunk are spatially nearby due to Morton ordering

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

After scene construction, call `# Context manager handles finalization automatically` to:
- Consolidate all metadata into `.zmetadata` file
- Improve load performance by reducing metadata requests
- Enable efficient streaming from remote stores

## Version History

- **0.1** (Current): Complete format with spatial index, nD support, HDR colors, transforms, scene dimensions

## Best Practices

1. **Use context managers** with LuxarZarrCompiler for automatic finalization
2. **Use appropriate chunk sizes** based on expected access patterns
3. **Store transforms at group level** for hierarchical transformations
4. **Define scene dimensions** for consistent coordinate systems
5. **Use consolidated metadata** for remote data access
6. **Validate all data** before writing to ensure consistency

## Example Creation (Python)

### Basic Example

```python
import numpy as np
from luxar import LuxarZarrCompiler, Dimensions, Dimension

# Create scene with dimensions
dims = Dimensions([
    Dimension("x", unit="um", display=True),
    Dimension("y", unit="um", display=True),
    Dimension("z", unit="um", display=True),
    Dimension("time", unit="ms", display=False, discrete=True)
])

with LuxarZarrCompiler("output.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    
    # Add points
    positions = np.random.randn(10000, 4).astype(np.float32)  # 4D points
    colors = np.random.rand(10000, 3).astype(np.float32)  # SDR colors (0.0-1.0)
    radii = np.ones(10000, dtype=np.float32) * 0.5
    
    scene.add_points("my_points", positions, colors, radii=radii)
    # Context manager handles finalization automatically
```

### Example with Spatial Index

```python
import numpy as np
from luxar import LuxarZarrCompiler, Dimensions, Dimension

# Create scene with dimensions
dims = Dimensions([
    Dimension("x", unit="um", display=True, range=[-100, 100]),
    Dimension("y", unit="um", display=True, range=[-100, 100]),
    Dimension("z", unit="um", display=True, range=[-100, 100]),
    Dimension("time", unit="ms", display=False, range=[0, 10], discrete=True)
])

# Enable spatial index for efficient nD slicing
with LuxarZarrCompiler("output.zarr", enable_spatial_index=True) as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Generate 4D points
    n_points = 100000
    positions = np.random.randn(n_points, 4).astype(np.float32) * 50
    colors = np.random.rand(n_points, 3).astype(np.float32)  # SDR colors (0.0-1.0)
    radii = np.random.uniform(0.1, 2.0, n_points).astype(np.float32)

    # Add points - Morton ordering and chunk bounds computed automatically
    scene.add_points(
        "my_points",
        positions,
        colors,
        radii=radii
    )
    # Points are reordered by (time) → Morton(x,y,z) and chunk_bounds are stored
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
- Hilbert curve option for improved spatial locality (vs Morton)
- Multi-resolution spatial indices for LOD