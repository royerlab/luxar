# Luxar Zarr Format Specification

## Version: 0.3

**Major Changes in v0.3:**
- Added spatial index for efficient nD point queries
- Points are reordered during compilation for spatial locality
- Grid-based sparse index structure for all dimensions

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
    ├── spatial_index/      # Spatial index for efficient nD queries (v0.3+)
    │   ├── .zattrs         # Index metadata
    │   ├── occupied_cells/ # Sparse list of occupied grid cells
    │   └── cell_ranges/    # Point ranges for each occupied cell
    └── <child_nodes>/      # Nested child nodes (recursive structure)
```

## Scene-Level Metadata (.zattrs)

The root `.zattrs` file contains scene-wide configuration:

```json
{
  "luxar_version": "0.2",
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
  "blending_mode": "additive"  // normal, additive
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

## Spatial Index (v0.3+)

The spatial index enables efficient nD range queries for point visibility determination during slicing operations. Points are reordered during compilation to ensure spatial locality aligns with the index structure.

### Index Structure

The spatial index uses a regular grid partitioning of the nD space, stored as a sparse representation containing only occupied cells.

#### spatial_index/.zattrs
```json
{
  "grid_shape": [10, 10, 10, 5],       // Number of cells per dimension
  "grid_origin": [-100, -100, -100, 0], // Minimum coordinate per dimension
  "cell_size": [20, 20, 20, 2],        // Size of each cell per dimension
  "num_occupied": 234,                   // Number of occupied cells
  "total_cells": 5000,                  // Total possible cells (product of grid_shape)
  "dimensions": 4,                      // Number of dimensions indexed
  "build_version": "0.3",               // Version of index builder
  "max_points_per_cell": 1024          // Maximum points in any single cell
}
```

#### spatial_index/occupied_cells/
- **Shape:** `(num_occupied, D)` where D = number of dimensions
- **Dtype:** `uint32`
- **Chunks:** `(min(num_occupied, 4096), D)`
- **Compression:** Blosc with zstd, level 3
- **Description:** nD coordinates of occupied grid cells
- **Example:** `[2, 3, 1, 0]` means cell at grid position (2,3,1,0)

#### spatial_index/cell_ranges/
- **Shape:** `(num_occupied, 2)`
- **Dtype:** `uint64`
- **Chunks:** `(min(num_occupied, 4096), 2)`
- **Compression:** Blosc with zstd, level 3
- **Description:** Start and end indices (exclusive) for points in each occupied cell
- **Example:** `[1000, 1250]` means points 1000-1249 belong to this cell
- **Invariant:** Ranges are non-overlapping and sorted

### Index Properties

1. **Sparse Representation:** Only cells containing points are stored
2. **Sorted Points:** Points are reordered so each cell's points are contiguous
3. **Grid Resolution:** Typically 10-20 cells per dimension for balanced query performance
4. **Cell Size:** Chosen to contain ~100-1000 points per cell on average

### Query Algorithm

To find all points potentially visible at slice position `[s0, s1, ..., sD]` with tolerance `[t0, t1, ..., tD]`:

```python
def query_spatial_index(index, slice_pos, tolerance):
    # Calculate grid range to check
    min_grid = floor((slice_pos - tolerance - grid_origin) / cell_size)
    max_grid = ceil((slice_pos + tolerance - grid_origin) / cell_size)
    
    # Clamp to valid grid bounds
    min_grid = max(0, min_grid)
    max_grid = min(grid_shape - 1, max_grid)
    
    # Find matching occupied cells
    point_ranges = []
    for i, cell_coords in enumerate(occupied_cells):
        if all(min_grid[d] <= cell_coords[d] <= max_grid[d] for d in range(D)):
            start, end = cell_ranges[i]
            point_ranges.append((start, end))
    
    return point_ranges
```

### Building the Index

During compilation, points are reordered using a space-filling curve or grid-based sorting:

```python
import numpy as np

def decode_cell_id(cell_id, grid_shape):
    """Convert linear cell ID back to nD grid coordinates."""
    coords = []
    for dim_size in reversed(grid_shape):
        coords.append(cell_id % dim_size)
        cell_id //= dim_size
    return list(reversed(coords))

def build_spatial_index(positions, grid_shape):
    """Build spatial index for nD points."""
    D = positions.shape[1]
    N = positions.shape[0]
    
    # Calculate grid bounds
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    grid_origin = min_coords
    cell_size = (max_coords - min_coords) / grid_shape
    
    # Assign points to grid cells
    grid_indices = np.floor((positions - grid_origin) / cell_size).astype(np.uint32)
    grid_indices = np.clip(grid_indices, 0, grid_shape - 1)
    
    # Convert nD grid indices to linear cell IDs
    cell_ids = np.zeros(N, dtype=np.uint64)
    stride = 1
    for d in range(D-1, -1, -1):
        cell_ids += grid_indices[:, d] * stride
        stride *= grid_shape[d]
    
    # Sort points by cell ID for spatial locality
    sort_order = np.argsort(cell_ids)
    sorted_positions = positions[sort_order]
    sorted_cell_ids = cell_ids[sort_order]
    
    # Build sparse index
    occupied_cells = []
    cell_ranges = []
    
    current_cell = sorted_cell_ids[0]
    start_idx = 0
    
    for i in range(1, N):
        if sorted_cell_ids[i] != current_cell:
            # Record the completed cell
            grid_coords = decode_cell_id(current_cell, grid_shape)
            occupied_cells.append(grid_coords)
            cell_ranges.append([start_idx, i])
            
            # Start new cell
            current_cell = sorted_cell_ids[i]
            start_idx = i
    
    # Record final cell
    grid_coords = decode_cell_id(current_cell, grid_shape)
    occupied_cells.append(grid_coords)
    cell_ranges.append([start_idx, N])
    
    return {
        'occupied_cells': np.array(occupied_cells, dtype=np.uint32),
        'cell_ranges': np.array(cell_ranges, dtype=np.uint64),
        'sorted_positions': sorted_positions,
        'sort_order': sort_order,
        'grid_origin': grid_origin,
        'cell_size': cell_size
    }
```

### Integration with Viewer

The TypeScript viewer uses the spatial index for efficient lazy loading:

```typescript
interface SpatialIndex {
  gridShape: number[];
  gridOrigin: number[];
  cellSize: number[];
  occupiedCells: Uint32Array;  // Shape: [numOccupied, D]
  cellRanges: BigUint64Array;   // Shape: [numOccupied, 2]
}

function queryVisiblePoints(
  index: SpatialIndex,
  slicePos: number[],
  tolerance: number[]
): Array<[number, number]> {
  const ranges: Array<[number, number]> = [];
  const D = index.gridShape.length;
  
  // Calculate grid bounds to query
  const minGrid = new Uint32Array(D);
  const maxGrid = new Uint32Array(D);
  
  for (let d = 0; d < D; d++) {
    minGrid[d] = Math.max(0, 
      Math.floor((slicePos[d] - tolerance[d] - index.gridOrigin[d]) / index.cellSize[d])
    );
    maxGrid[d] = Math.min(index.gridShape[d] - 1,
      Math.ceil((slicePos[d] + tolerance[d] - index.gridOrigin[d]) / index.cellSize[d])
    );
  }
  
  // Check each occupied cell
  const numOccupied = index.cellRanges.length / 2;
  for (let i = 0; i < numOccupied; i++) {
    let inRange = true;
    for (let d = 0; d < D; d++) {
      const cellCoord = index.occupiedCells[i * D + d];
      if (cellCoord < minGrid[d] || cellCoord > maxGrid[d]) {
        inRange = false;
        break;
      }
    }
    
    if (inRange) {
      const start = Number(index.cellRanges[i * 2]);
      const end = Number(index.cellRanges[i * 2 + 1]);
      ranges.push([start, end]);
    }
  }
  
  return ranges;
}
```

### Performance Characteristics

- **Query Time:** O(num_occupied_cells) - typically much smaller than O(N)
- **Memory:** O(num_occupied_cells * (D + 2)) - sparse representation
- **Build Time:** O(N log N) for sorting points
- **Cache Efficiency:** Points in same cell are contiguous in memory

### Benefits of Spatial Indexing

1. **Efficient nD Slicing**: Only load points near the current slice plane
2. **Reduced Memory Usage**: Load only relevant chunks instead of entire dataset
3. **Better Cache Utilization**: Spatially close points are stored contiguously
4. **Scalability**: Enables visualization of datasets with millions of points
5. **Progressive Loading**: Can load visible regions first for faster interaction

### Backward Compatibility

Point clouds without spatial indices will continue to work but with slower nD queries. The viewer detects the presence of `spatial_index/` and uses it when available, falling back to linear scanning otherwise.

### Implementation in LuxarZarrCompiler

When building a point cloud with spatial index:

1. **Compute Grid Parameters**: Based on data bounds and desired resolution
2. **Assign Points to Cells**: Map each point to its grid cell
3. **Sort Points**: Reorder points by cell ID for spatial locality
4. **Build Sparse Index**: Record occupied cells and their point ranges
5. **Store in Zarr**: Write index arrays and metadata
6. **Reorder Other Arrays**: Apply same sort order to colors, radii, sharpness

```python
def add_points_with_spatial_index(group, positions, colors=None, radii=None, 
                                  sharpness=None, grid_shape=None):
    """Add point cloud with spatial index to zarr group."""
    import zarr
    
    # Build spatial index
    if grid_shape is None:
        # Auto-determine grid shape based on data
        grid_shape = np.array([10] * positions.shape[1], dtype=np.uint32)
    
    index_data = build_spatial_index(positions, grid_shape)
    
    # Reorder all arrays according to spatial index
    sorted_positions = index_data['sorted_positions']
    sort_order = index_data['sort_order']
    
    if colors is not None:
        sorted_colors = colors[sort_order]
    if radii is not None:
        sorted_radii = radii[sort_order]
    if sharpness is not None:
        sorted_sharpness = sharpness[sort_order]
    
    # Store sorted data
    group.create_dataset('positions', data=sorted_positions, 
                        chunks=(min(len(sorted_positions), 32768), positions.shape[1]),
                        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2))
    
    if colors is not None:
        group.create_dataset('colors', data=sorted_colors,
                           chunks=(min(len(sorted_colors), 32768), 3),
                           compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2))
    
    if radii is not None:
        group.create_dataset('radii', data=sorted_radii,
                           chunks=(min(len(sorted_radii), 32768),),
                           compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2))
    
    # Store spatial index
    index_group = group.create_group('spatial_index')
    
    # Store index metadata
    index_group.attrs.update({
        'grid_shape': grid_shape.tolist(),
        'grid_origin': index_data['grid_origin'].tolist(),
        'cell_size': index_data['cell_size'].tolist(),
        'num_occupied': len(index_data['occupied_cells']),
        'total_cells': int(np.prod(grid_shape)),
        'dimensions': positions.shape[1],
        'build_version': '0.3',
        'max_points_per_cell': int(np.max(np.diff(index_data['cell_ranges'], axis=1)))
    })
    
    # Store index arrays
    index_group.create_dataset('occupied_cells', 
                              data=index_data['occupied_cells'],
                              chunks=(min(len(index_data['occupied_cells']), 4096), 
                                     positions.shape[1]),
                              compressor=zarr.Blosc(cname='zstd', clevel=3))
    
    index_group.create_dataset('cell_ranges',
                              data=index_data['cell_ranges'],
                              chunks=(min(len(index_data['cell_ranges']), 4096), 2),
                              compressor=zarr.Blosc(cname='zstd', clevel=3))
    
    return group
```

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

### Chunking with Spatial Index (v0.3+)

When using spatial indices:
- **Chunk Alignment**: Zarr chunks should align with spatial index cells when possible
- **Typical Strategy**: Each zarr chunk contains points from 1-4 spatial cells
- **Benefits**: Minimizes chunks loaded for range queries
- **Trade-off**: Balance between chunk size (I/O efficiency) and spatial locality

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

- **0.3** (Current): Added spatial index for efficient nD queries, point reordering for spatial locality
- **0.2**: Added scene dimensions, improved nD support, HDR color support (float32), sharpness attribute, rendering parameters
- **0.1**: Initial format with positions, colors, radii

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
    
    # Add point cloud
    positions = np.random.randn(10000, 4).astype(np.float32)  # 4D points
    colors = np.random.rand(10000, 3).astype(np.float32)  # SDR colors (0.0-1.0)
    radii = np.ones(10000, dtype=np.float32) * 0.5
    
    scene.add_points("my_points", positions, colors, radii=radii)
    # Context manager handles finalization automatically
```

### Example with Spatial Index (v0.3+)

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

with LuxarZarrCompiler("output.zarr", enable_spatial_index=True) as compiler:
    scene = compiler.create_scene(dimensions=dims)
    
    # Generate 4D points
    n_points = 100000
    positions = np.random.randn(n_points, 4).astype(np.float32) * 50
    colors = np.random.rand(n_points, 3).astype(np.float32)  # SDR colors (0.0-1.0)
    radii = np.random.uniform(0.1, 2.0, n_points).astype(np.float32)
    
    # Add points with spatial index (points will be automatically reordered)
    scene.add_points(
        "my_points", 
        positions, 
        colors, 
        radii=radii,
        grid_shape=[10, 10, 10, 5]  # Grid resolution for spatial index
    )
    # Spatial index is built automatically during finalization
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
- Adaptive spatial index resolution based on point density
- Multi-resolution spatial indices for LOD
- Integration with octree/kd-tree structures