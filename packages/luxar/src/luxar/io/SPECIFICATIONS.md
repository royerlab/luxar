# luxar.io - Technical Specification

## Purpose

The `io` package implements progressive writing to Zarr stores and spatial indexing for efficient nD point cloud queries. It enables handling of arbitrarily large datasets by writing data immediately without keeping it in memory.

---

## Progressive Writing System

### LuxarZarrCompiler

**Specification**:
- Entry point for creating Luxar scenes
- Implements context manager protocol (__enter__, __exit__)
- Writes data immediately to Zarr (no buffering)
- Returns only metadata (not data)
- Automatically finalizes on context exit

**Key Parameters**:
- `store_path`: Path to Zarr store (None creates temporary directory)
- `compressor`: Zarr compression (default: Blosc with zstd, level 3, bitshuffle)
- `version`: Luxar format version (default: "0.1")
- `enable_spatial_index`: Whether to build spatial indices (default: True)
- `dtype_config`: Data type configuration (default: auto-detect)

**Operations**:
1. `create_scene(dimensions=None)` - Create scene with optional dimension specs
2. `write_group(path, **attrs)` - Create group structure
3. `write_points(path, positions, ...)` - Write point data with attributes
4. `create_resizable_dataset(...)` - For streaming writes
5. `finalize()` - Consolidate metadata, close store

**Finalization**:
- Close store to flush all data
- Re-open to consolidate metadata
- Create `.zmetadata` file for efficient loading
- Mark as finalized (idempotent - safe to call multiple times)

---

## Point Data Writing Algorithm

### write_points() Specification

**Input**:
- `positions`: (N, D) float32 array
- `colors`: Optional (N, 3) float32 array (HDR) or uint8 (SDR)
- `radii`: Optional (N,) float32 array
- `sharpness`: Optional (N,) float32 array
- `grid_shape`: Optional grid resolution for spatial index
- `**attrs`: Additional attributes (transform, opacity, etc.)

**Output**:
- Metadata dictionary: {n_points, dims, path, has_colors, has_radii, has_sharpness, max_radius}

**Algorithm**:
1. **Validate** positions (must be 2D array, N points, D dimensions)
2. **Build spatial index** (if enabled and has non-displayed dimensions)
   - Extract displayed vs non-displayed dimensions from scene metadata
   - Calculate grid shape (auto or user-provided)
   - Partition non-displayed dimensions into spatial grid cells
   - Sort points by cell ID for spatial locality
   - Return sorted data + sort order
3. **Reorder all arrays** according to spatial index sort order
4. **Write positions dataset** with optimal dtype and chunking
5. **Write optional datasets** (colors, radii, sharpness) with validation
6. **Process transform** (if present) using centralized conversion
7. **Set default attributes** (opacity=1.0, gamma=1.0, blending_mode="additive")
8. **Write spatial index metadata** (if built)
9. **Return metadata** (no data retained in memory)

**Key Invariant**: All arrays (positions, colors, radii, sharpness) must stay synchronized during reordering

---

## Spatial Index Specification

### Purpose
Enable efficient range queries on nD points by indexing non-displayed dimensions.

### Design Decisions
- **Only index non-displayed dimensions** (displayed dims loaded fully)
- **Use regular grid partitioning** (simple, predictable)
- **Store sparse index** (only occupied cells)
- **Reorder points** for spatial locality (improves compression and I/O)

### Data Structures

**Grid Parameters**:
- `grid_shape`: Number of cells per dimension (uint32 array)
- `grid_origin`: Minimum coordinate per dimension (float32 array)
- `cell_size`: Size of each cell per dimension (float32 array)

**Index Data**:
- `occupied_cells`: (M, D) uint32 array - coordinates of non-empty cells
- `cell_ranges`: (M, 2) uint64 array - [start, end) indices for each cell
- `sort_order`: (N,) int64 array - mapping original → sorted positions

**Metadata**:
- `indexed_dimensions`: List of dimension indices that are indexed
- `displayed_dimensions`: List of dimension indices that are displayed
- `full_dimensions`: Total number of dimensions
- `total_points`: Total number of points

### Cell ID Linearization

**Formula** (row-major order):
```
cell_id = i₀ * stride₀ + i₁ * stride₁ + ... + iₙ₋₁ * strideₙ₋₁

where:
  stride_d = product of all grid dimensions after d
  stride_{n-1} = 1 (innermost dimension)
```

**Example** (grid shape [2, 3, 4]):
- Cell (0,0,0) → ID 0
- Cell (0,0,1) → ID 1
- Cell (0,1,0) → ID 4
- Cell (1,0,0) → ID 12

**Decoding** (reverse operation):
```
For each dimension from last to first:
    coord_d = cell_id % grid_shape_d
    cell_id = cell_id // grid_shape_d
Reverse coordinate list to get original order
```

### Grid Shape Determination

**For Discrete Dimensions** (time, channel, category):
- Use one cell per unique value
- Enables exact lookups (e.g., "all points at time=5")
- Cap at 10,000 cells (safety limit)

**For Continuous Dimensions**:
- Adaptive sizing based on data density
- Single indexed dimension: 10-100 cells (sqrt based on point count)
- Multiple indexed dimensions: Fewer cells per dimension (keep total manageable)
- Goal: ~100-1000 points per cell
- Heuristic: `min(10, max(3, sqrt(n_points / 10000)))` cells

**Spatial Dimension Cell Sizing**:
- For dimensions where points extend spatially (spatial=True)
- Cell size must be at least 2 * max_radius
- Ensures capturing all potentially visible points during queries

### Query Algorithm

**Input**:
- `slice_pos`: Current position in non-displayed dimensions
- `tolerance`: Radius/tolerance per dimension

**Algorithm**:
1. Convert slice position to grid coordinates: `grid_pos = (slice_pos - origin) / cell_size`
2. Calculate grid range: `min_grid = floor(grid_pos - tolerance/cell_size)`, `max_grid = ceil(grid_pos + tolerance/cell_size)`
3. Clamp to valid grid bounds
4. Find occupied cells within range
5. Return point ranges from matching cells

**Output**: List of (start, end) index pairs for points to load

---

## Intelligent Chunking

**Purpose**: Align Zarr chunks with spatial index for efficient I/O

**Specification**:

**Default Chunking** (no spatial index):
- 1D arrays: chunk size = min(length, 32768)
- 2D arrays (positions): chunk along first dimension, keep second dimension intact
  - Chunk size: min(n_points, 32768 / n_dims)
- Higher D: reasonable defaults per dimension

**Spatial Index-Aligned Chunking**:
- Calculate average points per cell
- Aim for chunks containing 4-8 cells worth of points
- Balance between I/O efficiency and spatial locality
- Clamp to range [1024, 32768] elements

---

## Data Type Configuration

### Auto-Detection Strategy

**Positions**:
- Default: float32 (accuracy critical)
- Could use float16 for small-scale data (not implemented)

**Colors**:
- If any value > 1.0 → float32 (HDR)
- If all values in [0, 1] → uint8 (memory efficient)
- If normalization needed → uint8 with 0-255 range

**Radii/Sharpness**:
- If all values in [0, 1] → uint8 (normalized)
- If values in [0, 65536] → float16
- Otherwise → float32

### Conversion Rules

**Float → uint8** (with normalization):
```
normalized = (value - input_min) / (input_max - input_min)
uint8_value = clip(normalized * 255, 0, 255)
```

**uint8 → Float** (with denormalization):
```
normalized = uint8_value / 255.0
float_value = normalized * (output_max - output_min) + output_min
```

---

## Streaming Points

### Purpose
Append point data in batches without loading existing data into memory.

### Specification

**Initialization**:
- Create resizable datasets with initial shape (0, D)
- Set maxshape to (None, D) for unlimited growth
- No data written initially

**Batch Appending**:
1. Validate batch (must match expected_dims)
2. Resize datasets to accommodate new points
3. Write batch to slice [current_position:new_position]
4. Update position counter
5. Create attribute datasets lazily (when first batch with that attribute arrives)

**Finalization**:
- Write group metadata
- Set attributes (type="points", n_points=total, etc.)
- Return metadata

**Key Invariant**: All datasets must be resized together to stay synchronized

---

## Zarr Store Structure

**Root Attributes** (minimum required):
- `type`: "scene"
- `luxar_version`: "0.1"
- `scene_dimensions`: Optional dimension specifications

**Points Group**:
- `positions`: (N, D) array
- `colors`: (N, 3) array (optional)
- `radii`: (N,) array (optional)
- `sharpness`: (N,) array (optional)
- `spatial_index/`: Group with index data (optional)

**Points Attributes**:
- `type`: "points"
- `n_points`: Number of points
- `opacity`, `gamma`, `blending_mode`: Rendering attributes
- `transform`: Optional 16-element list (column-major)
- `*_dtype`: Dtype metadata for each array
- `max_radius`: Maximum radius (if radii present)
- `spatial_extend_dims`: Boolean list indicating spatial extension

**Spatial Index Group**:
- Attributes: grid_shape, grid_origin, cell_size, num_occupied, total_cells, etc.
- `occupied_cells`: (M, D_indexed) uint32 array
- `cell_ranges`: (M, 2) uint64 array

---

## Compression Strategy

**Default Compressor**: Blosc
- Algorithm: zstd
- Level: 3 (fast with good compression)
- Shuffle: BITSHUFFLE (better for floating point data)

**Rationale**:
- zstd: Good balance of speed and compression ratio
- Level 3: Fast decompression for interactive viewing
- Bitshuffle: Exploits bit-level patterns in scientific data

---

## Constants and Tuning Parameters

**Spatial Index Grid Sizing**:
- `SPATIAL_INDEX_MAX_CELLS_DISCRETE = 10000` - Safety cap for discrete dimensions
- `SPATIAL_INDEX_MAX_CELLS_CONTINUOUS = 10` - Max cells for continuous dimensions
- `SPATIAL_INDEX_TARGET_POINTS_PER_CELL = 10000` - Target density
- `SPATIAL_INDEX_CHUNK_SIZE = 4096` - Chunk size for index datasets

All constants defined in `typing_utils/constants.py` for easy tuning.

---

## Error Handling

**Write-Time Validation**:
- Positions: Must be 2D array, at least 1 point, at least 1 dimension
- Colors: Must be (N, 3), no negative values, HDR warning > 10.0
- Radii: Must be (N,), all positive, no zeros
- Sharpness: Must be (N,), all positive, range [0, 15]

**Helpful Error Messages**:
- Include actual vs expected values
- Provide suggestions for fixing
- Use custom ValidationError class with suggestion field

---

## This specification provides sufficient detail to re-implement the I/O system while maintaining compatibility with existing Zarr stores.
