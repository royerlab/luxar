# luxar.io - Technical Specification

**Version**: 1.4.0
**Last Updated**: 2025-11-29

## Purpose

The `io` package implements progressive writing to Zarr stores and spatial indexing for efficient nD point cloud queries. It enables handling of arbitrarily large datasets by writing data immediately without keeping it in memory.

**Related Specifications**:
- `luxar.core` - Data structures and scene graph (see `core/SPECIFICATIONS.md`)
- `luxar.encoding` - Array encoding and semantic types (see `encoding/SPECIFICATIONS.md`)
- `luxar.validation` - Input validation (see `validation/SPECIFICATIONS.md`)
- `luxar.typing_utils` - Type definitions and constants (see `typing_utils/SPECIFICATIONS.md`)

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
- `encoding_mode`: Encoding mode for array storage (default: AUTO) - see encoding/SPECIFICATIONS.md

**Operations**:
1. `create_scene(dimensions=None)` - Create scene with optional dimension specs
2. `write_group(path, **attrs)` - Create group structure
3. `write_points(path, positions, radii, ...)` - Write point data with attributes
4. `create_resizable_dataset(...)` - Part of protocol (rarely used directly)
5. `finalize()` - Consolidate metadata, close store

**Finalization**:
- Close store to flush all data
- Re-open to consolidate metadata
- Create `.zmetadata` file for efficient loading
- Mark as finalized (idempotent - safe to call multiple times)

---

## write_points() Specification

**Input**:
- `positions`: (N, D) float32 array (required - **NOT** broadcastable, see encoding/SPECIFICATIONS.md)
- `radii`: (N,) float32 array, (1,) array, or scalar float (required)
- `colors`: Optional - (N, 3) array, (1, 3) array, scalar tuple/list, or None
- `color_mode`: Required for float32 colors: `"sdr"` or `"hdr"`
- `sharpness`: Optional - (N,) array, (1,) array, scalar float, or None
- `scene_dimensions`: Optional dimension specifications for compound ordering
- `**attrs`: Additional attributes (transform, opacity, extend_to_all, etc.)

**Scalar Convenience** (v1.4.0):
For uniform attributes, callers can provide scalars directly instead of arrays:
- `radii=0.5` instead of `np.full(N, 0.5)`
- `colors=(1.0, 0.0, 0.0)` instead of `np.full((N, 3), [1.0, 0.0, 0.0])`
- `sharpness=2.0` instead of `np.full(N, 2.0)`

The compiler passes scalars directly to the encoder with `n_elements=N`, eliminating intermediate array allocation. See `encoding/SPECIFICATIONS.md` v0.6.0 for complete scalar passthrough specification.

**Output**:
- Metadata dictionary: {n_points, ndim, path, has_colors, has_sharpness, max_radius}

**Validation**:
1. Positions must be 2D array with N ≥ 1 points, D ≥ 1 dimensions (arrays only, no broadcasting)
2. Radii must be positive (> 0) - validated whether scalar or array
3. Float32 colors require explicit `color_mode` ("sdr" or "hdr")
4. Sharpness must be in [0, 31] - validated whether scalar or array

**Algorithm**: See [Write Algorithm](#write-algorithm) in Spatial Index section for the full compound ordering algorithm.

**Key Invariant**: All arrays (positions, colors, radii, sharpness) must stay synchronized during sorting. Scalars are passed directly to encoder and don't participate in sorting.

---

## Spatial Index Specification

### Purpose
Enable efficient lazy loading of nD point and splat data by:
1. Sorting data by spatial locality (Morton order)
2. Providing lightweight chunk-level bounding boxes for visibility queries

**Supported Node Types**: Points, GSplats
**Not Supported**: Lines (spatial indexing for lines requires a different approach - TBD)

### Design Principles
- **Compound ordering**: Discrete dimensions first, then Morton within each slice
- **Chunk-aligned**: Loading unit = Zarr chunk = query unit (no indirection)
- **Radius-aware bounds**: Chunk bounding boxes include element extent (radius/size)
- **Dimension-aware**: Optimized for both discrete slicing AND spatial queries

---

### Compound Ordering for nD Data

**Problem**: Pure Morton ordering over ALL dimensions scatters discrete slice data:
- Points at time=5 would be distributed throughout the Morton-sorted array
- A time-slice query might touch every chunk
- Poor locality for discrete dimension navigation

**Solution**: Compound ordering - sort by discrete dimensions FIRST, then Morton within each slice.

**Ordering Hierarchy**:
```
Primary:   Discrete dimensions (time, channel, etc.) - lexicographic order
Secondary: Morton code of spatial/continuous dimensions only
```

**Result**:
```
[time=0, channel=0]: Morton-sorted by (x,y,z)
[time=0, channel=1]: Morton-sorted by (x,y,z)
[time=0, channel=2]: Morton-sorted by (x,y,z)
[time=1, channel=0]: Morton-sorted by (x,y,z)
...
```

**Benefits**:
- Discrete slice queries load contiguous chunks
- Spatial locality preserved within each slice
- Chunk boundaries can align with discrete slice boundaries

**Terminology**:
- `slice_dims`: Dimensions for exact-match slicing (discrete, non-displayed) - sorted first
- `ordering_dims`: Dimensions for spatial locality (displayed OR non-discrete) - curve-sorted within each slice

Note: A non-displayed spatial dimension (`display=False, spatial=True, discrete=False`) goes in `ordering_dims` because it benefits from spatial locality, even though it's not displayed.

**Algorithm**:
```python
# 1. Identify dimension types from scene_dimensions
slice_dims = [d for d in dims if d.discrete and not d.display]
ordering_dims = [d for d in dims if not d.discrete or d.display]

# 2. Create compound sort key
def compound_sort_key(point):
    # Primary: tuple of slice dimension values (lexicographic)
    slice_key = tuple(point[d] for d in slice_dims)

    # Secondary: Space-filling curve code of ordering dimensions only
    ordering_coords = [point[d] for d in ordering_dims]
    curve_key = compute_hilbert_code(ordering_coords)  # or Morton

    return (slice_key, curve_key)

# 3. Sort all arrays by compound key
sort_indices = argsort(points, key=compound_sort_key)
sorted_positions = positions[sort_indices]
sorted_colors = colors[sort_indices]  # etc.
```

**Metadata stored**:
- `slice_dims`: List of slice dimension indices (for exact-match queries)
- `ordering_dims`: List of ordering dimension indices (for spatial locality)
- `ordering_min`, `ordering_max`: Bounds for ordering dimensions only
- `ordering_bits_per_dim`: Bits per ordering dimension

**Fallback**: If no discrete dimensions exist, pure Hilbert ordering is used (equivalent to empty discrete key).

---

### Space-Filling Curves: Morton and Hilbert

The system supports two space-filling curve options for spatial ordering:

#### Morton Code (Z-Order Curve)

**Concept**: Map nD coordinates to a 1D integer by interleaving bit representations. Points nearby in nD space have similar Morton codes.

**Pros**:
- Simple bit-interleaving implementation
- Fast to compute
- Trivial nD extension

**Cons**:
- Occasional "jumps" in locality at quadrant boundaries

**Algorithm**:
```
Input: coordinates c[0..n-1], each normalized to [0, 2^B - 1]
Output: 64-bit Morton code

morton = 0
for bit in 0..(B-1):
    for dim in 0..(n-1):
        morton |= ((c[dim] >> bit) & 1) << (bit * n + dim)
```

**Normalization** (required before Morton coding):
```
For each dimension d:
    if max[d] == min[d]:
        normalized[d] = 0  # All coordinates identical in this dimension
    else:
        normalized[d] = floor((coord[d] - min[d]) / (max[d] - min[d]) * (2^B - 1))
```

Where `min[d]` and `max[d]` are the coordinate bounds for dimension d.

**Edge Case**: When all coordinates in a dimension are identical (`max[d] == min[d]`), set normalized value to 0 to avoid division by zero. This is valid because the dimension provides no spatial discrimination.

**Precision**: 64-bit Morton codes
- **Important**: With compound ordering, Morton codes only cover `ordering_dims` (not all dimensions)
- Bits per dimension: `B = floor(64 / len(ordering_dims))` (integer division)
- Unused bits: `64 - (B * len(ordering_dims))` remain zero (left-padded)

Examples (for `len(ordering_dims)`):
- 3 dims: 21 bits each (~2M levels), 1 unused bit
- 4 dims: 16 bits each (~65K levels), 0 unused bits
- 5 dims: 12 bits each (~4K levels), 4 unused bits
- 6 dims: 10 bits each (~1K levels), 4 unused bits

#### Hilbert Curve

**Concept**: Alternative space-filling curve that better preserves locality compared to Morton ordering.

**Pros**:
- Better locality preservation (never jumps far)
- ~10% better compression than Morton in practice
- Smoother traversal of space

**Cons**:
- More complex algorithm
- Requires external library (`hilbertcurve` package)

**Implementation**: Uses the `hilbertcurve` package which implements the Skilling (2004) algorithm for nD Hilbert curves.

**Libraries**:
- [`hilbertcurve`](https://pypi.org/project/hilbertcurve/) - Python package for nD Hilbert curves

**Note**: Hilbert and Morton use the same metadata format (ordering_min, ordering_max, ordering_bits_per_dim) for consistency. The ordering method is distinguished by the `ordering` attribute ("morton" or "hilbert").

#### Ordering Method Selection

Both Morton and Hilbert ordering are available via the `luxar.io.ordering` module:

```python
# For Points (with compound ordering):
sort_indices, metadata = sort_points_compound(
    positions,
    dimensions,
    method="hilbert",  # or "morton" (default: "morton")
)

# For GSplats (simple spatial ordering):
sort_indices, metadata = sort_splats_spatial(
    centers,
    method="hilbert",  # or "morton" (default: "hilbert")
)
```

**Default choices**:
- **Points**: Morton (simpler, well-tested with compound ordering)
- **GSplats**: Hilbert (better compression for pure spatial data)

**Concrete Example: 5D Dataset (X, Y, Z, Time, Channel)**

Consider a 5D point cloud with dimensions:
```python
Dimensions:
- X (dim 0): displayed, spatial
- Y (dim 1): displayed, spatial
- Z (dim 2): displayed, spatial
- Time (dim 3): non-displayed, discrete
- Channel (dim 4): non-displayed, discrete
```

**Compound Ordering Result**:
```python
# Dimension categorization
slice_dims = [3, 4]      # Time, Channel (discrete, non-displayed)
ordering_dims = [0, 1, 2]  # X, Y, Z (displayed/spatial)

# Morton code calculation
ordering_bits_per_dim = floor(64 / 3) = 21 bits per dimension
ordering_min = [x_min, y_min, z_min]  # Only for X, Y, Z
ordering_max = [x_max, y_max, z_max]  # Only for X, Y, Z

# Positions array shape
positions.shape = (N, 5)  # Stores ALL dimensions

# Sort key for each point
sort_key = (
    (time_value, channel_value),  # Primary: discrete dimensions (lexicographic)
    morton_code(x, y, z)          # Secondary: Morton code of X, Y, Z only
)

# Result: Points grouped by (time, channel), Morton-sorted within each group
# Example ordering:
#   [time=0, channel=0]: Morton-sorted by (x,y,z)
#   [time=0, channel=1]: Morton-sorted by (x,y,z)
#   [time=0, channel=2]: Morton-sorted by (x,y,z)
#   [time=1, channel=0]: Morton-sorted by (x,y,z)
#   ...
```

**Key Insight**: Morton codes only use **spatial/continuous dimensions**. Discrete dimensions are sorted separately (lexicographically) for slice-aligned chunking.

**Metadata stored**:
- `ordering_min`: (len(ordering_dims),) float32 - minimum coordinate per morton dimension
- `ordering_max`: (len(ordering_dims),) float32 - maximum coordinate per morton dimension
- `ordering_bits_per_dim`: int - bits allocated per morton dimension

---

### Chunk Bounding Boxes

**Purpose**: Allow client to determine which chunks intersect a view/slice without downloading point data.

**Schema**: `/chunk_bounds` array
- **Shape**: `(num_chunks, n_dims, 2)`
- **Dtype**: float32
- **Semantics**:
  - `chunk_bounds[i, d, 0]` = minimum bound for chunk i in dimension d
  - `chunk_bounds[i, d, 1]` = maximum bound for chunk i in dimension d

**Calculation** (during write):
```
For each chunk i:
    For each dimension d:
        chunk_bounds[i, d, 0] = min({position[p, d] - radius[p] for p in chunk_i})
        chunk_bounds[i, d, 1] = max({position[p, d] + radius[p] for p in chunk_i})
```

**Critical**: Bounds MUST include element extent (radius for points, ellipsoid extent for splats). This ensures no elements are missed when their center is outside the query region but their extent intersects it.

**For GSplats**: Use the maximum extent of the ellipsoid in each dimension:
```
# Covariance from Cholesky: Σ = L @ L.T
# Diagonal element: covariance[d, d] = sum(L[d, i]^2 for i in 0..d)
#
# For packed Cholesky factors (lower triangular, row-major):
# 2D (k=3): L = [[L[0], 0], [L[1], L[2]]]
#   covariance[0,0] = L[0]^2
#   covariance[1,1] = L[1]^2 + L[2]^2
#
# 3D (k=6): L = [[L[0], 0, 0], [L[1], L[2], 0], [L[3], L[4], L[5]]]
#   covariance[0,0] = L[0]^2
#   covariance[1,1] = L[1]^2 + L[2]^2
#   covariance[2,2] = L[3]^2 + L[4]^2 + L[5]^2
#
# General formula for diagonal index d:
#   start_idx = d * (d + 1) / 2
#   covariance[d,d] = sum(L[start_idx + i]^2 for i in 0..d)

extent[d] = sqrt(covariance[d, d]) * k  # k = 3.0 for 99.7% coverage
```

---

### Chunk Sizing Strategy

**Goal**: Balance between HTTP overhead (too many small chunks) and wasted data (too few large chunks).

**Constants** (defined in `typing_utils/SPECIFICATIONS.md` - single source of truth):
- `TARGET_CHUNK_BYTES = 65536` (64KB) - target chunk size
- `MIN_CHUNK_BYTES = 16384` (16KB) - minimum to amortize HTTP overhead
- `MAX_CHUNK_BYTES = 262144` (256KB) - maximum for responsive loading

**IMPORTANT**: Zarr chunks by **elements**, not bytes. This package is responsible for converting the byte target to element counts based on each array's dtype and shape.

**Bytes per point** (conservative estimate for points):
```python
bytes_per_point = n_dims * 4 + 16
# positions (n_dims * 4) + colors (12) + radii (4) + sharpness (4) ≈ n_dims*4 + 16
# Note: Positions store ALL dimensions; ordering_dims only affects spatial locality
```

**Target elements per chunk**:
```python
target_chunk_elements = TARGET_CHUNK_BYTES / bytes_per_point

# Examples:
# 3D data: 64KB / 28 bytes ≈ 2,300 points per chunk
# 4D data: 64KB / 32 bytes ≈ 2,000 points per chunk
# 5D data: 64KB / 36 bytes ≈ 1,800 points per chunk
```

**Discrete slice alignment** (when discrete dimensions exist):

Chunk boundaries should align with discrete slice boundaries to ensure:
- A time-slice query loads contiguous chunks
- No chunk spans multiple discrete slices

```python
# Number of discrete slices
num_discrete_slices = product(unique_values_per_discrete_dim)

# Points per discrete slice (average)
points_per_slice = total_points / num_discrete_slices

# Target chunks per slice (at least 1)
chunks_per_slice = max(1, round(points_per_slice / target_chunk_elements))

# Actual chunk size for this dataset
chunk_elements = ceil(points_per_slice / chunks_per_slice)
```

**Example calculations**:

| Dataset | Points | Discrete Slices | Pts/Slice | Chunks/Slice | Chunk Size |
|---------|--------|-----------------|-----------|--------------|------------|
| Small 4D | 100K | 10 time | 10,000 | 4-5 | 2,000-2,500 |
| Medium 5D | 1M | 300 (100×3) | 3,333 | 1-2 | 1,667-3,333 |
| Large 5D | 10M | 300 (100×3) | 33,333 | 14-17 | 1,960-2,380 |

**Edge cases**:
- **No discrete dimensions**: Use pure Morton ordering with target_chunk_elements
- **Very small slices** (< MIN_CHUNK_BYTES): Merge multiple slices per chunk (degrades slice query performance but maintains minimum chunk size)
- **Very large slices**: Multiple chunks per slice, aligned to target size

**Metadata stored**:
- `chunk_size`: int - elements per chunk (may vary slightly at slice boundaries)

---

### Write Algorithm

**Input**: positions (N, D), radii (N,), colors, scene_dimensions, etc.

**Steps**:
1. **Identify dimension types** from scene_dimensions:
   - `slice_dims`: non-displayed discrete dimensions (time, channel)
   - `ordering_dims`: displayed dimensions + non-displayed spatial dimensions

2. **Compute coordinate bounds** for continuous dimensions only:
   ```
   For each d in ordering_dims:
       ordering_min[d] = positions[:, d].min()
       ordering_max[d] = positions[:, d].max()
   ```

3. **Compute compound sort keys**:
   ```
   For each point p:
       discrete_key[p] = tuple(positions[p, d] for d in slice_dims)
       continuous_coords = [positions[p, d] for d in ordering_dims]
       morton_key[p] = compute_morton_code(normalize(continuous_coords))
       sort_key[p] = (discrete_key[p], morton_key[p])
   ```

4. **Sort by compound key**: Get sort indices, reorder ALL arrays (positions, colors, radii, sharpness)

5. **Calculate chunk size**: Using discrete slice alignment (see Chunk Sizing Strategy)

6. **Write sorted arrays**: To Zarr with calculated chunk size

7. **Compute chunk bounds**: For each chunk, compute AABB including radii (all dimensions)

8. **Write chunk_bounds**: As separate lightweight array

9. **Write metadata**: slice_dims, ordering_dims, ordering_min, ordering_max, ordering_bits_per_dim, chunk_size

**Key invariant**: All point attribute arrays (positions, colors, radii, sharpness) are stored in the same compound-sorted order. Chunk boundaries align with discrete slice boundaries.

---

### Query Algorithm (Client-Side)

**Input**:
- `chunk_bounds`: Pre-loaded (num_chunks, n_dims, 2) array
- `view_bounds`: For each dimension, the visible range `[min, max]`
  - Displayed dimensions: view frustum bounds
  - Non-displayed dimensions: slice position ± tolerance

**Algorithm**:
```
visible_chunks = []
for chunk_idx in 0..(num_chunks - 1):
    intersects = true
    for dim in 0..(n_dims - 1):
        chunk_min = chunk_bounds[chunk_idx, dim, 0]
        chunk_max = chunk_bounds[chunk_idx, dim, 1]
        view_min = view_bounds[dim, 0]
        view_max = view_bounds[dim, 1]

        # AABB intersection test
        if chunk_max < view_min or chunk_min > view_max:
            intersects = false
            break

    if intersects:
        visible_chunks.append(chunk_idx)

return visible_chunks
```

**Output**: List of chunk indices to load

**Loading**: Client requests specific chunks from Zarr arrays (positions, colors, radii, sharpness) using chunk indices directly.

**Performance benefit of compound ordering**: For discrete dimension queries (e.g., "show time=5"), the matching chunks are contiguous in the chunk list (because data is sorted by discrete dimensions first). This means fewer chunks typically match, and they can be loaded in a single range request.

---

### Zarr Storage Schema

**Points Group** (`/points_name/`):
```
positions/          # (N, D) float32, compound-sorted, chunked (required)
radii/              # (N,) or (1,) float32, compound-sorted, chunked (required)
colors/             # (N, 3) or (1, 3) float32/uint8, compound-sorted, chunked (optional)
sharpness/          # (N,) or (1,) float32, compound-sorted, chunked (optional)
chunk_bounds        # (num_chunks, D, 2) float32, single chunk
```

**Attributes** (on points group):
```json
{
  "type": "points",
  "n_points": 1000000,
  "slice_dims": [3, 4],
  "ordering_dims": [0, 1, 2],
  "ordering_min": [0.0, 0.0, 0.0],
  "ordering_max": [100.0, 100.0, 100.0],
  "ordering_bits_per_dim": 21,
  "chunk_size": 2000,
  "extend_to_all": ["Time"]
}
```

**Notes**:
- `ordering_min/max` only covers morton dimensions (used for Morton normalization)
- `extend_to_all` is optional - only present if points should appear at all values of specified dimensions (see core/SPECIFICATIONS.md)

**GSplats Group** (`/splats_name/`):
```
centers/            # (N, D) float32, compound-sorted, chunked
colors/             # (N, 3) float32/uint8, compound-sorted, chunked
amplitudes/         # (N,) float32, compound-sorted, chunked
cholesky_factors/   # (N, k) float32, compound-sorted, chunked
sharpness/          # (N,) float32, compound-sorted, chunked
chunk_bounds        # (num_chunks, D, 2) float32, single chunk
```

**GSplats Attributes** (on splats group):
```json
{
  "type": "gsplats",
  "n_splats": 1000000,
  "slice_dims": [3],
  "ordering_dims": [0, 1, 2],
  "ordering_min": [0.0, 0.0, 0.0],
  "ordering_max": [100.0, 100.0, 100.0],
  "ordering_bits_per_dim": 21,
  "chunk_size": 2000
}
```

---

### Chunk Metadata Size

**Formula for chunk_bounds size**:
```
metadata_bytes = num_chunks × n_dims × 2 × 4
              = (n_points / chunk_size) × n_dims × 8
```

Example: 10M points, 5D, chunk_size=2K → 5000 chunks × 5 × 8 = 200KB metadata

### Zarr Array Chunking Rules

**1D arrays** (radii, sharpness, amplitudes):
- Chunk shape: `(chunk_size,)`

**2D arrays** (positions, colors, cholesky_factors):
- Chunk along first dimension only
- Shape: `(chunk_size, n_cols)` - keep all columns together

**chunk_bounds array**:
- Single chunk (loaded entirely for queries)

### Compression Interaction

- Compound ordering improves compression by grouping similar values
- Blosc bitshuffle works well with sorted floating-point data
- Larger chunks = better compression ratio, slower random access (trade-off)

---

## Data Type Configuration

### Type Selection Strategy

**Positions**:
- Always: float32 (accuracy critical)
- Could use float16 for small-scale data (not implemented)

**Colors**:
- **uint8**: Implicit SDR, values [0, 255] normalized to [0, 1]
- **float32 with `color_mode="sdr"`**: Values must be in [0, 1], out-of-range raises error
- **float32 with `color_mode="hdr"`**: Unbounded non-negative values allowed

**Note**: Auto-detection (values > 1 = HDR) was rejected because buggy SDR data would silently be treated as HDR instead of raising an error. Explicit `color_mode` is required.

**Radii/Sharpness**:
- Default: float32
- Future: Could use uint8 (normalized) or float16 for memory efficiency (not implemented)

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

## Large Dataset Strategy

For very large datasets (TB-scale), the recommended approach is:

**Split into Multiple Nodes**:
```python
with LuxarZarrCompiler('huge.zarr', ordering_method="hilbert") as compiler:
    scene = compiler.create_scene()

    # Process in chunks, each chunk becomes a separate node
    for i in range(100):
        chunk_positions = load_chunk(i)  # Load 10M points at a time
        scene.add_points(f'chunk_{i}', chunk_positions, colors, radii)
        # Each node is independently sorted and has chunk_bounds
```

**Benefits**:
- Each node fits in memory for spatial ordering
- Each node has proper chunk_bounds for efficient queries
- Viewer can query all nodes in parallel
- Better than single unsorted blob

**Memory**: Process one chunk at a time (10M points ~1GB), not entire dataset.

---

## Reading Luxar Scenes

### Purpose

Provide Python API to read and validate Luxar scene files (.zarr format). Enables:
- Round-trip testing (write → read → verify)
- Python-based scene analysis and inspection
- Format validation and debugging
- Data extraction for processing

### API Design

#### LuxarScene Class

```python
from luxar.io import LuxarScene

# Load scene (read-only)
scene = LuxarScene.load('scene.zarr')

# Scene metadata
scene.version: str                    # Luxar format version
scene.dimensions: Optional[Dimensions]  # Scene dimensions (if present)
scene.root_attrs: Dict[str, Any]      # All root attributes
scene.path: Path                      # Path to zarr store

# Node discovery
scene.nodes: List[Dict[str, Any]]     # All node metadata
# Returns: [
#   {'name': 'my_points', 'type': 'points', 'n_points': 1000,
#    'ordering': 'morton', 'has_colors': True, ...},
#   {'name': 'my_splats', 'type': 'gsplats', 'n_splats': 500, ...},
# ]

scene.list_points() -> List[str]      # Names of all points nodes
scene.list_gsplats() -> List[str]     # Names of all gsplats nodes
scene.list_lines() -> List[str]       # Names of all lines nodes
scene.list_groups() -> List[str]      # Names of all group nodes

# Node queries
scene.has_node(name: str) -> bool
scene.get_node_type(name: str) -> str  # 'points', 'gsplats', 'lines', 'group'
scene.get_node_metadata(name: str) -> Dict[str, Any]  # Metadata only, no data

# Load node data (automatic decoding via ArrayDecoder)
scene.get_points(name: str) -> Dict[str, Any]
scene.get_gsplats(name: str) -> Dict[str, Any]
scene.get_lines(name: str) -> Dict[str, Any]
```

#### Return Value Structure

**get_points() returns**:
```python
{
    # Data arrays (decoded via ArrayDecoder)
    'positions': ndarray,      # (N, D) float32
    'colors': ndarray,         # (N, 3) float32 (if present, else None)
    'radii': ndarray,          # (N,) float32 (if present, else None)
    'sharpness': ndarray,      # (N,) float32 (if present, else None)

    # Spatial ordering (arrays and metadata in points group directly)
    'chunk_bounds': ndarray,   # (num_chunks, D, 2) float32 (if ordered, else None)

    # Metadata (from points group .zattrs)
    'metadata': {
        'type': 'points',
        'n_points': int,
        'max_radius': float,     # (if radii present, else not in attrs)
        'opacity': float,        # (default 1.0)
        'gamma': float,          # (default 1.0)
        'blending_mode': str,    # (default 'additive')
        'transform': ndarray,    # (4, 4) if present, else None)
        'extend_to_all': List[str],  # (if present, else not in attrs)

        # Spatial ordering metadata (if ordered)
        'ordering': str,          # 'morton', 'hilbert', or 'none'
        'slice_dims': List[int],  # Discrete dimension indices (if ordered)
        'ordering_dims': List[int], # Spatial dimension indices (if ordered)
        'ordering_min': List[float],  # Min bounds for Morton dims (if ordered)
        'ordering_max': List[float],  # Max bounds for Morton dims (if ordered)
        'ordering_bits_per_dim': int,  # Bits per dimension (if ordered)
        'chunk_size': int,        # Elements per chunk (if ordered)

        # ... any custom attrs
        # Note: has_colors, has_radii, has_sharpness are NOT stored
        # Reader detects presence by checking if arrays exist in group
    }
}
```

**get_gsplats() returns**:
```python
{
    # Data arrays (decoded)
    'centers': ndarray,           # (N, D) float32
    'amplitudes': ndarray,        # (N,) float32
    'cholesky_factors': ndarray,  # (N, k) float32
    'colors': ndarray,            # (N, 3) float32 (if present, else None)
    'sharpness': ndarray,         # (N,) float32 (if present, else None)

    # Spatial ordering (NOTE: stored DIRECTLY in gsplats group, not sub-group like Points)
    'chunk_bounds': ndarray,      # (num_chunks, D, 2) (if ordered, else None)

    # Metadata (from gsplats group .zattrs, includes ordering metadata)
    'metadata': {
        'type': 'gsplats',
        'n_splats': int,
        'ndim': int,
        'ordering': str,          # 'morton', 'hilbert', or 'none'
        'ordering_min': List[float],  # (if ordered)
        'ordering_max': List[float],  # (if ordered)
        'ordering_bits_per_dim': int,  # (if ordered)
        'chunk_size': int,          # (if ordered)
        'has_colors': bool,
        'has_sharpness': bool,
        'amplitude_range': Dict[str, float],  # min/max
        'center_bounds': Dict[str, List[float]],  # min/max
        'transform': ndarray,  # (4, 4) if present, else None
        # ... custom attrs
    }
}
```

**Note**: Both Points and GSplats now store ordering metadata and chunk_bounds directly in their group (simple, consistent structure).

**get_lines() returns**:
```python
{
    # Data arrays (decoded)
    'vertices': ndarray,     # (N, D) float32
    'widths': ndarray,       # (N,) float32
    'colors': ndarray,       # (N, 3) float32 (if present)
    'sharpness': ndarray,    # (N,) float32 (if present)
    'indices': ndarray,      # (M,) uint32 (if indexed line type)

    # Metadata
    'metadata': {
        'n_vertices': int,
        'n_segments': int,
        'ndim': int,
        'line_type': str,  # 'segments', 'polyline', 'loop', 'indexed'
        'has_colors': bool,
        'has_sharpness': bool,
        'max_width': float,
        'transform': ndarray,  # (4, 4) if present
        # ... custom attrs
    }
}
```

### Usage Examples

**Inspection**:
```python
scene = LuxarScene.load('scene.zarr')

# Scene overview
print(f"Luxar version: {scene.version}")
if scene.dimensions:
    print(f"Dimensions: {[d.name for d in scene.dimensions.dimensions]}")

# List all nodes
print(f"Scene has {len(scene.nodes)} nodes:")
for node in scene.nodes:
    print(f"  - {node['name']} ({node['type']})")
```

**Data loading**:
```python
# Load points data
points = scene.get_points('my_cloud')
positions = points['positions']  # Already decoded to float32
colors = points['colors']        # Decoded (or None if not present)
metadata = points['metadata']

# Check spatial ordering
if points['chunk_bounds'] is not None:
    print(f"Ordered with {metadata['ordering']}")
    print(f"Chunks: {len(points['chunk_bounds'])}")
    print(f"Discrete dims: {metadata['slice_dims']}")
    print(f"Spatial dims: {metadata['ordering_dims']}")
```

**Round-trip test**:
```python
# Write
with LuxarZarrCompiler('test.zarr') as compiler:
    scene_w = compiler.create_scene()
    scene_w.add_points('test', positions, colors, radii)

# Read back
scene_r = LuxarScene.load('test.zarr')
data = scene_r.get_points('test')

# Verify data
assert np.allclose(data['positions'], positions)
assert np.allclose(data['colors'], colors)

# Verify spatial ordering
assert data['chunk_bounds'] is not None
assert data['metadata']['ordering'] in ('morton', 'hilbert')
assert len(data['chunk_bounds']) > 0
```

### Implementation Notes

- Uses `ArrayDecoder` from `luxar.encoding` for automatic decoding
- Transforms are read and un-transposed (reverse of prepare_transform_for_zarr)
- Read-only (no modification)
- Lazy loading (data only loaded when get_* methods called)
- Validates format version on load

---

## Zarr Store Structure (Summary)

**Root Attributes** (minimum required):
- `type`: "scene"
- `luxar_version`: "0.1"
- `scene_dimensions`: Dimension specifications (names, display, discrete, spatial flags)
- `spatial_extend_dims`: Boolean array indicating which dimensions are spatial (for query tolerance calculation)

**Points Group**:
- `positions`: (N, D) float32 array, compound-sorted
- `radii`: (N,) or (1,) float32 array, compound-sorted (required)
- `colors`: (N, 3) or (1, 3) float32/uint8 array, compound-sorted (optional)
- `sharpness`: (N,) or (1,) float32 array, compound-sorted (optional)
- `chunk_bounds`: (num_chunks, D, 2) float32 array

**Points Attributes**:
- `type`: "points"
- `n_points`: Number of points
- `slice_dims`, `ordering_dims`: Dimension indices for compound ordering
- `ordering_min`, `ordering_max`: Bounds for morton dimensions only
- `ordering_bits_per_dim`: int - bits per morton dimension
- `chunk_size`: int - elements per chunk
- `max_radius`: Maximum radius
- `extend_to_all`: Optional list of dimension names for visibility extension
- `opacity`, `gamma`, `blending_mode`: Rendering attributes
- `transform`: Optional 16-element list (column-major)

**GSplats Group**:
- `centers`: (N, D) float32 array, compound-sorted
- `colors`: (N, 3) float32/uint8 array, compound-sorted (optional)
- `amplitudes`: (N,) float32 array, compound-sorted
- `cholesky_factors`: (N, k) float32 array, compound-sorted
- `sharpness`: (N,) float32 array, compound-sorted (optional)
- `chunk_bounds`: (num_chunks, D, 2) float32 array

**Lines Group** (no spatial indexing):
- `vertices`: (N, D) float32 array (NOT sorted - connectivity matters)
- `colors`: (N, 3) float32/uint8 array (optional)
- `widths`: (N,) float32 array
- `sharpness`: (N,) float32 array (optional)
- `indices`: (M,) uint32 array (for indexed type only)
- Note: Lines do not support spatial indexing in current version

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

**Morton Ordering** (to be added to `typing_utils/constants.py`):
- `MORTON_CODE_BITS = 64` - Total bits for Morton code
- Morton bits per dimension = `floor(64 / n_dims)` (computed dynamically)

**Chunk Sizing** (existing in `typing_utils/constants.py`):
- `CHUNK_SIZE_DEFAULT` - Default elements per chunk
- `CHUNK_SIZE_MIN` - Minimum chunk size
- `CHUNK_SIZE_MAX` - Maximum chunk size

**Sharpness** (existing in `typing_utils/constants.py`):
- `SHARPNESS_MIN = 0.0`
- `SHARPNESS_MAX = 31.0`
- `SHARPNESS_DEFAULT = 2.0`

**GSplat Extent Coverage** (to be added to `typing_utils/constants.py`):
- `GSPLAT_EXTENT_SIGMA = 3.0` - Number of standard deviations for bounding box (99.7% coverage)


---

## Error Handling

**Write-Time Validation**:
- Positions: Must be 2D array, at least 1 point, at least 1 dimension
- Colors (uint8): Must be (N, 3), values in [0, 255]
- Colors (float32 SDR): Must be (N, 3), values in [0, 1], requires `color_mode="sdr"`
- Colors (float32 HDR): Must be (N, 3), non-negative, requires `color_mode="hdr"`
- Radii: Must be (N,) or (1,), all positive (> 0), no zeros
- Sharpness: Must be (N,) or (1,), range [0, 31]

**Helpful Error Messages**:
- Include actual vs expected values
- Provide suggestions for fixing
- Use custom ValidationError class with suggestion field

---

## This specification provides sufficient detail to re-implement the I/O system while maintaining compatibility with existing Zarr stores.

---

## Changelog

- **v1.4.0** (2025-11-29): Scalar input support for uniform attributes
  - **NEW FEATURE**: Compiler accepts scalar inputs directly for uniform attributes
  - `radii=0.5` instead of `np.full(N, 0.5)` - no intermediate array created
  - `colors=(1.0, 0.0, 0.0)` instead of `np.full((N, 3), [1.0, 0.0, 0.0])`
  - `sharpness=2.0` instead of `np.full(N, 2.0)`
  - Compiler passes scalars directly to encoder with `n_elements=N`
  - **Performance**: Zero memory allocation for uniform attributes
  - Applies to write_points(), write_lines(), write_gsplats()
  - Requires `luxar.encoding` v0.6.0 (ArrayEncoder scalar support)
  - **Note**: Positions cannot be scalar (COORDINATE type blocks broadcasting)

- **v1.3.1** (2025-11-28): StreamingPoints removed
  - **BREAKING**: Deleted StreamingPoints class (incompatible with spatial ordering)
  - Streaming appends data progressively but can't apply spatial ordering
  - Without ordering → no chunk_bounds → viewer must load all data
  - **Solution**: Split large datasets into multiple nodes (see "Large Dataset Strategy")
  - Each node fits in memory, gets proper spatial ordering and chunk_bounds
  - Deleted streaming.py (301 lines) and test_streaming_comprehensive.py (12 tests)

- **v1.3.0** (2025-11-28): Hilbert curve support
  - Added Hilbert curve as alternative to Morton ordering
  - New module: `luxar.io.ordering` with Morton/Hilbert implementations
  - `sort_points_compound()` supports both Morton and Hilbert (default: Morton)
  - `sort_splats_spatial()` supports both Morton and Hilbert (default: Hilbert)
  - Hilbert provides ~10% better compression than Morton
  - Requires `hilbertcurve>=2.0.5` package for Hilbert support
  - Metadata format unchanged (same ordering_min/max/bits_per_dim for both methods)
  - `ordering` attribute distinguishes: "morton" or "hilbert"
  - **StreamingPoints removed**: Incompatible with spatial ordering (see v1.3.1)

- **v1.2.2** (2025-11-28): Encoding system integration
  - **BREAKING**: Replaced `dtype_config` parameter with `encoding_mode` (EncodingMode enum)
  - All array writes use ArrayEncoder from luxar.encoding package
  - Supports broadcasting, LUT encoding, and array reference deduplication
  - See encoding/SPECIFICATIONS.md for encoding system details

- **v1.2.1** (2025-11-28): Chunk sizing source of truth clarification
  - Clarified that chunk size constants are defined in `typing_utils/SPECIFICATIONS.md` (single source of truth)
  - Removed local constant redefinition, now references typing_utils
  - Added note that this package is responsible for bytes→elements conversion

- **v1.2.0** (2025-11-28): Consolidation and clarifications
  - Consolidated write_points() specification (removed duplicate algorithm section)
  - Consolidated chunking sections (removed outdated "Chunking Strategy" section)
  - Morton bits calculation clarified: uses `len(ordering_dims)`, not total dimensions
  - Fixed bytes_per_point formula: uses `n_dims` (all dimensions), not `n_ordering_dims`
  - Added `extend_to_all` to Points Attributes schema
  - Added `spatial_extend_dims` to Root Attributes schema
  - Fixed examples to use `chunk_size: 2000` consistently
  - Updated all array descriptions to say "compound-sorted" instead of "Morton-sorted"

- **v1.1.0** (2025-11-27): Compound ordering and chunk sizing strategy
  - **Compound ordering**: Discrete dimensions sorted first, then Morton within each slice
  - Dramatically improves discrete dimension slicing (time, channel queries load contiguous chunks)
  - **Chunk sizing strategy**: Target 64KB chunks, aligned with discrete slice boundaries
  - New metadata: `slice_dims`, `ordering_dims` arrays
  - `ordering_min/max` now only covers morton dimensions (not slice dimensions)
  - Write algorithm updated for compound key sorting

- **v1.0.0** (2025-11-27): Morton ordering spatial index
  - **BREAKING**: Complete rewrite replacing grid-based spatial index with Morton ordering
  - Morton code (Z-order curve) for spatial locality sorting
  - Chunk bounding boxes (`chunk_bounds`) for efficient visibility queries
  - All dimensions indexed (not just non-displayed)
  - Radius-aware bounds include element extent
  - Added Cholesky→Covariance formula for GSplat extent calculation
  - Explicit `color_mode` flag required for float32 colors (SDR vs HDR)
  - `radii` now required (not optional)
  - Sharpness range fixed to [0, 31]
  - Division by zero edge case handling in normalization
  - GSplats attributes section added with Morton metadata
  - Lines explicitly documented as NOT Morton-sorted

- **v0.x** (prior): Grid-based spatial index (deprecated)
  - Grid partitioning of non-displayed dimensions
  - `occupied_cells` and `cell_ranges` arrays
  - Cell ID linearization
