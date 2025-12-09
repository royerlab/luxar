# Lines Spatial Index Specification

**Version**: 1.0.0
**Status**: Draft
**Last Updated**: 2025-12-09

## Overview

This specification defines spatial indexing for Lines in Luxar, enabling efficient lazy loading of line segment data for large nD datasets. The design extends the existing Points/GSplats spatial indexing approach with adaptations specific to line geometry.

**Related Specifications**:
- `luxar.io` - Spatial indexing for Points/GSplats (see `io/SPECIFICATIONS.md`)
- `luxar.core` - Lines node structure (see `core/SPECIFICATIONS.md`)

---

## Table of Contents

1. [Design Goals](#design-goals)
2. [Unified Indexed Representation](#unified-indexed-representation)
3. [Dual Spatial Ordering](#dual-spatial-ordering)
4. [Extended Morton Encoding](#extended-morton-encoding)
5. [Compound Ordering for Discrete Dimensions](#compound-ordering-for-discrete-dimensions)
6. [Chunk Bounding Boxes](#chunk-bounding-boxes)
7. [Query Algorithm](#query-algorithm)
8. [Storage Schema](#storage-schema)
9. [API Design](#api-design)
10. [Implementation Notes](#implementation-notes)

---

## 1. Design Goals

1. **Efficient lazy loading**: Only load line segments visible in current view/slice
2. **nD support**: Work with arbitrary dimensional data (3D, 4D, 5D+)
3. **Preserve line geometry**: Capture full segment extent, not just centroids
4. **Consistent with Points**: Follow same patterns for compound ordering, chunk bounds
5. **API stability**: Internal optimizations don't affect user-facing API

---

## 2. Unified Indexed Representation

### Problem

The current Lines implementation supports 4 connectivity types:
- `segments`: Independent pairs `(0,1), (2,3), (4,5)...`
- `polyline`: Connected sequence `(0,1), (1,2), (2,3)...`
- `loop`: Closed polyline `(0,1), (1,2)...(N-1,0)`
- `indexed`: Explicit index pairs

This complexity makes spatial ordering difficult because connectivity constraints vary.

### Solution

**Internally convert all line types to indexed representation.**

| User-Specified Type | Internal Conversion |
|---------------------|---------------------|
| `segments` | Direct: vertices 0,1 → segment (0,1), vertices 2,3 → segment (2,3), ... |
| `polyline` | Sequential: (0,1), (1,2), (2,3), ..., (N-2, N-1) |
| `loop` | Sequential + wrap: (0,1), (1,2), ..., (N-2, N-1), (N-1, 0) |
| `indexed` | Direct use of provided indices |

**Benefits**:
- Single internal representation simplifies spatial indexing
- User API remains unchanged (convenience preserved)
- Internal optimizations are decoupled from API surface

### Conversion Algorithm

```python
def convert_to_indexed(n_vertices: int, line_type: str, indices: Optional[np.ndarray]) -> np.ndarray:
    """Convert any line type to indexed segment pairs.

    Returns:
        segments: (S, 2) uint32 array of vertex index pairs
    """
    if line_type == "indexed":
        # Already indexed - reshape to (S, 2)
        return indices.reshape(-1, 2)

    elif line_type == "segments":
        # Pairs of consecutive vertices
        return np.arange(n_vertices, dtype=np.uint32).reshape(-1, 2)

    elif line_type == "polyline":
        # Connected sequence
        indices = np.column_stack([
            np.arange(n_vertices - 1, dtype=np.uint32),
            np.arange(1, n_vertices, dtype=np.uint32)
        ])
        return indices

    elif line_type == "loop":
        # Connected sequence + closing segment
        indices = np.column_stack([
            np.arange(n_vertices, dtype=np.uint32),
            np.roll(np.arange(n_vertices, dtype=np.uint32), -1)
        ])
        return indices
```

---

## 3. Dual Spatial Ordering

### Core Insight

Lines have **two arrays** that benefit from spatial ordering:

1. **Vertices**: The actual nD positions - order for vertex data locality
2. **Segments**: Index pairs - order for segment query locality

Both are ordered independently, with index remapping to maintain correctness.

### Vertex Ordering (D-dimensional)

Vertices are ordered using standard Morton/Hilbert in D-dimensional space:

```python
# vertices: (V, D) float32
vertex_sort_indices = morton_encode_nd(vertices)
sorted_vertices = vertices[vertex_sort_indices]
```

### Segment Ordering (2D-dimensional)

**Key Innovation**: Treat each segment as a point in **2D-dimensional space**.

A segment connecting P1 → P2 becomes the concatenation `(P1, P2)`:

```
Segment: (x1, y1, z1) → (x2, y2, z2)
Becomes: (x1, y1, z1, x2, y2, z2)  ← 6D point
```

This captures the **full geometric nature** of the segment:
- Similar start points cluster together
- Similar end points cluster together
- Similar orientation/length segments cluster together

**Why not midpoint?** Midpoint loses orientation information:

| Segment | Endpoints | Midpoint | 2D Representation |
|---------|-----------|----------|-------------------|
| A | (0,0,0)→(2,2,2) | (1,1,1) | (0,0,0,2,2,2) |
| B | (0,2,0)→(2,0,2) | (1,1,1) | (0,2,0,2,0,2) |
| C | (1,1,0)→(1,1,2) | (1,1,1) | (1,1,0,1,1,2) |

Midpoint ordering would interleave these very different segments. 2D ordering keeps geometrically similar segments together.

### Index Remapping

When vertices are reordered, segment indices must be remapped:

```python
def reorder_with_remapping(vertices, segments, vertex_sort_indices):
    """Reorder vertices and remap segment indices."""
    # Sort vertices
    sorted_vertices = vertices[vertex_sort_indices]

    # Create inverse mapping: old_index → new_index
    inverse_map = np.argsort(vertex_sort_indices)

    # Remap segment indices
    remapped_segments = inverse_map[segments]

    return sorted_vertices, remapped_segments
```

### Complete Dual Ordering Algorithm

```python
def order_lines_spatial(vertices, segments, dimensions, method="morton"):
    """Apply dual spatial ordering to lines.

    Args:
        vertices: (V, D) float32 vertex positions
        segments: (S, 2) uint32 index pairs
        dimensions: List of Dimension objects
        method: "morton" or "hilbert"

    Returns:
        sorted_vertices: Reordered vertices
        sorted_segments: Reordered and remapped segments
        vertex_sort_indices: For reordering vertex attributes
        segment_sort_indices: For reordering segment attributes (if any)
        metadata: Ordering metadata
    """
    V, D = vertices.shape
    S = segments.shape[0]

    # 1. Order vertices in D-space (with compound ordering for discrete dims)
    vertex_sort_indices, vertex_metadata = sort_points_compound(
        vertices, dimensions, method=method
    )
    sorted_vertices = vertices[vertex_sort_indices]

    # 2. Create inverse mapping for index remapping
    inverse_map = np.argsort(vertex_sort_indices)
    remapped_segments = inverse_map[segments]

    # 3. Build 2D segment coordinates for ordering
    # Each segment (v1, v2) becomes (P1, P2) in 2D space
    segment_coords_2d = np.concatenate([
        sorted_vertices[remapped_segments[:, 0]],  # Start points
        sorted_vertices[remapped_segments[:, 1]]   # End points
    ], axis=1)  # Shape: (S, 2*D)

    # 4. Order segments in 2D-space (with compound ordering)
    segment_sort_indices, segment_metadata = sort_segments_compound(
        segment_coords_2d, remapped_segments, dimensions, method=method
    )
    sorted_segments = remapped_segments[segment_sort_indices]

    return (
        sorted_vertices,
        sorted_segments,
        vertex_sort_indices,
        segment_sort_indices,
        {
            "vertex_ordering": vertex_metadata,
            "segment_ordering": segment_metadata,
        }
    )
```

---

## 4. Extended Morton Encoding

### Problem

Standard 64-bit Morton codes provide limited precision for high-dimensional data:

| Dimensions | Bits per dimension | Precision |
|------------|-------------------|-----------|
| 3D | 21 bits | ~2M levels |
| 6D (3D segments) | 10 bits | ~1K levels |
| 10D (5D segments) | 6 bits | ~64 levels |

For 5D segment data (10 dimensions in 2D space), 64 levels is insufficient.

### Solution: 128-bit Morton Encoding

Extend Morton codes to 128 bits using Python's arbitrary-precision integers or paired uint64:

| Dimensions | Bits/dim (64-bit) | Bits/dim (128-bit) |
|------------|-------------------|---------------------|
| 3D | 21 bits | 42 bits |
| 6D (3D segments) | 10 bits | 21 bits |
| 10D (5D segments) | 6 bits | 12 bits |
| 12D (6D segments) | 5 bits | 10 bits |

### Auto-Selection

```python
def select_morton_precision(n_dims: int, min_bits_per_dim: int = 10) -> int:
    """Select Morton precision based on dimensionality.

    Args:
        n_dims: Number of dimensions to encode
        min_bits_per_dim: Minimum acceptable bits per dimension

    Returns:
        Total bits (64 or 128)
    """
    if 64 // n_dims >= min_bits_per_dim:
        return 64
    elif 128 // n_dims >= min_bits_per_dim:
        return 128
    else:
        # Fall back to 128-bit even if still low precision
        return 128
```

### Implementation

**Option A: Python arbitrary-precision integers**

```python
def morton_encode_extended(coords: np.ndarray, bits_per_dim: int) -> np.ndarray:
    """Encode coordinates to arbitrary-precision Morton codes."""
    n_points, n_dims = coords.shape
    morton = np.empty(n_points, dtype=object)  # Python int objects

    for i in range(n_points):
        code = 0
        for bit in range(bits_per_dim):
            for dim in range(n_dims):
                if (coords[i, dim] >> bit) & 1:
                    code |= 1 << (bit * n_dims + dim)
        morton[i] = code

    return morton
```

**Option B: Paired uint64 (faster, Numba-compatible)**

```python
def morton_encode_128bit(coords: np.ndarray, bits_per_dim: int) -> tuple:
    """Encode to 128-bit Morton as (high, low) uint64 pairs."""
    n_points, n_dims = coords.shape
    high = np.zeros(n_points, dtype=np.uint64)
    low = np.zeros(n_points, dtype=np.uint64)

    for i in range(n_points):
        for bit in range(bits_per_dim):
            for dim in range(n_dims):
                bit_pos = bit * n_dims + dim
                if (coords[i, dim] >> bit) & 1:
                    if bit_pos < 64:
                        low[i] |= np.uint64(1) << bit_pos
                    else:
                        high[i] |= np.uint64(1) << (bit_pos - 64)

    return high, low

# Sort lexicographically
sort_indices = np.lexsort((low, high))
```

**Note**: We only need sort indices - Morton codes themselves are not stored.

---

## 5. Compound Ordering for Discrete Dimensions

Like Points, Lines support compound ordering for nD data with discrete dimensions.

### Dimension Classification

```python
# From scene dimensions
slice_dims = [i for i, d in enumerate(dimensions) if d.discrete and not d.display]
ordering_dims = [i for i, d in enumerate(dimensions) if not d.discrete or d.display]
```

### Vertex Compound Ordering

Same as Points - discrete dimensions first, then Morton/Hilbert of spatial dimensions.

### Segment Compound Ordering

For segments in 2D space:
- **Slice dimensions**: The discrete dimensions from BOTH endpoints
- **Ordering dimensions**: The spatial dimensions from BOTH endpoints

```python
def sort_segments_compound(segment_coords_2d, segments, dimensions, method):
    """Compound ordering for segments in 2D space.

    segment_coords_2d: (S, 2*D) - concatenated endpoint coordinates
    """
    D = len(dimensions)

    # Identify slice/ordering dims in the 2D space
    # Each dimension appears twice (for P1 and P2)
    slice_dims_2d = []
    ordering_dims_2d = []

    for i, dim in enumerate(dimensions):
        if dim.discrete and not dim.display:
            slice_dims_2d.extend([i, D + i])  # Both endpoints
        else:
            ordering_dims_2d.extend([i, D + i])  # Both endpoints

    # Apply compound ordering in 2D space
    # Primary: slice dimensions (lexicographic)
    # Secondary: Morton/Hilbert of ordering dimensions
    ...
```

---

## 6. Chunk Bounding Boxes

### Dual Chunk Bounds

Lines require **two** chunk bounds arrays:

1. **`vertex_chunk_bounds`**: Bounds for vertex chunks (for vertex data loading)
2. **`segment_chunk_bounds`**: Bounds for segment chunks (for visibility queries)

Both are stored in **D-dimensional space** (not 2D) for view frustum intersection tests.

### Vertex Chunk Bounds

Same as Points - bounding box of vertex positions:

```python
def compute_vertex_chunk_bounds(vertices, chunk_size):
    """Compute bounding boxes for vertex chunks."""
    V, D = vertices.shape
    num_chunks = (V + chunk_size - 1) // chunk_size
    bounds = np.zeros((num_chunks, D, 2), dtype=np.float32)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, V)
        chunk_verts = vertices[start:end]
        bounds[i, :, 0] = chunk_verts.min(axis=0)
        bounds[i, :, 1] = chunk_verts.max(axis=0)

    return bounds
```

### Segment Chunk Bounds (Width-Aware)

Segment bounds must include line width (segments are "capsules", not infinitely thin):

```python
def compute_segment_chunk_bounds(vertices, segments, widths, chunk_size, slice_dims=None):
    """Compute bounding boxes for segment chunks.

    Args:
        vertices: (V, D) sorted vertex positions
        segments: (S, 2) sorted and remapped segment indices
        widths: (V,) per-vertex widths (already sorted with vertices)
        chunk_size: Segments per chunk
        slice_dims: Discrete dimension indices (no width expansion)

    Returns:
        bounds: (num_chunks, D, 2) float32
    """
    S = segments.shape[0]
    D = vertices.shape[1]
    num_chunks = (S + chunk_size - 1) // chunk_size
    discrete_dims = set(slice_dims) if slice_dims else set()

    bounds = np.zeros((num_chunks, D, 2), dtype=np.float32)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, S)
        chunk_segs = segments[start:end]

        # Get endpoint positions and widths
        p1 = vertices[chunk_segs[:, 0]]  # (chunk_size, D)
        p2 = vertices[chunk_segs[:, 1]]  # (chunk_size, D)
        w1 = widths[chunk_segs[:, 0]]    # (chunk_size,)
        w2 = widths[chunk_segs[:, 1]]    # (chunk_size,)

        # Max width at each segment (conservative)
        max_w = np.maximum(w1, w2)[:, np.newaxis]  # (chunk_size, 1)

        for d in range(D):
            if d in discrete_dims:
                # Discrete dimension: no width expansion
                bounds[i, d, 0] = min(p1[:, d].min(), p2[:, d].min()) - 0.5
                bounds[i, d, 1] = max(p1[:, d].max(), p2[:, d].max()) + 0.5
            else:
                # Spatial dimension: include width
                bounds[i, d, 0] = min((p1[:, d] - max_w[:, 0]).min(),
                                      (p2[:, d] - max_w[:, 0]).min())
                bounds[i, d, 1] = max((p1[:, d] + max_w[:, 0]).max(),
                                      (p2[:, d] + max_w[:, 0]).max())

    return bounds
```

---

## 7. Query Algorithm

### Viewer-Side Loading Strategy

```
1. Load segment_chunk_bounds (lightweight, cached)
2. Query: which segment chunks intersect view bounds?
3. Load matching segment chunks → get (v1, v2) pairs
4. Collect unique vertex indices from loaded segments
5. Determine which vertex chunks contain those indices
6. Load required vertex chunks
7. Render segments using loaded vertex data
```

### Segment Visibility Query

```typescript
function queryVisibleSegmentChunks(
  segmentChunkBounds: Float32Array,  // (numChunks, D, 2)
  viewBounds: Float32Array,          // (D, 2) - [min, max] per dimension
  numChunks: number,
  numDims: number
): number[] {
  const visibleChunks: number[] = [];

  for (let chunk = 0; chunk < numChunks; chunk++) {
    let intersects = true;

    for (let d = 0; d < numDims; d++) {
      const chunkMin = segmentChunkBounds[(chunk * numDims + d) * 2];
      const chunkMax = segmentChunkBounds[(chunk * numDims + d) * 2 + 1];
      const viewMin = viewBounds[d * 2];
      const viewMax = viewBounds[d * 2 + 1];

      if (chunkMax < viewMin || chunkMin > viewMax) {
        intersects = false;
        break;
      }
    }

    if (intersects) {
      visibleChunks.push(chunk);
    }
  }

  return visibleChunks;
}
```

### Vertex Chunk Lookup

Given segment indices, determine which vertex chunks to load:

```typescript
function getRequiredVertexChunks(
  segments: Uint32Array,           // Loaded segment data
  vertexChunkSize: number,
  numVertexChunks: number
): Set<number> {
  const requiredChunks = new Set<number>();

  for (let i = 0; i < segments.length; i++) {
    const vertexIndex = segments[i];
    const chunkIndex = Math.floor(vertexIndex / vertexChunkSize);
    requiredChunks.add(chunkIndex);
  }

  return requiredChunks;
}
```

---

## 8. Storage Schema

### Zarr Structure

```
/lines_name/
  vertices/                  # (V, D) float32, Morton-ordered in D-space
  segments/                  # (S, 2) uint32, Morton-ordered in 2D-space
  widths/                    # (V,) or (1,) float32, ordered with vertices
  colors/                    # (V, 3) float32/uint8, ordered with vertices (optional)
  sharpness/                 # (V,) float32, ordered with vertices (optional)
  vertex_chunk_bounds        # (num_vertex_chunks, D, 2) float32
  segment_chunk_bounds       # (num_segment_chunks, D, 2) float32
```

### Attributes

```json
{
  "type": "lines",
  "n_vertices": 100000,
  "n_segments": 150000,
  "ndim": 5,

  "ordering": "morton",
  "ordering_precision": 128,

  "vertex_ordering": {
    "slice_dims": [3, 4],
    "ordering_dims": [0, 1, 2],
    "ordering_min": [0.0, 0.0, 0.0],
    "ordering_max": [100.0, 100.0, 100.0],
    "ordering_bits_per_dim": 21,
    "chunk_size": 2000
  },

  "segment_ordering": {
    "slice_dims": [3, 4, 8, 9],
    "ordering_dims": [0, 1, 2, 5, 6, 7],
    "ordering_min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ordering_max": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    "ordering_bits_per_dim": 10,
    "chunk_size": 3000
  },

  "max_width": 0.5,
  "has_colors": true,
  "has_sharpness": false,

  "original_line_type": "polyline"
}
```

**Notes**:
- `original_line_type`: Preserved for reference (internal representation is always indexed)
- `segment_ordering.slice_dims/ordering_dims`: Indices in 2D space (0..2D-1)
- `ordering_precision`: 64 or 128 bits

---

## 9. API Design

### User-Facing API (Unchanged)

```python
# All these remain valid - internal conversion is transparent
scene.add_lines("trajectory", vertices, widths=0.1, line_type="polyline")
scene.add_lines("connections", vertices, widths, line_type="indexed", indices=pairs)
scene.add_lines("outline", vertices, widths=0.05, line_type="loop")
scene.add_lines("edges", vertices, widths, line_type="segments")
```

### Compiler Integration

```python
class LuxarZarrCompiler:
    def __init__(
        self,
        store_path: Optional[str] = None,
        enable_spatial_index: bool = True,  # Applies to Lines too
        ordering_method: Literal["morton", "hilbert"] = "morton",
        ...
    ):
        ...
```

### Lines Class Updates

```python
class Lines(DataNode):
    """Lines node with optional spatial indexing."""

    @property
    def ordering(self) -> str:
        """Spatial ordering type: 'morton', 'hilbert', or 'none'."""
        return self._metadata.get("ordering", "none")

    @property
    def n_segments(self) -> int:
        """Number of line segments."""
        return self._metadata.get("n_segments", 0)

    @property
    def original_line_type(self) -> str:
        """Original line type before internal conversion."""
        return self._metadata.get("original_line_type", "indexed")
```

---

## 10. Implementation Notes

### Performance Considerations

1. **Index remapping**: O(S) operation - negligible for typical datasets
2. **2D Morton encoding**: 2x dimensions means ~half bits per dim - use 128-bit when needed
3. **Dual chunk bounds**: Small overhead (2 lightweight arrays vs 1)
4. **Vertex deduplication**: Shared vertices are stored once, referenced multiple times

### Edge Cases

1. **Single-segment lines**: Still get spatial index (one chunk)
2. **Very long polylines**: Many segments share vertices - good vertex locality
3. **Highly connected graphs**: Many segments per vertex - segment ordering helps
4. **Uniform widths**: Scalar width stored once, bounds use that value

### Backward Compatibility

- `ordering="none"`: No spatial index, viewer loads all data (existing behavior)
- `enable_spatial_index=False`: Explicitly disable ordering

### Future Extensions

1. **Hierarchical LOD**: Coarse line approximations for distant views
2. **Curve fitting**: Store control points + interpolation type
3. **Variable-width segments**: Per-segment width arrays (currently per-vertex)

---

## Changelog

- **v1.0.0** (2025-12-09): Initial specification
  - Unified indexed representation for all line types
  - Dual spatial ordering (vertices in D-space, segments in 2D-space)
  - Extended 128-bit Morton encoding for high-dimensional data
  - Compound ordering support for discrete dimensions
  - Dual chunk bounds (vertex + segment)
  - Width-aware segment bounding boxes
