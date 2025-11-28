# luxar.core - Technical Specification

**Version**: 0.10.2
**Last Updated**: 2025-11-28

## Purpose

The `core` package defines the fundamental data structures and scene graph system for organizing and manipulating large-scale visualization data. It provides a hierarchical scene graph with transformation support, nD dimensional specifications, and extensible data node types (Points, Lines, GSplats).

**Related Specifications**:
- `luxar.encoding` - Array encoding, semantic types, and quantization (see `encoding/SPECIFICATIONS.md`)
- `luxar.io` - I/O operations and writer protocol (see `io/SPECIFICATIONS.md`)
- `luxar.gsplats.io` - GSplats storage format (see `gsplats/io/SPECIFICATIONS.md`)

---

## Node Type Taxonomy

The scene graph consists of two categories of nodes:

### Container Nodes
Nodes that organize the hierarchy but don't hold visualization data:
- **Scene**: Root container, holds dimensions and global settings
- **Group**: Intermediate container for organizing data nodes

### Data Nodes
Nodes that hold actual visualization data (positions, colors, etc.):
- **Points**: Point cloud data (spheres at positions)
- **Lines**: Line/curve data (connected vertices)
- **GSplats**: Gaussian splat data (oriented ellipsoids)

All nodes share common properties (transforms, rendering attributes) but data nodes additionally have:
- Array data stored in Zarr
- Type-specific metadata
- Element count (`n_elements`)

### Node Type Discriminator

Each node has a `type` attribute stored in Zarr that identifies its kind:

| Node Class | `type` Value | Category |
|------------|--------------|----------|
| Scene | `"scene"` | Container |
| Group | `"group"` | Container |
| Points | `"points"` | Data |
| Lines | `"lines"` | Data |
| GSplats | `"gsplats"` | Data |

This discriminator enables the viewer to determine how to render each node.

---

## Core Data Structures

### 1. Node (Base Class)

**Specification**:
- Base class for all scene graph nodes
- Nodes form a hierarchical tree structure
- Each node has: name, parent reference, list of children
- Each node can have a 4x4 transformation matrix
- Each node has rendering attributes: opacity (0-1), gamma (0.1-10), blending_mode
- Container nodes do NOT store array data - only metadata and references
- Nodes write data immediately through a writer interface (progressive writing)

**Invariants**:
- Parent-child relationships are bidirectional (parent knows children, children know parent)
- Root node has no parent (parent = None)
- Transforms compose hierarchically: `world_transform = parent_world_transform @ local_transform`
  - `local_transform`: The node's own transform (stored in Zarr as `transform` attribute)
  - `parent_world_transform`: The world transform of the parent node (computed recursively)
  - `world_transform`: The effective transform that positions this node in world coordinates
  - For root nodes: `world_transform = local_transform` (since there's no parent)
- Attribute changes are immediately persisted through writer interface

**Node Name Rules**:
- Names must be non-empty strings
- Names must NOT contain `/` (used as path separator)
- Names must NOT contain null characters
- Names should be valid Zarr group names (alphanumeric, underscore, hyphen recommended)
- Names must be unique among siblings (no two children of same parent with same name)
- **Duplicate names raise `ValueError`**: Attempting to add a child with the same name as an existing sibling is an error. No auto-renaming or silent overwriting.

**Key Operations**:
- `add_group(name, **attrs)` - Create child group node
- `walk()` - Depth-first traversal yielding (depth, node) tuples
- Property getters/setters for transform, opacity, gamma, blending_mode

---

### 2. Scene (Root Container)

**Specification**:
- Special node that serves as scene root (extends Node)
- Created through `LuxarZarrCompiler.create_scene()`
- Requires a writer interface (cannot exist without one)
- Holds scene-level dimension definitions
- Provides factory methods for creating data nodes
- **Can have a transform** like any other Node (applies to all children)

**Key Operations**:
- `add_group(name, **attrs)` - Create top-level group
- `add_points(name, positions, radii, ...)` - Create Points node (radii required)
- `add_lines(name, vertices, widths, ...)` - Create Lines node (widths required)
- `add_gsplats(name, centers, amplitudes, cholesky_factors, ...)` - Create GSplats node
- `dimensions` property - Get/set scene-level dimensional specifications

**Zarr Attributes**:
```json
{
  "type": "scene",
  "luxar_format_version": "1.0",
  "scene_dimensions": { ... }
}
```
*Note: `luxar_format_version` is the data format version, not the spec document version. This version changes only when the zarr storage format has breaking changes.*

---

### 3. Group (Container Node)

**Specification**:
- Container node for organizing data nodes hierarchically
- Does NOT hold array data (only metadata and children)
- Can have transforms and rendering attributes
- Created via `Scene.add_group()` or `Node.add_group()`

**Key Properties**:
- `name`: Group identifier
- `children`: List of child nodes (Groups or DataNodes)
- `transform`: Optional 4x4 transformation matrix
- `opacity`, `gamma`, `blending_mode`: Rendering attributes (inherited by children)

**Zarr Attributes**:
```
{
  "type": "group",
  "transform": [...],         # optional, 16-element column-major
  "opacity": 1.0,             # optional
  "gamma": 1.0,               # optional
  "blending_mode": "additive" # optional
}
```
*Note: Comments shown for documentation; actual JSON has no comments.*

**Use Cases**:
- Organizing related data (e.g., "cells" group containing multiple Points)
- Applying shared transform to multiple children
- Applying shared rendering attributes

**Nesting Rules**:
- Groups can contain other Groups (arbitrary nesting depth allowed)
- Groups can contain any DataNode type (Points, Lines, GSplats)
- Empty Groups (zero children) are valid but should raise a warning during `finalize()` - the user may have forgotten to add children
- A Group with only Group children is valid (for purely organizational purposes)

---

### 4. DataNode (Abstract Base Class)

**Specification**:
`DataNode` is an abstract base class that all data-bearing nodes inherit from. It extends `Node` and adds data-specific capabilities.

**Class Definition** (abstract):
```python
class DataNode(Node, ABC):
    """Abstract base class for nodes that contain visualization data.

    All data nodes share:
    - Immediate writing to Zarr (no data kept in memory)
    - Type-specific metadata
    - Element count property
    - Semantic type mapping for encoding
    """

    @property
    @abstractmethod
    def n_elements(self) -> int:
        """Number of primary elements.

        What this counts depends on node type:
        - Points: number of points (n_points)
        - Lines: number of vertices (n_vertices)
        - GSplats: number of splats (n_splats)
        """
        ...
```

**Common Properties** (all DataNodes):

| Property | Type | Description |
|----------|------|-------------|
| `n_elements` | int | Count of primary elements (points, vertices, or splats) |
| `ndim` | int | Dimensionality of primary data |
| `path` | str | Zarr path (Node attribute, computed from hierarchy) |
| `type` | str | Node type discriminator (`"points"`, `"lines"`, `"gsplats"`) |
| `metadata` | dict | Type-specific metadata |
| Rendering attrs | float/str | opacity, gamma, blending_mode (inherited from Node) |
| Transform | 4x4 matrix | Optional transformation (inherited from Node) |

**Note**: `path` is a Node attribute computed from the hierarchy, not stored in the metadata dict. The metadata dict contains only data-specific information.

**Path Computation**:
```
child.path = f"{parent.path}/{name}" if parent.path else name
```
This handles the empty root path correctly: `"" + child` → `"child"`, not `"/child"`.

**Scene Root Path**: The Scene node has `path = ""` (empty string). This was chosen for the following reasons:
- Avoids double-slash issues (`"//" + "child"` vs `"" + "child"`)
- Matches Zarr internal path conventions (no leading slash)
- Simplifies path computation: if `parent.path` is empty, child path is just `name`
- Child paths are naturally relative: `"child"`, `"parent/child"`, `"parent/child/grandchild"`

**Type-Specific Properties**:

| Node Type | Required Arrays | Optional Arrays | Metadata Fields |
|-----------|-----------------|-----------------|-----------------|
| Points | positions (N, d), radii | colors, sharpness | n_points, ndim, has_colors, has_sharpness, max_radius |
| Lines | vertices (N, d), widths | colors, sharpness, indices | n_vertices, n_segments, ndim, line_type, has_colors, has_sharpness, max_width |
| GSplats | centers (N, d), amplitudes, cholesky_factors | colors, sharpness | n_splats, ndim, has_colors, has_sharpness, ordering, amplitude_range, center_bounds |

*Note: All "Required Arrays" must be provided. "Optional Arrays" have defaults if not provided.*

**Constructor Order Invariant**:
When subclass and parent both initialize the same attribute:
- CORRECT: Call `super().__init__()` FIRST, then set subclass attributes
- WRONG: Set attributes first, then call `super().__init__()` (parent overwrites)

---

### 5. Points Node

**Specification**:
- Data node for point cloud visualization
- Each point is rendered as a sphere with soft edges at a position
- Inherits all Node capabilities

**Data Arrays**:

| Array | Shape | Dtype | Required | Description |
|-------|-------|-------|----------|-------------|
| positions | (N, d) | float32 | Yes | Point centers in d dimensions |
| colors | (N, 3) or (1, 3) | float32/uint8 | No | RGB colors (HDR or SDR) |
| radii | (N,) or (1,) | float32 | Yes | Point radii in scene units |
| sharpness | (N,) or (1,) | float32 | No | Edge sharpness (polynomial falloff) |

**Sharpness Parameter** (Points/Lines):
Sharpness controls the edge falloff profile. The intensity falloff from center to edge follows a **polynomial** formula:
```
intensity(r) = (1 - r/radius)^sharpness
```
Where `r` is the distance from center (0 at center, radius at edge).
- **sharpness = 1.0**: Linear falloff
- **sharpness = 2.0**: Quadratic falloff (default)
- **sharpness > 2.0**: Sharper edges, more abrupt falloff
- **sharpness → ∞**: Approaches hard-edged disk

Valid range: [0, 31]. This is a polynomial falloff, NOT a Gaussian.

**Note**: GSplats use a different formula (generalized Gaussian) - see GSplats section and `gsplats/SPECIFICATIONS.md`.

**Metadata**:
```python
{
    "n_points": int,        # Number of points
    "ndim": int,            # Dimensionality (d)
    "has_colors": bool,
    "has_sharpness": bool,
    "max_radius": float,    # Maximum radius (for spatial queries)
}
```

**Computed Metadata**:
- `max_radius`: `max(radii)` - the maximum value in the radii array. For broadcast radii `(1,)`, this is that single value.

**Default Values** (when optional arrays not provided):
- colors: white `[1.0, 1.0, 1.0]`
- sharpness: `2.0` (quadratic polynomial falloff)

**Validation Rules**:
- N ≥ 1 (at least one point)
- Empty Points (N=0) are NOT valid for finalized nodes
  - **Exception**: During streaming writes (see `../../io/SPECIFICATIONS.md` → "Streaming Points")
  - StreamingPoints starts empty (N=0), becomes valid after first `append_batch()`
  - Validation enforced at write/finalize time, not construction time
- `radii` must be positive (> 0)
- `sharpness` must be in [0, 31] range
- `colors` (SDR float): values must be in [0, 1] range. Values outside this range raise `ValueError`.
- `colors` (HDR float): unbounded, any values allowed.
- `colors` (uint8): implicitly [0, 255], converted to [0, 1].
- All validation failures raise `ValueError`

**Color Mode (SDR vs HDR)**:
Float colors require an explicit `color_mode` parameter to distinguish SDR from HDR:
- `color_mode="sdr"`: Values must be in [0, 1] range. Values outside raise `ValueError`.
- `color_mode="hdr"`: Unbounded, any non-negative values allowed.

This explicit flag is **required** for float32 colors to avoid ambiguity. Auto-detection (values > 1 = HDR) was rejected because buggy SDR data with out-of-range values would silently be treated as HDR instead of raising an error.

**Implementation notes**:
- The `color_mode` flag must be passed to `write_points()`, `write_lines()`, `write_gsplats()`
- The flag is stored in array metadata: `{"encoding": {..., "color_mode": "sdr"|"hdr"}}`
- uint8 colors are implicitly SDR (no flag needed)
- The encoding layer uses this flag to determine validation and encoding strategy

**Encoding**: Uses `luxar.encoding` with semantic types:
- positions: COORDINATE
- colors: COLOR
- radii: POSITIVE_SCALAR
- sharpness: BOUNDED_SCALAR [0, 31]

**Storage Order** (Morton/Z-Order):
All point arrays are stored sorted by Morton code for spatial locality:
1. Coordinates are normalized to integer range based on data bounds
2. Morton code computed by bit-interleaving normalized coordinates
3. All arrays (positions, colors, radii, sharpness) sorted by Morton code
4. This ordering improves compression and enables efficient spatial queries

**Spatial Index** (`chunk_bounds`):
Points support chunk-level spatial indexing:
- `chunk_bounds`: (num_chunks, n_dims, 2) float32 array
- Each chunk's axis-aligned bounding box, **including point radii**
- Enables client to determine visible chunks without loading point data
- Bounds calculation: `min(position - radius)`, `max(position + radius)` per chunk

**Broadcasting** (`broadcast_dims`):
For nD scenes with non-displayed dimensions (e.g., Time, Channel), points can be broadcast to appear at all values of specified dimensions without data duplication.

*Parameter*: `broadcast_dims: Optional[Union[List[str], str]]`

| Value | Behavior |
|-------|----------|
| `None` (default) | No broadcasting - points only visible at their explicit dimension values |
| `["Time"]` | Broadcast across Time dimension only |
| `["Time", "Channel"]` | Broadcast across both Time and Channel |
| `"all"` | Broadcast across ALL non-displayed dimensions |

*Example*:
```python
# Points at time=0, channel=0 - only visible at that slice
scene.add_points("static", positions, broadcast_dims=None)

# Points appear at all time values (channel=0 only)
scene.add_points("time_invariant", positions, broadcast_dims=["Time"])

# Points appear everywhere (all times, all channels)
scene.add_points("global_markers", positions, broadcast_dims="all")
```

*Storage*:
- `broadcast_dims` is stored as a zarr attribute: `{"broadcast_dims": ["Time", "Channel"]}`
- **Position values**: Original input coordinates are stored unchanged
  - Example: If input has `time=0`, the stored position has `time=0`
  - The viewer ignores these values for broadcast dimensions
- **Compound ordering**: Points are sorted normally (by their actual time value)
  - Broadcast points end up in their "natural" position in the sorted order
  - This is intentional - the viewer handles loading specially

*Query Behavior* (viewer-side):
- When querying the spatial index, the viewer checks `broadcast_dims` attribute
- If any broadcast dimension matches the current navigation dimension:
  - **Bypass spatial index entirely**
  - **Load ALL points** for this group (return full range `[0, totalPoints]`)
- This simple approach works because broadcast groups are typically small (markers, labels)

*Use Cases*:
- Reference markers that should be visible regardless of time/channel
- Axis labels or coordinate guides
- Background structures that don't change over time
- Reducing storage for data that's constant across dimensions

*Performance Note*:
- Broadcast groups should be **small** (hundreds to thousands of points)
- Large broadcast groups will be loaded entirely on every frame
- For large static data, consider writing separate groups per slice instead

*Validation*:
- Dimension names must exist in scene dimensions
- Only non-displayed dimensions can be broadcast (broadcasting displayed dimensions is meaningless)

---

### 6. Lines Node

**Specification**:
- Data node for line/curve visualization
- Represents connected sequences of vertices with interpolated attributes
- Supports multiple connectivity types: segments, polylines, closed loops, indexed
- Rendered as thick, anti-aliased tubes/ribbons with polynomial edge falloff

**Data Arrays**:

| Array | Shape | Dtype | Required | Description |
|-------|-------|-------|----------|-------------|
| vertices | (N, d) | float32 | Yes | Vertex positions in d dimensions |
| colors | (N, 3) or (1, 3) | float32/uint8 | No | Per-vertex RGB colors (interpolated along segments) |
| widths | (N,) or (1,) | float32 | Yes | Per-vertex line thickness in scene units (interpolated along segments) |
| sharpness | (N,) or (1,) | float32 | No | Per-vertex edge softness (interpolated along segments) |
| indices | (M*2,) | uint32 | Conditional | Flat array of vertex indices for `indexed` type only |

**Understanding Vertices and Indices**:

The `vertices` array stores coordinate data:
```python
vertices = [
    [x0, y0, z0],   # vertex 0
    [x1, y1, z1],   # vertex 1
    [x2, y2, z2],   # vertex 2
    [x3, y3, z3],   # vertex 3
]
```

Segments are defined by pairs of indices into the vertices array. For example, segment `(0, 1)` means "draw a line from `vertices[0]` to `vertices[1]`".

**Line Types and Connectivity**:

| Type | `line_type` Value | Index Array | Segment Generation |
|------|-------------------|-------------|-------------------|
| Segments | `"segments"` | None (implicit) | Vertex pairs: (0,1), (2,3), (4,5)... |
| Polyline | `"polyline"` | None (implicit) | Consecutive: (0,1), (1,2), (2,3)... |
| Loop | `"loop"` | None (implicit) | Consecutive + wrap: (0,1), (1,2)...(N-1,0) |
| Indexed | `"indexed"` | Required | Explicit: (indices[0],indices[1]), (indices[2],indices[3])... |

**Examples by Line Type**:

```python
# SEGMENTS: 6 vertices → 3 independent segments
vertices = [[0,0], [1,0], [2,0], [2,1], [3,0], [3,1]]  # N=6
line_type = "segments"
# Segments: (0,1), (2,3), (4,5) - pairs of vertices

# POLYLINE: 4 vertices → 3 connected segments
vertices = [[0,0], [1,0], [1,1], [0,1]]  # N=4
line_type = "polyline"
# Segments: (0,1), (1,2), (2,3) - consecutive vertices

# LOOP: 4 vertices → 4 segments (closed)
vertices = [[0,0], [1,0], [1,1], [0,1]]  # N=4
line_type = "loop"
# Segments: (0,1), (1,2), (2,3), (3,0) - closes back to start

# INDEXED: arbitrary connectivity with vertex reuse
vertices = [[0,0], [1,0], [0.5,0.866]]  # N=3 (triangle corners)
indices = [0, 1, 1, 2, 2, 0]  # M*2=6 → 3 segments
line_type = "indexed"
# Segments: (0,1), (1,2), (2,0) - forms a triangle
```

**Attribute Interpolation**:

All per-vertex attributes (colors, widths, sharpness) are **linearly interpolated** along each segment:
- At segment start: use start vertex's attribute value
- At segment end: use end vertex's attribute value
- Along segment: linear blend between start and end values

This enables smooth color gradients, tapered lines, and varying edge softness.

**Metadata**:
```python
{
    "n_vertices": int,      # Number of vertices (N)
    "n_segments": int,      # Number of line segments
    "ndim": int,            # Dimensionality (d)
    "line_type": str,       # "segments", "polyline", "loop", "indexed"
    "has_colors": bool,
    "has_sharpness": bool,
    "max_width": float,     # Maximum width (for rendering bounds)
}
```

**Computed Metadata**:
- `max_width`: `max(widths)` - the maximum value in the widths array. For broadcast widths `(1,)`, this is that single value.
- `n_segments`: computed from `n_vertices` and `line_type` per the segment count formula below.

**Sharpness Interpolation**:
Sharpness is linearly interpolated along each segment, affecting the cross-sectional falloff at each point along the line. The interpolated sharpness value at position `t` along segment `(v0, v1)` is:
```
sharpness_t = sharpness[v0] * (1-t) + sharpness[v1] * t
```

**Default Values** (when optional arrays not provided):
- colors: white `[1.0, 1.0, 1.0]`
- sharpness: `2.0` (quadratic polynomial falloff)

**Segment Count Formula** (integer division):
- `segments`: N // 2 (must have even N)
- `polyline`: N - 1
- `loop`: N
- `indexed`: len(indices) // 2

**Validation Rules**:
- `line_type` must be one of: `"segments"`, `"polyline"`, `"loop"`, `"indexed"`
- `widths` must be positive (> 0)
- `sharpness` must be in [0, 31] range
- Minimum vertex counts:
  - `segments`: N ≥ 2 (at least one segment)
  - `polyline`: N ≥ 2 (at least one segment)
  - `loop`: N ≥ 3 (minimum closed shape is a triangle)
  - `indexed`: N ≥ 2 (at least two vertices to form a segment)
- For `segments` type: N must be even (pairs of vertices)
- For `indexed` type: `indices` array is required and must have len ≥ 2
- For `indexed` type: `len(indices)` must be even (pairs form segments)
- For `indexed` type: all index values must be < N (valid vertex references)
- For non-`indexed` types: `indices` array must not be present
- Empty Lines (N=0) are NOT valid for finalized nodes (see Points validation for streaming exception)
- **Zero segments is always an error**: Lines must have at least one segment. For `indexed` type, this means `len(indices) >= 2`. For other types, this means sufficient vertices per the segment count formula.
- All validation failures raise `ValueError`

**Encoding**: Uses `luxar.encoding` with semantic types:
- vertices: COORDINATE
- colors: COLOR
- widths: POSITIVE_SCALAR
- sharpness: BOUNDED_SCALAR [0, 31]
- indices: INDEX (adaptive bit-depth: uint8/uint16/uint32 based on max vertex count)

**Storage Order**:
Lines are stored in their original order (NOT Morton-sorted). The connectivity constraints (segments, polylines, loops, indexed) make spatial reordering non-trivial.

**Spatial Index**:
**Lines do NOT support spatial indexing** in the current version. A different indexing approach is needed due to:
- Segments span between two vertices (can't assign to single cell)
- Indexed lines have complex vertex sharing
- Polylines/loops must maintain vertex sequence

This is a known limitation - spatial indexing for lines will be addressed in a future version.

---

### 7. GSplats Node

**Specification**:
- Data node for Gaussian splat visualization
- Each splat is an oriented ellipsoid defined by center, covariance, color, amplitude
- Supports RGB colors with scalar amplitude multiplier
- Full storage specification in `gsplats/io/SPECIFICATIONS.md`

**Data Arrays**:

| Array | Shape | Dtype | Required | Description |
|-------|-------|-------|----------|-------------|
| centers | (N, d) | float32 | Yes | Splat centers (not broadcastable) |
| amplitudes | (N,) or (1,) | float32 | Yes | Non-negative intensity values (scalar multiplier) |
| cholesky_factors | (N, k) or (1, k) | float32 | Yes | Packed Cholesky factors, k = d*(d+1)/2 |
| colors | (N, 3) or (1, 3) | float32/uint8 | No | RGB colors (HDR or SDR) |
| sharpness | (N,) or (1,) | float32 | No | Generalized Gaussian exponent [0, 31] |

**Note on Colors and Amplitudes**: The final rendered color is `color * amplitude`. When `colors` is not provided, white `[1.0, 1.0, 1.0]` is used, so amplitudes alone determine grayscale intensity.

**Sharpness Parameter** (GSplats):
GSplats use a **generalized Gaussian** formula, different from Points/Lines:
```
I(x) = amplitude * exp(-0.5 * ||y||^s)
```
Where s is the sharpness exponent and y is the Mahalanobis-transformed coordinate.
- **s = 2**: Standard Gaussian (default)
- **s > 2**: Sharper edges, faster decay

This is NOT the same as the polynomial falloff used by Points/Lines.

**Note on Effective Bounds**: The `gsplats.fitting` package uses an internal parameterization `s = 2 * exp(s')` where `s'` is clamped to `[-2.5, 2.5]`, producing effective bounds of `s ∈ [0.164, 24.47]`. This is narrower than the full [0, 31] range but is sufficient for practical use cases. The narrower range is a consequence of the parameterization chosen for numerical stability during optimization.

**Metadata**:
```python
{
    "n_splats": int,         # Number of splats
    "ndim": int,             # Dimensionality (d)
    "has_colors": bool,      # Only for optional arrays
    "has_sharpness": bool,   # Only for optional arrays
    "ordering": str,         # "none", "morton", "hilbert"
    "amplitude_range": {"min": float, "max": float},
    "center_bounds": {"min": [...], "max": [...]},
}
```

**Note**: `has_colors` and `has_sharpness` track optional arrays only. Required arrays (`centers`, `amplitudes`, `cholesky_factors`) are always present and don't need tracking flags. Sharpness bounds [0, 31] are stored in the encoding metadata (BOUNDED_SCALAR), not duplicated in node metadata.

**Default Values** (for optional arrays only):
- colors: white `[1.0, 1.0, 1.0]`
- sharpness: `2.0` (standard Gaussian, s=2)

**Validation Rules**:
- N ≥ 1 (at least one splat)
- Empty GSplats (N=0) are NOT valid for finalized nodes (see Points validation for streaming exception)
- `centers` array must have shape `(N, d)` - NOT broadcastable (unlike other arrays). Rationale: Broadcasting centers would place all N splats at the same location, which is degenerate and not meaningful for visualization. Each splat must have a distinct center position.
- `amplitudes` must be non-negative (>= 0)
- `sharpness` must be in [0, 31] range
- `cholesky_factors` diagonal elements must be positive (> 0) for valid positive-definite covariance. The packed diagonal elements are at indices 0, 2, 5, 9, ... (triangular numbers).
- All validation failures raise `ValueError`

**Cholesky Packing**:
Lower-triangular matrix packed row-major:
- 2D (k=3): `[L00, L10, L11]`
- 3D (k=6): `[L00, L10, L11, L20, L21, L22]`
- 4D (k=10): `[L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]`

**Encoding**: Uses `luxar.encoding` with semantic types:
- centers: COORDINATE
- colors: COLOR
- amplitudes: POSITIVE_SCALAR
- cholesky_factors: CHOLESKY
- sharpness: BOUNDED_SCALAR [0, 31]

**Storage Order** (Morton/Z-Order):
All GSplat arrays are stored sorted by Morton code (same as Points):
1. Center coordinates normalized to integer range based on data bounds
2. Morton code computed by bit-interleaving normalized coordinates
3. All arrays (centers, colors, amplitudes, cholesky_factors, sharpness) sorted by Morton code

**Spatial Index** (`chunk_bounds`):
GSplats support chunk-level spatial indexing:
- `chunk_bounds`: (num_chunks, n_dims, 2) float32 array
- Each chunk's axis-aligned bounding box, **including ellipsoid extent**
- Ellipsoid extent computed from Cholesky factors: `extent[d] = sqrt(covariance[d,d]) * 3.0`
- The factor 3.0 corresponds to ~99.7% coverage (3 standard deviations)
- Bounds calculation: `min(center - extent)`, `max(center + extent)` per chunk

**Reference**: See `gsplats/io/SPECIFICATIONS.md` for complete storage format specification.

---

### 8. Dimension System

**Specification**:

#### Dimension (Single Dimension)
**Fields**:
- `name` (str, required) - Dimension identifier
- `unit` (str) - Physical unit (e.g., "um", "s", "px")
- `range` (tuple[float, float] | None) - Optional bounds
- `step` (float | None) - Navigation step size (auto if None)
- `display` (bool, default True) - Whether dimension is visualized (max 3)
- `discrete` (bool, default False) - Whether values are integer steps
- `categories` (List[str] | None, default None) - Category labels for categorical dimensions
- `cyclic` (bool, default False) - Whether dimension wraps around
- `scale` (float, default 1.0) - Physical scale factor
- `spatial` (bool | None) - Whether elements extend through dimension (auto if None)
- `description` (str) - Optional human-readable description

**Dimension Types**:
- **Continuous**: Standard dimension with any numeric value (e.g., spatial coordinates)
- **Discrete (numeric)**: Integer-stepped dimension where values are meaningful numbers (e.g., time=0,1,2)
- **Categorical**: Dimension where values are labels, not numbers (e.g., channel=["DAPI","GFP","mCherry"])

Categorical dimensions are discrete dimensions with string labels. Values are stored as integer indices into the `categories` list:
- Index 0 → categories[0]
- Index 1 → categories[1]
- etc.

**Auto-behaviors** (applied in order during `__post_init__`):

0. **Categorical dimension handling** (if `categories` is not None):
   ```
   if categories is not None:
       discrete = True  # Categories are inherently discrete
       if range is None:
           range = (0, len(categories) - 1)  # Implicit range
       step = 1.0  # Always step by one category
   ```

   **Rationale**: Categorical dimensions are a special case of discrete dimensions where values are indices into a label list. The range and step are fully determined by the categories list.

1. **Spatial flag auto-determination** (if `spatial` is None):
   ```
   if display == True:
       spatial = True   # Displayed dimensions are always spatial
   else:
       spatial = False  # Non-displayed dimensions are never spatial
   ```

   **Rationale**: Displayed dimensions define the visible coordinate space, so elements naturally extend through them. Non-displayed dimensions are navigated via slicing, so elements exist at discrete positions within them.

2. **Discrete auto-correction** (if `spatial=False` and `display=False`):
   ```
   if discrete == False:
       discrete = True  # With warning
   ```

   **Rationale**: A non-spatial, non-displayed dimension represents categorical data (time points, channels, etc.) where elements exist at specific values, not continuously through the dimension.

3. **Step size auto-calculation** (if `step` is None):
   ```
   if discrete:
       step = 1.0       # Discrete dimensions step by 1
   elif range is not None:
       step = (range[1] - range[0]) * 0.01  # 1% of range
   else:
       step = 0.1       # Fallback default
   ```

**Flag Combinations and Meanings**:

| display | spatial | discrete | categories | Meaning | Example |
|---------|---------|----------|------------|---------|---------|
| True | True | False | None | Visible, continuous spatial axis | x, y, z |
| True | True | True | None | Visible, discrete spatial axis | quantized z |
| False | False | True | None | Navigated dimension with discrete numeric values | time (0,1,2...) |
| False | False | True | [...] | Navigated categorical dimension | channel=["DAPI","GFP"] |
| False | True | False | None | Non-displayed spatial dimension (smooth slicing) | depth, z-stack |

Note: `display=False` + `spatial=True` requires explicit `spatial=True` - auto-determination sets `spatial=False` for non-displayed dimensions.

**RESOLVED - Explicit spatial=True with display=False**:
This configuration IS valid and useful. Use cases:
- **Z-stack slicing**: Microscopy data where Z is not displayed but elements extend through it
- **Continuous depth**: 4D data where the 4th dimension represents physical depth
- **Smooth transitions**: Elements fade in/out smoothly as you navigate (hypersphere intersection)

When `display=False, spatial=True, discrete=False`:
- Elements extend as hyperspheres through this dimension
- Visibility determined by `R_effective = √(R² - D²)` formula
- Query tolerance = max_radius (not exact match)
- NOT included in `slice_dims` for compound ordering (treated as Morton-sorted dimension)

**Validation Rules**:
- range[0] < range[1]
- step > 0 (if specified)
- scale > 0
- Cannot be discrete AND spatial unless displayed (raises `ValueError`)
- If `categories` provided:
  - Must have at least 1 element
  - All category names must be non-empty strings
  - Category names must be unique (no duplicates)
  - Point coordinate values must be valid indices: `0 ≤ value < len(categories)`

#### Dimensions (Collection)
**Specification**:
- Container for list of Dimension objects
- Empty Dimensions collection is valid (no dimensional constraints on data)
- Maximum 3 displayed dimensions
- If the collection is non-empty, at least one dimension must have `display=True`
- Dimension names must be unique

**Key Properties**:
- `ndim` - Total number of dimensions
- `names` - List of dimension names
- `displayed` - Indices of displayed dimensions
- `non_displayed` - Indices of non-displayed dimensions
- `spatial_extend_dims` - Boolean list indicating spatial extension per dimension

---

### 9. nD Slicing Visual Semantics

When navigating through non-displayed dimensions, the viewer must determine which elements are visible. This section specifies the intended visual behavior.

#### Dimension Types and Visibility

| Dimension Type | Visibility Rule | Example |
|----------------|-----------------|---------|
| Displayed (spatial) | Always visible if in view frustum | x, y, z |
| Non-displayed, spatial | Visible if hypersphere intersects slice plane | continuous depth |
| Non-displayed, discrete | Visible only at exact value match | time, channel |

**Key Principle**: Non-spatial dimensions are always discrete by design (auto-corrected in `Dimension.__post_init__`).

#### Hypersphere Intersection for Spatial Dimensions

For non-displayed spatial dimensions, elements (points, splats) extend as hyperspheres through those dimensions. The viewer computes visibility using the Pythagorean theorem:

**Formula**:
```
Given:
- R = element radius
- D = distance from element center to current slice position (in non-displayed spatial dimensions only)

Effective radius at slice:
R_effective = √(R² - D²)   if D < R
            = 0             if D ≥ R (element not visible)
```

**Algorithm** (per element):
```python
# Calculate distance only in non-displayed SPATIAL dimensions
distance_squared = 0
for d in non_displayed_dimensions:
    if dimensions[d].spatial:  # Only spatial dimensions contribute
        delta = element_position[d] - current_slice[d]
        distance_squared += delta * delta

# Apply Pythagorean theorem
if distance_squared < radius**2:
    effective_radius = sqrt(radius**2 - distance_squared)
    # Element is visible with effective_radius
else:
    # Element is not visible at this slice
```

**Visual Result**: As you navigate through a spatial dimension, spheres appear to grow from zero radius, reach their full size when the slice passes through their center, then shrink back to zero.

#### Discrete Dimension Matching

For non-displayed discrete dimensions (time, channel, category), elements are only visible when the slice position exactly matches their value:

```python
for d in non_displayed_dimensions:
    if dimensions[d].discrete:
        if element_position[d] != current_slice[d]:
            # Element is not visible - dimension value doesn't match
            return False
```

**Visual Result**: As you navigate through time or channels, elements instantly appear/disappear rather than smoothly transitioning.

#### Spatial Query Tolerance

When querying the spatial index for visible elements, the tolerance per dimension depends on dimension type:

| Dimension | Query Tolerance |
|-----------|-----------------|
| Displayed | 0 (query all, frustum culling later) |
| Non-displayed, spatial | max_radius (elements may extend into view) |
| Non-displayed, discrete | 0 (exact match only) |

**Storage**: The `spatial_extend_dims` boolean array (stored as zarr attribute) indicates which dimensions are spatial for efficient query tolerance calculation.

#### Example: 6D Dataset (X, Y, Z, Time, Channel, Depth)

```python
dimensions = [
    Dimension("X", display=True),              # Displayed, spatial
    Dimension("Y", display=True),              # Displayed, spatial
    Dimension("Z", display=True),              # Displayed, spatial
    Dimension("Time", display=False, discrete=True),  # Non-displayed, discrete numeric
    Dimension("Channel", display=False,               # Non-displayed, categorical
              categories=["DAPI", "GFP", "mCherry"]),
    Dimension("Depth", display=False, spatial=True),  # Non-displayed, spatial
]
```

At slice position `[*, *, *, time=5, channel=1, depth=10.0]`:
- X, Y, Z: All in view are visible (standard 3D rendering)
- Time: Only elements with `time == 5` are visible (exact match)
- Channel: Only elements with `channel == 1` (i.e., "GFP") are visible (exact match)
- Depth: Elements with `|depth - 10.0| < radius` are visible (hypersphere intersection)

**Viewer Navigation for Categorical Dimensions**:
- Keyboard navigation ([ / ]) steps through categories
- **Cyclic behavior respects the `cyclic` flag**:
  - If `cyclic=True`: At last category, pressing ] wraps to first category; at first, pressing [ wraps to last
  - If `cyclic=False` (default): Navigation stops at boundaries (first/last category)
- UI shows category label (e.g., "Channel: GFP") not index ("Channel: 1")
- UI shows dropdown/selector instead of slider for category selection

**Example with cyclic**:
```python
# Cyclic categorical (e.g., angles, periodic states)
Dimension("Phase", categories=["G1", "S", "G2", "M"], cyclic=True)  # Wraps: M → G1

# Non-cyclic categorical (default)
Dimension("Channel", categories=["DAPI", "GFP", "mCherry"])  # Stops at ends
```

---

### 10. Transformation System

**Specification**:

#### Transform Representation
- All transforms are 4x4 homogeneous matrices (float32)
- Stored in column-major order for THREE.js compatibility
- NumPy uses row-major, storage uses column-major
- Translation components at indices [0,3], [1,3], [2,3] in row-major
- Translation components at indices [12], [13], [14] in column-major (storage)

#### Coordinate System Conventions
Luxar follows THREE.js conventions for consistency with the WebGL viewer:
- **Right-handed coordinate system**
- **Y-axis is up** by default
- **Positive rotation**: Counter-clockwise when looking down the positive axis toward the origin (right-hand rule)
- **Camera looks down -Z**: Standard OpenGL/WebGL convention

These conventions ensure that transforms created in Python render correctly in the THREE.js viewer without additional conversion.

#### nD Limitation
**Transforms only apply to the 3 displayed dimensions**. For nD data (d > 3):
- Transforms affect only the displayed dimensions (whichever 3 are currently visualized)
- Non-displayed dimensions are unaffected by transforms
- This is a fundamental limitation of 4x4 homogeneous matrices

This design choice keeps transforms simple and compatible with standard 3D graphics. Higher-dimensional transforms would require (d+1)×(d+1) matrices and significantly more complexity.

#### Storage Conversion
**NumPy (row-major) → Storage (column-major)**:
```
storage_list = numpy_matrix.T.ravel().tolist()
```

**Storage (column-major) → NumPy (row-major)**:
```
numpy_matrix = np.array(storage_list).reshape(4, 4).T
```

#### Transform Functions

**Basic Transforms**:
- `identity()` - 4x4 identity matrix
- `translate(x, y, z)` - Translation matrix
- `scale(x, y, z, uniform=None)` - Scaling matrix (uniform overrides x,y,z)
- `rotate_x/y/z(degrees)` - Rotation around principal axes
- `rotate(degrees, axis)` - Rotation around arbitrary axis (uses Rodrigues' formula)

**Composition**:
- `compose(T1, T2, T3, ...)` - Combines transforms in application order
- Transforms are applied left-to-right: T1 first, then T2, then T3

**Mathematical Formula**:
Given `compose(T1, T2, T3)`, the resulting matrix M satisfies:
```
M = T3 @ T2 @ T1

When applied to a point p:
M @ p = T3 @ (T2 @ (T1 @ p))
```

The matrix multiplication is right-to-left, but transforms apply left-to-right to points.

**Concrete Example**:
```python
# Goal: Move object to (5,0,0), then rotate it 90° around Z
T1 = translate(5, 0, 0)   # First: translate
T2 = rotate_z(90)          # Second: rotate

combined = compose(T1, T2)
# Internally: combined = T2 @ T1

# Applied to origin point [0, 0, 0]:
# Step 1: T1 @ [0,0,0] → [5, 0, 0]  (translated)
# Step 2: T2 @ [5,0,0] → [0, 5, 0]  (rotated around origin)

# Final position: [0, 5, 0]
```

**Implementation Algorithm**:
```python
def compose(*transforms):
    result = identity()
    for T in reversed(transforms):  # Iterate [T3, T2, T1]
        result = result @ T          # Right-multiply
    return result
    # Result: I @ T3 @ T2 @ T1 = T3 @ T2 @ T1
```

**Utilities**:
- `inverse(T)` - Matrix inverse using np.linalg.inv()
- `look_at(eye, target, up)` - Camera-style transformation
- `to_list(T)` - Convert to storage format (transposes automatically)
- `from_list(L)` - Convert from storage format (transposes automatically)

---

## Memory Efficiency Strategy

**Core Principle**: NEVER keep large data in memory

**Implementation**:
1. Nodes are lightweight metadata containers
2. Data arrays written immediately to Zarr via writer interface
3. Only metadata returned (n_elements, has_colors, etc.)
4. No Zarr groups kept in memory after writing
5. Attribute cache maintains fast access to metadata

**Data Flow**:
```
User provides data → Node validates → Writer writes to Zarr → Metadata returned → Node stores metadata
```

---

## Rendering Attributes

**Valid Ranges**:
- `opacity`: 0.0 to 1.0 (float)
- `gamma`: 0.1 to 10.0 (float)
- `blending_mode`: "normal" | "additive" (string)

**Validation**: All values validated on assignment, invalid values raise ValueError

**Hierarchical Composition**: Rendering attributes compose (multiply) through the hierarchy, they do NOT override:
- `effective_opacity = clamp(parent_opacity × child_opacity, 0.0, 1.0)`
- `effective_gamma = clamp(parent_gamma × child_gamma, 0.1, 10.0)`
- `blending_mode`: Child inherits parent's mode if not explicitly set; if set, child's mode is used

Example: If parent has `opacity=0.5` and child has `opacity=0.5`, the effective opacity is `0.25`.

**Clamping**: Effective values are clamped to valid ranges after composition to prevent invalid states.

**No Transform = Identity**: Nodes without an explicit transform use the identity matrix (no transformation).

**Default Values**:
- opacity: 1.0
- gamma: 1.0
- blending_mode: "additive"

---

## nD Representation

**Specification**:

All data nodes can have arbitrary dimensions (not limited to 3D).

**Visualization Strategy**:
- Maximum 3 dimensions can be displayed simultaneously
- Non-displayed dimensions are "sliced"
- Elements visible when within tolerance of current slice position
- Tolerance determined by element size (radius for Points, width for Lines) for spatial dimensions
- GSplats tolerance: implementation-specific (derived from Cholesky factors) - deferred for future specification
- Exact match required for discrete dimensions

**Spatial Extension**:
- Spatial dimensions: Elements treated as extended objects through dimension
- Non-spatial dimensions: Elements exist at specific discrete values only
- Example: Time dimension typically non-spatial (points at t=5, not spreading through time)
- Example: Z dimension typically spatial (points as 3D objects extending through depth)

**Broadcasting**:
- Data nodes can be broadcast to all values of specified dimensions
- Useful for static objects that appear across all time points/channels
- Applies to all data node types (Points, Lines, GSplats)

---

## Integration with luxar.encoding

Data arrays are stored using the `luxar.encoding` infrastructure:

**Encoding Modes**:
- `AUTO`: Automatic selection (may be lossy)
- `PRECISION`: Full float32 precision
- `MEMORY`: Quantized for minimal storage
- `CUSTOM`: Explicit dtype specification

**Semantic Types Used**:

| Semantic Type | Used By | Bounds | Description |
|---------------|---------|--------|-------------|
| COORDINATE | positions, vertices, centers | unbounded | Spatial coordinates |
| COLOR | colors | [0, 1] SDR or unbounded HDR | RGB values |
| POSITIVE_SCALAR | radii, widths, amplitudes | [0, ∞) | Non-negative values |
| BOUNDED_SCALAR | sharpness | [0, 31] | Edge sharpness (all node types) |
| CHOLESKY | cholesky_factors | unbounded | Covariance decomposition |
| INDEX | indices | [0, N) | Integer references (adaptive uint8/16/32) |

**Two-Layer Validation Note**:
The `POSITIVE_SCALAR` semantic type accepts values >= 0 at the encoding layer. However, the core layer applies stricter validation for specific arrays:
- `radii` must be > 0 (strictly positive) - zero radius is degenerate
- `widths` must be > 0 (strictly positive) - zero width is invisible
- `amplitudes` can be >= 0 (non-negative) - zero amplitude = invisible splat, valid use case

**Broadcasting Support**:
Arrays with shape `(1,)` or `(1, k)` are broadcast to all elements:
- Scalars (radii, widths, sharpness, amplitudes): `(1,)` broadcasts to `(N,)`
- Colors: `(1, 3)` broadcasts to `(N, 3)`
- Cholesky factors: `(1, k)` broadcasts to `(N, k)` where k = d*(d+1)/2

Metadata: `{"encoding": {"name": "broadcasted", "n_elements": N}}`

---

## Writer Protocol

The `ZarrWriterProtocol` defines the interface for progressive writing:

```python
class ZarrWriterProtocol(Protocol):
    @property
    def store_path(self) -> str:
        """Path to the underlying Zarr store."""
        ...

    def write_group(self, path, **attrs) -> None: ...

    def write_points(
        self, path, positions, radii,
        colors=None, sharpness=None,
        **attrs
    ) -> PointsMetadata: ...

    def write_lines(
        self, path, vertices, widths,
        colors=None, sharpness=None, indices=None,
        line_type="polyline",
        **attrs
    ) -> LinesMetadata: ...

    def write_gsplats(
        self, path, centers, amplitudes, cholesky_factors,
        colors=None, sharpness=None,
        **attrs
    ) -> GSplatsMetadata: ...

    def create_resizable_dataset(
        self, path, dtype, shape, maxshape=None, chunks=True
    ) -> "zarr.Array":
        """Create a resizable dataset for streaming writes."""
        ...

    def finalize(self) -> None: ...
```

Each `write_*` method:
1. Validates input arrays
2. Applies encoding based on semantic types
3. Writes immediately to Zarr
4. Returns only metadata (not the data)

**Metadata Type Aliases**:
```python
PointsMetadata = Dict[str, Any]   # n_points, ndim, has_colors, has_sharpness, max_radius
LinesMetadata = Dict[str, Any]    # n_vertices, n_segments, ndim, line_type, has_colors, has_sharpness, max_width
GSplatsMetadata = Dict[str, Any]  # n_splats, ndim, has_colors, has_sharpness, ordering, amplitude_range, center_bounds
```

---

## Dependencies Between Core Components

```
Scene (root container)
  └─ requires: Writer (for progressive writing)
  └─ optional: Dimensions (for nD specification)
  └─ creates: Group instances (containers)
  └─ creates: DataNode instances (Points, Lines, GSplats)

Node (base class)
  └─ has: children (list of Node)
  └─ has: parent (reference to parent Node)
  └─ has: transform (optional 4x4 matrix)
  └─ has: rendering attrs (opacity, gamma, blending_mode)

Group (container, extends Node)
  └─ type: "group"
  └─ children: Group or DataNode instances

DataNode (abstract base class, extends Node)
  └─ has: n_elements (count of primary data)
  └─ has: ndim (dimensionality)
  └─ has: type-specific metadata
  └─ stores: NO actual data (written immediately to Zarr)

Points (extends DataNode)
  └─ type: "points"
  └─ n_elements: n_points
  └─ metadata: n_points, ndim, has_colors, has_sharpness, max_radius

Lines (extends DataNode)
  └─ type: "lines"
  └─ n_elements: n_vertices
  └─ metadata: n_vertices, n_segments, ndim, line_type, has_colors, has_sharpness, max_width

GSplats (extends DataNode)
  └─ type: "gsplats"
  └─ n_elements: n_splats
  └─ metadata: n_splats, ndim, has_colors, has_sharpness, ordering, amplitude_range, center_bounds

Dimensions
  └─ contains: list of Dimension
  └─ validates: position arrays
  └─ provides: displayed/non-displayed indices
```

---

## Extensibility

The system is designed for reasonable extensibility:

**Adding a New Data Node Type**:
1. Define the data arrays (required and optional)
2. Define the metadata structure
3. Map arrays to semantic types from `luxar.encoding`
4. Add `write_*` method to `ZarrWriterProtocol`
5. Add `add_*` method to `Scene`
6. Add node type to `NodeType` enum
7. Implement viewer support

**Current Node Types**:
- Points: Point clouds (soft-edged spheres)
- Lines: Line segments, polylines, loops
- GSplats: Oriented Gaussian ellipsoids

**Potential Future Types** (not yet implemented):
- Meshes: Triangle/polygon meshes
- Volumes: Voxel/volumetric data
- Arrows: Directed vectors
- Text: 3D labels

**Future API Extensions** (planned):
- **Node Deletion**: `remove_child(name)` method to remove nodes from hierarchy. Requires careful handling of Zarr group cleanup.
- **Scene Loading**: `Scene.load(path)` to load an existing scene from Zarr for reading/modification. Enables round-trip testing and scene editing workflows.
- **Node Moving**: `move_to(new_parent)` to reorganize hierarchy without re-writing data.

These features depend on implementing scene loading from Zarr, which is a prerequisite for any modification operations.

---

## Critical Implementation Notes

### 1. Constructor Initialization Order
When subclass and parent both initialize the same attribute:
- CORRECT: Call super().__init__() FIRST, then set subclass attributes
- WRONG: Set attributes first, then call super().__init__() (parent overwrites)

Example (Points):
```python
# CORRECT:
def __init__(self, metadata=None):
    super().__init__(...)  # Parent initializes _metadata = {}
    self._metadata = metadata or {}  # Then override with actual data

# WRONG:
def __init__(self, metadata=None):
    self._metadata = metadata or {}  # Set first
    super().__init__(...)  # Parent OVERWRITES with {}
```

### 2. Transform Composition Order
Matrix multiplication is right-to-left:
- To apply T1, then T2: result = T2 @ T1
- To build from list [T1, T2, T3]: iterate reversed, right-multiply

```python
# CORRECT:
result = identity()
for transform in reversed([T1, T2, T3]):  # iterates [T3, T2, T1]
    result = result @ transform  # right-multiply

# Step by step:
#   result = I @ T3 = T3
#   result = T3 @ T2
#   result = T3 @ T2 @ T1
# Final: T3 @ T2 @ T1 (applies T1 first when multiplied with vector)
```

### 3. Dimension Spatial Flags
- `display=True` → automatically `spatial=True`
- `display=False` + `spatial=False` → automatically `discrete=True`
- This ensures consistency: non-spatial, non-displayed must be discrete

### 4. Node Type Attribute
Every node MUST have a `type` attribute for the viewer to determine rendering:
```python
# In Node subclass __init__:
super().__init__(name, parent=parent, writer=writer, type="points", **attrs)
```

---

## Testing Requirements

**Core Functionality Tests**:
- Scene creation with/without dimensions
- Hierarchical node creation (all types)
- Transform composition and application
- Dimension validation
- Metadata preservation for all data node types
- Rendering attribute validation

**Data Node Tests**:
- Points: positions, colors, radii, sharpness
- Lines: vertices, colors, widths, sharpness, indices, all line_type values
- GSplats: centers, cholesky packing, amplitude ranges, sharpness bounds

**Critical Bug Prevention**:
- Test compose() with non-commutative operations (rotate+translate, order matters)
- Test metadata is preserved for all data node types
- Test dimension auto-behaviors (spatial flag, discrete correction)
- Test transform storage round-trip (NumPy → storage → NumPy)
- Test node type discriminator is correctly set

---

## Format Compatibility

**Transform Storage**:
- Internal (NumPy): Row-major 4x4 matrix
- External (Zarr/THREE.js): Column-major 16-element list
- Conversion MUST transpose matrix

**Dimension Storage**:
- Serialized to dict via to_dict()
- All fields stored (including computed ones like spatial)
- Deserialized via from_dict()

**Dimension Serialization Format** (JSON in zarr `.zattrs`):
```json
// Continuous spatial dimension
{
  "name": "X",
  "unit": "um",
  "range": [0.0, 256.0],
  "step": 1.0,
  "display": true,
  "discrete": false,
  "categories": null,
  "cyclic": false,
  "scale": 1.0,
  "spatial": true,
  "description": ""
}

// Discrete numeric dimension
{
  "name": "Time",
  "unit": "",
  "range": [0, 100],
  "step": 1.0,
  "display": false,
  "discrete": true,
  "categories": null,
  "cyclic": false,
  "scale": 1.0,
  "spatial": false,
  "description": "Time point index"
}

// Categorical dimension
{
  "name": "Channel",
  "unit": "",
  "range": [0, 2],
  "step": 1.0,
  "display": false,
  "discrete": true,
  "categories": ["DAPI", "GFP", "mCherry"],
  "cyclic": false,
  "scale": 1.0,
  "spatial": false,
  "description": "Fluorescence channel"
}
```

**Categories Field**:
- `null` if not categorical
- Array of strings if categorical (0-indexed: first string = index 0)
- Must have at least 1 element
- All strings must be non-empty and unique

**Data Node Metadata**:
- Stored in Zarr group attributes
- Type-specific structure per node type
- `type` attribute always present

---

## Performance Considerations

**Memory**:
- Nodes are lightweight (< 1KB each)
- No data arrays kept in memory after writing
- Only metadata and attribute cache in memory

**Hierarchy Traversal**:
- Depth-first walk is generator (memory efficient)
- O(n) where n is number of nodes
- Each node yielded once

**Transform Composition**:
- Composition is O(k) where k is number of transforms in hierarchy depth
- Each composition is 4x4 matrix multiplication (64 FLOPs)
- **Memoization**: `world_transform` should be memoized (cached) per node
- Cache invalidation: When a node's `local_transform` changes, invalidate its `world_transform` cache and all descendants' caches
- This ensures repeated access to `world_transform` is O(1) after first computation

**Data Writing**:
- Immediate write to Zarr (no buffering)
- Encoding applied during write
- Compression via Blosc + zstd

---

This specification is sufficient to re-implement the core package in any language while maintaining compatibility with the Luxar ecosystem.

---

## Changelog

- **v0.10.2** (2025-11-28): Cyclic flag + categorical clarification
  - Clarified that `cyclic` flag controls navigation wrapping for categorical dimensions
  - `cyclic=True`: wraps at boundaries (M → G1)
  - `cyclic=False` (default): stops at boundaries
  - Added example showing cyclic vs non-cyclic categorical dimensions

- **v0.10.1** (2025-11-28): Dimension serialization format
  - Added explicit JSON examples for dimension serialization
  - Shows continuous, discrete numeric, and categorical dimension formats
  - Documents `categories` field serialization (null vs array of strings)

- **v0.10.0** (2025-11-28): Categorical dimensions
  - Added `categories` field to Dimension for labeled discrete values
  - Categorical dimensions store integer indices mapping to string labels
  - Auto-behaviors: categories implies discrete=True, auto-sets range and step
  - Validation: categories must be non-empty, unique, non-empty strings
  - Viewer navigation: categorical dimensions cycle (wrap around at ends)
  - Updated flag combinations table to include categories column
  - Updated 5D→6D example with categorical Channel dimension

- **v0.9.3** (2025-11-28): Empty nodes clarification
  - Clarified empty nodes (N=0) are invalid for finalized nodes only
  - Added exception for streaming writes (StreamingPoints starts empty)
  - Validation enforced at write/finalize time, not construction time

- **v0.9.2** (2025-11-28): GSplats I/O reference update
  - Updated GSplats storage format references to `gsplats/io/SPECIFICATIONS.md` (moved from `gsplats/GSPLATS_ZARR_FORMAT.md`)

- **v0.9.1** (2025-11-28): Cross-reference fix
  - Fixed: Related Specifications now correctly references `io/SPECIFICATIONS.md` (was `io/README.md`)

- **v0.9.0**: Broadcasting, nD slicing semantics, and dimension flag resolution
  - **broadcast_dims**: New API for Points to appear across all values of specified dimensions
    - `broadcast_dims=None` (default): No broadcasting
    - `broadcast_dims=["Time"]`: Broadcast across Time dimension
    - `broadcast_dims="all"`: Broadcast across ALL non-displayed dimensions
    - Clarified storage: Original position values stored unchanged, viewer ignores them
    - Clarified query: Viewer bypasses spatial index, loads all points for broadcast groups
    - Added performance note: Broadcast groups should be small (markers, labels)
  - **nD Slicing Visual Semantics**: New section documenting viewer behavior
    - Hypersphere intersection formula: `R_effective = √(R² - D²)`
    - Spatial dimensions: Smooth visibility based on hypersphere intersection
    - Discrete dimensions: Exact matching only
    - Spatial query tolerance specification per dimension type
  - **RESOLVED**: `display=False, spatial=True, discrete=False` is now explicitly valid
    - Use cases: Z-stack slicing, continuous depth, smooth transitions
    - Added to flag combinations table with "depth, z-stack" example
  - Section renumbered: Transformation System is now Section 10 (was 9)

- **v0.8.0**: Gamma and sharpness range updates
  - Updated gamma range from [0.2, 2.0] to [0.1, 10.0] (symmetric: gamma and 1/gamma have equal range)
  - Updated sharpness range from [0, 32] to [0, 31]
  - Added note about GSplats effective bounds [0.164, 24.47] due to internal parameterization
  - Fixed stale sharpness reference in v0.6.3 changelog entry

- **v0.7.0**: Morton ordering and chunk-based spatial indexing
  - **BREAKING**: Replaced grid-based spatial index with Morton ordering + chunk bounding boxes
  - Points: All arrays stored Morton-sorted, with `chunk_bounds` for spatial queries
  - GSplats: All arrays stored Morton-sorted, with `chunk_bounds` including ellipsoid extent
  - Lines: Explicitly documented as NOT supporting spatial indexing (connectivity constraints)
  - Morton codes: 64-bit, bits per dimension = 64 / n_dims
  - `chunk_bounds`: (num_chunks, n_dims, 2) float32 - AABB per chunk including element extent
  - Query algorithm: Simple AABB intersection test between chunk bounds and view/slice bounds
  - Removed: `occupied_cells`, `cell_ranges`, grid-based metadata

- **v0.6.3**: Validation rules and color mode
  - Renamed `luxar_version` to `luxar_format_version` with clarifying note (format version vs spec version)
  - Fixed path computation formula to handle empty root path correctly
  - Fixed "gaussian" terminology → "soft-edged spheres" for Points (polynomial falloff, not Gaussian)
  - **RESOLVED**: SDR vs HDR requires explicit `color_mode` flag - auto-detection rejected due to error-masking risk
  - **OPEN QUESTION**: Explicit `spatial=True` with `display=False` case - behavior may change
  - Added "Two-Layer Validation Note" explaining POSITIVE_SCALAR (>= 0) vs radii/widths (> 0)
  - Added `widths > 0` validation to Lines (was implicit, now explicit)
  - Added `sharpness` validation [0, 31] to Points, Lines, GSplats
  - Clarified transform composition memoization (cache `world_transform`, invalidate on change)
  - Clarified empty Groups warning triggers on `finalize()`

- **v0.6.2**: Clarifications and validation rules
  - Clarified transform hierarchy: distinguished `local_transform`, `parent_world_transform`, and `world_transform`
  - Rewrote dimension auto-behavior section with explicit code, rationale, and flag combination table
  - Clarified Scene can have a transform like any other Node
  - Added validation: `amplitudes` must be non-negative (>= 0)
  - Added validation: `cholesky_factors` diagonal elements must be positive (> 0)
  - Added documentation for `max_radius` and `max_width` computed metadata
  - Added validation: SDR float colors must be in [0, 1] range
  - Added Lines sharpness interpolation documentation
  - Fixed terminology: "polynomial edge falloff" for Points/Lines (not "gaussian")

- **v0.6.1**: Formula corrections and cleanup
  - **CRITICAL**: Fixed Points/Lines sharpness formula: polynomial `(1-r/radius)^s`, NOT Gaussian
  - **CRITICAL**: Fixed sharpness default to `2.0` (was incorrectly `1.0`)
  - Added explicit GSplats sharpness formula: generalized Gaussian `exp(-0.5 * ||y||^s)`
  - Fixed Writer Protocol: `amplitudes` and `cholesky_factors` now shown as required parameters
  - Removed `has_amplitudes` and `has_cholesky` from GSplats metadata (required arrays don't need tracking flags)
  - Standardized `sharpness` naming (singular) across all node types
  - Added gamma/opacity clamping after hierarchical composition
  - Clarified blending mode inheritance: child inherits parent's mode if not explicitly set

- **v0.6**: Specification consistency and clarifications
  - Standardized gamma range to 0.1-10.0 (was inconsistent)
  - Added sharpness parameter documentation (formula was incorrect, fixed in v0.6.1)
  - Fixed GSplats contradiction: `amplitudes` and `cholesky_factors` are now clearly required (removed conflicting defaults)
  - Simplified Type-Specific Properties table: "Required Arrays" and "Optional Arrays" columns
  - Added rationale for non-broadcastable `centers` in GSplats
  - Rewrote transform composition section with concrete example and clearer mathematical formula
  - Resolved Scene root path: `path = ""` (empty string), with rationale
  - Clarified rendering attribute composition: opacity and gamma **multiply** through hierarchy (not override)
  - Added coordinate system conventions section (Three.js compatible: right-handed, Y-up)
  - Added Group nesting rules: Groups can contain Groups, empty Groups raise warning
  - Added Future API Extensions section: node deletion, scene loading, node moving
  - Clarified duplicate node names raise `ValueError`
  - Clarified zero segments in Lines is always an error
  - Added Related Specifications cross-references at top of document

- **v0.5**: Required arrays, validation, and GSplats colors
  - Added `colors` array to GSplats (RGB, optional, default white) for consistency with Points/Lines
  - Made `radii` required for Points (no universal default)
  - Made `widths` required for Lines (no universal default)
  - Removed `has_radii` and `has_widths` from metadata (now always present)
  - Updated Writer Protocol signatures to show radii/widths as required positional args
  - Updated Scene factory methods to show required params (`add_points(name, positions, radii, ...)`)
  - Added node name validation rules (no `/`, unique among siblings)
  - Fixed transform composition code comment (was describing wrong iteration order)
  - Added `amplitude_range` and `center_bounds` to GSplats metadata summary and aliases
  - Updated Type-Specific Properties table: renamed "Required" to "Also Required", added clarifying note
  - Added validation: empty data nodes (N=0) are NOT valid
  - Added validation: indexed Lines must have len(indices) ≥ 2
  - Added validation: GSplats centers must be (N, d), NOT broadcastable
  - Added validation: all validation failures raise `ValueError`
  - Clarified cholesky_factors rendering default (isotropic splats if not provided)
  - Fixed JSON comments in code blocks (added note about documentation-only comments)
  - Fixed segment count formula to use integer division (`//`)
  - Added Open Question about Scene root path
  - Clarified nD tolerance is node-type specific (GSplats deferred)
  - Clarified Dimensions display requirement wording
  - Clarified empty Dimensions collection is valid
  - Clarified broadcasting shapes for colors `(1, 3)` and cholesky `(1, k)`

- **v0.4**: Critical review fixes
  - Fixed `dims` → `ndim` consistency in metadata aliases
  - Clarified transform nD limitation wording ("displayed dimensions" not "first 3")
  - Renumbered sections (removed "2b" awkwardness)
  - Added `store_path` property and `create_resizable_dataset` method to Writer Protocol
  - Added minimum vertex count validation rules for Lines
  - Clarified `indices` array must have even length
  - Removed redundant `sharpness_bounds` from GSplats metadata (stored in encoding)
  - Clarified radii/widths have no universal default (must be specified)

- **v0.3**: Extended node types
  - Added Lines node type with full specification
  - Added GSplats node type (referencing GSPLATS_ZARR_FORMAT.md)
  - Added Group node section
  - Added DataNode abstract base class
  - Standardized on `ndim` (not `dims`)
  - Unified sharpness bounds to [0, 32] across all node types
  - Added transform nD limitation documentation
  - Added validation rules for Lines

- **v0.2**: Initial version with Points only
