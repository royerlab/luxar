# luxar.core - Technical Specification

**Version**: 0.3
**Last Updated**: 2025-01-26

## Purpose

The `core` package defines the fundamental data structures and scene graph system for organizing and manipulating large-scale visualization data. It provides a hierarchical scene graph with transformation support, nD dimensional specifications, and extensible data node types (Points, Lines, GSplats).

---

## Node Type Taxonomy

The scene graph consists of two categories of nodes:

### Container Nodes
Nodes that organize the hierarchy but don't hold visualization data:
- **Scene**: Root container, holds dimensions and global settings
- **Group**: Intermediate container for organizing data nodes

### Data Nodes
Nodes that hold actual visualization data (positions, colors, etc.):
- **Points**: Point cloud data (spheres/gaussians at positions)
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
- Each node has rendering attributes: opacity (0-1), gamma (0.2-5.0), blending_mode
- Container nodes do NOT store array data - only metadata and references
- Nodes write data immediately through a writer interface (progressive writing)

**Invariants**:
- Parent-child relationships are bidirectional (parent knows children, children know parent)
- Root node has no parent (parent = None)
- Transforms compose hierarchically: `final_transform = parent_transform @ local_transform`
- Attribute changes are immediately persisted through writer interface

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

**Key Operations**:
- `add_group(name, **attrs)` - Create top-level group
- `add_points(name, positions, ...)` - Create Points node
- `add_lines(name, vertices, ...)` - Create Lines node
- `add_gsplats(name, centers, ...)` - Create GSplats node
- `dimensions` property - Get/set scene-level dimensional specifications

**Zarr Attributes**:
```json
{
  "type": "scene",
  "luxar_version": "0.3.0",
  "scene_dimensions": { ... }
}
```

---

### 2b. Group (Container Node)

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
```json
{
  "type": "group",
  "transform": [...],       // optional, 16-element column-major
  "opacity": 1.0,           // optional
  "gamma": 1.0,             // optional
  "blending_mode": "additive"  // optional
}
```

**Use Cases**:
- Organizing related data (e.g., "cells" group containing multiple Points)
- Applying shared transform to multiple children
- Applying shared rendering attributes

---

### 3. DataNode (Abstract Base Class)

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

**Note**: `path` is a Node attribute computed from the hierarchy (`parent.path + "/" + name`), not stored in the metadata dict. The metadata dict contains only data-specific information.

**Type-Specific Properties**:

| Node Type | Primary Data | Optional Data | Metadata Fields |
|-----------|--------------|---------------|-----------------|
| Points | positions (N, d) | colors, radii, sharpness | n_points, ndim, has_colors, has_radii, has_sharpness, max_radius |
| Lines | vertices (N, d) | colors, widths, sharpness, indices | n_vertices, n_segments, ndim, line_type, has_colors, has_widths, has_sharpness, max_width |
| GSplats | centers (N, d) | amplitudes, cholesky_factors, sharpnesses | n_splats, ndim, has_amplitudes, has_cholesky, has_sharpnesses, ordering |

**Constructor Order Invariant**:
When subclass and parent both initialize the same attribute:
- CORRECT: Call `super().__init__()` FIRST, then set subclass attributes
- WRONG: Set attributes first, then call `super().__init__()` (parent overwrites)

---

### 4. Points Node

**Specification**:
- Data node for point cloud visualization
- Each point is rendered as a sphere/gaussian at a position
- Inherits all Node capabilities

**Data Arrays**:

| Array | Shape | Dtype | Required | Description |
|-------|-------|-------|----------|-------------|
| positions | (N, d) | float32 | Yes | Point centers in d dimensions |
| colors | (N, 3) or (1, 3) | float32/uint8 | No | RGB colors (HDR or SDR) |
| radii | (N,) or (1,) | float32 | No | Point radii |
| sharpness | (N,) or (1,) | float32 | No | Edge sharpness (gaussian falloff) |

**Metadata**:
```python
{
    "n_points": int,        # Number of points
    "ndim": int,            # Dimensionality (d)
    "has_colors": bool,
    "has_radii": bool,
    "has_sharpness": bool,
    "max_radius": float,    # Maximum radius (for spatial queries)
}
```

**Default Values** (when optional arrays not provided):
- colors: white `[1.0, 1.0, 1.0]`
- radii: `0.1` (in scene units)
- sharpness: `1.0` (standard gaussian falloff)

**Encoding**: Uses `luxar.encoding` with semantic types:
- positions: COORDINATE
- colors: COLOR
- radii: POSITIVE_SCALAR
- sharpness: BOUNDED_SCALAR [0, 32]

---

### 5. Lines Node

**Specification**:
- Data node for line/curve visualization
- Represents connected sequences of vertices with interpolated attributes
- Supports multiple connectivity types: segments, polylines, closed loops, indexed
- Rendered as thick, anti-aliased tubes/ribbons with gaussian edge falloff

**Data Arrays**:

| Array | Shape | Dtype | Required | Description |
|-------|-------|-------|----------|-------------|
| vertices | (N, d) | float32 | Yes | Vertex positions in d dimensions |
| colors | (N, 3) or (1, 3) | float32/uint8 | No | Per-vertex RGB colors (interpolated along segments) |
| widths | (N,) or (1,) | float32 | No | Per-vertex line thickness (interpolated along segments) |
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
    "has_widths": bool,
    "has_sharpness": bool,
    "max_width": float,     # Maximum width (for rendering bounds)
}
```

**Default Values** (when optional arrays not provided):
- colors: white `[1.0, 1.0, 1.0]`
- widths: `0.1` (in scene units)
- sharpness: `1.0` (standard gaussian edge falloff)

**Segment Count Formula**:
- `segments`: N / 2 (must have even N)
- `polyline`: N - 1
- `loop`: N
- `indexed`: len(indices) / 2

**Validation Rules**:
- `line_type` must be one of: `"segments"`, `"polyline"`, `"loop"`, `"indexed"`
- For `segments` type: N must be even (pairs of vertices)
- For `indexed` type: `indices` array is required
- For `indexed` type: all index values must be < N (valid vertex references)
- For non-`indexed` types: `indices` array must not be present

**Encoding**: Uses `luxar.encoding` with semantic types:
- vertices: COORDINATE
- colors: COLOR
- widths: POSITIVE_SCALAR
- sharpness: BOUNDED_SCALAR [0, 32]
- indices: INDEX (adaptive bit-depth: uint8/uint16/uint32 based on max vertex count)

---

### 6. GSplats Node

**Specification**:
- Data node for Gaussian splat visualization
- Each splat is an oriented ellipsoid defined by center, covariance, amplitude
- **Single-channel intensity**: GSplats use amplitude (scalar) for intensity, not RGB colors
- Full specification in `gsplats/GSPLATS_ZARR_FORMAT.md`

**Data Arrays**:

| Array | Shape | Dtype | Required | Description |
|-------|-------|-------|----------|-------------|
| centers | (N, d) | float32 | Yes | Splat centers (not broadcastable) |
| amplitudes | (N,) or (1,) | float32 | No | Non-negative intensity values (single-channel, not RGB) |
| cholesky_factors | (N, k) or (1, k) | float32 | No | Packed Cholesky factors, k = d*(d+1)/2 |
| sharpnesses | (N,) or (1,) | float32 | No | Gaussian sharpness [0, 32] |

**Note on Colors**: Unlike Points and Lines, GSplats do not have RGB colors. The `amplitudes` array provides single-channel intensity. Color mapping (e.g., colormap lookup, channel assignment) is handled by the viewer or post-processing, not stored in the data format.

**Metadata**:
```python
{
    "n_splats": int,         # Number of splats
    "ndim": int,             # Dimensionality (d)
    "has_amplitudes": bool,
    "has_cholesky": bool,
    "has_sharpnesses": bool,
    "ordering": str,         # "none", "morton", "hilbert"
    "amplitude_range": {"min": float, "max": float},
    "sharpness_bounds": {"min": 0.0, "max": 32.0},
    "center_bounds": {"min": [...], "max": [...]},
}
```

**Default Values** (when optional arrays not provided):
- amplitudes: `1.0`
- cholesky_factors: identity (isotropic unit sphere)
- sharpnesses: `1.0`

**Cholesky Packing**:
Lower-triangular matrix packed row-major:
- 2D (k=3): `[L00, L10, L11]`
- 3D (k=6): `[L00, L10, L11, L20, L21, L22]`
- 4D (k=10): `[L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]`

**Encoding**: Uses `luxar.encoding` with semantic types:
- centers: COORDINATE
- amplitudes: POSITIVE_SCALAR
- cholesky_factors: CHOLESKY
- sharpnesses: BOUNDED_SCALAR [0, 32]

**Reference**: See `luxar/gsplats/GSPLATS_ZARR_FORMAT.md` for complete specification.

---

### 7. Dimension System

**Specification**:

#### Dimension (Single Dimension)
**Fields**:
- `name` (str, required) - Dimension identifier
- `unit` (str) - Physical unit (e.g., "um", "s", "px")
- `range` (tuple[float, float] | None) - Optional bounds
- `step` (float | None) - Navigation step size (auto if None)
- `display` (bool, default True) - Whether dimension is visualized (max 3)
- `discrete` (bool, default False) - Whether values are categorical/discrete
- `cyclic` (bool, default False) - Whether dimension wraps around
- `scale` (float, default 1.0) - Physical scale factor
- `spatial` (bool | None) - Whether elements extend through dimension (auto if None)
- `description` (str) - Optional human-readable description

**Auto-behaviors**:
1. `spatial` flag: If None, auto-determined:
   - displayed → always spatial
   - non-displayed → never spatial
2. Non-spatial, non-displayed dimensions → automatically discrete (with warning)
3. Step size: If None, auto-calculated:
   - discrete → 1.0
   - continuous with range → 1% of range
   - fallback → 0.1

**Validation Rules**:
- range[0] < range[1]
- step > 0 (if specified)
- scale > 0
- Cannot be discrete AND spatial unless displayed

#### Dimensions (Collection)
**Specification**:
- Container for list of Dimension objects
- Maximum 3 displayed dimensions
- At least 1 displayed dimension if any exist
- Dimension names must be unique

**Key Properties**:
- `ndim` - Total number of dimensions
- `names` - List of dimension names
- `displayed` - Indices of displayed dimensions
- `non_displayed` - Indices of non-displayed dimensions
- `spatial_extend_dims` - Boolean list indicating spatial extension per dimension

---

### 8. Transformation System

**Specification**:

#### Transform Representation
- All transforms are 4x4 homogeneous matrices (float32)
- Stored in column-major order for THREE.js compatibility
- NumPy uses row-major, storage uses column-major
- Translation components at indices [0,3], [1,3], [2,3] in row-major
- Translation components at indices [12], [13], [14] in column-major (storage)

#### nD Limitation
**Transforms only apply to the 3 displayed dimensions**. For nD data (d > 3):
- Transforms affect the first 3 spatial coordinates only
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
- `compose(T1, T2, T3, ...)` - Combines transforms
- **Critical**: Applies T1 first, then T2, then T3
- **Implementation**: Produces T3 @ T2 @ T1 (right-to-left matrix multiplication)
- **Accumulation**: iterate reversed transforms, right-multiply: `result = result @ transform`

**Mathematical Formula**:
```
compose(T1, T2, T3) = T3 @ T2 @ T1

Applied to vector v:
(T3 @ T2 @ T1) @ v = T3 @ (T2 @ (T1 @ v))

This applies T1 first, then T2, then T3.
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
- `gamma`: 0.2 to 5.0 (float)
- `blending_mode`: "normal" | "additive" (string)

**Validation**: All values validated on assignment, invalid values raise ValueError

**Inheritance**: Child nodes can override parent rendering attributes

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
- Tolerance determined by element size (radius, width) for spatial dimensions
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
| BOUNDED_SCALAR | sharpness, sharpnesses | [0, 32] | Edge sharpness (all node types) |
| CHOLESKY | cholesky_factors | unbounded | Covariance decomposition |
| INDEX | indices | [0, N) | Integer references (adaptive uint8/16/32) |

**Broadcasting Support**:
Arrays with shape `(1,)` or `(1, d)` are broadcast to all elements.
Metadata: `{"encoding": {"name": "broadcasted", "n_elements": N}}`

---

## Writer Protocol

The `ZarrWriterProtocol` defines the interface for progressive writing:

```python
class ZarrWriterProtocol(Protocol):
    def write_group(self, path, **attrs) -> None: ...

    def write_points(
        self, path, positions,
        colors=None, radii=None, sharpness=None,
        **attrs
    ) -> PointsMetadata: ...

    def write_lines(
        self, path, vertices,
        colors=None, widths=None, sharpness=None, indices=None,
        line_type="polyline",
        **attrs
    ) -> LinesMetadata: ...

    def write_gsplats(
        self, path, centers,
        amplitudes=None, cholesky_factors=None, sharpnesses=None,
        **attrs
    ) -> GSplatsMetadata: ...

    def finalize(self) -> None: ...
```

Each `write_*` method:
1. Validates input arrays
2. Applies encoding based on semantic types
3. Writes immediately to Zarr
4. Returns only metadata (not the data)

**Metadata Type Aliases**:
```python
PointsMetadata = Dict[str, Any]   # n_points, dims, has_colors, has_radii, has_sharpness, max_radius
LinesMetadata = Dict[str, Any]    # n_vertices, n_segments, dims, line_type, has_colors, has_widths, has_sharpness, max_width
GSplatsMetadata = Dict[str, Any]  # n_splats, ndim, has_amplitudes, has_cholesky, has_sharpnesses, ordering
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
  └─ metadata: n_points, ndim, has_colors, has_radii, has_sharpness, max_radius

Lines (extends DataNode)
  └─ type: "lines"
  └─ n_elements: n_vertices
  └─ metadata: n_vertices, n_segments, ndim, line_type, has_colors, has_widths, has_sharpness, max_width

GSplats (extends DataNode)
  └─ type: "gsplats"
  └─ n_elements: n_splats
  └─ metadata: n_splats, ndim, has_amplitudes, has_cholesky, has_sharpnesses, ordering

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
- Points: Point clouds (spheres/gaussians)
- Lines: Line segments, polylines, loops
- GSplats: Oriented Gaussian ellipsoids

**Potential Future Types** (not yet implemented):
- Meshes: Triangle/polygon meshes
- Volumes: Voxel/volumetric data
- Arrows: Directed vectors
- Text: 3D labels

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
for transform in reversed([T1, T2, T3]):  # [T3, T2, T1]
    result = result @ transform  # right-multiply

# Produces: I @ T1 = T1, then T1 @ T2, then (T1@T2) @ T3 = T3 @ T2 @ T1
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
- Composition is O(k) where k is number of transforms
- Each composition is 4x4 matrix multiplication (64 FLOPs)
- Result cached, not recomputed on each access

**Data Writing**:
- Immediate write to Zarr (no buffering)
- Encoding applied during write
- Compression via Blosc + zstd

---

This specification is sufficient to re-implement the core package in any language while maintaining compatibility with the Luxar ecosystem.
