# luxar.core - Technical Specification

## Purpose

The `core` package defines the fundamental data structures and scene graph system for organizing and manipulating large-scale point cloud data. It provides a hierarchical scene graph with transformation support and nD dimensional specifications.

---

## Core Data Structures

### 1. Scene Graph Node

**Specification**:
- Nodes form a hierarchical tree structure
- Each node has: name, parent reference, list of children
- Each node can have a 4x4 transformation matrix
- Each node has rendering attributes: opacity (0-1), gamma (0.2-2.0), blending_mode (string)
- Nodes do NOT store actual data - only metadata and references
- Nodes write data immediately through a writer interface (progressive writing)

**Invariants**:
- Parent-child relationships are bidirectional (parent knows children, children know parent)
- Root node has no parent (parent = None)
- Transforms compose hierarchically: final_transform = parent_transform @ local_transform
- Attribute changes are immediately persisted through writer interface

**Key Operations**:
- `add_group(name, **attrs)` - Create child group node
- `walk()` - Depth-first traversal yielding (depth, node) tuples
- Property getters/setters for transform, opacity, gamma, blending_mode

---

### 2. Points Node

**Specification**:
- Specialized node type for point cloud data
- Inherits all Node capabilities
- Stores only metadata about the points (not the actual data)
- Metadata includes: n_points, dimensionality, has_colors, has_radii, has_sharpness, max_radius

**Critical Implementation Detail**:
- Must call parent __init__() BEFORE setting self._metadata
- Parent Node.__init__() initializes _metadata = {}, which would overwrite subclass value if set before

**Invariants**:
- Metadata accurately reflects what was written to storage
- has_* flags correctly indicate which attributes are present
- n_points matches actual number of points written

---

### 3. Scene Root Node

**Specification**:
- Special node that serves as scene root
- Created through LuxarZarrCompiler.create_scene()
- Requires a writer interface (cannot exist without one)
- Can have scene-level dimension definitions

**Key Operations**:
- `add_group(name, **attrs)` - Create top-level group
- `add_points(name, positions, ...)` - Write points immediately to storage
- `dimensions` property - Get/set scene-level dimensional specifications

**Broadcasting Behavior**:
- Points can specify `broadcast_dims` to appear at all values of certain dimensions
- Auto-detection available but explicit is preferred
- Broadcasting replicates points across dimension combinations

---

### 4. Dimension System

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
- `spatial` (bool | None) - Whether points extend through dimension (auto if None)
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

**Validation**:
- Can validate position arrays against dimension specifications
- Checks: correct dimensionality, values within ranges (if specified)

---

### 5. Transformation System

**Specification**:

#### Transform Representation
- All transforms are 4x4 homogeneous matrices (float32)
- Stored in column-major order for THREE.js compatibility
- NumPy uses row-major, storage uses column-major
- Translation components at indices [0,3], [1,3], [2,3] in row-major
- Translation components at indices [12], [13], [14] in column-major (storage)

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
2. Points data written immediately to Zarr via writer interface
3. Only metadata returned (n_points, has_colors, etc.)
4. No Zarr groups kept in memory after writing
5. Attribute cache maintains fast access to metadata

**Data Flow**:
```
User provides data → Node validates → Writer writes to Zarr → Metadata returned → Node stores metadata
```

---

## Hierarchical Transformation

**Specification**:

Transforms compose from root to leaf:
```
final_transform = root_transform @ parent_transform @ local_transform
```

Each node can have a local transform. The final transform for any node is the composition of all ancestor transforms.

**Storage**: Each transform stored independently in node attributes as 16-element list (column-major).

**Retrieval**: When reading transform, automatically transpose from column-major back to row-major NumPy format.

---

## Rendering Attributes

**Valid Ranges**:
- `opacity`: 0.0 to 1.0 (float)
- `gamma`: 0.2 to 2.0 (float)
- `blending_mode`: "normal" | "additive" (string)

**Validation**: All values validated on assignment, invalid values raise ValueError

**Inheritance**: Child nodes can override parent rendering attributes

**Default Values**:
- opacity: 1.0
- gamma: 1.0
- blending_mode: "additive"

---

## nD Point Representation

**Specification**:

Points can have arbitrary dimensions (not limited to 3D).

**Visualization Strategy**:
- Maximum 3 dimensions can be displayed simultaneously
- Non-displayed dimensions are "sliced"
- Points visible when within tolerance of current slice position
- Tolerance determined by point radius (for spatial dimensions) or exact match (for discrete dimensions)

**Spatial Extension**:
- Spatial dimensions: Points treated as hyperspheres, extend through dimension
- Non-spatial dimensions: Points exist at specific discrete values only
- Example: Time dimension typically non-spatial (points at t=5, not spreading through time)
- Example: Depth dimension could be spatial (points as 3D objects extending through depth)

**Broadcasting**:
- Points can be broadcast to all values of specified dimensions
- Useful for static objects that appear across all time points/channels
- Implemented by replicating point data with different dimension values

---

## Key Algorithms

### Auto-Broadcast Detection
**Purpose**: Automatically determine which dimensions should be broadcast based on data

**Algorithm**:
1. Calculate expected total points for full coverage (product of dimension sizes)
2. If actual points < expected * 0.8, check each non-displayed dimension
3. For each non-displayed dimension:
   - If only one unique value AND incomplete coverage → broadcast
   - If dimension not in position array → broadcast
4. Return list of dimension names to broadcast

**Note**: Auto-detection can be ambiguous, explicit is preferred

---

## Dependencies Between Core Components

```
Scene (root)
  └─ requires: Writer (for progressive writing)
  └─ optional: Dimensions (for nD specification)
  └─ creates: Node instances (groups)
  └─ creates: Points instances (point clouds)

Node (group)
  └─ requires: Writer (optional, for progressive mode)
  └─ has: children (list of Node)
  └─ has: parent (reference to parent Node)
  └─ has: transform (optional 4x4 matrix)

Points (point cloud)
  └─ extends: Node
  └─ has: metadata (dict with has_colors, has_radii, etc.)
  └─ stores: NO actual data (written immediately to Zarr)

Dimensions
  └─ contains: list of Dimension
  └─ validates: position arrays
  └─ provides: displayed/non-displayed indices

Dimension
  └─ standalone: can be used independently
  └─ validation: auto-behaviors in __post_init__()
```

---

## Critical Implementation Notes

### 1. Constructor Initialization Order
When subclass and parent both initialize the same attribute:
- ✅ CORRECT: Call super().__init__() FIRST, then set subclass attributes
- ❌ WRONG: Set attributes first, then call super().__init__() (parent overwrites)

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

---

## Testing Requirements

**Core Functionality Tests**:
- Scene creation with/without dimensions
- Hierarchical node creation
- Transform composition and application
- Dimension validation
- Metadata preservation
- Rendering attribute validation

**Critical Bug Prevention**:
- Test compose() with non-commutative operations (rotate+translate, order matters)
- Test Points metadata is preserved (has_colors, has_radii, has_sharpness)
- Test dimension auto-behaviors (spatial flag, discrete correction)
- Test transform storage round-trip (NumPy → storage → NumPy)

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

---

## Performance Considerations

**Memory**:
- Nodes are lightweight (< 1KB each)
- No point data kept in memory after writing
- Only metadata and attribute cache in memory

**Hierarchy Traversal**:
- Depth-first walk is generator (memory efficient)
- O(n) where n is number of nodes
- Each node yielded once

**Transform Composition**:
- Composition is O(k) where k is number of transforms
- Each composition is 4x4 matrix multiplication (64 FLOPs)
- Result cached, not recomputed on each access

---

## This specification is sufficient to re-implement the core package in any language while maintaining compatibility with the Luxar ecosystem.
