# Gaussian Splats Dimension Mapping Specification

**Status**: Implemented (via `dim_order`, `fill`, and `fill_sigma` parameters in `Group.add_gsplats()` and related methods in `packages/luxar/src/luxar/core/group.py`)
**Created**: 2026-01-15
**Author**: Royer Lab

## Problem Statement

Currently, Gaussian splats in Luxar must always span ALL scene dimensions, requiring awkward padding when adding lower-dimensional splats to higher-dimensional scenes.

**Current behavior (problematic):**
```python
# Scene is 5D: [x, y, z, time, channel]
scene = Scene(dimensions=["x", "y", "z", "time", "channel"])

# You fit 3D splats to a single timepoint/channel
splat_data = fit_gaussian_splats(volume_3d)  # Centers: (N, 3)

# FORCED to pad centers and covariance to 5D:
centers_5d = np.pad(centers_3d, ((0, 0), (0, 2)))  # Add dummy time/channel dims
cholesky_5d = pad_cholesky_to_5d(cholesky_3d)      # 6 → 15 elements

scene.add_gsplats("splats", centers_5d, ..., cholesky_5d)
```

**Issues with current approach:**
- ❌ Unnatural (3D data stored as 5D)
- ❌ Wasteful (15 Cholesky elements instead of 6)
- ❌ Ambiguous (what do the dummy dimension values mean?)
- ❌ Error-prone (easy to get padding wrong)

---

## Proposed Solution: Explicit Dimension Mapping

Splats should declare their **intrinsic dimensionality** with explicit mapping to scene dimensions.

### Design Principles

1. **Explicit over implicit**: Users must specify dimension mapping when `splat_ndim ≠ scene_ndim`
2. **Automatic only when safe**: Auto-map when `splat_ndim == scene_ndim` (unambiguous)
3. **Mathematically consistent**: Covariance must match splat dimensions exactly
4. **Mutually exclusive**: Dimensions are either *spanned* (position + covariance) or *extended* (always visible)

---

## API Design

### New Parameters

```python
def add_gsplats(
    self,
    name: str,
    centers: np.ndarray,           # Shape (N, D_splat)
    amplitudes: Union[float, np.ndarray],
    cholesky_factors: np.ndarray,  # Shape (N, k) where k = D_splat*(D_splat+1)/2
    colors: Optional[np.ndarray] = None,
    parent: Optional[Node] = None,
    dimensions: Optional[Union[List[str], List[int]]] = None,      # NEW
    extend_to_all: Optional[Union[List[str], List[int]]] = None,   # NEW (modified)
    **attrs: Any,
) -> GSplats:
    """
    Add Gaussian splats to the scene.

    Parameters
    ----------
    dimensions : list of str or list of int, optional
        Which scene dimensions the splats span (have position and covariance in).
        Must match the dimensionality of `centers` array.

        Can specify dimensions by name (str) or index (int):
        - By name: ["x", "y", "z"] or ["time", "x", "y"]
        - By index: [0, 1, 2] or [3, 0, 1]

        Examples:
        - centers.shape = (N, 3) → dimensions = ["x", "y", "z"] or [0, 1, 2]
        - centers.shape = (N, 4) → dimensions = ["x", "y", "z", "time"] or [0, 1, 2, 3]

        Auto-mapping behavior:
        - If None and splat_ndim == scene_ndim: auto-map to all scene dims (in order)
        - If None and splat_ndim == num_spatial_dims: auto-map to spatial dims (with warning)
        - Otherwise: raises an error (must be explicit)

    extend_to_all : list of str or list of int, optional
        Which scene dimensions the splats should be visible across (no attenuation).
        These dimensions must NOT appear in `dimensions` (mutually exclusive).

        Can specify dimensions by name (str) or index (int):
        - By name: ["time", "channel"]
        - By index: [3, 4]

        Together with `dimensions`, must cover ALL scene dimensions (strict requirement).

        Example: For 5D scene [x, y, z, time, channel] with 3D splats:
        - dimensions=["x", "y", "z"]
        - extend_to_all=["time", "channel"]
        - Result: Splats visible at all times and channels
    """
```

### Usage Examples

#### Case 1: 3D Splats in 4D Scene (Explicit Extension)

```python
# Fit splats to 3D microscopy data
volume_3d = load_microscopy_volume()  # Shape: (X, Y, Z)
result = fit_gaussian_splats(volume_3d, num_splats=10000)
# result.centers.shape = (10000, 3)
# result.cholesky_factors.shape = (10000, 6)

# Create 4D scene with time dimension
scene = Scene(dimensions=["x", "y", "z", "time"])

# Add splats: they span [x,y,z] and extend across all time
scene.add_gsplats(
    "cell_structure",
    centers=result.centers,              # (N, 3) - native dimensionality
    amplitudes=result.amplitudes,
    cholesky_factors=result.cholesky_factors,  # (N, 6) - 3D covariance
    colors=result.colors,
    dimensions=["x", "y", "z"],          # Splats span spatial dims
    extend_to_all=["time"]               # Visible at all timepoints
)
```

#### Case 2: Full-Dimensional Splats (Automatic)

```python
# Fit 4D spatiotemporal splats
data_4d = load_timeseries()  # Shape: (T, Z, Y, X)
result = fit_gaussian_splats(data_4d, num_splats=50000)
# result.centers.shape = (50000, 4)
# result.cholesky_factors.shape = (50000, 10)

# Create 4D scene
scene = Scene(dimensions=["x", "y", "z", "time"])

# No dimension mapping needed - splat_ndim == scene_ndim
scene.add_gsplats(
    "dynamic_process",
    centers=result.centers,              # (N, 4)
    amplitudes=result.amplitudes,
    cholesky_factors=result.cholesky_factors,  # (N, 10)
    dimensions=None,  # Auto: all scene dimensions
    extend_to_all=[]  # Empty list = all dimensions are spanned
)
```

#### Case 2b: Spatial Dimension Auto-Mapping (with Warning)

```python
# Fit 3D splats
result = fit_gaussian_splats(volume_3d, num_splats=10000)
# result.centers.shape = (10000, 3)

# Create 4D scene with 3 spatial dims + time
scene = Scene(dimensions=[
    Dimension("x", display=True),
    Dimension("y", display=True),
    Dimension("z", display=True),
    Dimension("time", display=False)  # Non-spatial
])

# Auto-maps to spatial dimensions with warning
scene.add_gsplats(
    "auto_mapped",
    centers=result.centers,              # (N, 3)
    amplitudes=result.amplitudes,
    cholesky_factors=result.cholesky_factors,  # (N, 6)
    # dimensions=None - Auto-detects: splat_ndim == num_spatial_dims
    extend_to_all=["time"]  # Must still specify extension explicitly
)
# ⚠️ Warning: "Auto-mapped 3D splats to spatial dimensions [x, y, z].
#             Specify dimensions explicitly to silence this warning."
```

#### Case 3: Multiple Extended Dimensions

```python
# 3D splats in 5D scene (x, y, z, time, channel)
scene = Scene(dimensions=["x", "y", "z", "time", "channel"])

# Fit splats to single channel at single timepoint
result = fit_gaussian_splats(volume_3d, num_splats=5000)

scene.add_gsplats(
    "nuclei",
    centers=result.centers,              # (N, 3)
    amplitudes=result.amplitudes,
    cholesky_factors=result.cholesky_factors,  # (N, 6)
    dimensions=["x", "y", "z"],
    extend_to_all=["time", "channel"]    # Visible at all times and channels
)
```

#### Case 4: Partial Coverage (Attenuation in Unmapped Dims)

```python
# 3D splats in 5D scene, but only at specific time/channel
scene = Scene(dimensions=["x", "y", "z", "time", "channel"])

result = fit_gaussian_splats(volume_3d, num_splats=5000)

scene.add_gsplats(
    "snapshot",
    centers=result.centers,              # (N, 3)
    amplitudes=result.amplitudes,
    cholesky_factors=result.cholesky_factors,  # (N, 6)
    dimensions=["x", "y", "z"],
    extend_to_all=None                   # Attenuate in time/channel
    # Splats will only be visible near time=0, channel=0
)
```

#### Case 5: Dimension Reordering

```python
# Splats fitted in [z, y, x] order, scene uses [x, y, z, time]
scene = Scene(dimensions=["x", "y", "z", "time"])

# Centers are in [z, y, x] order
result = fit_gaussian_splats(data_zyx, num_splats=1000)

scene.add_gsplats(
    "reordered",
    centers=result.centers,              # (N, 3) in [z,y,x] order
    amplitudes=result.amplitudes,
    cholesky_factors=result.cholesky_factors,
    dimensions=["z", "y", "x"],          # Explicit ordering by name
    extend_to_all=["time"]
)
```

#### Case 6: Using Integer Dimension Indices

```python
# Same as Case 1, but using indices instead of names
scene = Scene(dimensions=["x", "y", "z", "time"])

result = fit_gaussian_splats(volume_3d, num_splats=10000)

scene.add_gsplats(
    "cell_structure",
    centers=result.centers,              # (N, 3)
    amplitudes=result.amplitudes,
    cholesky_factors=result.cholesky_factors,
    dimensions=[0, 1, 2],                # Indices instead of names
    extend_to_all=[3]                    # Index 3 = "time"
)
```

---

## Validation Rules

The implementation must enforce these invariants:

### Rule 1: Shape Consistency
```python
len(dimensions) == centers.shape[1]
```
The number of specified dimensions must match the splat data dimensionality.

### Rule 2: Cholesky Consistency
```python
expected_k = splat_ndim * (splat_ndim + 1) // 2
cholesky_factors.shape[1] == expected_k
```
Cholesky factors must match splat dimensionality, not scene dimensionality.

### Rule 3: Mutual Exclusivity
```python
set(dimensions) ∩ set(extend_to_all) == ∅
```
A dimension cannot be both spanned and extended. These are mutually exclusive concepts:
- **Spanned** (`dimensions`): Splat has position and covariance in this dimension
- **Extended** (`extend_to_all`): Splat is visible everywhere in this dimension

### Rule 4: Complete Coverage (Strict Mode)
```python
set(dimensions) ∪ set(extend_to_all) == set(scene_dimension_names)
```
Every scene dimension must be either spanned or extended. No ambiguity allowed.

**No exceptions**: This is a strict requirement. Users must explicitly specify all dimensions to avoid ambiguity about visibility and attenuation behavior.

### Rule 5: Dimension Name Validity
```python
all(dim in scene.dimension_names for dim in dimensions)
all(dim in scene.dimension_names for dim in extend_to_all)
```
All dimension names must exist in the scene.

---

## Storage Format (Zarr)

### Metadata Additions

```json
{
  "type": "gsplats",
  "n_splats": 10000,
  "ndim": 3,
  "splat_dimensions": ["x", "y", "z"],
  "extend_to_all": ["time"],
  "scene_ndim": 4,
  "has_colors": true,
  "cholesky_k": 6
}
```

**Key fields:**
- `ndim`: Intrinsic splat dimensionality (e.g., 3)
- `splat_dimensions`: Scene dimension names the splats span
- `extend_to_all`: Scene dimension names to extend across
- `scene_ndim`: Total scene dimensionality (for validation)
- `cholesky_k`: Number of Cholesky elements (= ndim*(ndim+1)/2)

### Zarr Arrays

```
gsplat_node/
  ├── .zattrs                   (metadata above)
  ├── centers                   (N, ndim) float32
  ├── amplitudes                (N,) or (1,) float32
  ├── cholesky_factors          (N, k) or (1, k) float32, k=ndim*(ndim+1)/2
  └── colors                    (N, 3) or (1, 3) uint8/float32
```

**No padding**: Arrays use intrinsic dimensionality, not scene dimensionality.

---

## TypeScript Processing

### Loading Phase

```typescript
interface GSplatsMetadata extends BaseMetadata {
  type: 'gsplats';
  n_splats: number;
  ndim: number;                      // Intrinsic splat dims (e.g., 3)
  splat_dimensions: string[];        // Scene dim names (e.g., ["x", "y", "z"])
  extend_to_all?: string[];          // Extended dim names (e.g., ["time"])
  scene_ndim: number;                // Total scene dims (e.g., 4)
  cholesky_k: number;                // k = ndim*(ndim+1)/2
  // ... existing fields
}
```

### nD → 3D Projection Algorithm

When rendering in nD viewer with `displayDims = [d0, d1, d2]` and `slicePosition`:

#### Step 1: Build Dimension Mappings

```typescript
// Map display dimensions → splat dimension indices
const displayToSplat = new Map<number, number | null>();
for (let displayIdx = 0; displayIdx < 3; displayIdx++) {
  const sceneDimName = scene.dimensions[displayDims[displayIdx]].name;
  const splatIdx = metadata.splat_dimensions.indexOf(sceneDimName);
  displayToSplat.set(displayIdx, splatIdx >= 0 ? splatIdx : null);
}

// Identify hidden splat dimensions (not displayed, not extended)
const hiddenSplatDims: number[] = [];
for (let splatIdx = 0; splatIdx < metadata.ndim; splatIdx++) {
  const sceneDimName = metadata.splat_dimensions[splatIdx];
  const sceneDimIdx = scene.dimensions.findIndex(d => d.name === sceneDimName);

  if (!displayDims.includes(sceneDimIdx) &&
      !metadata.extend_to_all?.includes(sceneDimName)) {
    hiddenSplatDims.push(splatIdx);
  }
}
```

#### Step 2: Extract 3D Centers

```typescript
for (let i = 0; i < nSplats; i++) {
  for (let displayIdx = 0; displayIdx < 3; displayIdx++) {
    const splatIdx = displayToSplat.get(displayIdx);

    if (splatIdx !== null) {
      // Display dimension is spanned by splat
      centers3D[i*3 + displayIdx] = centers[i*metadata.ndim + splatIdx];
    } else {
      // Display dimension is extended or not in splat
      // Place at current slice position (doesn't matter for extended dims)
      const sceneDimIdx = displayDims[displayIdx];
      centers3D[i*3 + displayIdx] = slicePosition[sceneDimIdx];
    }
  }
}
```

#### Step 3: Extract 3D Cholesky Submatrix

```typescript
// Get splat dimension indices corresponding to display
const displayedSplatIndices = [0, 1, 2]
  .map(d => displayToSplat.get(d))
  .filter(idx => idx !== null) as number[];

const cholesky3D = extractCholeskySubmatrix(
  cholesky,
  splatIndex,
  metadata.ndim,
  displayedSplatIndices
);
```

#### Step 4: Compute Attenuation in Hidden Dimensions

```typescript
let attenuation = 1.0;

for (const splatIdx of hiddenSplatDims) {
  // Get scene dimension index for this splat dimension
  const sceneDimName = metadata.splat_dimensions[splatIdx];
  const sceneDimIdx = scene.dimensions.findIndex(d => d.name === sceneDimName);

  // Compute Mahalanobis distance in this dimension
  const centerValue = centers[i*metadata.ndim + splatIdx];
  const sliceValue = slicePosition[sceneDimIdx];
  const diff = sliceValue - centerValue;

  // Extract 1D variance from Cholesky (diagonal element)
  const variance = getCholeskyVariance(cholesky, i, metadata.ndim, splatIdx);
  const mahalDist = Math.abs(diff) / Math.sqrt(variance);

  // Standard Gaussian falloff
  attenuation *= Math.exp(-0.5 * mahalDist * mahalDist);
}

amplitudes3D[i] = amplitudes[i] * attenuation;
```

**Key insight**: Only attenuate in hidden dimensions that are **spanned** by the splat but not displayed. Extended dimensions never attenuate.

---

## Implementation Checklist

### Python (packages/luxar/)

- [ ] Update `Scene.add_gsplats()` signature with `dimensions` and modified `extend_to_all`
- [ ] Add validation rules (5 rules above)
- [ ] Update `LuxarZarrCompiler.write_gsplats()` to store dimension metadata
- [ ] Update `GSplats` dataclass to include dimension info
- [ ] Add helper: `_validate_gsplat_dimensions(centers, dimensions, extend_to_all, scene_dims)`
- [ ] Update docstrings and type hints
- [ ] Remove padding logic (no longer needed)

### TypeScript (packages/luxar-viewer/)

- [ ] Update `GSplatsMetadata` interface with new fields
- [ ] Update `processGSplats()` with dimension mapping logic
- [ ] Update `extractCholeskySubmatrix()` to handle arbitrary dimension subsets
- [ ] Add `buildDimensionMappings()` helper
- [ ] Add `computeAttenuationInHiddenDims()` helper
- [ ] Update `GSplatsViewState` to include dimension mappings
- [ ] Update loader to pass dimension metadata to processor
- [ ] Add validation for dimension consistency

### WASM (if applicable)

- [ ] Update Rust functions with dimension mapping parameters
- [ ] Handle variable splat dimensionality (not just scene ndim)
- [ ] Update Cholesky extraction to use dimension indices

### Tests

- [ ] Test Case 1: 3D splats in 4D scene with extension
- [ ] Test Case 2: Full-dimensional splats (auto-mapping)
- [ ] Test Case 3: Multiple extended dimensions
- [ ] Test Case 4: Partial coverage with attenuation
- [ ] Test Case 5: Dimension reordering
- [ ] Test Case 6: Validation errors (mismatched shapes, invalid dims)
- [ ] Test Case 7: Cholesky submatrix extraction correctness
- [ ] Test Case 8: Attenuation computation in hidden dims
- [ ] Cross-language test: Python encoder → TypeScript decoder

### Documentation

- [ ] Update `CLAUDE.md` with new API
- [ ] Update `LUXAR_ZARR_FORMAT.md` with storage changes
- [ ] Add examples to `gsplats/README.md`
- [ ] Update API reference (Sphinx docs)
- [ ] Add migration guide (breaking change)

---

## Design Decisions (Resolved)

### D1: Auto-mapping behavior when dimensions is None

**Decision**: ✅ **Automatic mapping in two cases:**
1. If `splat_ndim == scene_ndim`: Auto-map to all scene dimensions (in order), no warning
   - `extend_to_all` can be empty list `[]` (all dims spanned) or list of specific dims
2. If `splat_ndim == num_spatial_dimensions` **and** `extend_to_all` is specified: Auto-map to spatial dimensions (in order), **with warning**
   - `extend_to_all` is REQUIRED to ensure complete coverage (strict mode)
3. Otherwise: Error (user must be explicit)

**Rationale**: Safe when unambiguous. Spatial dimension mapping is common enough to support, but warrants a warning since it's making an assumption. Requiring `extend_to_all` in case 2 enforces complete coverage and avoids ambiguity about non-spatial dimensions.

**Implementation detail**: Spatial dimensions are identified by `dimension.display == True`. Auto-mapping collects all dimensions where `display=True` in order.

```python
# In scene.py
spatial_dims = [
    dim.name for dim in self.dimensions.dimensions
    if dim.display
]
num_spatial = len(spatial_dims)

if dimensions is None:
    if splat_ndim == scene_ndim:
        # Case 1: Exact match, auto-map all
        dimensions = list(range(scene_ndim))
    elif splat_ndim == num_spatial:
        # Case 2: Matches spatial dims, auto-map with warning
        dimensions = [i for i, dim in enumerate(self.dimensions.dimensions) if dim.display]
        warnings.warn(
            f"Auto-mapped {splat_ndim}D splats to spatial dimensions {spatial_dims}. "
            f"Specify dimensions explicitly to silence this warning.",
            UserWarning
        )
    else:
        # Case 3: Ambiguous, error
        raise ValueError(
            f"Cannot auto-map {splat_ndim}D splats to {scene_ndim}D scene "
            f"(spatial dims: {num_spatial}). Specify dimensions explicitly."
        )
```

---

### D2: Support for integer dimension indices

**Decision**: ✅ **Support both names and indices**

```python
# By name (more readable)
dimensions=["x", "y", "z"]
extend_to_all=["time"]

# By index (more concise)
dimensions=[0, 1, 2]
extend_to_all=[3]
```

**Rationale**: Names are more readable, indices are more concise. Both have valid use cases. The implementation can normalize internally.

---

### D3: Spatial index adaptation

**Decision**: ✅ **Adapt spatial indexing to handle dimension mapping**

Current spatial index uses scene-space bounding boxes. With dimension mapping:
- Spatial index is built in **splat space** (intrinsic dimensionality)
- Chunk bounds are computed for splat dimensions only
- Loader translates queries from scene space to splat space
- Index metadata stores dimension mapping for query translation

**Implementation approach**:
```python
# In compiler.py - spatial ordering
ordering_data = compute_spatial_ordering(
    centers,  # Shape (N, splat_ndim)
    splat_dimensions,  # Which scene dims these map to
    method="hilbert"
)

# Chunk bounds are in splat space
chunk_bounds = compute_chunk_bounds_gsplats(
    centers,  # (N, splat_ndim)
    cholesky_factors,  # (N, k)
    splat_ndim  # Not scene_ndim!
)

# In loader - query translation
def query_splats(scene_bbox, view_state):
    # Extract splat-space bbox from scene-space bbox
    splat_bbox = project_bbox_to_splat_space(
        scene_bbox,
        self.metadata.splat_dimensions,
        view_state.slicePosition  # For extended/hidden dims
    )
    # Query spatial index in splat space
    chunks = self.spatial_index.query(splat_bbox)
    return chunks
```

**Rationale**: More efficient to index in splat's native dimensionality. Clean separation between data space and scene space.

---

## Migration Path

This is a **breaking change** for GSplats API.

**Option A**: Break immediately (recommended for early-stage project)
```python
# Old code breaks with clear error message
scene.add_gsplats(name, centers_5d, ...)
# Error: "Dimension mismatch. GSplats API has changed.
#         See docs/specs/GSPLATS_DIMENSION_MAPPING.md"
```

**Option B**: Support both formats temporarily
- Detect old format: `centers.shape[1] == scene_ndim` and no `dimensions` field
- Emit deprecation warning
- Remove in next major version

**Recommendation**: Option A (clean break, clear error messages)

---

## Error Messages

Since this design enforces strict, explicit, unambiguous dimension specification, error messages must be informative and actionable.

### Error 1: Shape Mismatch

```python
# User provides 3 dimensions but 4D centers
scene.add_gsplats(..., centers=centers_4d, dimensions=["x", "y", "z"])

# Error:
ValueError: Dimension count mismatch: centers have 4 dimensions but
            dimensions=["x", "y", "z"] specifies 3.
            Either:
            - Use 3D centers matching dimensions, or
            - Specify all 4 dimensions in 'dimensions' parameter
```

### Error 2: Incomplete Coverage

```python
# 5D scene, but only 4 dimensions specified
scene.add_gsplats(
    ...,
    dimensions=["x", "y", "z"],
    extend_to_all=["time"]
    # Missing: "channel" dimension!
)

# Error:
ValueError: Incomplete dimension coverage. All scene dimensions must be
            either spanned or extended.
            Scene dimensions: ["x", "y", "z", "time", "channel"]
            Spanned: ["x", "y", "z"]
            Extended: ["time"]
            Missing: ["channel"]

            Add to extend_to_all=["time", "channel"] or dimensions as needed.
```

### Error 3: Overlapping Dimensions

```python
# Same dimension in both lists
scene.add_gsplats(
    ...,
    dimensions=["x", "y", "z", "time"],
    extend_to_all=["time"]
)

# Error:
ValueError: Dimension "time" appears in both 'dimensions' and 'extend_to_all'.
            These are mutually exclusive:
            - dimensions: Splat has position and covariance in this dimension
            - extend_to_all: Splat is visible everywhere in this dimension

            Remove "time" from one of the lists.
```

### Error 4: Cholesky Shape Mismatch

```python
# 3D splats but 4D Cholesky
scene.add_gsplats(
    ...,
    centers=centers_3d,  # (N, 3)
    cholesky_factors=cholesky_4d,  # (N, 10) - wrong!
    dimensions=["x", "y", "z"]
)

# Error:
ValueError: Cholesky factors shape mismatch: expected (N, 6) for 3D splats,
            got (N, 10).

            For D-dimensional splats, Cholesky must have k = D*(D+1)/2 elements:
            - 2D: k=3
            - 3D: k=6
            - 4D: k=10
            - 5D: k=15
```

### Error 5: Invalid Dimension Names/Indices

```python
# Typo in dimension name
scene.add_gsplats(..., dimensions=["x", "y", "zee"])  # Should be "z"

# Error:
ValueError: Unknown dimension "zee" in dimensions parameter.
            Valid dimensions: ["x", "y", "z", "time", "channel"]
            Did you mean "z"?

# Out of range index
scene.add_gsplats(..., dimensions=[0, 1, 5])  # Scene only has 4 dims

# Error:
ValueError: Dimension index 5 out of range. Scene has 4 dimensions (0-3).
            Valid indices: [0, 1, 2, 3]
```

### Error 6: Auto-mapping Ambiguity

```python
# 3D splats in 5D scene, no dimensions specified
scene = Scene(dimensions=["x", "y", "z", "time", "channel"])
result = fit_gaussian_splats(volume_3d)  # 3D

scene.add_gsplats(..., centers=result.centers)  # dimensions=None, extend_to_all=None

# Error:
ValueError: Cannot auto-map 3D splats to 5D scene (spatial dims: 3).
            Ambiguous mapping: splat dimensionality matches spatial dimensions
            but extend_to_all was not specified.

            Be explicit:
            scene.add_gsplats(
                ...,
                dimensions=["x", "y", "z"],
                extend_to_all=["time", "channel"]
            )
```

### Warning 1: Auto-mapping to Spatial Dimensions

```python
# 3D splats in 4D scene with 3 spatial dims, extension specified
scene.add_gsplats(
    ...,
    centers=result.centers,  # (N, 3)
    # dimensions=None - implicitly maps to spatial
    extend_to_all=["time"]
)

# Warning:
UserWarning: Auto-mapped 3D splats to spatial dimensions ["x", "y", "z"].
             Specify dimensions=["x", "y", "z"] explicitly to silence this warning.
```

---

## References

- Related: `extend_to_all` feature for Points/Lines
- Related: nD navigation system in viewer
- Related: Spatial indexing for efficient loading
- Format spec: `docs/guides/user/LUXAR_ZARR_FORMAT.md`
- Fitting API: `packages/luxar/src/luxar/gsplats/fit_gsplats.py`
