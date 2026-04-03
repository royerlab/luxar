# Gaussian Splats Dimension Mapping

**Version**: 1.0
**Last Updated**: 2026-01-15

## Overview

Gaussian splats declare their **intrinsic dimensionality** with explicit mapping to scene dimensions. This allows lower-dimensional splats (e.g., 3D) to be added to higher-dimensional scenes (e.g., 5D) without padding — splats store only the dimensions they actually span, and declare which remaining dimensions they should be visible across.

### Design Principles

1. **Explicit over implicit**: Users must specify dimension mapping when `splat_ndim != scene_ndim`
2. **Automatic only when safe**: Auto-map when `splat_ndim == scene_ndim` (unambiguous)
3. **Mathematically consistent**: Covariance must match splat dimensions exactly
4. **Mutually exclusive**: Dimensions are either *spanned* (position + covariance) or *extended* (always visible)

---

## API

### Parameters

The `add_gsplats()` method on `Scene` and `Group` accepts these dimension mapping parameters:

```python
def add_gsplats(
    self,
    name: str,
    centers: np.ndarray,           # Shape (N, D_splat)
    amplitudes: Union[float, np.ndarray],
    cholesky_factors: np.ndarray,  # Shape (N, k) where k = D_splat*(D_splat+1)/2
    colors: Optional[np.ndarray] = None,
    parent: Optional[Node] = None,
    dimensions: Optional[Union[List[str], List[int]]] = None,
    extend_to_all: Optional[Union[List[str], List[int]]] = None,
    **attrs: Any,
) -> GSplats:
```

**`dimensions`** — Which scene dimensions the splats span (have position and covariance in). Must match the dimensionality of `centers`. Can use names (`["x", "y", "z"]`) or indices (`[0, 1, 2]`).

Auto-mapping behavior when `None`:
- If `splat_ndim == scene_ndim`: auto-map to all scene dimensions in order
- If `splat_ndim == num_spatial_dims` and `extend_to_all` is specified: auto-map to spatial dimensions (with warning)
- Otherwise: raises an error

**`extend_to_all`** — Which scene dimensions the splats should be visible across without attenuation. Must not overlap with `dimensions`. Together with `dimensions`, must cover all scene dimensions.

---

## Usage Examples

### 3D Splats in 4D Scene (Explicit Extension)

```python
result = fit_gaussian_splats(volume_3d, num_splats=10000)
# result.centers.shape = (10000, 3)
# result.cholesky_factors.shape = (10000, 6)

scene = Scene(dimensions=["x", "y", "z", "time"])

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

### Full-Dimensional Splats (Automatic)

```python
result = fit_gaussian_splats(data_4d, num_splats=50000)
# result.centers.shape = (50000, 4)

scene = Scene(dimensions=["x", "y", "z", "time"])

scene.add_gsplats(
    "dynamic_process",
    centers=result.centers,              # (N, 4)
    amplitudes=result.amplitudes,
    cholesky_factors=result.cholesky_factors,  # (N, 10)
    # dimensions=None → auto-maps to all scene dimensions
)
```

### Multiple Extended Dimensions

```python
scene = Scene(dimensions=["x", "y", "z", "time", "channel"])

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

### Dimension Reordering

```python
scene = Scene(dimensions=["x", "y", "z", "time"])

# Centers are in [z, y, x] order from fitting
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

---

## Validation Rules

### Rule 1: Shape Consistency
```
len(dimensions) == centers.shape[1]
```
The number of specified dimensions must match the splat data dimensionality.

### Rule 2: Cholesky Consistency
```
cholesky_factors.shape[1] == splat_ndim * (splat_ndim + 1) // 2
```
Cholesky factors must match splat dimensionality, not scene dimensionality.

### Rule 3: Mutual Exclusivity
```
set(dimensions) & set(extend_to_all) == {}
```
A dimension cannot be both spanned and extended:
- **Spanned** (`dimensions`): Splat has position and covariance in this dimension
- **Extended** (`extend_to_all`): Splat is visible everywhere in this dimension

### Rule 4: Complete Coverage
```
set(dimensions) | set(extend_to_all) == set(scene_dimension_names)
```
Every scene dimension must be either spanned or extended.

### Rule 5: Dimension Name Validity
All dimension names must exist in the scene.

---

## Storage Format (Zarr)

### Metadata

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

Key fields:
- `ndim`: Intrinsic splat dimensionality (e.g., 3)
- `splat_dimensions`: Scene dimension names the splats span
- `extend_to_all`: Scene dimension names to extend across
- `cholesky_k`: Number of Cholesky elements (`ndim*(ndim+1)/2`)

### Zarr Arrays

```
gsplat_node/
  ├── .zattrs                   (metadata above)
  ├── centers                   (N, ndim) float32
  ├── amplitudes                (N,) or (1,) float32
  ├── cholesky_factors          (N, k) or (1, k) float32, k=ndim*(ndim+1)/2
  └── colors                    (N, 3) or (1, 3) uint8/float32
```

Arrays use intrinsic dimensionality, not scene dimensionality.

---

## Viewer Processing

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
}
```

### nD to 3D Projection

When rendering with `displayDims = [d0, d1, d2]` and a `slicePosition`:

1. **Map display dimensions to splat indices** — For each display dimension, find the corresponding splat dimension index (or `null` if the display dimension is extended/not in splat)

2. **Extract 3D centers** — For spanned display dimensions, read from splat centers. For extended display dimensions, place at current slice position.

3. **Extract 3D Cholesky submatrix** — Extract the 3x3 sub-covariance corresponding to displayed splat dimensions.

4. **Compute attenuation in hidden dimensions** — For spanned dimensions that are neither displayed nor extended, compute Gaussian falloff based on Mahalanobis distance between the splat center and slice position in that dimension. Extended dimensions never attenuate.

```typescript
// Attenuation for hidden spanned dimensions
let attenuation = 1.0;
for (const splatIdx of hiddenSplatDims) {
  const centerValue = centers[i * ndim + splatIdx];
  const sliceValue = slicePosition[sceneDimIdx];
  const diff = sliceValue - centerValue;
  const variance = getCholeskyVariance(cholesky, i, ndim, splatIdx);
  const mahalDist = Math.abs(diff) / Math.sqrt(variance);
  attenuation *= Math.exp(-0.5 * mahalDist * mahalDist);
}
amplitudes3D[i] = amplitudes[i] * attenuation;
```

---

## References

- [Luxar Zarr Format Specification](../guides/user/LUXAR_ZARR_FORMAT.md) — Scene-level format including GSplats nodes
- [GSplats Zarr Format](GSPLATS_ZARR_FORMAT.md) — Standalone `.gsplats.zarr` storage format
- [nD Transforms Specification](../guides/specs/ND_TRANSFORMS_SPEC.md) — Related nD navigation system
