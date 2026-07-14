# Gaussian Splats Dimension Mapping

**Version**: 2.0
**Last Updated**: 2026-07-13

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
    labels: Optional[Sequence[str]] = None,        # Per-splat hover labels (CSR text)
    image_labels: Optional[Any] = None,            # Per-splat hover thumbnails (CSR images)
    parent: Optional[Node] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    partition: Any = None,                         # True / dict(max_elements=..., rule=...) →
                                                   #   compile-time kind=partition wrapper
    **attrs: Any,
) -> Union[GSplats, Group]:
```

**`dim_order`** — Maps data columns to scene dimensions by name. Also reorders and embeds Cholesky factors automatically. For example, `dim_order=["z", "y", "x"]` declares that the first column of `centers` corresponds to the scene's "z" dimension, etc.

When `dim_order` is specified and the data has fewer dimensions than the scene, the `fill` and `fill_sigma` parameters provide fixed coordinates and covariance widths for the unmapped dimensions. The method automatically expands the centers and Cholesky factors to scene dimensionality.

Auto-mapping behavior when `None`:
- If `splat_ndim == scene_ndim`: auto-map to all scene dimensions in order
- If `splat_ndim < scene_ndim` and `extend_to_all` covers the remaining dims: auto-map to spatial dimensions
- Otherwise: raises an error

**`extend_to_all`** — Which scene dimensions the splats should be visible across without attenuation. Can be a list of dimension names or a single name string. Must not overlap with mapped dimensions. Extended dimensions are never attenuated by slice position.

**`fill`** — Fixed coordinate values for dimensions not covered by `dim_order`. For example, `fill={"time": 5.0}` places all splats at time=5.0.

**`fill_sigma`** — Standard deviations for unmapped dimensions in the Cholesky embedding (default 1.0). Controls splat extent in filled dimensions.

---

## Usage Examples

### 3D Splats in 4D Scene (Extension)

```python
result = fit_gaussian_splats(volume_3d, num_splats=10000)
# result.centers.shape = (10000, 3)
# result.cholesky_factors.shape = (10000, 6)

from luxar import LuxarZarrCompiler, Dimensions, Dimension

dims = Dimensions([
    Dimension("x", display=True), Dimension("y", display=True),
    Dimension("z", display=True), Dimension("time", display=False),
])
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    scene.add_gsplats(
        "cell_structure",
        centers=result.centers,              # (N, 3) - native dimensionality
        amplitudes=result.amplitudes,
        cholesky_factors=result.cholesky_factors,  # (N, 6) - 3D covariance
        colors=result.colors,
        dim_order=["x", "y", "z"],           # Splats span spatial dims
        extend_to_all=["time"]               # Visible at all timepoints
    )
```

### Full-Dimensional Splats (Automatic)

```python
result = fit_gaussian_splats(data_4d, num_splats=50000)
# result.centers.shape = (50000, 4)

from luxar import LuxarZarrCompiler, Dimensions, Dimension

dims = Dimensions([
    Dimension("x", display=True), Dimension("y", display=True),
    Dimension("z", display=True), Dimension("time", display=False),
])
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    scene.add_gsplats(
        "dynamic_process",
        centers=result.centers,              # (N, 4)
        amplitudes=result.amplitudes,
        cholesky_factors=result.cholesky_factors,  # (N, 10)
        # dim_order=None → auto-maps to all scene dimensions
    )
```

### Multiple Extended Dimensions

```python
from luxar import LuxarZarrCompiler, Dimensions, Dimension

dims = Dimensions([
    Dimension("x", display=True), Dimension("y", display=True),
    Dimension("z", display=True), Dimension("time", display=False),
    Dimension("channel", display=False),
])
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    result = fit_gaussian_splats(volume_3d, num_splats=5000)

    scene.add_gsplats(
        "nuclei",
        centers=result.centers,              # (N, 3)
        amplitudes=result.amplitudes,
        cholesky_factors=result.cholesky_factors,  # (N, 6)
        dim_order=["x", "y", "z"],
        extend_to_all=["time", "channel"]    # Visible at all times and channels
    )
```

### Dimension Reordering with Fill

```python
from luxar import LuxarZarrCompiler, Dimensions, Dimension

dims = Dimensions([
    Dimension("x", display=True), Dimension("y", display=True),
    Dimension("z", display=True), Dimension("time", display=False),
])
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Centers are in [z, y, x] order from fitting
    result = fit_gaussian_splats(data_zyx, num_splats=1000)

    scene.add_gsplats(
        "reordered",
        centers=result.centers,              # (N, 3) in [z,y,x] order
        amplitudes=result.amplitudes,
        cholesky_factors=result.cholesky_factors,
        dim_order=["z", "y", "x"],          # Explicit ordering by name
        extend_to_all=["time"]
    )
```

---

## Validation Rules

### Rule 1: Shape Consistency
```
len(dim_order) == centers.shape[1]
```
The number of specified dimensions must match the splat data dimensionality.

### Rule 2: Cholesky Consistency
```
cholesky_factors.shape[1] == splat_ndim * (splat_ndim + 1) // 2
```
Cholesky factors must match splat dimensionality, not scene dimensionality.

### Rule 3: Mutual Exclusivity
```
set(dim_order) & set(extend_to_all) == {}
```
A dimension cannot be both spanned and extended:
- **Spanned** (`dim_order`): Splat has position and covariance in this dimension
- **Extended** (`extend_to_all`): Splat is visible everywhere in this dimension

### Rule 4: Complete Coverage
```
set(dim_order) | set(extend_to_all) | set(fill.keys()) == set(scene_dimension_names)
```
Every scene dimension must be covered by `dim_order`, `extend_to_all`, or `fill`.

### Rule 5: Dimension Name Validity
All dimension names must exist in the scene.

---

## Storage Format (Zarr)

### Metadata

After dimension mapping is applied, the compiler writes scene-dimensionality data. The zarr metadata reflects the final (expanded) dimensionality:

```json
{
  "type": "gsplats",
  "n_splats": 10000,
  "ndim": 4,
  "has_colors": true,
  "extend_to_all": ["time"]
}
```

Key fields:
- `ndim`: Final splat dimensionality after expansion (matches scene dimensionality). The viewer derives `cholesky_k` from this as `ndim*(ndim+1)/2`.
- `extend_to_all`: Scene dimension names the splats extend across (visible everywhere). Stored only when non-empty.
- `has_colors`: Whether the node has per-splat colors

### Zarr Arrays

```
gsplat_node/
  ├── .zattrs                   (metadata above)
  ├── centers                   (N, ndim) uint16 (AUTO; float32 if an axis extent ≥ 2¹⁶) / float32 (PRECISION)
  ├── amplitudes                (N,) or (1,) uint8/uint16 (AUTO) / float32 (PRECISION)
  ├── cholesky_factors_diag     (N, d) or (1, d) uint8 (AUTO, certified — escalates to uint16 if the covariance certificate fails) / float32 (PRECISION), d=ndim (diagonal, scale-like terms)
  ├── cholesky_factors_offdiag  (N, k-d) or (1, k-d) uint8 (AUTO, certified as above) / float32 (PRECISION) (signed off-diagonal; omitted when ndim==1; k=ndim*(ndim+1)/2)
  └── colors                    (N, 3) or (1, 3) uint8/uint16 (AUTO) / float32 (PRECISION)
```

On-disk dtypes follow the encoding mode (default `AUTO` quantizes; all arrays
decode to float32 on read — see
[GSplats Zarr Format](GSPLATS_ZARR_FORMAT.md) for the per-array encodings).

Arrays use scene dimensionality (after `dim_order` expansion), not the original data dimensionality. Since format **v3.1** the Cholesky factors are stored on disk split into `cholesky_factors_diag` (N, ndim) + `cholesky_factors_offdiag` (N, k-ndim) — each encoded/quantized independently — and recombined into the packed (N, k) `cholesky_factors` form immediately on read (Python reader and viewer loader), so nothing downstream of the storage boundary sees the split. (1D splats have no off-diagonal terms, so `cholesky_factors_offdiag` is omitted; legacy v3.0 files store a single packed `cholesky_factors`, read via presence-detect fallback.)

---

## Viewer Processing

### Loading Phase

```typescript
interface GSplatsMetadata {
  type: 'gsplats';
  n_splats: number;
  ndim: number;                      // Splat dims (matches scene ndim after expansion)
  has_colors: boolean;
  chunk_size: number;
  ordering: 'morton' | 'hilbert' | 'none';
  extend_to_all?: string[];          // Extended dim names (e.g., ["time"])
  // ... plus ordering bounds, transform, rendering params
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
