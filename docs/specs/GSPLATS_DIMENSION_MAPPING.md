# Gaussian Splats Dimension Mapping

**Version**: 2.0
**Last Updated**: 2026-08-26

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

`cholesky_factors` packs the lower-triangular factor L of the covariance
(Σ = L·Lᵀ) in row-major order. Its diagonal is scale-like: isotropic std σ in
3D uses `[σ, 0, σ, 0, 0, σ]`, not `1/σ`.

**`dim_order`** — Maps data columns to scene dimensions by name. Also reorders and embeds Cholesky factors automatically. For example, `dim_order=["z", "y", "x"]` declares that the first column of `centers` corresponds to the scene's "z" dimension, etc.

When `dim_order` is specified and the data has fewer dimensions than the scene, the `fill` and `fill_sigma` parameters provide fixed coordinates and covariance widths for the unmapped dimensions. The method automatically expands the centers and Cholesky factors to scene dimensionality.

Auto-mapping behavior when `None`:
- If `splat_ndim == scene_ndim`: auto-map to all scene dimensions in order
- If `splat_ndim < scene_ndim` and `extend_to_all` covers the remaining dims: auto-map to spatial dimensions
- Otherwise: raises an error

**`extend_to_all`** — Which scene dimensions the splats should be visible across without attenuation. Can be a list of dimension names or a single name string. Must not overlap with mapped dimensions. Extended dimensions are never attenuated by slice position.

**`fill`** — Fixed coordinate values for dimensions not covered by `dim_order`. For example, `fill={"time": 5.0}` places all splats at time=5.0.

**`fill_sigma`** — Standard deviations for unmapped dimensions in the Cholesky embedding (default 1.0). Controls splat extent in filled dimensions.

For a hand-authored stacked time/channel axis from lower-dimensional splats,
use `dim_order=["x", "y", "z"]`, pair `fill={"time": t}` with
`fill_sigma={"time": 0.0}`, and pass `extend_to_all=[]` (omitting it
auto-broadcasts every unmapped dimension). The embedding treats zero as semantic
no-extent and regularizes that semantic zero to `1e-7` before Cholesky
decomposition. If the input already has full scene-dimensional centers and
factors without `dim_order` embedding, author a strictly positive diagonal
smaller than the coordinate step, because the public writer rejects a literal
zero. Use `extend_to_all` instead when the geometry should remain visible at every
coordinate.

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
Their values follow the covariance-factor packing described above.

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

2. **Extract 3D centers** — For spanned display dimensions, read from splat centers. For extended display dimensions, place at current slice position. Components with no display dimension behind them (a 2D or 1D view) are zero-filled. The visible centre is the stored coordinate on each displayed dimension, read **directly** — it is not shifted by any correlation between the displayed and hidden dimensions (see step 3).

3. **Extract 3D Cholesky submatrix** — Compute the **marginal** Cholesky of the sub-covariance over the displayed splat dimensions: the displayed 3D covariance is the marginal Σ_dd (the displayed rows/columns of Σ), **not** the conditional covariance Σ_dd − Σ_dh·Σ_hh⁻¹·Σ_hd (Schur complement) of the slice through the hidden coordinate. This is a deliberate visualisation choice: every splat is drawn with its full displayed footprint at its own centre, and the slice position only *attenuates* it (step 4). For categorical/discrete hidden axes such as time the distinction is moot — those axes are gated exactly, and carry no covariance worth conditioning on. For a continuous hidden axis correlated with a displayed one (a non-zero Σ_dh), the exact conditional slice would shift the visible centre by Σ_dh·Σ_hh⁻¹·(s − μ_h) and narrow the footprint to the Schur complement; that conditional projection is **not implemented**. Note the renderer's output buffer is **always** the 6-element packed-3D layout `[L00, L10, L11, L20, L21, L22]`, regardless of how many dimensions are displayed, so with `n = min(displayDims.length, 3) < 3` the marginal is only n×n and the remaining rows are **synthesized**, not extracted:

   - Off-diagonals are `0` — the phantom axis is uncorrelated with the real ones, leaving the in-plane profile exactly as authored.
   - The diagonal is the **geometric mean of the real Cholesky pivots**, which equals `(det Σ_S)^(1/2n)` and is therefore rotation-invariant. A 2D splat thus renders as a round blob at its own in-plane scale.

   The phantom diagonal is deliberately **not** a small epsilon. In sum projection (additive, luminous, volumetric) the shader scales amplitude by the Gaussian's extent along the view ray, `sigmaRay = 1/√(rᵀΣ⁻¹r)`, so an ε-thin splat viewed face-on is scaled by ~1e-5 and discarded — the scene renders black. `luxar.gsplats.lift` depends on this directly: it calibrates amplitude as `opacity / (rayIntegralFactor · σ)`, which holds for a 2D lift only because `√(σ·σ) == σ`.

   See `wasm/rust/src/gsplats_processing.rs::compute_display_cholesky_3d` and its TypeScript twin.

4. **Gate and attenuate in hidden dimensions** — Spanned dimensions that are neither displayed nor extended are handled in two passes. Extended dimensions never attenuate.

   - **Discrete hidden dims** (categorical/discrete axes, e.g. time) are a hard gate, precomputed on the TypeScript side before the kernel runs: a splat is visible only if its centre lies within half a step (`discreteSteps[dim] * 0.5`, default step `1`) of the slice position on *every* discrete hidden dim. No Gaussian falloff is applied on these axes.
   - **Continuous hidden dims** attenuate the amplitude by **one joint Gaussian**, not a per-dimension product. The kernel forms the offset vector `diff = slicePosition − center` over all continuous hidden dims, extracts the **marginal** Cholesky of Σ_hh over exactly those dims (`compute_marginal_cholesky`, so hidden–hidden correlations are honoured), and evaluates a single Mahalanobis distance `d² = diffᵀ·Σ_hh⁻¹·diff`. The falloff is a **shifted, truncated Gaussian** that reaches exactly zero at the truncation radius (C⁰-continuous, so a splat crossing the cutoff fades out rather than popping):

     `attenuation = max(0, (exp(−d²/2) − c) / (1 − c))`, with `c = exp(−truncate²/2)`

   Only the amplitude is multiplied by `attenuation`; the 3D centre and Cholesky from steps 2–3 are written unchanged. A splat whose attenuated amplitude falls below `min_amplitude` (or is NaN) is dropped from the compacted output.

```rust
// Fused kernel — gsplats_processing.rs::project_gsplats_nd_to_3d, per splat i.
if discrete_visibility[i] == 0 { continue; }            // hard gate (precomputed in TS)

let attenuation = if continuous_hidden_dims.is_empty() { 1.0 } else {
    for (h, &dim) in continuous_hidden_dims.iter().enumerate() {
        diff[h] = slice_position[dim] - positions[i * ndim + dim];
    }
    // Marginal Cholesky of Σ_hh over the continuous hidden dims (correlations kept).
    compute_marginal_cholesky(cholesky, i * packed_size, continuous_hidden_dims,
                              num_continuous, &mut hidden_cholesky);
    let d = mahalanobis_distance_internal(&diff[..num_continuous], &hidden_cholesky, num_continuous);
    let c = (-0.5 * truncate * truncate).exp();          // shift so the tail hits 0 at `truncate`
    (((-0.5 * d * d).exp() - c) / (1.0 - c)).max(0.0)
};

let amplitude3d = amplitudes[i] * attenuation;
if amplitude3d < min_amplitude || amplitude3d.is_nan() { continue; }   // culled
// centers3d / cholesky3d (steps 2–3) are emitted unattenuated; only the amplitude changes.
```

   The TypeScript reference kernel in `wasm/typescript/gsplats-processing.ts` (the >16D path) applies the same shifted-Gaussian truncation.

---

## References

- [Luxar Zarr Format Specification](../guides/user/LUXAR_ZARR_FORMAT.md) — Scene-level format including GSplats nodes
- [GSplats Zarr Format](GSPLATS_ZARR_FORMAT.md) — Standalone `.gsplats.zarr` storage format
- [nD Transforms Specification](../guides/specs/ND_TRANSFORMS_SPEC.md) — Related nD navigation system
