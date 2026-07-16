# Luxar WASM Kernels (Rust Source)

> Performance-critical kernels for spatial indexing, nD visibility, projection,
> data decoding, line clipping, and Gaussian-splat processing. Compiled to
> WebAssembly via `wasm-bindgen` and consumed by the Luxar viewer's worker
> pipeline; mirrored line-for-line by the TypeScript fallback in
> [`../../typescript/`](../../typescript/).

The crate root is [`lib.rs`](./lib.rs). It declares every module below and
re-exports the `#[wasm_bindgen]` functions that form the public WASM ABI. The
unified TypeScript-facing interface for both this Rust implementation and the
fallback lives in [`../../types.ts`](../../types.ts); shared documentation and
build instructions are in the parent [`../README.md`](../README.md).

---

## Module Map

```
src/
├── lib.rs                  — crate root: module declarations + WASM re-exports
├── common.rs               — shared constants, ndim validation, packed-Cholesky index
├── lines_clipping.rs       — Liang-Barsky nD slab clipping + attribute interpolation
├── gsplats_processing.rs   — Mahalanobis distance, marginal Cholesky, attenuation
├── effective_radii.rs      — Pythagorean radius shrinkage when slicing through hidden dims
├── projection.rs           — extract 3D positions, bounds, compact-by-mask
├── depth_sort.rs           — back-to-front splat ordering (depth-sorting Phase 2)
└── decode.rs               — quantized / log / LUT / broadcast decoders
```

Every module ships a `#[cfg(test)]` block of native Rust unit tests (run via
`make test-wasm` or `cargo test`). The kernels are organised so that the
TypeScript fallback in `../../typescript/` has a 1:1 file mapping
(`lines_clipping.rs` ↔ `lines-clipping.ts`, `decode.rs` ↔ `decode.ts`, etc.).

---

## Shared Conventions

### 16-dimension limit

All `validate_ndim`-guarded functions support **at most 16 dimensions**. Above
16, they panic with a descriptive message and the worker layer falls back to
the TypeScript implementation. See [`common.rs`](./common.rs):

- `MAX_SUPPORTED_DIMS = 16`
- `MAX_PACKED_CHOLESKY_SIZE = 136` (= 16·17/2)
- `CHOLESKY_EPSILON = 1e-10`
- `validate_ndim(ndim, fn_name)` — panic helper with a fallback hint
- `packed_index(row, col)` — `row·(row+1)/2 + col` for lower-triangular packed
  Cholesky storage `[L00, L10, L11, L20, L21, L22, …]`

Functions that depend on these constants allocate fixed-size stack arrays
(`[f32; MAX_SUPPORTED_DIMS]`, `[f32; MAX_PACKED_CHOLESKY_SIZE]`) instead of
heap buffers, which is why the dimension cap exists in the first place.

### Buffer ABI

All WASM-exported kernels are zero-copy: callers pre-allocate `&mut [T]`
output buffers sized to the worst case (`num_items × stride`) and the kernel
returns the **count of populated elements** (`u32`) plus optional per-element
masks. The TypeScript side then slices the buffer down to the returned count.
`debug_assert!` is used on every output length to catch sizing bugs in dev
builds without paying for the check in release.

### Hot-loop optimisation patterns

The kernels share a small bag of tricks documented inline:

- **Reciprocal multiply over division** (`let inv = 1.0 / x; a * inv`) — WASM
  `f32.div` is roughly 10× slower than `f32.mul` on most engines.
- **Loop fusion** — visibility and norm computation collapsed into a single
  pass over the per-element inner loop (see
  `gsplats_processing.rs::mahalanobis_distance_internal`).
- **Branchless mask writes** — `output_mask[i] = visible as u8; count +=
visible as u32;` instead of an `if/else`.
- **Fixed-size lookup arrays** in place of `HashSet<u32>` for `display_dims`
  (see `lines_clipping.rs`, `effective_radii.rs`).
- **Stride-specialised fast paths** for `stride == 1` and `stride == 3` in
  `projection.rs::compact_by_mask`.

Manual SIMD via the `wide` crate was benchmarked and dropped: LLVM
auto-vectorisation under `wasm-opt -O3 --enable-simd` matches or beats it on
these shapes (see the note in `Cargo.toml`).

---

## Modules

### `common.rs` — shared constants and validation

Single-file utility module that everything else depends on. Exports
`MAX_SUPPORTED_DIMS`, `MAX_PACKED_CHOLESKY_SIZE`, `CHOLESKY_EPSILON`, the
`validate_ndim(ndim, fn_name)` panic helper, and `packed_index(row, col)`
(`#[inline]`) for indexing lower-triangular Cholesky storage.

### `lines_clipping.rs` — Liang-Barsky clipping in nD

The most involved kernel. Handles the five segment cases relative to the
nD slab `slice_position ± tolerance` on non-displayed dimensions:

```
A: both endpoints IN  → unclipped (t1=0, t2=1)
B: P1 IN, P2 OUT      → clip P2 (t2 < 1)
C: P1 OUT, P2 IN      → clip P1 (t1 > 0)
D: both OUT, opposite sides → clip both (segment crosses slab)
E: both OUT, same side       → invisible
```

| Function                           | Purpose                                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `clip_segment_single`              | Reference single-segment clipper. Returns `[visible, t1, t2]` as a 3-vec for JS interop.                      |
| `clip_segments_batch`              | Workhorse: clips `num_segments` segments in one WASM call into `output_visibility`, `output_t1`, `output_t2`. |
| `interpolate_clipped_positions`    | After batch clip, compute 3D start/end positions for visible segments via `display_dims`.                     |
| `interpolate_scalars_batch`        | Same compaction for per-vertex scalar attributes (widths, sharpness, …).                                      |
| `interpolate_colors_batch`         | RGB version with the inner loop unrolled across the three channels.                                           |
| `calculate_segment_lengths`        | Euclidean 3D length per visible segment (for LOD / dash patterns).                                            |
| `mark_clipped_endpoints`           | Boolean flags `t1 > 0` / `t2 < 1` per visible segment (for end-cap factor adjustment).                        |
| `lerp`, `lerp_vec3`, `distance_3d` | Scalar math helpers exposed for the TS fallback to share semantics.                                           |

The batch path replaces the `HashSet<u32>` of display dims with a fixed-size
`[bool; 16]` lookup. `dv.abs() < 1e-7` short-circuits the
parallel-to-slice case before any division. `t1 >= t2` aborts the per-segment
loop the moment the valid interval collapses.

### `gsplats_processing.rs` — Mahalanobis, marginal Cholesky, attenuation

The most numerically subtle module. Key insight: when projecting an nD
Gaussian to a subset of dimensions `S`, **the Cholesky of the marginal
covariance is generally not the row/column sub-matrix of L**. Computing it
requires reconstructing `Σ_S[i,j] = Σ_k L[s_i,k]·L[s_j,k]` and re-factorising
via Cholesky-Crout. This is what `compute_marginal_cholesky` (private,
`#[inline]`) does, and it is what the TypeScript fallback must mirror exactly
to keep parity tests green.

| Function                        | Purpose                                                                                                                                                                                                                                                     |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mahalanobis_distance`          | Forward substitution on packed `L`: solve `L·y = diff`, return `‖y‖`.                                                                                                                                                                                       |
| `extract_cholesky_submatrix`    | **Raw** row/col extraction — correct only for block-diagonal L. Documented as such; callers should prefer the attenuation helpers below.                                                                                                                    |
| `compute_gsplats_attenuation`   | Per-splat: build hidden-dim diff vector, compute correct marginal Cholesky for hidden dims, Mahalanobis distance, then a C⁰-continuous shifted Gaussian `scale · max(0, exp(-D²/2) - exp(-trunc²/2))`. Writes `output_visibility` and `output_attenuation`. |
| `extract_visible_cholesky_3d`   | For each visible splat, compute the correct marginal 3D Cholesky over `display_dims` and pack into `[L00, L10, L11, L20, L21, L22]` ready for the renderer.                                                                                                 |
| `compact_attenuated_amplitudes` | Compact `amplitudes[i] * attenuation[i]` over the visibility mask.                                                                                                                                                                                          |

The shifted Gaussian truncation eliminates a popping artifact at the
splat boundary that a raw `exp(-D²/2)` would produce when splats cross the
slab edge. `truncate` is typically 3.0 (three sigma).

### `effective_radii.rs` — Pythagorean radius shrinkage

| Function                    | Purpose                                                                         |
| --------------------------- | ------------------------------------------------------------------------------- |
| `calculate_effective_radii` | `R_eff = sqrt(R² - D²)` where D is the nD distance in non-displayed dimensions. |

Supports a per-dimension `spatial_extend_dims: &[u8]` flag distinguishing
**spatial** (continuous, contributes to D) from **discrete** (categorical /
integer-labelled, must match `slice_position[d]` within `0.5` or the point
is culled). The hot loop fuses the discrete-match check and the spatial
distance accumulation, breaking out early on a discrete mismatch. Display
dims are looked up via a fixed-size `[bool; 16]` array.

### `depth_sort.rs` — back-to-front splat ordering

| Function              | Purpose                                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| `sort_splats_by_depth` | Camera-space z per splat, min/max-normalized uint16 keys, stable 65536-bucket counting sort (back-to-front permutation for `aSortedIndex`). |

Scale-invariant (per-sort normalization — nm..km units; raw f16 keys were
deliberately rejected), stable on ties, behind-camera splats key to the far
bucket, degenerate depth ranges fall back to the identity ordering. Input is
always projected 3D centers, so the 16-dimension cap does not apply. See
`docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §5.

### `projection.rs` — extraction, bounds, compaction

| Function                   | Purpose                                                                                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `extract_3d_positions`     | Project nD positions to 3D by indexing through `display_dims` (≤3); fills unused output dims with `0.0`.                                                     |
| `calculate_bounds_3d`      | Single-pass min/max sweep into a 6-element bounds buffer `[minX, minY, minZ, maxX, maxY, maxZ]`. Returns `0` and zero bounds on empty input.                 |
| `compact_by_mask`          | Stride-specialised compaction. Fast paths for `stride=1` (scalars) and `stride=3` (vec3) with an unrolled vec3 copy; generic fallback for arbitrary strides. |
| `count_visible`            | `popcount`-style scan over a `u8` mask.                                                                                                                      |
| `radii_to_visibility_mask` | `mask[i] = radii[i] > threshold` (strictly greater — `radii[i] == threshold` is treated as hidden).                                                          |

### `decode.rs` — array decoding

The decoders are the simplest kernels but dominate wall-clock time for
large chunks. All hot loops precompute the scale factor outside the loop and
use direct array indexing.

| Function                                         | Purpose                                                                                                                                                           |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `decode_quantized_u8` / `decode_quantized_u16`   | Linear dequantization: `output = min + raw · (max - min) / MAX_RAW`.                                                                                              |
| `decode_log_scalar_u8` / `decode_log_scalar_u16` | Log-space dequantization for wide-dynamic-range positive scalars (radii). Uses `f32::exp_m1` for accuracy at small values.                                        |
| `decode_lut_scalar_u8` / `decode_lut_scalar_u16` | Scalar LUT lookup: `output[i] = lut[indices[i]]`.                                                                                                                 |
| `decode_lut_row_u8` / `decode_lut_row_u16`       | Row LUT lookup with stride `k` (for vec3 / vec4 attributes); uses `slice::copy_from_slice` for the inner copy.                                                    |
| `decode_broadcasted`                             | Expand a single value (or short vector) to `num_points × elements_per_point`. Fast path when `value.len() >= elements_per_point`; otherwise pads with `value[0]`. |

---

## Build

These sources are not compiled directly by `cargo build`. The standard build
is driven from the repo root:

```bash
make build-wasm     # wasm-pack build (release) → packages/luxar-viewer/public/wasm/
make test-wasm      # cargo test (native target, runs every #[cfg(test)] block)
make benchmark-wasm # WASM-vs-TypeScript performance suite (browser harness)
```

Release profile (`Cargo.toml`): `opt-level=3`, `lto=true`, `codegen-units=1`,
`panic=abort`, `strip=true`. `wasm-opt` is invoked with `-O3 --enable-simd
--enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext`.

## Testing

Native unit tests live alongside each module (`#[cfg(test)] mod tests`) and
exercise the algorithms with small hand-checked fixtures: identity Cholesky
factors, 4D-with-hidden-dim splat slices, segments crossing a slab,
Pythagorean radius shrinkage with both spatial and discrete dimensions, etc.

The TypeScript fallback in [`../../typescript/`](../../typescript/) has a
corresponding `*.test.ts` parity suite that drives both implementations
through identical fixtures via [`../../types.ts`](../../types.ts) and asserts
byte-equivalent outputs.

## See Also

- [`../README.md`](../README.md) — Rust crate manifest and build wiring.
- [`../../README.md`](../../README.md) — WASM module overview, ABI, loader,
  fallback strategy.
- [`../../typescript/`](../../typescript/) — pure-TypeScript mirror of every
  kernel (used when WASM is unavailable or `ndim > 16`).
- [`../../types.ts`](../../types.ts) — unified `WasmModule` interface
  contracted against both implementations.
