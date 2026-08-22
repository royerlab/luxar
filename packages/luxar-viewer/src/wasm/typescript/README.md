# WASM TypeScript fallback

TypeScript re-implementations of every Rust/WASM kernel exposed in
`src/wasm/`. These run when the WASM module is unavailable (failed
to load, mobile Safari without WebAssembly, dev environment, unit
tests) and preserve the exact same numerical contract as the WASM
path. They are also the **production path for >16D data**: the WASM
kernels use fixed-size 16-dim arrays, so `pickBackend(ctx, ndim)` in
`src/workers/data-worker/state.ts` routes every `ndim > 16` operation
here automatically — not just when WASM is missing.

## Files

| File                    | Role                                                                                                                                                                                                |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `index.ts`              | Barrel re-exporting every kernel and the `TypeScriptFallback` class implementing the `WasmModule` interface                                                                                         |
| `lines-clipping.ts`     | Liang-Barsky segment clipping, batched position/scalar/color interpolation, segment lengths, and per-endpoint cap suppression; frounds every float step in the Rust op order for exact WASM parity  |
| `mesh-culling.ts`       | Whole-triangle nD culling: per-vertex slab membership + face compaction preserving original vertex indices. `Math.fround`s the slab bounds                                                          |
| `gsplats-processing.ts` | Marginal Cholesky factorization, Mahalanobis distance, fused nD→3D projection (`project_gsplats_nd_to_3d`). Frounds every float step in the Rust op order (#1820); exact vs WASM except `exp`/`log` |
| `effective-radii.ts`    | `calculate_effective_radii` — `R_eff = sqrt(R² − D²)` for nD points sliced by a hyperplane. Frounds every float step in the Rust op order (#1820); bit-exact vs WASM                                |
| `decode.ts`             | LUT / quantized / log-scalar / geolog-scalar / per-channel decoders + `decode_broadcasted`; log-scalar is bit-exact with WASM and is now the single route the main-thread `ArrayDecoder` uses too   |
| `float32-math.ts`       | Exact TypeScript port of Rust/WASM `expm1f`, used where narrowing the host's f64 `Math.expm1` result can choose the neighboring f32                                                                 |
| `projection.ts`         | nD→3D position extraction (`extract_3d_positions`)                                                                                                                                                  |
| `depth-sort.ts`         | `sort_splats_by_depth` — back-to-front splat ordering; frounds every float step in the Rust op order so WASM↔TS parity is exact-permutation                                                         |

## Public surface

A single `TypeScriptFallback` class that implements the `WasmModule`
interface from `../types.ts`. The WASM loader constructs a
`TypeScriptFallback` instance whenever the real WASM module fails to
load; the data worker also keeps one alongside compiled WASM
(`WasmCtx.tsFallback`) so `pickBackend` can serve `ndim > 16`
operations that the WASM kernels cannot handle. Every individual
kernel is also exported by name from `index.ts` for direct use
(e.g. by unit tests).

## Parity contract

Every TypeScript kernel here MUST produce **the same f32** as its Rust
counterpart for the same inputs — bit-for-bit, not "within a Float32
epsilon". This is locked in by
`src/tests/unit/wasm/wasm-vs-typescript.test.ts` (skipped when the
WASM binary isn't built; CI builds it and runs parity checks).

"Within Float32 epsilon" was the contract until #1820 and it was the
wrong one. These kernels feed **decisions**, not just displayed digits:
`calculate_effective_radii` compares a distance against a radius,
`project_gsplats_nd_to_3d` compares an amplitude against `minAmplitude`.
A sub-epsilon difference there changes the element COUNT one backend
emits, so a tolerance-based test passes while the two backends render
different scenes. New parity cases must assert exact equality wherever
exactness is reachable, and a MEASURED ulp/absolute bound (with the
fixture that produced it) only where it is not.

The two places exactness is not currently reachable are
`Math.exp`/`Math.log`: V8's ieee754 kernels are a different approximation
from the wasm libm's `expf`/`logf`. That is a scope decision, not a
limit — a frounded TypeScript port of musl's `expf` measures 0/200 000
mismatches against the real WASM — and it is tracked as **#1830**.

### Per-file f32 status

This directory is **not** uniformly verified. Current state:

| File                    | f32 discipline                                                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `depth-sort.ts`         | Frounded; sort permutation exact vs WASM                                                           |
| `mesh-culling.ts`       | Slab bounds frounded; parity-tested                                                                |
| `effective-radii.ts`    | Frounded end to end (#1820); bit-exact vs WASM over a 20k randomized sweep                         |
| `gsplats-processing.ts` | Frounded end to end (#1820); bit-exact except the `exp`/`log` residual (#1830)                     |
| `decode.ts`             | Frounds Rust-f32 inputs and decode arithmetic; log-scalar output is bit-exact with Rust's `expm1f` |
| `lines-clipping.ts`     | Frounded end to end (#1821); exact parity cases cover slab boundaries and t-parameter accumulation |
| `projection.ts`         | Pure copy, no arithmetic                                                                           |

The viewer's full-array and range-based `ArrayDecoder` log-scalar paths delegate to
`decode_log_scalar_u8` / `decode_log_scalar_u16` here. The Rust/WASM f32 kernel is the
viewer decode contract, so the main-thread and TypeScript worker-fallback routes now use
the same f32 operation order, and both are bit-exact with the Rust `expm1f` the WASM
kernel runs (#1843). The Python decoder remains a separate f64 metadata helper.

### Where `Math.fround` is mandatory

JS numbers are f64. Reading a `Float32Array` yields an exact f32 widened to f64,
and **arithmetic on those values is then done in f64**, which does not round the
way the Rust f32 kernel does. Anywhere a comparison can land on the boundary,
that difference is observable and must be closed with `Math.fround` at each step,
in the Rust operation order:

- `depth-sort.ts` frounds every step so the sort permutation is exact.
- `decode.ts` frounds linear bounds, geolog-scalar anchors, and the log-scalar
  `maxLog` bound, all of which arrive as f32 in WASM. Its log-scalar kernels use
  `float32-math.ts`'s `expm1f`, because narrowing host `Math.expm1` afterward can
  still choose the neighboring f32. The geolog-scalar, log-per-channel, and
  signed-log-per-channel decoders deliberately retain f64 `Math.exp`/`Math.expm1`
  because their Rust transforms also evaluate in f64 before narrowing.
- `mesh-culling.ts` frounds the slab bounds. The f64 difference of two f32 values
  is _exact_ while Rust's f32 subtraction rounds; the gap is under half an ulp,
  but when the rounding goes DOWN the rounded bound is itself a legal f32 vertex
  coordinate, and a vertex sitting exactly there is visible in WASM and culled in
  unfrounded TS (`slice=1.0, tolerance=0.1` is such a case, and is a parity test).
- `effective-radii.ts` frounds the Pythagorean accumulation, the discrete
  tolerance subtraction and the `R² − D²` cancellation. Skipping any of them lets
  a point on the rim of its own radius be visible on one backend and culled on
  the other, changing `visibleCount`.
- `gsplats-processing.ts` frounds the Σ_S dot product, the Crout reduction, both
  forward substitutions, the shifted-Gaussian chain and the final
  `amplitude × attenuation` product — and narrows four values Rust types as
  `f32`: `CHOLESKY_RELATIVE_EPSILON`, `CHOLESKY_EPSILON`, the
  forward-substitution epsilon, and the `min_amplitude`/`truncate` parameters
  wasm-bindgen narrows at the boundary.

An accumulator that lands in a `Float32Array` is **not** self-rounding: the
store rounds the final value only, so an f64 running sum silently carries extra
precision through the whole reduction. `sqrt` and `/` on operands that are
already f32 need no explicit fround (f64's 53 bits ≥ 2·24 + 2 makes the double
rounding provably benign) — it is the accumulators and the comparisons that do.

- `lines-clipping.ts` frounds every clipping and interpolation step in Rust's
  operation order. Exact parity tests cover both the `slice=1.0, tolerance=0.1`
  slab boundary and the separately rounded reciprocal used for t-parameters.

## Performance

The TypeScript path is intentionally slower (3-5×) than WASM for
projection and visibility kernels. It exists for correctness coverage,
as a fallback, and as the uncapped production backend for >16D data
(WASM supports at most 16 dimensions); users with large datasets
should build the WASM module (`make build-wasm`) for production
performance.

The exact `expm1f` port is about 4.4× slower than narrowing `Math.expm1`
for the transcendental step (2 million u16 codes on Node 22). Log-scalar
decoding reaches it on two routes: the wholesale TypeScript fallback
selected when the WASM artifact fails to load, and — since the main-thread
`ArrayDecoder` was unified onto this kernel — every main-thread log-scalar
decode, whether or not WASM loaded. For inputs larger than the code space,
`ArrayDecoder` decodes the at-most-65,536-entry code space once and fills from
that lookup table. On Node 22, 5M widened u16 codes measured ~50 ms through the
lookup-table path versus ~395 ms through the kernel directly, while remaining
bit-exact with the shared kernel.

## Constants

`MAX_SUPPORTED_DIMS = 16` is imported from
`../../config/constants.ts` (the single source of truth) and is used
by `gsplats-processing.ts` to size module-level workspace buffers for
the marginal Cholesky reconstruction. That size is only the initial
allocation: because this reference is the uncapped `ndim > 16` backend
(WASM panics above the cap, so `pickBackend` routes those operations
here), the buffers grow on demand when the continuous hidden-dim count
exceeds 16, and the `ndim <= 16` path stays allocation-free. The Rust
side (`src/wasm/rust/src/common.rs`) must mirror this value.

`CHOLESKY_EPSILON = 1e-10` comes from the same place: `gsplats-processing.ts`
imports `GSPLAT_CHOLESKY_EPSILON` from `../../config/constants.ts` under the
kernel-local name, because the loaders' gsplats chunk-fetch epsilon must cover
the band this constant makes a degenerate hidden dimension render in. The Rust
side keeps an independent copy (separate language) and is pinned against the
config value by
`src/tests/unit/data/loaders/spatial-query/tolerance-computer.test.ts`.
