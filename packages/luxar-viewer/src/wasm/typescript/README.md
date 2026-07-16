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

| File                    | Role                                                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `index.ts`              | Barrel re-exporting every kernel and the `TypeScriptFallback` class implementing the `WasmModule` interface      |
| `lines-clipping.ts`     | Liang-Barsky segment clipping, batched position/scalar/color interpolation, segment lengths, clipped-end flags   |
| `gsplats-processing.ts` | Marginal Cholesky factorization, Mahalanobis distance, shifted-Gaussian attenuation, visible-Cholesky extraction |
| `effective-radii.ts`    | `calculate_effective_radii` — `R_eff = sqrt(R² − D²)` for nD points sliced by a hyperplane                       |
| `decode.ts`             | LUT / quantized / log-scalar / geolog-scalar / per-channel (linear, log, signed-log, geolog) decoders + `decode_broadcasted` |
| `projection.ts`         | nD→3D position extraction, AABB bounds, `compact_by_mask`, `count_visible`, `radii_to_visibility_mask`           |
| `depth-sort.ts`         | `sort_splats_by_depth` — back-to-front splat ordering; frounds every float step in the Rust op order so WASM↔TS parity is exact-permutation |

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

Every TypeScript kernel here MUST produce the same numerical result
as its Rust counterpart for the same inputs, within Float32 epsilon
tolerance. This is locked in by
`src/tests/unit/wasm/wasm-vs-typescript.test.ts` (skipped when the
WASM binary isn't built; CI builds it and runs parity checks).

## Performance

The TypeScript path is intentionally slower (3-5×) than WASM for
projection and visibility kernels. It exists for correctness coverage,
as a fallback, and as the uncapped production backend for >16D data
(WASM supports at most 16 dimensions); users with large datasets
should build the WASM module (`make build-wasm`) for production
performance.

## Constants

`MAX_SUPPORTED_DIMS = 16` is imported from
`../../config/constants.ts` (the single source of truth) and is used
by `gsplats-processing.ts` to size module-level workspace buffers for
the marginal Cholesky reconstruction. The Rust side
(`src/wasm/rust/src/common.rs`) must mirror this value.
