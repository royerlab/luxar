# WASM TypeScript fallback

TypeScript re-implementations of every Rust/WASM kernel exposed in
`src/wasm/`. These run when the WASM module is unavailable (failed
to load, mobile Safari without WebAssembly, dev environment, unit
tests) and preserve the exact same numerical contract as the WASM
path.

## Files

| File                                  | Role                                                                        |
| ------------------------------------- | --------------------------------------------------------------------------- |
| `index.ts`                            | Aggregator implementing the `WasmModule` interface from `src/wasm/index.ts` |
| `effective-radii.ts`                  | nD effective-radius computation (Points spatial culling)                    |
| `decode.ts`                           | LUT / quantized / log-scalar decoding kernels                               |
| `projection.ts`                       | nD→3D position extraction                                                   |
| `spatial.ts`                          | nD visibility masks (per-element slab tests)                                |
| `points.ts`                           | Points-specific kernels (radius_to_visibility_mask, compact_by_mask)        |
| `lines.ts`, `lines-clipping.ts`       | Lines clipping + interpolation                                              |
| `gsplats.ts`, `gsplats-processing.ts` | GSplats Mahalanobis distance, marginal Cholesky, compaction                 |

## Public surface

A single `TypeScriptFallback` class that implements `WasmModule`. The
WASM loader (`src/wasm/index.ts`) constructs a `TypeScriptFallback`
instance whenever the real WASM module fails to load.

## Parity contract

Every TypeScript kernel here MUST produce the same numerical result
as its Rust counterpart for the same inputs, within Float32 epsilon
tolerance. This is locked in by `tests/unit/wasm/wasm-vs-typescript.test.ts`
(skipped when the WASM binary isn't built; CI builds it and runs
parity checks).

## Performance

The TypeScript path is intentionally slower (3-5×) than WASM for
projection and visibility kernels. It exists for correctness coverage
and as a fallback; users with large datasets should build the WASM
module (`make build-wasm`) for production performance.

## Constants

`MAX_SUPPORTED_DIMS = 16` is imported from
`src/config/constants.ts` — the single source of truth that the
Rust side (`src/wasm/rust/src/common.rs`) must mirror.
