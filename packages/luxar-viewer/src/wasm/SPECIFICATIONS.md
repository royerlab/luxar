# luxar-viewer.wasm - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-05-06

## Purpose

The `luxar-viewer.wasm` package provides a unified high-performance compute
layer for spatial queries, nD visibility, line clipping, GSplat math, and
array decoding. It exposes a single `WasmModule` interface backed by either
a Rust/WebAssembly implementation or a pure TypeScript fallback, chosen
at runtime so the rest of the viewer never has to fork.

**Core Responsibility**: Take the viewer's hot inner loops — the ones that
run per-chunk, per-frame, or per-element on millions of items — and run
them at the speed Rust gives us, while keeping a correctness-equivalent TS
fallback so dev/test/unsupported-browsers paths stay working.

---

## Table of Contents

1. [Loader Contract](#1-loader-contract)
2. [WasmModule Interface](#2-wasmmodule-interface)
3. [Validation Boundaries](#3-validation-boundaries)
4. [16-Dimension Limit](#4-16-dimension-limit)
5. [TypeScript Fallback](#5-typescript-fallback)
6. [Memory + Transferable Discipline](#6-memory--transferable-discipline)
7. [Build Pipeline](#7-build-pipeline)
8. [Versioning + Compatibility](#8-versioning--compatibility)

---

## 1. Loader Contract

```typescript
import { initWasm, isWasmSupported, getFallback } from './wasm';

const wasm = await initWasm();
```

Behaviour:

- **First call**: tries to fetch `public/wasm/luxar_wasm_bg.wasm`,
  instantiates it, and returns a `WasmModule`. On any failure (build
  artifact missing, fetch error, WASM disabled by the host) it logs a
  warning and returns the TypeScript fallback instead. Either way the
  shape of the returned object is the same.
- **Subsequent calls**: return the cached promise from the first call —
  the loader is idempotent.
- **`isWasmSupported()`**: synchronous probe. Returns `true` only when
  `WebAssembly` is defined *and* `instantiateStreaming` is available.
- **`getFallback()`**: forces the TypeScript implementation — useful for
  parity tests.

The DataWorker and the JS-side data loaders never see which backend they
got. The only difference between WASM and TS is performance.

---

## 2. WasmModule Interface

The interface is grouped by concern. Every method is synchronous (no
Promise returns) — both backends are CPU-bound and run on a worker
thread.

### 2.1 Spatial Queries

| Method                        | Purpose                                      |
| ----------------------------- | -------------------------------------------- |
| `query_chunks_for_view(...)`  | Find chunks intersecting an nD slice         |

### 2.2 nD Visibility

| Method                              | Purpose                                       |
| ----------------------------------- | --------------------------------------------- |
| `compute_nd_visibility_points(...)` | Hypersphere intersection for point clouds     |
| `compute_nd_visibility_lines(...)`  | Endpoint-based visibility for line segments   |
| `compute_nd_visibility_gsplats(...)`| Ellipsoid extent for Gaussian splats          |
| `calculate_effective_radii(...)`    | Slice-through-higher-D radius                 |
| `radii_to_visibility_mask(...)`     | Convert effective radii to a boolean mask     |

### 2.3 Decoding (multi-typed-array hot path)

| Method                                                  | Purpose                          |
| ------------------------------------------------------- | -------------------------------- |
| `decode_quantized_u8` / `decode_quantized_u16`          | Dequantise compressed arrays     |
| `decode_log_scalar_u8` / `decode_log_scalar_u16`        | Log-space decoding               |
| `decode_lut_scalar_u8` / `decode_lut_scalar_u16`        | LUT scalar decoding              |
| `decode_lut_row_u8` / `decode_lut_row_u16`              | LUT row (vector) decoding        |
| `decode_broadcasted`                                    | Expand broadcast-encoded arrays  |

### 2.4 Projection

| Method                       | Purpose                                       |
| ---------------------------- | --------------------------------------------- |
| `extract_3d_positions`       | Extract display dims from nD positions        |
| `calculate_bounds_3d`        | Compute 3D bbox from nD data                  |
| `compact_by_mask`            | Remove invisible elements by boolean mask     |
| `count_visible`              | Count non-zero mask values                    |

### 2.5 Gaussian Splats

| Method                              | Purpose                                  |
| ----------------------------------- | ---------------------------------------- |
| `compute_gsplats_attenuation`       | Gaussian attenuation from nD distance    |
| `extract_visible_cholesky_3d`       | Extract 3D Cholesky factors from nD      |
| `compact_attenuated_amplitudes`     | Remove fully attenuated splats           |
| `extract_cholesky_submatrix`        | Raw Cholesky submatrix for given dims    |
| `mahalanobis_distance`              | nD Mahalanobis distance                  |

### 2.6 Line Clipping

| Method                              | Purpose                                  |
| ----------------------------------- | ---------------------------------------- |
| `clip_segment_single`               | Clip one segment, return interp params   |
| `clip_segments_batch`               | Clip many segments to nD slice bounds    |
| `interpolate_clipped_positions`     | Reposition endpoints at clip boundaries  |
| `interpolate_scalars_batch`         | Interpolate scalar attributes            |
| `interpolate_colors_batch`          | Interpolate color attributes             |
| `calculate_segment_lengths`         | Compute segment lengths for LOD          |
| `mark_clipped_endpoints`            | Mark endpoints for cap-factor adjustment |
| `lerp` / `lerp_vec3` / `distance_3d`| Math helpers                             |

---

## 3. Validation Boundaries

Crossing into WASM with a malformed buffer either crashes the worker
(memory misalignment, OOB read) or produces silently wrong output. Every
public entry point now has a JS-side validator that throws a descriptive
`Error` *before* the call lands in WASM.

Two helpers cover the bulk:

- `validateNDArrays({ positions, ndim, length })` — checks element-count
  consistency for visibility entry points.
- `validateProjectionInputs(...)` — same for the projection family.
- `validateDecodeArgs(fnName, data, length, lut?)` — covers the decode
  family. Verifies length divisibility, LUT presence when required, and
  that quantised inputs are typed as `Uint8Array` / `Uint16Array`.

A negative test for every entry point lives in
`src/tests/unit/workers/data-worker-validation.test.ts`. Validators
mock the WASM module to a no-op stub so the tests assert the
JS-boundary behaviour, not the WASM implementation.

---

## 4. 16-Dimension Limit

WASM functions use fixed-size arrays sized at `MAX_NDIM = 16` for cache
locality. Calling a 16-dim limited function with `ndim > 16` throws:

```
Error: ndim=18 exceeds maximum supported dimensions (16)
```

For datasets that genuinely exceed 16D (rare; usually a sign the producer
should reduce via PCA / feature selection), the TypeScript fallback is
selected automatically — it has no dimension limit but is significantly
slower for large arrays.

The 16-dim cap applies to: `calculate_effective_radii`,
`mahalanobis_distance`, `compute_gsplats_attenuation`, and any function
that does fixed-size hyper-volume math.

The TypeScript `extract_*` and decode helpers have no such cap.

---

## 5. TypeScript Fallback

The fallback (`src/wasm/typescript/`) mirrors the Rust module structure
file-for-file. Each function is correctness-equivalent to its Rust
counterpart — the parity is enforced by the `wasm-vs-typescript.test.ts`
suite, which runs the same fixture inputs through both implementations
and asserts byte-identical (or float-tolerant for FP outputs) results.

Use `getFallback()` to force this path in tests:

```typescript
import { getFallback } from './wasm';
const ts = getFallback();
```

The fallback's primary roles:

1. **Dev convenience**: WASM artefacts not built yet.
2. **Browser support**: targeting environments without `WebAssembly`.
3. **Parity testing**: validating Rust changes against the reference TS.
4. **>16-dim datasets**: graceful handling without throwing.

---

## 6. Memory + Transferable Discipline

WASM operates on JS `Uint8Array` / `Uint16Array` / `Float32Array` /
`Int32Array` buffers via wasm-bindgen. Discipline points:

- **Inputs cross by view**: the array's `.buffer` is shared, not copied.
  The Rust side reads from JS memory directly. The caller must not
  resize / detach the buffer mid-call.
- **Outputs are returned**: WASM allocates output buffers in its own
  heap and returns a copy to JS. The fallback mimics this — it
  always returns *new* arrays, never a view into the input.
- **Worker boundary**: data crosses to/from the DataWorker as a
  *transferable* (`postMessage(payload, [buffer])`). This requires the
  output buffer to be detachable, which is why the WASM side allocates
  a fresh buffer rather than overwriting an input.

Misuse of these contracts is hard to debug — the
`validate*` helpers in §3 catch the common shape errors at the
boundary, so a wrong-typed-array failure surfaces immediately rather
than as a worker crash.

---

## 7. Build Pipeline

```bash
make build-wasm    # Compile Rust → wasm via wasm-pack
make test-wasm     # cargo test in src/wasm/rust/
make benchmark-wasm# Run wasm-vs-typescript benchmarks
```

Output (committed to `public/wasm/` so the dev server can fetch them):

| File                  | Purpose                              |
| --------------------- | ------------------------------------ |
| `luxar_wasm_bg.wasm`  | Compiled WASM binary                 |
| `luxar_wasm.js`       | wasm-bindgen-generated JS bindings   |
| `luxar_wasm.d.ts`     | TypeScript type definitions          |

The Rust source lives in `src/wasm/rust/`; the parallel TS fallback in
`src/wasm/typescript/`. New API additions touch all three: Rust impl,
TS fallback, and the shared `WasmModule` interface in `types.ts`.

---

## 8. Versioning + Compatibility

There is no explicit version field on the WASM artefact. Compatibility is
maintained through:

1. **The `WasmModule` interface in `types.ts`**: any breaking change
   here ripples to both the Rust and TS implementations.
2. **The parity test suite**: `wasm-vs-typescript.test.ts` catches
   silent divergence.
3. **The validator suite**: `data-worker-validation.test.ts` catches
   accidental input-shape relaxations.

Feature flags / fallbacks for partial WASM support are not exposed
publicly — the loader's all-or-nothing model is intentional, since
mixing some WASM functions with TS counterparts in a single hot loop
would defeat the whole point.

---

## File Structure

```
wasm/
├── index.ts              — Loader (initWasm, isWasmSupported, getFallback)
├── types.ts              — WasmModule interface (unified API)
├── typescript/           — Pure TypeScript fallback
│   ├── index.ts          — TypeScriptFallback class
│   ├── spatial.ts        — Spatial queries
│   ├── points.ts         — Point visibility
│   ├── lines.ts          — Line visibility
│   ├── lines-clipping.ts — Line clipping
│   ├── gsplats.ts        — GSplat visibility
│   ├── gsplats-processing.ts
│   ├── decode.ts         — Array decoding
│   ├── effective-radii.ts
│   └── projection.ts     — nD → 3D projection
└── rust/                 — Rust source (parallel to TypeScript)
    ├── README.md         — Detailed Rust/WASM build docs
    ├── Cargo.toml
    └── src/              — Mirror of TypeScript modules in Rust
```
