# luxar-viewer/src/wasm

High-performance WebAssembly module for spatial queries, nD visibility computation, and array decoding. Includes a pure TypeScript fallback for environments where WASM is unavailable.

## Architecture

```
Data Loaders
  └─► WorkerPool
        └─► DataWorker
              └─► WasmModule interface
                    ├── Rust/WASM implementation (3-5x faster)
                    └── TypeScript fallback (works everywhere)
```

Both implementations share the same `WasmModule` interface, so callers don't need to know which is active.

## Usage

```typescript
import { initWasm, isWasmSupported, setWasmJsUrl } from './wasm';

// Optional: override the WASM JS shim URL for non-default bundler setups
// (must be called before initWasm). LuxarApp forwards LuxarAppOptions.wasmPath
// through this hook automatically.
// setWasmJsUrl(new URL('/static/luxar/wasm/luxar_wasm.js', location.origin).href);

// Load WASM with automatic TypeScript fallback
const wasm = await initWasm();

// Use the unified API
const count = wasm.clip_segments_batch(/* ... */);
```

The default URL resolution (`new URL('../wasm/luxar_wasm.js', import.meta.url)`)
works for the standalone Vite app and most consumer bundlers (Vite, Rollup,
webpack 5). Use `setWasmJsUrl` only when shipping WASM files from a
non-standard location.

## WasmModule API

### nD Visibility

- `calculate_effective_radii()` — Radius when sliced through higher dimensions
  (per-element nD visibility/culling otherwise lives INSIDE the projection
  kernels: `clip_segments_batch` for Lines, the fused projection kernel for
  GSplats). Mesh is the exception: its cull is a standalone pair, below, because
  it produces an index buffer rather than compacted per-element attributes.

### Decoding

- `decode_quantized_u8()` / `decode_quantized_u16()` — Dequantize compressed arrays
- `decode_log_scalar_u8()` / `decode_log_scalar_u16()` — Log-space decoding
- `decode_geolog_scalar_u8()` / `decode_geolog_scalar_u16()` — Geometric-log decoding (reserved zero level)
- `decode_linear_perchannel_u8()` / `_u16()` — Per-column fixed-point (COORDINATE centers/positions)
- `decode_log_perchannel_u8()` / `_u16()` — Per-column log (Cholesky diagonal; `zero_level`-aware)
- `decode_signed_log_perchannel_u8()` / `_u16()` — Per-column signed-log (Cholesky off-diagonal)
- `decode_geolog_perchannel_u8()` / `_u16()` — Per-column TRUE-log (HDR colors; reserved zero level)
- `decode_lut_scalar_u8()` / `decode_lut_scalar_u16()` — Lookup table scalar decoding
- `decode_lut_row_u8()` / `decode_lut_row_u16()` — Lookup table row (vector) decoding
- `decode_broadcasted()` — Expand broadcast-encoded arrays

### Projection

- `extract_3d_positions()` — Extract display dimensions from nD positions

### Gaussian Splats

- `project_gsplats_nd_to_3d()` — Fused single-pass nD→3D projection (attenuation, visibility, compaction)
- `mahalanobis_distance()` — nD Mahalanobis distance

### Line Clipping

- `clip_segment_single()` — Clip a single segment and return interpolation parameters
- `clip_segments_batch()` — Clip line segments to nD slice bounds
- `interpolate_clipped_positions()` — Interpolate positions at clip boundaries
- `interpolate_scalars_batch()` / `interpolate_colors_batch()` — Interpolate attributes
- `calculate_segment_lengths()` — Compute segment lengths for LOD
- `compute_cap_suppression()` — Per-endpoint cap suppression in [0, 1] (clipped endpoints + interior polyline joints) for the shader cap factor

### Mesh Culling

Whole-triangle nD culling for indexed surfaces. Unlike Lines, nothing is clipped
or interpolated — a triangle is drawn iff **all three** of its vertices pass the
nD slab test, so a cut boundary is triangle-quantized (a documented v1 trade;
see `docs/specs/MESH_NODE_SPEC.md` §5).

- `mesh_vertex_visibility_mask()` — Per-vertex nD slab membership → `u8` mask
- `compact_visible_faces()` — Keep faces whose three vertices are all visible,
  writing ORIGINAL (un-remapped) vertex indices

Only the index buffer is rebuilt on a slice change; vertex attribute buffers are
uploaded once and left alone, so vertices are never compacted.

## 16-Dimension Limit

WASM functions use fixed-size arrays for performance and support a **maximum of 16 dimensions**. For data with more than 16 dimensions, the TypeScript fallback is used automatically (slower but has no dimension limit).

## Performance: WASM vs TypeScript Fallback

The TypeScript fallback is correctness-equivalent and useful for development,
unsupported browsers, or missing build artifacts, but it is not the performance
target for large interactive scenes. WASM is recommended for:

- nD point visibility and effective-radius queries over large chunks
- line clipping/projection in 4D+ scenes
- GSplat nD→3D projection (attenuation, visibility, marginal Cholesky, compaction)
- quantization/LUT/log decoding for large arrays

Run the benchmark suite on the target machine to measure real speedups:

```bash
make benchmark-wasm
# or, from packages/luxar-viewer/
pnpm test src/tests/unit/wasm/wasm-performance.test.ts --run
```

Benchmark results depend on browser/runtime, CPU, memory bandwidth, array size,
and dimensionality. Keep benchmark output with performance investigations rather
than treating static numbers in this README as release guarantees.

## Build

The WASM module must be compiled from Rust source before use:

```bash
make build-wasm    # Build WASM module
make test-wasm     # Run Rust unit tests
```

Build output goes to `public/wasm/`:

- `luxar_wasm_bg.wasm` — Compiled WASM binary
- `luxar_wasm.js` — JavaScript bindings
- `luxar_wasm.d.ts` — TypeScript type definitions

If WASM is not built, `initWasm()` logs a warning with build instructions and falls back to TypeScript.

## File Structure

```
wasm/
├── index.ts              — Loader (initWasm, isWasmSupported, getFallback,
│                           setWasmJsUrl, isWasmFallback)
├── types.ts              — WasmModule interface (unified API)
├── typescript/           — Pure TypeScript fallback
│   ├── index.ts          — TypeScriptFallback class
│   ├── decode.ts         — Array decoding
│   ├── effective-radii.ts — Radius calculations
│   ├── gsplats-processing.ts — GSplat processing
│   ├── lines-clipping.ts — Line clipping
│   ├── mesh-culling.ts   — Whole-triangle nD culling
│   └── projection.ts     — nD → 3D projection
└── rust/                 — Rust source (parallel to TypeScript)
    ├── Cargo.toml
    ├── README.md         — Crate overview + build/test commands
    └── src/              — Mirror of TypeScript modules in Rust
        └── README.md     — Detailed per-module kernel docs
```

## Subpackages

- [`rust/`](./rust/src/README.md) — Rust source compiled to WASM. Implements
  the kernels for Points, Lines, GSplats, Mesh, projection, and line clipping.
  See `rust/src/README.md` for the full Rust/WASM build pipeline.
- [`typescript/`](./typescript/README.md) — Pure TypeScript fallback
  matching the Rust kernels function-for-function. Used when WASM fails
  to load or for environments without WebAssembly support.
