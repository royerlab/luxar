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
  kernels: `clip_segments_batch` for Lines, the attenuation/fused kernel for
  GSplats)

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
- `calculate_bounds_3d()` — Compute 3D bounding box from nD data
- `compact_by_mask()` — Remove invisible elements by boolean mask
- `count_visible()` — Count non-zero mask values
- `radii_to_visibility_mask()` — Convert effective radii to visibility mask

### Gaussian Splats

- `compute_gsplats_attenuation()` — Gaussian attenuation from nD distance
- `extract_visible_cholesky_3d()` — Extract 3D Cholesky factors from nD
- `compact_attenuated_amplitudes()` — Remove fully attenuated splats
- `extract_cholesky_submatrix()` — Extract raw Cholesky submatrix for specified dimensions
- `mahalanobis_distance()` — nD Mahalanobis distance

### Line Clipping

- `clip_segment_single()` — Clip a single segment and return interpolation parameters
- `clip_segments_batch()` — Clip line segments to nD slice bounds
- `interpolate_clipped_positions()` — Interpolate positions at clip boundaries
- `interpolate_scalars_batch()` / `interpolate_colors_batch()` — Interpolate attributes
- `calculate_segment_lengths()` — Compute segment lengths for LOD
- `mark_clipped_endpoints()` — Mark clipped endpoints for cap factor adjustment
- `lerp()` / `lerp_vec3()` / `distance_3d()` — Math helpers

## 16-Dimension Limit

WASM functions use fixed-size arrays for performance and support a **maximum of 16 dimensions**. For data with more than 16 dimensions, the TypeScript fallback is used automatically (slower but has no dimension limit).

## Performance: WASM vs TypeScript Fallback

The TypeScript fallback is correctness-equivalent and useful for development,
unsupported browsers, or missing build artifacts, but it is not the performance
target for large interactive scenes. WASM is recommended for:

- nD point visibility and effective-radius queries over large chunks
- line clipping/projection in 4D+ scenes
- GSplat attenuation and Cholesky submatrix extraction
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
│   ├── points.ts         — Point visibility
│   ├── lines.ts          — Line visibility
│   ├── lines-clipping.ts — Line clipping
│   ├── gsplats.ts        — GSplat visibility
│   ├── gsplats-processing.ts — GSplat processing
│   ├── decode.ts         — Array decoding
│   ├── effective-radii.ts — Radius calculations
│   └── projection.ts     — nD → 3D projection
└── rust/                 — Rust source (parallel to TypeScript)
    ├── Cargo.toml
    └── src/              — Mirror of TypeScript modules in Rust
        └── README.md     — Detailed Rust/WASM build docs
```

## Subpackages

- [`rust/`](./rust/src/README.md) — Rust source compiled to WASM. Implements
  the kernels for Points, Lines, GSplats, projection, and line clipping.
  See `rust/src/README.md` for the full Rust/WASM build pipeline.
- [`typescript/`](./typescript/README.md) — Pure TypeScript fallback
  matching the Rust kernels function-for-function. Used when WASM fails
  to load or for environments without WebAssembly support.
