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
import { initWasm, isWasmSupported } from './wasm';

// Load WASM with automatic TypeScript fallback
const wasm = await initWasm();

// Use the unified API
const chunks = wasm.query_chunks_for_view(/* ... */);
const visibility = wasm.compute_nd_visibility_points(/* ... */);
```

## WasmModule API

### Spatial Queries
- `query_chunks_for_view()` — Find chunks intersecting an nD slice

### nD Visibility
- `compute_nd_visibility_points()` — Hypersphere intersection for point clouds
- `compute_nd_visibility_lines()` — Endpoint-based visibility for line segments
- `compute_nd_visibility_gsplats()` — Ellipsoid extent for Gaussian splats
- `calculate_effective_radii()` — Radius when sliced through higher dimensions

### Decoding
- `decode_quantized_u8()` / `decode_quantized_u16()` — Dequantize compressed arrays
- `decode_log_scalar_u8()` / `decode_log_scalar_u16()` — Log-space decoding
- `decode_lut_scalar()` / `decode_lut_row()` — Lookup table decoding
- `decode_broadcasted()` — Expand broadcast-encoded arrays

### Projection
- `extract_3d_positions()` — Extract display dimensions from nD positions
- `calculate_bounds_3d()` — Compute 3D bounding box from nD data
- `compact_by_mask()` — Remove invisible elements by boolean mask
- `radii_to_visibility_mask()` — Convert effective radii to visibility mask

### Gaussian Splats
- `compute_gsplats_attenuation()` — Gaussian attenuation from nD distance
- `extract_visible_cholesky_3d()` — Extract 3D Cholesky factors from nD
- `compact_attenuated_amplitudes()` — Remove fully attenuated splats
- `mahalanobis_distance()` — nD Mahalanobis distance

### Line Clipping
- `clip_segments_batch()` — Clip line segments to nD slice bounds
- `interpolate_clipped_positions()` — Interpolate positions at clip boundaries
- `interpolate_scalars_batch()` / `interpolate_colors_batch()` — Interpolate attributes
- `calculate_segment_lengths()` — Compute segment lengths for LOD

## 16-Dimension Limit

WASM functions use fixed-size arrays for performance and support a **maximum of 16 dimensions**. For data with more than 16 dimensions, the TypeScript fallback is used automatically (slower but has no dimension limit).

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
├── index.ts              — Loader (initWasm, isWasmSupported, getFallback)
├── types.ts              — WasmModule interface (unified API)
├── typescript/           — Pure TypeScript fallback
│   ├── index.ts          — TypeScriptFallback class
│   ├── spatial.ts        — Spatial queries
│   ├── points.ts         — Point visibility
│   ├── lines.ts          — Line visibility
│   ├── lines_clipping.ts — Line clipping
│   ├── gsplats.ts        — GSplat visibility
│   ├── gsplats_processing.ts — GSplat processing
│   ├── decode.ts         — Array decoding
│   ├── effective_radii.ts — Radius calculations
│   └── projection.ts     — nD → 3D projection
└── rust/                 — Rust source (parallel to TypeScript)
    ├── README.md         — Detailed Rust/WASM build docs
    ├── Cargo.toml
    └── src/              — Mirror of TypeScript modules in Rust
```

For Rust implementation details, see [`rust/README.md`](rust/README.md).
