# Luxar WASM Module

High-performance Rust/WebAssembly module for spatial queries, nD visibility computation, and data decoding.

**Status**: ✅ **PRODUCTION-READY** - Fully integrated with all data loaders

## Overview

The WASM module provides accelerated implementations of CPU-intensive operations used throughout the data loading pipeline. It runs in Web Workers for parallel execution, with automatic TypeScript fallback if WASM is unavailable.

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LOADERS                             │
│  (point/lines/gsplats-spatial-index-loader.ts)              │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │      WorkerPool       │                       │
│              │   (1-N DataWorkers)   │                       │
│              │           │           │                       │
│              │           ▼           │                       │
│              │  ┌─────────────────┐  │                       │
│              │  │   WASM Module   │  │                       │
│              │  │  (or TypeScript │  │                       │
│              │  │    fallback)    │  │                       │
│              │  └─────────────────┘  │                       │
│              └───────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/wasm/
├── rust/                      # Rust source (this directory)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs             # Module exports
│       ├── spatial.rs         # Chunk AABB queries
│       ├── points.rs          # Point visibility
│       ├── lines.rs           # Line segment visibility
│       ├── gsplats.rs         # GSplat visibility
│       ├── decode.rs          # Quantized/LUT decoding
│       ├── projection.rs      # nD → 3D projection
│       ├── effective_radii.rs # Effective radius calculation
│       ├── gsplats_processing.rs # GSplat-specific processing
│       └── lines_clipping.rs  # Line segment clipping
├── index.ts                   # WASM loader with fallback detection
├── types.ts                   # WasmModule interface (525 lines)
└── typescript/                # TypeScript fallback implementations
    ├── index.ts
    ├── spatial.ts
    ├── points.ts
    ├── lines.ts
    ├── gsplats.ts
    ├── decode.ts
    ├── projection.ts
    ├── effective_radii.ts
    ├── gsplats_processing.ts
    └── lines_clipping.ts
```

## Build Output

After building, `public/wasm/` contains:

- `luxar_wasm_bg.wasm` - WebAssembly binary (~44KB optimized)
- `luxar_wasm.js` - JavaScript bindings
- `luxar_wasm.d.ts` - TypeScript type definitions

## Prerequisites

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Or visit: https://rustup.rs/

### 2. Install wasm-pack

```bash
cargo install wasm-pack
```

Or visit: https://rustwasm.github.io/wasm-pack/installer/

### Quick Setup (Makefile)

```bash
# From project root - installs Rust + wasm-pack automatically
make setup-rust
```

## Building

### Production Build

```bash
# From luxar-viewer root:
pnpm build:wasm

# Or via Makefile (from project root):
make wasm-build

# Or directly:
cd src/wasm/rust
wasm-pack build --target web --out-dir ../../../public/wasm --release
```

### Development Build (faster, larger)

```bash
pnpm build:wasm:dev
```

## Testing

### Run Rust Unit Tests (39 tests)

```bash
# From project root:
make test-wasm

# Or from luxar-viewer:
pnpm test:wasm

# Or directly:
cd src/wasm/rust
cargo test
```

### Test Coverage

All modules have comprehensive unit tests:

**Spatial queries** (`spatial.rs`):

- `test_query_chunks_basic_3d` - 3D chunk intersection
- `test_query_chunks_4d_with_hidden_dimension` - 4D with hidden dim
- `test_query_chunks_no_matches` - Empty result handling

**Point visibility** (`points.rs`):

- `test_point_visibility_3d` - Basic 3D visibility
- `test_point_visibility_4d_hidden_dimension` - 4D with slicing
- `test_point_visibility_edge_cases` - Edge cases

**Line visibility** (`lines.rs`):

- `test_line_visibility_both_endpoints_visible` - Both endpoints in slice
- `test_line_visibility_one_endpoint_visible` - Partial visibility
- `test_line_visibility_both_endpoints_hidden` - Full occlusion

**GSplat visibility** (`gsplats.rs`):

- `test_gsplat_visibility_basic` - 3D ellipsoid visibility
- `test_gsplat_visibility_4d` - 4D with slicing

**Decoding** (`decode.rs`):

- `test_decode_quantized_*` - Quantized data decoding
- `test_decode_lut_*` - LUT data decoding
- `test_decode_log_scalar_*` - Log-space decoding

**Projection** (`projection.rs`):

- `test_project_points_to_3d_*` - nD to 3D projection

### Run TypeScript Integration Tests

```bash
# Includes WASM comparison tests (59 tests)
pnpm test src/tests/unit/wasm/
```

## Functions Exported

### Spatial Queries

| Function                | Description                       |
| ----------------------- | --------------------------------- |
| `query_chunks_for_view` | Find chunks intersecting nD slice |

### Visibility Computation

| Function                        | Description                         |
| ------------------------------- | ----------------------------------- |
| `compute_nd_visibility_points`  | Filter visible points (hypersphere) |
| `compute_nd_visibility_lines`   | Filter visible line segments        |
| `compute_nd_visibility_gsplats` | Filter visible GSplats (ellipsoid)  |

### Data Decoding

| Function             | Description               |
| -------------------- | ------------------------- |
| `decode_quantized`   | Uint8/Uint16 → Float32    |
| `decode_lut`         | Index → value lookup      |
| `decode_log_scalar`  | Dequantize with log scale |
| `decode_broadcasted` | Replicate single value    |

### Projection

| Function                | Description                       |
| ----------------------- | --------------------------------- |
| `project_points_to_3d`  | nD → 3D with visibility filtering |
| `project_lines_to_3d`   | Line nD → 3D projection           |
| `project_gsplats_to_3d` | GSplat nD → 3D projection         |

## Usage from TypeScript

```typescript
import { initWasm } from '../wasm';

// Load WASM (falls back to TypeScript if unavailable)
const wasm = await initWasm();

// Spatial query example
const count = wasm.query_chunks_for_view(
  chunkBounds,
  slicePosition,
  tolerance,
  ndim,
  numChunks,
  output
);

// Decoding example
const decoded = wasm.decode_quantized(quantizedData, minVal, maxVal, output);
```

## Performance

**Expected Speedups vs TypeScript fallback:**

- Spatial index queries: **3-5x faster**
- nD visibility computation: **3-5x faster**
- Quantized decoding: **2-3x faster**

**Optimizations:**

- Fixed-size arrays (avoids heap allocations)
- Bulk memory operations
- Zero-copy where possible (wasm-bindgen)

## Dimension Limits

**Maximum supported dimensions: 16**

Functions that process nD data (visibility, effective radii, GSplats processing) use
fixed-size arrays for performance. If your dataset has more than 16 dimensions:

1. WASM functions will panic with a clear error message
2. The TypeScript fallback has no dimension limit
3. Consider reducing dimensions for performance-critical datasets

```typescript
// Example error when ndim > 16:
// [WASM] calculate_effective_radii: ndim=20 exceeds maximum supported dimensions (16).
// Luxar WASM functions support up to 16 dimensions.
// For higher dimensions, use TypeScript fallback or reduce dataset dimensionality.
```

**Functions with this limit:**

- `calculate_effective_radii`
- `mahalanobis_distance`
- `compute_gsplats_attenuation`
- `extract_visible_cholesky_3d`

## Integration Points

The WASM module is used through the unified loader architecture:

**Direct loader calls**:

- `point-spatial-index-loader.ts`: Spatial queries (~line 727), projection (~line 1500)
- `lines-spatial-index-loader.ts`: Spatial queries (~line 534)
- `gsplats-spatial-index-loader.ts`: Spatial queries (~line 377)

**Via unified loaders** (`src/data/loaders/`):

- `range-loader.ts`: All encoding dispatch (broadcast, quantized, LUT decode)
- `spatial-query-builder.ts`: Spatial query execution (~line 162)

Note: Line numbers are approximate. Encoding-specific methods have been consolidated
into `RangeLoader` as part of the unified loader architecture (Phase 1 complete).

## TypeScript Fallback

The TypeScript fallback in `src/wasm/typescript/` provides full parity with Rust implementations. It's automatically used when:

- WASM module fails to load
- Running in environments without WebAssembly support
- Configuration flag `useWASM: false`

## Troubleshooting

### Build Fails

- Ensure Rust toolchain is up to date: `rustup update`
- Ensure wasm-pack is installed: `cargo install wasm-pack`
- Clear target directory: `cd src/wasm/rust && cargo clean`

### WASM Not Loading in Browser

- Check browser console for errors
- Verify files exist in `public/wasm/`
- Ensure Vite is serving public directory correctly
- Check WASM file size (should be ~44KB optimized)

### Tests Fail

- Run Rust tests first: `cargo test`
- Check cargo test output for specific failures
- Verify test data matches expected format

## Development Workflow

1. **Make changes** to Rust files in `src/`
2. **Run Rust tests**: `cargo test` (39 tests)
3. **Build WASM**: `pnpm build:wasm`
4. **Run TypeScript tests**: `pnpm test` (includes WASM comparison)
5. **Test in browser**: `pnpm dev` and check console

## Verification Commands

```bash
# From project root
make test-wasm       # Run 39 Rust tests
make wasm-build      # Build WASM module

# From luxar-viewer
pnpm test:wasm       # Rust tests
pnpm build:wasm      # Build
pnpm test --run      # All TypeScript tests (includes WASM comparison)
```
