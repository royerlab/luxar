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

`initWasm` resolves the shim URL first-match-wins: a `setWasmJsUrl` override;
else, in a **dev build** with a `location` global, `/wasm/luxar_wasm.js` on the
dev-server origin; else an ordered list of candidates relative to the bundled
chunk (`import.meta.url`), which works for the standalone Vite app and most
consumer bundlers (Vite, Rollup, webpack 5). The dev-origin branch reads the bare
`location` global only under `typeof`, so a host without one — Node, SSR, the
`node` test environment — falls through to the bundle-relative candidates rather
than throwing (#1642). That branch is a SINGLE candidate: the dev server serves
`public/wasm/` from the root while this module is served from
`/src/wasm/index.ts`, so the bundle-relative ones could only add guaranteed 404s.

Why a list and not one literal (#1649): the compiled artifact always lands in a
`wasm/` directory at the output root, but the chunk carrying the loader sits at
one of two depths — `assets/index-*.js` and the library build's worker chunks
are one level down, while the library build's entry chunk (`luxar-viewer.js`)
is at the root itself. So `initWasm` tries the shim one directory ABOVE the chunk
first (the app build and both worker chunks, so the hot paths still cost one
request), then the shim BESIDE the chunk (the library entry chunk). In built terms
those are `dist/wasm/luxar_wasm.js` and `dist/lib/wasm/luxar_wasm.js`; the
specifiers themselves are in `WASM_SHIM_RELATIVE_SPECIFIERS`. The list is
deduplicated after resolution, so it is not always two requests: a chunk served at
the URL root (`dist/lib/*` copied to a site root) resolves both specifiers to the
same href. A candidate counts as a hit only if it exposes a **callable**
`default`, so a 200 with an empty body — or a JavaScript stub/redirect module
served in place of the absent artifact — falls through instead of ending the walk
on a module that cannot initialize. (An HTML error page needs no such help: it
does not parse as an ES module, so the import itself rejects.) Only the import is
retried; once a candidate wins, initialization and the staleness check run against
it alone. Use `setWasmJsUrl` only when shipping WASM files from a non-standard
location.

Dev-tree gotcha, and only under one precondition: if you serve the whole `dist/`
after running both `pnpm build` and `pnpm build:lib`, the app build's
`dist/wasm/` is the FIRST candidate for `dist/lib/luxar-viewer.js` and shadows
the freshly copied `dist/lib/wasm/`. Back to back that is benign — both scripts
begin with `pnpm build:wasm` and copy the same `public/wasm/`, so the two
artifacts are byte-identical. It only bites when `public/wasm/` changed between
the two builds, and then the likely outcome is the silent one:
`assertRequiredWasmExports` compares export NAMES only, so an older binary with
the same export set loads and runs, and you profile or debug the wrong build
with nothing on the console. It turns loud — a "Loaded WASM module is stale"
throw plus the TypeScript fallback, from an artifact that looks correctly placed
— only when the shadowing build predates a kernel since added to
`required-exports.ts`.

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
- `compute_joint_codes()` — Per-endpoint joint code: a sentinel (free end / slice-clipped / degree-≥​3 hub) or a signed reference to the partner segment's storage slot, driving the shader's join geometry and endpoint cap

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
pnpm bench:wasm
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

If WASM is not built, `initWasm()` logs a warning with build instructions and falls back to
TypeScript. The warning names the candidate URLs it RESOLVED, in order, or says resolution failed
before the import — "the artifact isn't there" and "the loader never computed a URL" read
identically otherwise. It deliberately does not claim each listed candidate was fetched and
rejected: the same `catch` also covers a failure AFTER one of them loaded fine — `default()`
throwing, or `assertRequiredWasmExports` rejecting a stale artifact — and the swallowed `error`
(logged alongside) is what says which happened.

### Loading the built artifact directly

Tests and benchmarks that need the compiled kernels rather than the fallback can't use
`initWasm()`: it reaches the shim through a `new Function('url', 'return import(url)')`
indirection that vitest's VM module runner does not service
(`ERR_VM_DYNAMIC_IMPORT_CALLBACK_MISSING`), so under vitest every `initWasm()` ends in
`TypeScriptFallback` in **any** environment, whatever URL it resolved. They all go through
one shared loader, `src/tests/helpers/wasm-artifact.ts`:

```typescript
import {
  loadWasmArtifact,
  tryLoadWasmArtifact,
  wasmArtifactExists,
} from '../helpers/wasm-artifact';

const wasmAvailable = wasmArtifactExists(); // gate describe.skipIf / it.runIf

const wasm = await loadWasmArtifact(); // strict: absent, incompatible or stale all throw
const maybe = await tryLoadWasmArtifact(); // soft: null on a load failure, still throws when stale
```

It reads the `.wasm` bytes, `initSync`s the shim, and calls `assertRequiredWasmExports` on the
namespace **and** on the instantiated exports `initSync` returns (a mixed build — only one of
`luxar_wasm.js` / `luxar_wasm_bg.wasm` overwritten — still instantiates, and the shim's statically
declared wrappers hide the gap), **before** the `as unknown as WasmModule` cast and **outside** the
catch that downgrades a
load failure to a skip. The cast promises the whole `WasmModule` interface while a stale gitignored
build may be missing newer kernels, so without the check a stale build fails as an opaque
`x is not a function` deep in an unrelated kernel test instead of naming the missing export and
pointing at `pnpm build:wasm (or make build-wasm)`; inside that catch the named message would be
swallowed instead. `src/tests/unit/wasm/direct-import-guard.test.ts` keeps the rule structural
rather than policed: it fails if any `.ts` under `src/`, `tools/` or `scripts/` other than that
helper (and `src/wasm/index.ts`, the production loader) both mentions `luxar_wasm.js` and calls
`initSync(`.

## File Structure

```
wasm/
├── index.ts              — Loader (initWasm, isWasmSupported, getFallback,
│                           setWasmJsUrl, isWasmFallback,
│                           resolveWasmShimUrls, importFirstWasmShim,
│                           instantiateWasmShim, assertRequiredWasmExports)
├── shared-module.ts      — Compiles one cloneable WebAssembly.Module for all
│                           data workers, with bounded per-worker fallback
├── types.ts              — WasmModule interface (unified API)
├── required-exports.ts   — Kernels a stale build may lack; shared by the
│                           loader's staleness check and the vitest global setup
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
