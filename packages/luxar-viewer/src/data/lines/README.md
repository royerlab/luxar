# Lines data path

Lines-specific data loaders and projection. The wider data pipeline
lives in `src/data/`; this folder contains the bits that are specific
to the Lines node type.

## Files

| File | Role |
|------|------|
| `lines-spatial-index-loader.ts` | Spatial-index loader for Lines: dual chunk index (segment ordering + vertex ordering), two-stage load (segment chunks → unique vertex set → vertex-attr ranges). Emits a `LoadedLinesData` payload with optional per-vertex scalars for colormap mode. |
| `projection.ts` | Main-thread `buildInstanceBuffers`: clips segments to the nD slice, interpolates per-vertex attributes (colors, widths, sharpness, scalars) at clipped endpoints, and emits per-segment GPU instance buffers. Used as the fallback when the worker is unavailable. The primary path is the WASM batch projection in `workers/data-worker.ts::projectLinesTo3D`. |
| `chunk-index-loader.ts` | Loads the Lines dual chunk-bounds index (segment + vertex orderings); exposes `registerLinesArrayBounds` as a per-type wrapper around `ChunkPrefetcher.registerArrayBounds`. |

## Public surface

`LinesSpatialIndexLoader` implements the same `SpatialIndexLoader`
contract as Points / GSplats (constructor, `loadForView`,
`prefetchChunks`, `dispose`, monitor events). Scene-loader code never
imports the concrete class — it goes through `loader-factory.ts`.

`buildInstanceBuffers` is exported for the main-thread fallback path
and for unit tests that exercise per-vertex interpolation without a
WASM context.

## Invariants

- **Dual chunk index.** Lines maintain two independent chunk
  orderings: one in `(2 × ndim)` segment-endpoint space (for spatial
  queries) and one in `ndim` vertex space (for direct vertex-attr
  loading). The query algorithm walks the segment index to find
  visible segments, then collects the unique vertex set into
  contiguous vertex ranges so each attribute array (positions,
  widths, colors, sharpness, scalars) is fetched with one zarr read
  per range.
- **Per-vertex scalars ride the worker path.** Since `873690c3` the
  worker payload carries `scalars: Float32Array | Float16Array |
  Uint8Array | null`; `interpolate_scalars_batch` (the same WASM
  kernel that handles widths and sharpness) produces compacted
  `startScalars` / `endScalars`. Uint8 inputs are normalized by
  `1/255` to match the colormap shader's `[0, 1]` contract.
- **Main-thread `buildInstanceBuffers` is the fallback.** It is
  slower (~600 ms / 1M segments per
  `src/tests/benchmarks/lines-ts-fallback-alloc-bench.ts`) but
  preserves numerical parity with the worker path. The fallback
  fires only on WASM init failure / worker timeout / dataset-switch
  abort (the abort path now re-throws rather than falling back —
  see `WorkerAbortError`).
- **Scalar length validation is fail-closed.** If the scalar array
  length doesn't match the vertex count,
  `buildInstanceBuffers` logs a warning and emits the output with
  no scalars; the line shader's `USE_COLORMAP` path stays inactive
  rather than reading garbage scalar values.

## See also

- `src/types/lines.ts` — type definitions and metadata schema
- `src/rendering/line-material.ts` — GPU-side rendering
- `src/rendering/line-geometry.ts` — instanced mesh construction +
  update helpers
- `src/wasm/typescript/lines.ts` and `lines-clipping.ts` —
  TypeScript fallback that mirrors the Rust WASM kernel
- `src/data/loaders/SPECIFICATIONS.md` — encoding dispatch /
  range-loader contract
- `src/data/SPECIFICATIONS.md` §7 — full Lines spatial-index spec
  including the worker scalar path and the dual chunk index
- `src/tests/benchmarks/lines-ts-fallback-alloc-bench.ts` —
  performance benchmark for the main-thread fallback
