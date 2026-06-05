# Lines data path

Lines-specific data loaders and projection. The wider data pipeline
lives in `src/data/`; this folder contains the bits that are specific
to the Lines node type.

## Files

| File                            | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lines-spatial-index-loader.ts` | Spatial-index loader for Lines: dual chunk index (segment ordering + vertex ordering), two-stage load (segment chunks → unique vertex set → vertex-attr ranges). Emits a `LoadedLinesData` payload with optional per-vertex scalars for colormap mode.                                                                                                                                                                                                                                               |
| `projection.ts`                 | Main-thread `projectLinesTo3D` (TS fallback) and `projectLinesTo3DWASM` (batch WASM): clip segments to the nD slice, interpolate per-vertex attributes (colors, widths, sharpness, scalars) at clipped endpoints, and emit per-segment GPU instance buffers. The TS path is the fallback; the primary path is the WASM batch projection registered at `workers/data-worker/projection/lines.ts` (dispatched from `workers/data-worker.ts`). Also exports `createEmptyLinesData` and `initLinesWASM`. |
| `chunk-index-loader.ts`         | Loads the Lines dual chunk-bounds index (segment + vertex orderings); exposes `registerLinesArrayBounds` as a per-type wrapper around `ChunkPrefetcher.registerArrayBounds`, and `computeVertexRangesFromIndices` for coalescing referenced vertex indices into contiguous zarr ranges.                                                                                                                                                                                                              |
| `handler.ts`                    | Per-type scene-loader wiring (`loadAndStage`) — mirrors `points/handler.ts` and `gsplats/handler.ts`. Derives the per-node `LinesViewState`, calls `loader.updateView`, hands the payload to `processLinesData`, and dispatches predictive prefetch via the shared `ViewStateQueue`. Lines use `applyPartialExtendTolerance: false` (verbatim legacy behaviour).                                                                                                                                     |
| `lines-progressive-loader.ts`   | `LinesProgressiveLoader` — Composite-pattern multi-additive-LOD facade wrapping N `LinesSpatialIndexLoader` instances (one per `additive_<i>` subgroup). Loads LODs sequentially per `updateView`, stopping after a cache miss / wall-clock budget (`CACHE_HIT_THRESHOLD_MS`); `concatenateLinesData` offset-adjusts each subgroup's local `segments` indices by the cumulative vertex count and fills missing-LOD colors with white. Exposes `hasMoreLODs` / `loadedLODCount` / `totalLODCount` / `lastAllResident`. Mirrors `GSplatsProgressiveLoader` / `PointsProgressiveLoader`. |
| `lod-refinement.ts`             | `runLinesRefinement` — thin wrapper over the generic `scene-loader/progressive/refinement.ts` helper. Re-derives each progressive node's `LinesViewState`, calls `loader.updateView` to stream the next LOD, then `processLines` → `commitLines`. Mirrors `gsplats/lod-refinement.ts` / `points/lod-refinement.ts`. Driven by `scene-loader.ts`.                                                                                                  |

## Public surface

`LinesSpatialIndexLoader` implements `LinesDataLoader` — same shape
as the Points and GSplats facades (constructor, `loadLines` /
`updateView`, `prefetchChunks`, `dispose`, monitor events via
`addEventListener` / `getMetrics` / `getActiveQueries`). Scene-loader
code never imports the concrete class — it goes through
`scene-loader/loaders/loader-factory.ts`.

`LinesProgressiveLoader` (in `lines-progressive-loader.ts`) wraps N
single-LOD `LinesSpatialIndexLoader`s for multi-additive-LOD datasets
and satisfies the same `LinesDataLoader` interface, so scene-loader code
is agnostic to whether a node is single- or multi-LOD. The factory
returns it for `additive_<i>` subgroups; `scene-loader.ts` drives
`runLinesRefinement` (from `lod-refinement.ts`) to stream the remaining
LODs in the background once the first level is on screen.

`projectLinesTo3D` is exported for the main-thread fallback path
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
- **Progressive concatenation remaps segment indices.** When
  `LinesProgressiveLoader` concatenates per-LOD `LoadedLinesData`, each
  subgroup's `segments` array indexes its *own local* vertex buffer, so
  `concatenateLinesData` offsets every segment index by the cumulative
  vertex count of earlier levels. Per-vertex fields go through the shared
  `concatRequiredField` / `concatOptionalField` helpers (dtype
  preserved); colors fill missing-LOD ranges with white (or the
  dtype-max for Uint8/Uint16). LOD 0 always loads; later levels stop on a
  cache miss or once the wall-clock budget (`CACHE_HIT_THRESHOLD_MS`) is
  exceeded, leaving the rest to the refinement loop.
- **Per-vertex scalars ride the worker path.** Since `873690c3` the
  worker payload carries `scalars: Float32Array | Float16Array |
Uint8Array | null`; `interpolate_scalars_batch` (the same WASM
  kernel that handles widths and sharpness) produces compacted
  `startScalars` / `endScalars`. Uint8 inputs are normalized by
  `1/255` to match the colormap shader's `[0, 1]` contract.
- **Main-thread `projectLinesTo3D` is the fallback.** It is
  slower (~600 ms / 1M segments per
  `src/tests/benchmarks/lines-ts-fallback-alloc-bench.ts`) but
  preserves numerical parity with the worker path. The fallback
  fires only on WASM init failure / worker timeout / dataset-switch
  abort (the abort path now re-throws rather than falling back —
  see `WorkerAbortError`).
- **Scalar length validation is fail-closed.** If the scalar array
  length doesn't match the vertex count,
  `projectLinesTo3D` logs a warning and emits the output with
  no scalars; the line shader's `USE_COLORMAP` path stays inactive
  rather than reading garbage scalar values.

## See also

- `src/types/lines.ts` — type definitions and metadata schema
- `src/rendering/materials/line/material-glsl.ts` and `material-tsl.ts` — GPU-side rendering (WebGL2 + WebGPU dual stack)
- `src/rendering/line-geometry.ts` — instanced mesh construction +
  update helpers
- `src/wasm/typescript/lines.ts` and `lines-clipping.ts` —
  TypeScript fallback that mirrors the Rust WASM kernel
- `src/data/scene-loader/progressive/refinement.ts` — generic
  multi-LOD refinement loop that `runLinesRefinement` delegates to
- `src/data/loaders/progressive/concat-helpers.ts` — shared
  `concatRequiredField` / `concatOptionalField` used by the
  progressive loader's `concatenateLinesData`
- `src/data/loaders/README.md` — encoding dispatch /
  range-loader contract
- `src/data/README.md` — Lines spatial-index overview
- `src/tests/benchmarks/lines-ts-fallback-alloc-bench.ts` —
  performance benchmark for the main-thread fallback
