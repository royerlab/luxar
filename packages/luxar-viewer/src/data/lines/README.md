# Lines data path

Lines-specific data loaders and projection. The wider data pipeline
lives in `src/data/`; this folder contains the bits that are specific
to the Lines node type.

## Files

| File                            | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lines-spatial-index-loader.ts` | Spatial-index loader for Lines: dual chunk index (segment ordering + vertex ordering), two-stage load (segment chunks → unique vertex set → vertex-attr ranges). Emits a `LoadedLinesData` payload with optional per-vertex scalars for colormap mode.                                                                                                                                                                                                                                                                                                                                                                                       |
| `projection.ts`                 | Exports `createEmptyLinesData` (the "no visible lines" payload) only. The nD→3D clip/interpolate math lives solely in the worker dispatcher `workers/data-worker/projection/lines.ts` (run on a worker, or on the main thread via `workers/data-worker/projection/in-process.ts`); the hand-written main-thread copy (`projectLinesTo3D` / `projectLinesTo3DWASM` / `clipSegmentToSlice` / `initLinesWASM`) was deleted in W4 to leave one implementation.                                                                                                                                                                                   |
| `chunk-index-loader.ts`         | Loads the Lines dual chunk-bounds index (segment + vertex orderings); exposes `registerLinesArrayBounds` as a per-type wrapper around `ChunkPrefetcher.registerArrayBounds`, and the per-segment vertex-index helpers: `sortedUniqueVertexIndices` (typed-array sort + dedupe — no JS `Set`, which V8 caps at 2^24 entries; issue #1049), `computeVertexRangesFromIndices` for coalescing the sorted unique indices into contiguous zarr ranges, and `remapSegmentIndices` (global → local remap via prefix offsets + binary search — no per-vertex `Map`, same 2^24 cap).                                                                   |
| `handler.ts`                    | Per-type scene-loader wiring (`loadAndStage`) — mirrors `points/handler.ts` and `gsplats/handler.ts`. Derives the per-node `LinesViewState`, calls `loader.updateView`, hands the payload to `processLinesData`, and dispatches predictive prefetch via the shared `ViewStateQueue`. Lines use `applyPartialExtendTolerance: false` (verbatim legacy behaviour).                                                                                                                                                                                                                                                                             |
| `lines-progressive-loader.ts`   | `LinesProgressiveLoader` — Composite-pattern multi-additive-LOD facade wrapping N `LinesSpatialIndexLoader` instances (one per `additive_<i>` subgroup). Loads LODs sequentially per `updateView`, stopping after a cache miss / wall-clock budget (`CACHE_HIT_THRESHOLD_MS`); `concatenateLinesData` offset-adjusts each subgroup's local `segments` indices by the cumulative vertex count and fills missing-LOD colors with white. Folded cumulative payloads release each decoded rung's pooled accumulator while logical ladder depth remains separate. Exposes `hasMoreLODs` / `loadedLODCount` / `totalLODCount` / `lastAllResident`. |
| `lod-refinement.ts`             | `runLinesRefinement` — thin wrapper over the generic `scene-loader/progressive/refinement.ts` helper. Re-derives each progressive node's `LinesViewState`, calls `loader.updateView` to stream the next LOD, then `processLines` → `commitLines`. Mirrors `gsplats/lod-refinement.ts` / `points/lod-refinement.ts`. Driven by `scene-loader.ts`.                                                                                                                                                                                                                                                                                             |

## Public surface

`LinesSpatialIndexLoader` implements `LinesDataLoader` (constructor,
`loadLines` / `updateView`, `prefetchChunks`, `dispose`, monitor events via
`addEventListener` / `getMetrics` / `getActiveQueries`) and additionally exposes
`releaseAccumulator()` for the progressive Lines ownership handoff. Scene-loader
code never imports the concrete class — it goes through
`scene-loader/loaders/loader-factory.ts`.

`LinesProgressiveLoader` (in `lines-progressive-loader.ts`) wraps N
single-LOD `LinesSpatialIndexLoader`s for multi-additive-LOD datasets
and satisfies the same `LinesDataLoader` interface, so scene-loader code
is agnostic to whether a node is single- or multi-LOD. The factory
returns it for `additive_<i>` subgroups; `scene-loader.ts` drives
`runLinesRefinement` (from `lod-refinement.ts`) to stream the remaining
LODs in the background once the first level is on screen.

There is no main-thread projection export in this folder: the
main-thread fallback is `projectLinesInProcess` from
`src/workers/data-worker/projection/in-process.ts`, which runs the same
worker dispatcher kernel in-process. `projection.ts` here exports only
`createEmptyLinesData`.

## Invariants

- **Dual chunk index.** Lines maintain two independent chunk
  orderings: one in `(2 × ndim)` segment-endpoint space (for spatial
  queries) and one in `ndim` vertex space (for direct vertex-attr
  loading). The query algorithm walks the segment index to find
  visible segments, then collects the unique vertex set into
  contiguous vertex ranges so each attribute array (positions,
  widths, colors, sharpness, scalars) is fetched with one zarr read
  per range.
- **Labelled nodes publish their on-disk vertex ranges.** A node declaring
  `has_labels` / `has_image_labels` / `has_keys` gets `LoadedLinesData.vertexRangeBounds` — the
  ascending, disjoint ON-DISK vertex ranges the loaded per-vertex arrays
  concatenate, flattened as `[start0, end0, start1, end1, …]` in a `Uint32Array`
  so the SliceCache measures and deep-copies them like any other per-vertex
  array (an object array would be billed 0 bytes and shared by reference). They are one link of the slot → on-disk map picking resolves
  per-vertex labels through (issue #1424; the full chain is documented in
  `data/scene-loader/process/data-processor-lines.ts` and
  `rendering/picking/picking-system/element-id-map.ts`). Gated because nothing
  else reads them and they otherwise ride along in every SliceCache snapshot.
  A ladder never publishes them: `createProgressiveLinesLoader` clears both
  label flags on each synthesized `additive_<i>` node, and
  `concatenateLinesData` strips the field defensively — a sub-LOD's ranges
  describe that level's own on-disk space, not the parent's per-vertex union CSR
  spanning all the levels (#1422); offsetting each level's ranges by the preceding
  levels' on-disk vertex counts, which is that union's index space, is what #1439
  did for the Points ladder — lines has no counterpart yet, so a laddered lines
  node still hovers at the raw segment slot.
- **Progressive concatenation remaps segment indices.** When
  `LinesProgressiveLoader` concatenates per-LOD `LoadedLinesData`, each
  subgroup's `segments` array indexes its _own local_ vertex buffer, so
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
- **In-process `projectLinesInProcess` is the fallback.** On worker
  failure or timeout, `processLinesData` re-runs the SAME dispatcher
  kernel on the main thread via
  `workers/data-worker/projection/in-process.ts`, so numerical parity
  with the worker path holds by construction. A dataset-switch abort
  re-throws rather than falling back (see `WorkerAbortError`).
- **Scalar length validation is fail-closed.** The projection
  dispatcher's input validation (`workers/data-worker/validation.ts`)
  rejects a scalars array shorter than the referenced vertex count
  with a clear error, rather than letting the interpolation kernel
  read past the end of the buffer.

## See also

- `src/types/lines.ts` — type definitions and metadata schema
- `src/rendering/materials/line/material-glsl.ts` and `material-tsl.ts` — GPU-side rendering (WebGL2 + WebGPU dual stack)
- `src/rendering/line-geometry.ts` — instanced mesh construction +
  update helpers
- `src/wasm/typescript/lines-clipping.ts` —
  TypeScript fallback that mirrors the Rust WASM kernel
- `src/data/scene-loader/progressive/refinement.ts` — generic
  multi-LOD refinement loop that `runLinesRefinement` delegates to
- `src/data/loaders/progressive/concat-helpers.ts` — shared
  `concatRequiredField` / `concatOptionalField` used by the
  progressive loader's `concatenateLinesData`
- `src/data/loaders/README.md` — encoding dispatch /
  range-loader contract
- `src/data/README.md` — Lines spatial-index overview
- `src/workers/data-worker/projection/in-process.ts` —
  `projectLinesInProcess`, the in-process (main-thread) fallback for
  the worker projection path
