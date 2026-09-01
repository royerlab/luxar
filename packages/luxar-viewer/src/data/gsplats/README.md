# GSplats data path

Gaussian-splat-specific data loaders and processing. The wider data
pipeline lives in `src/data/`; this folder contains the bits that are
specific to the GSplats node type.

## Files

| File                              | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `gsplats-spatial-index-loader.ts` | Spatial-index loader for GSplats: queries the chunk-bounds index, fetches encoded ranges through `RangeLoader`, and emits a `LoadedGSplatsData` payload                                                                                                                                                                                                                                                                                                                              |
| `gsplats-progressive-loader.ts`   | Composite multi-LOD facade: wraps N `GSplatsSpatialIndexLoader` instances (one per LOD subgroup), loads them sequentially, and folds the loaded levels into one cumulative payload while tracking logical ladder depth separately. Stops at the first LOD whose load exceeds the cache-hit threshold and prefetches the next one.                                                                                                                                                    |
| `chunk-index-loader.ts`           | Loads the GSplats `chunk_bounds` index from zarr metadata, reconciles the metadata-implied chunk count against the array's implied count (taking the smaller value on mismatch), and exposes `registerGSplatsArrayBounds` as a per-type wrapper around `ChunkPrefetcher.registerArrayBounds`                                                                                                                                                                                         |
| `label-channel.ts`                | Compacts exact unsigned `label_ids` against `label_vocabulary`, projects their GPU-safe indices with visible splats, and resolves picked indices back to exact ids and names                                                                                                                                                                                                                                                                                                         |
| `projection.ts`                   | Exports `createEmptyGSplatsData` (the "no visible splats" payload) only. The nD→3D projection math (marginal Cholesky, Mahalanobis attenuation, 3D center extraction) lives solely in the worker dispatcher `workers/data-worker/projection/gsplats.ts` (run on a worker, or on the main thread via `workers/data-worker/projection/in-process.ts`), backed by the fused WASM kernel + TS reference; the hand-written main-thread copy was deleted in W4 to leave one implementation |
| `handler.ts`                      | Per-type wiring for the scene-loader load+stage path. Derives the gsplats view state (with `applyPartialExtendTolerance: true`), calls `loader.updateView()`, hands off to `processGSplatsData`, and dispatches predictive prefetch                                                                                                                                                                                                                                                  |
| `lod-refinement.ts`               | Progressive LOD refinement loop: after the initial commit, loads remaining LODs one pass per `requestAnimationFrame`, with cancellation when the view-state queue has pending state                                                                                                                                                                                                                                                                                                  |

## Public surface

`GSplatsSpatialIndexLoader` implements the `GSplatsDataLoader`
contract — same shape as the Points and Lines facades (constructor,
`loadGSplats`, `updateView`, `prefetchChunks`, `dispose`, monitor
events via `addEventListener` / `getMetrics` / `getActiveQueries`).
The concrete loader also exposes `releaseAccumulator()` for the progressive
ownership handoff.
Scene-loader code never imports the concrete class — it goes through
`loader-factory.ts`.

## Invariants

- Marginal Cholesky workspace is **pre-allocated** (`_sigmaWorkspace`,
  `_lSubWorkspace` at module scope; sized to
  `MAX_SUPPORTED_DIMS = 16` from `src/config/constants.ts`). Hot
  loops (called per splat, 100k+ times) must not re-allocate.
- Amplitude-ordered LOD priority is structural, not runtime: LOD 0
  carries the coarsest (highest-amplitude) splats and each subsequent
  LOD adds residual detail. `GSplatsProgressiveLoader` concatenates
  loaded LODs in order, so the brightest splats are always present
  first regardless of how many LODs have streamed in.
- Folded ladders keep one cumulative payload and release the decoded rung
  accumulators after each concatenation. Because payload count no longer
  identifies logical depth, an oversized folded SliceCache snapshot cannot be
  trimmed safely and is skipped rather than stored with a cursor ahead of data.
- The picking slot → on-disk element index map is composed at
  **projection** time, not load time (issue #1423). GSplats projection —
  and with it the hidden-dim visibility compaction that renumbers the
  slots — runs downstream of the loader, so the loader only publishes
  its half: `LoadedGSplatsData.ranges`, and only for a node declaring
  `has_labels` / `has_image_labels` / `has_keys`. The fused kernel records the
  surviving source indices, `data-processor-gsplats.ts` composes the two
  via `buildElementIdMap`, and `commit-gsplats-geometry.ts` stamps the
  result onto the MESH (`types/committed-data::setElementIdMap`) — never
  onto the loaded payload, which can be a SliceCache-owned snapshot whose
  byte size was measured at store time and which must not be mutated.
  A ladder payload must never carry the loader's half: a sub-LOD's ranges
  are in that level's on-disk space, so `concatenateGSplatsData` strips
  `ranges`, the one such field on `LoadedGSplatsData`. The composed map
  never passes through the concat at all — it lives one stage later, on
  `ProcessedGSplatsData`. A gsplat ladder carries no string `labels` CSR at
  any level — the authoring path has no such channel for it (the Points / Lines
  ladders put one union CSR on the parent instead, #1422). The categorical
  `label_ids` channel is separate and is preserved on every ladder level.
- Hidden-dim attenuation uses **marginal** Σ, not conditional. The
  marginal path is correct for diagonal hidden-display covariance
  (the typical case for time-stamped / channel-stamped splats). See the
  `wasm/typescript/gsplats-processing.ts` reference kernel (and its Rust
  twin) for the conditional-slicing trade-off.

## See also

- `src/types/gsplats.ts` — type definitions and metadata schema
- `src/rendering/materials/gsplat/` — GPU-side material/rendering
- `src/wasm/typescript/gsplats-processing.ts` — TypeScript fallback
  that mirrors the Rust WASM kernel
- `src/data/loaders/README.md` — encoding dispatch / range
  loader contract
