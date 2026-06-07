# GSplats data path

Gaussian-splat-specific data loaders and processing. The wider data
pipeline lives in `src/data/`; this folder contains the bits that are
specific to the GSplats node type.

## Files

| File                              | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `gsplats-spatial-index-loader.ts` | Spatial-index loader for GSplats: queries the chunk-bounds index, fetches encoded ranges through `RangeLoader`, and emits a `LoadedGSplatsData` payload                                                                                                                                                                                                                                                                                                                              |
| `gsplats-progressive-loader.ts`   | Composite multi-LOD facade: wraps N `GSplatsSpatialIndexLoader` instances (one per LOD subgroup) and loads them sequentially. Stops at the first LOD whose load exceeds the cache-hit threshold and prefetches the next one                                                                                                                                                                                                                                                          |
| `chunk-index-loader.ts`           | Loads the GSplats `chunk_bounds` index from zarr metadata, reconciles the metadata-implied chunk count against the array's implied count (taking the smaller value on mismatch), and exposes `registerGSplatsArrayBounds` as a per-type wrapper around `ChunkPrefetcher.registerArrayBounds`                                                                                                                                                                                         |
| `projection.ts`                   | Exports `createEmptyGSplatsData` (the "no visible splats" payload) only. The nD→3D projection math (marginal Cholesky, Mahalanobis attenuation, 3D center extraction) lives solely in the worker dispatcher `workers/data-worker/projection/gsplats.ts` (run on a worker, or on the main thread via `workers/data-worker/projection/in-process.ts`), backed by the fused WASM kernel + TS reference; the hand-written main-thread copy was deleted in W4 to leave one implementation |
| `handler.ts`                      | Per-type wiring for the scene-loader load+stage path. Derives the gsplats view state (with `applyPartialExtendTolerance: true`), calls `loader.updateView()`, hands off to `processGSplatsData`, and dispatches predictive prefetch                                                                                                                                                                                                                                                  |
| `lod-refinement.ts`               | Progressive LOD refinement loop: after the initial commit, loads remaining LODs one pass per `requestAnimationFrame`, with cancellation when the view-state queue has pending state                                                                                                                                                                                                                                                                                                  |

## Public surface

`GSplatsSpatialIndexLoader` implements the `GSplatsDataLoader`
contract — same shape as the Points and Lines facades (constructor,
`loadGSplats`, `updateView`, `prefetchChunks`, `dispose`, monitor
events via `addEventListener` / `getMetrics` / `getActiveQueries`).
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
