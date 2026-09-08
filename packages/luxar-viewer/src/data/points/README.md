# Points data path

Points-specific data loaders and projection. The wider data pipeline
lives in `src/data/`; this folder contains the bits that are specific
to the Points node type.

## Files

| File                             | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `points-spatial-index-loader.ts` | Spatial-index loader for Points: queries the chunk-bounds index, fetches encoded ranges through `RangeLoader`, and emits a `LoadedPointsData` payload. `loadPoints` is a thin wrapper (S-cache restore → `loadPointsInternal` → shared `finishQueryTracking` close-out → S-cache store / error emit) mirroring the Lines and GSplats facades. Owns the per-loader `LoadedPointsDataAccumulator` for the zero-allocation hot path.                                                                                                                    |
| `points-progressive-loader.ts`   | Composite-pattern multi-additive-LOD facade. Wraps N `PointsSpatialIndexLoader` instances (one per `additive_<i>` subgroup), loads LODs sequentially from LOD 0, stops at the first cache miss / `CACHE_HIT_THRESHOLD_MS` overrun, keeps one-rung lookahead, and folds the loaded levels into one cumulative `LoadedPointsData` while tracking logical ladder depth separately. GSplats instead use deeper cache-bounded lookahead.                                                                                                                  |
| `lod-refinement.ts`              | Progressive Points LOD refinement — thin wrapper over the generic `data/scene-loader/progressive/refinement.ts`. Drives the per-frame refinement loop for progressive loaders (those exposing `hasMoreLODs`); commits each refined load directly via `updatePointsGeometry`. Mirrors `data/gsplats/lod-refinement.ts`.                                                                                                                                                                                                                               |
| `projection.ts`                  | The single, **WASM-accelerated** nD→3D Points projection (`projectPointsTo3D`) + `createEmptyPointsData`. Runs the `extract_3d_positions` / `calculate_effective_radii` WASM kernels (via a `wasm` backend from `getPointsBackend`), then filters by visibility, normalizes uint8 radii (`/255`), and writes through `targetBuffers` (zero-alloc accumulator). Runs on the **main thread** — Points projection is bandwidth-bound and pairs with the accumulator, so it isn't worker-offloaded (W4b removed the former dead worker dispatcher copy). |
| `effective-radius-calculator.ts` | `calculateSpatialQueryTolerance` + `shouldApplyEffectiveRadius` (used by the loader), plus a TS reference `calculateEffectiveRadii` (the production path now uses the WASM kernel; the TS twin lives in `wasm/typescript/effective-radii.ts`). Points-only — Lines and GSplats carry equivalent info in segment bounds / Cholesky factors.                                                                                                                                                                                                           |
| `chunk-index-loader.ts`          | Loads the Points chunk-bounds index from zarr metadata; exposes `registerPointsArrayBounds` as a per-type wrapper around `ChunkPrefetcher.registerArrayBounds`.                                                                                                                                                                                                                                                                                                                                                                                      |
| `handler.ts`                     | Per-type wiring for the scene-loader's load + stage phase. Exports `loadAndStage` (skip → `loader.updateView` → failure-clear → metadata → predictive-prefetch dispatch), plus `kind`/`label` constants and the `StagedPointsCommit` / `PointsHandlerCtx` shapes. Lines and GSplats mirror this shape so all first-class geometry kinds stay symmetrical.                                                                                                                                                                                            |

## Public surface

`PointsSpatialIndexLoader` implements the `DataLoader` contract —
same shape as the Lines and GSplats facades (constructor, `loadPoints`,
`updateView`, `prefetchChunks`, `dispose`, monitor events via
`addEventListener` / `getMetrics` / `getActiveQueries`).
The concrete loader also exposes `releaseAccumulator()` for the progressive
ownership handoff.
Scene-loader code never imports the concrete class — it goes through
`loader-factory.ts`. For multi-additive-LOD datasets the factory
constructs a `PointsProgressiveLoader` instead, which implements the
same contract (plus `hasMoreLODs` / `loadedLODCount` / `totalLODCount`
/ `lastAllResident` for the refinement loop) over N per-LOD loaders.

`projectPointsTo3D` (in `projection.ts`) is the single projection
implementation; the loader calls it directly with a WASM backend from
`getPointsBackend(ndim)`. It is always on the main thread (WASM-accelerated)
— there is no worker round-trip for Points.

It also emits `elementIds` (via the shared `buildElementIdMap` in
`data/loaders/element-ids.ts`, which GSplats and Lines compose with too): the
visible-buffer slot → on-disk element index map that picking uses for
per-element string/image lookups. It is built only for a node declaring
`has_labels` / `has_image_labels` / `has_keys` — the readers it exists for, and
it costs 4 B/point on the zero-allocation path — and omitted on the identity
path (one range starting at 0, no effective-radius compaction). A node without
one of those channels can still be picked when an interaction template or an
embedder `selection` / element-action listener provisions picking, and its
`elementIndex` keeps reporting the storage slot. Across an additive ladder,
`PointsProgressiveLoader` composes the per-level maps into the PARENT node's
union label/key CSR space (`additive_0 || additive_1 || …`) by offsetting level
`i` with the preceding levels' on-disk `n_points` — and strips them entirely
when the parent declares no union label/key CSR, since a sub-LOD's own index
space is not one any reader can key by (#1439).

## Invariants

- **Effective radius is Points-only.** Lines use precomputed segment
  bounds (already in the chunk-bounds index); GSplats use
  Cholesky-derived tolerance. Points need a per-dim radius adjustment
  because a `pointRadius` of `r` enlarges visibility by `r` in every
  non-displayed axis — that's exactly what
  `EffectiveRadiusCalculator` computes.
- **Zero-allocation accumulator hot path.** The loader owns a
  `LoadedPointsDataAccumulator` sized to the dataset's `pointCount`. Every
  nD scrub writes through pre-allocated `positions3D` / `colors` /
  `radii` / `sharpness` / `scalars` buffers — no per-update
  allocation. The accumulator preserves Uint8/Uint16 dtypes natively
  for colors/radii/sharpness so a Uint8 zarr array round-trips
  without intermediate Float32 widening. One bounded exception: the
  `elementIds` map is a fresh `Uint32Array(numPoints)` per update, and
  only for a node declaring a string/image channel on a non-identity range set
  — a node without one, a single `[0, N)` range, and every sub-LOD of a ladder
  whose parent declares no union label/key CSR all skip it.
- **Folded ladders trade prefix caching for bounded residency.** After each
  concatenation, progressive Points keeps one cumulative payload and releases
  the decoded rung accumulators that backed it. A cumulative payload cannot be
  trimmed back to a coarse prefix without recreating those rungs, so an
  oversized folded snapshot is not cached and re-decodes on revisit.
- **Dtype-aware scale propagation.** `radiusScale` lives on
  `geometry.userData` and is propagated to the material's uniforms via
  `syncPointMaterialWithGeometry` (the only one of the three
  material-sync helpers — Lines and GSplats don't have dtype-tagged
  scalar attributes). Sharpness has no scale: it is authored natively
  in `[0, 1]` (a `uint8/255` value already lands in range).
- **`MAX_SUPPORTED_DIMS = 16`** from `src/config/constants.ts` bounds
  the WASM stack-array sizes; the TypeScript fallback in
  `wasm/typescript/effective-radii.ts` mirrors this.

## See also

- `src/types/points.ts` — type definitions and metadata schema
- `src/rendering/materials/point/` — GPU-side rendering
  (`material-glsl.ts` / `material-tsl.ts` variants share the
  `shader-glsl.ts` / `shader-tsl.ts` sources)
- `src/rendering/material-sync-helpers.ts` —
  `syncPointMaterialWithGeometry` propagates dtype scales
- `src/wasm/typescript/effective-radii.ts` —
  TypeScript fallback that mirrors the Rust WASM kernel
- `src/data/loaders/README.md` — encoding dispatch /
  range-loader contract
