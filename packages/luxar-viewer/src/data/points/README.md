# Points data path

Points-specific data loaders and projection. The wider data pipeline
lives in `src/data/`; this folder contains the bits that are specific
to the Points node type.

## Files

| File                             | Role                                                                                                                                                                                                                                                                               |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `points-spatial-index-loader.ts` | Spatial-index loader for Points: queries the chunk-bounds index, fetches encoded ranges through `RangeLoader`, and emits a `LoadedPointsData` payload. Owns the per-loader `DataAccumulator` for the zero-allocation hot path.                                                     |
| `projection.ts`                  | nD→3D projection: extracts displayed coordinates, computes effective radii, filters by visibility, and writes through `targetBuffers` when present. Mirrors the WASM kernel (`workers/data-worker.ts::projectPointsTo3D`) so the main-thread fallback stays numerically identical. |
| `effective-radius-calculator.ts` | Points-only nD effective-radius computation. Combines `maxRadius` with per-dimension extend offsets and hidden-axis distances. Lines and GSplats don't need this — segment bounds and Cholesky factors carry the equivalent info inline.                                           |
| `chunk-index-loader.ts`          | Loads the Points chunk-bounds index from zarr metadata; exposes `registerPointsArrayBounds` as a per-type wrapper around `ChunkPrefetcher.registerArrayBounds`.                                                                                                                    |

## Public surface

`PointsSpatialIndexLoader` implements the same `SpatialIndexLoader`
contract as Lines / GSplats (constructor, `loadForView`,
`prefetchChunks`, `dispose`, monitor events). Scene-loader code never
imports the concrete class — it goes through `loader-factory.ts`.

`projectPointsTo3D` is also exported from `projection.ts` for direct
main-thread use (worker-disabled environments, unit tests).

## Invariants

- **Effective radius is Points-only.** Lines use precomputed segment
  bounds (already in the chunk-bounds index); GSplats use
  Cholesky-derived tolerance. Points need a per-dim radius adjustment
  because a `pointRadius` of `r` enlarges visibility by `r` in every
  non-displayed axis — that's exactly what
  `EffectiveRadiusCalculator` computes.
- **Zero-allocation accumulator hot path.** The loader owns a
  `PointsDataAccumulator` sized to the dataset's `pointCount`. Every
  nD scrub writes through pre-allocated `positions3D` / `colors` /
  `radii` / `sharpness` / `scalars` buffers — no per-update
  allocation. The accumulator preserves Uint8/Uint16 dtypes natively
  for colors/radii/sharpness so a Uint8 zarr array round-trips
  without intermediate Float32 widening.
- **Dtype-aware scale propagation.** `radiusScale` /
  `sharpnessScale` / `scalarScale` live on
  `geometry.userData` and are propagated to the material's uniforms
  via `syncPointMaterialWithGeometry` (the only one of the three
  material-sync helpers — Lines and GSplats don't have dtype-tagged
  scalar attributes).
- **`MAX_SUPPORTED_DIMS = 16`** from `src/config/constants.ts` bounds
  the WASM stack-array sizes; the TypeScript fallback in
  `wasm/typescript/effective-radii.ts` mirrors this.

## See also

- `src/types/points.ts` — type definitions and metadata schema
- `src/rendering/point-material.ts` — GPU-side rendering
- `src/rendering/material-sync-helpers.ts` —
  `syncPointMaterialWithGeometry` propagates dtype scales
- `src/wasm/typescript/points.ts` and `effective-radii.ts` —
  TypeScript fallback that mirrors the Rust WASM kernel
- `src/data/loaders/README.md` — encoding dispatch /
  range-loader contract
