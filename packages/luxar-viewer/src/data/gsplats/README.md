# GSplats data path

Gaussian-splat-specific data loaders and processing. The wider data
pipeline lives in `src/data/`; this folder contains the bits that are
specific to the GSplats node type.

## Files

| File | Role |
|------|------|
| `gsplats-spatial-index-loader.ts` | Spatial-index loader for GSplats: queries the chunk-bounds index, fetches encoded ranges through `RangeLoader`, and emits a `LoadedGSplatsData` payload |
| `gsplats-processor.ts` | nD→3D projection: marginal Cholesky extraction, Mahalanobis-distance attenuation, 3D center extraction. Mirrors the WASM kernel for use as a TypeScript fallback. |
| `gsplats-progressive-loader.ts` | Progressive amplitude-ordered loading — surfaces the brightest splats first while the rest stream in |
| `chunk-index-loader.ts` | Loads the GSplats chunk-bounds index from zarr metadata |

## Public surface

`GSplatsSpatialIndexLoader` implements the same `SpatialIndexLoader`
contract as Points / Lines (constructor, `loadForView`,
`prefetchChunks`, `dispose`, monitor events). Scene-loader code never
imports the concrete class — it goes through `loader-factory.ts`.

## Invariants

- Marginal Cholesky workspace is **pre-allocated** (`_sigmaWorkspace`,
  `_lSubWorkspace` at module scope; sized to
  `MAX_SUPPORTED_DIMS = 16` from `src/config/constants.ts`). Hot
  loops (called per splat, 100k+ times) must not re-allocate.
- Amplitude-ordered loading preserves visual priority: the loader
  consumes a precomputed `amplitude_order` array stored in zarr so
  the brightest splats arrive in the first range, regardless of
  spatial chunk order.
- Hidden-dim attenuation uses **marginal** Σ, not conditional. The
  marginal path is correct for diagonal hidden-display covariance
  (the typical case for time-stamped / channel-stamped splats). See
  `gsplats-processor.ts` header for the conditional-slicing trade-
  off.

## See also

- `src/types/gsplats.ts` — type definitions and metadata schema
- `src/rendering/gsplat-material.ts` — GPU-side rendering
- `src/wasm/typescript/gsplats-processing.ts` — TypeScript fallback
  that mirrors the Rust WASM kernel
- `src/data/loaders/SPECIFICATIONS.md` — encoding dispatch / range
  loader contract
