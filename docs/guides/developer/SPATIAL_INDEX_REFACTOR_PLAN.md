# Spatial Index Consolidation Plan

**Status:** Plan only — no implementation has started.
**Owner:** TBD (see "Open questions" below).

## Problem

Three near-parallel TypeScript files implement chunk-based spatial indices for the three geometry types (points, gsplats, lines):

| File | LOC | Geometry |
|------|-----|----------|
| `packages/luxar-viewer/src/data/chunk-spatial-index.ts` | 392 | Points |
| `packages/luxar-viewer/src/data/gsplats-chunk-spatial-index.ts` | 248 | GSplats |
| `packages/luxar-viewer/src/data/lines-chunk-spatial-index.ts` | 391 | Lines |

A fourth file, `packages/luxar-viewer/src/data/loaders/spatial-query-builder.ts` (435 LOC), already factors out the shared logic (AABB intersection, range merging, tolerance helpers, a fluent `SpatialQueryBuilder` class) — but **none of the three geometry-specific files use it yet**. The duplication is ~150 LOC of nearly-identical AABB scans, range merges, and `chunkIndicesToRanges()` reimplementations.

## What is genuinely shared

1. **AABB intersection loop** — identical min/max overlap test across all three.
2. **Range merging** — identical merge-by-start logic.
3. **Chunk-to-range conversion** — `chunkIndicesToRanges()` is identical (only the wrapper type names differ).
4. **Load skeleton** — `zarr.open() → get() → validate` is the same pattern; only the array names and validation invariants differ.

## What is genuinely geometry-specific

1. **Tolerance calculation**
   - Points: `step × multiplier`.
   - GSplats: 3σ Gaussian extent (default), discrete dims = 0.5.
   - Lines: 0 for spatial dims (bounds already include line width), 0.5 for discrete, `1e10` for displayed.
2. **Number of bounds arrays**
   - Points / GSplats: single bounds array.
   - Lines: two — vertex bounds **and** segment bounds.
3. **Vertex derivation** (lines only) — `computeVertexChunksForIndices()`, `computeVertexRangesFromIndices()` translate segment indices to vertex ranges for two-phase loading.

## Recommended approach: composition + strategy

Keep the three loaders calling shared utilities with thin geometry-specific strategies. **Avoid an inheritance base class** — there is no shared mutable state, just shared pure functions and one geometry-specific config bundle.

### Step-by-step migration

1. **Adopt `spatial-query-builder.ts`'s shared utilities in the three loaders.** No new abstraction yet — just delete the duplicated `mergeRanges`, `chunkIndicesToRanges`, and AABB-scan implementations and call the shared ones. This alone removes ~150 LOC and is mechanical/safe.
2. **Define a `GeometryIndexStrategy` interface** in a new `spatial-index/strategy.ts`:
   ```ts
   interface GeometryIndexStrategy<TIndex, TBounds> {
     loadBounds(location: Location, attrs: Attrs): Promise<TBounds>;
     computeTolerance(view: BaseViewState, dimensions: Dimension[]): number[];
     // Optional: for geometries with multiple bounds arrays.
     queryAuxiliary?(index: TIndex, query: SpatialQueryParams): LoadRange[];
   }
   ```
3. **Implement three strategies** — `pointsStrategy`, `gsplatsStrategy`, `linesStrategy` — each ~40-80 LOC, holding only the geometry-specific tolerance and bounds-loading logic.
4. **Migrate one loader at a time** (start with the simplest — gsplats, since it has no auxiliary index). Each migration is a self-contained PR with unchanged public API.
5. **Delete the three legacy `*-chunk-spatial-index.ts` files** once their consumers have migrated.

### Public-API impact

Existing consumers (call sites enumerated below) continue calling `point-spatial-index-loader`, `gsplats-spatial-index-loader`, `lines-spatial-index-loader` — those entry points stay; only their internals shift to use shared utilities. **Zero call-site changes** in the first phase.

The exported interfaces (`ChunkSpatialIndex`, `GSplatsChunkSpatialIndex`, `LinesChunkSpatialIndex`) remain. They could later be unified, but that touches several `types/*.ts` files and unit tests — defer to a phase 2.

## Call-site inventory

Primary consumers (3 loaders):
- `data/point-spatial-index-loader.ts` — 4 callsites
- `data/gsplats-spatial-index-loader.ts` — 4 callsites
- `data/lines-spatial-index-loader.ts` — 4+ callsites

Secondary surface:
- `data/index.ts` — re-exports (7 entries).
- `types/points.ts`, `types/gsplats.ts`, `types/lines.ts` — interface definitions.
- 6 unit test files (one per geometry × 2 phases).

## Estimated effort

| Phase | Description | LOC delta | Risk |
|-------|-------------|-----------|------|
| 1 | Adopt `spatial-query-builder` utilities in three loaders | −150 | Low (mechanical) |
| 2 | Strategy interface + three strategies | +200 / −300 | Medium (new abstraction) |
| 3 | Delete legacy `*-chunk-spatial-index.ts` files | −600 | Low (after callers migrated) |

Net: ~−500 LOC, plus a clearer extension point for adding new geometry types (e.g., meshes).

## Open questions

1. **Should the three `*ChunkSpatialIndex` interfaces be unified into a single discriminated-union type?** Pro: removes redundancy in `types/`. Con: touches Python-side type generation if any.
2. **Does `SpatialQueryBuilder` (the fluent class) survive the refactor, or do we standardize on free functions?** Currently unused — picking one path before phase 2 avoids a half-adopted API.
3. **Two-phase load (lines only)** — is it worth generalizing to other geometries (e.g., gsplats with a coarse + fine tile structure)? If yes, the strategy interface needs `queryAuxiliary?`. If no, lines stays the only consumer and the optional method can be inlined.

## Non-goals

- Performance changes — this is a structure-only refactor.
- New geometry types — the strategy interface is designed to make adding them easier, but implementing them is out of scope.
- Touching the worker-side spatial code (`workers/data-worker.ts` does not use these files directly).
