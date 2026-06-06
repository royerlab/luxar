# Data stats

Scene-level and per-loader statistics for the data layer. Two small,
pure helpers feed the data-loading monitor and the debug-logging
emitted at the end of `loadScene`: one walks the THREE.js scene graph
and counts loaded geometry; the other sums the per-loader
`AccumulatorStats` reported by the zero-allocation buffer pools.

Both helpers are deliberately I/O-free and side-effect-free so they
can be unit-tested directly against synthetic scenes and stub loader
maps without spinning up WebGL or a zarr store.

## Files

| File             | Role                                                                                                                                                                                                                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `scene-stats.ts` | Pure scene-graph traversal: `computeSceneStats(scene)` walks the THREE object tree once and returns counts of points/lines/gsplats meshes, total instance counts (points instances, line segments, splats), and how many of those nodes carry a spatial index. Returns `null` for scenes whose `traverse` is not a function (mocks). |
| `aggregator.ts`  | Sums `AccumulatorStats` across a `Map<path, loader>` of spatial-index loaders. One per-geometry public wrapper (`getAggregatedPointsAccumulatorStats`, `...Lines...`, `...GSplats...`) calls a shared private `aggregateStats` over any iterable of loaders exposing `getAccumulatorStats()`.                                        |

## Public surface

```typescript
// scene-stats.ts
export interface SceneStats {
  pointsObjects: number;
  totalPoints: number;
  linesObjects: number;
  totalSegments: number;
  gsplatsObjects: number;
  totalGSplats: number;
  spatialIndexed: number;
}
export function computeSceneStats(scene: THREE.Object3D | null | undefined): SceneStats | null;

// aggregator.ts — one wrapper per geometry kind, all returning AccumulatorStats
export function getAggregatedPointsAccumulatorStats(
  loaders: ReadonlyMap<string, DataLoader>
): AccumulatorStats;
export function getAggregatedLinesAccumulatorStats(
  loaders: ReadonlyMap<string, LinesDataLoader>
): AccumulatorStats;
export function getAggregatedGSplatsAccumulatorStats(
  loaders: ReadonlyMap<string, GSplatsDataLoader>
): AccumulatorStats;
```

## How the counts are derived

`computeSceneStats` traverses the scene once and inspects each
`THREE.Mesh` tagged by `userData.nodeType`:

- **`nodeType === 'points'`** — increments `pointsObjects`. The
  per-mesh point count is taken from the most authoritative source
  available, in this order:
  1. `geometry.instanceCount` (when the geometry is an
     `InstancedBufferGeometry`) — the source of truth for the
     **visible** instance count, since pooled `aCenter` attributes may
     be over-allocated beyond what's currently drawn.
  2. `userData.visiblePointCount` — set by the commit phase when the
     instance count is not directly readable.
  3. `geometry.getAttribute('aCenter').count` — last-resort total
     capacity.
- **`nodeType === 'lines'`** — increments `linesObjects`. Like Points,
  Lines are instanced quad meshes (one instance per segment), so the
  per-mesh segment count is read from:
  1. `geometry.instanceCount` (when the geometry is an
     `InstancedBufferGeometry`) — the source of truth for visible
     segments.
  2. `userData.visibleSegmentCount` — fallback for tests that
     synthesize Lines meshes without setting the instance count.
     Both feed `totalSegments`. (There is no `aCenter` last-resort branch
     for lines.)
- **`nodeType === 'gsplats'`** — increments `gsplatsObjects` and adds
  `userData.visibleSplatCount` (default `0`) to `totalGSplats`.
- For all three kinds, `userData.attrs?.has_spatial_index` truthy
  increments `spatialIndexed`.

## How aggregation works

`aggregateStats` (private to `aggregator.ts`) walks a loader iterable
and sums four fields from each loader's `AccumulatorStats`:
`capacity`, `allocations`, `growthEvents`, `memoryMB`. Loaders that
do not implement `getAccumulatorStats` — or that return `null` (e.g.
before the first fill) — are skipped silently, so the aggregator is
safe to call mid-load.

The three public wrappers differ only in their input map type
(`DataLoader` / `LinesDataLoader` / `GSplatsDataLoader`); the
aggregation logic is identical. Keeping them separate matches the
three-geometry symmetry rule used elsewhere in `data/` and lets the
monitor wire one provider per geometry kind in
`scene-loader/monitor/monitor-wiring.ts`.

## Invariants

- **Pure functions.** Both helpers are read-only over their inputs
  and have no module-level state. Calling them in a hot path is cheap
  enough that the monitor invokes them on every UI refresh.
- **Defensive against partial scenes.** `computeSceneStats` tolerates
  `null`/`undefined` scenes and scenes whose `traverse` has been
  mocked away (returns `null`); `aggregateStats` tolerates loaders
  without `getAccumulatorStats` and `null` stats. Neither helper
  throws on shape mismatches — the monitor stays robust during async
  load transitions.
- **Single source of truth for visible counts.** When a points or
  lines mesh is `InstancedBufferGeometry`, `instanceCount` overrides
  any `userData.visiblePointCount` / `userData.visibleSegmentCount`.
  Producers that need to report a different visible count must set the
  instance count on the geometry, not the userData.

## Callers

- `data/zarr-loader.ts` — calls `computeSceneStats` at the end of
  `loadScene` and emits one debug log line per stat.
- `data/scene-loader/monitor/monitor-wiring.ts` — wires the three
  `getAggregated*AccumulatorStats` wrappers into the
  `AccumulatorProviderPort` consumed by the data-loading monitor.

## See also

- `../accumulators/README.md` — defines `AccumulatorStats` and the
  per-geometry accumulators that produce it.
- `../scene-loader/monitor/monitor-wiring.ts` — the only consumer of the
  aggregator wrappers.
- `../zarr-loader.ts` — the only caller of `computeSceneStats`.
