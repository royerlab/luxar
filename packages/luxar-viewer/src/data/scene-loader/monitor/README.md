# Data-monitor wiring

Everything the SceneLoader pushes to the data-monitor UI after a
load completes and on every update cycle: provider registration, the
per-update visible-count tally, the `SceneNode → SceneGraphNode`
conversion that drives the monitor's tree view, and the live
LOD/refinement/residency provider.

## LOD & partition awareness

The monitor distinguishes the three LOD/partition loading shapes so its
numbers stay honest and its tree reads clearly:

- **Substitutive LOD** (`kind=lod` groups): K mutually-exclusive levels,
  one rendered at a time. The converter tags the node with `kind='lod'`
  and `lodGroupChildCount`; the UI aggregator counts the **finest** level
  for dataset totals (not the sum — that would inflate ~K×) and
  `getGlobalStats` collapses the K level loaders to one logical layer.
- **Additive LOD** (`n_additive_sublods > 1` leaves): progressively
  refined. The converter records `additiveSublods`; the LOD-progress
  provider reports `loaded / total` levels, a refining flag, and the
  last-load cache-residency.
- **Partition** (`kind=partition` groups): N disjoint BSP parts, all
  rendered. The converter tags `kind='partition'` + `partCount`; totals
  sum across parts (correct — parts are disjoint). Partition groups are
  static (no per-frame selector), so `monitor-wiring.ts` snapshots their
  `{ path, partCount }` once at wire-time and feeds them to the
  LOD-progress provider, which surfaces them as `kind:'partition'` states.

`visible-counts.ts` prunes hidden subtrees so inactive LOD levels (and
toggled-off layers) don't double-count toward the visible HUD totals.

## Files

| File                       | Role                                                                                                                                                                                                                                                                                                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `monitor-wiring.ts`        | `wireMonitorAfterLoad(...)` — pushes the resolved `CacheTelemetryState`, then registers cache stats / L0 / GPU buffer pool / per-geometry accumulator-stats / profiler / LOD-progress providers, converts the scene graph for the monitor's tree view, runs the initial visible-counts pass, and calls `forceUpdate()`. The LOD-progress provider is wired with `collectPartitionGroups(sceneGraph)` (a local walk gathering `kind=partition` groups as `{ path, partCount }`).                                                                     |
| `scene-graph-converter.ts` | `convertToSceneGraphNode(node)` — pure recursive conversion from the loader's `SceneNode` to the monitor UI's `SceneGraphNode`: type whitelisting, display-name derivation (`/` → `"Scene"`), per-type stats (`pointCount`, `segmentCount` + `vertexCount`, `splatCount`), and specialized-group fields (`kind`, `displayType`, `lodGroupChildCount` / `partCount`, `additiveSublods`).     |
| `visible-counts.ts`        | `updateVisibleCountsInMonitor(rootGroup, monitor)` — recurses the root group, **skipping `visible === false` subtrees**, sums per-mesh `visiblePointCount` / `visibleSegmentCount` / `visibleSplatCount` userData (points + lines + gsplats), and pushes the totals to the monitor. Called once per update cycle after the commits so the HUD shows post-clipping, post-LOD visible counts. |
| `lod-progress-provider.ts` | `createLODProgressProvider({ loaderMaps, lodGroupRegistry, partitionGroups? })` — builds the `path → LODProgressState` snapshot the monitor polls each tick: additive `loaded/total/refining/lastAllResident` from the progressive loaders (duck-typed via the `ProgressiveLike` getter surface), substitutive `activeLevel/levelCount/selector` from the `LODGroupRegistry`, and static `partCount` from the partition-group snapshot.                                                                                       |

## Consumers

- `../lifecycle/load-scene.ts` calls `wireMonitorAfterLoad` once at
  the tail of every `loadScene`.
- `../../scene-loader.ts` (parent orchestrator) calls
  `updateVisibleCountsInMonitor` at the end of every `updateView`
  cycle.
- `monitor-wiring.ts` internally uses `scene-graph-converter.ts` to
  produce the SceneGraphNode tree.
