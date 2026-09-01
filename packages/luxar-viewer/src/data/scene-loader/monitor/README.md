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
  provider reports committed `loaded / total` levels (falling back to the
  progressive loader's cursor when no commit stamp exists), a refining flag,
  and the last-load cache-residency.
- **Partition** (`kind=partition` groups): N disjoint BSP parts, all
  eligible to render together when in the camera frustum. The converter tags
  `kind='partition'` + `partCount`; totals sum across parts (correct — parts
  are disjoint). The structural `{ path, partCount }` does not change as the
  frustum selector hides and reveals parts, so `monitor-wiring.ts` snapshots it
  once at wire-time and feeds it to the LOD-progress provider, which surfaces
  the group as a `kind:'partition'` state.

`visible-counts.ts` prunes hidden subtrees so inactive LOD levels (and
toggled-off layers) don't double-count toward the visible HUD totals.

## Files

| File                       | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `monitor-wiring.ts`        | `wireMonitorAfterLoad(...)` — pushes the resolved `CacheTelemetryState`, then registers cache stats / L0 / GPU buffer pool / per-geometry accumulator-stats / profiler / LOD-progress providers, converts the scene graph for the monitor's tree view, runs the initial visible-counts pass, and calls `forceUpdate()`. The LOD-progress provider is wired with `collectPartitionGroups(sceneGraph)` (a local walk gathering `kind=partition` groups as `{ path, partCount }`).                                                         |
| `scene-graph-converter.ts` | `convertToSceneGraphNode(node)` — pure recursive conversion from the loader's `SceneNode` to the monitor UI's `SceneGraphNode`: type whitelisting, display-name derivation (`/` → `"Scene"`), per-type stats (`pointCount`, `segmentCount` + `vertexCount`, `splatCount`, `faceCount` + `vertexCount`), and specialized-group fields (`kind`, `displayType`, `lodGroupChildCount` / `partCount`, `additiveSublods`).                                                                                                                    |
| `visible-counts.ts`        | `updateVisibleCountsInMonitor(rootGroup, monitor)` — recurses the root group, **skipping `visible === false` subtrees**, sums per-mesh `visiblePointCount` / `visibleSegmentCount` / `visibleSplatCount` / `visibleTriangleCount` userData (all four geometry types) plus `droppedElementCount` (points + lines + gsplats only — mesh is not element-texture backed), and pushes the visible and dropped totals to the monitor. Called once per update cycle after the commits so the HUD shows post-clipping, post-LOD visible counts. |
| `committed-lod-reader.ts`  | `createCommittedLODCountReader(rootGroup)` — reads each node's commit-time `committedLODCount` stamp from the live THREE scene so additive progress reports rungs actually on screen rather than the loader cursor.                                                                                                                                                                                                                                                                                                                     |
| `lod-progress-provider.ts` | `createLODProgressProvider({ loaderMaps, lodGroupRegistry, partitionGroups?, committedLODCounts? })` — builds the `path → LODProgressState` snapshot the monitor polls each tick: additive `loaded/total/refining/lastAllResident` from the progressive loaders (duck-typed via the `ProgressiveLike` getter surface), with `loaded` taken from the committed-count reader when available; substitutive `activeLevel/levelCount/selector` from the `LODGroupRegistry`; and static `partCount` from the partition-group snapshot.        |

## Consumers

- `../lifecycle/load-scene.ts` calls `wireMonitorAfterLoad` once at
  the tail of every `loadScene`.
- `../../scene-loader.ts` (parent orchestrator) calls
  `updateVisibleCountsInMonitor` at the end of every `updateView`
  cycle.
- `monitor-wiring.ts` internally uses `scene-graph-converter.ts` to
  produce the SceneGraphNode tree.
