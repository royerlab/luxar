# Data-monitor wiring

Everything the SceneLoader pushes to the data-monitor UI after a
load completes and on every update cycle: provider registration, the
per-update visible-count tally, and the `SceneNode → SceneGraphNode`
conversion that drives the monitor's tree view.

## Files

| File                       | Role                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `monitor-wiring.ts`        | `wireMonitorAfterLoad(...)` — pushes the resolved `CacheTelemetryState`, then registers cache stats / L0 / GPU buffer pool / per-geometry accumulator-stats / profiler providers, converts the scene graph for the monitor's tree view, runs the initial visible-counts pass, and calls `forceUpdate()`.                                                   |
| `scene-graph-converter.ts` | `convertToSceneGraphNode(node)` — pure recursive conversion from the loader's `SceneNode` to the monitor UI's `SceneGraphNode`, with type whitelisting (`scene` / `group` / `points` / `lines` / `gsplats` / `mesh`), display-name derivation (`/` → `"Scene"`), and per-type stats (`pointCount`, `segmentCount` + `vertexCount`, `splatCount`).           |
| `visible-counts.ts`        | `updateVisibleCountsInMonitor(rootGroup, monitor)` — traverses the root group, sums per-mesh `visibleSegmentCount` / `visibleSplatCount` userData (lines + gsplats), and pushes the totals to the monitor. Called once per update cycle after the lines/gsplats commits so the HUD shows post-clipping visible counts.                                     |

## Consumers

- `../lifecycle/load-scene.ts` calls `wireMonitorAfterLoad` once at
  the tail of every `loadScene`.
- `../../scene-loader.ts` (parent orchestrator) calls
  `updateVisibleCountsInMonitor` at the end of every `updateView`
  cycle.
- `monitor-wiring.ts` internally uses `scene-graph-converter.ts` to
  produce the SceneGraphNode tree.
