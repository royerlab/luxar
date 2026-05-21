# SceneLoader extracted helpers

Modules composed by `data/scene-loader.ts`. The orchestrator class
stays in the parent `data/` folder; the focused, testable units that
implement each non-trivial sub-step of its lifecycle live here. Every
file in this folder is consumed by `SceneLoader` (and a few are
consumed by the `Subpackages` listed at the bottom) — none are
re-exported from `data/index.ts`.

The SceneLoader has three external entry points:

- **`loadScene(url)`** — initial load (cache stack, zarr open, scene
  graph build, recursive per-node load, post-load monitor wiring,
  GSplats LOD kick).
- **`updateView(viewState)`** — slider / animation / keyboard nav. Runs
  the per-type async process step then the synchronous atomic commit,
  serialized through a view-state queue with predictive prefetch.
- **`dispose()`** — release dataset-scoped resources before the next
  `loadScene`.

This folder is split along that lifecycle. Top-level files here are the
shared scaffolding; the four subpackages below own the per-geometry
work each entry point dispatches into.

## Files

| File                         | Role                                                                                                                                                                                                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load-scene.ts`              | Initial-load orchestrator body. Aborts in-flight worker tasks, disposes the previous loader, opens the L0/L1/L2 cache stack, opens the zarr root, builds the empty root `THREE.Group`, initializes scene dimensions, persists `viewer_config` + `position_bounds`, builds the scene graph, recursively loads every leaf, loads overlay configs, wires the post-load monitor providers, and kicks GSplats LOD refinement when multi-LOD loaders still have higher LODs pending. |
| `cache-setup.ts`             | `setupCaches(url, flags)` — builds the L0 (`DecompressedChunkCache`) + L1/L2 (`MultiLevelCachingStore`) layers, attaches the `ChunkPrefetcher`, registers L0 invalidation on L1/L2 clear, and resolves the `CacheTelemetryState` for the monitor UI. Returns the raw store the Luxar Zarr facade opens.                                                                                                                                                                          |
| `cache-api.ts`               | Pure get/clear/list surface for the three cache levels — `getCacheStats`, `listCachedDatasets`, `clearL0Cache`, `clearL1Cache`, `clearL2Cache`, `clearAllCaches`. Each no-ops when its layer is `null` (URL `?no-cache` or app-config disable). Defines the `CacheStatsSnapshot` shape `__luxarDebug.cache.getStats()` returns.                                                                                                                                                |
| `monitor-wiring.ts`          | `wireMonitorAfterLoad(...)` — pushes the resolved `CacheTelemetryState`, then registers cache stats / L0 / GPU buffer pool / per-geometry accumulator-stats / profiler providers, converts the scene graph for the monitor's tree view, runs the initial visible-counts pass, and calls `forceUpdate()`.                                                                                                                                                                       |
| `dispose.ts`                 | `disposeSceneLoader(ctx)` — aborts the dataset signal, disposes every registered loader, the GPU buffer pool, the caching store (awaited so L2 flushes drain), the L0 cache, the predictive-prefetch baseline, dataset-scoped custom colormap LUTs, and the monitor's loader-bound closures. Worker pool itself is NOT terminated — `core/app.ts` owns that.                                                                                                                |
| `loader-registry.ts`         | `LoaderRegistry` class — holds the three per-geometry loader maps (`loaders`, `linesLoaders`, `gsplatLoaders`), the `failedLoaders` error-tracking map, plus registration / lookup / `recordFailure` / `clearFailure` / `disposeAll` helpers.                                                                                                                                                                                                                                  |
| `loader-factory.ts`          | Four constructor helpers — `createPointsLoader`, `createLinesLoader`, `createGSplatsLoader`, `createProgressiveGSplatsLoader` — each resolving the node's zarr `Location` and instantiating the matching `*SpatialIndexLoader`. The progressive variant opens each `lod_i/` subgroup and wraps N per-LOD loaders in a `GSplatsProgressiveLoader`, propagating the parent's pre-composed effective attrs into every LOD's synthetic SceneNode.                                  |
| `view-state-queue.ts`        | `ViewStateQueue` — owns `_pendingViewState` (single Partial<ViewState> queued while the in-progress lock is held; drained on a microtask) and `_prevPerNodeViewState` (per-node ViewState snapshots used as the predictive-prefetch baseline). Exposes `setPending` / `takePending` / `hasPending` / `drain` / `dispatchPrefetch` / `forgetPath` / `clearPrev`.                                                                                                                |
| `predicted-view-state.ts`    | `predictNextViewState(prev, current)` extrapolates the next-frame view state from the per-dimension delta (display axes never extrapolated; NaN-protected). `dispatchPredictivePrefetch(prev, current, loaders)` calls `prefetchChunks(predicted)` on every loader implementing the `PrefetchableLoader` structural type. Errors are swallowed — prefetch is best-effort cache warming.                                                                                       |
| `derive-node-view-state.ts`  | `deriveNodeViewState(path, attrs, baseViewState, sceneGraph, opts)` — single source of truth for per-node query derivation. Returns `{ skip: 'extend_to_all' }` when the node's `extend_to_all` dims fully cover all non-displayed dims, otherwise the base view state folded with the partial-extend tolerance override (Points + GSplats) and the inverse `nd_transform` for the node's path. Same helper backs initial load, update, and retry — query regions stay aligned. |
| `extend-tolerance.ts`        | Pure helpers feeding `derive-node-view-state.ts`: the `EXTEND_TO_ALL_TOLERANCE = 1e10` sentinel, `validateExtendDims` (actionable error listing valid dim names), `getOrComputeExtendedTolerance` (cached extended-tolerance arrays keyed by sorted dim-name set), and the `isSceneDimensions` type guard.                                                                                                                                                                     |
| `effective-attrs.ts`         | `applyEffectiveAttrs(sceneGraph, node)` — returns a node-attrs record with `opacity` / `gamma` / `intensity` / `offset` / `blending_mode` replaced by the values from `attrs-composer.getEffectiveAttrs` (root → leaf composition). Falls back to the raw attrs when no scene graph is available.                                                                                                                                                                              |
| `run-loader-updates.ts`      | `runLoaderUpdates(loaders, loaderType, updateFn, ctx)` — shared scaffolding for the per-geometry update loops in `updateView`. Wraps each loader call in a profiler session, records failures into `failedLoaders` with a retry count, drops the predictive-prefetch baseline for failed paths, and returns the staged commits for the atomic commit phase.                                                                                                                  |
| `retry.ts`                   | `retryFailedLoaderUnlocked(path, ctx)` and `retryAllFailedLoadersUnlocked(paths, ctx)` — lock-free recovery for paths recorded in `failedLoaders`. Re-runs the same `deriveNodeViewState` → `loader.updateView` → process+commit chain the main update path uses, so retry can never load a different query region than a fresh update would. Guards against scene-object disappearance between failure and retry.                                                            |
| `scene-graph-converter.ts`   | `convertToSceneGraphNode(node)` — pure recursive conversion from the loader's `SceneNode` to the monitor UI's `SceneGraphNode`, with type whitelisting (`scene` / `group` / `points` / `lines` / `gsplats` / `mesh`), display-name derivation (`/` → `"Scene"`), and per-type stats (`pointCount`, `segmentCount` + `vertexCount`, `splatCount`).                                                                                                                              |
| `url-normalization.ts`       | `normalizeURL(url, windowOrigin)` — pure: absolute URLs (case-insensitive `https?://`) get a trailing slash; relative paths are prepended with the supplied origin. Origin passed in (not read from `window`) so the helper is testable without a DOM.                                                                                                                                                                                                                          |
| `visible-counts.ts`          | `updateVisibleCountsInMonitor(rootGroup, monitor)` — traverses the root group, sums per-mesh `visibleSegmentCount` / `visibleSplatCount` userData (lines + gsplats), and pushes the totals to the monitor. Called once per update cycle after the lines/gsplats commits so the HUD shows post-clipping visible counts.                                                                                                                                                          |

## Invariants

- **Single source of truth for per-node view state.** Initial load,
  update, and retry all route through `derive-node-view-state.ts`. The
  `extend_to_all` skip semantics, partial-extend tolerance override,
  and `nd_transform` inverse-query all live in that one helper —
  diverging means a retry can succeed against a different region than
  the update that failed.
- **Async process / synchronous commit split.** `process/` runs the nD
  → 3D projection (worker-preferred, main-thread fallback) and returns
  `Staged*Commit` payloads without touching geometry; `commit/` writes
  those payloads into THREE.js buffers atomically. The two halves are
  routed by `run-loader-updates.ts` upstream and `update-view/atomic-commit.ts`
  downstream so every participating mesh updates in the same rendered
  frame.
- **Lock-free retry.** `retry.ts` helpers never touch
  `_updateInProgress`. The orchestrator holds the lock once around the
  entire retry call so retry-from-inside-retry cannot deadlock and a
  slider event mid-retry queues as `_pendingViewState` instead of
  racing into a concurrent `updateView`.
- **Dataset abort is wired into the worker pool.** `load-scene.ts`
  aborts the previous dataset's signal **before** `dispose()` runs so
  in-flight `runWithTimeout` callers settle immediately; `dispose.ts`
  clears the pool's signal so the next loader's workers see no signal
  until `load-scene.ts` installs the new one.
- **Best-effort prefetch.** Predictive prefetch errors are swallowed
  (`predicted-view-state.ts`, `view-state-queue.ts`) — a failed
  prefetch must never block the next demand fetch or surface noise in
  the console.

## Subpackages

- [commit](./commit/README.md) — synchronous GPU-commit step. One
  module per geometry (Points/Lines/GSplats) plus a shared renderer-
  cache eviction helper. Runs atomically at the tail of `updateView`.
- [nodes](./nodes/README.md) — initial-load helpers that walk the
  zarr scene graph and attach matching THREE.js objects, with empty
  placeholders before the first fetch and per-leaf error isolation
  (`LoaderError` + `loadLeafNode`).
- [process](./process/README.md) — async nD→3D projection step for
  Lines and GSplats (worker preferred, main-thread fallback). Returns
  `Staged*Commit` payloads without mutating geometry.
- [update-view](./update-view/README.md) — orchestration helpers
  extracted from `scene-loader.ts::updateView`: per-type ctx
  construction, atomic Stage 2 commit, and the `finally`-phase
  dispatcher that chooses between rAF re-entry, GSplats refinement,
  and lock release.

## See also

- `../scene-loader.ts` — the orchestrator class that composes every
  module in this folder.
- `../scene-loader-manager.ts` — singleton manager for `SceneLoader`
  instances; injects the monitor factory.
- `../scene-loader-monitor-port.ts` — the `SceneLoaderMonitorPort`
  interface consumed by `monitor-wiring.ts` and `dispose.ts`.
- `../attrs-composer.ts` — `getEffectiveAttrs` used by
  `effective-attrs.ts`.
- `../transforms/nd-transform.ts` — `computeWorldNdTransform` /
  `invertNdTransformForQuery` used by `derive-node-view-state.ts`.
- `../view-state-manager.ts` — dimension validation invoked by
  `nodes/initialize-scene-dimensions.ts`.
- `../../workers/worker-pool.ts` — `setAbortSignal` consumed by
  `load-scene.ts` and `dispose.ts`.
