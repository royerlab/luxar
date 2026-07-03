# Scene-graph node construction

Initial-load helpers that walk a zarr scene graph and attach the
matching THREE.js objects to the scene. One file per first-class
geometry kind (Points, Lines, GSplats) wraps the `loader-factory.ts`
construction + placeholder + initial-fetch + commit dance. Two more
files cover the specialized-`Group` kinds (`load-lod-group-node.ts`
for `kind="lod"` view-driven LOD selection, `load-partition-group-node.ts`
for `kind="partition"` compile-time spatial decomposition). Shared
infrastructure handles store enumeration, scene-graph building,
dimension initialization, monitor wiring, error dispatch, and the
recursive walk.

This folder is the **initial-load** path. The per-update path lives
in the sibling `../../points/handler.ts`, `../../lines/handler.ts`,
and `../../gsplats/handler.ts` modules. Together the two paths cover
every way a leaf node gets data: first construction, slice updates,
and retry-after-failure.

## Files

| File                             | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build-ctx.ts`                   | `NodeBuildCtx` interface — the per-call snapshot the orchestrator passes down to every leaf loader (viewState, factoryDeps, registry, nodeFactory, optional `lodGroupRegistry`, plus callback closures for attrs composition, view-state derivation, monitor wiring, dataset-liveness check `isDatasetLive`, lazy-gsplats release `releaseLazyGSplats`, and per-type commit). Never carries `this`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `build-scene-graph.ts`           | `buildSceneGraph(rootLoc, rootAttrs, store)` — enumerates the store, walks groups breadth-first by path depth, opens each group's attrs, skips the `/overlays` subtree, marks every Points/Lines/GSplats node's subtree internal (so its `additive_<i>/` LOD subgroups don't appear as spurious scene-graph children — the leaf loader walks them itself), and eagerly fetches the sibling `colormap_lut` zarr array for any node declaring `colormap: 'custom'`. Finally sorts every sibling list by the `child_index` attr (insertion order stamped by the Python `Node`) so the scene graph — and the layers panel — follow napari-style add order rather than the store's alphabetical enumeration; siblings without `child_index` keep their relative enumeration order. Returns the `SceneNode` tree.                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `enumerate-store.ts`             | `enumerateStore(store)` — thin wrapper around the consolidated-metadata `contents()` method with a single-root fallback for stores that don't expose it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `initialize-scene-dimensions.ts` | `initializeSceneDimensions(sceneDims)` — validates a scene-level `scene_dimensions` blob through `ViewStateManager`, logs the validation, and returns a fresh `ViewState`. Returns `null` on invalid input so the caller can leave the previous ViewState untouched.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `load-scene-nodes.ts`            | `loadSceneNodes(node, parentThree, parentLoc, ctx)` — recursive walk. For each leaf, dispatches to the matching per-type loader through `loadLeafNode`. For each group, creates a `THREE.Group`, applies its 4x4 transform via `NodeFactory.applyTransform`, and recurses.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `load-points-node.ts`            | `loadPointsNode` — initial-load wrapper for a single Points leaf. Branches on `n_additive_sublods`: multi-additive nodes (`> 1`) use a progressive loader (with composed effective attrs so sub-LOD synthetic nodes inherit opacity/intensity); single-LOD nodes use the standard loader. Registers it, attaches an empty placeholder, derives the per-node view state (with `applyPartialExtendTolerance: true`), fetches points, and commits through `updatePointsGeometry`. Throws `LoaderError` on failure. Exported as two halves (mirroring `load-gsplats-node.ts`) so `load-lod-group-node.ts` can defer a Points child of a `kind=lod` group: `loadPointsNodeCheap` (build loader + attach placeholder, no fetch, **no registry registration**) and `loadPointsNodeExpensive` (derive + fetch + commit, with an `isDatasetLive()` gate). The combined `loadPointsNode` runs both and registers immediately.                                                                          |
| `load-lines-node.ts`             | `loadLinesNode` — mirror of `loadPointsNode` for Lines (same `n_additive_sublods` progressive branch). Differs in one place: the data fetch uses `applyPartialExtendTolerance: false` because segment bounds already encode non-displayed spatial extent (the override would double-apply during clipping). Commits via `processLinesData` + `commitLinesGeometry`. Exported as two halves (mirroring points/gsplats) so `load-lod-group-node.ts` can defer a Lines lod child: `loadLinesNodeCheap` (build loader + attach placeholder, no fetch, **no registry registration**) and `loadLinesNodeExpensive` (derive + fetch + project + commit, with an `isDatasetLive()` gate). The combined `loadLinesNode` runs both and registers immediately.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `load-gsplats-node.ts`           | `loadGSplatsNode` — mirror of `loadPointsNode` for GSplats. Branches on `n_additive_sublods`: multi-additive nodes (`> 1`) use the progressive loader (with parent's composed effective attrs so sub-LOD synthetic nodes inherit opacity/intensity); single-LOD nodes use the standard loader. Commits via `processGSplatsData` + `commitGSplatsGeometry`. Exported as two halves so `load-lod-group-node.ts` can defer the costly part: `loadGSplatsNodeCheap` (build loader + attach placeholder, no fetch, **no registry registration**) and `loadGSplatsNodeExpensive` (fetch + process + commit, with an `isDatasetLive()` gate so a deferred load can't write into a switched/disposed scene). The combined `loadGSplatsNode` runs both and registers immediately.                                                                                                                                                                                                   |
| `load-lod-group-node.ts`         | `loadLodGroupNode` — initial-load for a kind=`lod` `Group` (also exports the `LoadSceneChildren` recursion-handle type). Creates a `THREE.Group`, applies its transform, then registers a `LODGroupEntry` (children + their `coverage_fraction` / `position_bounds` + `default_level`) with the `LODGroupRegistry` so the per-frame selector picks which child renders. Legacy (pre-v3.2) stores whose children carry `min_pixel_size` instead of `coverage_fraction` are auto-adapted (thresholds normalized by the finest value) with a warning naming `luxar gsplat migrate-format`; children with neither attr get a loud, actionable error instead of a silent 0-default. **Lazy loading**: only the default level's geometry is fetched eagerly; every other **gsplats / points / lines** child is cheap-attached (placeholder + loader via `loadGSplatsNodeCheap` / `loadPointsNodeCheap` / `loadLinesNodeCheap`, no fetch) with an `ensureLoaded` thunk (runs the matching `*Expensive` + registers via `registerGSplatsLoader` / `registerPointsLoader` / `registerLinesLoader` on first selection) and a `release` thunk (returns GPU geometry to the evictable pool via `releaseLazyGSplats` / `releaseLazyPoints` / `releaseLazyLines`, resets readiness) so the LOD registry's byte-budget pass can evict off-screen levels. The shared `attachLazyChild` helper holds the ready/failed/loading state machine for all three types. Deferring the points/lines child matters for the points-/lines-substitutive ladders, whose finest child is the full cloud / line set — otherwise it would be fetched eagerly. Other child types and the no-registry fallback load fully. Children are hidden immediately on attach so the sequential-load loop can't flash all levels at once; `register()` re-enables exactly one synchronously at the end. |
| `load-partition-group-node.ts`   | `loadPartitionGroupNode` — initial-load for a kind=`partition` `Group`. Creates a `THREE.Group`, marks `userData.kind = 'partition'` (so picking resolves an inner `part_<i>` hit back to the wrapper path), applies its transform, recurses children via the injected `loadChildren` handle. All children stay visible — no per-frame selector — and THREE's per-mesh frustum culling handles per-part culling. The wrapper's `position_bounds` (union over children) sits on its on-disk attrs already, so picking / scene-bounds-cache treat the layer as one logical entity.                                                                                                                                                                                                                                                                                                                                                                                           |
| `connect-loader-to-monitor.ts`   | `connectLoaderToMonitor(path, loader, monitor)` — duck-type-guarded wiring. Points loaders always implement the full `LoaderMonitor` surface; Lines/GSplats expose it optionally. Checks for the four-method shape (`addEventListener`/`removeEventListener`/`getMetrics`/`getActiveQueries`) before wiring. Null monitor short-circuits.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `load-leaf-error-dispatch.ts`    | `LoaderError` class + `classifyLoaderError(error)` heuristic + `loadLeafNode(load, path)` wrapper. Catches `LoaderError` thrown by a leaf, logs+toasts by kind (Network warns, Decode/Validation/Unexpected error-logs and toasts), returns `null` so the failing leaf doesn't sink the rest of the scene. Re-throws non-LoaderError exceptions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

## Invariants

- **Three-geometry symmetry.** `load-points-node.ts`,
  `load-lines-node.ts`, and `load-gsplats-node.ts` follow the same
  shape: `createXLoader` helper → `registry.registerXLoader` →
  `nodeFactory.createEmptyXNode` placeholder → `parentThree.add` →
  `deriveNodeViewState` → `loader.loadX` → process+commit through
  the same helpers the update/retry paths use → `recordFailure` +
  `throw new LoaderError` on error. Filename matches the single
  exported function.
- **Placeholder before fetch.** Every leaf attaches an empty
  placeholder mesh _before_ the initial data fetch. A transient fetch
  failure then leaves a findable, retryable THREE node in the scene
  rather than a hole — `retryFailedLoader(path)` can target it, and
  the placeholder's `userData.attrs` is the source of truth the retry
  view state is derived from. The same node is later populated in
  place by the matching `../commit/commit-*-geometry.ts` helper, so
  the 0-points → N-points transition uses the same code path as every
  subsequent update.
- **Single source of truth for view state.** Every leaf calls
  `ctx.deriveNodeViewState(path, attrs, opts)` — the same helper the
  main update loop and `retry.ts` use. Initial/update/retry can never
  silently load different query regions. Points and GSplats pass
  `applyPartialExtendTolerance: true`; Lines passes `false` (segment
  bounds already encode the equivalent extent).
- **Initial load never skips.** `deriveNodeViewState` may return
  `{ skip: 'extend_to_all' }` to mean "leave the existing node
  alone" — but on initial load we always want to construct the
  THREE node so future slice changes can populate it. Each leaf
  handles the skip return by falling back to the orchestrator's
  base `ctx.viewState`.
- **ViewState snapshot is captured by value.** `NodeBuildCtx` carries
  a snapshot of the orchestrator's viewState at the time
  `loadSceneNodes` is invoked, not a reference. A concurrent
  `updateView` mutating the orchestrator's field cannot corrupt an
  in-flight initial-load query.
- **Partial-scene resilience.** Each leaf is wrapped in
  `loadLeafNode`, which catches `LoaderError` and returns `null` so
  one failing leaf doesn't take down the whole scene. Non-LoaderError
  exceptions still propagate.
- **Overlays are not 3D scene nodes.** `buildSceneGraph` skips any
  path under `/overlays` because screen-space overlays live in a
  parallel tree built by a different code path.

## See also

- `../loaders/loader-factory.ts` — `create{Points,Lines,GSplats}Loader`
  and `createProgressive{Points,Lines,GSplats}Loader` helpers
  that the leaf loaders call. The orchestrator passes the live
  `LoaderFactoryDeps` snapshot through `NodeBuildCtx`.
- `../loaders/loader-registry.ts` — `LoaderRegistry.registerXLoader` and
  `recordFailure` invoked by every leaf.
- `../view-state/derive-node-view-state.ts` — `deriveNodeViewState` and the
  `DerivedNodeViewState` / `DeriveOpts` shapes consumed via
  `NodeBuildCtx`.
- `../process/data-processor-{lines,gsplats}.ts` — async projection
  step that produces the `StagedLinesCommit` / `StagedGSplatsCommit`
  bundles consumed by commit.
- `../commit/commit-{points,lines,gsplats}-geometry.ts` — synchronous
  GPU-commit helpers invoked at the tail of each initial-load.
- `../../points/handler.ts`, `../../lines/handler.ts`,
  `../../gsplats/handler.ts` — the matching **update**-path helpers;
  this folder owns the **initial-load** path.
- `../../../rendering/node-factory.ts` — `createEmptyXNode` and
  `applyTransform` used to build the placeholder meshes and group
  transforms.
- `../../scene-loader-monitor-port.ts` — `SceneLoaderMonitorPort`
  contract used by `connect-loader-to-monitor.ts`.
- `../../view-state-manager.ts` — `ViewStateManager` used by
  `initialize-scene-dimensions.ts` for dimension validation.
