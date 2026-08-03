# View-state derivation, prediction, and queueing

Everything the SceneLoader does with a view-state value before it
reaches a loader: deriving the per-node query (with `extend_to_all`

- partial-extend + inverse-`nd_transform` folding), predicting the
  next frame for prefetch warming, serializing concurrent
  `updateView` calls through a single-slot queue, and composing the
  hierarchical render attributes the geometry loaders read off each
  node's `attrs`.

## Files

| File                        | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `derive-node-view-state.ts` | `deriveNodeViewState(path, attrs, baseViewState, sceneGraph, opts)` — single source of truth for per-node query derivation. Returns a single `{ skip: false; viewState }` shape: the base view state folded with the partial-extend tolerance override (Points + GSplats) and the inverse `nd_transform` for the node's path. When `extend_to_all` covers ALL non-displayed dims the node is derived as a slice-INVARIANT query (the `1e10` extend-to-all tolerance sentinel on every extended dim + its `slicePosition` pinned to `0`), so per-sweep re-queries hit the loader's same-view no-op. Same helper backs initial load, update, and retry — query regions stay aligned. |
| `extend-tolerance.ts`       | Pure helpers feeding `derive-node-view-state.ts`: the `EXTEND_TO_ALL_TOLERANCE = 1e10` sentinel, `validateExtendDims` (actionable error listing valid dim names), `getOrComputeExtendedTolerance` (cached extended-tolerance arrays keyed by sorted dim-name set), and the `isSceneDimensions` type guard.                                                                                                                                                                      |
| `effective-attrs.ts`        | `applyEffectiveAttrs(sceneGraph, node)` — returns a node-attrs record with `opacity` / `gamma` / `intensity` / `offset` / `blending_mode` replaced by the values from `attrs-composer.getEffectiveAttrs` (root → leaf composition). Falls back to the raw attrs when no scene graph is available.                                                                                                                                                                               |
| `predicted-view-state.ts`   | `predictNextViewState(prev, current)` extrapolates the next-frame view state from the per-dimension delta (display axes never extrapolated; NaN-protected). `dispatchPredictivePrefetch(prev, current, loaders)` calls `prefetchChunks(predicted)` on every loader implementing the `PrefetchableLoader` structural type. Errors are swallowed — prefetch is best-effort cache warming.                                                                                         |
| `view-state-queue.ts`       | `ViewStateQueue` — owns `_pendingViewState` (single Partial<ViewState> queued while the in-progress lock is held; drained on a microtask) and `_prevPerNodeViewState` (per-node ViewState snapshots used as the predictive-prefetch baseline). Exposes `setPending` / `takePending` / `hasPending` / `drain` / `dispatchPrefetch` / `forgetPath` / `clearPrev`.                                                                                                                 |

## Consumers

- `../lifecycle/load-scene.ts` resets the predictive-prefetch
  baseline (the queue's `clearPrev`) on dataset switch, so the
  first `updateView` doesn't extrapolate from the prior dataset's
  slice position. The initial per-leaf loads themselves run through
  `../nodes/*` below.
- `../lifecycle/retry.ts` re-runs `derive-node-view-state.ts` so
  retry uses the same query region the failed update used.
- `../nodes/*` consume `derive-node-view-state.ts` to issue
  the first per-leaf load and `extend-tolerance.ts::isSceneDimensions`
  to validate `scene_dimensions` blobs.
- `../update-view/*` consume `derive-node-view-state.ts` to assemble
  per-type update contexts and the queue for re-entry decisions.
- `../loaders/run-loader-updates.ts` clears the queue's per-node
  baseline (`forgetPath`) when a path's update fails, so the next
  success re-baselines instead of extrapolating across the error.
- `../process/data-processor-lines.ts` uses
  `extend-tolerance.ts::EXTEND_TO_ALL_TOLERANCE` to flag covered dims.
- `../../scene-loader.ts` (parent orchestrator) owns the
  `ViewStateQueue` instance and forwards `applyEffectiveAttrs` /
  `deriveNodeViewState` through its ctx builders.
- `data/{points,lines,gsplats}/handler.ts` call
  `ViewStateQueue.dispatchPrefetch(path, derivedViewState, loader)`
  (carried through their per-step ctx) to warm the cache from each
  node's derived view-state; `data/gsplats/lod-refinement.ts`
  threads the queue through as a type on its ctx.
