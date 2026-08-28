# Lifecycle: load, dispose, retry

The three phase-transition orchestrator bodies — initial load,
teardown, and mid-life recovery — plus the load-only URL helper. Each
file owns one entry point of the SceneLoader's lifetime; the
orchestrator class in `../../scene-loader.ts` is a thin wrapper that
builds the ctx and calls into here.

## Files

| File                   | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `load-scene.ts`        | Initial-load orchestrator body. Aborts in-flight worker tasks, disposes the previous loader, opens the L0/L1/L2 cache stack, opens the zarr root, builds the empty root `THREE.Group`, initializes scene dimensions, persists `viewer_config` + `position_bounds`, builds the scene graph, recursively loads every leaf, loads overlay configs, wires the post-load monitor providers, and kicks GSplats LOD refinement when multi-LOD loaders still have higher LODs pending. |
| `dispose.ts`           | `disposeSceneLoader(ctx)` — aborts the dataset signal, disposes every registered loader, the GPU buffer pool, the caching store (awaited so L2 flushes drain), the L0 cache, the predictive-prefetch baseline, dataset-scoped custom colormap LUTs, and the monitor's loader-bound closures. Worker pool itself is NOT terminated — `core/app.ts` owns that.                                                                                                                   |
| `retry.ts`             | `retryFailedLoaderUnlocked(path, ctx)` and `retryAllFailedLoadersUnlocked(paths, ctx)` — lock-free recovery for paths recorded in `failedLoaders`. Re-runs the same `deriveNodeViewState` → `loader.updateView` → process+commit chain the main update path uses, so retry can never load a different query region than a fresh update would. Lines share eager admission; retry-all is eight-wide. Guards against scene-object disappearance between failure and retry.       |
| `url-normalization.ts` | `normalizeURL(url, windowOrigin)` — pure: absolute URLs (case-insensitive `https?://`) get a trailing slash; relative paths are prepended with the supplied origin. Origin passed in (not read from `window`) so the helper is testable without a DOM. Nests here because `load-scene.ts` is its only consumer inside this folder.                                                                                                                                             |

## Invariants

- **Dataset abort precedes dispose.** `load-scene.ts` aborts the
  previous dataset's signal before `dispose()` runs so in-flight
  `runWithTimeout` callers settle immediately; `dispose.ts` then
  clears the worker pool's signal so the next loader's tasks see a
  clean slate.
- **Retry is lock-free.** `retry.ts` never touches
  `_updateInProgress` — the orchestrator holds the lock once around
  the whole retry call so a slider event mid-retry queues as
  `_pendingViewState` instead of racing into a concurrent
  `updateView`.
- **Line retry admission is session-scoped.** Single and batched registered line
  retries acquire the same working-set gate as the eager scene walk; retry-all
  adds only the shared eight-slot worker cap.

## Consumers

- `../../scene-loader.ts` (parent orchestrator) — the only caller
  for all four files. `normalizeURL` is also re-tested at the
  parent's class method (`SceneLoader.normalizeURL`).
