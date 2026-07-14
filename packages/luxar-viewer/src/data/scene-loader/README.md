# SceneLoader extracted helpers

Modules composed by `data/scene-loader.ts`. The orchestrator class
stays in the parent `data/` folder; every helper that implements one
of its lifecycle steps lives here. Nothing in this folder is
re-exported through `data/index.ts` — the parent orchestrator is the
only consumer of the top-level files of each subfolder, with a small
number of exceptions noted in each cluster's README (the
`progressive/` loop is driven from the per-geometry `lod-refinement.ts`
modules, and `lod-load-stats.ts` is read by the debug interface).

The SceneLoader has three external entry points:

- **`loadScene(url)`** — initial load (cache stack, zarr open, scene
  graph build, recursive per-node load, post-load monitor wiring,
  GSplats LOD kick).
- **`updateView(viewState)`** — slider / animation / keyboard nav. Runs
  the per-type async process step then the synchronous atomic commit,
  serialized through a view-state queue with predictive prefetch. Each
  cycle gets a fresh per-update `AbortController` (distinct from the
  per-dataset one): when a newer view-state supersedes the in-flight
  one, its signal is aborted so the superseded load's chunk reads/worker
  decodes bail (an `AbortError`/`WorkerAbortError` is treated as
  superseded, not a failure) and its atomic commit is skipped — the
  winning view-state commits the correct frame. The signal flows from
  `updateView` through the handler ctx and `loader.updateView` to two read
  surfaces: the `wrapWithCache` L0 chokepoint (covers warm-cache hits), and
  `RangeLoader` — the single owner of every demand chunk read for all three
  geometry types, which threads the signal into both `get(array, …, { signal })`
  (cold / L0-disabled / `array_ref`-target reads, regardless of wrapping) and
  the worker decodes. The atomic commit-skip is the final, always-present
  correctness backstop.
- **`dispose()`** — release dataset-scoped resources before the next
  `loadScene`.

## Layout

This folder is split into eleven subpackages — thematic clusters plus the
pre-existing per-step folders — named for their concern. Two `.ts`
files sit directly under `scene-loader/`:

- **`lod-load-stats.ts`** — debug-only per-stage timing accumulator for
  lazy LOD level loads (fetch/decode, projection+pack, GPU commit,
  release). These loads run outside any `updateView` cycle, so the
  `UpdateProfiler` / data-loading-monitor never sees them; this fills
  the gap. Disabled by default (zero cost), enabled under `?debug` by
  `installDebugInterface`, which also exposes
  `window.__luxarDebug.getLodLoadStats()` / `resetLodLoadStats()`.

The subfolders:

| Subpackage                             | Concern                                                                                                                                                                                                                                                                                                         |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [cache](./cache/README.md)             | L0/L1/L2 cache stack setup + the get/clear/list surface used by `__luxarDebug.cache`.                                                                                                                                                                                                                           |
| [loaders](./loaders/README.md)         | Per-geometry loader constructors, the `LoaderRegistry` lifecycle bookkeeping, and the shared update-loop scaffolding.                                                                                                                                                                                           |
| [view-state](./view-state/README.md)   | Single source of truth for per-node view-state derivation, predictive prefetch, the serialized update queue, and `extend_to_all` tolerance helpers.                                                                                                                                                             |
| [lifecycle](./lifecycle/README.md)     | The three phase-transition orchestrator bodies — `loadScene`, `dispose`, `retry` — and the load-only `normalizeURL` helper.                                                                                                                                                                                     |
| [monitor](./monitor/README.md)         | Post-load data-monitor provider wiring, `SceneNode → SceneGraphNode` UI conversion, and the per-update visible-count tally.                                                                                                                                                                                     |
| [commit](./commit/README.md)           | Synchronous GPU-commit step. One module per geometry plus the renderer-cache eviction helper. Runs atomically at the tail of `updateView`.                                                                                                                                                                      |
| [nodes](./nodes/README.md)             | Initial-load helpers that walk the zarr scene graph and attach matching THREE.js objects, with empty placeholders before the first fetch and per-leaf error isolation.                                                                                                                                          |
| [process](./process/README.md)         | Async nD→3D projection step for Lines and GSplats (worker preferred, main-thread fallback). Returns `Staged*Commit` payloads without mutating geometry.                                                                                                                                                         |
| [prefetch](./prefetch/README.md)       | `SlicePrefetcher` — background t+1 slice prefetch during dimension playback: shadow loaders warm the NEXT tick's decoded ladder into the shared SliceCache (S-cache) so the real tick's restore hits instantly. Driven via `SceneLoader.prefetchSlice`; aborted at the top of every foreground `updateView`.    |
| [update-view](./update-view/README.md) | Orchestration helpers extracted from `scene-loader.ts::updateView`: per-type ctx construction, atomic Stage 2 commit, and the `finally`-phase dispatcher that chooses between rAF re-entry, GSplats refinement, and lock release.                                                                               |
| progressive (`refinement.ts`)          | The generic progressive-LOD refinement loop (`runProgressiveRefinement`) shared by all three leaf types via their `data/{points,lines,gsplats}/lod-refinement.ts` wrappers: per-frame rAF yield, view-state-queue cancellation handoff, per-loader processing, and lock release when no LODs remain. No README. |

## Cross-cluster invariants

- **Single source of truth for per-node view state.** Initial load,
  update, and retry all route through
  `view-state/derive-node-view-state.ts`. The `extend_to_all` skip
  semantics, partial-extend tolerance override, and `nd_transform`
  inverse-query all live in that one helper — diverging means a retry
  can succeed against a different region than the update that failed.
- **Async process / synchronous commit split.** `process/` runs the nD
  → 3D projection (worker-preferred, main-thread fallback) and returns
  `Staged*Commit` payloads without touching geometry; `commit/` writes
  those payloads into THREE.js buffers atomically. The two halves are
  routed by `loaders/run-loader-updates.ts` upstream and
  `update-view/atomic-commit.ts` downstream so every participating
  mesh updates in the same rendered frame.
- **Lock-free retry.** `lifecycle/retry.ts` helpers never touch
  `_updateInProgress`. The orchestrator holds the lock once around the
  entire retry call so retry-from-inside-retry cannot deadlock and a
  slider event mid-retry queues as `_pendingViewState` instead of
  racing into a concurrent `updateView`.
- **Dataset abort is wired into the worker pool.** `lifecycle/load-scene.ts`
  aborts the previous dataset's signal **before** `dispose()` runs so
  in-flight `runWithTimeout` callers settle immediately;
  `lifecycle/dispose.ts` clears the pool's signal so the next loader's
  workers see no signal until `lifecycle/load-scene.ts` installs the
  new one.
- **Best-effort prefetch.** Predictive prefetch errors are swallowed
  (`view-state/predicted-view-state.ts`, `view-state/view-state-queue.ts`)
  — a failed prefetch must never block the next demand fetch or
  surface noise in the console.

## See also

- `../scene-loader.ts` — the orchestrator class that composes every
  module in this folder.
- `../scene-loader-manager.ts` — singleton manager for `SceneLoader`
  instances; injects the monitor factory.
- `../scene-loader-monitor-port.ts` — the `SceneLoaderMonitorPort`
  interface consumed by `monitor/monitor-wiring.ts` and
  `lifecycle/dispose.ts`.
- `../attrs-composer.ts` — `getEffectiveAttrs` used by
  `view-state/effective-attrs.ts`.
- `../transforms/nd-transform.ts` — `computeWorldNdTransform` /
  `invertNdTransformForQuery` used by
  `view-state/derive-node-view-state.ts`.
- `../view-state-manager.ts` — dimension validation invoked by
  `nodes/initialize-scene-dimensions.ts`.
- `../../workers/worker-pool.ts` — `setAbortSignal` consumed by
  `lifecycle/load-scene.ts` and `lifecycle/dispose.ts`.
