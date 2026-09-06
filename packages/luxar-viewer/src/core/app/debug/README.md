# Debug Interface

`window.__luxarDebug` wiring for the Luxar viewer — the runtime hook that AI
debug drivers, Playwright specs, embedders, and the data-monitor's Cache tab
all reach into.

## Overview

This folder owns the `__luxarDebug` global. The bootstrap entry point
(`core/bootstrap.ts`) seeds a minimal stub (`app`, `consoleInterceptor`,
`version`) so the global exists from the first JS tick. Once `LuxarApp.init()`
finishes building the subsystem graph it calls `installDebugInterface(ports)`
to extend that stub with live runtime components, helper functions, and the
synthetic-scene injector.

Everything here is private to `LuxarApp` — only `app.ts` imports these files
(`LuxarApp.setupDebugInterface()` calls `installDebugInterface`, and a private
`openCacheStatsView()` wrapper delegates to `cache-stats-view.ts`). The one
exception is `capture-readiness.ts`, which is deliberately dependency-free so
the out-of-bundle capture tool (`tools/capture-hires.ts`) can consume the
`getState()` snapshot from Node.

## File Structure

```
debug/
├── debug-interface.ts       # installDebugInterface() — populates window.__luxarDebug
├── debug-state.ts           # computeDebugState() + computeDrawOrder() — pure scene walks
├── capture-readiness.ts     # summarizeCaptureReadiness() — "did anything load?" verdict
├── debug-cache-helpers.ts   # buildDebugCacheHelpers() — __luxarDebug.cache.* wrappers
└── cache-stats-view.ts      # openCacheStatsView() — pops the data-monitor Cache tab
```

## Files

### `debug-interface.ts`

Entry point. Exports `installDebugInterface(ports: InstallDebugInterfacePorts)`
and the `InstallDebugInterfacePorts` interface. No-ops when `ports.debug` is
false, so production bundles pay only the import cost.

The port object carries opaque handles for every subsystem the debug surface
touches: `app`, `sceneManager`, `animationController`, `inputHandler`,
`renderingControls`, `recordingPanel`, plus accessor closures
`getPickingSystem` / `getOverlayManager` / `isInitialized`. Accessors (rather
than direct references) are used for systems that are torn down and rebuilt
across dataset reloads.

The function preserves whatever bootstrap already seeded (`app`,
`consoleInterceptor`, `version`) by spreading the existing object, then layers
on the runtime fields. When `LuxarApp` is instantiated outside the standalone
entry point — e.g. embeds or unit tests — bootstrap hasn't run, so the helper
falls back to a fresh stub.

On entry it calls `setLodLoadStatsEnabled(true)` (from
`data/scene-loader/lod-load-stats`) so per-stage timing for lazy LOD level
loads and additive per-level ladder loads is captured under `?debug` only.
Those lazy `ensureLoaded` loads run outside any `updateView` cycle, so the
`UpdateProfiler` never sees them — this fills the gap. The captured stats are
reachable as `__luxarDebug.getLodLoadStats()` (snapshot) and
`resetLodLoadStats()`.

`injectSyntheticScene`'s body is wrapped in try/catch: on failure it logs via
`log.error` and surfaces the message through the user-facing error overlay
(`showError`), then re-throws so callers that `await` still see the rejection.
This prevents a broken dynamic import / synthetic-scene builder from becoming a
silent unhandled promise rejection.

After populating the global, it emits a series of `log.info(Modules.LUXAR, …)`
lines documenting the available commands; this is the in-console help text
users see after enabling `?debug`.

### `debug-state.ts`

Pure helper behind `__luxarDebug.getState()`. Exports
`computeDebugState(ctx: DebugStateContext): DebugState`, the input-surface
interface `DebugStateContext`, and the result-shape interfaces (`DebugState`,
`PointCloudInfo`, `GSplatMeshInfo`, `LineMeshInfo`, `MeshNodeInfo`,
`GPUPoolDebugStats`, `LODGroupDebugInfo`, `PartitionDebugInfo`). It also exports
`computeDrawOrder(scene): DrawOrderEntry[]` and its `DrawOrderEntry` shape, the
helper behind `__luxarDebug.getDrawOrder()`.

`computeDebugState` walks the scene graph once and tallies per-node detail for
all four geometry types by inspecting `userData.nodeType`. For the three
instanced-quad types — Points, Lines, GSplats — it uses
`InstancedBufferGeometry.instanceCount` as the source of truth, because pooled
attribute arrays are over-allocated and `drawRange` only covers the 6-index
base quad. Points additionally fall back to `userData.visiblePointCount`, then
to 0, when the geometry is not instanced or its `instanceCount` is not finite.
Mesh is not instanced, so its `meshNodes[]` entries count triangles from the
current `drawRange` (falling back to the index length, since `drawRange.count`
defaults to `Infinity`) — the draw range is precisely what the nD slice
compaction narrows, so the index length alone would report the whole surface
regardless of slice position — and read the live shader variant flags off the
material's `defines`.

Every per-node geometry entry also exposes additive-ladder state when its live
loader has more than one level: `loadedLODCount`, `totalLODCount`,
`lastAllResident`, and the commit-time `committedLadderComplete` /
`committedEnergyFraction` stamps. This is how an additive-only leaf advertises
that it is laddered; `lodGroups[]` remains reserved for substitutive
`kind=lod` containers.

Mesh reveal ladders do not use `SliceCache`, so they contribute additive timing
and per-node residency state but never appear in `ladderDepthHistogram`.

The same traversal also summarises specialized-group containers by their
`userData.kind`: `kind=lod` groups become `lodGroups[]` (level count + the
index of the visible child as `activeLevel`, `-1` when none), and
`kind=partition` groups become `partitions[]` (part count + visible-part
count).

For lines, the `hasColormap` flag is read structurally from
`material.defines.USE_COLORMAP` so the result is identical whether the line mesh
is running on the GLSL `ShaderMaterial` or the TSL `NodeMaterial` backend.

The returned `DebugState` carries `totalPoints`, `totalGSplats`, `totalLines`,
`totalTriangles`, `totalElements` (their sum), `totalDroppedElements`, the per-node arrays
(`pointClouds`, `gsplatMeshes`, `lineMeshes`, `meshNodes`), `lodGroups`, `partitions`, an
optional `gpuPool` byte-stats block, `dimensions`, a nested `camera`
(`{position, fov}`) plus flat `cameraPosition` / `cameraFov` mirrors kept for
back-compat, and the `isAnimating` / `initialized` / `isLoading` flags.

`isLoading` is true while a LOAD PASS is in flight on any registered scene
loader — an `updateView` sweep (fetch/decode/upload) up to its geometry commit,
a failed-loader retry (which takes the same lock), or a view-state that is
QUEUED behind either and has not begun loading yet. That third clause is why
subtracting the refinement drain below opens no hole: a nav arriving during a
refinement hold parks in the queue without touching the lock, so without it the
flag would read idle while the requested slice had not started. Three things are
outside that scope:

- the **initial `loadScene`**, which only touches the loader's lock at its very
  end (handing it to the post-load refinement kick). Wait on `initialized` for
  the first load. An in-page **dataset switch** is covered by neither flag:
  `initialized` stays true and the fresh loader is registered before its
  `loadScene` runs, so `isLoading` reads false throughout the switch's load.
- **lazy substitutive-LOD / deferred-partition `ensureLoaded` promotions**,
  which run outside any `updateView` cycle and surface as content-change
  notifications instead.
- the **progressive-LOD refinement drain**, which inherits the same lock after
  the current view has already committed. Excluded deliberately, so the flag
  reports first-commit latency rather than full-ladder latency — the same
  distinction `update-view/queue-next.ts` draws when it resolves its pass
  waiters at refinement entry. `SceneLoader.isUpdateInProgress()` keeps the
  broader "lock is held at all" meaning for the adaptive-DPR manager and
  `core/app/init/pipeline.ts`.

Eight helpers in `tests/e2e/helpers.ts` poll this flag to decide when a load has
settled — `waitForDataLoaded`, `waitForDimensionNavigation`,
`waitForSpatialQuery`, `waitForSpatialQueryOrThrow`,
`waitForNavigationComplete`, `waitForNavigationCompleteOrThrow`, and the
state-based fallbacks inside `waitForRenderStable` and `waitForNextRender` — as
do `tests/e2e/real-dataset-loading.spec.ts` and the two capture specs under
`tests/screenshots/`. So it must stay a real boolean: an absent field reads as
"not loading" (`!undefined` is `true`) and gates on nothing.

Dependencies arrive as parameters (`scene`, `camera`, `currentFov`,
`isAnimating`, `initialized`, `isLoading`, `dims`, optional `gpuPoolStats` and
`refinementResidency`), so the helper is callable from unit tests against real
`THREE.Points` / `THREE.Mesh` fixtures without bringing up the WebGL renderer.
The two optional providers are supplied in production from
`SceneLoaderManager` — `gpuPoolStats()` (projected onto the `GPUPoolDebugStats`
subset) and `refinementResidencyStop()` — and read INSIDE the getter, per
snapshot, for the same reason `isLoading` is. Omitting either leaves its field
absent, which must read as "unknown / never happened" and never as a
synthesised zero: `capture-readiness.ts` treats a PRESENT
`refinementResidency` record as the stop signal regardless of its contents.

### `capture-readiness.ts`

Pure verdict over a `getState()` snapshot: exports
`summarizeCaptureReadiness(state: Partial<DebugState> | null | undefined)` and
its `CaptureReadinessSummary` result shape. Answers the one question a
screenshot/capture driver asks — "does this scene graph carry drawable
elements?" — as `ok` plus all four per-type totals (`totalPoints`,
`totalGSplats`, `totalLines`, `totalTriangles`), a `totalElements`, and the four
per-node counts (`pointCloudCount`, `gsplatCount`, `lineCount`,
`meshNodeCount`). `ok` is true iff `totalElements > 0` and nothing made those
counts a property of the RUN rather than of the store: `totalDroppedElements
=== 0` (per-node renderer clamps), no `refinementResidency` stop (progressive
refinement declining scene-wide at the residency byte ceiling, #2508) and no
`gpuPool.byteBudgetEvictions` (the pool going over its VRAM byte budget and
shedding pooled geometry to get back under it — including levels the LOD
registry had demoted from active, though on its own it does not prove rendered
geometry was lost, and it stays 0 when active bytes alone exceed the budget).
The last two are also reported numerically as
`refinementDeclinedPathCount` and `gpuByteBudgetEvictions`, and concurrent
causes are `; `-joined into one `reason` — dropped elements first, then the
caps, an order the tests pin — rather than the first shadowing the rest. No
clause may contain `; ` itself, or a caller splitting on it would read one cause
as several: the residency clause comma-separates its own parenthetical, and any
snapshot-supplied string it echoes (the stop `reason`, `firstPath`) has its
semicolons flattened and its length bounded first, since a `page.evaluate`
payload could otherwise forge a clause. Both new signals are LOADER-SCOPED and
cumulative within that life, with no reset: they say "this happened at some
point while this scene was loaded", not "this is true now", so a scene that
stopped once and later refined fully is still refused. That is deliberate —
refinement order is path-dependent. The scope is the loader, not the page: an
in-page dataset switch builds a fresh `SceneLoader` (and so a fresh reporter and
GPU pool) through `SceneLoaderManager.createLoaderAsync`, so the next scene gets
a clean verdict without a reload.
Note the deliberate asymmetry between them: a residency record is only
written once a rung has actually been declined, so its PRESENCE refuses even
when its fields are unreadable, whereas `gpuPool` rides on every snapshot from
a pooled build and only a positive, readable `byteBudgetEvictions` refuses
(refusing on the pool's total `evictions` would fail every normal nD capture,
since that counter is dominated by routine LRU recycling). Here `totalElements`
is `max(the snapshot's own totalElements field, sum of the four per-type totals)`.
That max is a version-skew hedge, not arithmetic: the capture tool talks to
whatever viewer build is served at `APP_URL`, so a missing total is re-derived
from the per-type ones and a stale or partial snapshot's own field can only
under-claim relative to itself — never under-claim against the per-type totals it
is carrying. The current viewer sets the field to exactly that sum
(`debug-state.ts`), so on a live snapshot the max is inert and it is simply the
sum. A false verdict therefore covers retryable "nothing loaded yet",
permanent "loaded but renderer-truncated", and "loaded but capped by a
memory ceiling" states; callers distinguish them from the human-readable
`reason`.

Every not-ready path (no state, an unexpected snapshot shape,
present-but-non-finite totals, an empty scene, a residency stop, a byte-budget
pool eviction) carries a human-readable `reason`
instead of leaking `NaN`/`undefined`. `reason` is not exclusive to `ok: false`:
it doubles as a CAVEAT channel, so an otherwise-ready verdict that had to count a
total as 0 — because it was non-finite, or because a partial (version-skewed)
snapshot did not carry it at all — still names the affected fields rather than
printing a silent zero. Negative totals are clamped at 0 for the same reason — an
element count cannot be negative, and an unclamped one could cancel a real
positive in the sum.

It measures the scene GRAPH, not the framebuffer: like the `debug-state.ts`
aggregates it mirrors, the totals include HIDDEN nodes and sum every level of a
substitutive `kind=lod` group. An all-hidden scene therefore reports `ok: true`
and can still screenshot blank — deliberately, so the two modules can never
disagree about what a total means. Filter on the per-node `visible` flags for
the stricter question.

Imports nothing but the `DebugState` _type_ — no THREE, no browser globals — so
it runs under vitest and under `tsx` in a Node tool alike. That is the point:
`tools/capture-hires.ts` used to compute this verdict inside its
`page.evaluate` closure, where no test could reach it, and read the totals from
a `state.performance` sub-object `computeDebugState` has never produced.
`DebugState` is FLAT, so every total was `undefined`, `undefined > 0` made `ok`
false for every scene ever captured, and `JSON.stringify` dropping the
`undefined` keys hid the mismatch from the printed diagnostics (#1579). The
totals now come from the flat fields, and the tool's browser closure does
nothing but return the snapshot verbatim.

### `debug-cache-helpers.ts`

Backing for the `window.__luxarDebug.cache.*` sub-object. Exports
`buildDebugCacheHelpers(getLoader: LoaderProvider)`, the `DebugCacheHelpers`
result shape, the `CacheCapableLoader` port (the subset of `SceneLoader` the
helpers depend on), and the `LoaderProvider` alias.

Six thin wrappers — `getStats`, `listDatasets`, `clearL0`, `clearL1`,
`clearL2`, `clearAll` — each resolve the active loader through the injected
provider, check availability (and `hasCachingStore` for L1/L2 + listDatasets),
then delegate. `getStats` / `listDatasets` return a `{ error }` envelope when
no loader is available; `clear*` log a warning and silently no-op. Both
behaviors mirror the pre-extraction contract.

The port-based factory shape lets tests inject a stub loader without standing
up the `SceneLoaderManager` singleton. In production, `debug-interface.ts`
wires the provider as
`() => SceneLoaderManager.getInstance().getDefaultLoader()`.

### `cache-stats-view.ts`

Single function `openCacheStatsView(): void`. Resolves
`DataMonitorManager.getInstance().getDefaultMonitor()`, calls `.show()`,
`.expand()`, `.setActiveTab('cache')`. Best-effort — returns immediately when
no monitor exists (embedded contexts that disable the monitor).

Triggered from two places: `LuxarApp.init()` when `options.openCacheStats` is
set, and `loadDataset` when the `?cache-stats` URL flag is present
(`core/app.ts` wires its private `openCacheStatsView()` wrapper into the
load-dataset ports, and `core/app/dataset/load-dataset.ts` calls
`ports.openCacheStatsView()` after the monitor exists).

## `window.__luxarDebug` Surface

Available once `installDebugInterface` runs (after `LuxarApp.init()`):

| Field                                                                                          | Source                                                               | Purpose                                                                                                                                                                                                                                                                                               |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app`                                                                                          | bootstrap stub, preserved                                            | `LuxarApp` instance                                                                                                                                                                                                                                                                                   |
| `consoleInterceptor`                                                                           | bootstrap stub, preserved                                            | Captured console buffer                                                                                                                                                                                                                                                                               |
| `version`                                                                                      | bootstrap stub, preserved                                            | `buildInfo().version`                                                                                                                                                                                                                                                                                 |
| `scene` / `camera` / `renderer` / `controls` / `postProcessing`                                | `sceneManager.*`                                                     | Live THREE.js refs                                                                                                                                                                                                                                                                                    |
| `animationController` / `inputHandler` / `renderingControls` / `recordingPanel`                | ports                                                                | Subsystem handles                                                                                                                                                                                                                                                                                     |
| `sceneDimsManager`                                                                             | singleton                                                            | nD dimension state                                                                                                                                                                                                                                                                                    |
| `workers.getQueueDepth()` / `workers.getStats()`                                               | `getWorkerPool()`                                                    | Backpressure diagnostic                                                                                                                                                                                                                                                                               |
| `fps` / `fpsSamplingEnabled`                                                                   | `AdaptiveDPRManager`                                                 | Healthy-cadence estimate plus whether sampling is active. `fps` is `undefined` until at least two samples exist and deliberately rejects isolated stalls as outliers; it is not perceived frame rate. `renderer.info.render.frame` is a render-pass counter, not an FPS reading.                      |
| `getState()`                                                                                   | `computeDebugState` + `SceneLoaderManager.isAnyLoadPassInProgress()` | JSON-serialisable scene snapshot, including the `isLoading` flag the E2E data-wait helpers poll                                                                                                                                                                                                       |
| `getDrawOrder()`                                                                               | `computeDrawOrder`                                                   | Per-mesh blending bucket, depthWrite, renderOrder and element count, in draw order                                                                                                                                                                                                                    |
| `renderOnce()`                                                                                 | `animationController.startAnimation()`                               | Kick a frame for stable screenshots                                                                                                                                                                                                                                                                   |
| `getSceneLoader()`                                                                             | `SceneLoaderManager.getInstance()`                                   | Cache inspection root                                                                                                                                                                                                                                                                                 |
| `getPickingSystem()` / `getOverlayManager()`                                                   | port accessors                                                       | Live (survive reloads)                                                                                                                                                                                                                                                                                |
| `cache.getStats()` / `listDatasets()` / `clearL0()` / `clearL1()` / `clearL2()` / `clearAll()` | `buildDebugCacheHelpers`                                             | Cache tier control                                                                                                                                                                                                                                                                                    |
| `showError(message)`                                                                           | `ui/error-overlay`                                                   | Render the error dialog directly (visual-regression hook)                                                                                                                                                                                                                                             |
| `injectSyntheticScene({type, count, bounds?, seed?, clusters?, blending?})`                    | dynamic import                                                       | Perf-bench injector — builds a Points, Lines, or GSplats payload and wires it through `materialManager` + the node-factory pipeline; resolves to the discriminated union `{type, elementCount, <per-type count>, mesh}` (capacity-clamped `elementCount`; per-type alias carries the requested count) |
| `getLodLoadStats()` / `resetLodLoadStats()`                                                    | `data/scene-loader/lod-load-stats`                                   | Per-stage timing for lazy loads plus additive per-level loads, keyed by geometry, level, and residency, with an `:aborted` key when a level load is cancelled. Additive nodes of the same geometry type share keys; node paths are deliberately excluded.                                             |
| `runtimeReady`                                                                                 | `true`                                                               | Sentinel flag for E2E waits                                                                                                                                                                                                                                                                           |

`injectSyntheticScene` dynamically imports `scene/synthetic-scene` so the
builders stay out of the main chunk (that chunk is never loaded unless the
API is invoked under `?debug`); `rendering/line-geometry` and
`rendering/material-manager` are statically imported because they're already
part of the main bundle (production modules), so dynamic-importing them would
save no chunk bytes.

It builds one of three geometry types — Points, Lines, or GSplats. The
shared options are `count` (elements to generate), `bounds` (half-extent of
the generation volume, default 100) and `seed` (deterministic PRNG seed,
default 1; the same `(type, count, bounds, seed, clusters)` yields
byte-identical generated geometry — `blending` selects the node's material
and does not touch the generated buffers).
`clusters` (default 256) is the number of gaussian blobs the points/gsplats
samplers draw from and is ignored by lines. `blending` defaults to
`additive` for lines (historical bench contract) and `normal` for
points/gsplats — the latter so the depth-sort subsystem engages — and
overrides either when set. The resolved `elementCount` is the
capacity-clamped drawn count (uniform across types); the per-type alias
(`segmentCount` / `pointCount` / `splatCount`) carries the requested count.
Points/gsplats injection also emits the production depth-sort commit signals,
so the SortWorker registers the node and the first ordering lands
asynchronously a frame or two after the promise resolves.

```js
// Lines (default additive blending)
const { elementCount } = await window.__luxarDebug.injectSyntheticScene({
  type: 'lines',
  count: 1_000_000,
});
// Points — clustered gaussians, default 'normal' (depth-sorted) blending
await window.__luxarDebug.injectSyntheticScene({ type: 'points', count: 2_000_000, clusters: 512 });
// GSplats — deterministic given (type, count, bounds, seed, clusters)
await window.__luxarDebug.injectSyntheticScene({ type: 'gsplats', count: 500_000, seed: 7 });
```

## Activation

`installDebugInterface` is a no-op unless `ports.debug` is true. `LuxarApp`
sets that flag when the standalone bootstrap detects `?debug` in the URL or
`localStorage['luxar.debug'] === 'true'`. Embedders that want the surface in
their own host page pass `debug: true` to `LuxarApp.init()`.

## See Also

- `../../README.md` — Core package overview (debug section)
- `../../bootstrap.ts` — Where the `__luxarDebug` stub is first seeded
- `../../app.ts` — Caller; `setupDebugInterface()` wires the ports
- `../../../utils/console-interceptor.ts` — Console buffer surfaced as `__luxarDebug.consoleInterceptor`
- `../../../ui/data-monitor-manager.ts` — Backs `openCacheStatsView`
