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
(see `installDebugInterface` at `core/app.ts:435`, `openCacheStatsView` re-export
at `core/app.ts:45`).

## File Structure

```
debug/
├── debug-interface.ts       # installDebugInterface() — populates window.__luxarDebug
├── debug-state.ts           # computeDebugState() — pure scene-walking snapshot
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
loads is captured under `?debug` only. Those lazy `ensureLoaded` loads run
outside any `updateView` cycle, so the `UpdateProfiler` never sees them — this
fills the gap. The captured stats are reachable as
`__luxarDebug.getLodLoadStats()` (snapshot) and `resetLodLoadStats()`.

`injectSyntheticScene`'s body is wrapped in try/catch: on failure it logs via
`log.error` and surfaces the message through the user-facing error overlay
(`showError`), then re-throws so callers that `await` still see the rejection.
This prevents a broken dynamic import / synthetic-line builder from becoming a
silent unhandled promise rejection.

After populating the global, it emits a series of `log.info(Modules.LUXAR, …)`
lines documenting the available commands; this is the in-console help text
users see after enabling `?debug`.

### `debug-state.ts`

Pure helper behind `__luxarDebug.getState()`. Exports
`computeDebugState(ctx: DebugStateContext): DebugState`, the input-surface
interface `DebugStateContext`, and the result-shape interfaces (`DebugState`,
`PointCloudInfo`, `GSplatMeshInfo`, `LineMeshInfo`, `GPUPoolDebugStats`,
`LODGroupDebugInfo`, `PartitionDebugInfo`).

`computeDebugState` walks the scene graph once and tallies per-mesh detail for
three geometry types — Points, Lines, GSplats — by inspecting
`userData.nodeType`. For each it uses `InstancedBufferGeometry.instanceCount`
as the source of truth, because pooled attribute arrays are over-allocated and
`drawRange` only covers the 6-index base quad. Points additionally fall back
to `userData.visiblePointCount` and then attribute count when `instanceCount`
is absent.

The same traversal also summarises specialized-group containers by their
`userData.kind`: `kind=lod` groups become `lodGroups[]` (level count + the
index of the visible child as `activeLevel`, `-1` when none), and
`kind=partition` groups become `partitions[]` (part count + visible-part
count).

For lines, the `hasColormap` flag is read structurally from
`material.defines.USE_COLORMAP` so the result is identical whether the mesh is
running on the GLSL `ShaderMaterial` or the TSL `NodeMaterial` backend.

The returned `DebugState` carries `totalPoints`, `totalGSplats`, `totalLines`,
`totalElements` (their sum), the per-mesh arrays, `lodGroups`, `partitions`, an
optional `gpuPool` byte-stats block, `dimensions`, and a nested `camera`
(`{position, fov}`) plus flat `cameraPosition` / `cameraFov` mirrors kept for
back-compat.

Dependencies arrive as parameters (`scene`, `camera`, `currentFov`,
`isAnimating`, `initialized`, `dims`, optional `gpuPoolStats`), so the helper
is callable from unit tests against real `THREE.Points` / `THREE.Mesh`
fixtures without bringing up the WebGL renderer. (`gpuPoolStats` is not wired
in the production `installDebugInterface` call, so `gpuPool` is `undefined`
there; tests pass it explicitly.)

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
set, and `loadDataset` when the `?cache-stats` URL flag is present (the
re-export lives at `core/app.ts:46`, called by
`core/app/dataset/load-dataset.ts:99`).

## `window.__luxarDebug` Surface

Available once `installDebugInterface` runs (after `LuxarApp.init()`):

| Field                                                                                          | Source                                 | Purpose                                                                                                                                                                       |
| ---------------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app`                                                                                          | bootstrap stub, preserved              | `LuxarApp` instance                                                                                                                                                           |
| `consoleInterceptor`                                                                           | bootstrap stub, preserved              | Captured console buffer                                                                                                                                                       |
| `version`                                                                                      | bootstrap stub, preserved              | `'1.0.0'`                                                                                                                                                                     |
| `scene` / `camera` / `renderer` / `controls` / `postProcessing`                                | `sceneManager.*`                       | Live THREE.js refs                                                                                                                                                            |
| `animationController` / `inputHandler` / `renderingControls` / `recordingPanel`                | ports                                  | Subsystem handles                                                                                                                                                             |
| `sceneDimsManager`                                                                             | singleton                              | nD dimension state                                                                                                                                                            |
| `workers.getQueueDepth()` / `workers.getStats()`                                               | `getWorkerPool()`                      | Backpressure diagnostic                                                                                                                                                       |
| `getState()`                                                                                   | `computeDebugState`                    | JSON-serialisable scene snapshot                                                                                                                                              |
| `renderOnce()`                                                                                 | `animationController.startAnimation()` | Kick a frame for stable screenshots                                                                                                                                           |
| `getSceneLoader()`                                                                             | `SceneLoaderManager.getInstance()`     | Cache inspection root                                                                                                                                                         |
| `getPickingSystem()` / `getOverlayManager()`                                                   | port accessors                         | Live (survive reloads)                                                                                                                                                        |
| `cache.getStats()` / `listDatasets()` / `clearL0()` / `clearL1()` / `clearL2()` / `clearAll()` | `buildDebugCacheHelpers`               | Cache tier control                                                                                                                                                            |
| `showError(message)`                                                                           | `ui/error-overlay`                     | Render the error dialog directly (visual-regression hook)                                                                                                                     |
| `injectSyntheticScene({type, count, bounds?, seed?})`                                          | dynamic import                         | Lines perf-bench injector — builds an `InstancedLinesMeshConfig` and adds it through `materialManager` + `createInstancedLinesMesh`; resolves to `{type, segmentCount, mesh}` |
| `getLodLoadStats()` / `resetLodLoadStats()`                                                    | `data/scene-loader/lod-load-stats`     | Per-stage timing for lazy LOD level loads (fetch/decode, process, commit, release)                                                                                            |
| `runtimeReady`                                                                                 | `true`                                 | Sentinel flag for E2E waits                                                                                                                                                   |

`injectSyntheticScene` dynamically imports `scene/synthetic-scene` so the
synthetic-line builder stays out of the main chunk;
`rendering/line-geometry` and `rendering/material-manager` are statically
imported because they're already part of the main bundle (production
modules), so dynamic-importing them would save no chunk bytes.

## Activation

`installDebugInterface` is a no-op unless `ports.debug` is true. `LuxarApp`
sets that flag when the standalone bootstrap detects `?debug` in the URL or
`localStorage['luxar.debug'] === 'true'`. Embedders that want the surface in
their own host page pass `debug: true` to `LuxarApp.init()`.

## See Also

- `../../README.md` — Core package overview (debug section)
- `../../bootstrap.ts` — Where the `__luxarDebug` stub is first seeded
- `../../app.ts` — Caller; line 435 wires the ports
- `../../../utils/console-interceptor.ts` — Console buffer surfaced as `__luxarDebug.consoleInterceptor`
- `../../../ui/data-monitor-manager.ts` — Backs `openCacheStatsView`
