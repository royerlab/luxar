# Debug Interface Guide

## Overview

Luxar exposes a `window.__luxarDebug` object that provides programmatic access
to the viewer's internal state. This is intended for:

- Diagnosing rendering or data-loading issues during development
- Writing and debugging Playwright E2E tests
- AI-assisted debugging via the agent driver (`pnpm agent:debug`)
- Generating state dumps for bug reports

The interface is defined in two stages: `src/core/bootstrap.ts` creates the
base object with the `app` reference and version string, then
`installDebugInterface()` (in `src/core/app/debug/debug-interface.ts`, invoked
from `app.ts` via `setupDebugInterface()`) extends it with runtime components
(scene, camera, cache helpers, etc.) after initialization completes.

## Enabling the Debug Interface

**URL parameter** (recommended):

```
http://localhost:5173/?src=http://127.0.0.1:8000&debug
```

**localStorage** (persists across page loads):

```js
localStorage.setItem('luxar.debug', 'true');
```

Note: either flag — the `?debug` URL parameter or the persisted `luxar.debug`
localStorage entry — enables full debug mode: `bootstrap.ts` seeds the base
debug object, and `setupDebugInterface()` in `app.ts` (which delegates to
`installDebugInterface()` in `src/core/app/debug/debug-interface.ts`) adds the
runtime components (scene, camera, controls, etc.) once initialization completes.

When enabled, the viewer logs `Debug interface available at window.__luxarDebug`
to the console and, after initialization, prints a summary of available commands.

When the `?debug` URL parameter is absent *and* localStorage is not set, the
object is never created, so there is zero overhead in production.

## Available Properties and Methods

### Base properties (from `bootstrap.ts`, available immediately)

| Property | Type | Description |
|---|---|---|
| `window.__luxarBuild` | `BuildInfo` | Unconditional pre-initialization build identity: version, commit, build time, and whether the bundle was stamped. |
| `app` | `LuxarApp` | The application instance. |
| `consoleInterceptor` | `ConsoleInterceptor` | Captures all console output for replay. |
| `version` | `string` | `VIEWER_VERSION` (`src/version.ts`): the `package.json` version injected at build time (a semver-normalized CalVer such as `2026.6.5`), or `0.0.0-dev` in a consumer that bundles `src/` without the define. Commit and build time live on `window.__luxarBuild`. |

### Runtime components (from `installDebugInterface()`, available once `runtimeReady` is `true`)

| Property | Type | Description |
|---|---|---|
| `scene` | `THREE.Scene` | The Three.js scene graph root. |
| `camera` | `LuxarCamera` | Active camera (perspective or orthographic). |
| `renderer` | `THREE.WebGLRenderer` | The WebGL renderer instance. |
| `controls` | `ControlsManager` | Orbit/fly controls manager. |
| `postProcessing` | `PostProcessingManager` | Bloom, AO, AA pipeline. |
| `animationController` | `AnimationController` | Manages the render loop. |
| `inputHandler` | `InputHandler` | Keyboard and mouse input system. |
| `renderingControls` | `RenderingControls` | UI panel for rendering settings. |
| `recordingPanel` | `RecordingPanel` | Screenshot and video capture panel. |
| `sceneDimsManager` | `SceneDimsManager` | Dimension navigation state. |
| `runtimeReady` | `boolean` | `true` once all components are initialized. |

### Helper functions

| Method | Return type | Description |
|---|---|---|
| `getState()` | `object` | JSON-serializable snapshot of current state (point counts per cloud, camera position/FOV, dimension info, animation status, initialization status, the two memory-ceiling fields tabled below, and `isLoading` — true while a load pass, i.e. an `updateView` fetch/decode/upload sweep up to its geometry commit, is in flight on any registered scene loader, or a view-state is queued behind one; see the scope notes below). |
| `renderOnce()` | `void` | Kicks the animation loop to force a single render frame. Useful for stable screenshots. |
| `getSceneLoader()` | `SceneLoaderManager` | Returns the singleton scene loader manager for inspecting loaded data. |

#### Memory-ceiling fields on `getState()`

Two fields answer "were these element counts decided by the store, or by this
machine?". `core/app/debug/capture-readiness.ts` refuses a capture on either
(#2508). `refinementResidency` and `gpuPool.byteBudgetEvictions` are
**cumulative for the life of the scene loader and never reset within it**: they
report that something happened at some point while this scene was loaded, not
that it is true right now. The scope is the LOADER, not the page — an in-page
dataset switch builds a fresh `SceneLoader` (fresh reporter, fresh GPU pool), so
the next scene starts clean with no reload. The rest of `gpuPool` is *not*
cumulative: `activeBytes`,
`pooledBytes`, `totalBytes`, `largestPooledBytes`, `activeBuffers` and
`pooledBuffers` are instantaneous readings, and `evictions` is a lifetime
counter dominated by routine recycling.

| Field | Type | Description |
|---|---|---|
| `gpuPool` | `GPUPoolDebugStats` (absent when unavailable) | GPU buffer-pool usage — the instantaneous `activeBuffers` / `pooledBuffers` / `activeBytes` / `pooledBytes` / `totalBytes` / `largestPooledBytes`, plus the cumulative `evictions` and `byteBudgetEvictions`. The last is the subset charged by the VRAM byte-budget pass: `evictions` alone is dominated by routine LRU recycling and means nothing on its own. Not new to the snapshot TYPE, but only **populated in production** as of #2508 (a `gpuPoolStats` provider had been declared and never passed), so an older build reports `undefined` here. Absent when no loader is registered or pooling is disabled. |
| `refinementResidency` | `RefinementResidencyStop` (absent when nothing was declined) | Present once progressive refinement declined at least one rung at the residency byte ceiling; carries the first refusal's `reason` / `residentBytes` / `budgetBytes` / `firstPath`, the distinct `declinedPathCount`, and a bounded `declinedPaths` sample. **Presence is the signal** — absence means "never stopped" (or an older build), never "in trouble". |

### Cache helpers (`__luxarDebug.cache`)

| Method | Return type | Description |
|---|---|---|
| `getStats()` | `{ l0, l1, l2 }` | Returns hit/miss/size statistics for all three cache tiers. |
| `listDatasets()` | `object` | Lists datasets stored in the caching store. |
| `clearL0()` | `void` | Clears the L0 decompressed chunk cache (in-memory). |
| `clearL1()` | `void` | Clears the L1 memory cache. |
| `clearL2()` | `Promise<void>` | Clears the L2 OPFS (Origin Private File System) persistent cache. |
| `clearAll()` | `Promise<void>` | Clears all three cache tiers. |

## Common Debug Workflows

### Inspecting scene state

Open the browser console and run:

```js
const state = __luxarDebug.getState();
console.table(state.pointClouds);  // Per-cloud point counts, visibility, attributes
console.log('Total points:', state.totalPoints);
console.log('Camera:', state.camera);
console.log('Dimensions:', state.dimensions);
```

To walk the Three.js scene graph directly:

```js
__luxarDebug.scene.traverse(obj => {
  if (obj.type === 'Points') console.log(obj.name, obj.geometry.attributes);
});
```

### Checking cache performance

```js
const stats = __luxarDebug.cache.getStats();
console.log('L0 (decompressed):', stats.l0);
console.log('L1 (memory):', stats.l1);
console.log('L2 (OPFS):', stats.l2);
```

To isolate a cache tier during testing:

```js
__luxarDebug.cache.clearL0();  // Force re-decompression on next access
await __luxarDebug.cache.clearL2();  // Force re-fetch from network
```

### Forcing re-renders

```js
__luxarDebug.renderOnce();
```

This restarts the animation loop briefly, producing at least one fresh frame.
Useful after programmatically changing material uniforms or camera position.

### Dumping state for bug reports

Copy-paste the following into the console to produce a JSON blob suitable for
attaching to an issue:

```js
JSON.stringify(__luxarDebug.getState(), null, 2);
```

For cache state:

```js
JSON.stringify(__luxarDebug.cache.getStats(), null, 2);
```

## Using with Playwright (E2E Tests)

E2E tests load the viewer with `?debug` in the URL and wait for the debug
interface before making assertions. The helpers in
`src/tests/e2e/helpers.ts` encapsulate the common patterns.

Eight helpers poll `getState().isLoading`: `waitForDataLoaded`,
`waitForDimensionNavigation`, `waitForSpatialQuery`,
`waitForSpatialQueryOrThrow`, `waitForNavigationComplete`,
`waitForNavigationCompleteOrThrow`, and the state-based fallbacks inside
`waitForRenderStable` and `waitForNextRender`. (`waitForPointsLoaded` does not —
it gates on `state.totalPoints >= minPoints`.) So that flag has to be a real
boolean in every snapshot, not merely absent when nothing is loading: `!undefined`
is `true`, which gates on nothing.

`isLoading` is scoped to a load pass — an `updateView` sweep (fetch / decode /
upload) up to its geometry commit, a failed-loader retry (which takes the same
lock), or a view-state that is **queued** behind either and has not begun
loading yet. That last clause is what keeps the refinement exclusion below from
opening a hole: a nav arriving during a refinement hold parks in the queue
without touching the lock, so an `isLoading: true` with no fetch in flight is
the expected reading on any laddered dataset. It deliberately does **not**
cover:

- the **initial `loadScene`** (that path only touches the loader's lock at its
  very end, to hand it to the post-load refinement kick). Wait on `initialized`
  for the first load. An in-page **dataset switch** is covered by neither flag:
  `initialized` stays true and the replacement loader is registered before its
  `loadScene` runs, so `isLoading` reads false throughout the switch's load.
- **lazy substitutive-LOD / deferred-partition `ensureLoaded` promotions**,
  which run outside any `updateView` cycle and surface as content-change
  notifications instead.
- the **progressive-LOD refinement drain**, which inherits the same lock after
  the current view has already committed. Excluding it keeps the flag reporting
  first-commit latency rather than full-ladder latency, so a wait doesn't sit
  through every additive ladder. `SceneLoader.isUpdateInProgress()` retains the
  broader "the lock is held at all" meaning for the adaptive-DPR manager and the
  init pipeline.

### Waiting for initialization

```typescript
import { waitForLuxarReady } from './helpers';

await waitForLuxarReady(page);  // Waits for getState().initialized === true
```

Under the hood this polls `window.__luxarDebug.getState().initialized`:

```typescript
await page.waitForFunction(() => {
  const debug = (window as any).__luxarDebug;
  return debug && debug.getState && debug.getState().initialized;
}, null, { timeout: 45000 });
```

### Reading state from a test

```typescript
import { getLuxarState } from './helpers';

const state = await getLuxarState(page);
expect(state.totalPoints).toBeGreaterThan(0);
```

### Forcing a render for screenshots

```typescript
import { renderOnce } from './helpers';

await renderOnce(page);  // Triggers render + 100ms settle
await page.screenshot({ path: 'screenshot.png' });
```

### Waiting for data to load

```typescript
import { waitForPointsLoaded } from './helpers';

await waitForPointsLoaded(page, 100);  // Wait until >= 100 points are loaded
```

### Direct `page.evaluate` access

For one-off checks not covered by helpers:

```typescript
const cacheStats = await page.evaluate(() => {
  return (window as any).__luxarDebug.cache.getStats();
});
```

## Using with the Agent Debugger

The agent debugger (`pnpm agent:debug`) launches a headless Playwright browser
with `?debug` in the URL and dumps the debug interface state automatically.

```bash
cd packages/luxar-viewer
pnpm agent:debug           # Headless, captures state + screenshot
pnpm agent:debug:visible   # Headed browser for visual inspection
```

The output includes:

- `[BROWSER-CONSOLE-*]` -- all console messages captured by `consoleInterceptor`
- A JSON state dump from `getState()` (point counts, camera, dimensions)
- A screenshot saved to `test-results/debug/debug-view.png`

Typical workflow:

1. Run `pnpm agent:debug` to capture current viewer state.
2. Inspect the JSON dump and screenshot for anomalies.
3. If needed, add `console.log()` calls in source code and re-run.
4. Fix the issue and verify with another `pnpm agent:debug` run.
5. Remove any temporary logging before committing.

## Source Files

| File | Role |
|---|---|
| `packages/luxar-viewer/src/core/bootstrap.ts` | Creates the base `__luxarDebug` object and checks activation flags (`?debug` URL param / `luxar.debug` localStorage). |
| `packages/luxar-viewer/src/core/app.ts` | Calls `setupDebugInterface()` after init, delegating to `installDebugInterface()`. |
| `packages/luxar-viewer/src/core/app/debug/debug-interface.ts` | `installDebugInterface()` — assigns the runtime components (scene, camera, cache helpers, `getState`, etc.) onto `__luxarDebug`. |
| `packages/luxar-viewer/src/tests/e2e/helpers.ts` | Playwright helper functions that consume the debug interface. |
