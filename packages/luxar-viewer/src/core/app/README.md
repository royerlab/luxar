# core/app — LuxarApp's private support tree

`LuxarApp` (in `../app.ts`) is intentionally thin: every method delegates
to a helper in this folder. The class itself owns state and ordering; the
helpers own the actual work. With two exceptions, nothing under `app/` is
re-exported from the package barrel — these files exist as the
orchestrator's private support code, grouped thematically. The exceptions:
`embedder/`, whose public value/event types (`LuxarEmbedderEventMap`,
`EmbedderDimensions`, `ScreenshotOptions`, …) are re-exported from
`src/index.ts` as part of the programmatic embedder API, and
`snapshot/viewer-snapshot.ts`, whose `type ViewerSnapshot` is also
re-exported from `src/index.ts`.

Tests mirror the layout under `tests/unit/core/app/<theme>/`.

## Top-level files

| File              | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `factories.ts`    | `AppFactories` interface, `defaultFactories`, and `resolveFactories(overrides)`. Optional construction overrides for the heavy subsystems built in `init()` (SceneManager, AnimationController, RenderingControls, RecordingPanel, LayersPanel). Embedders + tests inject pre-built stubs here without subclassing.                                                                                                                                                                                                                            |
| `options.ts`      | `LuxarAppOptions` interface — every init-time knob the orchestrator accepts: `canvas`, `container`, `src`, `debug`, `loaderConfig`, `gpuPoolMaxBytes`, `updateBrowserUrl`, `wasmPath`, `workerPath`, `openCacheStats`, `factories`, `renderer`, `webgpuForceWebGL`, `perfTimestamp`, `renderAlways`, `renderAudit`, `pinnedDPR`, `lodFade`, `lodEnergyComp`, `lodFinest`, `lodBias`, `bakeEnvironment`, `blendWarmup`, `depthSort`, `densityGuard`, `densityCap`, `allowLinks`, `control`, `controlToken`, `kiosk`. Re-exported from `app.ts`. |
| `error-dialog.ts` | `showViewerError(message, shortcutForAction, options?)` + `ERROR_DIALOG_SHORTCUTS` — the error dialog with its dataset-browser / help hints, shared by the bootstrap, the dataset browser and the debug surface.                                                                                                                                                                                                                                                                                                                               |

## Subpackages

```
app/
├── init/           # init() pipeline: environment guards, module overrides, subsystem graph build
├── lifecycle/      # dispose / focus / unload — teardown and runtime power-management glue
├── dataset/        # src URL → scene routing, the load sequence, and DatasetSession (per-dataset state)
├── viewer-config/  # Apply zarr viewer_config onto the live app (incl. scene audio) + panel-visibility capture
├── camera/         # CameraFlight (flyTo tween) + story waypoints (WaypointDriver, installStoryWaypoints)
├── kiosk/          # ?kiosk / ui.kiosk lock-down + the GPU-loss reload watchdog
├── control/        # Remote-control client (?control=ws://…) and its method allowlist
├── interaction/    # Canvas gesture / context-menu ownership, element actions, double-tap-to-fit
├── snapshot/       # JSON capture/restore of camera + per-dimension slice position
├── embedder/       # Public programmatic embedder API — event/value types + headless screenshot
├── debug/          # window.__luxarDebug surface — runtime hook for AI drivers + Playwright
├── picking/        # GPU picking session lifecycle + pick-result → hover overlay routing
└── overlays/       # Screen-space HUD layers — OverlayManager, ScaleBar, ColormapLegend
```

- **`init/`** — Modules composed by `LuxarApp.init()`: browser + `THREE.REVISION`
  guards, `wasmPath`/`workerPath` forwarding into module singletons, and the
  subsystem-graph pipeline that populates a partial accumulator so a thrown
  step still leaves disposable references behind.
- **`lifecycle/`** — Fault-tolerant ordered teardown of every subsystem, plus
  window-focus + document-visibility → animate pause/resume and a
  `beforeunload` → `app.dispose()` listener. Does not drive frames; it only
  starts, stops, and disposes the controller that does.
- **`dataset/`** — URL classification (must-browse / probe / load), the async
  zarr metadata HEAD probe, the `DatasetBrowser` modal lifecycle, the
  scene-dependent UI initialization sequence that follows a successful load,
  and `DatasetSession`: the one owner of what a dataset installs in the app
  (archive-fault subscription, story waypoints, kiosk watchdog, control-panel
  authoring), replaced at every load start and disposed with the app.
- **`viewer-config/`** — Routes the non-rendering fields of the per-dataset
  `viewer_config` blob (panel show/hide flags, theme, dimension-nav step, and
  animation dispatch) onto the running app, binds the sound layer to the scene
  (`install-audio.ts`), plus a pure helper for snapshotting panel visibility
  around recording. Rendering knobs are routed separately via
  `RenderingControls`.
- **`camera/`** — `CameraFlight`: the tween behind `LuxarApp.flyTo()`. Interpolates
  between two `CameraSnapshot`s in the orbit parameterisation (target lerp, direction
  slerp, log distance, up slerp), runs as a `continuous` per-frame callback paired with
  `startAnimation()`, hands every frame to the controls the way `restoreCamera` does,
  and cancels on canvas pointer/wheel/touch or document keydown. `buildFlightPath` /
  `easeFlight` are pure and exported for tests. Also `WaypointDriver`: binds the
  scene's authored `viewer_config.waypoints` (camera poses keyed on hidden-dimension
  positions with the overlay `visible_range` matching rule) to the dims manager —
  snap at load, `flyTo` on a change of matched waypoint; `installStoryWaypoints`
  wires it to the app through ports and returns the binding the dataset session
  owns.
- **`kiosk/`** — `applySceneKiosk` resolves kiosk mode from the scene's
  `ui.kiosk` block and `?kiosk` (URL wins) and applies it (`applyKioskMode`:
  input permissions, hidden panels, the `watchdog.ts` reload on a WebGL context that
  never returns or a lost WebGPU device),
  returning separate teardowns: the app owns input restoration across loads,
  while the dataset session owns the watchdog.
- **`control/`** — The `?control=` remote-control WebSocket client, its
  callable-method allowlist and wire-value sanitizers.
- **`interaction/`** — Claims the canvas's gestures and context menu for the
  viewer, element actions (`element_actions` templates), the picked-element
  cache, and touch double-tap-to-fit.
- **`bookmark-state.ts`** — serializes the current camera, dimensions, rendering and layer appearance into a shareable URL and restores it through the public app setters.
- **`snapshot/`** — `captureSnapshot` / `restoreSnapshot` + `ViewerSnapshot`
  types. JSON-serialisable view state (camera + slice position only) for
  tests, share-view links, and regression harnesses. Layer-panel and
  rendering-controls settings live on other abstractions and are excluded.
- **`embedder/`** — The public programmatic embedder API. `events.ts` defines
  the app-scoped event catalog (`LuxarEmbedderEventMap`) and value types
  (`EmbedderDimensions`, `ScreenshotOptions`, …) behind `LuxarApp.on()`,
  `getDimensions()` / `setDimensionValue()`, and `screenshot()`;
  `screenshot.ts` implements `captureScreenshot` (headless frame → encoded
  Blob, no Recording-panel dependency). Unlike the rest of `app/`, these
  types are re-exported from the package root (`src/index.ts`).
- **`debug/`** — Owns the `window.__luxarDebug` global. Under `?debug` the
  bootstrap seeds a stub before `init()` (app handle, version, the perf readers
  `getPerf` / `getPerfRecords` / `resetPerfCounters`, `showError`) and installs
  the debug perf instruments; `installDebugInterface(ports)` extends it with
  live runtime components, cache helpers, scene-walking state, and the
  synthetic-scene injector once `init()` finishes.
- **`picking/`** — Stands up `rendering/picking/PickingSystem` for the current
  scene when any node has `has_labels` / `has_image_labels` / `has_keys`, an
  element-action template, or an embedder listener that consumes picks. It tears
  down any prior session and wires DOM + Three.js EventDispatcher listeners. The
  pure pick-result → hover-payload branch logic is split into a separate handler
  so it can be unit-tested with stub ports.
- **`overlays/`** — Builds and disposes the three non-3D HUD layers
  (`OverlayManager`, `ScaleBar`, `ColormapLegend`) on `loadDataset()` /
  `dispose()`. (Other folders reach `ui/` too: `dataset/` for the dataset
  browser, `error-dialog.ts` for the error overlay, `debug/` and `init/` for
  the panels they build.)

## Conventions

Every helper takes an explicit `Ports` interface — the orchestrator wires
its long-lived components (SceneManager, InputHandler, RenderingControls,
overlay-init callbacks) in without the helper needing to know how they're
implemented. Helpers never mutate orchestrator state directly; they return
what they installed (a teardown, an `InstalledWaypoints` binding) and the
orchestrator — or the `DatasetSession` it owns — keeps it.

## See Also

- `../README.md` — Core package overview, the `LuxarApp` public API,
  initialization sequence, and the standalone vs embedded bootstrap split.
- `../app.ts` — The orchestrator that calls into every subpackage here.
- `../bootstrap.ts` — Pre-init sequence and the `bootstrapStandalone()` factory.
