# core/app — LuxarApp's private support tree

`LuxarApp` (in `../app.ts`) is intentionally thin: every method delegates
to a helper in this folder. The class itself owns state and ordering; the
helpers own the actual work. With one exception, nothing under `app/` is
re-exported from the package barrel — these files exist as the
orchestrator's private support code, grouped thematically. The exception is
`embedder/`: its public value/event types (`LuxarEmbedderEventMap`,
`EmbedderDimensions`, `ScreenshotOptions`, …) are re-exported from
`src/index.ts` as part of the programmatic embedder API.

Tests mirror the layout under `tests/unit/core/app/<theme>/`.

## Top-level files

| File           | Role                                                                                                                                                                                                                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `factories.ts` | `AppFactories` interface, `defaultFactories`, and `resolveFactories(overrides)`. Optional construction overrides for the heavy subsystems built in `init()` (SceneManager, AnimationController, RenderingControls, RecordingPanel, LayersPanel). Embedders + tests inject pre-built stubs here without subclassing. |
| `options.ts`   | `LuxarAppOptions` interface — every init-time knob the orchestrator accepts: `canvas`, `container`, `src`, `debug`, `loaderConfig`, `updateBrowserUrl`, `wasmPath`, `workerPath`, `openCacheStats`, `factories`, `renderer`, `webgpuForceWebGL`, `perfTimestamp`, `pinnedDPR`. Re-exported from `app.ts`.           |

## Subpackages

```
app/
├── init/           # init() pipeline: environment guards, module overrides, subsystem graph build
├── lifecycle/      # dispose / focus / unload — teardown and runtime power-management glue
├── dataset/        # src URL → scene routing: browser-vs-load decision + load sequence
├── viewer-config/  # Apply zarr viewer_config onto the live app + panel-visibility capture
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
  zarr metadata HEAD probe, the `DatasetBrowser` modal lifecycle, and the
  scene-dependent UI initialization sequence that follows a successful load.
- **`viewer-config/`** — Routes the non-rendering fields of the per-dataset
  `viewer_config` blob (panel show/hide flags, theme, dimension-nav step) onto
  the running app, plus a pure helper for snapshotting panel visibility around
  recording. The 47 rendering knobs are routed separately via `RenderingControls`.
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
- **`debug/`** — Owns the `window.__luxarDebug` global. The bootstrap seeds a
  minimal stub from the first JS tick; `installDebugInterface(ports)` extends
  it with live runtime components, cache helpers, scene-walking state, and
  the synthetic-scene injector once `init()` finishes.
- **`picking/`** — Stands up `rendering/picking/PickingSystem` for the current
  scene when any node has `has_labels` / `has_image_labels`, tears down any
  prior session, and wires DOM + Three.js EventDispatcher listeners. The pure
  pick-result → hover-payload branch logic is split into a separate handler
  so it can be unit-tested with stub ports.
- **`overlays/`** — Builds and disposes the three non-3D HUD layers
  (`OverlayManager`, `ScaleBar`, `ColormapLegend`) on `loadDataset()` /
  `dispose()`. The only files in `app/` that touch the matching `ui/` modules.

## Conventions

Every helper takes an explicit `Ports` interface — the orchestrator wires
its long-lived components (SceneManager, InputHandler, RenderingControls,
overlay-init callbacks) in without the helper needing to know how they're
implemented. Helpers never mutate orchestrator state directly; callbacks
(`onDisposed`, `disposePrevious`) keep ownership with `LuxarApp`.

## See Also

- `../README.md` — Core package overview, the `LuxarApp` public API,
  initialization sequence, and the standalone vs embedded bootstrap split.
- `../app.ts` — The orchestrator that calls into every subpackage here.
- `../bootstrap.ts` — Pre-init sequence and the `bootstrapStandalone()` factory.
