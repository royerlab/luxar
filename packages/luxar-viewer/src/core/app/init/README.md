# LuxarApp init pipeline

Modules composed by `LuxarApp.init()`. The orchestrator class
(`core/app.ts`) keeps `init()` itself thin — it calls the helpers in
this folder to validate the host environment, forward asset-URL
overrides into module singletons, and build the full viewer subsystem
graph. Dataset routing, `beforeunload` / focus listeners, the debug
surface, and overlays all run _after_ this pipeline returns (see the
sibling `lifecycle/`, `dataset/`, `debug/`, `overlays/` folders).

## Files

| File                      | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build-rail-items.ts`     | `buildRailItems(deps)` — assembles the left control-rail's `ControlRailItem` descriptors. Extracted from the pipeline so the rail's wiring lives in one focused, independently-testable place. Buttons dispatch through `inputHandler.getUiActions()` so on-screen and keyboard behaviour share one command surface; rich controls open rail popovers (see `ui/rail-panels/`). Two gates are read here, not in CSS: `getInputProfile().coarsePointer` adds a momentary **Hide panels** item that always closes the open surfaces and makes Help/Monitor activation close the docked panels (one floating surface on a phone); the Fullscreen chip is emitted only when `document.fullscreenEnabled` (or the WebKit flag) is true — the API is absent on iPhone Safari. |
| `density-guard-wiring.ts` | Wires projected-density tracking into materials, refinement caps, and the per-frame scheduler without making the pipeline restate that policy.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `environment-guards.ts`   | Two upfront fail-fast checks: `assertBrowserEnvironment()` (rejects SSR / non-browser callers when `window` / `document` is missing) and `assertThreeRevision(min=185)` (parses `THREE.REVISION`, rejects hosts whose `three` peer is below what the viewer's Timer / post-processing APIs require). Both throw with a remediation message instead of letting a cryptic `ReferenceError` surface mid-init.                                                                                                                                                                                                                                                                                                                                                             |
| `environment-wiring.ts`   | Wires scene-environment capture, settled-state refresh triggers, and the one-shot `?bake-env` path.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `load-activity.ts`        | Builds the shared predicate for active loader passes and refinement work used by adaptive DPR and scene-environment capture.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `module-overrides.ts`     | `applyModuleOverrides({ wasmPath, workerPath })` — forwards optional asset-URL overrides into the `wasm/` and `workers/worker-pool` module singletons via `setWasmJsUrl` / `setDataWorkerUrl`. Each call is skipped when the option is undefined so default `import.meta.url` resolution still kicks in. Overrides are module-level and persist across `init()` calls (one `LuxarApp` per page in v1).                                                                                                                                                                                                                                                                                                                                                                 |
| `pipeline.ts`             | `runInitPipeline(ports, partial)` — builds the full subsystem graph in order, populating a `Partial<InitPipelineResult>` accumulator the orchestrator pre-allocates so a thrown step still leaves disposable references behind. Returns the same object cast to the full `InitPipelineResult` once every field is set.                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

## Pipeline order

`runInitPipeline` executes the steps below strictly in this order —
order is observable behaviour (e.g. `PerformanceMonitor` subscribes to
the bus the `AnimationController` emits each frame, so the controller
must exist first).

```
runInitPipeline(ports, partial)
├── 1. Log expected-404 notice (spatial index / array attrs probes)
├── 2. Resolve sceneSrc       options.src ?? config.defaultZarrPath
├── 3. resolveFactories       merge options.factories over defaults
├── 4. SceneManager           factories.sceneManager() then init({canvas,
│                              debug, renderer, webgpuForceWebGL,
│                              perfTimestamp}); stored on partial BEFORE
│                              awaiting init so a throw stays disposable
├── 5. AnimationController    factories.animationController(controls,
│                              postProcessing); setContextLostPredicate
│                              → sceneManager.isWebGLContextLost
├── 6. PerformanceMonitor     subscribes to controller's per-frame bus
├── 7. DebugConsole
├── 8. Per-frame callback     'dynamic-clipping' → updateDynamicClippingPlanes
├── 9. LOD registry factory   SceneLoaderManager.setLODGroupRegistryFactory
│                              (→ new LODGroupRegistry closing over live
│                              sceneManager: camera, viewport, displayDims via
│                              sceneDimsManager, byte budget via
│                              getGpuByteBudget, resident bytes via the default
│                              loader's gpuBufferPool)
├── 10. Density-guard wiring  wireDensityGuard; register projected-density
│                              material/refinement hooks and per-frame callback
├── 11. LOD-group selector    'lod-group-selector' per-frame callback that
│                              calls evaluatePerFrame() on the current default
│                              loader and refreshVisibleCounts() on a swap
├── 12. AdaptiveDPRManager    wired to sceneManager + controller
├── 13. Load-activity         buildLoadActivityPredicate; suppress adaptive-DPR
│                              learning while loaders/refinement are active
├── 14. Scene environment     wireSceneEnvironment with the inverse settled
│                              predicate for capture/refresh/bake scheduling
├── 15. ResolutionIndicator   targetFPS = ceil(maxFPS/5)*5; show/reset
│                              on DPR change callback (shown value is
│                              dpr/nativeDPR — percent of native)
├── 16. WebGL/WebGPU loss     webgl-context-restored → NodeFactory
│      listeners              .rebuildAfterContextRestore on loaded scene;
│                              webgpu-device-lost → notifier.error
│                              ("reload to continue"). Both tracked via
│                              ports.events for dispose.
├── 17. SceneLoaderManager    setMonitorFactory(monitorId → DataMonitor
│      monitor injection      Manager.getInstance() lookup/create) so
│                              data/ never imports ui/
├── 18. InputHandler          new + init(); DimensionSliders factory
│                              injected so input/ never imports ui/
├── 19. RenderingControls     factories.renderingControls; cross-link
│                              ↔ AnimationController, AdaptiveDPRManager,
│                              InputHandler
├── 20. RecordingPanel        factories.recordingPanel; setPanelState
│                              Callbacks(ports.get/restorePanelVisibility);
│                              setAdaptiveDPRManager; setRecordingPanel
│                              on input handler; then the controller's
│                              three predicates: setIdleRestorePredicate
│                              (idle native-DPR restore gated on the panel
│                              not currently recording),
│                              setRenderSkipPredicate (loop render skipped
│                              while panel.isLoopRenderSuppressed()) and
│                              setPacingSuspendPredicate (frame pacing off
│                              while panel.isCurrentlyRecording())
├── 21. LayersPanel           factories.layersPanel(document.body, ctrl);
│                              setLayersPanel on input handler
├── 22. ControlRail           buildRailItems(deps) → new ControlRail(items,
│                              performanceMonitor.element); each button fires
│                              the same command as its keyboard shortcut
└── 23. animationController.startAnimation()   render background first,
                                               before any dataset load
```

The pipeline returns the populated `InitPipelineResult`; the orchestrator
copies the fields onto `LuxarApp` and then runs dataset routing +
lifecycle wiring outside this folder.

## Invariants

- **Partial accumulator survives exceptions.** Every constructed
  subsystem is assigned to `partial.<field>` immediately, before any
  follow-up `await` or method call. If a step throws, the orchestrator's
  catch can still call `LuxarApp.dispose()` and reach a disposable
  reference for everything built so far.
- **Construction order is observable.** `pipeline.ts` is deliberately
  flat — no helper extraction that would reorder steps. The monitor
  factory must be injected before any `SceneLoader` is created, the
  animation controller must exist before `PerformanceMonitor`, etc.
- **Listeners are tracked.** Every `addEventListener` the pipeline
  registers on `sceneManager` is paired with a `ports.events.add(...)`
  cleanup closure so app dispose tears them down — no anonymous arrows
  that would leak if `sceneManager` outlives the app.
- **Module overrides leak forward.** `applyModuleOverrides` mutates
  module-level state in `wasm/` and `workers/worker-pool`. A second
  `init()` without the option will not reset to defaults; v1 supports
  one `LuxarApp` per page.

## See also

- `../../app.ts` — the orchestrator that calls `assertBrowserEnvironment`,
  `assertThreeRevision`, `applyModuleOverrides`, then `runInitPipeline`.
- `../factories.ts` — `resolveFactories` and the `AppFactories` shape
  consumed at step 3.
- `../options.ts` — `LuxarAppOptions` interface (canvas, src, debug,
  renderer, wasmPath, workerPath, factories, …).
- `../lifecycle/` — `beforeunload` / focus / visibility wiring that runs
  after the pipeline returns.
- `../dataset/` — `shouldShowBrowser` / `loadDataset` routing that
  consumes `sceneSrc`.
