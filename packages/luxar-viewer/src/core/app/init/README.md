# LuxarApp init pipeline

Modules composed by `LuxarApp.init()`. The orchestrator class
(`core/app.ts`) keeps `init()` itself thin — it calls the helpers in
this folder to validate the host environment, forward asset-URL
overrides into module singletons, and build the full viewer subsystem
graph. Dataset routing, `beforeunload` / focus listeners, the debug
surface, and overlays all run _after_ this pipeline returns (see the
sibling `lifecycle/`, `dataset/`, `debug/`, `overlays/` folders).

## Files

| File                    | Role                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `environment-guards.ts` | Two upfront fail-fast checks: `assertBrowserEnvironment()` (rejects SSR / non-browser callers when `window` / `document` is missing) and `assertThreeRevision(min=184)` (parses `THREE.REVISION`, rejects hosts whose `three` peer is below what the viewer's Timer / post-processing APIs require). Both throw with a remediation message instead of letting a cryptic `ReferenceError` surface mid-init. |
| `module-overrides.ts`   | `applyModuleOverrides({ wasmPath, workerPath })` — forwards optional asset-URL overrides into the `wasm/` and `workers/worker-pool` module singletons via `setWasmJsUrl` / `setDataWorkerUrl`. Each call is skipped when the option is undefined so default `import.meta.url` resolution still kicks in. Overrides are module-level and persist across `init()` calls (one `LuxarApp` per page in v1).     |
| `pipeline.ts`           | `runInitPipeline(ports, partial)` — builds the full subsystem graph in order, populating a `Partial<InitPipelineResult>` accumulator the orchestrator pre-allocates so a thrown step still leaves disposable references behind. Returns the same object cast to the full `InitPipelineResult` once every field is set.                                                                                     |

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
├── 9. LOD-group wiring        SceneLoaderManager.setLODGroupRegistryFactory
│                              (→ new LODGroupRegistry closing over live
│                              sceneManager: camera, viewport, displayDims via
│                              sceneDimsManager, byte budget via
│                              getGpuByteBudget, resident bytes via
│                              getSceneLoader('default').gpuBufferPool); then a
│                              'lod-group-selector' per-frame callback that
│                              calls evaluatePerFrame() on the current default
│                              loader and refreshVisibleCounts() on a swap
├── 10. AdaptiveDPRManager    wired to sceneManager + controller
├── 11. ResolutionIndicator   targetFPS = ceil(maxFPS/5)*5; show/reset
│                              on DPR change callback
├── 12. WebGL/WebGPU loss     webgl-context-restored → NodeFactory
│      listeners              .rebuildAfterContextRestore on loaded scene;
│                              webgpu-device-lost → notifier.error
│                              ("reload to continue"). Both tracked via
│                              ports.events for dispose.
├── 13. SceneLoaderManager    setMonitorFactory(monitorId → DataMonitor
│      monitor injection      Manager.getInstance() lookup/create) so
│                              data/ never imports ui/
├── 14. InputHandler          new + init(); DimensionSliders factory
│                              injected so input/ never imports ui/
├── 15. RenderingControls     factories.renderingControls; cross-link
│                              ↔ AnimationController, AdaptiveDPRManager,
│                              InputHandler
├── 16. RecordingPanel        factories.recordingPanel; setPanelState
│                              Callbacks(ports.get/restorePanelVisibility);
│                              setAdaptiveDPRManager; setRecordingPanel
│                              on input handler
├── 17. LayersPanel           factories.layersPanel(document.body, ctrl);
│                              setLayersPanel on input handler
└── 18. animationController.startAnimation()   render background first,
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
