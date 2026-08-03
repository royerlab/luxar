# App lifecycle

`LuxarApp`'s teardown and runtime power-management glue. Four helpers
the orchestrator wires up at the end of `init()` to keep the render
loop in step with page state, retry failed loads when connectivity
returns, and tear every subsystem down in the correct order when the
app — or the host page — goes away.

The render loop itself lives in `../../../scene/animation/` (see
`animation-controller.ts`). This folder does not drive frames; it only
**starts**, **stops**, and **disposes** the controller in response to
external lifecycle signals (window focus, document visibility,
`beforeunload`, explicit `app.dispose()`).

## Files

```
lifecycle/
├── dispose-pipeline.ts   # Ordered, fault-tolerant teardown of every subsystem
├── focus-handling.ts     # window focus + document visibilitychange → animate pause/resume
├── online-retry.ts       # window online → retry every failed loader (failed-load recovery)
└── unload-handling.ts    # window beforeunload → app.dispose()
```

| File                  | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dispose-pipeline.ts` | `runDisposePipeline(ports)` — runs every subsystem's `dispose()` through a `safeDispose` wrapper that catches and logs without bubbling. Component order is the teardown contract; singletons (`ThemeManager`, `DataMonitorManager`, `SceneLoaderManager`, `disposeWorkerPool`) tear down last. `DisposePipelinePorts` declares the snapshot of orchestrator state plus per-field clear callbacks.                                                                                                                                                       |
| `focus-handling.ts`   | `installFocusHandling(ports)` registers `window.focus` and `document.visibilitychange` listeners on the shared `EventGroup`. Both early-return when `getRecordingPanel()?.isCurrentlyRecording()` reports an active capture so offline recording keeps a stable loop.                                                                                                                                                                                                                                                                                    |
| `online-retry.ts`     | `installOnlineRetry(ports)` registers a `window` `online` listener that retries every failed loader when connectivity is restored — the trigger half of the failed-load recovery story (`SceneLoader.retryAllFailedLoaders` is the engine). No failures → silent no-op; deferred batches (update lock held) re-attempt every `DEFERRED_RETRY_DELAY_MS` up to `MAX_DEFERRED_RETRY_ATTEMPTS` times; re-entrancy guarded; `getLoader` is a live accessor so dataset switches don't go stale. Listener + pending timer are owned by the shared `EventGroup`. |
| `unload-handling.ts`  | `installUnloadHandler(ports)` registers a `beforeunload` listener that invokes the supplied `dispose` callback inside a defensive try/catch (unload is terminal — a throw would otherwise bubble to `window.onerror` and block other unload work). The listener is owned by the shared `EventGroup`, so it is removed automatically by `events.dispose()` inside `runDisposePipeline`.                                                                                                                                                                   |

## Animate-tick orchestration

The render loop is driven by `AnimationController.startAnimation()` /
`stopAnimation()`. This folder's helpers translate page lifecycle into
those calls:

| Event                                   | Action                                   | Suppressed during recording? |
| --------------------------------------- | ---------------------------------------- | ---------------------------- |
| `window` `focus`                        | `animationController.startAnimation()`   | yes                          |
| `document` `visibilitychange` → hidden  | `animationController.stopAnimation()`    | yes                          |
| `document` `visibilitychange` → visible | `animationController.startAnimation()`   | yes                          |
| `window` `beforeunload`                 | `dispose()` (which stops the loop first) | n/a                          |

`getRecordingPanel` is passed as a **live accessor**, not a snapshot,
so a focus event that fires mid-dispose — after the orchestrator
cleared its `recordingPanel` field but before `events.dispose()`
removes the listener — sees the updated `undefined` and the
optional-chain short-circuits.

## Dispose flow

`runDisposePipeline` is idempotent and safe to call from partial-init
states (any port may be `undefined`). The order encodes a teardown
contract:

```
1.  animation + adaptive-DPR + resolution-indicator
2.  scaleBar, colormapLegend
3.  pickingEvents                        ← BEFORE overlayManager (HIGH-12 mousemove race)
4.  overlayManager (clears recordingPanel back-ref first)
5.  recordingPanel, layersPanel
6.  pickingSystem → labelLoader, imageLabelLoader
7.  datasetBrowser (close + unbind from inputHandler)
8.  inputHandler                         ← child panels gone before host
9.  renderingControls
10. sceneManager
11. ThemeManager.disposeInstance() + cleanupUI()
12. events.dispose()                     ← all listeners (focus, visibility, beforeunload, …)
13. DataMonitorManager → SceneLoaderManager → disposeWorkerPool()
```

`pickingEvents` is released ahead of `overlayManager` because the
picking system's mousemove handler closes over the overlay manager;
disposing the overlay first would leave a window in which a
synchronously-dispatched mousemove null-derefs the disposed overlay.

A throwing component must NOT skip later cleanup — `safeDispose`
records the failure label and continues, and a final aggregated
`log.error` lists every component that threw. The per-field
`clear*` callbacks (`clearScaleBar`, `clearOverlayManager`, …) let the
helper drop the orchestrator's references without reaching into
`LuxarApp` directly, keeping the pipeline a pure function over its
ports.

## See also

- `../../../scene/animation/README.md` — the `AnimationController`
  these helpers start and stop.
- `../init/pipeline.ts` — builds the subsystems the dispose pipeline
  tears down; the install-helpers here run at the end of `init()`.
- `../../../utils/cross-layer/event-group.ts` — the `EventGroup`
  abstraction every install-helper uses, so listener removal is one
  call (`events.dispose()`) in step 12 of the dispose flow.
