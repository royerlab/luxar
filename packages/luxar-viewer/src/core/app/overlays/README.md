# core/app/overlays

Screen-space overlay subsystems for `LuxarApp`. These helpers stand up and tear down the three non-3D HUD layers — `OverlayManager` (zarr-driven annotations + hover tooltips), `ScaleBar`, and `ColormapLegend` — each on its own lifecycle hook off `loadDataset()` / `dispose()`. They are the only files in `app/` that touch `ui/overlay-manager`, `ui/scale-bar`, and `ui/colormap-legend`.

## Files

```
overlays/
├── init-overlays.ts          # Build OverlayManager + load zarr overlay_groups
├── init-scale-bar.ts         # Build ScaleBar + register per-frame update
├── init-colormap-legend.ts   # Build ColormapLegend from the LayersPanel state
└── dispose-overlays.ts       # Tear down OverlayManager
```

Every helper takes a `Ports` interface (explicit dependencies, no globals) and returns the new instance for `LuxarApp` to store on its own field. Helpers never mutate orchestrator state directly — callbacks (`onDisposed`, `disposePrevious`) keep ownership with `LuxarApp`.

## How it hooks into the app

`LuxarApp.loadDataset()` (in `../dataset/load-dataset.ts`) runs these helpers in a fixed order after the scene is parsed:

1. `disposeOverlays({ manager, onDisposed })` — clear the previous dataset's `OverlayManager` up front so a second `loadDataset()` call doesn't leak annotations. The helper calls `manager.dispose()` and then the `onDisposed` callback, which the orchestrator uses to clear its own field reference.
2. `initScaleBar({ previous, sceneManager, animationController, inputHandler })` — reads `sceneManager.camera` / `.controls` / `.renderer.domElement`, constructs the bar with `config.ui.scaleBar.targetWidthPx` + `position`, and registers a per-frame callback under the key `'scale-bar'` on the `AnimationController` so the bar tracks live camera changes. The previous instance's callback is removed via `removePerFrameCallback('scale-bar')` before disposal.
3. `initColormapLegend({ previous, layersPanel, inputHandler })` — built after the `LayersPanel` exists so the legend can subscribe to `layersPanel.layerState`. Returns `undefined` when no `LayersPanel` is set.
4. `initOverlays({ disposePrevious, sceneManager, inputHandler, recordingPanel })` — looks up the `LuxarScene` group in `sceneManager.scene.children`, pulls `overlayConfigs` + `zarrBaseUrl` from its `userData`, and asynchronously calls `manager.loadOverlays(overlayConfigs, zarrBaseUrl)` when both are present.

Each helper is **idempotent**: a `previous` / `disposePrevious` port disposes the prior instance before constructing the new one, so reloading a dataset (or calling the helper twice) is safe. `initOverlays` is defensive about this even though `loadDataset()` already calls `disposeOverlays()` first.

## Z-stack and rendering model

These overlays are HTML/CSS layers appended to `document.body`, **not** WebGL geometry. They sit above the canvas in DOM stacking order; the helpers here don't manage z-index themselves — `ScaleBar` and `ColormapLegend` are absolute-positioned widgets owned by `ui/`, and `OverlayManager` injects its annotation DOM via `ui/overlay-manager`. The 3D scene continues rendering normally underneath; only `ScaleBar` participates in the animation loop (its `update()` reads camera state each frame to recompute the bar length).

## Wiring to the rest of the app

Every helper plugs its instance into the `InputHandler` so the existing keyboard toggles keep working across reloads:

- `inputHandler.setOverlayManager(manager)` — owned by `initOverlays()`
- `inputHandler.setScaleBar(scaleBar)` — owned by `initScaleBar()`
- `inputHandler.setColormapLegend(legend)` — owned by `initColormapLegend()`

`initOverlays()` additionally calls `recordingPanel?.setOverlayManager(manager)` when a recording panel exists, so the recording subsystem can mirror the same annotations.

## Failure modes

- `initColormapLegend()` wraps the constructor in `try/catch` and returns `undefined` if it throws (`ColormapLegend` requires DOM access; the call site explicitly tolerates failure in test / headless environments).
- `initOverlays()` always returns a manager: when the scene has no `LuxarScene` group, no `overlayConfigs`, or no `zarrBaseUrl`, it builds an empty `OverlayManager` and skips `loadOverlays()` rather than throwing.
- `disposeOverlays()` is a no-op when `manager` is `undefined`, so calling it before any dataset has loaded is safe.

## Managed elsewhere

Other on-screen overlays in the viewer — `HelpOverlay`, `ErrorOverlay`, `DatasetBrowser`, layers panel, recording panel — are **not** managed by this folder. They are constructed directly by `LuxarApp` or by other helpers in `core/app/` and follow their own lifecycle (typically the `UIComponent` base in `ui/overlay-widgets/`). This folder is scoped to the three scene-dependent HUD overlays whose lifetime tracks `loadDataset()`.

## Design note: ports-and-callbacks

Each helper accepts a small `Ports` interface (e.g. `InitScaleBarPorts`, `DisposeOverlaysPorts`) listing only the dependencies it actually reads. State ownership stays on `LuxarApp`: the helper returns the new instance, and the orchestrator assigns it onto its own field. For disposal the helper takes a `previous` instance (or a `disposePrevious` thunk) and an `onDisposed` callback rather than reaching back into the caller. This pattern keeps each helper unit-testable without spinning up the full `LuxarApp` and matches the convention used across the rest of `app/`.

## See also

- [`../dataset/load-dataset.ts`](../dataset/load-dataset.ts) — orchestrator that calls these helpers in sequence
- [`../picking/pick-result-handler.ts`](../picking/pick-result-handler.ts) — feeds hover content into the `OverlayManager` built here
- [`../../../ui/overlay-manager.ts`](../../../ui/overlay-manager.ts) — annotation overlay implementation
- [`../../../ui/scale-bar.ts`](../../../ui/scale-bar.ts) — scale-bar widget
- [`../../../ui/colormap-legend.ts`](../../../ui/colormap-legend.ts) — colormap-legend widget
