# App-Level Picking Wiring

> Stands up `rendering/picking/PickingSystem` for the current scene and routes its results into the hover overlay. The pure GPU work lives in `rendering/picking/`; this folder is the thin glue that hooks it into `LuxarApp`'s lifecycle, pointer events, and `OverlayManager`.

## Overview

`LuxarApp.loadDataset()` calls `initPicking()` on every scene load. The function decides whether picking is needed (any node with `userData.attrs.has_labels` or `has_image_labels`), tears down any prior session, constructs a fresh `PickingSystem` with a result callback built by `buildPickResultHandler`, and wires the DOM + Three.js EventDispatcher listeners that drive it. The handler closures and the listener wire-up are intentionally split so the branch logic can be unit-tested with stub ports — no `WebGLRenderer`, no zarr store, no real `PickingSystem`.

## File Structure

```
picking/
├── init-picking.ts          # initPicking(ports) — session lifecycle + event wiring
└── pick-result-handler.ts   # buildPickResultHandler(ports) — PickResult → hover payload
```

## `init-picking.ts`

`initPicking({ sceneManager, pickingEvents, previous, getOverlayManager })` returns `{ pickingSystem, labelLoader, imageLabelLoader }` (all `undefined` when picking is intentionally inactive). The flow:

1. **Teardown** — `pickingEvents.dispose()` plus `dispose()` on the previous `PickingSystem` / `LabelLoader` / `ImageLabelLoader`, so a dataset switch never leaks state.
2. **Label detection** — walks the `LuxarScene` group looking for `userData.attrs.has_labels` / `has_image_labels`. No labels → return three `undefined`s, picking stays off for this session.
3. **Loaders** — pulls the active scene loader from `getSceneLoader('default')` and constructs `LabelLoader` / `ImageLabelLoader` against its `zarrStore` + `zarr.root(store)`.
4. **System** — `new PickingSystem(renderer, capabilities, camera, buildPickResultHandler({ labelLoader, imageLabelLoader, overlayManager: getOverlayManager() }))`.
5. **NodeFactory hookup** — `sceneLoader.nodeFactory.setPickingSystem(pickingSystem)` so future node loads get pick materials; `registerExistingSceneNodes(root)` retroactively registers the already-loaded nodes (scene loads before picking init).
6. **Post-processing hookup** — `pickingSystem.setPostProcessing(...)` so the lens-distortion port stays in sync with the visible frame.
7. **Pick gating** — `pickingSystem.setShouldPick(() => getOverlayManager()?.hasVisibleHoverOverlay() ?? false)` — no consumer, no work.
8. **Event wiring** — all listeners are registered through the shared `EventGroup` so a single `pickingEvents.dispose()` removes them:

| Source                                    | Event                  | Action                                                                     |
| ----------------------------------------- | ---------------------- | -------------------------------------------------------------------------- |
| `renderer.domElement`                     | `mousemove` (passive)  | `pickingSystem.onMouseMove(e)`                                             |
| `renderer.domElement`                     | `mouseleave` (passive) | `pickingSystem.onMouseLeave()` — drops pending cursor                      |
| `sceneManager.controls` (EventDispatcher) | `change`               | `pickingSystem.markDirty()`                                                |
| `sceneManager.controls`                   | `start`                | `pickingSystem.suppress(true)` + `overlayManager.updateHoverContent(null)` |
| `sceneManager.controls`                   | `end`                  | `pickingSystem.suppress(false)` — camera-settle re-pick re-arms naturally  |
| `window`                                  | `resize`               | `pickingSystem.markDirty()`                                                |
| `window`                                  | `scroll` (capture, passive) | `pickingSystem.invalidateCanvasRect()` — page scroll moves the canvas on screen without changing the view, so bust just the cached rect (cheap) rather than `markDirty()` |
| `sceneManager` (EventDispatcher)          | `camera-changed`       | `pickingSystem.setCamera(...)` for perspective ↔ ortho swaps               |

Three.js EventDispatcher sources are registered via `pickingEvents.add(() => …removeEventListener)` since their signatures don't match `EventTarget`.

## `pick-result-handler.ts`

`buildPickResultHandler({ labelLoader, imageLabelLoader, overlayManager })` returns the `(result: PickResult | null) => Promise<void>` callback handed to `PickingSystem`. Branch contract:

- `null` result → `overlayManager.updateHoverContent(null)`; no loader calls.
- Non-null → resolve `nodePath`, then `Promise.all` on `labelLoader.getLabel(nodePath, elementId)` and `imageLabelLoader.getImageUrl(nodePath, elementId)`. Emit `{ label, imageUrl, nodeName, elementIndex }` only when at least one is truthy; otherwise clear hover.
- Either fetch rejects → `log.warning` and clear hover. Errors must not kill the hover loop.

**Partition-aware node path.** `findOutermostPartitionWrapperName(result.mainNode)` walks the hit leaf's parent chain and returns the `name` of the **outermost** ancestor whose `userData.kind === 'partition'`. When present that wrapper name (= zarr path) is used as `nodePath`; otherwise the handler falls back to `result.mainNode.name`. This matches the layers-panel's outermost-as-layer convention, so a hit inside a nested `kind=partition` group reports the topmost wrapper rather than the inner `part_<i>`.

**Stale-drop ordering.** A label/image fetch is async, so a slow fetch from an older cursor position could resolve after a newer one and re-show a stale tooltip. A monotonic `latest` token, captured per call as `seq`, gates the post-`await` emit: a superseded invocation (`seq !== latest`) drops its result — both on the success path and in the `catch`. A `null` (hide) call applies immediately and bumps the token, so it also cancels any in-flight content fetch.

Ports are narrow structural interfaces (just `getLabel` / `getImageUrl` / `updateHoverContent`) so tests can stub with plain `vi.fn()`s.

## See Also

- [`../README.md`](../README.md) — `LuxarApp`'s private support tree (this folder lives under `core/app/`)
- [`../../../rendering/picking/README.md`](../../../rendering/picking/README.md) — the GPU picking subsystem this folder wires up
- [`../../../rendering/picking/picking-system/README.md`](../../../rendering/picking/picking-system/README.md) — pure helpers (ray-AABB, vote, settle, lens-distortion) behind `PickingSystem`
- `../../../ui/overlay-manager.ts` — `updateHoverContent` + `hasVisibleHoverOverlay` consumers
- `../../../data/loaders/picking/label-loader.ts` / `picking/image-label-loader.ts` — zarr-backed label fetchers
- `../../../utils/cross-layer/event-group.ts` — `EventGroup` used for bulk listener teardown
