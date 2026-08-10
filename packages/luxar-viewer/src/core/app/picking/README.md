# App-Level Picking Wiring

> Stands up `rendering/picking/PickingSystem` for the current scene and routes its results into the hover overlay. The pure GPU work lives in `rendering/picking/`; this folder is the thin glue that hooks it into `LuxarApp`'s lifecycle, pointer events, and `OverlayManager`.

## Overview

`LuxarApp.loadDataset()` calls `initPicking()` on every scene load. The function decides whether picking is needed (any node with `userData.attrs.has_labels` or `has_image_labels`, or an embedder `selection` listener existing at load time), tears down any prior session, constructs a fresh `PickingSystem` with a result callback built by `buildPickResultHandler`, and wires the DOM + Three.js EventDispatcher listeners that drive it. The handler closures and the listener wire-up are intentionally split so the branch logic can be unit-tested with stub ports — no `WebGLRenderer`, no zarr store, no real `PickingSystem`.

## File Structure

```
picking/
├── init-picking.ts          # initPicking(ports) — session lifecycle + event wiring
└── pick-result-handler.ts   # buildPickResultHandler(ports) — PickResult → hover payload
```

## `init-picking.ts`

`initPicking({ sceneManager, pickingEvents, previous, getOverlayManager, onSelection?, hasSelectionConsumer? })` returns `{ pickingSystem, labelLoader, imageLabelLoader }` (all `undefined` when picking is intentionally inactive). The flow:

1. **Teardown** — `pickingEvents.dispose()` plus `dispose()` on the previous `PickingSystem` / `LabelLoader` / `ImageLabelLoader`, so a dataset switch never leaks state.
2. **Label detection** — walks the `LuxarScene` group looking for `userData.attrs.has_labels` / `has_image_labels`. No labels **and** no embedder `selection` listener (`hasSelectionConsumer()`) → return three `undefined`s, picking stays off for this session. A selection listener alone still provisions picking; the loaders below are built only for the flags that were actually found.
3. **Loaders** — pulls the active scene loader from `getSceneLoader('default')` and constructs `LabelLoader` / `ImageLabelLoader` against its `zarrStore` + `zarr.root(store)`.
4. **System** — `new PickingSystem(renderer, capabilities, camera, buildPickResultHandler({ labelLoader, imageLabelLoader, overlayManager: getOverlayManager(), onSelection }))` — the `onSelection` port is the sink for the public `selection` embedder event.
5. **NodeFactory hookup** — `sceneLoader.nodeFactory.setPickingSystem(pickingSystem)` so future node loads get pick materials; `registerExistingSceneNodes(root)` retroactively registers the already-loaded nodes (scene loads before picking init).
6. **Post-processing hookup** — `pickingSystem.setPostProcessing(...)` so the lens-distortion port stays in sync with the visible frame.
7. **Pick gating** — `pickingSystem.setShouldPick(() => (getOverlayManager()?.hasVisibleHoverOverlay() ?? false) || (hasSelectionConsumer?.() ?? false))` — a disjunction: picks run while there is a visible hover overlay (tooltips) **or** a live embedder `selection` listener, so a label-less scene (which auto-injects no hover overlay) still picks for selection. Both sides are read LIVE, so unsubscribing stops the pick renders without re-initialising the pipeline. No consumer, no work.
8. **Event wiring** — all listeners are registered through the shared `EventGroup` so a single `pickingEvents.dispose()` removes them:

| Source                                    | Event                       | Action                                                                                                                                                                    |
| ----------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renderer.domElement`                     | `mousemove` (passive)       | `pickingSystem.onMouseMove(e)`                                                                                                                                            |
| `renderer.domElement`                     | `mouseleave` (passive)      | `pickingSystem.onMouseLeave()` — drops pending cursor                                                                                                                     |
| `sceneManager.controls` (EventDispatcher) | `change`                    | `pickingSystem.markDirty()`                                                                                                                                               |
| `sceneManager.controls`                   | `start`                     | `pickingSystem.suppress(true)` + `overlayManager.updateHoverContent(null)`                                                                                                |
| `sceneManager.controls`                   | `end`                       | `pickingSystem.suppress(false)` — camera-settle re-pick re-arms naturally                                                                                                 |
| `window`                                  | `resize`                    | `pickingSystem.markDirty()`                                                                                                                                               |
| `window`                                  | `scroll` (capture, passive) | `pickingSystem.invalidateCanvasRect()` — page scroll moves the canvas on screen without changing the view, so bust just the cached rect (cheap) rather than `markDirty()` |
| `sceneManager` (EventDispatcher)          | `camera-changed`            | `pickingSystem.setCamera(...)` for perspective ↔ ortho swaps                                                                                                              |

Three.js EventDispatcher sources are registered via `pickingEvents.add(() => …removeEventListener)` since their signatures don't match `EventTarget`.

## `pick-result-handler.ts`

`buildPickResultHandler({ labelLoader, imageLabelLoader, overlayManager })` returns the `(result: PickResult | null) => Promise<void>` callback handed to `PickingSystem`. Branch contract:

- `null` result → `overlayManager.updateHoverContent(null)`; no loader calls.
- Non-null → `Promise.all` on `labelLoader.getLabel(lookupPath, elementId)` and `imageLabelLoader.getImageUrl(lookupPath, elementId)`. Emit `{ label, imageUrl, nodeName: reportPath, elementIndex }` only when at least one is truthy; otherwise clear hover.
- Either fetch rejects → `log.warning` and clear hover. Errors must not kill the hover loop.

**Partition-aware node path — reported vs queried.** The handler resolves two paths, and they differ under a partition:

| Path         | Value                                             | Used for                                            |
| ------------ | ------------------------------------------------- | --------------------------------------------------- |
| `reportPath` | outermost `kind=partition` wrapper, else the leaf | selection `nodeName`, overlay title                 |
| `lookupPath` | the hit leaf scene node, `result.mainNode.name`   | `getLabel` / `getImageUrl`, selection `hitNodeName` |

`findOutermostPartitionWrapperName(result.mainNode)` walks the hit leaf's parent chain and returns the `name` of the **outermost** ancestor whose `userData.kind === 'partition'`. Reporting that wrapper mirrors how the layers panel treats a `kind=partition` wrapper — the wrapper is the layer the user sees — so a hit inside a nested `kind=partition` group names the topmost wrapper rather than the inner `part_<i>`. It is not a general outermost-ancestor rule: `ui/layers/layer-state.ts` also admits `kind=lod` wrappers as layers, and those are deliberately not matched, so a substitutive-LOD layer still reports its internal level path.

The **lookup** must not follow it (#1415). `lookupPath` is the hit leaf _scene node_, which is the CSR owner both for a flat node and for a `part_<i>` of a partition. A partition wrapper is a bare group: `add_points` / `add_lines` / `add_gsplats` / `add_mesh` slice `labels` per part and write the CSR (`label_offsets` / `label_bytes`) on each `part_<i>` leaf. And `result.elementId` is an index in that leaf's own element space, so it only indexes the part's array anyway. Querying the wrapper missed on both counts and left every hover on a partitioned layer with an empty tooltip — silently, since `LabelLoader` demotes the absent array to an info log and caches `[]`. Of the two lookups only `getLabel` is reachable under a partition today — all four adders refuse `image_labels` alongside `partition=`, so no `part_<i>` ever owns an image CSR — and `getImageUrl` moves to `lookupPath` for consistency (and so it is already right if that combination is ever allowed), not because it is broken today.

One known limit survives this fix, plus a narrowed caveat on the element id — neither fixable by the path choice:

- **Additive ladders.** A progressive (additive) node writes its CSR per `additive_<i>` sub-group, which is not a scene-graph node, while `points-progressive-loader.ts` concatenates every loaded level into one committed buffer. No single path can index that buffer, so labels on a laddered node are unusable regardless of `lookupPath` — a producer-side gap (#1422).
- **Element id vs on-disk index — now resolved where the node can resolve it.** `result.elementId` reaches the handler already translated into the on-disk element index the CSR is keyed by, wherever the node can resolve one: `resolveOnDiskElementId` (`rendering/picking/picking-system/element-id-map.ts`) applies the published slot → on-disk map at the single `PickResult` construction site, and where the identity already holds no map is published and the raw slot already _is_ the on-disk index. Points, GSplats and Lines all resolve, for a node declaring `has_labels` / `has_image_labels`; Mesh needs none, since its `gl_VertexID` already is the on-disk vertex ordinal. Lines resolves through the longest chain and to a different granularity than the slot it started from (#1424): its pick shader reports a visible **segment** slot while line labels are per-**vertex**, so the resolved value is the picked segment's **start** vertex row in the on-disk (spatially sorted) vertex ordering — a segment carries a single `flat` pick id, so exactly one of its two endpoints can be reported. Otherwise the id stays a _visible-buffer storage slot_ and diverges from the on-disk index as soon as a hidden dimension culls chunks (the loaded ranges no longer start at 0) or triggers effective-radius compaction (`data/points/projection.ts`). That now leaves label-less nodes: a label-less Points / GSplats node has no CSR to miss, but an embedder reading `SelectionPayload.elementIndex` on one is reading a slot — and on a label-less Lines node it is a per-segment slot against a per-vertex CSR whatever the slicing.

**Stale-drop ordering.** A label/image fetch is async, so a slow fetch from an older cursor position could resolve after a newer one and re-show a stale tooltip. A monotonic `latest` token, captured per call as `seq`, gates the post-`await` emit: a superseded invocation (`seq !== latest`) drops its result — both on the success path and in the `catch`. A `null` (hide) call applies immediately and bumps the token, so it also cancels any in-flight content fetch.

Ports are narrow structural interfaces (just `getLabel` / `getImageUrl` / `updateHoverContent`) so tests can stub with plain `vi.fn()`s.

## See Also

- [`../README.md`](../README.md) — `LuxarApp`'s private support tree (this folder lives under `core/app/`)
- [`../../../rendering/picking/README.md`](../../../rendering/picking/README.md) — the GPU picking subsystem this folder wires up
- [`../../../rendering/picking/picking-system/README.md`](../../../rendering/picking/picking-system/README.md) — pure helpers (ray-AABB, vote, settle, lens-distortion) behind `PickingSystem`
- `../../../ui/overlay-manager.ts` — `updateHoverContent` + `hasVisibleHoverOverlay` consumers
- `../../../data/loaders/picking/label-loader.ts` / `picking/image-label-loader.ts` — zarr-backed label fetchers
- `../../../utils/cross-layer/event-group.ts` — `EventGroup` used for bulk listener teardown
