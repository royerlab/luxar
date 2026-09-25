# App-Level Picking Wiring

> Stands up `rendering/picking/PickingSystem` for the current scene and routes its results into the hover overlay. The pure GPU work lives in `rendering/picking/`; this folder is the thin glue that hooks it into `LuxarApp`'s lifecycle, pointer events, and `OverlayManager`.

## Overview

`LuxarApp.loadDataset()` calls `initPicking()` on every scene load. The function decides whether picking is needed (any node with `userData.attrs.has_labels`, `has_label_ids`, `has_keys`, `has_image_labels`, `link`, or `copy`, or an embedder selection/element-action listener existing at load time), tears down any prior session, constructs a fresh `PickingSystem` with a result callback built by `buildPickResultHandler`, and wires the DOM + Three.js EventDispatcher listeners that drive it. The handler closures and the listener wire-up are intentionally split so the branch logic can be unit-tested with stub ports — no `WebGLRenderer`, no zarr store, no real `PickingSystem`.

## File Structure

```
picking/
├── init-picking.ts          # initPicking(ports) — session lifecycle + event wiring
└── pick-result-handler.ts   # buildPickResultHandler(ports) — PickResult → hover payload
```

The click/context-menu half of the same session lives one folder over, in
[`../interaction/`](../interaction) — `initPicking` constructs it and registers
its listeners through the same `EventGroup`, so one `pickingEvents.dispose()`
still tears the whole session down.

## `init-picking.ts`

`initPicking({ sceneManager, pickingEvents, previous, getOverlayManager, onSelection?, hasSelectionConsumer? })` returns `{ pickingSystem, labelLoader, keyLoader, imageLabelLoader }` (all `undefined` when picking is intentionally inactive). The flow:

1. **Teardown** — `pickingEvents.dispose()` plus `dispose()` on the previous `PickingSystem` / label, key, and image-label loaders, so a dataset switch never leaks state.
2. **Consumer detection** — walks the `LuxarScene` group looking for `userData.attrs.has_labels` / `has_label_ids` / `has_keys` / `has_image_labels` and `link` / `copy` templates. No channels/templates **and** no embedder selection or element-action listener → return four `undefined`s, picking stays off for this session. A listener alone still provisions picking; the loaders below are built only for the flags that were actually found.
3. **Loaders** — pulls the active scene loader from `getSceneLoader('default')` and constructs label/key `LabelLoader` instances plus `ImageLabelLoader` against its `zarrStore` + `zarr.root(store)`.
4. **System** — `new PickingSystem(renderer, capabilities, camera, buildPickResultHandler({ labelLoader, keyLoader, imageLabelLoader, overlayManager: getOverlayManager(), onSelection }))` — the `onSelection` port is the sink for the public `selection` embedder event.
5. **NodeFactory hookup** — `sceneLoader.nodeFactory.setPickingSystem(pickingSystem)` so future node loads get pick materials; `registerExistingSceneNodes(root)` retroactively registers the already-loaded nodes (scene loads before picking init).
6. **Post-processing hookup** — `pickingSystem.setPostProcessing(...)` so the lens-distortion port stays in sync with the visible frame.
7. **Pick gating** — picks run while there is a visible hover overlay, a scene interaction template, or a live embedder selection/element-action listener. Listener and overlay predicates are read live; scene templates are constant for the loaded session. No consumer, no work.
8. **Event wiring** — all listeners are registered through the shared `EventGroup` so a single `pickingEvents.dispose()` removes them:

| Source                           | Event                       | Action                                                                                                                                                                                                                                                                                                                                                   |
| -------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renderer.domElement`            | `mousemove` (passive)       | `pickingSystem.onMouseMove(e)`                                                                                                                                                                                                                                                                                                                           |
| `renderer.domElement`            | `mouseleave` (passive)      | `pickingSystem.onMouseLeave()` — drops pending cursor                                                                                                                                                                                                                                                                                                    |
| `sceneManager` (EventDispatcher) | `change`                    | `pickingSystem.markDirty()` after camera pose or projection changes; FOV and static near/far updates both change a fresh pick render and must dispatch this event. The per-frame dynamic near/far path deliberately stays silent because its triggering camera/geometry change already dirties picking, and dispatching every frame would keep rAF awake |
| `sceneManager.controls`          | `start`                     | `pickingSystem.suppress(true)` + `overlayManager.updateHoverContent(null)`                                                                                                                                                                                                                                                                               |
| `sceneManager.controls`          | `end`                       | `pickingSystem.suppress(false)` — camera-settle re-pick re-arms naturally                                                                                                                                                                                                                                                                                |
| `window`                         | `resize`                    | `pickingSystem.markDirty()`                                                                                                                                                                                                                                                                                                                              |
| `window`                         | `scroll` (capture, passive) | `pickingSystem.invalidateCanvasRect()` — page scroll moves the canvas on screen without changing the view, so bust just the cached rect (cheap) rather than `markDirty()`                                                                                                                                                                                |
| `sceneManager` (EventDispatcher) | `camera-changed`            | `pickingSystem.setCamera(...)` for perspective ↔ ortho swaps                                                                                                                                                                                                                                                                                             |

Three.js EventDispatcher sources are registered via `pickingEvents.add(() => …removeEventListener)` since their signatures don't match `EventTarget`.

## `pick-result-handler.ts`

`buildPickResultHandler({ labelLoader, keyLoader, imageLabelLoader, overlayManager })` returns the `(result: PickResult | null) => Promise<void>` callback handed to `PickingSystem`. Branch contract:

- `null` result → `overlayManager.updateHoverContent(null)`; no loader calls.
- Non-null → `Promise.all` on `labelLoader.getLabel(lookupPath, elementId)`, `keyLoader.getLabel(lookupPath, elementId)`, and `imageLabelLoader.getImageUrl(lookupPath, elementId)`. When no baked string label exists, a GSplats `label_ids` channel supplies `name (exact-id)` from the visible storage slot. Emit `{ label, key, imageUrl, nodeName: reportPath, elementIndex }` only when at least one is truthy; otherwise clear hover.
- Any fetch rejects → `log.warning` and clear hover. Errors must not kill the hover loop.

**Partition-aware node path — reported vs queried.** The handler resolves two paths, and they differ under a partition:

| Path         | Value                                             | Used for                                                      |
| ------------ | ------------------------------------------------- | ------------------------------------------------------------- |
| `reportPath` | outermost `kind=partition` wrapper, else the leaf | selection `nodeName`, overlay title                           |
| `lookupPath` | the hit leaf scene node, `result.mainNode.name`   | label/key `getLabel` / `getImageUrl`, selection `hitNodeName` |

`findOutermostPartitionWrapperName(result.mainNode)` walks the hit leaf's parent chain and returns the `name` of the **outermost** ancestor whose `userData.kind === 'partition'`. Reporting that wrapper mirrors how the layers panel treats a `kind=partition` wrapper — the wrapper is the layer the user sees — so a hit inside a nested `kind=partition` group names the topmost wrapper rather than the inner `part_<i>`. It is not a general outermost-ancestor rule: `ui/layers/layer-state.ts` also admits `kind=lod` wrappers as layers, and those are deliberately not matched, so a substitutive-LOD layer still reports its internal level path.

The **lookup** must not follow it (#1415). `lookupPath` is the hit leaf _scene node_, which is the CSR owner both for a flat node and for a `part_<i>` of a partition. A partition wrapper is a bare group: `add_points` / `add_lines` / `add_gsplats` / `add_mesh` slice `labels` / `keys` per part and write each CSR on the `part_<i>` leaf. And `result.elementId` is an index in that leaf's own element space, so it only indexes the part's array anyway. Querying the wrapper missed on both counts and left every hover on a partitioned layer with an empty tooltip — silently, since `LabelLoader` demotes the absent array to an info log and caches the missing node. Of the three lookups only the label/key `getLabel` calls are reachable under a partition today — all four adders refuse `image_labels` alongside `partition=`, so no `part_<i>` ever owns an image CSR — and `getImageUrl` moves to `lookupPath` for consistency (and so it is already right if that combination is ever allowed), not because it is broken today.

One narrowed caveat on the element id remains, plus a partial one on ladders — neither fixable by the path choice:

- **Additive ladders.** A progressive (additive) node carries ONE union CSR per present labels/keys channel on its parent node (#1422), spanning the levels in `additive_<i>` order — which is the same node `lookupPath` names, and the same space `points-progressive-loader.ts` produces when it concatenates the committed levels. So `lookupPath` indexes it correctly, and for **Points** it now does so under nD slicing too: `loader-factory.ts` derives CSR-style `levelOffsets` from the levels' on-disk `n_points` and the progressive loader composes each level's slot → on-disk map into that union index space, bounding every composed id to its own level's rows and refusing the whole map on any inconsistency (#1439). Refusing is not suppression — `resolveOnDiskElementId` then returns the raw slot, which on a sliced ladder can itself be a wrong CSR row; what it buys is narrower and exact: never an id composed out of data known to be inconsistent, so never worse than before. Not for Lines: its raw slot is a per-**segment** one while each union CSR is per-**vertex** (#1424), so a laddered lines node with a string channel hovers at the wrong granularity whatever the slicing, not merely at a shifted offset — and nothing composes the LEVELS of a lines ladder either. Gsplat ladders carry no labels/keys at all.
- **Element id vs on-disk index — now resolved where the node can resolve it.** `result.elementId` reaches the handler already translated into the on-disk element index the CSR is keyed by, wherever the node can resolve one: `resolveOnDiskElementId` (`rendering/picking/picking-system/element-id-map.ts`) applies the published slot → on-disk map at the single `PickResult` construction site, and where the identity already holds no map is published and the raw slot already _is_ the on-disk index. Points, GSplats and Lines all resolve, for a node declaring `has_labels` / `has_image_labels` / `has_keys`; Mesh needs none, since its `gl_VertexID` already is the on-disk vertex ordinal. Lines resolves through the longest chain and to a different granularity than the slot it started from (#1424): its pick shader reports a visible **segment** slot while line string channels are per-**vertex**, so the resolved value is the picked segment's **start** vertex row in the on-disk (spatially sorted) vertex ordering — a segment carries a single `flat` pick id, so exactly one of its two endpoints can be reported. Otherwise the id stays a _visible-buffer storage slot_ and diverges from the on-disk index as soon as a hidden dimension culls chunks (the loaded ranges no longer start at 0) or triggers effective-radius compaction (`data/points/projection.ts`). That now leaves nodes without per-element string channels: a Points / GSplats node without one has no CSR to miss, but an embedder reading `SelectionPayload.elementIndex` there is reading a slot — and on a Lines node without one it is a per-segment slot in what is a per-vertex element space whatever the slicing.

**Stale-drop ordering.** A string/image fetch is async, so a slow fetch from an older cursor position could resolve after a newer one and re-show a stale tooltip. A monotonic `latest` token, captured per call as `seq`, gates the post-`await` emit: a superseded invocation (`seq !== latest`) drops its result — both on the success path and in the `catch`. A `null` (hide) call applies immediately and bumps the token, so it also cancels any in-flight content fetch.

Ports are narrow structural interfaces (just `getLabel` / `getImageUrl` / `updateHoverContent`) so tests can stub with plain `vi.fn()`s.

## Click actions on the picked element (#1917)

Hovering names an element; clicking it can also _do_ something. The wiring for
that is `initPicking`'s third responsibility, and it is deliberately built on
the hover pick rather than on a pick of its own:

- **`../interaction/picked-element-cache.ts`** — retains the settled pick,
  stamped with `pickingSystem.pickGeneration` + `visibleSignature`, and refuses
  to hand it back once either has moved (or the cursor has left a 4 px radius).
  Fed by the handler's `onPicked` port, which fires on exactly the same terms
  as `onSelection` — what is picked, not what has a tooltip.
- **`../interaction/element-actions.ts`** — pure resolution of the node's
  `link` / `copy` / `link_target` attrs into a safe URL and a copy string.
- **`../interaction/canvas-actions.ts`** — the DOM half: `pointerdown` /
  `pointerup` / `contextmenu` on the canvas, the shared `openContextMenu`, the
  clipboard, the pointer cursor.

Why a cache and not a fresh pick on click: there is no synchronous
"pick at (x, y)", and an `await` on a GPU readback risks spending the browser's
transient user activation, which turns "open the link" into "popup blocked".
Acting on the pick the tooltip is already showing is both synchronous and more
honest about what the user chose.

`PickingSystem.suppress()` deliberately does NOT advance `pickGeneration`,
because pointerdown → controls `start` → `suppress(true)` is the first half of
an ordinary click; if it did, every click would refuse itself. Tests pin both
that and the positive invalidators.

Provisioning note: a node declaring `link` / `copy` counts as a picking
consumer in its own right, both in the early-return gate and in `shouldPick`.
Without that, a layer with a link but no labels (a link built from
`{hover_index}` is perfectly usable) would never pick, and the link would
silently never fire.

## See Also

- [`../README.md`](../README.md) — `LuxarApp`'s private support tree (this folder lives under `core/app/`)
- [`../../../rendering/picking/README.md`](../../../rendering/picking/README.md) — the GPU picking subsystem this folder wires up
- [`../../../rendering/picking/picking-system/README.md`](../../../rendering/picking/picking-system/README.md) — pure helpers (ray-AABB, vote, settle, lens-distortion) behind `PickingSystem`
- [`../interaction/`](../interaction) — the click / context-menu half of a picking session
- `../../../ui/overlay-manager.ts` — `updateHoverContent` + `hasVisibleHoverOverlay` consumers
- `../../../utils/hover-template.ts` — the `{hover_*}` vocabulary, shared by the tooltip, `link` and `copy`
- `../../../data/loaders/picking/label-loader.ts` / `picking/image-label-loader.ts` — zarr-backed label fetchers
- `../../../utils/cross-layer/event-group.ts` — `EventGroup` used for bulk listener teardown
