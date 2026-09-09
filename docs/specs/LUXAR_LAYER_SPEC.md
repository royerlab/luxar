# LuxarLayer Embed Contract

## Scope

`LuxarLayer` is the headless embedding API for adding a Luxar scene to a host application's own
Three.js renderer, camera, controls, scene graph, and render loop. It contributes one
`THREE.Group` plus Luxar's loading, nD slicing, LOD selection, depth sorting, and teardown
bookkeeping. It does not create or own host rendering infrastructure.

The supported public entry point is:

```ts
import { LuxarLayer } from '@luxar/viewer';
```

The runnable reference host is `packages/luxar-viewer/examples/layer/`.

## Ownership and lifecycle

The host owns the renderer, scene, camera, controls, canvas, render loop, post-processing, and UI.
It passes live camera and viewport getters because either can change after construction.

```ts
const layer = new LuxarLayer({
  renderer,
  scene,
  getCamera: () => camera,
  getViewportSize: () => ({ width, height }),
  requestRender: () => {
    frameRequested = true;
  },
});

await layer.load(src);
```

`load()` attaches the returned root to the host scene. A later successful load replaces and
disposes the previous subtree. A failed replacement leaves the layer empty rather than displaying
geometry backed by a disposed loader. Concurrent `load()` calls are rejected.

The host must await `dispose()` before constructing another `LuxarLayer`. Disposal waits for an
active load and dimension update, detaches layer geometry, and releases the process-wide loader,
material, LOD, sorting, and worker resources.

## Public surface

The constructor also accepts loader and asset-path overrides plus `gpuPoolMaxBytes`, `lodFade`,
`lodEnergyComp`, `lodFinest`, and `depthSort` feature switches. `renderOrder` defaults to 10.
`requestRender` is optional only for hosts that render continuously; an on-demand host must use it
to schedule a future frame when progressive geometry, a lazy LOD level, or a retry commits
asynchronously.

| Member | Contract |
| --- | --- |
| `root` | Current layer root, or `null` before load and after disposal. |
| `load(src)` | Load and attach a scene, replacing the previous successful load. |
| `update()` | Run depth sorting and LOD selection before the host renders. |
| `resize()` | Publish the current camera projection and drawing-buffer size. |
| `getDimensions()` | Return cloned dimension state, or `null` when unavailable. |
| `findDimension(name)` | Find a dimension case-insensitively, or return `null`. |
| `getDimensionNames()` | Return names in center-column order. |
| `setDimensionValue(index, value)` | Coalesce and commit a foreground slice update. |
| `prefetchDimensionValue(index, value, budgetMs?)` | Warm a slice without committing it. |
| `awaitDimensionUpdate()` | Resolve when no dimension update remains in flight. |
| `setVisible(visible)` / `isVisible()` | Hide or show without discarding caches. |
| `setExposure(multiplier)` / `getExposure()` | Scale opacity relative to authored values. |
| `getBounds()` | Return world-space bounds for the current root, or `null`. |
| `alignTo(matrix)` | Place the root in host world space before or after `load()`. |
| `onDatasetFault(listener)` | Subscribe to `DatasetFaultPayload` archive faults, replaying the current fault immediately; returns an unsubscribe function and throws after `dispose()`. |
| `getDatasetFault()` | Return the current `DatasetFaultPayload`, or `null` before a fault occurs, once a new `load()` begins, and after disposal. No callback fires when a fault clears, so a host that renders recoverable state must poll this method rather than latch the callback result. |
| `handleContextLost()` / `handleContextRestored()` | Re-arm Luxar-owned GPU resources around host context recovery. |
| `dispose()` | Asynchronously release the layer and process-wide Luxar resources. |

## Per-frame ordering

Every host frame must execute:

```ts
layer.update();
renderer.render(scene, camera);
```

`update()` deliberately runs the depth-sort scheduler before the LOD selector. Sorting establishes
cross-node render ranks; LOD evaluation can then change the visible level without leaving those
ranks stale for the frame. Calling only `renderer.render()` may leave lazy geometry, LOD, and
depth-order state stale.

`requestRender` does not replace `update()`. It wakes an on-demand host after asynchronous work;
the requested frame must still call `update()` before `renderer.render()`.

Call `resize()` after changing the viewport or camera projection. The layer reads the host camera
and viewport but cannot observe those changes itself.

## Dimensions

`getDimensionNames()` returns center-column order; embedders must not assume `x,y,z,time` ordering.
`setDimensionValue()` coalesces rapid requests and resolves after the pass containing that value
commits. Playback should issue foreground updates without awaiting each frame and use
`prefetchDimensionValue()` for the next value.

## Draw order and visibility

The layer stamps its configured `renderOrder` (10 by default) onto every owned `THREE.Group`,
including groups attached lazily. The host remains responsible for assigning compatible orders to
its own transparent groups.

`setVisible(false)` keeps caches and in-flight requests alive. Lazy LOD loads pause while hidden;
resident hidden levels are preferred eviction candidates under GPU pressure.

## Context loss and restore

The host owns the WebGL context event handlers. On loss it must prevent the browser default and
notify the layer; after rebuilding its own renderer and post-processing resources it must notify
the layer again:

```ts
canvas.addEventListener('webglcontextlost', (event) => {
  event.preventDefault();
  layer.handleContextLost();
});

canvas.addEventListener('webglcontextrestored', () => {
  layer.handleContextRestored();
});
```

The loss hook clears shader warm-up state and reduces the GPU budget. The restore hook marks only
the layer subtree dirty and re-arms Luxar-owned resources; it does not restore host resources.

## Single-instance rule

Only one `LuxarLayer` may exist per page, and it must not coexist with `LuxarApp`. The scene-loader,
dimension, material, LOD, and worker managers are process singletons. Multiple owners would share
and tear down each other's state.

## Known limitations

- Scene `viewer_config` values implemented by Luxar UI, camera, or post-processing code are inert:
  tone mapping, global exposure/gamma/offset, bloom, background, camera settings, and related UI.
- The host must provide its own tone mapping. Without a rolloff, overlapping emissive additive
  geometry can clip flat in the framebuffer.
- Near-plane culling is not derived in layer mode; the host should keep its near plane clear of the
  data.
- `setDimensionValue()` before `load()` is a no-op because dimension metadata comes from the scene.
- Default opaque mesh materials interpret opacity as alpha-cutout coverage rather than a smooth
  dimmer; use normal blending when smooth transparency is required.
- KTX2 mesh textures decode through a renderer-owned decoder installed by the layer. Other host
  texture and renderer resources remain the host's responsibility.
- The UI-layer `notifier` is not registered, so Luxar toasts and error overlays are silently
  dropped. Hosts must provide their own user-facing status and error surface. Archive failures
  remain observable through `onDatasetFault()` and `getDatasetFault()`.

## Coverage note

The runnable example's LOD E2E fixture exercises the legacy `coverage` selector path. The current
`screen-area` selector used by newly authored ladders requires separate fixture coverage.
