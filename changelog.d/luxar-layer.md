#### `LuxarLayer` — render a Luxar scene inside a host application's own renderer

`LuxarApp` embeds *the viewer*: it owns the renderer, camera, controls,
post-processing, render loop, and UI. That is the right shape when Luxar is the
page. It is the wrong shape when a host application already has a Three.js scene
and wants Luxar's data as one more thing inside it — sharing a single WebGL
context, one camera, and one set of controls. Stacking two canvases and syncing
cameras almost works, but it costs a second GL context, forbids any depth
interaction between the two, and gives the host two post chains to reconcile.

`LuxarLayer` is the headless sibling for that case. The host keeps its pipeline;
the layer contributes a `THREE.Group` plus the per-frame bookkeeping that keeps
streaming, LOD selection, and depth sorting correct. It owns no renderer,
camera, controls, post-processing, or UI.

```ts
import { LuxarLayer } from '@royerlab/luxar-viewer';

const layer = new LuxarLayer({ renderer, getCamera: () => camera, getViewportSize, scene });
await layer.load('https://example.com/imaging.luxar.zarr');
// each frame, before the host renders:
layer.update();
```

No new machinery was needed to make this possible, which is the interesting
part: `SceneLoader.loadScene()` already returns a plain `THREE.Group`, and
`LODGroupRegistryDeps` and `configureDepthSort()` are already defined purely in
terms of injectable getters — `getCamera()`, `getViewportSize()`,
`getDisplayDims()`. Nothing in the data, cache, LOD, or material path reaches for
`SceneManager`. `LuxarLayer` supplies those getters from the host instead, so
all of the renderer/camera/controls/UI coupling stays where it already was, in
`SceneManager` and the app init pipeline.

Two details are worth knowing before wiring a host to it. First, `update()` runs
the depth-sort scheduler *before* the LOD-group selector, because sorting
assigns the cross-node render order that a LOD swap can then invalidate; hosts
call one method and get the right order. Second, `setDimensionValue()`
**coalesces** — a host scrubbing a timeline at frame rate outruns the loader by
roughly 30×, and every call but the last is already stale by the time it could
commit. Callers arriving during an in-flight pass are all served by the next
one, and each is resolved by the pass that actually included its value. The
intended playback pattern is fire-and-forget plus `prefetchDimensionValue()` for
the next tick, so the host's own timeline never stalls on streamed slices.

`dispose()` is async and tears down the loader and its caches, the data-worker
pool, the depth-sort worker, and the material cache — a full teardown of Luxar
in the page. That is consistent with the single-instance rule the layer shares
with `LuxarApp`: the scene-loader manager, dimension manager, material manager,
and worker pool are process singletons, so one layer per page, and never
alongside a `LuxarApp`.
