# `core/layer` — Layer mode

> Render a Luxar scene inside a **host application's own** Three.js renderer,
> camera, and scene graph.

## What this is

`LuxarLayer` is the headless sibling of `LuxarApp`.

|                        | `LuxarApp`                                    | `LuxarLayer`                          |
| ---------------------- | --------------------------------------------- | ------------------------------------- |
| Renderer / camera      | owns them                                      | host owns them                        |
| Controls               | owns them (orbit / fly / ortho)                | host owns them                        |
| Post-processing        | owns the pipeline                              | host owns it                          |
| UI (panels, rail, …)   | owns it                                        | none                                  |
| Render loop            | owns it (`AnimationController`, idle-pausing)  | host calls `update()` per frame       |
| Contributes            | the whole viewer                               | one `THREE.Group` + per-frame upkeep  |

Use `LuxarApp` to embed *the viewer* in a page. Use `LuxarLayer` when the host
already has a 3D scene and wants Luxar's data as one more thing in it — sharing
one WebGL context, one camera, one set of controls.

## Why it works

The seam predates this module. `SceneLoader.loadScene()` returns a plain
`THREE.Group` (its own docstring says `threeScene.add(scene)`), and both
`LODGroupRegistryDeps` and `configureDepthSort()` are defined purely in terms of
injectable getters — `getCamera()`, `getViewportSize()`, `getDisplayDims()`.
Nothing in the data, cache, LOD, or material path reaches for `SceneManager`.
`LuxarLayer` supplies those getters from the host instead.

## Usage

```ts
import { LuxarLayer } from '@royerlab/luxar-viewer';

const layer = new LuxarLayer({
  renderer,                                              // host-owned
  getCamera: () => camera,                               // host-owned, live getter
  getViewportSize: () => renderer.getSize(new THREE.Vector2()),
  scene,                                                 // host-owned
});

await layer.load('https://example.com/imaging.luxar.zarr');

function animate() {
  requestAnimationFrame(animate);
  layer.update();          // BEFORE the host renders
  renderer.render(scene, camera);
}

// teardown
await layer.dispose();
```

### nD navigation

```ts
const t = layer.findDimension('time');
if (t !== null) {
  layer.prefetchDimensionValue(t, frame + 1);   // warm the next slice
  void layer.setDimensionValue(t, frame);        // do NOT await during playback
}
```

`setDimensionValue` **coalesces**: while a slice is in flight, further calls
replace a single queued update rather than stacking. A host scrubbing a slider
at frame rate would otherwise issue tens of full view updates per second, all
but the last already stale.

Awaiting it during playback makes the host's own timeline stutter on network
latency. The intended pattern is fire-and-forget plus a prefetch of the next
value, so the layer shows the nearest committed slice and catches up.

### Placing the data in the host's world

```ts
layer.alignTo(matrix);   // e.g. the host normalizes its own data into a unit box
```

Applied to the root's matrix, so nothing downstream needs to know: LOD selection
reads projected screen area, which is transform-invariant.

## Host responsibilities

- **`update()` once per frame, before rendering.** It runs the depth-sort
  scheduler and the LOD-group selector, in that order (sorting assigns the
  cross-node render order that a LOD swap can invalidate).
- **`resize()`** after a viewport, DPR, or camera-projection change. The layer
  cannot observe the host's canvas.
- **`requestRender`** if the host renders on demand. Without it, geometry that
  commits outside a user interaction — progressive refinement, lazy LOD loads,
  retries — will not repaint. Hosts that render continuously can omit it.
- **Draw order for the host's own geometry.** Luxar assigns `renderOrder` across
  its own nodes; a host with its own transparent geometry should set explicit
  values rather than rely on insertion order.

## Limits

- **One layer per page, and never alongside a `LuxarApp`.** `SceneLoaderManager`,
  `sceneDimsManager`, `materialManager`, and the worker pool are process
  singletons. Two owners share, then corrupt, each other's state. Same
  restriction as `LuxarApp`, same reason.
- **No UI.** No panels, no picking UI, no monitor, no keyboard handling. The
  cross-layer `notifier` stays unregistered, so Luxar's toasts and error
  overlays are silently dropped unless the host registers a backend.
- **A scene's `tone_mapping` is inert.** Luxar tone-maps in the mega-shader,
  which is a post-processing pass the layer does not own (`PostProcessingManager`
  even forces `renderer.toneMapping = NoToneMapping` because of it). It is not a
  per-material setting that could be pushed onto the nodes, so
  `viewer_config.tone_mapping` reaches nothing and changing it has no effect.

  This matters more than it sounds. `additive` blending sums contributions into a
  framebuffer that clamps at 1.0, and normalising amplitudes fixes the per-splat
  scale, not the accumulated one — so in a host with no tone mapping, overlapping
  bright structure clips flat to white with no gradient, and the scene file gives
  no hint that this will happen. A host wanting the filmic rolloff must set
  `renderer.toneMapping` itself (an `OutputPass` picks it up), which also applies
  to the host's own geometry. Otherwise exposure is the only control — the
  scene's authored `opacity` plus `setExposure()` — and `max` is the one blending
  mode that cannot saturate at all.
- **`dispose()` is async** and tears down process singletons — the loader and its
  caches, the data-worker pool, the depth-sort worker, the material cache. It is
  a full teardown of Luxar in the page, not a partial one, which is consistent
  with the single-instance rule.

## Files

| File             | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| `luxar-layer.ts` | The `LuxarLayer` class and its options type.   |
