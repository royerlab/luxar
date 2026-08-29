# `core/layer` — Layer mode

The normative public embedding contract is
[`docs/specs/LUXAR_LAYER_SPEC.md`](../../../../../docs/specs/LUXAR_LAYER_SPEC.md).
This package note explains the internal seam and implementation-specific rationale.

> Render a Luxar scene inside a **host application's own** Three.js renderer,
> camera, and scene graph.

## What this is

`LuxarLayer` is the headless sibling of `LuxarApp`.

|                      | `LuxarApp`                                    | `LuxarLayer`                         |
| -------------------- | --------------------------------------------- | ------------------------------------ |
| Renderer / camera    | owns them                                     | host owns them                       |
| Controls             | owns them (orbit / fly / ortho)               | host owns them                       |
| Post-processing      | owns the pipeline                             | host owns it                         |
| UI (panels, rail, …) | owns it                                       | none                                 |
| Render loop          | owns it (`AnimationController`, idle-pausing) | host calls `update()` per frame      |
| Contributes          | the whole viewer                              | one `THREE.Group` + per-frame upkeep |

Use `LuxarApp` to embed _the viewer_ in a page. Use `LuxarLayer` when the host
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
  renderer, // host-owned
  getCamera: () => camera, // host-owned, live getter
  getViewportSize: () => renderer.getSize(new THREE.Vector2()),
  scene, // host-owned
});

await layer.load('https://example.com/imaging.luxar.zarr');

function animate() {
  requestAnimationFrame(animate);
  layer.update(); // BEFORE the host renders
  renderer.render(scene, camera);
}

// teardown
await layer.dispose();
```

### nD navigation

```ts
const t = layer.findDimension('time');
if (t !== null) {
  layer.prefetchDimensionValue(t, frame + 1); // warm the next slice
  void layer.setDimensionValue(t, frame); // do NOT await during playback
}
```

`setDimensionValue` **coalesces**: while a slice is in flight, further calls
replace a single queued update rather than stacking. A host scrubbing a slider
at frame rate would otherwise issue tens of full view updates per second, all
but the last already stale.

Awaiting it during playback makes the host's own timeline stutter on network
latency. The intended pattern is fire-and-forget plus a prefetch of the next
value, so the layer shows the nearest committed slice and catches up.

A host must not assume which centre column is which. Producers disagree about
axis order for the same specimen — a fit handed over as `(Z, Y, X)` may be
published as a scene storing `(X, Y, Z)`, and a scene authored by
`luxar gsplat convert` names its axes `dim0…dimN` and declares nothing at all.
Guessing wrong renders a plausible, silently transposed scene. Read the names:

```ts
const names = layer.getDimensionNames(); // e.g. ['Z', 'Y', 'X', 'Time'] — or ['dim0', …]
```

### Placing the data in the host's world

```ts
layer.alignTo(matrix); // e.g. the host normalizes its own data into a unit box
```

Applied to the root's matrix, so nothing downstream needs to know: LOD selection
reads projected screen area, which is transform-invariant.

`alignTo` is **order-independent with `load`**: a matrix declared first is
remembered and applied when the scene arrives. A host usually derives its
placement from its own metadata, which resolves on a schedule unrelated to the
scene fetch, so requiring one order would make correctness a race.

### Visibility and exposure

```ts
layer.setVisible(false); // hide without discarding caches or in-flight streams
layer.setExposure(0.75); // scale authored node opacity; 1 = as authored
```

`setVisible` toggles `visible` on the root rather than detaching it, so
caches and in-flight fetches survive. Lazy LOD loads pause while hidden and
resume on the next update after re-showing; under GPU-budget pressure, resident
levels in a hidden layer are evicted before visible ones. The setting survives
initial load and later dataset switches.

`setExposure` scales the _authored_ opacity, not the live value, so the result
does not depend on how a slider was dragged, and it is re-applied as geometry
streams in so late-arriving nodes match the ones already on screen. It exists
because a scene's authored exposure was tuned against whatever post chain
authored it, and the host's is a different one.

For Points, Lines, and Gaussian Splats, that opacity controls the emissive
contribution. For a Mesh in its default `opaque` mode, it is instead cutout
coverage: values below `alphaCutoff` (0.5 by default) discard the surface, while
values above it do not dim surviving fragments. Author the mesh with `normal`
blending when exposure should produce smooth surface transparency.

### API summary

| Method                             | Purpose                                                           |
| ---------------------------------- | ----------------------------------------------------------------- |
| `load(src)`                        | Load a scene; resolves once the first slice commits               |
| `update()`                         | Per-frame bookkeeping — depth sort, then LOD selection            |
| `resize()`                         | After a viewport, DPR, or camera-projection change                |
| `alignTo(m)`                       | Place the root in host world space; order-independent with `load` |
| `getBounds()`                      | World-space bounds, or `null` before load                         |
| `getDimensions()`                  | Dimension metadata (ndim, displayed, currentStep, ranges)         |
| `getDimensionNames()`              | Axis names in centre-column order                                 |
| `findDimension(name)`              | Index by name, case-insensitive, or `null`                        |
| `setDimensionValue(i, v)`          | Move a non-displayed axis; coalesces                              |
| `prefetchDimensionValue(i, v)`     | Warm a slice without committing it                                |
| `awaitDimensionUpdate()`           | Resolve once no slice update is in flight                         |
| `setVisible(v)` / `isVisible()`    | Show/hide while preserving caches and in-flight fetches           |
| `setExposure(m)` / `getExposure()` | Scale exposure relative to authored                               |
| `getDatasetFault()`                | Current `{ src, error }` archive fault, or `null`                 |
| `onDatasetFault(fn)`               | Subscribe to faults; replays current state; returns unsubscribe   |
| `handleContextLost()`              | Back off the Luxar GPU budget after WebGL context loss            |
| `handleContextRestored()`          | Rebuild Luxar resources after host WebGL context recovery         |
| `dispose()`                        | Async full teardown of Luxar in the page                          |

## Host responsibilities

- **`update()` once per frame, before rendering.** It runs the depth-sort
  scheduler and the LOD-group selector, in that order (sorting assigns the
  cross-node render order that a LOD swap can invalidate).
- **`resize()`** after a viewport, DPR, or camera-projection change. The layer
  cannot observe the host's canvas.
- **`requestRender`** if the host renders on demand. Without it, geometry that
  commits outside a user interaction — progressive refinement, lazy LOD loads,
  retries — will not repaint. Hosts that render continuously can omit it.
- **Draw order for the host's own geometry.** `renderOrder` defaults to 10 and is
  stamped onto every Group in the Luxar subtree, including groups that stream in
  later. Three.js compares that Group key before per-mesh `renderOrder`, so host
  transparent groups should use explicit lower/higher values rather than rely on
  insertion order.
- **WebGL context recovery.** The host owns the canvas events. Call
  `handleContextLost()` on loss so Luxar drops warm-up programs and reduces its
  GPU-resident byte budget, then reset the renderer / post-processing and call
  `handleContextRestored()` so Luxar rebuilds materials, geometry uploads,
  loader registrations, and blend-program warm-up state.

## Limits

- **One layer per page, and never alongside a `LuxarApp`.** `SceneLoaderManager`,
  `sceneDimsManager`, `materialManager`, and the worker pool are process
  singletons. Two owners share, then corrupt, each other's state. Same
  restriction as `LuxarApp`, same reason.
- **No near-cull is pushed.** `SceneManager` derives a near-cull distance from
  its dynamic scene-bounds cache and passes it as a fourth argument to
  `updateCameraParams`, which fades geometry approaching the near plane.
  `resize()` omits it, so the shared near fade stays at its default and elements
  pop instead of fading. Wiring it would mean reproducing the bounds cache. A
  host that cares should keep its near plane clear of the data.
- **`setDimensionValue()` before `load()` is a no-op.** The dimension set comes
  from the scene, and `initFromScene` would overwrite a pre-load value with the
  scene's own defaults anyway. Restore a saved timepoint _after_ `load()`
  resolves.
- **A second `load()` is a dataset switch, and overlapping loads throw.** The
  previous root is detached, because `loadScene` has already disposed its loader
  and leaving it attached would draw over disposed backing stores. A failed
  switch also detaches the previous root for the same reason, leaving the layer
  empty until a later load succeeds. Two _concurrent_ loads cannot be resolved
  that way — the second's
  `createLoaderAsync` disposes the first's loader mid-flight, and whichever
  resolves last wins the root slot — so `load()` refuses to start while another
  is in flight rather than silently producing dead geometry.
- **No UI.** No panels, no picking UI, no monitor, no keyboard handling. The
  cross-layer `notifier` stays unregistered, so Luxar's toasts and error
  overlays are silently dropped unless the host registers a backend. Archive
  failures are the exception: `onDatasetFault()` receives a
  `{ src, error }` payload, and `getDatasetFault()` reads the current state.
  The last complete frame stays visible. Fault delivery is one-way: an
  explicit retry can clear the loader's latch without a callback, and a later
  failure can notify again, so a host that renders recoverable state must poll
  `getDatasetFault()` rather than permanently latch the callback result.
- **Mesh textures use the host renderer.** The layer installs a renderer-owned
  KTX2 decoder for compressed mesh textures; raw and JPEG textures keep using
  the portable decode path.
- **A scene's post-processing / camera / UI config is inert.** Everything under
  `viewer_config` that `ui/rendering-controls.ts` applies rather than the node
  path reaches nothing here: `tone_mapping`, `exposure`, `global_gamma`,
  `global_offset`, the `bloom_*` family, `background_color`, and the whole
  `camera` block. The layer owns no post-processing, no camera, and no UI. Only
  per-node appearance — colormap, blending mode, opacity, absorption, the
  intensity/offset window — travels with the geometry.

  Tone mapping is the one that bites, because it changes what the data looks
  like and nothing surfaces it. Luxar tone-maps in the mega-shader, a
  post-processing pass (`PostProcessingManager` even forces
  `renderer.toneMapping = NoToneMapping` because of it), so it is not a
  per-material setting that could be pushed onto the nodes. Measured in a host application, a scene authored with
  ACES rendered pixel-identical to one authored with `None` — measured, not
  asserted by a test in this repo.

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

| File             | Purpose                                      |
| ---------------- | -------------------------------------------- |
| `luxar-layer.ts` | The `LuxarLayer` class and its options type. |
