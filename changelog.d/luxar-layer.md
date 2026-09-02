#### `LuxarLayer` — render a Luxar scene inside a host application's own renderer

`LuxarApp` embeds _the viewer_: it owns the renderer, camera, controls,
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
import { LuxarLayer } from "@luxar/viewer";

const layer = new LuxarLayer({
    renderer,
    getCamera: () => camera,
    getViewportSize,
    scene,
});
await layer.load("https://example.com/imaging.luxar.zarr");
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
the depth-sort scheduler _before_ the LOD-group selector, because sorting
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

The rest of the surface exists because integrating a real host demanded it, and
each entry is there to prevent a _silent_ wrong result rather than an error:

- `alignTo()` is **order-independent with `load()`** — a matrix declared first is
  remembered and applied when the scene arrives. A host derives its placement
  from its own metadata, which resolves on a schedule unrelated to the scene
  fetch, so requiring one order would make correctness a race, and losing the
  matrix puts the data somewhere plausible but wrong.
- `getDimensionNames()` returns the axis names in centre-column order, because a
  host must not assume one. Producers disagree for the same specimen — a fit
  handed over as `(Z, Y, X)` may be published as a scene storing `(X, Y, Z)`, and
  a scene authored by `gsplat convert` names its axes `dim0…dimN` and declares
  nothing. Guessing wrong renders a transposed scene that still looks like a
  scene.
- `setVisible()` / `isVisible()` toggle the root's `visible` rather than
  detaching it, so caches and in-flight fetches survive. Lazy LOD loads pause
  while hidden and resume on the next `update()` after re-showing; under
  GPU-budget pressure, resident levels in a hidden layer are evicted before
  visible ones.
- `setExposure()` / `getExposure()` scale exposure relative to what the scene was
  authored with. A scene's authored exposure was tuned against whichever post
  chain authored it; a host's is a different one. It composes against the
  authored value rather than the live one, so the result does not depend on how a
  slider was dragged, and it re-applies as geometry streams in so late-arriving
  nodes match those already on screen. On a Mesh in the default `opaque` mode,
  this scales cutout coverage rather than brightness: values below `alphaCutoff`
  (0.5 by default) discard the surface, while `normal` blending provides smooth
  transparency.
- `renderOrder` is stamped across nested scene / LOD / partition Groups (including
  groups that stream in later). WebGL context-loss/restoration hooks reduce the
  resident GPU budget, rebuild Luxar-owned resources, and re-arm blend-program
  warm-up against the host renderer.

One limitation is worth stating loudly because nothing surfaces it: **a scene's
`viewer_config` post-processing, camera, and UI block has no effect in layer
mode.** Everything applied by `ui/rendering-controls.ts` rather than by the node
path is inert — `tone_mapping`, `exposure`, `global_gamma`, `global_offset`, the
`bloom_*` family, `background_color`, and `camera.*`. Only per-node appearance
(colormap, blending mode, opacity, absorption, the intensity/offset window)
travels with the geometry.

Tone mapping is the one that bites. Luxar tone-maps in
the mega-shader, which is a post-processing pass the layer deliberately does not
own — `PostProcessingManager` even forces `renderer.toneMapping = NoToneMapping`
because of it. It is not a per-material setting that could be pushed onto the
nodes. A scene authored with ACES therefore renders pixel-identical to one
authored with `None`, and since `additive` blending sums into a framebuffer that
clamps at 1.0, a host with no tone mapping of its own will see bright overlapping
structure clip flat to white. Hosts wanting the filmic rolloff must set
`renderer.toneMapping` themselves.
