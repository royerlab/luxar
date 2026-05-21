# Scene-manager helpers

Pure-helper subtree extracted from `scene/scene-manager.ts`. Each
subfolder owns one slice of `SceneManager`'s responsibilities — camera
construction & framing, clipping-plane policy & bounds math, renderer
/ post-processing wiring & context-loss recovery, and viewport sizing
& DPR policy. Helpers receive everything they need as function
arguments or via small ctx interfaces; nothing here imports
`SceneManager` or holds a back-reference to it.

The split exists so each concern can be unit-tested in isolation and
so the `SceneManager` class body stays an orchestrator instead of a
megaclass. Behaviour is identical to the pre-extraction inline code;
the public class methods on `SceneManager` are now thin shims that
delegate into this tree.

## Layout

```
scene-manager/
├── camera/          # camera construction, framing, mode swap, material push
├── clipping/        # bounds math, scene-bounds cache, clipping-plane policy
├── render-pipeline/ # backend selection, post-processing init, disposal, context recovery
└── viewport/        # DPR policy and rAF-coalesced resize orchestrator
```

There are no top-level source files in this folder — it is a pure
container for the four subpackages below.

## Subpackages

- **`camera/`** — Pure helpers for default-camera construction, zarr
  viewer-config overrides, FOV-aware fit math, the perspective ↔
  orthographic mode swap, and pushing the current camera projection
  into `materialManager`. Event dispatch stays in `SceneManager`;
  helpers report `{ cameraChanged }` instead.
- **`clipping/`** — Side-effect-free `BoundingBox` / `BoundingSphere`
  geometry, the lazy `SceneBoundsCache` (nD-metadata-projected onto
  displayed dims), and three policy helpers (`applyClippingPlanes`,
  `autoAdjustFromBounds`, `updateDynamicFromCache`) operating on a
  read-only `ClippingCtx`. Dynamic clipping is sphere-based and
  zero-alloc per frame, gated by a 0.1% change threshold.
- **`render-pipeline/`** — Backend selection (`?renderer=` ladder),
  `createWebGLRenderer` / `createWebGPURenderer` with HDR
  configuration and the WebGPU compat-mode fallback, the
  `PostProcessingManager` factory, three independent scene-disposal
  helpers, and the `WebGLContextRecovery` class with its fixed
  rebuild order.
- **`viewport/`** — Stateless DPR policy
  (`getActivePixelRatio` / `computePixelRatioOverride` /
  `getNormalizedDPRScale` / `syncPostProcessingDPRScale`) plus the
  `ResizeOrchestrator` class that rAF-coalesces window resizes,
  applies adaptive-DPR resizes immediately, and honours
  `resizeLocked` while a recording is in progress.

## Invariants shared across the subtree

- **No back-references to `SceneManager`.** Each helper takes camera,
  controls, scene, and post-processing references as parameters or
  via a small ctx / deps interface. This is what makes the split
  worth doing: every unit is testable in isolation.
- **Event dispatch stays in `SceneManager`.** Helpers update state
  and may log, but never call `dispatchEvent` on the host's event
  channel — they report changes through return values so the public
  class method's call site owns the event.
- **Cache lifecycle owned by `SceneManager`.** Helpers only call
  `ensure()` / `getSphere()` / `getBounds()` on the bounds cache;
  invalidation happens at scene-load / clear boundaries on the host.
- **Pre-extraction behaviour preserved.** Each file was lifted as-is
  from its inline counterpart in `scene-manager.ts`. Refactors here
  must keep the observable behaviour of the corresponding public
  `SceneManager` method intact.

## See also

- `../scene-manager.ts` — the host class that composes these helpers
  during `init()` / `loadSceneData()` / `dispose()`; public methods
  like `setControlType`, `centerCameraOnScene`, `updateFOV`,
  `updateSize`, `autoAdjustClippingPlanes`, and `setDynamicClipping`
  are thin shims that delegate into this tree.
- `../README.md` — Scene package overview, including the
  context-loss UX section that mirrors the
  `render-pipeline/webgl-context-recovery.ts` flow.
- `../scene-dims-manager.ts` — provides the `displayed` dimension
  indices consumed by `clipping/scene-bounds-cache.ts` to project
  nD metadata bounds into 3D.
- `../animation/` — `AdaptiveDPRManager` calls
  `SceneManager.setAdaptivePixelRatio`, which composes
  `viewport/dpr-policy.ts` + `viewport/resize-orchestrator.ts`.
