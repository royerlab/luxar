# Render-pipeline helpers

Focused helpers extracted from `scene/scene-manager.ts` that own the
**graphics-API negotiation, post-processing wiring, scene-graph
disposal, and WebGL context-loss recovery** concerns. Each file is a
single-purpose unit that the `SceneManager` composes during
`init()` / `loadSceneData()` / `dispose()`; nothing here reaches back
into `SceneManager` fields directly — every dependency arrives via
the function arguments or a deps interface.

These files were lifted as-is from their inline counterparts on the
class. Behavior is identical to the pre-extraction code; the split
exists so each concern can be unit-tested independently and so the
SceneManager body stays an orchestrator instead of a megaclass.

## Files

| File                        | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renderer-setup.ts`         | Backend selection + renderer construction. `selectBackend()` applies the URL-param / env-var / default precedence ladder. `createWebGLRenderer()` builds a `THREE.WebGLRenderer` from an explicit WebGL2 context. `createWebGPURenderer()` negotiates a "core" `GPUAdapter` + `GPUDevice` with raised `maxVertexBuffers` / `maxBufferSize` / `maxStorageBufferBindingSize` to bypass r184's compat-mode caps, awaits `renderer.init()`, and returns `{ fallback: true }` when the adapter sits below the 8-vertex-buffer spec minimum (unless `?renderer=webgpu` overrides). Both `create*Renderer` paths configure HDR via `configureHDRRenderer` / `logHDRCapabilities` before returning. Each returns `{ renderer, capabilities }`; the host wires `materialManager.setCaps`, clear color, and initial resize. |
| `post-processing-setup.ts`  | Thin factory over `PostProcessingManager`. Reads canvas dimensions from `renderer.domElement` to seed the render-target pyramid and forwards an `onResize` callback so the host can refresh material uniforms that cache `renderer.getDrawingBufferSize()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `scene-disposal.ts`         | Three independent disposal helpers. `disposeObjectTree(obj)` walks an Object3D depth-first, disposing `Mesh` / `InstancedMesh` geometry and material(s) while removing each child from its parent. `clearLoadedSceneContent(scene)` removes the loaded subtree from a scene root while preserving lights and `userData.isBackground` markers, returning the number of removed objects. `disposeSceneGraphResources(scene)` is the final-shutdown traversal — disposes geometry + material(s) on every renderable but does NOT detach from the graph (caller is dropping the scene anyway).                                                                                                                                                                                                                        |
| `webgl-context-recovery.ts` | `WebGLContextRecovery` class + `markSceneResourcesDirtyForContextRestore()` helper. Owns the canvas-side `webglcontextlost` / `webglcontextrestored` listeners and the deterministic rebuild sequence on restore: `renderer.resetState` → `postProcessing.rebuildAfterContextRestore` → `materialManager.rebuildAfterContextRestore` → mark scene attributes/materials dirty → `updateRendererSize` → `onContextRestored` → `triggerChange`. Communicates with the host SceneManager via a `WebGLContextRecoveryDeps` interface (lookup functions for scene + post-processing, callbacks for resize / restore-broadcast / change).                                                                                                                                                                                |

## Invariants

- **No back-references to SceneManager.** Each helper receives its
  dependencies explicitly (function arguments or
  `WebGLContextRecoveryDeps`). The SceneManager wires them up but the
  helpers never import or query it. This is what makes the split
  worth doing: each unit is testable in isolation.
- **`createWebGPURenderer` fallback contract.** Returning
  `{ fallback: true }` means the WebGPU adapter advertises
  `maxVertexBuffers < 8` (WebGPU spec minimum) AND the user did not
  pass `?renderer=webgpu`. The caller must drop to
  `createWebGLRenderer` — no partial WebGPU state is left behind.
  When `rendererOverride === 'webgpu'`, the function pushes through
  on the degenerate adapter so debug sessions can still inspect it.
- **Why `getPostProcessing()` / `getScene()` are functions, not
  references.** `setupContextLossHandling()` runs BEFORE `setupScene()`
  and post-processing init in the SceneManager's `init()`. Closing
  over a direct reference at construction would capture `undefined`.
  The getters resolve at restore-time, by which point the host has
  finished initialising both.
- **PostProcessing identity is preserved across context restore.**
  `rebuildAfterContextRestore()` mutates GPU-bound resources in place
  rather than replacing the manager; downstream caches (PickingSystem,
  AnimationController, RenderingControls) keep their references valid
  and user settings (bloom, exposure, tone mapping) survive end-to-end.
  The recovery class only calls the in-place rebuild — it never
  replaces the manager.
- **Disposal walk shape.** `disposeObjectTree` always disposes
  `children[0]` until the array is empty (rather than iterating an
  index). This avoids re-indexing surprises when `remove` shifts the
  array under us. `clearLoadedSceneContent` snapshots the removable
  children first (back-to-front) before mutating the scene, for the
  same reason.
- **Final-shutdown traversal is detach-free.** `disposeSceneGraphResources`
  does NOT call `scene.remove()` — the caller is about to drop the
  scene, renderer, and camera, so a single `scene.traverse()` is
  faster than walk-and-remove. The split between this and
  `clearLoadedSceneContent` is intentional: clear-on-reload preserves
  the scene root and lights; clear-on-shutdown is permitted to leave
  dangling references.
- **Restore rebuild order is fixed.** `resetState` →
  `rebuildAfterContextRestore` (post-processing) →
  `rebuildAfterContextRestore` (materials) → mark scene dirty →
  `updateRendererSize` → `onContextRestored` (node-factory and other
  subscribers) → `triggerChange`. Reordering breaks the contract:
  subscribers expect post-processing + materials to already be valid
  when they re-register their per-context state.

## See also

- `../../scene-manager.ts` — the orchestrator that composes these
  helpers during `init()` / `loadSceneData()` / `dispose()`.
- `../../README.md` — Scene package overview, including WebGL
  context-loss UX section that mirrors the recovery flow here.
- `../../../rendering/renderer-capabilities.ts` —
  `createRendererCapabilities`, `Renderer`, `RendererCapabilities`
  consumed by `renderer-setup.ts`.
- `../../../rendering/index.ts` (`PostProcessingManager`),
  `../../../rendering/material-manager.ts` (`materialManager`) —
  rebuild targets used during context restore.
- `../../../utils/hdr/hdr-detection.ts` — `configureHDRRenderer` /
  `logHDRCapabilities` invoked at the tail of both `create*Renderer`
  paths.
- `../viewport/` — `ResizeOrchestrator` + the pure `dpr-policy.ts`
  functions; the resize / DPR wiring downstream of
  `createPostProcessing`'s `onResize` callback. (The
  `AdaptiveDPRManager` that drives that wiring lives in
  `../../../rendering/adaptive-dpr-manager.ts`, not under `viewport/`.)
