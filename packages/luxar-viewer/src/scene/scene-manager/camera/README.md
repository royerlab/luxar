# Scene-manager camera helpers

Pure helpers extracted from `scene/scene-manager.ts`. Each module owns
one slice of camera behavior — construction, framing, mode swap, or
material parameter push — and reads everything it needs from
parameters or a small ctx object. None of them keep their own state
or hold a back-reference to `SceneManager`.

## Files

| File                  | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `camera-setup.ts`     | Default-camera construction and per-scene zarr overrides. `createDefaultPerspectiveCamera(canvas)` builds a `THREE.PerspectiveCamera` from `config.renderingControls.defaults` + `config.camera.initialPosition`. `resetCameraToInitialPosition(camera)` returns it to that home pose. `applyZarrViewerConfig(root, camera, controls, scene)` applies the loaded zarr's `viewer_config` (position, target, `target_node` lookup via `resolveTargetNodeCenter`, up vector, background color) and reports whether an explicit position was applied so the caller can suppress auto-framing.                                                                                                                                                                                                                     |
| `camera-framing.ts`   | Shared FOV-aware fit math for the two camera-framing callers. `computeSceneBoundingBox(scene)` walks the scene graph and aggregates world-space bounds across Points / Lines / GSplats meshes (instanced + Luxar `InstancedBufferGeometry`). `fitCameraToBounds(camera, controls, bounds, options)` owns the shared distance/zoom math, `lookAt`, controls `reinitialize` + `saveState` ordering, and applies asymmetric `ZOOM_IN_FACTOR = 1000` / `ZOOM_OUT_FACTOR = 10000` distance/zoom limits. `centerCameraOnScene` (F-key path) and `autoFrameCamera` (scene-load path) wrap that core; `centerOnOrigin` resets target to `(0, 0, 0)` at the current distance.                                                                                                                                          |
| `camera-materials.ts` | Pushes the current camera state into the global `materialManager`. `updateMaterialsForCurrentCamera(ctx)` passes the drawing-buffer size, the orthographic flag, the near-cull margin from the bounds cache and the pixel ratio — no FOV or frustum height, because every shader reads its projection terms from the projection matrix per draw (an ortho zoom or FOV change therefore needs no push). `adjustFOV(ctx, deltaY)` mutates `camera.fov` (clamped via `validateFOV` to `[fovMin, fovMax]`), refreshes the projection matrix, and re-pushes — no-op for orthographic cameras.                                                                                                                                                                                                                      |
| `camera-mode.ts`      | Perspective ↔ orthographic camera-swap policy. `setControlType(type, ctx)` swaps the camera when needed, forwards the type to `ControlsManager`, and re-pushes materials; it returns `{ cameraChanged }` so the caller can dispatch `'camera-changed'` (event dispatch stays in `SceneManager`). `swapToOrthographic` matches the visible frustum at the current target distance and is pose-preserving (copies the perspective camera's position, quaternion and up verbatim, changing only the projection), stashing the perspective FOV for the inverse swap; `swapToPerspective` is its exact inverse — it restores that stashed FOV and dollies along the view direction so the apparent size at the pivot depth matches the ortho frustum (compensating for ortho zoom), preserving orientation and up. |

## Invariants

- **Pure helpers.** Each function takes the camera, controls, scene
  references it needs as parameters or via a small ctx interface.
  Helpers do not import `SceneManager` and do not hold per-call
  state. The host owns the `camera` field; mode-swap helpers read
  and replace it through `CameraModeCtx.getCamera()` / `setCamera()`.
- **Reinitialize order matters.** After repositioning the camera,
  `fitCameraToBounds` calls `controls.setTarget(...)` then
  `controls.reinitialize()` then `controls.update()` then
  `controls.saveState()`. Doing `reinitialize()` before `setTarget`
  would re-derive the orbit distance against the old target;
  `update()` before `reinitialize()` would snap the camera back to
  the previous distance.
- **Event dispatch stays in SceneManager.** `setControlType` reports
  `{ cameraChanged }` instead of dispatching `'camera-changed'`
  itself, so the event fires from the public class method's call
  site (non-goal: helpers never dispatch on the host's event
  channel).
- **`target_node` precedes `target`.** In `applyZarrViewerConfig`,
  an explicit `target_node` resolves to the named node's
  world-space bbox center via `resolveTargetNodeCenter` and wins
  over a numeric `target`. A missing name logs a warning and the
  helper falls through to the numeric `target` (not an error).
- **Position alone suppresses auto-framing.** `applyZarrViewerConfig`
  returns `positionApplied: true` only when the zarr specified an
  explicit camera position. An author target without a position
  sets the orbit pivot but still lets auto-framing pick a sensible
  distance.

## See also

- `../../scene-manager.ts` — the host class; public methods
  `setControlType`, `centerCameraOnScene`, `updateFOV`, etc. are
  thin shims that delegate here.
- `../clipping/bounds-math.ts` — `calculateCameraDistance`,
  `getBoundingBoxCenter`, `validateFOV`, `BoundingBox` consumed by
  framing and FOV helpers.
- `../clipping/scene-bounds-cache.ts` — `SceneBoundsCache.ensure` /
  `getNearCull` used by `updateMaterialsForCurrentCamera`.
- `../../../utils/camera-utils.ts` — `LuxarCamera` union and the
  `isPerspectiveCamera` / `isOrthographicCamera` /
  `getCameraFovRadians` / `getOrthoFrustumHeight` discriminators.
- `../../../config/zarr-bridge/viewer-config-utils.ts` —
  `extractCameraOverrides` / `extractBackgroundColor` parsed in
  `applyZarrViewerConfig`.
- `../../../rendering/material-manager.ts` — `materialManager`
  global pushed from `camera-materials.ts`.
- `../../../controls/controls-manager.ts` — `ControlsManager` /
  `ControlType` consumed throughout.
