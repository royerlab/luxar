# Clipping policy

Camera clipping-plane policy and scene-bounds math extracted from
`SceneManager`. Three concerns live side-by-side: pure geometric
helpers over `BoundingBox` / `BoundingSphere`, a lazily-computed
cache of the scene's 3D bounds (projected from nD metadata onto the
displayed dimensions), and the policy functions that turn those
bounds into camera near/far values.

The split lets `SceneManager` stay free of allocation-per-frame
bounds traversals: dynamic clipping reads from the cache once per
scene load, and the per-frame update is zero-traversal, zero-alloc.

## Files

| File                    | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bounds-math.ts`        | Side-effect-free geometry: `BoundingBox` / `BoundingSphere` / `CameraConfig` types, `calculateBoundingBoxFromPositions`, `mergeBoundingBoxes`, `getBoundingBoxCenter` / `Size` / `MaxDimension` / `Diagonal`, `expandBoundingBox`, `isPointInBoundingBox`, `isValidBoundingBox` (loose "has any extent" predicate — true for degenerate 2D slabs / 1D segments, false only for a single point), `transformBoundingBox` (4x4 column-major; skips corners with `\|w\| < 1e-12`, falls back to the input box if all 8 are degenerate), `boundingBoxToSphere`, `calculateClippingPlanesFromSphere`, `calculateCameraDistance` (FOV + aspect fit-ratio with 20% margin), `validateFOV`, and `projectBoundsToDisplayDims` (nD min/max → 3D box via `displayDims`). Exports the `MIN_NEAR_PLANE = 0.0001` and `SPHERE_SAFETY_EXPANSION = 1.05` constants used by the policy. |
| `scene-bounds-cache.ts` | `SceneBoundsCache` class — lazy, invalidatable cache of the 3D `BoundingBox`, derived `BoundingSphere`, and near-cull margin (~0.1% of diagonal) projected from `userData.positionBounds` to `sceneDimsManager.getDims().displayed`. `ensure(scene)` walks the scene graph once to find metadata bounds; subsequent calls are O(1) until `invalidate()`. Also exports the free helpers `computeBoundsFromMetadata` and `findPositionBoundsInScene` used by the cache and by `SceneManager.getSceneBoundsFromMetadata`.                                                                                                 |
| `clipping-policy.ts`    | Three policy helpers operating on a narrow `ClippingCtx` (`camera`, `controls`, `scene`, `boundsCache`, `getSceneBoundsFromMetadata`): `applyClippingPlanes` (validates `near < far`, warns on `far/near > 10000`, calls `updateProjectionMatrix`), `autoAdjustFromBounds` (metadata first, `THREE.Box3.setFromObject` fallback, also pushes scene diagonal into `controls.setSceneScale`), and `updateDynamicFromCache` (per-frame sphere-based near/far, with a 0.1% change threshold to avoid projection-matrix thrash on sub-pixel camera moves).                                                                  |

## Invariants

- **Sphere, not box, for clipping.** The dynamic and auto-adjust
  paths convert the cached `BoundingBox` to a circumscribed
  `BoundingSphere` (with `SPHERE_SAFETY_EXPANSION = 1.05`) before
  computing near/far. The sphere is direction-independent, so near
  values transition smoothly as the camera moves around the scene
  instead of jumping at box edges/corners. No exponential smoothing
  is needed.
- **Inside-sphere clamp.** When `dist(camera, center) < R * 1.05`,
  near collapses to `MIN_NEAR_PLANE = 0.0001` so all surrounding
  geometry stays visible — matches the
  `calculateClippingPlanesFromSphere` documented behaviour and is
  re-implemented inline by `updateDynamicFromCache` for the zero-
  alloc per-frame path.
- **Metadata bounds preferred over scene-graph bounds.** Both
  `autoAdjustFromBounds` and the cache reach for
  `userData.positionBounds` (full dataset extent set by the loader)
  before falling back to `THREE.Box3.setFromObject`. Metadata is
  available before geometry finishes loading and reflects the full
  nD dataset extent rather than the currently-visible slice.
- **nD → 3D projection lives in `projectBoundsToDisplayDims`.** The
  cache and `computeBoundsFromMetadata` defer to this helper so the
  mapping is testable without a populated `THREE.Scene`. Unmapped
  axes default to 0 (degenerate on that axis); out-of-range
  `displayDims` indices also default to 0 — same defensive fallback
  the original inline code used to avoid OOB reads.
- **0.1% change gate on dynamic updates.** `updateDynamicFromCache`
  only mutates `camera.near` / `camera.far` and calls
  `updateProjectionMatrix` when either value changed by more than
  0.1%. Sub-pixel camera moves do not invalidate the projection
  matrix.
- **Cache lifecycle owned by `SceneManager`.** The policy helpers
  never call `boundsCache.invalidate()`; `SceneManager` invalidates
  on scene load / clear. Helpers only call `ensure()` (idempotent)
  and read via `getSphere()` / `getBounds()` / `getNearCull()`.
- **No event dispatch from helpers.** `applyClippingPlanes` and
  friends update camera state and log, but never call
  `dispatchEvent` — that stays at the `SceneManager` call sites so
  the policy stays a pure function of `ClippingCtx`.
- **`ClippingCtx` is read-only.** Helpers read camera, controls,
  scene, and the bounds cache through the ctx but never mutate the
  host class. The only mutation surface is the camera's
  `near` / `far` / projection matrix and the controls'
  `setSceneScale` — both via documented setters.

## See also

- `../../scene-manager.ts` — host class; owns the
  `SceneBoundsCache` instance, builds the `ClippingCtx`, and
  invalidates the cache on `loadSceneData` / clear.
- `../../scene-dims-manager.ts` — provides `displayed` dimension
  indices consumed by `computeBoundsFromMetadata`.
- `../../../controls/controls-manager.ts` — `setSceneScale` consumer
  fed by the bounding-box diagonal in `autoAdjustFromBounds`.
- `../../../utils/camera-utils.ts` — `LuxarCamera` union
  (`PerspectiveCamera | OrthographicCamera`) accepted by the policy
  helpers.
- `../../../config` — `scene.defaultFitRatio` and
  `renderingControls.defaults.{near,far}` used as fallbacks.
