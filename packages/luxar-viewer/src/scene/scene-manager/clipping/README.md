# Clipping policy

Camera clipping-plane policy and scene-bounds math extracted from
`SceneManager`. Three concerns live side-by-side: pure geometric
helpers over `BoundingBox` / `BoundingSphere`, a lazily-computed
cache of the scene's 3D bounds (projected from nD metadata onto the
displayed dimensions), and the policy functions that turn those
bounds into camera near/far values.

The split lets `SceneManager` stay free of allocation-per-frame
bounds traversals: once the metadata bounds are cached (on the
first successful `ensure()`), dynamic clipping reads from the cache
and the per-frame update is zero-traversal, zero-alloc. A
metadata-less scene is the exception — it has no negative caching,
so the per-frame `ensure()` re-walks the graph each frame (see the
`scene-bounds-cache.ts` row).

## Files

| File                    | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bounds-math.ts`        | Side-effect-free geometry: `BoundingBox` / `BoundingSphere` / `CameraConfig` types, `getBoundingBoxCenter` / `Size` / `MaxDimension` / `Diagonal`, `transformBoundingBox` (4x4 column-major; skips corners with `\|w\| < 1e-12`, falls back to the input box if all 8 are degenerate), `boundingBoxToSphere`, `calculateClippingPlanesFromSphere`, `calculateCameraDistance` (FOV + aspect fit-ratio with 20% margin), `validateFOV`, and `projectBoundsToDisplayDims` (nD min/max → 3D box via `displayDims`). Exports the `MIN_NEAR_PLANE = 1e-9`, `MIN_NEAR_RADIUS_FACTOR = 2e-6`, `MAX_NEAR_FAR_RATIO = 1000` and `SPHERE_SAFETY_EXPANSION = 1.05` constants used by the policy, plus `nearPlaneFloor` (the near floor both clipping paths actually clamp to) and `minNearForRadius` (its dominated last-resort backstop). |
| `scene-bounds-cache.ts` | `SceneBoundsCache` class — lazy, invalidatable cache of the 3D `BoundingBox`, derived `BoundingSphere`, and near-cull margin (~0.1% of diagonal) projected from `userData.positionBounds` to `sceneDimsManager.getDims().displayed`. `ensure(scene)` walks the scene graph to find metadata bounds; after a successful hit subsequent calls are O(1) until `invalidate()`, but a metadata-less scene has no negative caching and re-walks the graph on every call. Also exports the free helpers `computeBoundsFromMetadata` and `findPositionBoundsInScene` used by the cache and by `SceneManager.getSceneBoundsFromMetadata`.                                                                                                                                                                                               |
| `clipping-policy.ts`    | Three policy helpers operating on a narrow `ClippingCtx` (`camera`, `controls`, `scene`, `boundsCache`, `getSceneBoundsFromMetadata`): `applyClippingPlanes` (validates `near < far`, warns on `far/near > 10000`, calls `updateProjectionMatrix`), `autoAdjustFromBounds` (metadata first, `THREE.Box3.setFromObject` fallback, also pushes scene diagonal into `controls.setSceneScale`), and `updateDynamicFromCache` (per-frame sphere-based near/far, with a 0.1% change threshold to avoid projection-matrix thrash on sub-pixel camera moves).                                                                                                                                                                                                                                                                          |

## Invariants

- **Sphere, not box, for clipping.** The dynamic and auto-adjust
  paths convert the cached `BoundingBox` to a circumscribed
  `BoundingSphere` (with `SPHERE_SAFETY_EXPANSION = 1.05`) before
  computing near/far. The sphere is direction-independent, so near
  values transition smoothly as the camera moves around the scene
  instead of jumping at box edges/corners. No exponential smoothing
  is needed.
- **Inside-sphere clamp is RATIO-BOUNDED.** When
  `dist(camera, center) < R * 1.05` the sphere-surface distance is
  meaningless, so near becomes the floor
  `nearPlaneFloor(R, far) = max(minNearForRadius(R), far / MAX_NEAR_FAR_RATIO)`.
  The `far / 1000` term is the operative one and it exists for
  DEPTH-BUFFER PRECISION: depth quantization goes as
  `d² · (far − near) / (near · far) · 2⁻²⁴` on the 24-bit depth
  renderbuffer, so an unbounded near/far ratio z-fights (and pops as the
  camera orbits, since near tracks camera distance). Being inside the
  circumscribed sphere is not exotic — it is just "zoomed in", because
  the sphere is 1.73x the half-side of a cube.

  The bound is lossless for Points / Lines / GSplats, not a tradeoff:
  their shaders already discard anything closer than
  `nearCull = 1e-3 * diagonal` (`perspectiveNearFade`), and the floor
  stays inside that reject band — by only 0.8%, which is why a property
  test enforces it rather than a comment asserting it. The full
  derivation of the constant lives on `MAX_NEAR_FAR_RATIO`; it is not
  restated here, so there is one place to change. Mesh has no near fade
  in either backend and is the one type the floor can clip.
  Under PERSPECTIVE, `MIN_NEAR_RADIUS_FACTOR` is dominated everywhere
  except a zero-radius sphere (where it yields `MIN_NEAR_PLANE`, keeping
  the degenerate-frustum guard tripping instead of NaN-ing); under ortho
  it is the operative floor, since ortho opts out of the ratio bound
  (next invariant). Re-implemented
  inline by `updateDynamicFromCache` for the zero-alloc per-frame path,
  with a parity test pinning the two to identical values.

- **The ratio bound is PERSPECTIVE-ONLY.** Orthographic depth is linear
  in eye space, so its resolution is `(far - near) / 2^24` whatever
  `near` is — the bound buys ortho nothing, while `perspectiveNearFade`
  returns 1.0 under ortho so all four geometry types draw right up to
  `near` there. Both paths therefore pass
  `!isOrthographicCamera(camera)` into `nearPlaneFloor`, read from the
  LIVE camera because the viewer swaps projections at runtime (V key).

- **Metadata bounds preferred over scene-graph bounds.** Both
  `autoAdjustFromBounds` and the cache reach for
  `userData.positionBounds` (full dataset extent set by the loader).
  When it is absent, `autoAdjustFromBounds` falls back to
  `THREE.Box3.setFromObject`; the cache has no such fallback — it
  simply stays empty, `getBounds()` / `getSphere()` return null, and
  `getNearCull()` keeps its 0.1 default. Metadata is
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
