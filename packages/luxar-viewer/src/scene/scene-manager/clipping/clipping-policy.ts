/**
 * Camera clipping-plane policy extracted from SceneManager.
 *
 * Three independent behaviours live here as pure helpers that take
 * a narrow `ClippingCtx` rather than reading SceneManager fields:
 *
 *   - `applyClippingPlanes` — apply explicit near/far with
 *     validation and Z-precision warnings.
 *   - `autoAdjustFromBounds` — derive near/far from scene bounds
 *     (metadata first, geometry fallback), feed scene scale into
 *     controls, and apply.
 *   - `updateDynamicFromCache` — per-frame near/far from a cached
 *     bounding sphere; zero allocations, and zero scene-graph
 *     traversal once the bounds are cached (a metadata-less scene
 *     re-walks each frame). Skips updates < 0.1% change for stability.
 *
 * Both bounds-derived paths clamp `near` to the SHARED
 * `nearPlaneFloor(R, far)`, which bounds the near/far ratio to
 * `MAX_NEAR_FAR_RATIO` and thereby the depth-buffer quantization. The
 * two paths compute the same math twice on purpose (the per-frame one
 * must not allocate); `clipping-policy.test.ts` pins them to identical
 * values so the duplication cannot drift.
 *
 * The host owns `ClippingState` (enabled flag) and threads it
 * through alongside camera / controls / scene refs. Event dispatch
 * remains at SceneManager call sites — these helpers never call
 * `dispatchEvent`.
 *
 * @module scene/scene-manager/clipping/clipping-policy
 */

import * as THREE from 'three';
import { config } from '../../../config';
import { log, Modules } from '../../../utils/log';
import type { ControlsManager } from '../../../controls/controls-manager';
import { isOrthographicCamera, type LuxarCamera } from '../../../utils/camera-utils';
import {
  type BoundingBox,
  boundingBoxToSphere,
  calculateClippingPlanesFromSphere,
  getBoundingBoxDiagonal,
  nearPlaneFloor,
  SPHERE_SAFETY_EXPANSION,
} from './bounds-math';
import type { SceneBoundsCache } from './scene-bounds-cache';

/**
 * Ctx supplied by SceneManager to the policy helpers. Kept narrow:
 * helpers read these refs but never mutate the host class.
 */
export interface ClippingCtx {
  readonly camera: LuxarCamera;
  readonly controls: ControlsManager;
  readonly scene: THREE.Scene;
  readonly boundsCache: SceneBoundsCache;
  /** Compute 3D bounds from scene metadata. Provided by SceneManager. */
  readonly getSceneBoundsFromMetadata: () => BoundingBox | null;
}

/**
 * Ratio above which the Z-buffer precision warning fires. Deliberately
 * looser than {@link MAX_NEAR_FAR_RATIO} (the bound the automatic paths
 * clamp to): this is the "you are in trouble" line for MANUALLY set or
 * zarr-authored planes, not the target.
 */
const RATIO_WARN_THRESHOLD = 10000;

/**
 * A camera plane the 0.1%-change gate in {@link updateDynamicFromCache}
 * cannot reason about, and which must therefore count as "changed" so a
 * healthy value can be written back.
 *
 * NaN fails every comparison, so `Math.abs(NaN - near) / NaN > 0.001` is
 * false. A NEGATIVE current value is just as sticky and less obvious: the
 * quotient `Math.abs(current - near) / current` is then negative, so the
 * gate stays shut for every possible new value. `restoreCamera`
 * (`core/app/snapshot/viewer-snapshot.ts`) assigns a snapshot's planes to
 * the camera verbatim, so a poisoned pair really can arrive.
 */
function isUnusablePlane(value: number): boolean {
  return !Number.isFinite(value) || value <= 0;
}

/**
 * Apply explicit near/far to the camera with validation. Logs a
 * Z-precision warning when far/near > 10000.
 *
 * This is the only path that can produce a pathological ratio: the two
 * bounds-derived paths clamp to `nearPlaneFloor`, which caps the ratio
 * at `MAX_NEAR_FAR_RATIO` by construction. So the warning lives
 * here, where MANUAL slider values and zarr-authored `camera.near` /
 * `camera.far` arrive, and `updateDynamicFromCache` deliberately does
 * NOT route through it — a per-frame call has nothing to warn about and
 * would allocate a log string 60x a second to say so. The
 * `bounds-math.property.test.ts` ratio property is the regression
 * tripwire for the automatic paths instead.
 *
 * @returns true when the planes were applied; false when validation rejected them.
 */
export function applyClippingPlanes(camera: LuxarCamera, near: number, far: number): boolean {
  // Non-finite check FIRST, and separately from `near >= far`: every comparison
  // against NaN is false, so `NaN >= far` does not reject and a NaN would sail
  // through into the projection matrix, blanking the view with no diagnostic.
  // Infinity is refused for the same reason — an infinite plane makes the matrix
  // degenerate rather than meaning "see everything".
  //
  // DEFENSIVE, not currently reachable: every present caller of
  // `SceneManager.updateClippingPlanes` passes values that already went through
  // `validateRenderingSettings` (reset / load / zarr paths) or the number
  // controller's numeric input parsing (the sliders), so none can deliver a
  // non-finite pair today. The guard exists because that is a property of the
  // CALLERS, not of this function's contract, and because the sibling
  // `restoreCamera` (`core/app/snapshot/viewer-snapshot.ts`) shows the shape of
  // the hazard: it assigns `camera.near` / `camera.far` directly and so bypasses
  // this validation entirely. Anything routed here in future is covered.
  // `near <= 0` is refused alongside: a zero or negative near is invalid for a
  // perspective frustum (the projection divides by it) and is equally
  // unreachable from `near >= far` when `far` is also non-positive. Same
  // rationale as above — the callers happen to guarantee positivity today
  // (`positiveOrDefault`, and the controllers' clamp to a `minNearForRadius`
  // minimum), but that is their property, not this function's contract.
  if (!Number.isFinite(near) || !Number.isFinite(far) || near <= 0) {
    log.warning(
      Modules.SCENE_MANAGER,
      `Ignoring invalid clipping planes (near: ${near}, far: ${far})`
    );
    return false;
  }

  if (near >= far) {
    log.warning(Modules.SCENE_MANAGER, 'Near plane must be less than far plane');
    return false;
  }

  const ratio = far / near;
  if (ratio > RATIO_WARN_THRESHOLD) {
    log.warning(
      Modules.SCENE_MANAGER,
      `High near/far ratio (${ratio.toFixed(0)}:1) may cause Z-buffer precision issues. Consider adjusting clipping planes.`
    );
  }

  camera.near = near;
  camera.far = far;
  camera.updateProjectionMatrix();

  log.info(
    Modules.SCENE_MANAGER,
    `Clipping planes updated - Near: ${near < 0.001 ? near.toExponential(1) : near.toFixed(3)}, Far: ${far.toFixed(1)} (ratio: ${ratio.toFixed(0)}:1)`
  );
  return true;
}

/**
 * Auto-adjust clipping planes from scene bounds.
 *
 * Prefers metadata bounds (full dataset extent, available before
 * geometry loads); falls back to a Box3 over the loaded scene
 * graph. Also feeds the bounding-box diagonal into the scale-aware
 * controls.
 *
 * @returns the derived near/far and whether they were applied; on empty
 *   scenes, returns the configured defaults with `applied: false` without
 *   touching the camera.
 *
 * Note on that empty-scene return: those configured defaults
 * (`near` 0.1 / `far` 1000) are a ratio of 10,000 — well past
 * `MAX_NEAR_FAR_RATIO`. That is deliberate and inert, not an oversight
 * to "fix": the branch touches no camera, and its only production caller
 * (`SceneManager.autoAdjustClippingPlanes`) uses `applied: false` to avoid
 * dispatching a view change. It is reached only when the scene has neither
 * metadata bounds nor geometry — i.e. when there is nothing to z-fight. The
 * camera keeps the same defaults it was constructed with, and the first real
 * bounds put it back under the bound.
 */
export function autoAdjustFromBounds(ctx: ClippingCtx): {
  near: number;
  far: number;
  applied: boolean;
} {
  const cameraPos = {
    x: ctx.camera.position.x,
    y: ctx.camera.position.y,
    z: ctx.camera.position.z,
  };
  // Read from the LIVE camera, never cached: the viewer swaps projections at
  // runtime (V key), and the ratio bound must not follow a stale one.
  const boundRatio = !isOrthographicCamera(ctx.camera);

  const sceneBounds = ctx.getSceneBoundsFromMetadata();

  if (sceneBounds) {
    const diagonal = getBoundingBoxDiagonal(sceneBounds);
    if (diagonal > 0) {
      ctx.controls.setSceneScale(diagonal);
    }

    const sphere = boundingBoxToSphere(sceneBounds);
    const { near, far } = calculateClippingPlanesFromSphere(sphere, cameraPos, boundRatio);
    const applied = applyClippingPlanes(ctx.camera, near, far);

    if (applied) {
      log.success(
        Modules.SCENE_MANAGER,
        `Clipping planes set from metadata bounds (near: ${near.toFixed(4)}, far: ${far.toFixed(1)})`
      );
    }

    return { near, far, applied };
  }

  // Fallback: bounds from the loaded scene graph.
  const box = new THREE.Box3().setFromObject(ctx.scene);
  if (box.isEmpty()) {
    log.warning(Modules.SCENE_MANAGER, 'No scene content for clipping plane calculation');
    return {
      near: config.renderingControls.defaults.near,
      far: config.renderingControls.defaults.far,
      applied: false,
    };
  }

  const fallbackBounds = {
    min: { x: box.min.x, y: box.min.y, z: box.min.z },
    max: { x: box.max.x, y: box.max.y, z: box.max.z },
  };

  const diagonal = getBoundingBoxDiagonal(fallbackBounds);
  if (diagonal > 0) {
    ctx.controls.setSceneScale(diagonal);
  }

  const sphere = boundingBoxToSphere(fallbackBounds);
  const { near, far } = calculateClippingPlanesFromSphere(sphere, cameraPos, boundRatio);
  const applied = applyClippingPlanes(ctx.camera, near, far);

  return { near, far, applied };
}

/**
 * Per-frame near/far update from the cached bounding sphere.
 *
 * Zero allocations, and zero scene-graph traversal once the bounds
 * are cached — the `ensure()` below is O(1) after the first
 * successful hit. (A metadata-less scene has no negative caching, so
 * that `ensure()` re-walks the graph each frame — see
 * `scene-bounds-cache.ts`.) Only updates the camera when values
 * changed more than 0.1% (stability gate to avoid projection-matrix
 * thrash from sub-pixel camera moves).
 *
 * No-op (returns early) when:
 *  - the bounds cache is empty (no metadata available);
 *  - the sphere is degenerate (near >= far — e.g. a zero-extent
 *    single-point scene, whose radius-0 sphere yields no valid frustum);
 *  - changes are below the 0.1% threshold.
 */
export function updateDynamicFromCache(ctx: ClippingCtx): void {
  ctx.boundsCache.ensure(ctx.scene);
  const s = ctx.boundsCache.getSphere();
  if (!s) return;

  const cam = ctx.camera.position;
  const dx = cam.x - s.center.x;
  const dy = cam.y - s.center.y;
  const dz = cam.z - s.center.z;
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const R = s.radius * SPHERE_SAFETY_EXPANSION;
  const far = dist + R;
  // Live projection check (not cached): ortho opts out of the ratio bound —
  // its depth is linear, so the bound is pure clipping cost there.
  const minNear = nearPlaneFloor(R, far, !isOrthographicCamera(ctx.camera));
  const near = dist < R ? minNear : Math.max(minNear, dist - R);

  // Degenerate guard (zero-extent scene → radius-0 sphere → near >= far):
  // writing that to the camera puts (far - near) = 0 into the projection
  // matrix and NaNs the frustum. Same contract as applyClippingPlanes,
  // which refuses near >= far on the explicit path.
  //
  // The `isFinite` half is EXPLICIT rather than relying on `near >= far`,
  // which is false for NaN and so would let one through. It is load-bearing
  // for INFINITE bounds specifically: there `near`/`far` are both Infinity,
  // `Math.abs(camera.near - Infinity) / camera.near` IS > 0.001, so the change
  // gate below fires and would write Infinity into the projection matrix. (For
  // NaN bounds the gate happens to block the write too, since every NaN
  // comparison is false — but that is an accident of comparison semantics, not
  // a guard.) Cheap enough for a per-frame path: two register compares.
  if (!Number.isFinite(near) || !Number.isFinite(far) || near >= far) return;

  // Only update when values changed > 0.1% — avoids thrashing the projection
  // matrix on sub-pixel camera moves.
  //
  // A non-finite or non-positive CURRENT value counts as changed — see
  // `isUnusablePlane`. Otherwise a camera already holding one is stuck
  // forever, because the relative-change quotient can never exceed the
  // threshold. The guard above stops us writing bad values; this is what lets
  // us recover from one.
  const nearChanged =
    isUnusablePlane(ctx.camera.near) || Math.abs(ctx.camera.near - near) / ctx.camera.near > 0.001;
  const farChanged =
    isUnusablePlane(ctx.camera.far) || Math.abs(ctx.camera.far - far) / ctx.camera.far > 0.001;

  if (nearChanged || farChanged) {
    ctx.camera.near = near;
    ctx.camera.far = far;
    ctx.camera.updateProjectionMatrix();
  }
}
