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
 *     bounding sphere; zero allocations, zero scene-graph
 *     traversal. Skips updates < 0.1% change for stability.
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
import type { LuxarCamera } from '../../../utils/camera-utils';
import {
  type BoundingBox,
  boundingBoxToSphere,
  calculateClippingPlanesFromSphere,
  getBoundingBoxDiagonal,
  minNearForRadius,
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
 * Apply explicit near/far to the camera with validation. Logs a
 * Z-precision warning when far/near > 10000.
 */
export function applyClippingPlanes(camera: LuxarCamera, near: number, far: number): void {
  if (near >= far) {
    log.warning(Modules.SCENE_MANAGER, 'Near plane must be less than far plane');
    return;
  }

  const ratio = far / near;
  if (ratio > 10000) {
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
}

/**
 * Auto-adjust clipping planes from scene bounds.
 *
 * Prefers metadata bounds (full dataset extent, available before
 * geometry loads); falls back to a Box3 over the loaded scene
 * graph. Also feeds the bounding-box diagonal into the scale-aware
 * controls.
 *
 * @returns the near/far that were applied; on empty scenes,
 *   returns the configured defaults without touching the camera.
 */
export function autoAdjustFromBounds(ctx: ClippingCtx): { near: number; far: number } {
  const cameraPos = {
    x: ctx.camera.position.x,
    y: ctx.camera.position.y,
    z: ctx.camera.position.z,
  };

  const sceneBounds = ctx.getSceneBoundsFromMetadata();

  if (sceneBounds) {
    const diagonal = getBoundingBoxDiagonal(sceneBounds);
    if (diagonal > 0) {
      ctx.controls.setSceneScale(diagonal);
    }

    const sphere = boundingBoxToSphere(sceneBounds);
    const { near, far } = calculateClippingPlanesFromSphere(sphere, cameraPos);
    applyClippingPlanes(ctx.camera, near, far);

    log.success(
      Modules.SCENE_MANAGER,
      `Clipping planes set from metadata bounds (near: ${near.toFixed(4)}, far: ${far.toFixed(1)})`
    );

    return { near, far };
  }

  // Fallback: bounds from the loaded scene graph.
  const box = new THREE.Box3().setFromObject(ctx.scene);
  if (box.isEmpty()) {
    log.warning(Modules.SCENE_MANAGER, 'No scene content for clipping plane calculation');
    return {
      near: config.renderingControls.defaults.near,
      far: config.renderingControls.defaults.far,
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
  const { near, far } = calculateClippingPlanesFromSphere(sphere, cameraPos);
  applyClippingPlanes(ctx.camera, near, far);

  return { near, far };
}

/**
 * Per-frame near/far update from the cached bounding sphere.
 *
 * Zero allocations, zero scene-graph traversal. Only updates the
 * camera when values changed more than 0.1% (stability gate to
 * avoid projection-matrix thrash from sub-pixel camera moves).
 *
 * Returns false (no-op) when:
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
  const minNear = minNearForRadius(R);
  const near = dist < R ? minNear : Math.max(minNear, dist - R);

  // Degenerate guard (zero-extent scene → radius-0 sphere → near >= far):
  // writing that to the camera puts (far - near) = 0 into the projection
  // matrix and NaNs the frustum. Same contract as applyClippingPlanes,
  // which refuses near >= far on the explicit path.
  if (near >= far) return;

  // Only update when values changed > 0.1% — avoids thrashing the
  // projection matrix on sub-pixel camera moves.
  const nearChanged = Math.abs(ctx.camera.near - near) / ctx.camera.near > 0.001;
  const farChanged = Math.abs(ctx.camera.far - far) / ctx.camera.far > 0.001;

  if (nearChanged || farChanged) {
    ctx.camera.near = near;
    ctx.camera.far = far;
    ctx.camera.updateProjectionMatrix();
  }
}
