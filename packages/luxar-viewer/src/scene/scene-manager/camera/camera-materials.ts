/**
 * Camera-aware material updates extracted from SceneManager.
 *
 * Two helpers:
 *
 *   - `updateMaterialsForCurrentCamera` — push current camera
 *     projection (FOV-based for perspective, frustum-based for
 *     orthographic) + drawing-buffer size + near-cull margin into
 *     the global materialManager. Called on resize, camera swap,
 *     and FOV change.
 *
 *   - `adjustFOV` — mutate `camera.fov` with clamping, refresh the
 *     projection matrix, and push the new value into materials.
 *     No-op for orthographic cameras.
 *
 * Both helpers are pure with respect to SceneManager — they read
 * everything they need from the supplied ctx.
 *
 * @module scene/scene-manager/camera/camera-materials
 */

import * as THREE from 'three';
import { config } from '../../../config';
import { materialManager } from '../../../rendering/material-manager';
import type { Renderer } from '../../../rendering/renderer-capabilities';
import {
  type LuxarCamera,
  getCameraFovRadians,
  getOrthoFrustumHeight,
  isOrthographicCamera,
  isPerspectiveCamera,
} from '../../../utils/camera-utils';
import { validateFOV } from '../clipping/bounds-math';
import type { SceneBoundsCache } from '../clipping/scene-bounds-cache';

/**
 * Ctx supplied by SceneManager. The materials helpers read these
 * refs but never mutate the host class. `_bufferSize` is a
 * pre-allocated Vector2 owned by SceneManager so resize callbacks
 * don't allocate on every frame.
 */
export interface CameraMaterialsCtx {
  readonly renderer: Renderer;
  readonly camera: LuxarCamera;
  readonly scene: THREE.Scene;
  readonly boundsCache: SceneBoundsCache;
  /**
   * Pre-allocated Vector2 receiver for getDrawingBufferSize.
   *
   * **Borrow contract**: This Vector2 is owned by SceneManager (a single
   * shared instance) and is mutated on every camera-materials update.
   * Downstream consumers MUST treat it as read-only and scoped to the
   * current call. To retain the value across frames, `.copy()` it into
   * private storage — never store the reference itself, or a subsequent
   * resize will silently mutate your "saved" value.
   */
  readonly bufferSize: THREE.Vector2;
}

/**
 * Push current camera projection into the global material manager.
 *
 * Perspective: projection = FOV radians, orthographic flag = false.
 * Orthographic: projection = frustum height, orthographic flag = true.
 *
 * Always ensures the bounds cache is populated (so the materials
 * see a consistent near-cull margin) and uses the supplied
 * pre-allocated Vector2 to avoid per-call allocation.
 */
export function updateMaterialsForCurrentCamera(ctx: CameraMaterialsCtx): void {
  ctx.renderer.getDrawingBufferSize(ctx.bufferSize);
  ctx.boundsCache.ensure(ctx.scene);
  const nearCull = ctx.boundsCache.getNearCull();
  const pixelRatio = ctx.renderer.getPixelRatio();

  if (isOrthographicCamera(ctx.camera)) {
    const frustumHeight = getOrthoFrustumHeight(ctx.camera);
    materialManager.updateCameraParams(frustumHeight, ctx.bufferSize, true, nearCull, pixelRatio);
  } else {
    materialManager.updateCameraParams(
      getCameraFovRadians(ctx.camera),
      ctx.bufferSize,
      false,
      nearCull,
      pixelRatio
    );
  }
}

/**
 * Adjust the perspective camera FOV by `deltaY * fovSensitivity`,
 * clamped to `[fovMin, fovMax]`. Updates the projection matrix and
 * pushes the new projection into materials.
 *
 * Returns true when the FOV was applied, false for orthographic
 * cameras (no-op) — callers use this to skip FOV-coupled UI updates
 * (e.g. flipping the rendering-controls preset to "Custom") when
 * nothing actually changed.
 */
export function adjustFOV(ctx: CameraMaterialsCtx, deltaY: number): boolean {
  if (!isPerspectiveCamera(ctx.camera)) return false;
  const fovChange = deltaY * config.camera.fovSensitivity;
  ctx.camera.fov = validateFOV(
    ctx.camera.fov + fovChange,
    config.camera.fovMin,
    config.camera.fovMax
  );
  ctx.camera.updateProjectionMatrix();
  updateMaterialsForCurrentCamera(ctx);
  return true;
}
