/**
 * Camera-mode helpers extracted from SceneManager.
 *
 * Owns the perspective ↔ orthographic camera-swap policy and the
 * wiring of the swap into the control-type setter. SceneManager
 * keeps the public `setControlType()` method (so the
 * `camera-changed` event dispatches at the class call site — helpers
 * never dispatch on the host's event channel) but delegates the body
 * to `setControlType()` here.
 *
 * The host owns the `camera` field; the ctx exposes a
 * `getCamera() / setCamera()` pair so helpers can read/replace it
 * across the swap without taking a back-pointer to SceneManager.
 *
 * @module scene/scene-manager/camera/camera-mode
 */

import * as THREE from 'three';
import { config } from '../../../config';
import type { ControlsManager, ControlType } from '../../../controls/controls-manager';
import type { PostProcessingManager } from '../../../rendering';
import type { Renderer } from '../../../rendering/renderer-capabilities';
import {
  type LuxarCamera,
  isOrthographicCamera,
  isPerspectiveCamera,
} from '../../../utils/camera-utils';

/**
 * Ctx supplied by SceneManager. `getCamera()` returns the current
 * camera reference (which may change across a swap), `setCamera()`
 * replaces the host's field.
 */
export interface CameraModeCtx {
  /** Read the current camera. Reflects the latest value after setCamera(). */
  getCamera(): LuxarCamera;
  /** Replace the host's camera field. */
  setCamera(camera: LuxarCamera): void;
  readonly controls: ControlsManager;
  readonly renderer: Renderer;
  readonly postProcessing: PostProcessingManager;
  /** Push current camera into materials. Called after each successful swap. */
  updateMaterialsForCurrentCamera(): void;
  /** Update the cached ortho zoom on the host after a perspective→ortho swap. */
  setLastOrthoZoom(zoom: number): void;
}

/**
 * Switch camera control type. Performs perspective ↔ orthographic
 * swap if needed, then forwards the type change to ControlsManager
 * and refreshes material projection parameters.
 *
 * @returns `{ cameraChanged: true }` if the camera object was
 *   replaced; caller is expected to dispatch the `camera-changed`
 *   event in that case. (Event dispatch stays at the SceneManager
 *   call site — helpers never dispatch on the host's event channel.)
 */
export function setControlType(type: ControlType, ctx: CameraModeCtx): { cameraChanged: boolean } {
  const needsOrtho = type === 'ortho';
  const hasOrtho = isOrthographicCamera(ctx.getCamera());
  const cameraChanged = needsOrtho !== hasOrtho;

  if (needsOrtho && !hasOrtho) {
    swapToOrthographic(ctx);
  } else if (!needsOrtho && hasOrtho) {
    swapToPerspective(ctx);
  }

  ctx.controls.setCamera(ctx.getCamera());
  ctx.controls.setControlType(type);
  ctx.updateMaterialsForCurrentCamera();

  return { cameraChanged };
}

/**
 * Swap perspective → orthographic, matching the visible frustum
 * at the current target distance. Resets to a clean front view
 * (down -Z, up = Y) — ortho mode is for 2D viewing, so carrying
 * over a tilted 3D orientation is confusing.
 *
 * No-op when the current camera is already orthographic.
 */
export function swapToOrthographic(ctx: CameraModeCtx): void {
  const camera = ctx.getCamera();
  if (!isPerspectiveCamera(camera)) return;

  const focusTarget = ctx.controls.getFocusTarget();
  // SCENE-RELATIVE floor (mirrors the orbit-controls re-init/reset fix):
  // an absolute 0.001 would mis-frame sub-milli-unit scenes on a
  // perspective->ortho switch. The active controls' minDistance is
  // scene-diagonal-derived (deriveScaleLimits / auto-frame); 0.001 stays
  // only as the last resort when no positive floor is known.
  const activeControls = ctx.controls.getControls?.();
  const minDist =
    activeControls && 'minDistance' in activeControls && activeControls.minDistance > 0
      ? activeControls.minDistance
      : 0.001;
  const distance = Math.max(camera.position.distanceTo(focusTarget), minDist);
  const fovRad = (camera.fov * Math.PI) / 180;
  const frustumHeight = 2 * distance * Math.tan(fovRad / 2);
  const aspect = camera.aspect || 1;

  const ortho = new THREE.OrthographicCamera(
    (-frustumHeight * aspect) / 2,
    (frustumHeight * aspect) / 2,
    frustumHeight / 2,
    -frustumHeight / 2,
    camera.near,
    camera.far
  );

  ortho.position.set(focusTarget.x, focusTarget.y, focusTarget.z + distance);
  ortho.up.set(0, 1, 0);
  ortho.lookAt(focusTarget);
  ortho.updateMatrixWorld();

  ctx.setCamera(ortho);
  ctx.setLastOrthoZoom(ortho.zoom);
  ctx.postProcessing.setCamera(ortho);
}

/**
 * Swap orthographic → perspective. Restores the default FOV
 * (from `config.renderingControls.defaults.fov`); preserves
 * position, orientation and up vector from the ortho camera.
 *
 * No-op when the current camera is already perspective.
 */
export function swapToPerspective(ctx: CameraModeCtx): void {
  const camera = ctx.getCamera();
  if (isPerspectiveCamera(camera)) return;

  const canvas = ctx.renderer.domElement;
  const aspect =
    (canvas.clientWidth || window.innerWidth) / (canvas.clientHeight || window.innerHeight);

  const persp = new THREE.PerspectiveCamera(
    config.renderingControls.defaults.fov,
    aspect,
    camera.near,
    camera.far
  );

  persp.position.copy(camera.position);
  persp.quaternion.copy(camera.quaternion);
  persp.up.copy(camera.up);
  persp.updateMatrixWorld();

  ctx.setCamera(persp);
  ctx.postProcessing.setCamera(persp);
}
