/**
 * JSON-serialisable snapshot of viewer state.
 *
 * Captures the visible camera placement and the current per-dimension
 * slice position so an external caller (a test, a "share view" link,
 * a regression harness) can reproduce a specific view across reloads.
 *
 * Layer-panel state and rendering-controls settings are *not* included
 * in v1 — they live on different abstractions (LayerStateManager and
 * the rendering-controls settings persistence path) and have their own
 * serialise/restore routes; bundling them here would couple this module
 * to the UI layer for no immediate gain. Adding them later is additive
 * (new fields under a higher `version`).
 */

import type { SceneManager } from '../../../scene/scene-manager';
import { isPerspectiveCamera, isOrthographicCamera } from '../../../utils/camera-utils';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import { log, Modules } from '../../../utils/log';
import * as THREE from 'three';

/** Schema version so future fields can be added without breaking old snapshots. */
export const VIEWER_SNAPSHOT_VERSION = 1 as const;

export interface CameraSnapshot {
  position: readonly [number, number, number];
  target: readonly [number, number, number];
  up: readonly [number, number, number];
  isOrtho: boolean;
  /** Vertical FOV in degrees. Present only for perspective cameras. */
  fov?: number;
  /** Camera zoom factor. Meaningful for orthographic cameras. */
  zoom?: number;
  near: number;
  far: number;
}

export interface DimsSnapshot {
  /** Total dimensions in the source dataset. Used to reject mismatched restores. */
  ndim: number;
  /** Indices currently shown on screen, length 1–3. */
  displayed: readonly number[];
  /** Current slice position per dimension, length === ndim. */
  currentStep: readonly number[];
}

export interface ViewerSnapshot {
  version: typeof VIEWER_SNAPSHOT_VERSION;
  camera: CameraSnapshot;
  dims?: DimsSnapshot;
}

/**
 * Capture the current camera + dim state into a JSON-serialisable object.
 *
 * If no scene has been loaded yet, `dims` is omitted (the scene-dims
 * manager has nothing meaningful to report).
 */
export function captureSnapshot(sceneManager: SceneManager): ViewerSnapshot {
  const camera = sceneManager.camera;
  const target = sceneManager.controls.getFocusTarget();

  const cameraSnapshot: CameraSnapshot = {
    position: [camera.position.x, camera.position.y, camera.position.z],
    target: [target.x, target.y, target.z],
    up: [camera.up.x, camera.up.y, camera.up.z],
    isOrtho: isOrthographicCamera(camera),
    near: camera.near,
    far: camera.far,
    ...(isPerspectiveCamera(camera) ? { fov: camera.fov } : {}),
    ...(isOrthographicCamera(camera) ? { zoom: camera.zoom } : {}),
  };

  const dims = sceneDimsManager.getDims();
  const dimsSnapshot: DimsSnapshot | undefined = dims
    ? {
        ndim: dims.ndim,
        displayed: [...dims.displayed],
        currentStep: [...dims.currentStep],
      }
    : undefined;

  return {
    version: VIEWER_SNAPSHOT_VERSION,
    camera: cameraSnapshot,
    ...(dimsSnapshot ? { dims: dimsSnapshot } : {}),
  };
}

/**
 * Write a {@link CameraSnapshot} back onto the live camera + controls.
 *
 * Position / up / near / far and the projection-specific parameter (fov for
 * perspective, zoom for ortho) are applied in place; then the controls are
 * re-targeted and re-initialised so subsequent orbit/fly updates don't snap
 * the camera back. Shared by {@link restoreSnapshot} and the embedder API's
 * `setCameraPose()`.
 */
export function restoreCamera(sceneManager: SceneManager, cam: CameraSnapshot): void {
  const camera = sceneManager.camera;

  camera.position.set(cam.position[0], cam.position[1], cam.position[2]);
  camera.up.set(cam.up[0], cam.up[1], cam.up[2]);
  camera.near = cam.near;
  camera.far = cam.far;
  if (isPerspectiveCamera(camera) && cam.fov !== undefined) {
    camera.fov = cam.fov;
  }
  if (isOrthographicCamera(camera) && cam.zoom !== undefined) {
    camera.zoom = cam.zoom;
  }
  camera.updateProjectionMatrix();

  // setTarget mirrors the orbit/fly difference internally; reinitialize()
  // re-derives orbit distance/orientation from the new (position, target).
  sceneManager.controls.setTarget(new THREE.Vector3(cam.target[0], cam.target[1], cam.target[2]));
  sceneManager.controls.reinitialize();
}

/**
 * Restore camera + dim state from a snapshot.
 *
 * - The camera position, target, up, near/far, and projection-specific
 *   parameters (fov for perspective, zoom for ortho) are written back
 *   in place. After updating, the controls are re-initialised so
 *   subsequent orbit/fly updates don't snap the camera back.
 * - Dims restore is best-effort: if the snapshot's `ndim` does not match
 *   the loaded dataset, the dims block is skipped with a warning. The
 *   `displayed` set is not changed (re-displaying dims would re-frame the
 *   scene); only the per-dim `currentStep` values are forwarded to
 *   `sceneDimsManager.setDimensionValue`.
 *
 * Returns the parts that were actually applied so callers can confirm.
 */
export function restoreSnapshot(
  sceneManager: SceneManager,
  snapshot: ViewerSnapshot
): { cameraApplied: boolean; dimsApplied: boolean } {
  if (snapshot.version !== VIEWER_SNAPSHOT_VERSION) {
    log.warning(
      Modules.SCENE_MANAGER,
      `ViewerSnapshot version ${snapshot.version} does not match expected ${VIEWER_SNAPSHOT_VERSION}; skipping restore`
    );
    return { cameraApplied: false, dimsApplied: false };
  }

  restoreCamera(sceneManager, snapshot.camera);

  let dimsApplied = false;
  if (snapshot.dims) {
    const dims = sceneDimsManager.getDims();
    if (!dims) {
      log.warning(
        Modules.SCENE_MANAGER,
        'ViewerSnapshot has dims but no scene is loaded yet; skipping dims restore'
      );
    } else if (dims.ndim !== snapshot.dims.ndim) {
      log.warning(
        Modules.SCENE_MANAGER,
        `ViewerSnapshot ndim=${snapshot.dims.ndim} does not match loaded scene ndim=${dims.ndim}; skipping dims restore`
      );
    } else {
      const target = snapshot.dims.currentStep;
      const limit = Math.min(target.length, dims.ndim);
      for (let i = 0; i < limit; i++) {
        if (target[i] !== dims.currentStep[i]) {
          sceneDimsManager.setDimensionValue(i, target[i]);
        }
      }
      dimsApplied = true;
    }
  }

  return { cameraApplied: true, dimsApplied };
}
