/**
 * Camera ↔ orientation sync for LuxarFlyControls.
 * Extracted from `luxar-fly-controls.ts` so the orchestrator stays
 * focused on lifecycle and the per-frame physics loop.
 *
 * Parallel to `luxar-orbit-controls/camera-application.ts` in spirit —
 * but fly controls drive the camera directly from a quaternion (no
 * separate target / distance pair), so the API is narrower.
 */

import * as THREE from 'three';
import type { LuxarCamera } from '../../utils/camera-utils';

/** Initialize orientation from the current camera quaternion. */
export function initializeFromCamera(camera: LuxarCamera, orientation: THREE.Quaternion): void {
  // Copy the camera's current orientation
  orientation.copy(camera.quaternion);
}

/**
 * Sync the camera transform from the orientation quaternion. Keeps
 * camera.up in sync so state export and non-screenSpacePanning pan work
 * correctly in adjacent controls.
 */
export function updateOrientation(camera: LuxarCamera, orientation: THREE.Quaternion): void {
  camera.quaternion.copy(orientation);
  camera.up.set(0, 1, 0).applyQuaternion(orientation);
}

/**
 * Smoothly slerp the orientation toward a quaternion that looks at
 * `target` from the current camera position, then sync the camera.
 *
 * @param smoothness - 0 = snap, 1 = no change.
 * @param up - The up direction the new orientation keeps. Defaults to the
 *   camera's own `up`, which for fly controls is the current up (roll
 *   included): a look-at turns the view without dropping the roll. It was a
 *   hard-coded world +Y, so re-targeting (a restored pose, a framing on a
 *   Z-up scene) silently replaced the authored up.
 */
export function lookAtSmooth(
  camera: LuxarCamera,
  orientation: THREE.Quaternion,
  target: THREE.Vector3,
  smoothness: number,
  up: THREE.Vector3 = camera.up
): void {
  // Calculate desired look direction
  const direction = new THREE.Vector3();
  direction.subVectors(target, camera.position);
  direction.normalize();

  // Create a quaternion that looks in the target direction
  const targetQuaternion = new THREE.Quaternion();
  const tempMatrix = new THREE.Matrix4();
  tempMatrix.lookAt(camera.position, target, up);
  targetQuaternion.setFromRotationMatrix(tempMatrix);

  // Smoothly interpolate to target orientation
  orientation.slerp(targetQuaternion, 1 - smoothness);

  updateOrientation(camera, orientation);
}
