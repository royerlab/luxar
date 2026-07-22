/**
 * Camera transform application for orbit controls.
 * Extracted from `luxar-orbit-controls.ts` so the orchestrator stays
 * focused on its sequenced update step and DOM lifecycle.
 *
 * `applyToCamera` is the hot path (called every frame from `update()`)
 * and uses a module-local scratch vector to match the orchestrator's
 * no-allocation pattern. `initializeFromCamera` is cold (constructor /
 * reset / reinitialize) and allocates freely.
 */

import * as THREE from 'three';
import type { LuxarCamera } from '../../utils/camera-utils';

const _v = new THREE.Vector3();

/** Apply orientation + distance + target to camera transform. */
export function applyToCamera(
  camera: LuxarCamera,
  target: THREE.Vector3,
  orientation: THREE.Quaternion,
  distance: number
): void {
  _v.set(0, 0, distance).applyQuaternion(orientation);
  camera.position.copy(target).add(_v);
  camera.up.set(0, 1, 0).applyQuaternion(orientation);
  camera.lookAt(target);
  // Ensure camera.matrix is up-to-date (needed by pan math which reads matrix columns)
  camera.updateMatrixWorld();
}

/**
 * Extract orientation and distance from current camera state and write
 * them into the supplied accumulators. Returns the new distance value
 * since `number` isn't passed by reference.
 *
 * `minDistance` is the caller's SCENE-RELATIVE orbit floor (the class's
 * own zoom lower bound — scene diagonal × minDistanceFactor once scale
 * limits are known). It guards the degenerate camera == target case
 * without imposing an absolute world-unit scale: the old fixed 0.001
 * floor flung the camera 1000× out of a tiny-unit scene (diagonal
 * ~1e-6) on every controls re-init / mode switch. 0.001 remains only as
 * the last-resort fallback when no positive floor is supplied (no scene
 * bounds known yet).
 */
export function initializeFromCamera(
  camera: LuxarCamera,
  target: THREE.Vector3,
  outOrientation: THREE.Quaternion,
  minDistance: number = 0
): number {
  const offset = new THREE.Vector3().subVectors(camera.position, target);
  const distance = Math.max(offset.length(), minDistance > 0 ? minDistance : 0.001);

  // Derive up from camera quaternion rather than camera.up — the quaternion
  // is always authoritative, whereas camera.up may be stale (fly controls
  // only update quaternion, not up). Prevents roll loss on fly→orbit switch.
  // lookAt() has a singularity when the view direction is parallel to the up vector.
  // Detect this and use a fallback up vector to prevent NaN.
  const viewDir = offset.clone().normalize();
  let up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
  const upDot = Math.abs(viewDir.dot(up));
  if (upDot > 0.999) {
    // Near singularity: pick a fallback up vector perpendicular to view direction
    up = Math.abs(viewDir.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
  }

  const lookMatrix = new THREE.Matrix4().lookAt(camera.position, target, up);
  outOrientation.setFromRotationMatrix(lookMatrix);
  return distance;
}
