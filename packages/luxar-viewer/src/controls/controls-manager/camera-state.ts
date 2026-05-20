/**
 * Capture and restore the camera-shaped state shared across control modes.
 * Extracted from `controls-manager.ts` so the orchestrator stays focused
 * on lifecycle and dispatch.
 *
 * `saveCameraState` snapshots position / rotation / up / target before
 * disposing one control instance; `restoreCameraState` re-derives the
 * new instance's internal orbit state from those snapshots. Fly controls
 * initialise themselves from the camera, so restore is a no-op there —
 * the snapshot still matters because it provides a sensible pivot point
 * for the *next* orbit/ortho switch out of fly mode.
 */

import * as THREE from 'three';
import { LuxarOrbitControls } from '../luxar-orbit-controls';
import type { LuxarFlyControls } from '../luxar-fly-controls';
import type { LuxarCamera } from '../../utils/camera-utils';

export interface CameraStateCtx {
  camera: LuxarCamera;
  currentControls: LuxarOrbitControls | LuxarFlyControls | null;
  sceneScale: number;
  savedCameraPosition: THREE.Vector3;
  savedCameraRotation: THREE.Euler;
  savedCameraUp: THREE.Vector3;
  savedTarget: THREE.Vector3;
}

export function saveCameraState(ctx: CameraStateCtx): void {
  ctx.savedCameraPosition.copy(ctx.camera.position);
  ctx.savedCameraRotation.copy(ctx.camera.rotation);
  ctx.savedCameraUp.copy(ctx.camera.up);

  // Save target for all control types.
  // For orbit/ortho: use the explicit orbit target.
  // For fly: derive from camera look direction so switching to orbit/ortho
  // gets a sensible pivot point (not a stale target from a previous mode).
  if (ctx.currentControls instanceof LuxarOrbitControls) {
    ctx.savedTarget.copy(ctx.currentControls.target);
  } else {
    const forward = new THREE.Vector3();
    ctx.camera.getWorldDirection(forward);
    ctx.savedTarget.copy(ctx.camera.position).add(forward.multiplyScalar(ctx.sceneScale || 10));
  }
}

export function restoreCameraState(ctx: CameraStateCtx): void {
  if (ctx.currentControls instanceof LuxarOrbitControls) {
    // Set the target, then re-derive orientation from the current camera state
    // (important: the constructor initialized with target=(0,0,0), which is wrong)
    ctx.currentControls.target.copy(ctx.savedTarget);
    ctx.currentControls.reinitialize();
    ctx.currentControls.update();
  }
  // Fly controls automatically initialize from current camera state
}
