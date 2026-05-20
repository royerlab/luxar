/**
 * Pan math (vendored from THREE.js OrbitControls).
 * Extracted from `luxar-orbit-controls.ts` so the orchestrator file
 * stays focused on its sequenced update step and DOM lifecycle.
 *
 * Mutates a caller-supplied accumulator (`out`) instead of returning
 * fresh Vector3s, so the orchestrator's per-frame allocation pattern
 * is preserved.
 */

import * as THREE from 'three';
import type { LuxarCamera } from '../../../utils/camera-utils';

// Module-local scratch vector to avoid per-call allocation. Mirrors the
// `_v` scratch the orchestrator uses for its other math.
const _v = new THREE.Vector3();

export interface PanCtx {
  camera: LuxarCamera;
  distance: number;
  panSpeed: number;
  screenSpacePanning: boolean;
  domElement: HTMLElement;
}

/** Accumulate a pan along the camera's X axis (negative distance). */
export function applyPanLeft(
  out: THREE.Vector3,
  distance: number,
  objectMatrix: THREE.Matrix4
): void {
  _v.setFromMatrixColumn(objectMatrix, 0); // camera X axis
  _v.multiplyScalar(-distance);
  out.add(_v);
}

/** Accumulate a pan along the camera's Y axis (or world up if non-screen-space). */
export function applyPanUp(
  out: THREE.Vector3,
  distance: number,
  objectMatrix: THREE.Matrix4,
  cameraUp: THREE.Vector3,
  screenSpacePanning: boolean
): void {
  if (screenSpacePanning) {
    _v.setFromMatrixColumn(objectMatrix, 1); // camera Y axis
  } else {
    _v.setFromMatrixColumn(objectMatrix, 0);
    _v.crossVectors(cameraUp, _v);
  }
  _v.multiplyScalar(distance);
  out.add(_v);
}

/** Accumulate the full pan delta for an X/Y pointer-move pair. */
export function applyPan(out: THREE.Vector3, deltaX: number, deltaY: number, ctx: PanCtx): void {
  if (ctx.camera instanceof THREE.PerspectiveCamera) {
    const fovRad = ctx.camera.fov * (Math.PI / 180);
    const height = 2 * ctx.distance * Math.tan(fovRad / 2);
    applyPanLeft(
      out,
      (deltaX * height * ctx.panSpeed) / ctx.domElement.clientHeight,
      ctx.camera.matrix
    );
    applyPanUp(
      out,
      (deltaY * height * ctx.panSpeed) / ctx.domElement.clientHeight,
      ctx.camera.matrix,
      ctx.camera.up,
      ctx.screenSpacePanning
    );
  } else {
    const cam = ctx.camera as THREE.OrthographicCamera;
    applyPanLeft(
      out,
      (deltaX * (cam.right - cam.left) * ctx.panSpeed) / cam.zoom / ctx.domElement.clientWidth,
      ctx.camera.matrix
    );
    applyPanUp(
      out,
      (deltaY * (cam.top - cam.bottom) * ctx.panSpeed) / cam.zoom / ctx.domElement.clientHeight,
      ctx.camera.matrix,
      ctx.camera.up,
      ctx.screenSpacePanning
    );
  }
}
