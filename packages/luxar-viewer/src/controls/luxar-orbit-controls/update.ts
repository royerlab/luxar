/**
 * Per-frame update sequencer for LuxarOrbitControls.
 * Extracted from `luxar-orbit-controls.ts` so the orchestrator stays
 * focused on lifecycle, public API surface, and DOM bindings.
 *
 * Applies (in order):
 *   1. Auto-rotation (about the configured camera-frame axis)
 *   2. Trackball-rotation damping
 *   3. View-axis roll damping
 *   4. Pan damping
 *   5. Zoom damping
 *   6/7. Distance / ortho-zoom clamping
 *   8. Camera transform
 *   9. Change-detection event
 *
 * Returns true when the camera actually moved — including an
 * orthographic zoom-only frame, which changes the view without touching
 * position or orientation (useful for render-on-demand). Uses
 * module-local scratch vectors/quaternions to preserve the
 * orchestrator's no-allocation pattern on the hot path.
 */

import * as THREE from 'three';
import type { LuxarCamera } from '../../utils/camera-utils';
import { applyZoomScale } from './math/zoom';
import { autoRotateAxisVector } from './math/auto-rotate';
import { applyToCamera } from './camera-application';
import type { AutoRotateAxis } from '../types';

const _IDENTITY_QUAT = new THREE.Quaternion();
const _v2 = new THREE.Vector3();
const _q1 = new THREE.Quaternion();

/**
 * State and callbacks {@link runUpdateStep} needs, projected from the
 * `LuxarOrbitControls` orchestrator. Object refs (orientation, the
 * rotation/pan deltas, target, last-frame position/quaternion) are mutated
 * in place; scalar accumulators (roll/zoom delta, distance) are read/written
 * through accessors so the orchestrator keeps ownership of the fields.
 */
export interface OrbitUpdateCtx {
  enableRotate: boolean;
  enableDamping: boolean;
  dampingFactor: number;
  autoRotate: boolean;
  autoRotateSpeed: number;
  /** Camera-frame axis the turntable revolves around (see {@link AutoRotateAxis}). */
  autoRotateAxis: AutoRotateAxis;

  orientation: THREE.Quaternion;
  rotationDelta: THREE.Quaternion;
  panDelta: THREE.Vector3;
  target: THREE.Vector3;

  getRollDelta: () => number;
  setRollDelta: (v: number) => void;
  getZoomDelta: () => number;
  setZoomDelta: (v: number) => void;
  getDistance: () => number;
  setDistance: (v: number) => void;

  camera: LuxarCamera;
  minDistance: number;
  maxDistance: number;
  minZoom: number;
  maxZoom: number;

  lastPosition: THREE.Vector3;
  lastQuaternion: THREE.Quaternion;

  dispatch: (type: 'change') => void;
}

/**
 * Run one per-frame orbit update: apply auto-rotation, then the damped
 * rotation / roll / pan / zoom deltas, clamp distance (and ortho zoom), write
 * the transform to the camera, and detect movement. With damping enabled each
 * delta is applied by `dampingFactor` and decayed by the complement; without
 * it each is applied fully and reset.
 *
 * @param deltaTime - Seconds since the last frame (defaults to 1/60), keeping
 *   auto-rotation frame-rate independent.
 * @returns true if the camera position, orientation, or orthographic zoom
 *   changed this frame (a `change` event is dispatched in that case) —
 *   useful for render-on-demand. Ortho zoom mutates only `camera.zoom`,
 *   so it is tracked separately from the position/orientation compare.
 */
export function runUpdateStep(ctx: OrbitUpdateCtx, deltaTime?: number): boolean {
  // Ortho zoom lives on camera.zoom (steps 5/7 mutate it in place without
  // moving the camera), so snapshot it here for the step-9 change test —
  // consumers like the scene manager's material refresh and pick-buffer
  // invalidation rely on `change` firing for the damped zoom tail.
  const zoomBefore = ctx.camera instanceof THREE.OrthographicCamera ? ctx.camera.zoom : null;

  // 1. Auto-rotation: around the chosen camera-frame axis — screen-up
  // (vertical, the default), screen-right (a tumble), or the view direction
  // (a pure roll). Speed=1.0 → one full rotation in 60 seconds (matches
  // THREE.js OrbitControls convention)
  if (ctx.autoRotate && ctx.enableRotate) {
    const dt = deltaTime ?? 1 / 60;
    const angle = ((2 * Math.PI) / 60) * ctx.autoRotateSpeed * dt;
    // Inline quaternion math (applyOrbitRotation also calls applyToCamera, redundant in update())
    autoRotateAxisVector(ctx.autoRotateAxis, ctx.orientation, _v2);
    _q1.setFromAxisAngle(_v2, angle);
    ctx.orientation.premultiply(_q1);
    ctx.orientation.normalize();
  }

  // 2. Apply trackball rotation with damping (local frame)
  if (ctx.enableDamping) {
    _q1.slerpQuaternions(_IDENTITY_QUAT, ctx.rotationDelta, ctx.dampingFactor);
    ctx.orientation.multiply(_q1);
    ctx.orientation.normalize();
    ctx.rotationDelta.slerp(_IDENTITY_QUAT, ctx.dampingFactor);
  } else {
    ctx.orientation.multiply(ctx.rotationDelta);
    ctx.orientation.normalize();
    ctx.rotationDelta.identity();
  }

  // 3. Apply view-axis roll with damping
  const rollDelta = ctx.getRollDelta();
  if (Math.abs(rollDelta) > 1e-6) {
    _v2.set(0, 0, -1).applyQuaternion(ctx.orientation).normalize();
    if (ctx.enableDamping) {
      const rollApply = rollDelta * ctx.dampingFactor;
      _q1.setFromAxisAngle(_v2, rollApply);
      ctx.orientation.premultiply(_q1);
      ctx.orientation.normalize();
      ctx.setRollDelta(rollDelta * (1 - ctx.dampingFactor));
    } else {
      _q1.setFromAxisAngle(_v2, rollDelta);
      ctx.orientation.premultiply(_q1);
      ctx.orientation.normalize();
      ctx.setRollDelta(0);
    }
  }

  // 4. Apply pan with damping
  if (ctx.enableDamping) {
    ctx.target.addScaledVector(ctx.panDelta, ctx.dampingFactor);
    ctx.panDelta.multiplyScalar(1 - ctx.dampingFactor);
  } else {
    ctx.target.add(ctx.panDelta);
    ctx.panDelta.set(0, 0, 0);
  }

  // 5. Apply zoom with damping
  const zoomDelta = ctx.getZoomDelta();
  if (Math.abs(zoomDelta) > 1e-8) {
    if (ctx.enableDamping) {
      const zoomApply = 1 + zoomDelta * ctx.dampingFactor;
      ctx.setDistance(
        applyZoomScale(ctx.camera, ctx.getDistance(), zoomApply, ctx.minZoom, ctx.maxZoom)
      );
      ctx.setZoomDelta(zoomDelta * (1 - ctx.dampingFactor));
    } else {
      ctx.setDistance(
        applyZoomScale(ctx.camera, ctx.getDistance(), 1 + zoomDelta, ctx.minZoom, ctx.maxZoom)
      );
      ctx.setZoomDelta(0);
    }
  }

  // 6. Clamp distance
  ctx.setDistance(THREE.MathUtils.clamp(ctx.getDistance(), ctx.minDistance, ctx.maxDistance));

  // 7. Clamp ortho zoom
  if (ctx.camera instanceof THREE.OrthographicCamera) {
    ctx.camera.zoom = THREE.MathUtils.clamp(ctx.camera.zoom, ctx.minZoom, ctx.maxZoom);
    ctx.camera.updateProjectionMatrix();
  }

  // 8. Apply to camera
  applyToCamera(ctx.camera, ctx.target, ctx.orientation, ctx.getDistance());

  // 9. Change detection
  const moved =
    !ctx.camera.position.equals(ctx.lastPosition) ||
    !ctx.camera.quaternion.equals(ctx.lastQuaternion) ||
    (zoomBefore !== null && (ctx.camera as THREE.OrthographicCamera).zoom !== zoomBefore);

  if (moved) {
    ctx.dispatch('change');
    ctx.lastPosition.copy(ctx.camera.position);
    ctx.lastQuaternion.copy(ctx.camera.quaternion);
  }

  return moved;
}
