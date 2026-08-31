/**
 * Per-frame update sequencer for LuxarOrbitControls.
 * Extracted from `luxar-orbit-controls.ts` so the orchestrator stays
 * focused on lifecycle, public API surface, and DOM bindings.
 *
 * Applies (in order):
 *   1. Auto-rotation (about the configured camera-frame or fixed scene axis)
 *   2. Auto-dolly (sinusoidal distance oscillation)
 *   3. Trackball-rotation damping
 *   4. View-axis roll damping
 *   5. Pan damping
 *   6. Zoom damping
 *   7/8. Distance / ortho-zoom clamping
 *   9. Camera transform
 *   10. Change-detection event
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
import { advanceDollyPhase, dollyScale } from './math/auto-dolly';
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
  /** Camera-frame or fixed scene axis (see {@link AutoRotateAxis}). */
  autoRotateAxis: AutoRotateAxis;
  /** Whether zoom input is allowed at all — also gates the auto-dolly. */
  enableZoom: boolean;
  autoDolly: boolean;
  /** Peak dolly swing as a fraction of distance (0.15 = ±15%). */
  autoDollyAmplitude: number;
  /** Seconds per full dolly oscillation. */
  autoDollyPeriod: number;

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
  getDollyPhase: () => number;
  setDollyPhase: (v: number) => void;

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
 *   auto-rotation and the auto-dolly frame-rate independent.
 * @returns true if the camera position, orientation, or orthographic zoom
 *   changed this frame (a `change` event is dispatched in that case) —
 *   useful for render-on-demand. Ortho zoom mutates only `camera.zoom`,
 *   so it is tracked separately from the position/orientation compare.
 */
export function runUpdateStep(ctx: OrbitUpdateCtx, deltaTime?: number): boolean {
  // Ortho zoom lives on camera.zoom (steps 2/6/8 mutate it in place without
  // moving the camera), so snapshot it here for the step-10 change test —
  // consumers like the scene manager's material refresh and pick-buffer
  // invalidation rely on `change` firing for the damped zoom tail.
  const zoomBefore = ctx.camera instanceof THREE.OrthographicCamera ? ctx.camera.zoom : null;

  // 1. Auto-rotation around the chosen camera-frame or fixed scene axis (see
  // AutoRotateAxis). `autoRotateSpeed` is REVOLUTIONS PER MINUTE: the turn
  // rate is (2π/60)·speed rad/s, so a full turn takes 60/speed seconds
  // (speed 1.0 → 60 s, measured). Matches the THREE.js OrbitControls
  // convention, and is the unit `auto_rotate_speed` carries on disk — the UI
  // shows the equivalent period instead (see secondsPerTurnFromRpm).
  if (ctx.autoRotate && ctx.enableRotate) {
    const dt = deltaTime ?? 1 / 60;
    const angle = ((2 * Math.PI) / 60) * ctx.autoRotateSpeed * dt;
    // Inline quaternion math (applyOrbitRotation also calls applyToCamera, redundant in update())
    autoRotateAxisVector(ctx.autoRotateAxis, ctx.orientation, _v2);
    _q1.setFromAxisAngle(_v2, angle);
    ctx.orientation.premultiply(_q1);
    ctx.orientation.normalize();
  }

  // 2. Auto-dolly: breathe the camera toward and away from the target on a
  // sine, in LOG distance, so it is the sinusoidal mousewheel it is named for
  // (see math/auto-dolly.ts). Applied straight to the distance rather than
  // through `zoomDelta` for the same reason auto-rotation bypasses
  // `rotationDelta`: routing driven motion through the damping filter would
  // low-pass it, shrinking the amplitude and lagging the phase behind the
  // period the user asked for. `applyZoomScale` carries the ortho branch, so
  // this modulates `camera.zoom` in 2D mode — where `enableRotate` is false
  // and the turntable is inert, but zoom is exactly what "closer" means.
  if (ctx.autoDolly && ctx.enableZoom) {
    const dt = deltaTime ?? 1 / 60;
    const phase = ctx.getDollyPhase();
    const nextPhase = advanceDollyPhase(phase, dt, ctx.autoDollyPeriod);
    const scale = dollyScale(phase, nextPhase, ctx.autoDollyAmplitude);
    if (scale !== 1) {
      ctx.setDistance(
        applyZoomScale(ctx.camera, ctx.getDistance(), scale, ctx.minZoom, ctx.maxZoom)
      );
    }
    ctx.setDollyPhase(nextPhase);
  }

  // 3. Apply trackball rotation with damping (local frame)
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

  // 4. Apply view-axis roll with damping
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

  // 5. Apply pan with damping
  if (ctx.enableDamping) {
    ctx.target.addScaledVector(ctx.panDelta, ctx.dampingFactor);
    ctx.panDelta.multiplyScalar(1 - ctx.dampingFactor);
  } else {
    ctx.target.add(ctx.panDelta);
    ctx.panDelta.set(0, 0, 0);
  }

  // 6. Apply zoom with damping
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

  // 7. Clamp distance
  ctx.setDistance(THREE.MathUtils.clamp(ctx.getDistance(), ctx.minDistance, ctx.maxDistance));

  // 8. Clamp ortho zoom
  if (ctx.camera instanceof THREE.OrthographicCamera) {
    ctx.camera.zoom = THREE.MathUtils.clamp(ctx.camera.zoom, ctx.minZoom, ctx.maxZoom);
    ctx.camera.updateProjectionMatrix();
  }

  // 9. Apply to camera
  applyToCamera(ctx.camera, ctx.target, ctx.orientation, ctx.getDistance());

  // 10. Change detection
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
