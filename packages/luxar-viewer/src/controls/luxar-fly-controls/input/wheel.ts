/**
 * Mouse-wheel handler for LuxarFlyControls.
 * Extracted from `luxar-fly-controls.ts` so the orchestrator stays
 * focused on lifecycle and the per-frame physics loop.
 *
 * - Plain scroll: forward/backward velocity impulse (scaled by the global
 *   wheel zoom sensitivity)
 * - Shift+scroll: roll (rotate around viewing axis)
 * - Ctrl/Meta+scroll: FOV (handled by InputHandler upstream, not
 *   intercepted here)
 */

import * as THREE from 'three';
import type { LuxarCamera } from '../../../utils/camera-utils';

// Module-local scratch to avoid per-event allocation.
const _v0 = new THREE.Vector3();
const _q0 = new THREE.Quaternion();

/**
 * State and callbacks the wheel handler needs, projected from the
 * `LuxarFlyControls` orchestrator. Object refs (camera, orientation,
 * velocity, angularVelocity) are mutated in place; `dispatch` routes the
 * `change` event back through the orchestrator.
 */
export interface FlyWheelCtx {
  enabled: boolean;
  inertialMode: boolean;
  movementSpeed: number;
  rotationSpeed: number;
  /**
   * Global per-machine multiplier on the plain-scroll forward/back impulse
   * (Settings > Input > Zoom Sensitivity; `config.controls.wheelZoomSensitivity`).
   * Shift+scroll roll is not scaled — it is not a zoom.
   */
  wheelZoomSensitivity: number;

  camera: LuxarCamera;
  orientation: THREE.Quaternion;
  velocity: THREE.Vector3;
  angularVelocity: THREE.Vector3;

  dispatch: (type: 'change') => void;
}

/**
 * Handle a scroll wheel event. Ctrl/Meta+scroll is left unhandled (ceded to
 * the upstream FOV handler). The sign of `deltaY` drives either roll about
 * the viewing axis (Shift held) or forward/backward motion (plain scroll):
 * inertial mode adds an angular/linear velocity impulse, non-inertial mode
 * applies the rotation or translation directly. Dispatches `change`; no-op
 * while disabled.
 */
export function handleWheel(ctx: FlyWheelCtx, event: WheelEvent): void {
  if (!ctx.enabled) return;

  // Let Ctrl/Meta+scroll pass through to InputHandler for FOV control
  if (event.ctrlKey || event.metaKey) return;

  event.preventDefault();

  // Normalize deltaY across browsers (line vs pixel vs page scrolling)
  const delta = -Math.sign(event.deltaY);

  if (event.shiftKey) {
    // Shift+scroll: roll around viewing axis
    _v0.set(0, 0, -1).applyQuaternion(ctx.orientation);
    const rollImpulse = delta * ctx.rotationSpeed * 0.06;

    if (ctx.inertialMode) {
      ctx.angularVelocity.addScaledVector(_v0, rollImpulse);
    } else {
      // Non-inertial: apply rotation directly
      _q0.setFromAxisAngle(_v0, rollImpulse);
      ctx.orientation.premultiply(_q0);
      ctx.orientation.normalize();
    }
  } else {
    // Plain scroll: move forward/backward
    _v0.set(0, 0, -1).applyQuaternion(ctx.orientation);
    const impulse = delta * ctx.movementSpeed * ctx.wheelZoomSensitivity * 0.3;

    if (ctx.inertialMode) {
      ctx.velocity.addScaledVector(_v0, impulse);
    } else {
      // Non-inertial: move directly
      ctx.camera.position.addScaledVector(_v0, impulse * 0.2);
    }
  }

  ctx.dispatch('change');
}
