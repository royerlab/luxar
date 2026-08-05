/**
 * Mouse drag handlers for LuxarFlyControls.
 * Extracted from `luxar-fly-controls.ts` so the orchestrator stays
 * focused on lifecycle and the per-frame physics loop.
 *
 * Left-drag = strafe (screen-space translation).
 * Right-drag = rotate (look around via angular impulse).
 */

import * as THREE from 'three';
import type { LuxarCamera } from '../../../utils/camera-utils';
import type { FlyMouseAction } from './keyboard';

// Module-local scratch vectors to avoid per-event allocation.
const _v0 = new THREE.Vector3();
const _v1 = new THREE.Vector3();

/**
 * State and callbacks the mouse handlers need, projected from the
 * `LuxarFlyControls` orchestrator. Object refs (camera, orientation,
 * velocity, angularVelocity) are mutated in place; primitive drag state
 * (active action, last mouse X/Y) is read/written through accessors, and
 * `dispatch` routes events back through the orchestrator.
 */
export interface FlyMouseCtx {
  enabled: boolean;
  inertialMode: boolean;
  lookSpeed: number;
  movementSpeed: number;

  camera: LuxarCamera;
  orientation: THREE.Quaternion;
  velocity: THREE.Vector3;
  angularVelocity: THREE.Vector3;

  getActiveMouseAction: () => FlyMouseAction;
  setActiveMouseAction: (v: FlyMouseAction) => void;
  getMouseX: () => number;
  setMouseX: (v: number) => void;
  getMouseY: () => number;
  setMouseY: (v: number) => void;

  dispatch: (type: 'change' | 'start' | 'end') => void;
}

/**
 * Begin a mouse drag: left button (0) starts a strafe, right button (2)
 * starts a rotate. Records the button as the active action, seeds the last
 * mouse position, prevents the default, and dispatches `start`. No-op while
 * disabled or for any other button.
 */
export function handleMouseDown(ctx: FlyMouseCtx, event: MouseEvent): void {
  if (!ctx.enabled) return;

  // Left button (0) = strafe, Right button (2) = rotate
  if (event.button === 0) {
    ctx.setActiveMouseAction('strafe');
    ctx.setMouseX(event.clientX);
    ctx.setMouseY(event.clientY);
    event.preventDefault();
    ctx.dispatch('start');
  } else if (event.button === 2) {
    ctx.setActiveMouseAction('rotate');
    ctx.setMouseX(event.clientX);
    ctx.setMouseY(event.clientY);
    event.preventDefault();
    ctx.dispatch('start');
  }
}

/**
 * End a mouse drag when the released button matches the active action
 * (left↔strafe, right↔rotate): clears the active action and dispatches
 * `end`. Ignores releases that don't match the in-progress gesture.
 */
export function handleMouseUp(ctx: FlyMouseCtx, event: MouseEvent): void {
  if (!ctx.enabled) return;

  const action = ctx.getActiveMouseAction();
  if ((event.button === 0 && action === 'strafe') || (event.button === 2 && action === 'rotate')) {
    ctx.setActiveMouseAction('none');
    ctx.dispatch('end');
  }
}

/**
 * Apply the active drag to the camera from the pointer delta since the last
 * move. Rotate (right-drag) adds a pitch/yaw angular impulse about the
 * camera's local axes; strafe (left-drag) translates in the camera's
 * screen plane — as a velocity impulse in inertial mode, or directly on
 * `camera.position` otherwise. Dispatches `change`; no-op when disabled or
 * no drag is active.
 */
export function handleMouseMove(ctx: FlyMouseCtx, event: MouseEvent): void {
  const action = ctx.getActiveMouseAction();
  if (!ctx.enabled || action === 'none') return;

  const deltaX = event.clientX - ctx.getMouseX();
  const deltaY = event.clientY - ctx.getMouseY();

  ctx.setMouseX(event.clientX);
  ctx.setMouseY(event.clientY);

  if (action === 'rotate') {
    // Right-drag: apply angular impulse for rotation (look around)
    const torquePitch = -deltaY * ctx.lookSpeed * 2.5;
    const torqueYaw = -deltaX * ctx.lookSpeed * 2.5;

    _v0.set(1, 0, 0).applyQuaternion(ctx.orientation);
    _v1.set(0, 1, 0).applyQuaternion(ctx.orientation);

    ctx.angularVelocity.addScaledVector(_v0, torquePitch);
    ctx.angularVelocity.addScaledVector(_v1, torqueYaw);
  } else if (action === 'strafe') {
    // Left-drag: screen-space translation (strafe up/down/left/right)
    // Drag direction matches on-screen movement, consistent with pan in orbit/ortho.
    _v0.set(1, 0, 0).applyQuaternion(ctx.orientation);
    _v1.set(0, 1, 0).applyQuaternion(ctx.orientation);

    // Scale by movementSpeed for scene-appropriate sensitivity.
    // The 0.005 factor converts pixel deltas to reasonable world-space distances.
    const strafeScale = ctx.movementSpeed * 0.005;

    if (ctx.inertialMode) {
      // Inertial: add velocity impulse
      ctx.velocity.addScaledVector(_v0, -deltaX * strafeScale);
      ctx.velocity.addScaledVector(_v1, deltaY * strafeScale);
    } else {
      // Non-inertial: move directly
      ctx.camera.position.addScaledVector(_v0, -deltaX * strafeScale);
      ctx.camera.position.addScaledVector(_v1, deltaY * strafeScale);
    }
  }

  ctx.dispatch('change');
}
