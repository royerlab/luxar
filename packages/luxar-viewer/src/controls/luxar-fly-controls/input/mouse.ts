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

export function handleMouseUp(ctx: FlyMouseCtx, event: MouseEvent): void {
  if (!ctx.enabled) return;

  const action = ctx.getActiveMouseAction();
  if ((event.button === 0 && action === 'strafe') || (event.button === 2 && action === 'rotate')) {
    ctx.setActiveMouseAction('none');
    ctx.dispatch('end');
  }
}

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
