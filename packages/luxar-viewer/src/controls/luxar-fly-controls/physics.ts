/**
 * Translation + rotation physics integration for LuxarFlyControls.
 * Extracted from the per-frame `update()` body in `luxar-fly-controls.ts`
 * so the orchestrator stays focused on lifecycle.
 *
 * `integrateTranslation` advances velocity from WASD input, applies
 * damping, integrates into camera.position, then zeros velocities below
 * a threshold. `integrateRotation` does the same for arrow-key/Q-E look
 * input via angular velocity and quaternion premultiply.
 *
 * Both return whether they produced motion this frame; the orchestrator
 * dispatches `change` if either reports true.
 */

import * as THREE from 'three';
import type { LuxarCamera } from '../../utils/camera-utils';
import { config } from '../../config';
import type { FlyMoveState, FlyLookState } from './input/keyboard';

// Module-local scratch — owned by the physics step.
const _v0 = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _q0 = new THREE.Quaternion();

/**
 * State the physics integrators read and mutate, projected from the
 * `LuxarFlyControls` orchestrator: the camera and orientation quaternion,
 * the translational/angular velocity accumulators, the current move/look
 * input, and the mode/damping/speed knobs (including the transient
 * speed-boost flag).
 */
export interface FlyPhysicsCtx {
  camera: LuxarCamera;
  orientation: THREE.Quaternion;
  velocity: THREE.Vector3;
  angularVelocity: THREE.Vector3;
  moveState: FlyMoveState;
  lookState: FlyLookState;

  inertialMode: boolean;
  damping: number;
  rotationDamping: number;
  movementSpeed: number;
  rotationSpeed: number;
  speedBoost: boolean;
}

/**
 * Advance translation velocity (WASD + Alt+W/S) and integrate it into
 * camera.position. Returns whether the body moved this frame.
 */
export function integrateTranslation(ctx: FlyPhysicsCtx, delta: number): boolean {
  // Get movement vectors from orientation quaternion for consistency
  // This ensures movement is perfectly tied to the control model
  _v0.set(0, 0, -1).applyQuaternion(ctx.orientation).normalize();
  _v1.set(1, 0, 0).applyQuaternion(ctx.orientation).normalize();
  _v2.set(0, 1, 0); // Keep world up for vertical rise/fall

  // Determine effective damping based on mode
  // Non-inertial mode uses high damping for immediate response
  const effectiveDamping = ctx.inertialMode ? ctx.damping : 0.5;

  // Apply speed boost multiplier (2x speed when Shift is held)
  const speedMultiplier = ctx.speedBoost ? 2.0 : 1.0;

  // Always use physics-based movement (unified approach)
  // Calculate acceleration from input.
  // movementSpeed is the user-facing "speed" parameter (controlled by UI slider
  // and scale-aware system).
  _v3.set(0, 0, 0);
  _v3.addScaledVector(
    _v0,
    (ctx.moveState.forward - ctx.moveState.back) * ctx.movementSpeed * speedMultiplier
  );
  _v3.addScaledVector(
    _v1,
    (ctx.moveState.right - ctx.moveState.left) * ctx.movementSpeed * speedMultiplier
  );
  _v3.addScaledVector(
    _v2,
    (ctx.moveState.up - ctx.moveState.down) * ctx.movementSpeed * speedMultiplier
  );

  // Update velocity
  ctx.velocity.addScaledVector(_v3, delta);

  // Apply damping
  ctx.velocity.multiplyScalar(
    Math.pow(effectiveDamping, delta * config.controls.fly.physics.dampingPower)
  );

  // Apply velocity to position
  ctx.camera.position.addScaledVector(ctx.velocity, delta);

  // Check if we're still moving (using configured threshold)
  if (ctx.velocity.length() < config.controls.fly.physics.velocityThreshold) {
    ctx.velocity.set(0, 0, 0);
    return false;
  }
  return true;
}

/**
 * Advance angular velocity from look state (arrow keys + Q/E roll) and
 * apply it to the orientation quaternion. Damping always runs. Returns
 * whether the orientation changed this frame.
 */
export function integrateRotation(ctx: FlyPhysicsCtx, delta: number): boolean {
  const effectiveRotationDamping = ctx.inertialMode ? ctx.rotationDamping : 0.5;

  // Handle angular velocity for rotation with arrow keys and Q/E roll
  // True airplane-like fly controls: all rotations relative to camera's local axes
  if (ctx.lookState.horizontal !== 0 || ctx.lookState.vertical !== 0 || ctx.lookState.roll !== 0) {
    // Get camera's local axes in world space
    // These define the rotation axes for consistent airplane-like controls
    _v0.set(1, 0, 0).applyQuaternion(ctx.orientation);
    _v1.set(0, 1, 0).applyQuaternion(ctx.orientation);
    _v2.set(0, 0, -1).applyQuaternion(ctx.orientation);

    if (ctx.inertialMode) {
      // Apply angular acceleration (torque)
      _v3.set(0, 0, 0);
      // Pitch: rotate around camera's local right axis (negative for correct up/down)
      _v3.addScaledVector(_v0, -ctx.lookState.vertical * ctx.rotationSpeed);
      // Yaw: rotate around camera's local up axis
      _v3.addScaledVector(_v1, -ctx.lookState.horizontal * ctx.rotationSpeed);
      // Roll: rotate around camera's local forward axis
      _v3.addScaledVector(_v2, ctx.lookState.roll * ctx.rotationSpeed);

      // Add torque to world-space angular velocity
      ctx.angularVelocity.addScaledVector(_v3, delta);
    } else {
      // Non-inertial: directly set angular velocity
      ctx.angularVelocity.set(0, 0, 0);
      ctx.angularVelocity.addScaledVector(_v0, -ctx.lookState.vertical * ctx.rotationSpeed);
      ctx.angularVelocity.addScaledVector(_v1, -ctx.lookState.horizontal * ctx.rotationSpeed);
      ctx.angularVelocity.addScaledVector(_v2, ctx.lookState.roll * ctx.rotationSpeed);
    }
  }
  // In non-inertial mode without input, high damping below stops rotation quickly.

  // Apply angular velocity to orientation
  let rotated = false;
  const angularSpeed = ctx.angularVelocity.length();
  if (angularSpeed > config.controls.fly.physics.angularVelocityThreshold) {
    // Create rotation from angular velocity
    const angle = angularSpeed * delta;
    _v3.copy(ctx.angularVelocity).normalize();
    _q0.setFromAxisAngle(_v3, angle);

    // Apply WORLD-space delta rotation (pre-multiply)
    ctx.orientation.premultiply(_q0);
    ctx.orientation.normalize();

    rotated = true;
  }

  // Apply angular damping to world-space angular velocity
  ctx.angularVelocity.multiplyScalar(
    Math.pow(effectiveRotationDamping, delta * config.controls.fly.physics.dampingPower)
  );

  // Stop tiny rotations
  if (ctx.angularVelocity.length() < config.controls.fly.physics.angularVelocityThreshold) {
    ctx.angularVelocity.set(0, 0, 0);
  }

  return rotated;
}
