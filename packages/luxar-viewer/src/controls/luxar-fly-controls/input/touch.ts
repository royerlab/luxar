/**
 * Touch handlers for LuxarFlyControls — the finger vocabulary of fly mode.
 *
 * Fly mode had no touch path at all: its listeners are legacy `mousedown` /
 * `mousemove` / `mouseup`, a touch drag produces no `mousemove`, and the only
 * thing a tap produced was one compatibility `mousedown`/`mouseup` pair. One
 * of three advertised camera modes did nothing on a phone or tablet.
 *
 * The model mirrors the orbit controls' finger count, so a user learns one
 * vocabulary:
 *   - 1 finger drag        → look (the right-drag angular impulse)
 *   - 2 fingers, midpoint  → strafe (the left-drag screen-plane translation)
 *   - 2 fingers, pinch     → forward / back thrust along the view direction
 *   - 2 fingers, twist     → roll about the view direction
 * All from the same pair of positions, per frame, through the SAME physics
 * state the mouse path drives (`velocity`, `angularVelocity`, `orientation`),
 * so inertia and damping behave identically.
 *
 * Touch `pointerdown` is `preventDefault`ed: the browser's compatibility mouse
 * events would otherwise reach the mouse path and start a phantom left-drag
 * strafe under every tap.
 */

import * as THREE from 'three';
import type { LuxarCamera } from '../../../utils/camera-utils';

// Module-local scratch vectors to avoid per-event allocation.
const _x = new THREE.Vector3();
const _y = new THREE.Vector3();
const _fwd = new THREE.Vector3();

/** Geometry of the first two fingers, remembered between moves. */
export interface FlyPinchState {
  distance: number;
  angle: number;
  midX: number;
  midY: number;
}

/**
 * State and callbacks the touch handlers need, projected from the
 * `LuxarFlyControls` orchestrator. Object refs are mutated in place; the
 * pinch snapshot is read/written through accessors.
 */
export interface FlyTouchCtx {
  enabled: boolean;
  inertialMode: boolean;
  lookSpeed: number;
  movementSpeed: number;

  camera: LuxarCamera;
  orientation: THREE.Quaternion;
  velocity: THREE.Vector3;
  angularVelocity: THREE.Vector3;

  /** Live finger positions by pointerId; insertion order is finger order. */
  pointers: Map<number, THREE.Vector2>;
  getPinch: () => FlyPinchState | null;
  setPinch: (p: FlyPinchState | null) => void;

  dispatch: (type: 'change' | 'start' | 'end') => void;
}

/** Look impulse per pixel of one-finger drag — the mouse right-drag factor. */
export const TOUCH_LOOK_GAIN = 2.5;
/** Strafe distance per pixel of two-finger drag — the mouse left-drag factor. */
export const TOUCH_STRAFE_SCALE = 0.005;
/**
 * Thrust per unit of `ln(distance / previous distance)`, times `movementSpeed`.
 * Logarithmic so a pinch from 100 → 200 px and one from 200 → 400 px travel
 * the same distance, and pinch-in exactly undoes pinch-out.
 */
export const PINCH_THRUST_GAIN = 2.0;
/** Radians of roll impulse per radian of two-finger twist. */
export const FLY_TWIST_ROLL_GAIN = 1.0;
/**
 * Sign that makes the SCENE follow the fingers (same reasoning as the orbit
 * twist): y-down screen `atan2` grows clockwise, a positive rotation about the
 * forward axis turns the camera clockwise as the user sees it.
 */
export const FLY_TWIST_ROLL_SIGN = -1;

/** Wrap an angle difference into (-π, π] so a twist across ±π does not jump. */
function wrapAngle(delta: number): number {
  if (!Number.isFinite(delta)) return 0;
  let d = delta;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d <= -Math.PI) d += 2 * Math.PI;
  return d;
}

function firstTwo(ctx: FlyTouchCtx): [THREE.Vector2, THREE.Vector2] | null {
  if (ctx.pointers.size < 2) return null;
  const it = ctx.pointers.values();
  return [it.next().value as THREE.Vector2, it.next().value as THREE.Vector2];
}

function pinchGeometry(a: THREE.Vector2, b: THREE.Vector2): FlyPinchState {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return {
    distance: Math.sqrt(dx * dx + dy * dy),
    angle: Math.atan2(dy, dx),
    midX: (a.x + b.x) * 0.5,
    midY: (a.y + b.y) * 0.5,
  };
}

/** Re-snapshot the pinch from whatever fingers are down (null below two). */
function reseedPinch(ctx: FlyTouchCtx): void {
  const pair = firstTwo(ctx);
  ctx.setPinch(pair ? pinchGeometry(pair[0], pair[1]) : null);
}

/** One-finger drag: the same angular impulse a mouse right-drag applies. */
function look(ctx: FlyTouchCtx, deltaX: number, deltaY: number): boolean {
  const scale = ctx.lookSpeed * TOUCH_LOOK_GAIN;
  if (scale === 0) return false;
  _x.set(1, 0, 0).applyQuaternion(ctx.orientation);
  _y.set(0, 1, 0).applyQuaternion(ctx.orientation);
  ctx.angularVelocity.addScaledVector(_x, -deltaY * scale);
  ctx.angularVelocity.addScaledVector(_y, -deltaX * scale);
  return true;
}

/** Two-finger midpoint drag: the same screen-plane translation as a left-drag. */
function strafe(ctx: FlyTouchCtx, deltaX: number, deltaY: number): boolean {
  const scale = ctx.movementSpeed * TOUCH_STRAFE_SCALE;
  if (scale === 0) return false;
  _x.set(1, 0, 0).applyQuaternion(ctx.orientation);
  _y.set(0, 1, 0).applyQuaternion(ctx.orientation);
  const target = ctx.inertialMode ? ctx.velocity : ctx.camera.position;
  target.addScaledVector(_x, -deltaX * scale);
  target.addScaledVector(_y, deltaY * scale);
  return true;
}

/** Pinch: thrust along the view direction; twist: roll about it. */
function thrustAndRoll(ctx: FlyTouchCtx, prev: FlyPinchState, next: FlyPinchState): boolean {
  if (prev.distance <= 0 || next.distance <= 0) return false; // coincident fingers: no orientation
  _fwd.set(0, 0, -1).applyQuaternion(ctx.orientation);
  const thrust = Math.log(next.distance / prev.distance) * ctx.movementSpeed * PINCH_THRUST_GAIN;
  let changed = false;
  if (thrust !== 0) {
    (ctx.inertialMode ? ctx.velocity : ctx.camera.position).addScaledVector(_fwd, thrust);
    changed = true;
  }
  const twist = wrapAngle(next.angle - prev.angle);
  if (twist !== 0) {
    ctx.angularVelocity.addScaledVector(_fwd, FLY_TWIST_ROLL_SIGN * FLY_TWIST_ROLL_GAIN * twist);
    changed = true;
  }
  return changed;
}

/**
 * A touch-like finger landed: track it, (re)snapshot the pinch, and start the
 * gesture when it is the first finger. Cancels the compatibility mouse events.
 */
export function handleTouchDown(ctx: FlyTouchCtx, event: PointerEvent): void {
  if (!ctx.enabled) return;
  event.preventDefault();
  const first = ctx.pointers.size === 0;
  ctx.pointers.set(event.pointerId, new THREE.Vector2(event.clientX, event.clientY));
  reseedPinch(ctx);
  if (first) ctx.dispatch('start');
}

/**
 * A tracked finger moved. One finger looks; two or more strafe by the midpoint
 * delta, thrust by the pinch ratio and roll by the twist — all against the
 * previous snapshot, which is then advanced. Dispatches `change` only when an
 * impulse was applied.
 */
export function handleTouchMove(ctx: FlyTouchCtx, event: PointerEvent): void {
  const pos = ctx.pointers.get(event.pointerId);
  if (!pos) return;
  if (!ctx.enabled) {
    pos.set(event.clientX, event.clientY);
    reseedPinch(ctx);
    return;
  }
  let changed = false;
  if (ctx.pointers.size === 1) {
    const deltaX = event.clientX - pos.x;
    const deltaY = event.clientY - pos.y;
    if (deltaX !== 0 || deltaY !== 0) {
      changed = look(ctx, deltaX, deltaY);
    }
    pos.set(event.clientX, event.clientY);
  } else {
    pos.set(event.clientX, event.clientY);
    const prev = ctx.getPinch();
    const pair = firstTwo(ctx);
    if (prev && pair) {
      const next = pinchGeometry(pair[0], pair[1]);
      const deltaX = next.midX - prev.midX;
      const deltaY = next.midY - prev.midY;
      if (deltaX !== 0 || deltaY !== 0) {
        changed = strafe(ctx, deltaX, deltaY);
      }
      changed = thrustAndRoll(ctx, prev, next) || changed;
      ctx.setPinch(next);
    }
  }
  if (changed) ctx.dispatch('change');
}

/**
 * A tracked finger lifted (or the browser cancelled it). The survivors
 * re-snapshot the pinch (2 → 1 continues as a look from the survivor's own last
 * position, so nothing jumps); the last finger ends the gesture. Runs even while
 * disabled so a finger that lifts after `enabled = false` is not tracked forever.
 */
export function handleTouchUp(ctx: FlyTouchCtx, event: PointerEvent): void {
  if (!ctx.pointers.delete(event.pointerId)) return;
  reseedPinch(ctx);
  if (ctx.pointers.size === 0) ctx.dispatch('end');
}
