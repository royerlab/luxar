/**
 * Touch-gesture handlers for LuxarOrbitControls.
 * Extracted from `luxar-orbit-controls.ts` so the orchestrator stays
 * focused on its sequenced update step and the lifecycle of its bound
 * handlers.
 *
 * Operates on the same `OrbitInputCtx` as the pointer handlers since
 * touch state lives in the same pointer arrays / accumulators.
 *
 * Gesture vocabulary (the one the mobile UI is built on):
 *   - 1 finger drag  → rotate (arcball), or pan when rotation is disabled
 *   - 2 fingers      → pinch = dolly, midpoint drag = pan, twist = view-axis
 *                      roll — all three at once, from one pair of positions
 * Lifting a finger out of a multi-finger gesture re-seeds the survivors
 * (`handleTouchStart` is re-run by `handlePointerUp`) instead of ending it.
 */

import * as THREE from 'three';
import { type OrbitInputCtx, pointerNDC } from './pointer';
import { computeArcballRotation } from '../math/trackball';

/**
 * Radians of camera roll per radian of two-finger twist. 1 = the scene turns
 * with the fingers; the damping in update() smooths pinch jitter, which is
 * small because fingers moving radially barely change the inter-finger angle.
 */
export const TWIST_ROLL_GAIN = 1.0;

/**
 * Sign that makes the SCENE follow the fingers. Screen space is y-down, so
 * `atan2` grows for a visually clockwise twist; a positive roll about the view
 * axis turns the camera clockwise as the user sees it, i.e. the scene
 * counter-clockwise — hence the negation.
 */
export const TWIST_ROLL_SIGN = -1;

/** Wrap an angle difference into (-π, π] so a twist across ±π does not jump. */
export function wrapAngle(delta: number): number {
  let d = delta;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d <= -Math.PI) d += 2 * Math.PI;
  return d;
}

/** Client coordinates → normalized device coordinates on `domElement`. */
function clientNDC(clientX: number, clientY: number, domElement: HTMLElement): THREE.Vector2 {
  const rect = domElement.getBoundingClientRect();
  return new THREE.Vector2(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -((clientY - rect.top) / rect.height) * 2 + 1
  );
}

/**
 * The i-th tracked finger's LIVE position when known, else its pointerdown
 * snapshot. Live matters when seeding runs because another finger lifted: the
 * survivors have moved since they landed, and seeding from the stale snapshot
 * would make the scene jump on the next move.
 */
function fingerPosition(ctx: OrbitInputCtx, index: number): { x: number; y: number } {
  const snapshot = ctx.pointers[index];
  const live = ctx.pointerPositions.get(snapshot.pointerId);
  return { x: live?.x ?? snapshot.clientX, y: live?.y ?? snapshot.clientY };
}

/** Geometry of a finger pair: separation distance, angle, midpoint. */
function pairGeometry(
  p0: { x: number; y: number },
  p1: { x: number; y: number }
): { distance: number; angle: number; midX: number; midY: number } {
  const dx = p0.x - p1.x;
  const dy = p0.y - p1.y;
  return {
    distance: Math.sqrt(dx * dx + dy * dy),
    angle: Math.atan2(dy, dx),
    midX: (p0.x + p1.x) * 0.5,
    midY: (p0.y + p1.y) * 0.5,
  };
}

/** One finger: rotate (or pan when rotation is disabled), seeded at its position. */
function seedSingleFinger(ctx: OrbitInputCtx): void {
  const { x, y } = fingerPosition(ctx, 0);
  if (ctx.enableRotate) {
    ctx.setState('rotate');
    ctx.rotateStart.copy(clientNDC(x, y, ctx.domElement));
  } else if (ctx.enablePan) {
    ctx.setState('pan');
    ctx.panStart.set(x, y);
  }
}

/**
 * Two fingers: combined dolly + pan + twist. `dollyStart` carries the pinch
 * distance in `.y` and the inter-finger angle in `.x` (unused by the mouse
 * dolly path, which reads `.y` only); `panStart` carries the midpoint.
 */
function seedTwoFingers(ctx: OrbitInputCtx): void {
  ctx.setState('zoom');
  const g = pairGeometry(fingerPosition(ctx, 0), fingerPosition(ctx, 1));
  ctx.dollyStart.set(g.angle, g.distance);
  ctx.panStart.set(g.midX, g.midY);
}

/**
 * Set the gesture from the current touch count and seed its start points.
 * Also re-run when a finger lifts out of a multi-finger gesture, so the
 * survivors seed the continuing gesture from their CURRENT positions.
 */
export function handleTouchStart(ctx: OrbitInputCtx): void {
  if (ctx.pointers.length === 1) seedSingleFinger(ctx);
  else if (ctx.pointers.length === 2) seedTwoFingers(ctx);
}

/** One-finger rotate: accumulate an arcball quaternion and re-anchor. */
function moveRotate(ctx: OrbitInputCtx): void {
  const endNDC = pointerNDC(ctx.pointers[0], ctx.domElement);
  const deltaQuat = computeArcballRotation(
    ctx.rotateStart,
    endNDC,
    ctx.trackballRadius,
    ctx.rotateSpeed
  );
  ctx.rotationDelta.multiply(deltaQuat);
  ctx.rotateStart.copy(endNDC);
}

/** One-finger pan: feed the client delta and re-anchor. */
function movePan(ctx: OrbitInputCtx): void {
  const { clientX, clientY } = ctx.pointers[0];
  ctx.pan(clientX - ctx.panStart.x, clientY - ctx.panStart.y);
  ctx.panStart.set(clientX, clientY);
}

/**
 * Two fingers: the change in pinch distance accumulates a zoom delta
 * (pinch-out = zoom in = negative, consistent with scroll-up), the change in
 * inter-finger angle accumulates a view-axis roll (when rotation is enabled),
 * and the change in midpoint drives the pan.
 */
function moveTwoFingers(ctx: OrbitInputCtx): void {
  const p0 = ctx.pointerPositions.get(ctx.pointers[0].pointerId);
  const p1 = ctx.pointerPositions.get(ctx.pointers[1].pointerId);
  if (!p0 || !p1) return;
  const g = pairGeometry(p0, p1);

  const dollyDelta = g.distance / ctx.dollyStart.y;
  if (dollyDelta > 0) ctx.addZoomDelta(-(dollyDelta - 1));

  if (ctx.enableRotate) {
    const twist = wrapAngle(g.angle - ctx.dollyStart.x);
    if (twist !== 0) ctx.addRollDelta(TWIST_ROLL_SIGN * TWIST_ROLL_GAIN * twist);
  }
  ctx.dollyStart.set(g.angle, g.distance);

  ctx.pan(g.midX - ctx.panStart.x, g.midY - ctx.panStart.y);
  ctx.panStart.set(g.midX, g.midY);
}

/** Advance the active touch gesture (see the module header for the vocabulary). */
export function handleTouchMove(ctx: OrbitInputCtx, _event: PointerEvent): void {
  const state = ctx.getState();
  if (ctx.pointers.length === 1 && state === 'rotate') moveRotate(ctx);
  else if (ctx.pointers.length === 1 && state === 'pan') movePan(ctx);
  else if (ctx.pointers.length >= 2) moveTwoFingers(ctx);
}
