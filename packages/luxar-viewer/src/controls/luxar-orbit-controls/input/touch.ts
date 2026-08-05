/**
 * Touch-gesture handlers for LuxarOrbitControls.
 * Extracted from `luxar-orbit-controls.ts` so the orchestrator stays
 * focused on its sequenced update step and the lifecycle of its bound
 * handlers.
 *
 * Operates on the same `OrbitInputCtx` as the pointer handlers since
 * touch state lives in the same pointer arrays / accumulators.
 */

import { type OrbitInputCtx, pointerNDC } from './pointer';
import { computeArcballRotation } from '../math/trackball';

/**
 * Set the gesture from the current touch count. One finger starts rotate (or
 * pan if rotation is disabled), seeding the matching start point; two fingers
 * start a combined dolly-pan, seeding `dollyStart` with the pinch distance
 * and `panStart` with the two-finger midpoint.
 */
export function handleTouchStart(ctx: OrbitInputCtx): void {
  if (ctx.pointers.length === 1) {
    // Single finger: rotate (or pan if rotation disabled)
    if (ctx.enableRotate) {
      ctx.setState('rotate');
      ctx.rotateStart.copy(pointerNDC(ctx.pointers[0], ctx.domElement));
    } else if (ctx.enablePan) {
      ctx.setState('pan');
      ctx.panStart.set(ctx.pointers[0].clientX, ctx.pointers[0].clientY);
    }
  } else if (ctx.pointers.length === 2) {
    // Two fingers: dolly-pan
    ctx.setState('zoom'); // Combined dolly + pan
    const dx = ctx.pointers[0].clientX - ctx.pointers[1].clientX;
    const dy = ctx.pointers[0].clientY - ctx.pointers[1].clientY;
    ctx.dollyStart.set(0, Math.sqrt(dx * dx + dy * dy));
    // Pan center
    ctx.panStart.set(
      (ctx.pointers[0].clientX + ctx.pointers[1].clientX) * 0.5,
      (ctx.pointers[0].clientY + ctx.pointers[1].clientY) * 0.5
    );
  }
}

/**
 * Advance the active touch gesture. One-finger rotate accumulates an arcball
 * quaternion into `rotationDelta`; one-finger pan feeds the client delta to
 * `ctx.pan`. Two fingers do dolly + pan together: the change in pinch
 * distance accumulates a zoom delta (pinch-out = zoom in), and the change in
 * finger midpoint drives the pan.
 */
export function handleTouchMove(ctx: OrbitInputCtx, _event: PointerEvent): void {
  const state = ctx.getState();
  if (ctx.pointers.length === 1 && state === 'rotate') {
    const endNDC = pointerNDC(ctx.pointers[0], ctx.domElement);
    const deltaQuat = computeArcballRotation(
      ctx.rotateStart,
      endNDC,
      ctx.trackballRadius,
      ctx.rotateSpeed
    );
    ctx.rotationDelta.multiply(deltaQuat);
    ctx.rotateStart.copy(endNDC);
  } else if (ctx.pointers.length === 1 && state === 'pan') {
    const deltaX = ctx.pointers[0].clientX - ctx.panStart.x;
    const deltaY = ctx.pointers[0].clientY - ctx.panStart.y;
    ctx.pan(deltaX, deltaY);
    ctx.panStart.set(ctx.pointers[0].clientX, ctx.pointers[0].clientY);
  } else if (ctx.pointers.length >= 2) {
    // Two-finger dolly + pan
    const p0 = ctx.pointerPositions.get(ctx.pointers[0].pointerId);
    const p1 = ctx.pointerPositions.get(ctx.pointers[1].pointerId);
    if (!p0 || !p1) return;

    // Dolly (pinch)
    // Negate so pinch-out (fingers spread) = zoom in = negative zoomDelta,
    // consistent with scroll-up = zoom in = negative zoomDelta.
    const dx = p0.x - p1.x;
    const dy = p0.y - p1.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const dollyDelta = distance / ctx.dollyStart.y;
    if (dollyDelta > 0) {
      ctx.addZoomDelta(-(dollyDelta - 1));
    }
    ctx.dollyStart.set(0, distance);

    // Pan (two-finger drag)
    const centerX = (p0.x + p1.x) * 0.5;
    const centerY = (p0.y + p1.y) * 0.5;
    ctx.pan(centerX - ctx.panStart.x, centerY - ctx.panStart.y);
    ctx.panStart.set(centerX, centerY);
  }
}
