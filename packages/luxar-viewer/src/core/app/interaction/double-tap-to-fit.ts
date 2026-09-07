/**
 * Double-tap to re-frame — the touch counterpart of the Home key (F).
 *
 * Deliberately NOT part of `canvas-actions.ts`. That module is installed by
 * the picking session, which is provisioned only for scenes with per-element
 * labels, keys or interaction templates (or an embedder consumer) — while a
 * re-frame must work on ANY scene, a bare point cloud included. So this is
 * installed once per app, for the canvas's lifetime, with no dependency on
 * picking. `canvas-actions` still recognises the second tap of a double-tap,
 * but only to cancel the first tap's deferred navigation, never to re-frame.
 *
 * Touch-like pointers only ({@link isTouchLikePointer}): a mouse double-click
 * stays inert, as it always was. A tap is a primary press that lifts within
 * {@link TOUCH_CLICK_SLOP_PX} with no second finger down at any point during
 * the gesture — a pinch is two fingers, not two taps — and a drag resets the
 * tap history, so drag-then-tap is one tap, not two.
 */

import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { isTouchLikePointer } from '../../../utils/input-capabilities';
import { LONG_PRESS_MS } from '../../../utils/long-press';
import { TOUCH_CLICK_SLOP_PX } from './picked-element-cache';

/**
 * Two taps at most this far apart in time are a double-tap. Tap navigation
 * waits this long only after the async pick/readback and label fetch finish.
 */
export const DOUBLE_TAP_MS = 300;
/** …and at most this far apart on screen (a finger re-lands imprecisely). */
export const DOUBLE_TAP_SLOP_PX = 24;

/**
 * Listen on `canvas` for a touch double-tap and call `fit`. Listeners are
 * registered through `events`, so the owner's dispose tears them down.
 */
export function installDoubleTapToFit(
  canvas: HTMLElement,
  events: EventGroup,
  fit: () => void,
  now: () => number = () => performance.now()
): void {
  const down = new Map<number, { x: number; y: number; t: number }>();
  let multiTouch = false;
  let lastTap: { t: number; x: number; y: number } | null = null;

  const forget = (id: number): void => {
    down.delete(id);
    if (down.size === 0) multiTouch = false;
  };

  events.on(canvas, 'pointerdown', (e) => {
    const ev = e as PointerEvent;
    if (!isTouchLikePointer(ev) || ev.button !== 0) return;
    down.set(ev.pointerId, { x: ev.clientX, y: ev.clientY, t: now() });
    if (down.size > 1) multiTouch = true;
  });

  const onPointerGone = (e: Event): void => forget((e as PointerEvent).pointerId);
  events.on(canvas, 'pointercancel', onPointerGone);
  events.on(canvas, 'pointerleave', onPointerGone);

  events.on(canvas, 'pointerup', (e) => {
    const ev = e as PointerEvent;
    const start = down.get(ev.pointerId);
    // Read the latch BEFORE releasing this pointer: lifting the last finger of
    // a pinch clears it, and this release must still read as a pinch.
    const wasMultiTouch = multiTouch;
    forget(ev.pointerId);
    if (!start) return;
    if (wasMultiTouch || down.size > 0) {
      lastTap = null;
      return;
    }
    if (Math.hypot(ev.clientX - start.x, ev.clientY - start.y) > TOUCH_CLICK_SLOP_PX) {
      lastTap = null; // a drag: whatever tap preceded it is not half of a double-tap
      return;
    }
    const t = now();
    if (t - start.t >= LONG_PRESS_MS) {
      lastTap = null;
      return;
    }
    if (
      lastTap &&
      t - lastTap.t < DOUBLE_TAP_MS &&
      Math.hypot(ev.clientX - lastTap.x, ev.clientY - lastTap.y) <= DOUBLE_TAP_SLOP_PX
    ) {
      lastTap = null;
      fit();
      return;
    }
    lastTap = { t, x: ev.clientX, y: ev.clientY };
  });
}
