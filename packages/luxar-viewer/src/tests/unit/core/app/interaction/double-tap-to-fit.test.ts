// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  installDoubleTapToFit,
  DOUBLE_TAP_MS,
  DOUBLE_TAP_SLOP_PX,
} from '../../../../../core/app/interaction/double-tap-to-fit';
import { TOUCH_CLICK_SLOP_PX } from '../../../../../core/app/interaction/picked-element-cache';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';

interface Harness {
  canvas: HTMLCanvasElement;
  events: EventGroup;
  fit: ReturnType<typeof vi.fn>;
  clock: { t: number };
}

function setup(): Harness {
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  const events = new EventGroup();
  const fit = vi.fn();
  const clock = { t: 1000 };
  installDoubleTapToFit(canvas, events, fit, () => clock.t);
  return { canvas, events, fit, clock };
}

function pointer(
  canvas: HTMLElement,
  type: 'pointerdown' | 'pointerup' | 'pointercancel' | 'pointerleave',
  opts: { x: number; y: number; pointerId?: number; pointerType?: string; button?: number }
): void {
  const { x, y, pointerId = 1, pointerType = 'touch', button = 0 } = opts;
  canvas.dispatchEvent(
    new PointerEvent(type, {
      pointerId,
      button,
      clientX: x,
      clientY: y,
      pointerType,
      bubbles: true,
    })
  );
}

/** A press that lifts `by` px away, `pointerType` defaulting to touch. */
function tap(
  canvas: HTMLElement,
  x: number,
  y: number,
  opts: { by?: number; pointerId?: number; pointerType?: string } = {}
): void {
  const { by = 0, pointerId = 1, pointerType = 'touch' } = opts;
  pointer(canvas, 'pointerdown', { x, y, pointerId, pointerType });
  pointer(canvas, 'pointerup', { x: x + by, y, pointerId, pointerType });
  // Chromium fires pointerleave right after a touch pointerup.
  pointer(canvas, 'pointerleave', { x: x + by, y, pointerId, pointerType });
}

beforeEach(() => {
  document.body.innerHTML = '';
});

describe('installDoubleTapToFit', () => {
  it('two touch taps within the window and slop re-frame once', () => {
    const h = setup();
    tap(h.canvas, 100, 80);
    h.clock.t += DOUBLE_TAP_MS - 50;
    tap(h.canvas, 100 + DOUBLE_TAP_SLOP_PX - 4, 80);
    expect(h.fit).toHaveBeenCalledTimes(1);
    // The pair is consumed: a third tap inside the window starts a new pair.
    h.clock.t += 50;
    tap(h.canvas, 100, 80);
    expect(h.fit).toHaveBeenCalledTimes(1);
  });

  it('two taps too far apart in time are two single taps', () => {
    const h = setup();
    tap(h.canvas, 100, 80);
    h.clock.t += DOUBLE_TAP_MS;
    tap(h.canvas, 100, 80);
    expect(h.fit).not.toHaveBeenCalled();
  });

  it('two taps too far apart on screen are two single taps', () => {
    const h = setup();
    tap(h.canvas, 100, 80);
    h.clock.t += 100;
    tap(h.canvas, 100 + DOUBLE_TAP_SLOP_PX + 1, 80);
    expect(h.fit).not.toHaveBeenCalled();
  });

  it('a press that drags past the tap slop is not a tap and resets the pair', () => {
    const h = setup();
    tap(h.canvas, 100, 80);
    h.clock.t += 100;
    tap(h.canvas, 100, 80, { by: TOUCH_CLICK_SLOP_PX + 1 });
    expect(h.fit).not.toHaveBeenCalled();
    // …so the next quick tap is the FIRST of a new pair, not a completion.
    h.clock.t += 100;
    tap(h.canvas, 100, 80);
    expect(h.fit).not.toHaveBeenCalled();
  });

  it('a long-press release is not half of a double-tap', () => {
    const h = setup();
    pointer(h.canvas, 'pointerdown', { x: 100, y: 80 });
    h.clock.t += 600;
    pointer(h.canvas, 'pointerup', { x: 100, y: 80 });
    h.clock.t += 120;
    tap(h.canvas, 100, 80);
    expect(h.fit).not.toHaveBeenCalled();
  });

  it('a pinch (two fingers) is never a tap, even when the fingers lift together', () => {
    const h = setup();
    pointer(h.canvas, 'pointerdown', { x: 60, y: 80, pointerId: 1 });
    pointer(h.canvas, 'pointerdown', { x: 140, y: 80, pointerId: 2 });
    pointer(h.canvas, 'pointerup', { x: 60, y: 80, pointerId: 1 });
    pointer(h.canvas, 'pointerleave', { x: 60, y: 80, pointerId: 1 });
    pointer(h.canvas, 'pointerup', { x: 140, y: 80, pointerId: 2 });
    pointer(h.canvas, 'pointerleave', { x: 140, y: 80, pointerId: 2 });
    h.clock.t += 100;
    // A single tap right after the pinch is a first tap, not a completion.
    tap(h.canvas, 100, 80);
    expect(h.fit).not.toHaveBeenCalled();
    h.clock.t += 100;
    tap(h.canvas, 100, 80);
    expect(h.fit).toHaveBeenCalledTimes(1);
  });

  it('a pinch resets a tap that preceded it', () => {
    const h = setup();
    tap(h.canvas, 100, 80);
    h.clock.t += 100;
    pointer(h.canvas, 'pointerdown', { x: 60, y: 80, pointerId: 1 });
    pointer(h.canvas, 'pointerdown', { x: 140, y: 80, pointerId: 2 });
    pointer(h.canvas, 'pointerup', { x: 60, y: 80, pointerId: 1 });
    pointer(h.canvas, 'pointerup', { x: 140, y: 80, pointerId: 2 });
    h.clock.t += 100;
    tap(h.canvas, 100, 80);
    expect(h.fit).not.toHaveBeenCalled();
  });

  it('a cancelled pointer drops out of the gesture', () => {
    const h = setup();
    pointer(h.canvas, 'pointerdown', { x: 100, y: 80 });
    pointer(h.canvas, 'pointercancel', { x: 100, y: 80 });
    h.clock.t += 100;
    tap(h.canvas, 100, 80);
    expect(h.fit).not.toHaveBeenCalled();
  });

  it('mouse double-clicks stay inert', () => {
    const h = setup();
    tap(h.canvas, 100, 80, { pointerType: 'mouse' });
    h.clock.t += 100;
    tap(h.canvas, 100, 80, { pointerType: 'mouse' });
    expect(h.fit).not.toHaveBeenCalled();
  });

  it('a secondary-button touch-like press does not count', () => {
    const h = setup();
    pointer(h.canvas, 'pointerdown', { x: 100, y: 80, button: 2 });
    pointer(h.canvas, 'pointerup', { x: 100, y: 80, button: 2 });
    h.clock.t += 100;
    tap(h.canvas, 100, 80);
    expect(h.fit).not.toHaveBeenCalled();
  });

  it('dispose removes the listeners', () => {
    const h = setup();
    h.events.dispose();
    tap(h.canvas, 100, 80);
    h.clock.t += 100;
    tap(h.canvas, 100, 80);
    expect(h.fit).not.toHaveBeenCalled();
  });
});
