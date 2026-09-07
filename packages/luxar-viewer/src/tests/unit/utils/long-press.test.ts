// @vitest-environment jsdom
/**
 * Unit tests for `utils/long-press.ts`.
 *
 * The timer is faked; `performance.now()` is not, which is fine because the
 * click-swallow window is measured against real time and these tests run in
 * far under 600 ms.
 */

import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { attachLongPress, LONG_PRESS_MS, LONG_PRESS_SLOP_PX } from '../../../utils/long-press';
import {
  resetInputProfileForTests,
  setInputProfileOverride,
} from '../../../utils/input-capabilities';

function pointer(
  type: string,
  init: { id?: number; x?: number; y?: number; pointerType?: string; button?: number } = {}
): PointerEvent {
  return new PointerEvent(type, {
    pointerId: init.id ?? 1,
    clientX: init.x ?? 50,
    clientY: init.y ?? 50,
    pointerType: init.pointerType ?? 'touch',
    button: init.button ?? 0,
    bubbles: true,
    cancelable: true,
  });
}

describe('attachLongPress', () => {
  let el: HTMLDivElement;
  let child: HTMLButtonElement;
  let onLongPress: Mock<(clientX: number, clientY: number, event: PointerEvent) => boolean>;
  let dispose: () => void;

  beforeEach(() => {
    vi.useFakeTimers();
    resetInputProfileForTests();
    el = document.createElement('div');
    child = document.createElement('button');
    el.appendChild(child);
    document.body.appendChild(el);
    onLongPress = vi.fn(() => true);
    dispose = attachLongPress(el, { onLongPress });
  });

  afterEach(() => {
    dispose();
    vi.useRealTimers();
    document.body.innerHTML = '';
    resetInputProfileForTests();
  });

  it('fires once after the hold duration at the press position, with the press event', () => {
    const down = pointer('pointerdown', { x: 120, y: 80 });
    child.dispatchEvent(down);
    vi.advanceTimersByTime(LONG_PRESS_MS - 1);
    expect(onLongPress).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(onLongPress).toHaveBeenCalledExactlyOnceWith(120, 80, down);
    vi.advanceTimersByTime(5000);
    expect(onLongPress).toHaveBeenCalledTimes(1);
  });

  it('a mouse press never arms it', () => {
    child.dispatchEvent(pointer('pointerdown', { pointerType: 'mouse' }));
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);
    expect(onLongPress).not.toHaveBeenCalled();
  });

  it('a pen arms it only on a coarse-pointer device', () => {
    child.dispatchEvent(pointer('pointerdown', { pointerType: 'pen' }));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    expect(onLongPress).not.toHaveBeenCalled();

    setInputProfileOverride('touch');
    child.dispatchEvent(pointer('pointerdown', { id: 2, pointerType: 'pen' }));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    expect(onLongPress).toHaveBeenCalledTimes(1);
  });

  it('cancels when the finger moves past the slop', () => {
    child.dispatchEvent(pointer('pointerdown', { x: 50, y: 50 }));
    vi.advanceTimersByTime(200);
    child.dispatchEvent(pointer('pointermove', { x: 50 + LONG_PRESS_SLOP_PX + 1, y: 50 }));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    expect(onLongPress).not.toHaveBeenCalled();
  });

  it('tolerates movement within the slop', () => {
    child.dispatchEvent(pointer('pointerdown', { x: 50, y: 50 }));
    child.dispatchEvent(pointer('pointermove', { x: 50 + LONG_PRESS_SLOP_PX - 1, y: 50 }));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    expect(onLongPress).toHaveBeenCalledTimes(1);
  });

  it('cancels on early release, pointercancel and pointerleave', () => {
    for (const end of ['pointerup', 'pointercancel', 'pointerleave']) {
      child.dispatchEvent(pointer('pointerdown'));
      vi.advanceTimersByTime(100);
      child.dispatchEvent(pointer(end));
      vi.advanceTimersByTime(LONG_PRESS_MS);
    }
    expect(onLongPress).not.toHaveBeenCalled();
  });

  it('a second finger cancels the press (pinch, not hold)', () => {
    child.dispatchEvent(pointer('pointerdown', { id: 1 }));
    child.dispatchEvent(pointer('pointerdown', { id: 2, x: 200 }));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    expect(onLongPress).not.toHaveBeenCalled();
    // Once both lift, a fresh single press arms again.
    child.dispatchEvent(pointer('pointerup', { id: 1 }));
    child.dispatchEvent(pointer('pointerup', { id: 2 }));
    child.dispatchEvent(pointer('pointerdown', { id: 3 }));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    expect(onLongPress).toHaveBeenCalledTimes(1);
  });

  it('a stray mouse pointer does not permanently disarm later touch presses', () => {
    child.dispatchEvent(pointer('pointerdown', { id: 1 }));
    child.dispatchEvent(pointer('pointerdown', { id: 2, pointerType: 'mouse' }));
    child.dispatchEvent(pointer('pointerup', { id: 1 }));

    child.dispatchEvent(pointer('pointerdown', { id: 3 }));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    expect(onLongPress).toHaveBeenCalledTimes(1);
  });

  it('swallows the platform contextmenu during and right after the press (Android double-menu)', () => {
    const seen = vi.fn();
    child.addEventListener('contextmenu', seen);
    child.dispatchEvent(pointer('pointerdown'));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    const cm = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    child.dispatchEvent(cm);
    expect(cm.defaultPrevented).toBe(true);
    expect(seen).not.toHaveBeenCalled(); // stopped in the capture phase
  });

  it("swallows the release click so the button's primary action does not also run", () => {
    const clicked = vi.fn();
    child.addEventListener('click', clicked);
    child.dispatchEvent(pointer('pointerdown'));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    child.dispatchEvent(pointer('pointerup'));
    child.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(clicked).not.toHaveBeenCalled();
    // Only ONE click is swallowed: the next is a real tap.
    child.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(clicked).toHaveBeenCalledTimes(1);
  });

  it('leaves click and contextmenu alone when the long press is declined', () => {
    dispose();
    onLongPress.mockReturnValue(false);
    dispose = attachLongPress(el, { onLongPress });
    const clicked = vi.fn();
    const contexted = vi.fn();
    child.addEventListener('click', clicked);
    child.addEventListener('contextmenu', contexted);

    child.dispatchEvent(pointer('pointerdown'));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    child.dispatchEvent(pointer('pointerup'));
    child.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    child.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));

    expect(contexted).toHaveBeenCalledTimes(1);
    expect(clicked).toHaveBeenCalledTimes(1);
  });

  it('a plain tap (no hold) leaves click and contextmenu alone', () => {
    const clicked = vi.fn();
    child.addEventListener('click', clicked);
    child.dispatchEvent(pointer('pointerdown'));
    vi.advanceTimersByTime(100);
    child.dispatchEvent(pointer('pointerup'));
    child.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(clicked).toHaveBeenCalledTimes(1);
    const cm = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    child.dispatchEvent(cm);
    expect(cm.defaultPrevented).toBe(false);
  });

  it('dispose removes the listeners and any pending timer', () => {
    child.dispatchEvent(pointer('pointerdown'));
    dispose();
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);
    expect(onLongPress).not.toHaveBeenCalled();
    child.dispatchEvent(pointer('pointerdown', { id: 9 }));
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);
    expect(onLongPress).not.toHaveBeenCalled();
    dispose = () => {}; // already disposed
  });

  it('honours custom duration and slop', () => {
    dispose();
    dispose = attachLongPress(el, { onLongPress, durationMs: 100, slopPx: 2 });
    child.dispatchEvent(pointer('pointerdown', { x: 10, y: 10 }));
    child.dispatchEvent(pointer('pointermove', { x: 13, y: 10 })); // 3 px > 2
    vi.advanceTimersByTime(100);
    expect(onLongPress).not.toHaveBeenCalled();
    child.dispatchEvent(pointer('pointerup'));
    child.dispatchEvent(pointer('pointerdown', { id: 2, x: 10, y: 10 }));
    vi.advanceTimersByTime(100);
    expect(onLongPress).toHaveBeenCalledTimes(1);
  });
});
