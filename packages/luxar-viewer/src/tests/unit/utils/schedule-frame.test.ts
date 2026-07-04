/**
 * Unit tests for `scheduleFrame` — frame-boundary scheduling that keeps
 * working in hidden tabs (rAF is suspended there) and stays synchronous in
 * non-browser contexts (the long-standing deterministic test behaviour).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { scheduleFrame, FRAME_FALLBACK_MS } from '../../../utils/schedule-frame';

let rafCallbacks: FrameRequestCallback[] = [];

function installFakeRaf(): void {
  rafCallbacks = [];
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
    rafCallbacks.push(cb);
    return rafCallbacks.length as unknown as number;
  }) as typeof globalThis.requestAnimationFrame;
}

function uninstallRaf(): void {
  delete (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame;
}

describe('scheduleFrame', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    installFakeRaf();
  });
  afterEach(() => {
    vi.useRealTimers();
    uninstallRaf();
  });

  it('runs via rAF when the document is visible (timer fallback cancelled)', () => {
    const cb = vi.fn();
    scheduleFrame(cb);
    expect(cb).not.toHaveBeenCalled();
    expect(rafCallbacks.length).toBe(1);

    rafCallbacks[0](performance.now());
    expect(cb).toHaveBeenCalledTimes(1);

    // The shadow timer must not fire the callback a second time.
    vi.advanceTimersByTime(FRAME_FALLBACK_MS * 2);
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('falls back to the shadow timer when the rAF never fires (tab hidden after scheduling)', () => {
    const cb = vi.fn();
    scheduleFrame(cb);
    expect(rafCallbacks.length).toBe(1); // scheduled, but never fired

    vi.advanceTimersByTime(FRAME_FALLBACK_MS);
    expect(cb).toHaveBeenCalledTimes(1);

    // A late rAF must not double-fire.
    rafCallbacks[0](performance.now());
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('uses setTimeout(0) when the document is already hidden (rAF suspended)', () => {
    Object.defineProperty(document, 'hidden', { value: true, configurable: true });
    try {
      const cb = vi.fn();
      scheduleFrame(cb);
      expect(rafCallbacks.length).toBe(0); // no rAF scheduled at all
      expect(cb).not.toHaveBeenCalled();
      vi.advanceTimersByTime(0);
      expect(cb).toHaveBeenCalledTimes(1);
    } finally {
      Object.defineProperty(document, 'hidden', { value: false, configurable: true });
    }
  });

  it('invokes synchronously when requestAnimationFrame is missing (non-browser)', () => {
    uninstallRaf();
    const cb = vi.fn();
    scheduleFrame(cb);
    expect(cb).toHaveBeenCalledTimes(1);
  });
});
