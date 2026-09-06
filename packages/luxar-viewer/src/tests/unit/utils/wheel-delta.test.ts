// @vitest-environment jsdom
/**
 * Unit tests for `utils/wheel-delta.ts` (issue #2531).
 *
 * jsdom honours `deltaMode` from the `WheelEvent` init dict but DEFAULTS it
 * to 0, which is exactly why every pre-existing wheel test was blind to the
 * bug: a pixel-mode event is the only shape they ever construct. Every test
 * here sets `deltaMode` explicitly.
 *
 * Assertions are pinned to the exported constants rather than magic numbers,
 * so re-tuning a constant updates the expectation with it while a severed
 * conversion still fails.
 */

import { describe, it, expect } from 'vitest';
import {
  MAX_NORMALIZED_DELTA_PX,
  NOMINAL_PAGE_HEIGHT_PX,
  PIXELS_PER_LINE,
  normalizeWheelDelta,
} from '../../../utils/wheel-delta';

/** Build a wheel event with an EXPLICIT deltaMode (jsdom defaults it to 0). */
function wheel(deltaY: number, deltaMode: number): WheelEvent {
  return new WheelEvent('wheel', { deltaY, deltaMode, cancelable: true });
}

/**
 * Build a wheel event carrying a NON-FINITE `deltaY`. `WheelEventInit.deltaY`
 * is a WebIDL restricted `double`, so the constructor refuses NaN/Infinity —
 * which is precisely why a non-finite delta can only reach us from a driver
 * quirk or a synthetic event, and why the helper has to defend against it.
 * Overwrite the property after construction to reproduce that shape.
 */
function wheelWithRawDelta(deltaY: number, deltaMode: number): WheelEvent {
  const event = new WheelEvent('wheel', { deltaMode, cancelable: true });
  Object.defineProperty(event, 'deltaY', { configurable: true, get: () => deltaY });
  return event;
}

/** An element reporting a fixed `clientHeight` (jsdom has no layout). */
function elementWithHeight(clientHeight: number): HTMLElement {
  const el = document.createElement('div');
  Object.defineProperty(el, 'clientHeight', { configurable: true, get: () => clientHeight });
  return el;
}

describe('normalizeWheelDelta — the constants themselves', () => {
  // Every other assertion here derives its expectation FROM these exports, so
  // without a literal they could be retuned silently and the suite stay green.
  it('pins MAX_NORMALIZED_DELTA_PX at 200', () => {
    expect(MAX_NORMALIZED_DELTA_PX).toBe(200);
  });

  it('pins NOMINAL_PAGE_HEIGHT_PX at 800', () => {
    expect(NOMINAL_PAGE_HEIGHT_PX).toBe(800);
  });

  // PIXELS_PER_LINE is pinned by the hard-coded 48 in the line-mode test
  // below (the one assertion in the repo that would catch an off-by-one).
});

describe('normalizeWheelDelta — pixel mode (DOM_DELTA_PIXEL) passes through', () => {
  it('passes a Chromium notch through verbatim', () => {
    expect(normalizeWheelDelta(wheel(100, 0))).toBe(100);
    expect(normalizeWheelDelta(wheel(-100, 0))).toBe(-100);
  });

  it('does NOT clamp pixel mode, even far above the cap', () => {
    // The anchor invariant: pixel-mode behaviour must be exactly what it was
    // before this helper existed, so nothing needed re-tuning and a fast
    // trackpad fling (which legitimately exceeds the cap) survives whole.
    expect(normalizeWheelDelta(wheel(5000, 0))).toBe(5000);
    expect(normalizeWheelDelta(wheel(-5000, 0))).toBe(-5000);
    expect(MAX_NORMALIZED_DELTA_PX).toBeLessThan(5000);
  });

  it('passes sub-pixel trackpad deltas through unchanged', () => {
    expect(normalizeWheelDelta(wheel(0.5, 0))).toBe(0.5);
  });
});

describe('normalizeWheelDelta — line mode (DOM_DELTA_LINE)', () => {
  it("converts Firefox's 3-line notch to pixels", () => {
    expect(normalizeWheelDelta(wheel(3, 1))).toBe(3 * PIXELS_PER_LINE);
    // Concretely: 48 px — same ballpark as Chromium's 100 px notch, and ~16x
    // the raw 3 the call sites used to consume. This literal is deliberately
    // NOT derived from PIXELS_PER_LINE: it is the only assertion in the repo
    // that catches the constant itself being retuned.
    expect(normalizeWheelDelta(wheel(3, 1))).toBe(48);
  });

  it('preserves the sign', () => {
    expect(normalizeWheelDelta(wheel(-3, 1))).toBe(-3 * PIXELS_PER_LINE);
  });

  it('clamps an outlier line count to the cap', () => {
    const outlier = MAX_NORMALIZED_DELTA_PX / PIXELS_PER_LINE + 10;
    expect(normalizeWheelDelta(wheel(outlier, 1))).toBe(MAX_NORMALIZED_DELTA_PX);
    expect(normalizeWheelDelta(wheel(-outlier, 1))).toBe(-MAX_NORMALIZED_DELTA_PX);
  });

  it('leaves an ordinary notch well under the cap', () => {
    expect(Math.abs(normalizeWheelDelta(wheel(3, 1)))).toBeLessThan(MAX_NORMALIZED_DELTA_PX);
  });
});

describe('normalizeWheelDelta — page mode (DOM_DELTA_PAGE)', () => {
  it("scales by the element's own clientHeight", () => {
    // A small fractional page must scale by the real height, not land on the
    // clamp — otherwise the element argument would be untestable.
    const el = elementWithHeight(600);
    expect(normalizeWheelDelta(wheel(0.25, 2), el)).toBe(150);
    expect(normalizeWheelDelta(wheel(-0.25, 2), el)).toBe(-150);
  });

  it('falls back to the nominal height when clientHeight is 0 (detached / unlaid-out)', () => {
    const el = elementWithHeight(0);
    expect(normalizeWheelDelta(wheel(0.1, 2), el)).toBeCloseTo(0.1 * NOMINAL_PAGE_HEIGHT_PX, 10);
  });

  it('falls back to the nominal height when no element is supplied', () => {
    expect(normalizeWheelDelta(wheel(0.1, 2))).toBeCloseTo(0.1 * NOMINAL_PAGE_HEIGHT_PX, 10);
    expect(normalizeWheelDelta(wheel(0.1, 2), null)).toBeCloseTo(0.1 * NOMINAL_PAGE_HEIGHT_PX, 10);
  });

  it('clamps a full page to the cap', () => {
    const el = elementWithHeight(600);
    expect(normalizeWheelDelta(wheel(1, 2), el)).toBe(MAX_NORMALIZED_DELTA_PX);
    expect(normalizeWheelDelta(wheel(-1, 2), el)).toBe(-MAX_NORMALIZED_DELTA_PX);
  });

  it('ignores a non-finite clientHeight and uses the nominal height', () => {
    const el = elementWithHeight(Number.POSITIVE_INFINITY);
    expect(normalizeWheelDelta(wheel(0.1, 2), el)).toBeCloseTo(0.1 * NOMINAL_PAGE_HEIGHT_PX, 10);
  });
});

describe('normalizeWheelDelta — unknown deltaMode', () => {
  it('passes an unrecognised mode through unchanged (nothing to convert)', () => {
    // A future spec mode is not something we can convert; preserving today's
    // behaviour is the least-surprise default.
    expect(normalizeWheelDelta(wheel(7, 3))).toBe(7);
    expect(normalizeWheelDelta(wheel(-9999, 3))).toBe(-9999);
  });
});

describe('normalizeWheelDelta — degenerate deltas', () => {
  it.each([0, 1, 2, 3])('deltaY +0 in mode %i yields 0', (mode) => {
    expect(normalizeWheelDelta(wheel(0, mode), elementWithHeight(600))).toBe(0);
  });

  it.each([0, 1, 2, 3])('deltaY -0 in mode %i is normalized to +0', (mode) => {
    // `-0` really does survive the WheelEvent constructor, and every mode
    // would otherwise carry it straight through (`-0 * 16` is `-0`, and
    // `clamp` passes it too) — the `deltaY === 0` guard is what normalizes it.
    // `toBe` uses `Object.is` semantics, so this fails on `-0`. Contract
    // hygiene rather than a consumer requirement: no call site can currently
    // tell `+0` from `-0`.
    expect(normalizeWheelDelta(wheel(-0, mode), elementWithHeight(600))).toBe(0);
  });

  it.each([0, 1, 2, 3])('NaN in mode %i yields 0', (mode) => {
    expect(normalizeWheelDelta(wheelWithRawDelta(NaN, mode))).toBe(0);
  });

  it.each([0, 1, 2, 3])('Infinity in mode %i yields 0', (mode) => {
    // An Infinity reaching computeZoomScale or rollDelta would poison the
    // camera state permanently.
    expect(normalizeWheelDelta(wheelWithRawDelta(Number.POSITIVE_INFINITY, mode))).toBe(0);
    expect(normalizeWheelDelta(wheelWithRawDelta(Number.NEGATIVE_INFINITY, mode))).toBe(0);
  });
});
