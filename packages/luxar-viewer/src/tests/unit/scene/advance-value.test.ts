/**
 * Unit tests for the pure playback-advance math
 * (scene/animation/advance-value.ts) — extracted from
 * DimensionAnimationManager's former calculateNextValue + handleBoundary so
 * it can also PEEK (t+1 prefetch) without mutating animation state.
 *
 * Expectations are parity-ported from the manager's behavior (see
 * dimension-animation-manager.test.ts) to pin the extraction as
 * behavior-preserving.
 */

import { describe, it, expect } from 'vitest';
import { advanceDimensionValue, type AdvanceArgs } from '../../../scene/animation/advance-value';

function args(over: Partial<AdvanceArgs> = {}): AdvanceArgs {
  return {
    current: 5,
    min: 0,
    max: 99,
    step: 1,
    direction: 'forward',
    loopMode: 'loop',
    targetFPS: 10,
    continuousTraverseMs: 10_000,
    ...over,
  };
}

describe('advanceDimensionValue — discrete stepping', () => {
  it('steps forward by +step', () => {
    const r = advanceDimensionValue(args());
    expect(r).toEqual({
      value: 6,
      direction: 'forward',
      shouldStop: false,
      directionChanged: false,
    });
  });

  it('steps backward by -step', () => {
    const r = advanceDimensionValue(args({ direction: 'backward' }));
    expect(r.value).toBe(4);
    expect(r.direction).toBe('backward');
  });

  it('respects a non-unit step', () => {
    expect(advanceDimensionValue(args({ step: 5 })).value).toBe(10);
  });
});

describe('advanceDimensionValue — continuous increment', () => {
  it('advances by range/traverseTime × frameTime', () => {
    // range 99, traverse 10s, 10 fps → 99/10000 × 100 = 0.99 per tick.
    const r = advanceDimensionValue(args({ step: null }));
    expect(r.value).toBeCloseTo(5.99, 10);
  });

  it('scales inversely with targetFPS', () => {
    const r = advanceDimensionValue(args({ step: null, targetFPS: 1 }));
    expect(r.value).toBeCloseTo(5 + 9.9, 10);
  });
});

describe('advanceDimensionValue — boundaries', () => {
  it('loop mode wraps forward max → min (the t99→t0 wrap)', () => {
    const r = advanceDimensionValue(args({ current: 99 }));
    expect(r.value).toBe(0);
    expect(r.shouldStop).toBe(false);
    expect(r.directionChanged).toBe(false);
  });

  it('loop mode wraps backward min → max', () => {
    const r = advanceDimensionValue(args({ current: 0, direction: 'backward' }));
    expect(r.value).toBe(99);
  });

  // Boundary is INCLUSIVE (`value >= max`), so a step landing EXACTLY on max
  // wraps — the last index is skipped on the forward loop. This pins the
  // pre-existing (baseline `handleBoundary`) contract that the extracted pure
  // function must preserve: without the exact-max case, a `>=`→`>` drift is
  // undetectable (every other boundary test overshoots max). Symmetric at min.
  it('loop wraps when a step lands EXACTLY on max (>= boundary, not >)', () => {
    // range [0, 50] step 1: t=49 → 50 === max → wraps to min (t=50 skipped).
    const r = advanceDimensionValue(args({ current: 49, min: 0, max: 50, step: 1 }));
    expect(r.value).toBe(0);
    expect(r.shouldStop).toBe(false);
  });

  it('loop wraps when a backward step lands EXACTLY on min (<= boundary, not <)', () => {
    const r = advanceDimensionValue(
      args({ current: 1, min: 0, max: 50, step: 1, direction: 'backward' })
    );
    expect(r.value).toBe(50);
  });

  it('once mode STOPS when a step lands exactly on max', () => {
    const r = advanceDimensionValue(
      args({ current: 49, min: 0, max: 50, step: 1, loopMode: 'once' })
    );
    expect(r.value).toBe(50);
    expect(r.shouldStop).toBe(true);
  });

  it('once mode clamps at max and stops', () => {
    const r = advanceDimensionValue(args({ current: 99, loopMode: 'once' }));
    expect(r.value).toBe(99);
    expect(r.shouldStop).toBe(true);
  });

  it('once mode clamps at min and stops (backward)', () => {
    const r = advanceDimensionValue(args({ current: 0, direction: 'backward', loopMode: 'once' }));
    expect(r.value).toBe(0);
    expect(r.shouldStop).toBe(true);
  });

  it('bounce mode clamps at max and RETURNS the flipped direction (no mutation contract)', () => {
    const a = args({ current: 99, loopMode: 'bounce' });
    const r = advanceDimensionValue(a);
    expect(r.value).toBe(99);
    expect(r.direction).toBe('backward');
    expect(r.directionChanged).toBe(true);
    // Purity: the input args are untouched — a second call is identical.
    expect(a.direction).toBe('forward');
    expect(advanceDimensionValue(a)).toEqual(r);
  });

  it('bounce mode flips backward → forward at min', () => {
    const r = advanceDimensionValue(
      args({ current: 0, direction: 'backward', loopMode: 'bounce' })
    );
    expect(r.value).toBe(0);
    expect(r.direction).toBe('forward');
    expect(r.directionChanged).toBe(true);
  });

  it('mid-range steps never flag a boundary', () => {
    const r = advanceDimensionValue(args({ current: 50, loopMode: 'bounce' }));
    expect(r).toEqual({
      value: 51,
      direction: 'forward',
      shouldStop: false,
      directionChanged: false,
    });
  });
});
