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

  // #2944 A7: the endpoints are FRAMES, not wrap triggers. A step that lands
  // on (or overshoots) max SHOWS max; only a step taken FROM max wraps/stops.
  // Previously `value >= max` wrapped on arrival, so the last timepoint of a
  // [0, 50] axis was never displayed in loop or once mode (49 -> 0, once
  // stopped at 49 as "complete"), and backward playback skipped min.
  it('loop forward VISITS max before wrapping (49 -> 50, then 50 -> 0)', () => {
    const toMax = advanceDimensionValue(args({ current: 49, min: 0, max: 50, step: 1 }));
    expect(toMax).toEqual({
      value: 50,
      direction: 'forward',
      shouldStop: false,
      directionChanged: false,
    });
    const wrap = advanceDimensionValue(args({ current: 50, min: 0, max: 50, step: 1 }));
    expect(wrap.value).toBe(0);
    expect(wrap.shouldStop).toBe(false);
  });

  it('loop backward VISITS min before wrapping (1 -> 0, then 0 -> 50)', () => {
    const toMin = advanceDimensionValue(
      args({ current: 1, min: 0, max: 50, step: 1, direction: 'backward' })
    );
    expect(toMin.value).toBe(0);
    expect(toMin.shouldStop).toBe(false);
    const wrap = advanceDimensionValue(
      args({ current: 0, min: 0, max: 50, step: 1, direction: 'backward' })
    );
    expect(wrap.value).toBe(50);
  });

  it('once forward SHOWS max (49 -> 50, not stopped), then completes from max', () => {
    const toMax = advanceDimensionValue(
      args({ current: 49, min: 0, max: 50, step: 1, loopMode: 'once' })
    );
    expect(toMax.value).toBe(50);
    expect(toMax.shouldStop).toBe(false);
    const done = advanceDimensionValue(
      args({ current: 50, min: 0, max: 50, step: 1, loopMode: 'once' })
    );
    expect(done.value).toBe(50);
    expect(done.shouldStop).toBe(true);
  });

  it('once backward SHOWS min (1 -> 0, not stopped), then completes from min', () => {
    const toMin = advanceDimensionValue(
      args({ current: 1, min: 0, max: 50, step: 1, loopMode: 'once', direction: 'backward' })
    );
    expect(toMin.value).toBe(0);
    expect(toMin.shouldStop).toBe(false);
    const done = advanceDimensionValue(
      args({ current: 0, min: 0, max: 50, step: 1, loopMode: 'once', direction: 'backward' })
    );
    expect(done.shouldStop).toBe(true);
  });

  it('a step that OVERSHOOTS max clamps onto max instead of wrapping', () => {
    // step 3 on [0, 50]: 48 -> 51 overshoots; show 50 first.
    const r = advanceDimensionValue(args({ current: 48, min: 0, max: 50, step: 3 }));
    expect(r.value).toBe(50);
    expect(r.shouldStop).toBe(false);
  });

  it('continuous: an overshooting increment clamps onto max, the next tick wraps', () => {
    // range 99, 10 fps, 10 s traverse: +0.99 per tick.
    const toMax = advanceDimensionValue(args({ current: 98.5, step: null }));
    expect(toMax.value).toBe(99);
    const wrap = advanceDimensionValue(args({ current: 99, step: null }));
    expect(wrap.value).toBe(0);
  });

  it('a max OFF the discrete grid: the last grid point counts as the boundary', () => {
    // Grid 0, 3, 6, 9 on [0, 10]: clamping 9 + 3 onto 10 would snap straight
    // back to 9 and freeze playback, so 9 is the endpoint and the step wraps.
    const r = advanceDimensionValue(args({ current: 9, min: 0, max: 10, step: 3, gridStep: 3 }));
    expect(r.value).toBe(0);
    const once = advanceDimensionValue(
      args({ current: 9, min: 0, max: 10, step: 3, gridStep: 3, loopMode: 'once' })
    );
    expect(once.shouldStop).toBe(true);
    expect(once.value).toBe(9);
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
