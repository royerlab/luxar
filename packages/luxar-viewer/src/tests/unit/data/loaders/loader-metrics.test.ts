/**
 * Unit tests for the spatial-index loader metric helpers.
 */

import { describe, it, expect } from 'vitest';
import {
  computeLoadLatency,
  recordLoadEvent,
  type LoaderMetricsCounters,
} from '../../../../data/loaders';

function makeCounters(): LoaderMetricsCounters {
  return { loads: 0, pointsLoaded: 0, bytesLoaded: 0, avgLoadTime: 0 };
}

describe('computeLoadLatency', () => {
  it('subtracts queryStart from now', () => {
    expect(computeLoadLatency(100, 350)).toBe(250);
    expect(computeLoadLatency(50, 1050)).toBe(1000);
  });

  it('returns 0 when queryStart is undefined', () => {
    expect(computeLoadLatency(undefined, 1000)).toBe(0);
  });

  it('returns 0 when queryStart is 0 (falsy fallback matches the original)', () => {
    // The original code used `start || Date.now()`, so a 0 start was
    // treated as missing and produced a 0 latency. We preserve that.
    expect(computeLoadLatency(0, 1000)).toBe(0);
  });

  it('returns negative when now < queryStart (clock skew)', () => {
    // The helper does no clamping — it just subtracts.
    expect(computeLoadLatency(200, 100)).toBe(-100);
  });

  // [P5] non-finite boundary inputs.
  it('returns Infinity when now is Infinity (truthy start, plain subtraction)', () => {
    // queryStart=100 is truthy, so the helper subtracts: Infinity - 100 = Infinity.
    expect(computeLoadLatency(100, Infinity)).toBe(Infinity);
  });

  it('returns 0 when queryStart is NaN (NaN is falsy → fallback branch)', () => {
    // ADAPTED from the finding's guess: NaN is falsy, so `!queryStartMs` is
    // true and the helper short-circuits to 0 — it never reaches the
    // subtraction, so the result is 0, NOT NaN. This matches the real
    // `start || Date.now()` falsy-fallback semantics documented in source.
    expect(computeLoadLatency(NaN, 1000)).toBe(0);
    expect(Number.isNaN(computeLoadLatency(NaN, 1000))).toBe(false);
  });
});

describe('recordLoadEvent', () => {
  it('first event sets avgLoadTime to the new load time', () => {
    const c = makeCounters();
    recordLoadEvent(c, 100, 1024, 50);
    expect(c).toEqual({ loads: 1, pointsLoaded: 100, bytesLoaded: 1024, avgLoadTime: 50 });
  });

  it('second event averages with the first using the rolling-mean formula', () => {
    const c = makeCounters();
    recordLoadEvent(c, 10, 100, 10);
    recordLoadEvent(c, 10, 100, 30);
    // (10 × 1 + 30) / 2 = 20
    expect(c.avgLoadTime).toBe(20);
    expect(c.loads).toBe(2);
    expect(c.pointsLoaded).toBe(20);
    expect(c.bytesLoaded).toBe(200);
  });

  it('three events: rolling mean continues to average correctly', () => {
    const c = makeCounters();
    for (const t of [10, 20, 30]) recordLoadEvent(c, 0, 0, t);
    // (10 + 20 + 30) / 3 = 20
    expect(c.avgLoadTime).toBeCloseTo(20, 9);
  });

  it('mutates and returns the same counters object (in-place)', () => {
    const c = makeCounters();
    expect(recordLoadEvent(c, 1, 1, 1)).toBe(c);
  });

  it('zero load times keep the average at zero', () => {
    const c = makeCounters();
    recordLoadEvent(c, 1, 1, 0);
    recordLoadEvent(c, 1, 1, 0);
    expect(c.avgLoadTime).toBe(0);
  });
});
