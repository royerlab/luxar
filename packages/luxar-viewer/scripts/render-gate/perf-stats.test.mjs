import { describe, expect, it } from 'vitest';

import { judgePerf, median, noiseFloor, ratioCI } from './perf-stats.mjs';

describe('median', () => {
  it('handles odd, even and empty inputs', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([])).toBeNaN();
  });
});

describe('ratioCI', () => {
  it('brackets the ratio of medians and is reproducible', () => {
    const base = [10, 10.2, 9.9, 10.1, 10, 9.8, 10.3];
    const cand = base.map((v) => v * 1.1);
    const a = ratioCI(base, cand);
    expect(a.ratio).toBeCloseTo(1.1, 10);
    expect(a.lo).toBeLessThanOrEqual(a.ratio);
    expect(a.hi).toBeGreaterThanOrEqual(a.ratio);
    expect(ratioCI(base, cand)).toEqual(a);
  });
});

describe('noiseFloor', () => {
  it('never drops below the minimum', () => {
    const same = [5, 5, 5, 5, 5];
    expect(noiseFloor(same, same)).toBe(0.01);
  });

  it('grows with the spread of the A/A samples', () => {
    const a = [10, 12, 8, 11, 9, 13, 7];
    const b = [9, 13, 7, 12, 8, 11, 10];
    expect(noiseFloor(a, b)).toBeGreaterThan(0.05);
  });
});

describe('judgePerf', () => {
  const base = [10, 10.1, 9.9, 10, 10.05, 9.95, 10];

  it('passes an unchanged metric', () => {
    expect(judgePerf(base, base.slice().reverse(), 0.02).verdict).toBe('pass');
  });

  it('fails a regression larger than the floor', () => {
    expect(
      judgePerf(
        base,
        base.map((v) => v * 1.1),
        0.02
      ).verdict
    ).toBe('fail');
  });

  it('passes an unchanged but noisy metric whose own CI straddles the floor', () => {
    // Same distribution, different sample order: the ratio is ~1 but the
    // resampled CI is wide. Judging the CI's upper bound would fail this.
    const noisyBase = [10, 11.5, 9, 10.4, 12, 8.8, 10.1];
    const noisyCand = [11.2, 9.1, 10.3, 8.9, 11.9, 10.2, 10.0];
    const v = judgePerf(noisyBase, noisyCand, 0.03);
    expect(v.hi).toBeGreaterThan(1.03);
    expect(v.verdict).toBe('pass');
  });

  it('does not judge a difference within the absolute tolerance (timer-resolution metrics)', () => {
    // A wake that blocks ~0 ms reads 0 or one 0.1 ms timer quantum: 0.1/0 is
    // an infinite ratio, which without the tolerance is a false FAIL.
    const zeros = [0, 0, 0, 0, 0, 0, 0];
    const quantum = [0, 0.1, 0, 0.1, 0.1, 0, 0.1];
    expect(judgePerf(zeros, quantum, 0.01).verdict).toBe('fail');
    expect(judgePerf(zeros, quantum, 0.01, 0.2).verdict).toBe('pass');
    // ...but a real change beyond it is still judged.
    expect(judgePerf([3, 3.1, 2.9, 3], [0, 0.1, 0, 0], 0.01, 0.2).verdict).toBe('win');
    expect(judgePerf([0, 0.1, 0, 0], [3, 3.1, 2.9, 3], 0.01, 0.2).verdict).toBe('fail');
  });

  it('reports a clear improvement as a win', () => {
    expect(
      judgePerf(
        base,
        base.map((v) => v * 0.8),
        0.02
      ).verdict
    ).toBe('win');
  });
});
