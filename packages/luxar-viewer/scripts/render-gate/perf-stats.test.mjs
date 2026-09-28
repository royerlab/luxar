import { describe, expect, it } from 'vitest';

import { anyMissing, judgeCounter, judgePerf, median, noiseFloor, ratioCI } from './perf-stats.mjs';

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

describe('judgePerf better=higher', () => {
  const base = [10, 10.1, 9.9, 10, 10.05, 9.95, 10];

  it('wins above 1 + floor and fails below 1 - floor', () => {
    const up = base.map((v) => v * 1.2);
    const down = base.map((v) => v * 0.8);
    expect(judgePerf(base, up, 0.02, 0, { better: 'higher' }).verdict).toBe('win');
    expect(judgePerf(base, down, 0.02, 0, { better: 'higher' }).verdict).toBe('fail');
    expect(judgePerf(base, base, 0.02, 0, { better: 'higher' }).verdict).toBe('pass');
  });
});

describe('judgeCounter', () => {
  const flat = [100, 100, 100, 100, 100];

  it('compares exact counters exactly when both baseline arms agree', () => {
    expect(judgeCounter(flat, flat, flat).verdict).toBe('pass');
    // One unit less is a win: no noise floor applies to a deterministic counter.
    expect(judgeCounter(flat, [99, 99, 99, 99, 99], flat).verdict).toBe('win');
    expect(judgeCounter(flat, [101, 101, 101, 101, 101], flat).verdict).toBe('fail');
    const v = judgeCounter(flat, [99, 99, 99, 99, 99], flat);
    expect(v.exact).toBe(true);
    expect(v.baseMedian).toBe(100);
    expect(v.candMedian).toBe(99);
  });

  it('honours tol and better=higher on the exact path', () => {
    expect(judgeCounter(flat, [101, 101, 101], flat, { tol: 1 }).verdict).toBe('pass');
    expect(judgeCounter(flat, [102, 102, 102], flat, { better: 'higher' }).verdict).toBe('win');
    expect(judgeCounter(flat, [98, 98, 98], flat, { better: 'higher' }).verdict).toBe('fail');
  });

  it('treats any increase over a constant-zero baseline as a fail', () => {
    const zeros = [0, 0, 0, 0, 0];
    expect(judgeCounter(zeros, zeros, zeros).verdict).toBe('pass');
    expect(judgeCounter(zeros, [1, 1, 1, 1, 1], zeros).verdict).toBe('fail');
  });

  it('falls back to the A/A noise floor when the baseline arms vary', () => {
    const a = [100, 104, 97, 101, 99, 103, 98];
    const b = [101, 99, 103, 98, 100, 97, 102];
    const v = judgeCounter(a, [100, 102, 99, 101, 98, 100, 101], b);
    expect(v.exact).toBe(false);
    expect(v.verdict).toBe('pass');
    expect(v.floor).toBeGreaterThan(0.01);
    expect(
      judgeCounter(
        a,
        a.map((x) => x * 2),
        b
      ).verdict
    ).toBe('fail');
    expect(
      judgeCounter(
        a,
        a.map((x) => x / 2),
        b
      ).verdict
    ).toBe('win');
    expect(
      judgeCounter(
        a,
        a.map((x) => x * 2),
        b,
        { better: 'higher' }
      ).verdict
    ).toBe('win');
  });

  it('reads n/a when any arm has a missing sample', () => {
    expect(judgeCounter(flat, [100, undefined, 100], flat).verdict).toBe('n/a');
    expect(judgeCounter([100, null, 100], flat, flat).verdict).toBe('n/a');
    expect(judgeCounter(flat, flat, [NaN, 100]).verdict).toBe('n/a');
    expect(judgeCounter([], flat, flat).verdict).toBe('n/a');
  });
});

describe('anyMissing', () => {
  it('flags undefined, null, NaN and empty arms', () => {
    expect(anyMissing([1, 2], [3])).toBe(false);
    expect(anyMissing([1, undefined])).toBe(true);
    expect(anyMissing([null])).toBe(true);
    expect(anyMissing([NaN])).toBe(true);
    expect(anyMissing([])).toBe(true);
  });
});

describe('judgeCounter relTol', () => {
  it('treats a timing-driven count within the relative tolerance as unchanged', () => {
    // Same build, one frame of jitter in a 189-render workload (seen in an A/A run).
    const base = [189, 189];
    const cand = [189, 190];
    expect(judgeCounter(base, cand, base).verdict).toBe('fail');
    expect(judgeCounter(base, cand, base, { relTol: 0.05 }).verdict).toBe('pass');
  });

  it('still reports a real change beyond the relative tolerance', () => {
    const base = [181, 181];
    expect(judgeCounter(base, [40, 41], base, { relTol: 0.05 }).verdict).toBe('win');
    expect(judgeCounter(base, [200, 201], base, { relTol: 0.05 }).verdict).toBe('fail');
  });
});
