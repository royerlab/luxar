import { describe, expect, it } from 'vitest';

import {
  judgeMetric,
  meetsExpectation,
  METRIC_DIRECTIONS,
  serverMetrics,
  validateExpectations,
  validateMetricDirections,
} from './suites.mjs';

const manifestSuites = {
  counters: {
    cases: [
      {
        id: 'a',
        metrics: [
          { name: 'render.count', kind: 'counter', better: 'lower' },
          { name: 'fps', kind: 'timing', better: 'higher' },
        ],
      },
    ],
  },
};

describe('validateExpectations', () => {
  it('accepts declared cases and metrics', () => {
    const e = { a: { 'render.count': 'zero', fps: 'win' } };
    expect(validateExpectations(e, manifestSuites)).toEqual(e);
  });

  it('rejects an unknown case, an undeclared metric and a bad value', () => {
    expect(() => validateExpectations({ b: {} }, manifestSuites)).toThrow(/unknown case 'b'/);
    expect(() => validateExpectations({ a: { nope: 'win' } }, manifestSuites)).toThrow(
      /no metric 'nope'/
    );
    expect(() => validateExpectations({ a: { fps: 'better' } }, manifestSuites)).toThrow(
      /not one of/
    );
  });
});

describe('meetsExpectation', () => {
  it('never counts n/a as met', () => {
    for (const want of ['win', 'zero', 'same', 'pass']) {
      expect(meetsExpectation(want, { verdict: 'n/a' }, 'counter')).toBe(false);
    }
  });

  it('applies each expectation', () => {
    expect(meetsExpectation('win', { verdict: 'win' }, 'counter')).toBe(true);
    expect(meetsExpectation('win', { verdict: 'pass' }, 'counter')).toBe(false);
    expect(meetsExpectation('zero', { verdict: 'win', candMedian: 0 }, 'counter')).toBe(true);
    expect(meetsExpectation('zero', { verdict: 'win', candMedian: 1 }, 'counter')).toBe(false);
    expect(
      meetsExpectation('same', { verdict: 'pass', baseMedian: 3, candMedian: 3 }, 'counter')
    ).toBe(true);
    expect(
      meetsExpectation('same', { verdict: 'pass', baseMedian: 3, candMedian: 3.2 }, 'counter')
    ).toBe(false);
    expect(
      meetsExpectation('same', { verdict: 'pass', baseMedian: 3, candMedian: 3.2 }, 'timing')
    ).toBe(true);
    expect(meetsExpectation('pass', { verdict: 'win' }, 'timing')).toBe(true);
    expect(meetsExpectation('pass', { verdict: 'fail' }, 'timing')).toBe(false);
  });
});

describe('judgeMetric', () => {
  const arm = (vals, key) => vals.map((v) => ({ [key]: v }));

  it('judges a per-tick ratio, not the timed total', () => {
    // A faster build plays 10% more ticks in the window and uploads 10% more
    // bytes in total: the same work per tick.
    const s = (bytes, ticks) => ({ 'gpu.uploadBytes': bytes, ticks });
    const m = { name: 'gpu.uploadBytes/tick', of: 'gpu.uploadBytes', per: 'ticks', relTol: 0.05 };
    const samples = {
      base: [s(1000, 50), s(1000, 50), s(1000, 50)],
      base2: [s(1000, 50), s(1000, 50), s(1000, 50)],
      cand: [s(1100, 55), s(1100, 55), s(1100, 55)],
    };
    expect(judgeMetric(m, samples)).toMatchObject({
      verdict: 'pass',
      baseMedian: 20,
      candMedian: 20,
    });
    expect(judgeMetric({ name: 'gpu.uploadBytes' }, samples).verdict).toBe('fail');
    const noTicks = { ...samples, cand: [s(1100, 0), s(1100, 0), s(1100, 0)] };
    expect(judgeMetric(m, noTicks).verdict).toBe('n/a');
  });

  it('judges counters exactly and timings with direction', () => {
    const counter = judgeMetric(
      { name: 'render.count', kind: 'counter' },
      {
        base: arm([5, 5, 5], 'render.count'),
        base2: arm([5, 5, 5], 'render.count'),
        cand: arm([0, 0, 0], 'render.count'),
      }
    );
    expect(counter).toMatchObject({
      verdict: 'win',
      baseMedian: 5,
      candMedian: 0,
      kind: 'counter',
    });
    const fps = [10, 10.1, 9.9, 10, 10.05];
    const timing = judgeMetric(
      { name: 'fps', kind: 'timing', better: 'higher' },
      {
        base: arm(fps, 'fps'),
        base2: arm(fps, 'fps'),
        cand: arm(
          fps.map((v) => v * 1.5),
          'fps'
        ),
      }
    );
    expect(timing.verdict).toBe('win');
  });

  it('reads n/a when a build does not report the metric', () => {
    const missing = judgeMetric(
      { name: 'decode.duplicates', kind: 'counter' },
      { base: [{}, {}], base2: [{}, {}], cand: arm([0, 0], 'decode.duplicates') }
    );
    expect(missing.verdict).toBe('n/a');
    const timing = judgeMetric(
      { name: 'x', kind: 'timing' },
      { base: [{}], base2: [{}], cand: [{ x: 1 }] }
    );
    expect(timing.verdict).toBe('n/a');
  });
});

describe('validateMetricDirections', () => {
  const suites = (metrics) => ({ playback: { cases: [{ id: 'scrub', metrics }] } });

  it('pins dragFrames as higher-is-better', () => {
    expect(METRIC_DIRECTIONS.dragFrames).toBe('higher');
  });

  it('rejects a manifest that judges dragFrames lower-is-better', () => {
    // The shape an audit's derived manifest declared: every gain read as a FAIL.
    const bad = suites([{ name: 'dragFrames', better: 'lower', kind: 'counter' }]);
    expect(() => validateMetricDirections(bad)).toThrow(
      /scrub\.dragFrames: declared better: 'lower', but dragFrames is higher-is-better/
    );
  });

  it('treats a missing better as lower and checks a ratio by its numerator', () => {
    expect(() => validateMetricDirections(suites([{ name: 'achievedFps' }]))).toThrow(
      /achievedFps is higher-is-better/
    );
    const ratio = { name: 'gpu.uploadBytes/tick', of: 'gpu.uploadBytes', per: 'ticks' };
    expect(() => validateMetricDirections(suites([{ ...ratio, better: 'lower' }]))).not.toThrow();
    expect(() => validateMetricDirections(suites([{ ...ratio, better: 'higher' }]))).toThrow(
      /gpu\.uploadBytes is lower-is-better/
    );
  });

  it('leaves a metric it does not know alone', () => {
    expect(() =>
      validateMetricDirections(suites([{ name: 'newCounter', better: 'higher' }]))
    ).not.toThrow();
  });
});

describe('serverMetrics', () => {
  const e = (url, startMs, bytes) => ({ url, startMs, endMs: startMs + 50, bytes });

  it('counts dataset requests before the first frame and their start waves', () => {
    const log = [
      e('/index.html', 0, 1000),
      e('/datasets/s/zarr.json', 10, 100),
      e('/datasets/s/a/zarr.json', 120, 50),
      e('/datasets/s/b/zarr.json', 125, 50),
      e('/datasets/s/a/c/0', 240, 4000),
      e('/datasets/s/a/c/1', 900, 4000),
    ];
    const m = serverMetrics(log, { navT: 0, firstFrameMs: 500, latencyMs: 100, maxInflight: 3 });
    expect(m).toEqual({
      serverRequests: 5,
      serverBytes: 8200,
      maxInflight: 3,
      requestsToFirstFrame: 4,
      bytesToFirstFrame: 4200,
      serialDepth: 3,
    });
  });

  it('omits the first-frame figures when the workload reported no first frame', () => {
    const m = serverMetrics([], { navT: 0, firstFrameMs: undefined, maxInflight: 0 });
    expect(m).toEqual({ serverRequests: 0, serverBytes: 0, maxInflight: 0 });
  });
});
