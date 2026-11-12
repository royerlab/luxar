import { expect, it } from 'vitest';

import { perfRowStatus, scorePerfMetric } from './perf-row.mjs';

it('does not judge a WebGL throughput regression against a paced arm', () => {
  const metric = scorePerfMetric('throughputMs', 'webgl', [15, 15], [17.3, 17.3], [15, 15]);
  expect(metric.verdict).toBe('pacing-bound');
  expect(metric.pacingBound).toEqual({ base: false, cand: true });
  expect(metric.ratio).toBeUndefined();
  expect(perfRowStatus({ throughputMs: metric, frameMs: { verdict: 'pass' } })).toBe('pass');
  expect(perfRowStatus({ throughputMs: metric, frameMs: { verdict: 'fail' } })).toBe('fail');
});

it('keeps unpaced and non-WebGL throughput comparable', () => {
  expect(scorePerfMetric('unpacedThroughputMs', 'webgl', [15], [17.3], [15]).verdict).toBe('fail');
  expect(scorePerfMetric('throughputMs', 'webgpu', [15], [17.3], [15]).verdict).toBe('fail');
});

it('widens the A/A floor across the run-wide timing family', () => {
  const base = [10, 10, 10, 10, 10];
  const cand = [10.12, 10.12, 10.12, 10.12, 10.12];
  const single = scorePerfMetric('gpuMs', 'webgpu', base, cand, base);
  const family = scorePerfMetric('gpuMs', 'webgpu', base, cand, base, 0, 4);

  expect(single.verdict).toBe('fail');
  expect(single.floor).toBe(0.01);
  expect(family.verdict).toBe('pass');
  expect(family.aaFloor).toBe(0.01);
  expect(family.floor).toBeCloseTo(0.012761, 6);
});

it('reports an unavailable baseline measurement as n/a', () => {
  const metric = scorePerfMetric('unpacedThroughputMs', 'webgl', [undefined], [1], [undefined]);
  expect(metric.verdict).toBe('n/a');
  expect(metric.baseMedian).toBeUndefined();
  expect(scorePerfMetric('throughputMs', 'webgl', [undefined], [17.3], [undefined]).verdict).toBe(
    'n/a'
  );
});
