import { expect, it } from 'vitest';

import { scorePerfMetric } from './perf-row.mjs';

it('does not judge a WebGL throughput regression against a paced arm', () => {
  const metric = scorePerfMetric('throughputMs', 'webgl', [15, 15], [17.3, 17.3], [15, 15]);
  expect(metric.verdict).toBe('pacing-bound');
  expect(metric.pacingBound).toEqual({ base: false, cand: true });
  expect(metric.ratio).toBeUndefined();
});

it('keeps unpaced and non-WebGL throughput comparable', () => {
  expect(scorePerfMetric('unpacedThroughputMs', 'webgl', [15], [17.3], [15]).verdict).toBe('fail');
  expect(scorePerfMetric('throughputMs', 'webgpu', [15], [17.3], [15]).verdict).toBe('fail');
});

it('reports an unavailable baseline measurement as n/a', () => {
  const metric = scorePerfMetric('unpacedThroughputMs', 'webgl', [undefined], [1], [undefined]);
  expect(metric.verdict).toBe('n/a');
  expect(metric.baseMedian).toBeUndefined();
  expect(scorePerfMetric('throughputMs', 'webgl', [undefined], [17.3], [undefined]).verdict).toBe(
    'n/a'
  );
});
