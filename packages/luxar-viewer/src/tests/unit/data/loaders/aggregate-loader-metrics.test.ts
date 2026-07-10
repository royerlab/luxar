/**
 * Unit tests for `aggregateLoaderMetrics` — the per-LOD → single-node
 * metrics roll-up used by the three progressive loaders.
 */

import { describe, it, expect } from 'vitest';
import { aggregateLoaderMetrics } from '../../../../data/loaders/aggregate-loader-metrics';
import type { LoaderMetrics, LoaderType } from '../../../../types/data-monitor-types';

function metrics(type: LoaderType, over: Partial<LoaderMetrics> = {}): LoaderMetrics {
  return {
    type,
    path: '/n/additive_x',
    queries: 0,
    loads: 0,
    evictions: 0,
    errors: 0,
    elementsLoaded: 0,
    bytesLoaded: 0,
    visibleElements: 0,
    avgQueryTime: 0,
    avgLoadTime: 0,
    memoryUsed: 0,
    memoryLimit: 0,
    ...over,
  };
}

describe('aggregateLoaderMetrics', () => {
  it('returns a zeroed point-spatial-index snapshot for an empty array', () => {
    const out = aggregateLoaderMetrics([], '/node');
    expect(out.path).toBe('/node');
    expect(out.type).toBe('point-spatial-index');
    expect(out.queries).toBe(0);
    expect(out.elementsLoaded).toBe(0);
  });

  it('sums counters and reports the supplied path + inner type', () => {
    const out = aggregateLoaderMetrics(
      [
        metrics('point-spatial-index', {
          queries: 2,
          loads: 1,
          evictions: 3,
          errors: 1,
          elementsLoaded: 100,
          bytesLoaded: 500,
          visibleElements: 40,
          memoryUsed: 10,
        }),
        metrics('point-spatial-index', {
          queries: 3,
          loads: 2,
          elementsLoaded: 50,
          bytesLoaded: 250,
          visibleElements: 20,
          memoryUsed: 25,
        }),
      ],
      '/points'
    );
    expect(out.type).toBe('point-spatial-index');
    expect(out.path).toBe('/points');
    expect(out.queries).toBe(5);
    expect(out.loads).toBe(3);
    expect(out.evictions).toBe(3);
    expect(out.errors).toBe(1);
    expect(out.elementsLoaded).toBe(150);
    expect(out.bytesLoaded).toBe(750);
    expect(out.visibleElements).toBe(60);
    expect(out.memoryUsed).toBe(35);
  });

  it('takes the max memoryLimit (shared cap, not additive)', () => {
    const out = aggregateLoaderMetrics(
      [
        metrics('lines-spatial-index', { memoryLimit: 100 }),
        metrics('lines-spatial-index', { memoryLimit: 250 }),
      ],
      '/lines'
    );
    expect(out.memoryLimit).toBe(250);
  });

  it('computes a query-weighted mean avgQueryTime', () => {
    // 10ms over 1 query + 30ms over 3 queries → (10*1 + 30*3) / 4 = 25
    const out = aggregateLoaderMetrics(
      [
        metrics('gsplats-spatial-index', { queries: 1, avgQueryTime: 10 }),
        metrics('gsplats-spatial-index', { queries: 3, avgQueryTime: 30 }),
      ],
      '/g'
    );
    expect(out.queries).toBe(4);
    expect(out.avgQueryTime).toBeCloseTo(25);
  });

  it('avgQueryTime is 0 when there were no queries', () => {
    const out = aggregateLoaderMetrics(
      [metrics('point-spatial-index'), metrics('point-spatial-index')],
      '/p'
    );
    expect(out.avgQueryTime).toBe(0);
  });
});
