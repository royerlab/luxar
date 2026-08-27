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
    errors: 0,
    elementsLoaded: 0,
    bytesLoaded: 0,
    visibleElements: 0,
    avgQueryTime: 0,
    avgLoadTime: 0,
    memoryUsed: 0,
    ...over,
  };
}

describe('aggregateLoaderMetrics', () => {
  it('returns a zeroed snapshot of the CALLER-SUPPLIED type for an empty array', () => {
    const out = aggregateLoaderMetrics([], '/node', 'point-spatial-index');
    expect(out.path).toBe('/node');
    expect(out.type).toBe('point-spatial-index');
    expect(out.queries).toBe(0);
    expect(out.elementsLoaded).toBe(0);
  });

  it('does not fall back to points for a non-points node with no levels left', () => {
    // A disposed ladder has cleared its level loaders, so `emptyType` is the
    // only thing left to name the node by. This used to be a hard-coded
    // 'point-spatial-index', which relabelled a disposed mesh/lines/gsplats
    // node as a points loader in the monitor's loader list.
    expect(aggregateLoaderMetrics([], '/surface', 'mesh-whole-node').type).toBe('mesh-whole-node');
    expect(aggregateLoaderMetrics([], '/curves', 'lines-spatial-index').type).toBe(
      'lines-spatial-index'
    );
  });

  it('sums counters and reports the supplied path + inner type', () => {
    const out = aggregateLoaderMetrics(
      [
        metrics('point-spatial-index', {
          queries: 2,
          loads: 1,
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
      '/points',
      'point-spatial-index'
    );
    expect(out.type).toBe('point-spatial-index');
    expect(out.path).toBe('/points');
    expect(out.queries).toBe(5);
    expect(out.loads).toBe(3);
    expect(out.errors).toBe(1);
    expect(out.elementsLoaded).toBe(150);
    expect(out.bytesLoaded).toBe(750);
    expect(out.visibleElements).toBe(60);
    expect(out.memoryUsed).toBe(35);
  });

  it('computes a query-weighted mean avgQueryTime', () => {
    // 10ms over 1 query + 30ms over 3 queries → (10*1 + 30*3) / 4 = 25
    const out = aggregateLoaderMetrics(
      [
        metrics('gsplats-spatial-index', { queries: 1, avgQueryTime: 10 }),
        metrics('gsplats-spatial-index', { queries: 3, avgQueryTime: 30 }),
      ],
      '/g',
      'point-spatial-index'
    );
    expect(out.queries).toBe(4);
    expect(out.avgQueryTime).toBeCloseTo(25);
  });

  it('avgQueryTime is 0 when there were no queries', () => {
    const out = aggregateLoaderMetrics(
      [metrics('point-spatial-index'), metrics('point-spatial-index')],
      '/p',
      'point-spatial-index'
    );
    expect(out.avgQueryTime).toBe(0);
  });

  it('computes a load-weighted mean avgLoadTime', () => {
    // 20ms over 1 load + 60ms over 3 loads → (20*1 + 60*3) / 4 = 50
    const out = aggregateLoaderMetrics(
      [
        metrics('gsplats-spatial-index', { loads: 1, avgLoadTime: 20 }),
        metrics('gsplats-spatial-index', { loads: 3, avgLoadTime: 60 }),
      ],
      '/g',
      'point-spatial-index'
    );
    expect(out.loads).toBe(4);
    expect(out.avgLoadTime).toBeCloseTo(50);
  });

  it('avgLoadTime is 0 when there were no loads', () => {
    const out = aggregateLoaderMetrics(
      [
        metrics('point-spatial-index', { avgLoadTime: 42 }),
        metrics('point-spatial-index', { avgLoadTime: 99 }),
      ],
      '/p',
      'point-spatial-index'
    );
    expect(out.loads).toBe(0);
    expect(out.avgLoadTime).toBe(0);
  });

  it('rolls up spatialIndex: cell counts summed, per-query rates query-weighted, density cell-weighted', () => {
    // out.queries = 1 + 3 = 4.
    //   occupiedCells   = 10 + 6            = 16   (summed)
    //   totalCells      = 20 + 12           = 32   (summed)
    //   avgCellsPerQuery   = (2*1 + 4*3)/4  = 3.5  (query-weighted)
    //   avgElementsPerCell = (100*10 + 200*6)/16 = 137.5 (cell-weighted:
    //     pools back to total elements / total occupied cells)
    //   queryEfficiency    = (0.1*1 + 0.5*3)/4 = 0.4 (query-weighted)
    const out = aggregateLoaderMetrics(
      [
        metrics('point-spatial-index', {
          queries: 1,
          spatialIndex: {
            occupiedCells: 10,
            totalCells: 20,
            avgCellsPerQuery: 2,
            avgElementsPerCell: 100,
            queryEfficiency: 0.1,
          },
        }),
        metrics('point-spatial-index', {
          queries: 3,
          spatialIndex: {
            occupiedCells: 6,
            totalCells: 12,
            avgCellsPerQuery: 4,
            avgElementsPerCell: 200,
            queryEfficiency: 0.5,
          },
        }),
      ],
      '/p',
      'point-spatial-index'
    );
    expect(out.spatialIndex).toBeDefined();
    expect(out.spatialIndex!.occupiedCells).toBe(16);
    expect(out.spatialIndex!.totalCells).toBe(32);
    expect(out.spatialIndex!.avgCellsPerQuery).toBeCloseTo(3.5);
    expect(out.spatialIndex!.avgElementsPerCell).toBeCloseTo(137.5);
    expect(out.spatialIndex!.queryEfficiency).toBeCloseTo(0.4);
  });

  it('excludes non-reporting LODs from the per-query rate denominators', () => {
    // A LOD with no chunk index reports no `spatialIndex` but still counts
    // queries. Folding those into the denominator would dilute both rates
    // (here: 3/7 of their true value) and could fire a spurious
    // low-efficiency recommendation.
    const out = aggregateLoaderMetrics(
      [
        metrics('point-spatial-index', {
          queries: 3,
          spatialIndex: {
            occupiedCells: 10,
            totalCells: 10,
            avgCellsPerQuery: 4,
            avgElementsPerCell: 50,
            queryEfficiency: 0.4,
          },
        }),
        // No spatialIndex, but 4 queries of its own.
        metrics('point-spatial-index', { queries: 4 }),
      ],
      '/p',
      'point-spatial-index'
    );
    expect(out.queries).toBe(7);
    expect(out.spatialIndex!.avgCellsPerQuery).toBeCloseTo(4);
    expect(out.spatialIndex!.queryEfficiency).toBeCloseTo(0.4);
  });

  it('takes optimization from the FIRST loader that reports it (no double-counting)', () => {
    const out = aggregateLoaderMetrics(
      [
        metrics('gsplats-spatial-index'), // no optimization
        metrics('gsplats-spatial-index', {
          optimization: { wasm: { loaded: true, queriesAccelerated: 5 } },
        }),
        metrics('gsplats-spatial-index', {
          optimization: { wasm: { loaded: true, queriesAccelerated: 999 } },
        }),
      ],
      '/g',
      'point-spatial-index'
    );
    // The first REPORTING loader wins; the third's differing value is ignored.
    expect(out.optimization?.wasm?.queriesAccelerated).toBe(5);
  });
});
