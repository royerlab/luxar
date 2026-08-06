/**
 * Unit tests for the spatial-index loader metric helpers.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  computeLoadLatency,
  recordLoadEvent,
  finishQueryTracking,
  makeInitialLoaderMetrics,
  buildSpatialIndexMetrics,
  type LoaderMetricsCounters,
  type QueryMetricsCounters,
} from '../../../../data/loaders';
import type { QueryInfo } from '../../../../types/data-monitor-types';

function makeCounters(): LoaderMetricsCounters {
  return { loads: 0, elementsLoaded: 0, bytesLoaded: 0, avgLoadTime: 0 };
}

function makeQueryCounters(queries = 1, avgQueryTime = 0): QueryMetricsCounters {
  return { queries, avgQueryTime };
}

function makeQuery(id: string, startTime: number): QueryInfo {
  return { id, loader: 'point-spatial-index', path: '/test', startTime, status: 'loading' };
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
    expect(c).toEqual({ loads: 1, elementsLoaded: 100, bytesLoaded: 1024, avgLoadTime: 50 });
  });

  it('second event averages with the first using the rolling-mean formula', () => {
    const c = makeCounters();
    recordLoadEvent(c, 10, 100, 10);
    recordLoadEvent(c, 10, 100, 30);
    // (10 × 1 + 30) / 2 = 20
    expect(c.avgLoadTime).toBe(20);
    expect(c.loads).toBe(2);
    expect(c.elementsLoaded).toBe(20);
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

describe('finishQueryTracking', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("marks the tracked query 'complete', stamps endTime, and removes it", () => {
    vi.useFakeTimers();
    vi.setSystemTime(1_000);
    const q = makeQuery('q1', 900);
    const active = new Map([['q1', q]]);
    const metrics = makeQueryCounters(1);
    finishQueryTracking(active, metrics, 'q1', 900, 'complete');
    expect(q.status).toBe('complete');
    expect(q.endTime).toBe(1_000);
    expect(active.size).toBe(0);
  });

  it("marks the tracked query 'error' and removes it (error path never leaks entries)", () => {
    const q = makeQuery('q1', Date.now());
    const active = new Map([['q1', q]]);
    finishQueryTracking(active, makeQueryCounters(1), 'q1', q.startTime, 'error');
    expect(q.status).toBe('error');
    expect(active.has('q1')).toBe(false);
  });

  it('first query sets avgQueryTime to the elapsed time', () => {
    vi.useFakeTimers();
    vi.setSystemTime(1_050);
    const metrics = makeQueryCounters(1);
    finishQueryTracking(new Map(), metrics, 'q1', 1_000, 'complete');
    expect(metrics.avgQueryTime).toBe(50);
  });

  it('second query folds into the rolling mean ((prev×(n-1)+elapsed)/n)', () => {
    vi.useFakeTimers();
    vi.setSystemTime(2_030);
    const metrics = makeQueryCounters(2, 10);
    finishQueryTracking(new Map(), metrics, 'q2', 2_000, 'complete');
    // (10 × 1 + 30) / 2 = 20
    expect(metrics.avgQueryTime).toBe(20);
  });

  it('queries === 0 leaves avgQueryTime untouched (failure before query registration)', () => {
    const metrics = makeQueryCounters(0, 0);
    finishQueryTracking(new Map(), metrics, 'q1', Date.now(), 'error');
    expect(metrics.avgQueryTime).toBe(0);
  });

  it('unknown queryId leaves the map untouched but still updates the average', () => {
    vi.useFakeTimers();
    vi.setSystemTime(1_010);
    const other = makeQuery('other', 500);
    const active = new Map([['other', other]]);
    const metrics = makeQueryCounters(1);
    finishQueryTracking(active, metrics, 'missing', 1_000, 'complete');
    expect(active.size).toBe(1);
    expect(other.status).toBe('loading');
    expect(metrics.avgQueryTime).toBe(10);
  });
});

describe('makeInitialLoaderMetrics', () => {
  it('returns a fully zeroed record stamped with type and path', () => {
    const m = makeInitialLoaderMetrics('lines-spatial-index', '/some/node');
    expect(m.type).toBe('lines-spatial-index');
    expect(m.path).toBe('/some/node');
    expect(m.queries).toBe(0);
    expect(m.loads).toBe(0);
    expect(m.errors).toBe(0);
    expect(m.elementsLoaded).toBe(0);
    expect(m.visibleElements).toBe(0);
    expect(m.avgQueryTime).toBe(0);
    expect(m.memoryUsed).toBe(0);
  });

  it('returns a fresh object per call (no shared state between loaders)', () => {
    const a = makeInitialLoaderMetrics('point-spatial-index', '/a');
    const b = makeInitialLoaderMetrics('point-spatial-index', '/b');
    expect(a).not.toBe(b);
    a.queries = 5;
    expect(b.queries).toBe(0);
  });
});

describe('buildSpatialIndexMetrics', () => {
  it('maps chunk-index figures onto the metric shape', () => {
    // 4 queries touched 8 cells cumulatively -> a true mean of 2 cells/query.
    const si = buildSpatialIndexMetrics(20, 4, 8, 4000);
    expect(si.occupiedCells).toBe(20);
    expect(si.totalCells).toBe(20);
    expect(si.avgCellsPerQuery).toBe(2);
    expect(si.avgElementsPerCell).toBe(200);
    expect(si.queryEfficiency).toBe(2 / 20);
  });

  it('efficiency is stable over session length (regression: the historical last/cumulative formula decayed ~1/n)', () => {
    // Same per-query behavior (2 cells each) at 10 vs 10,000 queries must
    // yield the SAME efficiency — the old formula divided the LAST query's
    // cells by the cumulative count, drifting toward 0 and eventually firing
    // the advisor's low-efficiency recommendation on any long session.
    const early = buildSpatialIndexMetrics(20, 10, 20, 4000);
    const late = buildSpatialIndexMetrics(20, 10_000, 20_000, 4_000_000);
    expect(early.avgCellsPerQuery).toBe(2);
    expect(late.avgCellsPerQuery).toBe(2);
    expect(late.queryEfficiency).toBe(early.queryEfficiency);
  });

  it('guards the divisions at zero queries / zero chunks', () => {
    const si = buildSpatialIndexMetrics(0, 0, 0, 0);
    expect(si.avgCellsPerQuery).toBe(0);
    expect(si.avgElementsPerCell).toBe(0);
    expect(si.queryEfficiency).toBe(0);
  });
});
