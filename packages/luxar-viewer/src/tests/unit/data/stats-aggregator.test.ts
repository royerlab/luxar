/**
 * Unit tests for the stats aggregator.
 *
 * Pure function module: tests run with no DOM, no THREE, no zarr — just
 * plain loader stand-ins exposing `getAccumulatorStats()`.
 */

import { describe, it, expect } from 'vitest';
import {
  getAggregatedPointsAccumulatorStats,
  getAggregatedLinesAccumulatorStats,
  getAggregatedGSplatsAccumulatorStats,
} from '../../../data/stats-aggregator';
import type { AccumulatorStats } from '../../../data/data-accumulator';
import type { DataLoader } from '../../../data/data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';

const makeStats = (over: Partial<AccumulatorStats> = {}): AccumulatorStats => ({
  capacity: 0,
  allocations: 0,
  growthEvents: 0,
  memoryMB: 0,
  ...over,
});

const makeLoader = <T>(stats: AccumulatorStats | null | undefined): T => {
  const obj: { getAccumulatorStats?: () => AccumulatorStats | null } = {};
  if (stats !== undefined) {
    obj.getAccumulatorStats = () => stats;
  }
  return obj as T;
};

describe('stats-aggregator', () => {
  it('returns all-zero stats for an empty loader map', () => {
    const stats = getAggregatedPointsAccumulatorStats(new Map());
    expect(stats).toEqual({
      capacity: 0,
      allocations: 0,
      growthEvents: 0,
      memoryMB: 0,
    });
  });

  it('sums each numeric field across multiple loaders', () => {
    const loaders = new Map<string, DataLoader>([
      [
        'a',
        makeLoader<DataLoader>(
          makeStats({ capacity: 100, allocations: 1, growthEvents: 0, memoryMB: 1.5 })
        ),
      ],
      [
        'b',
        makeLoader<DataLoader>(
          makeStats({ capacity: 250, allocations: 3, growthEvents: 2, memoryMB: 4.0 })
        ),
      ],
      [
        'c',
        makeLoader<DataLoader>(
          makeStats({ capacity: 50, allocations: 0, growthEvents: 1, memoryMB: 0.25 })
        ),
      ],
    ]);

    const aggregated = getAggregatedPointsAccumulatorStats(loaders);

    expect(aggregated).toEqual({
      capacity: 400,
      allocations: 4,
      growthEvents: 3,
      memoryMB: 5.75,
    });
  });

  it('skips loaders that do not expose getAccumulatorStats', () => {
    const loaders = new Map<string, DataLoader>([
      ['with', makeLoader<DataLoader>(makeStats({ capacity: 10, memoryMB: 2 }))],
      ['without', makeLoader<DataLoader>(undefined)],
    ]);

    const aggregated = getAggregatedPointsAccumulatorStats(loaders);

    expect(aggregated.capacity).toBe(10);
    expect(aggregated.memoryMB).toBe(2);
  });

  it('skips loaders whose getAccumulatorStats returns null', () => {
    const loaders = new Map<string, DataLoader>([
      ['real', makeLoader<DataLoader>(makeStats({ capacity: 7, allocations: 1 }))],
      ['nulled', makeLoader<DataLoader>(null)],
    ]);

    const aggregated = getAggregatedPointsAccumulatorStats(loaders);

    expect(aggregated.capacity).toBe(7);
    expect(aggregated.allocations).toBe(1);
  });

  it('aggregates lines loaders identically to points', () => {
    const loaders = new Map<string, LinesDataLoader>([
      ['a', makeLoader<LinesDataLoader>(makeStats({ capacity: 200, growthEvents: 1 }))],
      ['b', makeLoader<LinesDataLoader>(makeStats({ capacity: 100, growthEvents: 2 }))],
    ]);

    const aggregated = getAggregatedLinesAccumulatorStats(loaders);

    expect(aggregated.capacity).toBe(300);
    expect(aggregated.growthEvents).toBe(3);
  });

  it('aggregates gsplats loaders identically to points', () => {
    const loaders = new Map<string, GSplatsDataLoader>([
      ['a', makeLoader<GSplatsDataLoader>(makeStats({ allocations: 5, memoryMB: 12.5 }))],
      ['b', makeLoader<GSplatsDataLoader>(makeStats({ allocations: 2, memoryMB: 3.5 }))],
    ]);

    const aggregated = getAggregatedGSplatsAccumulatorStats(loaders);

    expect(aggregated.allocations).toBe(7);
    expect(aggregated.memoryMB).toBe(16.0);
  });

  it('produces identical output for points/lines/gsplats wrappers given identical input', () => {
    const stats = makeStats({ capacity: 42, allocations: 1, growthEvents: 1, memoryMB: 0.5 });
    const map: Map<string, DataLoader> = new Map([['x', makeLoader<DataLoader>(stats)]]);

    const points = getAggregatedPointsAccumulatorStats(map);
    const lines = getAggregatedLinesAccumulatorStats(
      map as unknown as ReadonlyMap<string, LinesDataLoader>
    );
    const gsplats = getAggregatedGSplatsAccumulatorStats(
      map as unknown as ReadonlyMap<string, GSplatsDataLoader>
    );

    expect(points).toEqual(lines);
    expect(points).toEqual(gsplats);
  });
});
