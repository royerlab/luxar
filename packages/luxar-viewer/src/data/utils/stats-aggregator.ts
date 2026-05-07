/**
 * Stats aggregation utilities for loader accumulators.
 *
 * Extracted from SceneLoader to reduce God Object complexity.
 * Aggregates AccumulatorStats across all loaders of a given geometry type.
 *
 * @module data/stats-aggregator
 */

import type { AccumulatorStats } from './data-accumulator';
import type { DataLoader } from '../data-loader-types';
import type { LinesDataLoader } from '../../types/lines';
import type { GSplatsDataLoader } from '../../types/gsplats';
import type { PointsSpatialIndexLoader } from '../points/points-spatial-index-loader';
import type { LinesSpatialIndexLoader } from '../lines/lines-spatial-index-loader';
import type { GSplatsSpatialIndexLoader } from '../gsplats/gsplats-spatial-index-loader';

/**
 * Aggregate accumulator stats from a collection of loaders.
 *
 * Sums capacity, allocations, growthEvents, and memoryMB across all loaders
 * that expose a `getAccumulatorStats()` method.
 *
 * @param loaders - Iterable of loaders (any type that may have getAccumulatorStats)
 * @returns Aggregated stats with all fields summed
 */
function aggregateStats(
  loaders: Iterable<{ getAccumulatorStats?: () => AccumulatorStats | null }>
): AccumulatorStats {
  let totalCapacity = 0;
  let totalAllocations = 0;
  let totalGrowthEvents = 0;
  let totalMemoryMB = 0;

  for (const loader of loaders) {
    const stats = loader.getAccumulatorStats?.();
    if (stats) {
      totalCapacity += stats.capacity;
      totalAllocations += stats.allocations;
      totalGrowthEvents += stats.growthEvents;
      totalMemoryMB += stats.memoryMB;
    }
  }

  return {
    capacity: totalCapacity,
    allocations: totalAllocations,
    growthEvents: totalGrowthEvents,
    memoryMB: totalMemoryMB,
  };
}

/**
 * Get aggregated accumulator stats for all points loaders.
 *
 * @param loaders - Map of path to DataLoader (PointsSpatialIndexLoader instances)
 * @returns Aggregated AccumulatorStats
 */
export function getAggregatedPointsAccumulatorStats(
  loaders: ReadonlyMap<string, DataLoader>
): AccumulatorStats {
  return aggregateStats(loaders.values() as Iterable<PointsSpatialIndexLoader>);
}

/**
 * Get aggregated accumulator stats for all lines loaders.
 *
 * @param loaders - Map of path to LinesDataLoader (LinesSpatialIndexLoader instances)
 * @returns Aggregated AccumulatorStats
 */
export function getAggregatedLinesAccumulatorStats(
  loaders: ReadonlyMap<string, LinesDataLoader>
): AccumulatorStats {
  return aggregateStats(loaders.values() as Iterable<LinesSpatialIndexLoader>);
}

/**
 * Get aggregated accumulator stats for all gsplats loaders.
 *
 * @param loaders - Map of path to GSplatsDataLoader (GSplatsSpatialIndexLoader instances)
 * @returns Aggregated AccumulatorStats
 */
export function getAggregatedGSplatsAccumulatorStats(
  loaders: ReadonlyMap<string, GSplatsDataLoader>
): AccumulatorStats {
  return aggregateStats(loaders.values() as Iterable<GSplatsSpatialIndexLoader>);
}
