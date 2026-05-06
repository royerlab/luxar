/**
 * Pure helpers for the spatial-index loader's metric bookkeeping.
 *
 * Extracted from `data/point-spatial-index-loader.ts` so the
 * moving-average load-time update + the load-counter increments are
 * unit-tested in isolation, without a zarr store or an active query
 * map.
 *
 * @module data/point-loader/loader-metrics
 */

/** Mutable subset of the loader's metrics record we need to update. */
export interface LoaderMetricsCounters {
  loads: number;
  pointsLoaded: number;
  bytesLoaded: number;
  avgLoadTime: number;
}

/**
 * Apply a single load event to the metrics counters in-place.
 *
 * Updates:
 *   - `loads`: incremented by 1
 *   - `pointsLoaded`: incremented by `points`
 *   - `bytesLoaded`: incremented by `bytes`
 *   - `avgLoadTime`: rolling mean
 *     `(prev_avg × (n-1) + new_load_time) / n` after the increment
 *
 * Pure-ish: mutates `counters` (the only side-effect) and returns it.
 */
export function recordLoadEvent(
  counters: LoaderMetricsCounters,
  points: number,
  bytes: number,
  loadTime: number
): LoaderMetricsCounters {
  counters.loads += 1;
  counters.pointsLoaded += points;
  counters.bytesLoaded += bytes;
  counters.avgLoadTime =
    (counters.avgLoadTime * (counters.loads - 1) + loadTime) / counters.loads;
  return counters;
}

/**
 * Compute load latency given a query-start timestamp and the current
 * timestamp. Returns 0 when the start is missing (covers the
 * defensive `?? Date.now()` fallback in the original code).
 */
export function computeLoadLatency(
  queryStartMs: number | undefined,
  nowMs: number = Date.now()
): number {
  if (queryStartMs === undefined) return 0;
  return nowMs - queryStartMs;
}
