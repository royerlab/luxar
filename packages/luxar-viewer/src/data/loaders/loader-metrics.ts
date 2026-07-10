/**
 * Pure helpers for the spatial-index loader's metric bookkeeping,
 * shared across the points / lines / gsplats facades.
 *
 * Originally lived in `data/points/`; hoisted to `data/loaders/` once
 * lines and gsplats grew the same monitor surface. The moving-average
 * load/query-time updates + the load-counter increments are unit-tested
 * in isolation, without a zarr store.
 *
 * `elementsLoaded` is the geometry-neutral throughput counter: it records
 * points for the points facade, vertices for lines, and splats for gsplats.
 *
 * @module data/loaders/loader-metrics
 */

import type {
  LoaderMetrics,
  LoaderType,
  QueryInfo,
  SpatialIndexMetrics,
} from '../../types/data-monitor-types';

/**
 * The zeroed initial {@link LoaderMetrics} record every spatial-index loader
 * starts from — one factory instead of three near-identical constructor
 * blocks. `elementsLoaded` / `visibleElements` are the geometry-neutral
 * counters (points / vertices-or-segments / splats).
 */
export function makeInitialLoaderMetrics(type: LoaderType, path: string): LoaderMetrics {
  return {
    type,
    path,
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
  };
}

/**
 * Build the chunk-index telemetry snapshot every facade's `getMetrics`
 * attaches as `metrics.spatialIndex` (consumed by the monitor advisor's
 * query-efficiency recommendations). Shared by all three geometry facades:
 * points/gsplats pass their single chunk index; lines passes its SEGMENT
 * index (the queried side of its dual index).
 *
 * The grid-flavored field names are the metric shape's legacy vocabulary for
 * a chunk-based index: one "cell" = one chunk. `avgCellsPerQuery` is a TRUE
 * rolling mean — CUMULATIVE queried cells over cumulative query count. (The
 * historical points formula divided the LAST query's cell count by the
 * cumulative query count, which decayed ~1/n with session length and made
 * `queryEfficiency` drift toward 0, eventually firing the advisor's
 * "Suboptimal Query Efficiency" recommendation on any long session.)
 */
export function buildSpatialIndexMetrics(
  chunkCount: number,
  chunkSize: number,
  queries: number,
  totalQueryCells: number,
  elementsLoaded: number
): SpatialIndexMetrics {
  const avgChunksPerQuery = queries > 0 ? totalQueryCells / queries : 0;
  return {
    gridShape: [chunkCount],
    gridOrigin: [0],
    cellSize: [chunkSize],
    occupiedCells: chunkCount,
    totalCells: chunkCount,
    avgCellsPerQuery: avgChunksPerQuery,
    avgElementsPerCell: chunkCount > 0 ? elementsLoaded / chunkCount : 0,
    queryEfficiency: avgChunksPerQuery / Math.max(chunkCount, 1),
    rangesInCache: 0, // Range-based per-loader cache is not used.
  };
}

/** Mutable subset of the loader's metrics record we need to update. */
export interface LoaderMetricsCounters {
  loads: number;
  elementsLoaded: number;
  bytesLoaded: number;
  avgLoadTime: number;
}

/**
 * Apply a single load event to the metrics counters in-place.
 *
 * Updates:
 *   - `loads`: incremented by 1
 *   - `elementsLoaded`: incremented by `elements` (points / vertices / splats)
 *   - `bytesLoaded`: incremented by `bytes`
 *   - `avgLoadTime`: rolling mean
 *     `(prev_avg × (n-1) + new_load_time) / n` after the increment
 *
 * Pure-ish: mutates `counters` (the only side-effect) and returns it.
 */
export function recordLoadEvent(
  counters: LoaderMetricsCounters,
  elements: number,
  bytes: number,
  loadTime: number
): LoaderMetricsCounters {
  counters.loads += 1;
  counters.elementsLoaded += elements;
  counters.bytesLoaded += bytes;
  counters.avgLoadTime = (counters.avgLoadTime * (counters.loads - 1) + loadTime) / counters.loads;
  return counters;
}

/**
 * Compute load latency given a query-start timestamp and the current
 * timestamp. Returns 0 when the start is falsy (undefined or 0). The
 * truthy check matches the original `startTime || Date.now()` fallback
 * pattern in the spatial-index loader, so a missing start yields a
 * latency of 0 rather than a huge nonsense value.
 */
export function computeLoadLatency(
  queryStartMs: number | undefined,
  nowMs: number = Date.now()
): number {
  if (!queryStartMs) return 0;
  return nowMs - queryStartMs;
}

/** Mutable subset of the loader's metrics needed to close out a query. */
export interface QueryMetricsCounters {
  queries: number;
  avgQueryTime: number;
}

/**
 * Close out a tracked query: mark the `QueryInfo` complete/errored, drop it
 * from `activeQueries`, and fold the elapsed time into the rolling
 * `avgQueryTime` mean (`(prev_avg × (n-1) + elapsed) / n` over `queries`).
 *
 * Runs on BOTH the success and error paths so the active-query map never
 * leaks an entry. The `queries > 0` guard skips the average when the
 * failure happened before query registration (e.g. an init error).
 * Shared verbatim by the points / lines / gsplats facades.
 */
export function finishQueryTracking(
  activeQueries: Map<string, QueryInfo>,
  metrics: QueryMetricsCounters,
  queryId: string,
  startTime: number,
  status: 'complete' | 'error'
): void {
  const query = activeQueries.get(queryId);
  if (query) {
    query.status = status;
    query.endTime = Date.now();
    activeQueries.delete(queryId);
  }
  const queryTime = Date.now() - startTime;
  if (metrics.queries > 0) {
    metrics.avgQueryTime =
      (metrics.avgQueryTime * (metrics.queries - 1) + queryTime) / metrics.queries;
  }
}
