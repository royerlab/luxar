/**
 * Pure aggregation of per-LOD {@link LoaderMetrics} into a single snapshot.
 *
 * The progressive (multi-additive-LOD) loaders wrap N inner spatial-index
 * loaders. To report honest telemetry to the data-loading monitor they expose
 * the {@link LoaderMonitor} surface and roll their inner loaders' metrics up
 * through this helper. Kept pure (no `this`, no side effects) so it can be
 * unit-tested in isolation and reused across the Points / Lines / GSplats
 * progressive loaders.
 *
 * @module data/loaders/aggregate-loader-metrics
 */

import type { LoaderMetrics, LoaderType } from '../../types/data-monitor-types';

/** A zeroed metrics object — used as the empty-array fallback. */
function zeroedMetrics(type: LoaderType, path: string): LoaderMetrics {
  return {
    type,
    path,
    queries: 0,
    loads: 0,
    errors: 0,
    elementsLoaded: 0,
    bytesLoaded: 0,
    visibleElements: 0,
    avgQueryTime: 0,
    avgLoadTime: 0,
    memoryUsed: 0,
  };
}

/**
 * Aggregate the per-LOD metrics of a progressive loader's inner loaders into a
 * single {@link LoaderMetrics} representing the whole node.
 *
 * Counters (`queries` / `loads` / `errors` / `elementsLoaded` /
 * `bytesLoaded` / `visibleElements` / `memoryUsed`) are summed. `avgQueryTime` /
 * `avgLoadTime` are weighted means by `queries` / `loads` respectively (so a
 * LOD that never queried doesn't skew the average). The optional
 * `spatialIndex` cell counts are summed with query-weighted rate means.
 * `optimization` (mostly app-global singletons like WASM/GPU-pool) is taken
 * from the first loader that reports it to avoid double-counting.
 *
 * @param inner per-LOD metrics snapshots (already produced by `getMetrics()`)
 * @param path the parent node path to report as this aggregate's `path`
 */
export function aggregateLoaderMetrics(inner: LoaderMetrics[], path: string): LoaderMetrics {
  if (inner.length === 0) {
    return zeroedMetrics('point-spatial-index', path);
  }

  const out = zeroedMetrics(inner[0].type, path);

  let queryTimeWeighted = 0;
  let loadTimeWeighted = 0;

  // spatialIndex accumulators
  let siCells = 0;
  let siTotalCells = 0;
  let siCellsPerQueryWeighted = 0;
  let siPointsPerCellWeighted = 0;
  let siEfficiencyWeighted = 0;
  let firstSpatialIndex: NonNullable<LoaderMetrics['spatialIndex']> | undefined;
  let firstOptimization: LoaderMetrics['optimization'] | undefined;

  for (const m of inner) {
    out.queries += m.queries;
    out.loads += m.loads;
    out.errors += m.errors;
    out.elementsLoaded += m.elementsLoaded;
    out.bytesLoaded += m.bytesLoaded;
    out.visibleElements += m.visibleElements;
    out.memoryUsed += m.memoryUsed;

    queryTimeWeighted += m.avgQueryTime * m.queries;
    loadTimeWeighted += m.avgLoadTime * m.loads;

    if (m.spatialIndex) {
      const si = m.spatialIndex;
      if (!firstSpatialIndex) firstSpatialIndex = si;
      siCells += si.occupiedCells;
      siTotalCells += si.totalCells;
      siCellsPerQueryWeighted += si.avgCellsPerQuery * m.queries;
      siPointsPerCellWeighted += si.avgElementsPerCell * m.queries;
      siEfficiencyWeighted += si.queryEfficiency * m.queries;
    }
    if (m.optimization && !firstOptimization) {
      firstOptimization = m.optimization;
    }
  }

  out.avgQueryTime = out.queries > 0 ? queryTimeWeighted / out.queries : 0;
  out.avgLoadTime = out.loads > 0 ? loadTimeWeighted / out.loads : 0;

  if (firstSpatialIndex) {
    out.spatialIndex = {
      occupiedCells: siCells,
      totalCells: siTotalCells,
      avgCellsPerQuery: out.queries > 0 ? siCellsPerQueryWeighted / out.queries : 0,
      avgElementsPerCell: out.queries > 0 ? siPointsPerCellWeighted / out.queries : 0,
      queryEfficiency: out.queries > 0 ? siEfficiencyWeighted / out.queries : 0,
    };
  }
  if (firstOptimization) {
    out.optimization = firstOptimization;
  }

  return out;
}
