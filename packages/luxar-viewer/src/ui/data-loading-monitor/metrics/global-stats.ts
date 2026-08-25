import type {
  GlobalStats,
  LoaderMetrics,
  LODProgressState,
  Recommendation,
  SceneGraphState,
} from '../../../types/data-monitor-types';

export interface AggregateGlobalStatsParams {
  metrics: ReadonlyMap<string, LoaderMetrics>;
  loaderPaths: Iterable<string>;
  loaderCount: number;
  lodStates: ReadonlyMap<string, LODProgressState>;
  rates: { queriesPerSec: number };
  sceneGraph: Pick<SceneGraphState, 'totalByType' | 'visibleByType'>;
  recommendations: Recommendation[];
}

export function aggregateGlobalStats({
  metrics,
  loaderPaths,
  loaderCount,
  lodStates,
  rates,
  sceneGraph,
  recommendations,
}: AggregateGlobalStatsParams): GlobalStats {
  let totalElementsLoaded = 0;
  let totalMemory = 0;
  let totalQueries = 0;
  let totalLoads = 0;
  let totalQueryTime = 0;
  let activeSpatial = 0;
  const isSpatialType = (type: string | undefined): boolean =>
    type === 'point-spatial-index' ||
    type === 'lines-spatial-index' ||
    type === 'gsplats-spatial-index';

  for (const loaderMetrics of metrics.values()) {
    totalElementsLoaded += loaderMetrics.elementsLoaded;
    totalMemory += loaderMetrics.memoryUsed;
    totalQueries += loaderMetrics.queries;
    totalLoads += loaderMetrics.loads;
    totalQueryTime += loaderMetrics.avgQueryTime * loaderMetrics.queries;
    if (isSpatialType(loaderMetrics.type)) activeSpatial++;
  }

  const paths = Array.from(loaderPaths);
  let lodLoaderExcess = 0;
  let lodSpatialExcess = 0;
  for (const [path, state] of lodStates) {
    if (state.kind !== 'lod') continue;
    let present = 0;
    let presentSpatial = 0;
    for (const loaderPath of paths) {
      if (loaderPath === path || loaderPath.startsWith(`${path}/`)) {
        present++;
        if (isSpatialType(metrics.get(loaderPath)?.type)) presentSpatial++;
      }
    }
    if (present > 1) lodLoaderExcess += present - 1;
    if (presentSpatial > 1) lodSpatialExcess += presentSpatial - 1;
  }

  const { totalByType, visibleByType } = sceneGraph;
  return {
    totalLoaders: Math.max(0, loaderCount - lodLoaderExcess),
    activeSpatialLoaders: Math.max(0, activeSpatial - lodSpatialExcess),
    totalElementsLoaded,
    totalMemory,
    datasetSize: totalByType.points,
    visiblePoints: visibleByType.points,
    datasetSegments: totalByType.lines,
    visibleSegments: visibleByType.lines,
    datasetSplats: totalByType.gsplats,
    visibleSplats: visibleByType.gsplats,
    totalQueries,
    totalLoads,
    avgQueryTime: totalQueries > 0 ? totalQueryTime / totalQueries : 0,
    queriesPerSecond: rates.queriesPerSec,
    recommendations,
  };
}
