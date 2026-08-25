/**
 * Global monitor-stat aggregation. Rolls loader snapshots, substitutive
 * LOD state, cached rates, and scene-graph counts into the public
 * `GlobalStats` shape. Pure and panel-agnostic: inputs are read-only and
 * the returned object is freshly allocated.
 */

import type {
  GlobalStats,
  LoaderMetrics,
  LODProgressState,
  Recommendation,
  SceneGraphState,
} from '../../../types/data-monitor-types';

export interface AggregateGlobalStatsParams {
  metrics: ReadonlyMap<string, LoaderMetrics>;
  loaders: ReadonlyMap<string, unknown>;
  lodStates: ReadonlyMap<string, LODProgressState>;
  rates: { queriesPerSec: number };
  sceneGraph: Pick<SceneGraphState, 'totalByType' | 'visibleByType'>;
  recommendations: Recommendation[];
}

/** Aggregate one monitor tick's loader and scene totals without mutating inputs. */
export function aggregateGlobalStats({
  metrics,
  loaders,
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

  // Substitutive kind=lod groups connect one loader per leaf level (eager
  // AND lazy levels are cheap-attached + connected up front, each reporting
  // a `*-spatial-index` metric), but only one level renders at a time.
  // Collapse each group's loaders to a single logical layer so the headline
  // counts don't read K× too high. The excess is derived from the loaders
  // *actually present under each group path* — not from the LOD level count
  // — so a level that is itself a multi-leaf subtree (>1 loader per level)
  // is collapsed correctly rather than under-subtracted. Child loaders are
  // registered at scene-graph paths nested under the group path. Excess is 0
  // unless the provider reports kind=lod groups, so plain scenes are
  // unaffected.
  //
  // Two excesses are tracked from matching populations: `lodLoaderExcess`
  // counts *all* loaders under each group (subtracted from `totalLoaders`,
  // which counts all loaders), while `lodSpatialExcess` counts only the
  // spatial-index–typed loaders (subtracted from `activeSpatial`, which is
  // built from spatial-typed metrics only). Drawing each from its own
  // population keeps a future non-spatial loader nested under a LOD group
  // from over-subtracting `activeSpatial`.
  let lodLoaderExcess = 0;
  let lodSpatialExcess = 0;
  for (const [path, state] of lodStates) {
    if (state.kind !== 'lod') continue;
    let present = 0;
    let presentSpatial = 0;
    for (const loaderPath of loaders.keys()) {
      if (loaderPath === path || loaderPath.startsWith(`${path}/`)) {
        present++;
        if (isSpatialType(metrics.get(loaderPath)?.type)) presentSpatial++;
      }
    }
    if (present > 1) lodLoaderExcess += present - 1;
    if (presentSpatial > 1) lodSpatialExcess += presentSpatial - 1;
  }

  // Dataset totals + visible counts come from the scene graph, identically
  // for every geometry type. Visible counts are refreshed each update
  // cycle by `updateVisibleCountsInMonitor` after nD clipping / LOD refine.
  // The display layer keeps per-type NAMED fields (each rendered with its own
  // label, unit noun and DOM id), so this is where the kind-keyed aggregation
  // model is projected onto them. Only three are projected here: mesh has no
  // headline field of its own — its triangle counts are shown per node in the
  // scene-graph tree (`templates/scene-graph.ts`, `faceCount`).
  const { totalByType, visibleByType } = sceneGraph;
  return {
    totalLoaders: Math.max(0, loaders.size - lodLoaderExcess),
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
