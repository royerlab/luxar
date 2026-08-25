import { describe, expect, it } from 'vitest';
import { aggregateGlobalStats } from '../../../../../ui/data-loading-monitor/metrics/global-stats';
import type { LoaderMetrics } from '../../../../../types/data-monitor-types';

function metrics(path: string, type: LoaderMetrics['type'], queries = 2): LoaderMetrics {
  return {
    path,
    type,
    queries,
    loads: 3,
    errors: 0,
    elementsLoaded: 4,
    bytesLoaded: 0,
    visibleElements: 0,
    avgQueryTime: 5,
    avgLoadTime: 0,
    memoryUsed: 6,
  };
}

describe('aggregateGlobalStats', () => {
  it('uses the supplied fresh rate and collapses substitutive LOD loaders', () => {
    const snapshots = new Map([
      ['/lod/0', metrics('/lod/0', 'point-spatial-index')],
      ['/lod/1', metrics('/lod/1', 'point-spatial-index')],
      ['/plain', metrics('/plain', 'lines-spatial-index')],
    ]);
    const result = aggregateGlobalStats({
      metrics: snapshots,
      loaders: snapshots,
      lodStates: new Map([['/lod', { kind: 'lod' }]]),
      rates: { queriesPerSec: 7.5 },
      sceneGraph: {
        totalByType: { points: 10, lines: 20, gsplats: 30, mesh: 40 },
        visibleByType: { points: 1, lines: 2, gsplats: 3, mesh: 4 },
      },
      recommendations: [],
    });

    expect(result.totalLoaders).toBe(2);
    expect(result.activeSpatialLoaders).toBe(2);
    expect(result.avgQueryTime).toBe(5);
    expect(result.queriesPerSecond).toBe(7.5);
    expect(result.datasetSplats).toBe(30);
  });

  it('keeps all-loader and spatial-loader LOD excesses independent across repeated calls', () => {
    const snapshots = new Map([
      ['/lod/0', metrics('/lod/0', 'point-spatial-index')],
      ['/lod/1', metrics('/lod/1', 'point-spatial-index')],
      [
        '/lod/metadata',
        metrics('/lod/metadata', 'metadata' as unknown as LoaderMetrics['type']),
      ],
    ]);
    const params = {
      metrics: snapshots,
      loaders: snapshots,
      lodStates: new Map([['/lod', { kind: 'lod' as const }]]),
      rates: { queriesPerSec: 0 },
      sceneGraph: {
        totalByType: { points: 0, lines: 0, gsplats: 0, mesh: 0 },
        visibleByType: { points: 0, lines: 0, gsplats: 0, mesh: 0 },
      },
      recommendations: [],
    };

    const first = aggregateGlobalStats(params);
    const second = aggregateGlobalStats(params);

    expect(first.totalLoaders).toBe(1);
    expect(first.activeSpatialLoaders).toBe(1);
    expect(second.totalLoaders).toBe(1);
    expect(second.activeSpatialLoaders).toBe(1);
  });
});
