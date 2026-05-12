/**
 * Post-load monitor wiring for `SceneLoader.loadScene()`.
 *
 * After the scene graph is built and per-node loaders are connected,
 * the SceneLoader hands the DataLoadingMonitor references to:
 *   - the L1/L2 caching store (Cache tab)
 *   - the L0 decompressed-chunk cache (Cache tab)
 *   - the GPU buffer pool (Memory tab)
 *   - the per-geometry accumulator aggregators (Memory tab)
 *   - the update profiler (Performance tab, if any)
 *   - the scene-graph snapshot for the tree view
 *
 * Splitting this out keeps loadScene focused on "what was built"
 * rather than "how monitor tabs are wired."
 */

import type { SceneNode } from '../data-loader-types';
import type { DataLoader } from '../data-loader-types';
import type { SceneGraphNode, CacheTelemetryState } from '../../types/data-monitor-types';
import type { SceneLoaderMonitorPort } from '../scene-loader-monitor-port';
import type { LinesDataLoader } from '../../types/lines';
import type { GSplatsDataLoader } from '../../types/gsplats';
import type { MultiLevelCachingStore, DecompressedChunkCache } from '../../cache';
import type { GPUBufferPool } from '../../rendering/gpu-buffer-pool';
import type { UpdateProfiler } from '../../profiling/update-profiler';
import {
  getAggregatedPointsAccumulatorStats,
  getAggregatedLinesAccumulatorStats,
  getAggregatedGSplatsAccumulatorStats,
} from '../utils/stats-aggregator';
import { convertToSceneGraphNode } from './scene-graph-converter';

export interface WireMonitorAfterLoadParams {
  /** The monitor port (no-op if undefined). */
  monitor: SceneLoaderMonitorPort | null | undefined;
  /** L1/L2 caching store, when caching is enabled. */
  cachingStore: MultiLevelCachingStore | null;
  /** L0 decompressed-chunk cache, when L0 is enabled. */
  l0Cache: DecompressedChunkCache | null;
  /**
   * Resolved cache telemetry state from `setupCaches()`. Pushed to
   * the monitor before provider wiring so the UI sees the right
   * disabled-reason during the brief pre-provider window.
   */
  cacheTelemetryState: CacheTelemetryState;
  /** GPU buffer pool for points/lines/gsplats geometry. */
  gpuBufferPool: GPUBufferPool | null;
  /** Update profiler (Performance tab), when SceneLoaderManager wired one in. */
  profiler: UpdateProfiler | null;
  /** Per-geometry loader maps for accumulator-stat aggregation. */
  loaders: Map<string, DataLoader>;
  linesLoaders: Map<string, LinesDataLoader>;
  gsplatLoaders: Map<string, GSplatsDataLoader>;
  /** The scene-graph SceneNode the loader just built. */
  sceneGraph: SceneNode;
  /**
   * Caller-supplied callback that traverses `rootGroup` and calls
   * `monitor.updateVisibleSegments` / `updateVisibleSplats`. Lives on
   * SceneLoader because it reads the live THREE scene; passing it in
   * keeps this helper free of THREE imports.
   */
  updateVisibleCounts: () => void;
}

/**
 * Wire all monitor-tab data providers and push the initial
 * scene-graph snapshot. No-op when `monitor` is null/undefined —
 * matches the pre-extraction `if (monitor)` guard.
 */
export function wireMonitorAfterLoad(params: WireMonitorAfterLoadParams): void {
  const {
    monitor,
    cachingStore,
    l0Cache,
    cacheTelemetryState,
    gpuBufferPool,
    profiler,
    loaders,
    linesLoaders,
    gsplatLoaders,
    sceneGraph,
    updateVisibleCounts,
  } = params;

  if (!monitor) return;

  // Push the explicit telemetry state BEFORE provider wiring so the
  // UI's brief pre-provider window reflects the policy decision (e.g.
  // `?no-cache` shows as "disabled-no-cache", not "not-wired").
  monitor.setCacheTelemetryState(cacheTelemetryState);

  // Cache tab — L1/L2 (LRU + OPFS) and L0 (decompressed chunks).
  if (cachingStore) {
    monitor.setCacheStatsProvider(cachingStore);
  }
  if (l0Cache) {
    monitor.setL0CacheProvider({
      getStats: () => l0Cache.getStats(),
      clear: () => l0Cache.clear(),
    });
  }

  // Memory tab — GPU buffer pool + per-geometry accumulator aggregates.
  if (gpuBufferPool) {
    monitor.setGPUBufferPoolProvider(gpuBufferPool);
  }
  monitor.setAccumulatorProvider('points', {
    getStats: () => getAggregatedPointsAccumulatorStats(loaders),
  });
  monitor.setAccumulatorProvider('lines', {
    getStats: () => getAggregatedLinesAccumulatorStats(linesLoaders),
  });
  monitor.setAccumulatorProvider('gsplats', {
    getStats: () => getAggregatedGSplatsAccumulatorStats(gsplatLoaders),
  });

  // Performance tab — update profiler timings (optional).
  if (profiler) {
    monitor.setProfiler(profiler);
  }

  // Scene-Graph tab — initial snapshot.
  const sceneGraphRoot: SceneGraphNode = convertToSceneGraphNode(sceneGraph);
  monitor.setSceneGraph(sceneGraphRoot);

  // Initial visible-counts pass; the caller's callback walks rootGroup.
  updateVisibleCounts();

  monitor.forceUpdate();
}
