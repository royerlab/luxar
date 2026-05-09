/**
 * Phase 18 W3: cache metrics aggregation extracted from
 * `DataLoadingMonitor.getCacheMetrics()`.
 *
 * The aggregation pulls L0/L1/L2/network stats from the optional
 * provider ports, walks the loader map to refresh per-loader
 * metric snapshots and accumulate memory limits + evictions, and
 * returns a single CacheMetrics object for the Cache tab to
 * render. It's panel-agnostic — no DOM or tab-state dependencies —
 * so lifting it out lets the Monitor's main file focus on event
 * dispatch + DOM patching rather than this multi-source roll-up.
 *
 * Two side effects are part of the contract:
 *   1. `metricsCache.set(path, ...)` is called for every loader so
 *      the monitor's "last seen metrics" snapshot stays fresh.
 *      (The original method did this inline; preserving it here
 *      keeps the loader-loop path unchanged.)
 *   2. `rates` is read but not mutated. Callers should call
 *      `calculateRates()` themselves before invoking this helper.
 */

import type {
  LoaderMonitor,
  LoaderMetrics,
  CacheMetrics,
  CacheStatsProvider,
} from '../../types/data-monitor-types';

/** Subset of `cachedRates` this aggregator reads. */
export interface CacheRatesSnapshot {
  queriesPerSec: number;
  loadsPerSec: number;
  bandwidth: number;
}

/** Provider port for L0 in-memory decompressed-chunk cache stats. */
export interface L0Provider {
  getStats: () => CacheMetrics['l0'];
  clear?: () => void;
}

export interface AggregateCacheMetricsParams {
  l0Provider: L0Provider | null;
  cacheStatsProvider: CacheStatsProvider | null;
  /** Active loaders to roll up memoryLimit + evictions from. */
  loaders: Map<string, LoaderMonitor>;
  /**
   * The monitor's metrics-snapshot map. Mutated in place: each
   * loader's current `getMetrics()` is written under its path so
   * subsequent reads see the latest values.
   */
  metricsCache: Map<string, LoaderMetrics>;
  /**
   * Pre-computed per-second rates from `calculateRates()`. Read-only
   * here; passing them in keeps the aggregator pure relative to the
   * monitor's mutable rates field.
   */
  rates: CacheRatesSnapshot;
}

/**
 * Aggregate cache metrics across all configured providers + active
 * loaders. Returns a fresh CacheMetrics object suitable for the
 * Cache tab's rendering. Mutates `metricsCache` (refreshes the
 * monitor's per-loader snapshots).
 */
export function aggregateCacheMetrics(params: AggregateCacheMetricsParams): CacheMetrics {
  const { l0Provider, cacheStatsProvider, loaders, metricsCache, rates } = params;

  let totalCacheMemory = 0;
  let memoryLimit = 0;
  let totalEntries = 0;
  let evictions = 0;

  let l0Stats: CacheMetrics['l0'] | undefined;
  let l1Stats: CacheMetrics['l1'] | undefined;
  let l2Stats: CacheMetrics['l2'] | undefined;
  let networkStats: CacheMetrics['network'] | undefined;
  let cacheEnabled = true;

  // L0 from in-memory decompressed-chunk cache provider.
  if (l0Provider) {
    l0Stats = l0Provider.getStats();
  }

  // L1/L2/network from the multi-level caching store.
  if (cacheStatsProvider) {
    const stats = cacheStatsProvider.getStats();
    cacheEnabled = cacheStatsProvider.isEnabled();

    l1Stats = {
      size: stats.l1.metadataSize + stats.l1.chunksSize,
      count: stats.l1.metadataCount + stats.l1.chunksCount,
      hits: stats.l1.hits,
      misses: stats.l1.misses,
      evictions: stats.l1.evictions,
    };

    l2Stats = {
      size: stats.l2.size,
      count: stats.l2.count,
      reads: stats.l2.reads,
      writes: stats.l2.writes,
      misses: stats.l2.misses,
    };

    networkStats = {
      bytesTransferred: stats.network.bytesTransferred,
      requestCount: stats.network.requestCount,
      bandwidth: stats.network.bandwidth,
    };

    totalCacheMemory = (l0Stats?.size ?? 0) + l1Stats.size + l2Stats.size;
    totalEntries = (l0Stats?.count ?? 0) + l1Stats.count + l2Stats.count;
  } else if (l0Stats) {
    // No L1/L2 provider — fall back to L0-only totals.
    totalCacheMemory = l0Stats.size;
    totalEntries = l0Stats.count;
  }

  // Walk loaders for memoryLimit / evictions; refresh metricsCache.
  // When no cacheStatsProvider is available, fall back to per-loader
  // memoryUsed + spatial-index entry counts.
  for (const [path, loader] of loaders) {
    const metrics = loader.getMetrics();
    metricsCache.set(path, metrics);

    memoryLimit += metrics.memoryLimit;
    evictions += metrics.evictions;

    if (!cacheStatsProvider) {
      totalCacheMemory += metrics.memoryUsed;
      if (metrics.spatialIndex && metrics.spatialIndex.rangesInCache !== undefined) {
        totalEntries += metrics.spatialIndex.rangesInCache;
      }
    }
  }

  const memoryPercent = memoryLimit > 0 ? (totalCacheMemory / memoryLimit) * 100 : 0;

  // Hit rate from L1 stats only — not a true demand hit rate, just
  // the L1-tier ratio. Higher tiers (L0 in-memory, L2 OPFS) carry
  // their own counters under l0Stats / l2Stats. Field is named
  // `recentHitRate` for back-compat; consumers wanting per-tier
  // breakdown should read `l0`, `l1`, `l2` directly.
  const totalL1Accesses = l1Stats ? l1Stats.hits + l1Stats.misses : 0;
  const recentHitRate = totalL1Accesses > 0 ? l1Stats!.hits / totalL1Accesses : 0;

  return {
    totalCacheMemory,
    memoryLimit,
    memoryPercent,
    totalEntries,
    totalAccesses: totalL1Accesses,
    recentHitRate,
    evictionsPerMin: evictions,
    avgEntrySize: totalEntries > 0 ? totalCacheMemory / totalEntries : 0,
    reuseRatio: 0,
    hitsPerSecond: l1Stats ? l1Stats.hits / 60 : 0,
    missesPerSecond: l1Stats ? l1Stats.misses / 60 : 0,
    avgAccessTime: 0,
    queriesPerSec: rates.queriesPerSec,
    loadsPerSec: rates.loadsPerSec,
    bandwidth: rates.bandwidth,
    l0: l0Stats,
    l1: l1Stats,
    l2: l2Stats,
    network: networkStats,
    enabled: cacheEnabled,
  };
}
