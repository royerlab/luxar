/**
 * Cache metrics aggregation. Pulls L0/L1/L2/network stats from the
 * optional provider ports, walks the loader map to refresh
 * per-loader metric snapshots and accumulate memory limits +
 * evictions, and returns a single `CacheMetrics` object for the
 * Cache tab to render. Panel-agnostic — no DOM or tab-state
 * dependencies — so the Monitor's main file can focus on event
 * dispatch + DOM patching rather than this multi-source roll-up.
 *
 * Side effects that callers depend on:
 *   1. `metricsCache.set(path, ...)` is called for every loader so
 *      the monitor's "last seen metrics" snapshot stays fresh.
 *   2. `rates` is read but not mutated. Callers should call
 *      `calculateRates()` themselves before invoking this helper.
 */

import type {
  LoaderMonitor,
  LoaderMetrics,
  CacheMetrics,
  CacheStatsProvider,
  CacheStatusBadge,
  CacheTelemetryState,
} from '../../../types/data-monitor-types';

// Re-export so existing imports keep working.
export type { CacheTelemetryState };

/**
 * Subset of `cachedRates` this aggregator reads. The aggregator
 * surfaces these as rolling per-second rates rather than computing
 * its own — `bandwidth` is already bytes/sec from `./rates.ts`.
 */
export interface CacheRatesSnapshot {
  queriesPerSec: number;
  loadsPerSec: number;
  hitsPerSec: number;
  missesPerSec: number;
  bandwidth: number;
}

/** Provider port for L0 in-memory decompressed-chunk cache stats. */
export interface L0Provider {
  getStats: () => CacheMetrics['l0'];
  clear?: () => void;
}

/** Provider port for SliceCache ("S-cache") stats. */
export interface SliceProvider {
  getStats: () => CacheMetrics['slice'];
  clear?: () => void;
}

export interface AggregateCacheMetricsParams {
  l0Provider: L0Provider | null;
  /** Optional so existing callers/tests need no change; the real monitor always
   *  passes it. Absent → no SliceCache row in the aggregated metrics. */
  sliceProvider?: SliceProvider | null;
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
  /**
   * Explicit cache telemetry state. When omitted, defaults to
   * `not-wired` (the most honest fallback when no provider exists).
   */
  telemetryState?: CacheTelemetryState;
}

/**
 * Aggregate cache metrics across all configured providers + active
 * loaders. Returns a fresh CacheMetrics object suitable for the
 * Cache tab's rendering. Mutates `metricsCache` (refreshes the
 * monitor's per-loader snapshots).
 */
export function aggregateCacheMetrics(params: AggregateCacheMetricsParams): CacheMetrics {
  const { l0Provider, sliceProvider, cacheStatsProvider, loaders, metricsCache, rates } = params;

  let totalCacheMemory = 0;
  let memoryLimit = 0;
  let totalEntries = 0;
  let evictions = 0;

  let l0Stats: CacheMetrics['l0'] | undefined;
  let sliceStats: CacheMetrics['slice'] | undefined;
  let l1Stats: CacheMetrics['l1'] | undefined;
  let l2Stats: CacheMetrics['l2'] | undefined;
  let networkStats: CacheMetrics['network'] | undefined;
  let demand: { l1Hits: number; l2Hits: number; networkRequests: number } | undefined;
  let health: CacheMetrics['health'] | undefined;
  // Counters that drive the cache-errors-detected and quota-constrained
  // badges below. Sourced from the L2 stats payload but not surfaced on
  // CacheMetrics.l2 itself (which stays a stable consumer-facing shape).
  let l2QuotaSkipped = 0;
  let l2WriteFailures = 0;
  let l2Corrupted = 0;
  let l2MetadataParseFailures = 0;

  // Derive telemetry state. Caller-supplied wins; otherwise infer
  // from provider presence. Default-on-no-provider is `not-wired`,
  // not `enabled` — `?no-cache` runs have no provider and must not
  // surface as enabled.
  let telemetryState: CacheTelemetryState;
  if (params.telemetryState) {
    telemetryState = params.telemetryState;
  } else if (cacheStatsProvider) {
    telemetryState = cacheStatsProvider.isEnabled()
      ? { kind: 'enabled' }
      : { kind: 'disabled-config' };
  } else {
    telemetryState = { kind: 'not-wired' };
  }
  const cacheEnabled = telemetryState.kind === 'enabled';

  // L0 from in-memory decompressed-chunk cache provider.
  if (l0Provider) {
    l0Stats = l0Provider.getStats();
  }

  // SliceCache ("S-cache") from its provider.
  if (sliceProvider) {
    sliceStats = sliceProvider.getStats();
  }

  // L1/L2/network from the multi-level caching store.
  if (cacheStatsProvider) {
    const stats = cacheStatsProvider.getStats();

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
      // R3: pass through OPFS health counters so the cache tab can
      // render them inline. Undefined entries from older providers
      // stay undefined.
      quotaWriteSkipped: stats.l2.quotaWriteSkipped,
      writeFailures: stats.l2.writeFailures,
      corruptedEntries: stats.l2.corruptedEntries,
      metadataParseFailures: stats.l2.metadataParseFailures,
    };

    // Pull badge-relevant L2 health counters (optional — older
    // providers omit them).
    l2QuotaSkipped = stats.l2.quotaWriteSkipped ?? 0;
    l2WriteFailures = stats.l2.writeFailures ?? 0;
    l2Corrupted = stats.l2.corruptedEntries ?? 0;
    l2MetadataParseFailures = stats.l2.metadataParseFailures ?? 0;

    networkStats = {
      bytesTransferred: stats.network.bytesTransferred,
      requestCount: stats.network.requestCount,
      bandwidth: stats.network.bandwidth,
      totalBytesServed: stats.network.totalBytesServed,
      totalRequestsServed: stats.network.totalRequestsServed,
    };

    // Per-tier demand counters (optional — providers without the
    // demand-counter feature simply omit the field).
    if (stats.demand) {
      demand = stats.demand;
    }

    // Cache validation health drives the unvalidated-external-dataset
    // and (S2) opfs-unavailable badges.
    if (stats.health) {
      health = {
        validationMode: stats.health.validationMode,
        lastValidatedAt: stats.health.lastValidatedAt,
        unvalidatedExternalDataset: stats.health.unvalidatedExternalDataset,
        // opfsAvailable was added in S2; older providers omit it,
        // in which case the badge stays unemitted (treat-as-true).
        opfsAvailable: stats.health.opfsAvailable,
      };
    }

    totalCacheMemory = (l0Stats?.size ?? 0) + (sliceStats?.size ?? 0) + l1Stats.size + l2Stats.size;
    totalEntries = (l0Stats?.count ?? 0) + (sliceStats?.count ?? 0) + l1Stats.count + l2Stats.count;
  } else if (l0Stats || sliceStats) {
    // No L1/L2 provider — fall back to in-memory (L0 + SliceCache) totals.
    totalCacheMemory = (l0Stats?.size ?? 0) + (sliceStats?.size ?? 0);
    totalEntries = (l0Stats?.count ?? 0) + (sliceStats?.count ?? 0);
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

  // Effective demand hit rate across L0/L1/L2/network. L0 hits come
  // from the L0Provider (decompressed-chunk cache); L1/L2/network
  // from the cache store's per-tier demand counters. We only compute
  // this when we have at least one usable signal; otherwise leave
  // `effectiveDemandHitRate` undefined so the UI can distinguish
  // "not wired up" from "0% hit rate."
  let effectiveDemandHitRate: number | undefined;
  const l0Hits = l0Stats?.hits ?? 0;
  // A SliceCache hit short-circuits the whole load (all lower tiers), so it
  // counts toward the effective hit rate — otherwise instant slice revisits
  // would understate it.
  const sliceHits = sliceStats?.hits ?? 0;
  if (demand) {
    const total = sliceHits + l0Hits + demand.l1Hits + demand.l2Hits + demand.networkRequests;
    if (total > 0) {
      effectiveDemandHitRate = (sliceHits + l0Hits + demand.l1Hits + demand.l2Hits) / total;
    } else {
      effectiveDemandHitRate = 0;
    }
  } else if (sliceStats && l0Stats) {
    // Both in-memory caches wired (no demand counters): combine their counters.
    const hits = sliceHits + l0Hits;
    const accesses = hits + (sliceStats.misses ?? 0) + (l0Stats.misses ?? 0);
    effectiveDemandHitRate = accesses > 0 ? hits / accesses : 0;
  } else if (sliceStats) {
    // Only SliceCache wired: use its reported hitRate.
    effectiveDemandHitRate = sliceStats.hitRate;
  } else if (l0Stats) {
    // Only L0 wired. Use the L0 provider's own hitRate.
    effectiveDemandHitRate = l0Stats.hitRate;
  }

  // Cache-status badges are surfaced in the UI and in programmatic
  // metrics for debug snapshots and E2E assertions.
  const status: CacheStatusBadge[] = [];
  switch (telemetryState.kind) {
    case 'enabled':
      status.push('cache-enabled');
      break;
    case 'disabled-no-cache':
      status.push('no-cache');
      break;
    case 'disabled-config':
      status.push('disabled-config');
      break;
    case 'not-wired':
      // Plain 'not-wired' means no provider is attached yet, usually
      // because a scene transition is mid-flight. Do not add a badge
      // here; the contradiction check below handles enabled-but-missing
      // providers.
      break;
  }
  // Provider-health: telemetry classifier said enabled but providers
  // are absent → contradiction; flag as provider-missing.
  if (telemetryState.kind === 'enabled' && !cacheStatsProvider && !l0Provider && !sliceProvider) {
    status.push('provider-missing');
  }
  if (l2QuotaSkipped > 0) status.push('quota-constrained');
  if (l2WriteFailures > 0 || l2Corrupted > 0 || l2MetadataParseFailures > 0) {
    status.push('cache-errors-detected');
  }
  if (health?.unvalidatedExternalDataset) {
    status.push('unvalidated-external-dataset');
  }
  // S2: emit `opfs-unavailable` only when the provider explicitly
  // says false. `undefined` means an older provider that doesn't
  // expose the field — assume available (no badge).
  if (health?.opfsAvailable === false) {
    status.push('opfs-unavailable');
  }

  return {
    totalCacheMemory,
    memoryLimit,
    memoryPercent,
    totalEntries,
    totalAccesses: totalL1Accesses,
    recentHitRate,
    effectiveDemandHitRate,
    // `evictions` accumulated across loaders.
    evictionsTotal: evictions,
    avgEntrySize: totalEntries > 0 ? totalCacheMemory / totalEntries : 0,
    reuseRatio: 0,
    // Rolling per-second rates from `./rates.ts`. Don't recompute
    // here as `lifetime/60` — that ratio drifts as history accumulates.
    hitsPerSecond: rates.hitsPerSec,
    missesPerSecond: rates.missesPerSec,
    avgAccessTime: 0,
    queriesPerSec: rates.queriesPerSec,
    loadsPerSec: rates.loadsPerSec,
    bandwidth: rates.bandwidth,
    l0: l0Stats,
    slice: sliceStats,
    l1: l1Stats,
    l2: l2Stats,
    network: networkStats,
    enabled: cacheEnabled,
    telemetryState,
    status,
    health,
  };
}
