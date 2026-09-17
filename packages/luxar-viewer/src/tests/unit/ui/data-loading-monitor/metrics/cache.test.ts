/**
 * Direct unit tests for the cache-metrics aggregator. Each branch
 * (no providers, L0 only, full provider, disabled provider, demand
 * counters, telemetry state, rolling rates) is pinned explicitly so
 * the aggregator's surface stays observable independent of the
 * monitor's integration tests.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  aggregateCacheMetrics,
  type CacheRatesSnapshot,
  type L0Provider,
  type SliceProvider,
} from '../../../../../ui/data-loading-monitor/metrics/cache';
import type {
  LoaderMonitor,
  LoaderMetrics,
  CacheStatsProvider,
} from '../../../../../types/data-monitor-types';

const ZERO_RATES: CacheRatesSnapshot = {
  queriesPerSec: 0,
  loadsPerSec: 0,
  bandwidth: 0,
};

function makeLoaderMetrics(overrides: Partial<LoaderMetrics> = {}): LoaderMetrics {
  return {
    type: 'point-spatial-index',
    path: '/p',
    queries: 0,
    loads: 0,
    errors: 0,
    elementsLoaded: 0,
    bytesLoaded: 0,
    visibleElements: 0,
    avgQueryTime: 0,
    avgLoadTime: 0,
    memoryUsed: 0,
    ...overrides,
  };
}

function makeLoader(metrics: LoaderMetrics): LoaderMonitor {
  return {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    getMetrics: () => metrics,
    getActiveQueries: () => [],
  };
}

function makeFullCacheProvider(opts: { enabled?: boolean } = {}): CacheStatsProvider {
  return {
    getStats: () => ({
      l1: {
        metadataSize: 100,
        chunksSize: 200,
        metadataCount: 5,
        chunksCount: 10,
        hits: 80,
        misses: 20,
        evictions: 3,
      },
      l2: {
        size: 1000,
        count: 50,
        reads: 30,
        writes: 25,
        misses: 5,
      },
      network: {
        bytesTransferred: 12345,
        requestCount: 7,
        bandwidth: 9999,
      },
    }),
    clearL1: vi.fn(),
    clearL2: vi.fn(),
    clearAll: vi.fn(),
    isEnabled: () => opts.enabled ?? true,
  };
}

describe('aggregateCacheMetrics', () => {
  it('returns zeros when there are no providers and no loaders', () => {
    const metricsCache = new Map<string, LoaderMetrics>();
    const result = aggregateCacheMetrics({
      l0Provider: null,
      cacheStatsProvider: null,
      loaders: new Map(),
      metricsCache,
      rates: ZERO_RATES,
    });

    expect(result.totalCacheMemory).toBe(0);
    expect(result.memoryLimit).toBe(0);
    expect(result.totalEntries).toBe(0);
    expect(result.l0).toBeUndefined();
    expect(result.l1).toBeUndefined();
    expect(result.l2).toBeUndefined();
    expect(result.network).toBeUndefined();
    // No provider → not-wired (must NOT default-enabled — that
    // would mislead `?noCache` users into thinking caching is on).
    expect(result.enabled).toBe(false);
    expect(result.telemetryState?.kind).toBe('not-wired');
    expect(metricsCache.size).toBe(0);
  });

  it('with only an L0 provider: totals come from L0 alone', () => {
    const l0Provider: L0Provider = {
      getStats: () => ({
        size: 500,
        count: 7,
        hits: 0,
        misses: 0,
        evictions: 0,
        hitRate: 0,
      }),
    };
    const result = aggregateCacheMetrics({
      l0Provider,
      cacheStatsProvider: null,
      loaders: new Map(),
      metricsCache: new Map(),
      rates: ZERO_RATES,
    });

    expect(result.l0?.size).toBe(500);
    expect(result.l0?.count).toBe(7);
    expect(result.totalCacheMemory).toBe(500);
    expect(result.totalEntries).toBe(7);
    expect(result.l1).toBeUndefined();
    expect(result.l2).toBeUndefined();
  });

  it('surfaces SliceCache stats and folds them into the in-memory totals', () => {
    const sliceProvider: SliceProvider = {
      getStats: () => ({
        size: 4096,
        count: 3,
        hits: 6,
        misses: 4,
        evictions: 1,
        hitRate: 0.6,
      }),
    };
    const l0Provider: L0Provider = {
      getStats: () => ({ size: 500, count: 7, hits: 0, misses: 0, evictions: 0, hitRate: 0 }),
    };
    const result = aggregateCacheMetrics({
      l0Provider,
      sliceProvider,
      cacheStatsProvider: null,
      loaders: new Map(),
      metricsCache: new Map(),
      rates: ZERO_RATES,
    });

    expect(result.slice?.size).toBe(4096);
    expect(result.slice?.count).toBe(3);
    expect(result.slice?.hits).toBe(6);
    // In-memory totals combine L0 (500/7) + SliceCache (4096/3).
    expect(result.totalCacheMemory).toBe(4596);
    expect(result.totalEntries).toBe(10);
    // SliceCache hits count toward the effective hit rate (6 hits / 10 accesses).
    expect(result.effectiveDemandHitRate).toBeCloseTo(0.6, 5);
  });

  it('omits the SliceCache breakdown when no sliceProvider is supplied', () => {
    const result = aggregateCacheMetrics({
      l0Provider: null,
      cacheStatsProvider: null,
      loaders: new Map(),
      metricsCache: new Map(),
      rates: ZERO_RATES,
    });
    expect(result.slice).toBeUndefined();
  });

  it('with full cache provider: totals are L0 + L1 + L2 sizes/counts', () => {
    const l0Provider: L0Provider = {
      getStats: () => ({
        size: 300,
        count: 3,
        hits: 0,
        misses: 0,
        evictions: 0,
        hitRate: 0,
      }),
    };
    const cacheStatsProvider = makeFullCacheProvider();
    const result = aggregateCacheMetrics({
      l0Provider,
      cacheStatsProvider,
      loaders: new Map(),
      metricsCache: new Map(),
      rates: ZERO_RATES,
    });

    // L0 size 300 + L1 (100+200=300) + L2 (1000) = 1600
    expect(result.totalCacheMemory).toBe(1600);
    // L0 count 3 + L1 (5+10=15) + L2 (50) = 68
    expect(result.totalEntries).toBe(68);
    expect(result.l1?.hits).toBe(80);
    expect(result.network?.bandwidth).toBe(9999);
  });

  it('cacheStatsProvider.isEnabled === false propagates to result.enabled', () => {
    const cacheStatsProvider = makeFullCacheProvider({ enabled: false });
    const result = aggregateCacheMetrics({
      l0Provider: null,
      cacheStatsProvider,
      loaders: new Map(),
      metricsCache: new Map(),
      rates: ZERO_RATES,
    });
    expect(result.enabled).toBe(false);
    expect(result.telemetryState?.kind).toBe('disabled-config');
  });

  describe('explicit telemetry state', () => {
    it("explicit 'disabled-no-cache' wins over inferred 'not-wired'", () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: null,
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
        telemetryState: { kind: 'disabled-no-cache' },
      });
      expect(result.enabled).toBe(false);
      expect(result.telemetryState?.kind).toBe('disabled-no-cache');
    });

    it("explicit 'enabled' wins over inferred 'not-wired'", () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: null,
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
        telemetryState: { kind: 'enabled' },
      });
      expect(result.enabled).toBe(true);
      expect(result.telemetryState?.kind).toBe('enabled');
    });
  });

  it('without cacheStatsProvider: totalCacheMemory falls back to per-loader memoryUsed', () => {
    const m1 = makeLoaderMetrics({ path: '/p1', memoryUsed: 100 });
    const m2 = makeLoaderMetrics({ path: '/p2', memoryUsed: 200 });
    const loaders = new Map([
      ['/p1', makeLoader(m1)],
      ['/p2', makeLoader(m2)],
    ]);
    const metricsCache = new Map<string, LoaderMetrics>();

    const result = aggregateCacheMetrics({
      l0Provider: null,
      cacheStatsProvider: null,
      loaders,
      metricsCache,
      rates: ZERO_RATES,
    });

    // No cache provider exposes a byte budget → no memory limit.
    expect(result.memoryLimit).toBe(0);
    expect(result.totalCacheMemory).toBe(300);
    expect(result.totalEntries).toBe(0);
    // metricsCache mutation: each loader's getMetrics() result was stored.
    expect(metricsCache.get('/p1')).toBe(m1);
    expect(metricsCache.get('/p2')).toBe(m2);
  });

  it('with cacheStatsProvider: loader memoryUsed is NOT double-counted into totalCacheMemory', () => {
    const m1 = makeLoaderMetrics({ path: '/p1', memoryUsed: 999 });
    const loaders = new Map([['/p1', makeLoader(m1)]]);

    const result = aggregateCacheMetrics({
      l0Provider: null,
      cacheStatsProvider: makeFullCacheProvider(),
      loaders,
      metricsCache: new Map(),
      rates: ZERO_RATES,
    });

    // totalCacheMemory comes from L1+L2 (300 + 1000 = 1300), NOT
    // from the loader's memoryUsed when cacheStatsProvider is set.
    expect(result.totalCacheMemory).toBe(1300);
    // memoryLimit comes solely from per-tier byte budgets; this stub
    // exposes no maxSize, so there is no limit.
    expect(result.memoryLimit).toBe(0);
  });

  it('memoryLimit is the sum of per-tier budgets (maxSize) when the providers expose them', () => {
    // Regression for the "no memory limit configured" gauge: the limit is the
    // sum of the resolved per-tier byte budgets, not the (hardcoded-0) per-loader
    // memoryLimit. Fixes the pressure bar reading "no memory limit configured".
    const l0Provider: L0Provider = {
      getStats: () => ({
        size: 0,
        count: 0,
        hits: 0,
        misses: 0,
        evictions: 0,
        hitRate: 0,
        maxSize: 200,
      }),
    };
    const sliceProvider: SliceProvider = {
      getStats: () => ({
        size: 0,
        count: 0,
        hits: 0,
        misses: 0,
        evictions: 0,
        hitRate: 0,
        maxSize: 1024,
      }),
    };
    const cacheStatsProvider: CacheStatsProvider = {
      ...makeFullCacheProvider(),
      getStats: () => ({
        l1: {
          metadataSize: 0,
          chunksSize: 0,
          metadataCount: 0,
          chunksCount: 0,
          hits: 0,
          misses: 0,
          evictions: 0,
          maxSize: 100,
        },
        l2: { size: 0, count: 0, reads: 0, writes: 0, misses: 0, maxSize: 2048 },
        network: { bytesTransferred: 0, requestCount: 0, bandwidth: 0 },
      }),
    };
    const result = aggregateCacheMetrics({
      l0Provider,
      sliceProvider,
      cacheStatsProvider,
      loaders: new Map([['/p', makeLoader(makeLoaderMetrics())]]),
      metricsCache: new Map(),
      rates: ZERO_RATES,
    });
    // 200 (L0) + 1024 (S-cache) + 100 (L1) + 2048 (L2).
    expect(result.memoryLimit).toBe(200 + 1024 + 100 + 2048);
  });

  it('rates are passed through unchanged', () => {
    const rates: CacheRatesSnapshot = {
      queriesPerSec: 12,
      loadsPerSec: 7,
      bandwidth: 4096,
    };
    const result = aggregateCacheMetrics({
      l0Provider: null,
      cacheStatsProvider: null,
      loaders: new Map(),
      metricsCache: new Map(),
      rates,
    });
    expect(result.queriesPerSec).toBe(12);
    expect(result.loadsPerSec).toBe(7);
    expect(result.bandwidth).toBe(4096);
  });

  it('memoryPercent = totalCacheMemory / memoryLimit × 100', () => {
    // 250 bytes used against a 1000-byte L0 budget → 25%.
    const l0Provider: L0Provider = {
      getStats: () => ({
        size: 250,
        count: 1,
        hits: 0,
        misses: 0,
        evictions: 0,
        hitRate: 0,
        maxSize: 1000,
      }),
    };
    const result = aggregateCacheMetrics({
      l0Provider,
      cacheStatsProvider: null,
      loaders: new Map(),
      metricsCache: new Map(),
      rates: ZERO_RATES,
    });
    expect(result.memoryPercent).toBeCloseTo(25.0, 5);
  });

  it('memoryPercent is 0 when memoryLimit is 0', () => {
    const result = aggregateCacheMetrics({
      l0Provider: null,
      cacheStatsProvider: null,
      loaders: new Map(),
      metricsCache: new Map(),
      rates: ZERO_RATES,
    });
    expect(result.memoryPercent).toBe(0);
  });

  describe('effectiveDemandHitRate', () => {
    function providerWithDemand(demand: {
      l1Hits: number;
      l2Hits: number;
      networkRequests: number;
    }): CacheStatsProvider {
      return {
        getStats: () => ({
          l1: {
            metadataSize: 0,
            chunksSize: 0,
            metadataCount: 0,
            chunksCount: 0,
            hits: demand.l1Hits,
            misses: 0,
            evictions: 0,
          },
          l2: {
            size: 0,
            count: 0,
            reads: demand.l2Hits,
            writes: 0,
            misses: 0,
          },
          network: {
            bytesTransferred: 0,
            requestCount: demand.networkRequests,
            bandwidth: 0,
          },
          demand,
        }),
        clearL1: vi.fn(),
        clearL2: vi.fn(),
        clearAll: vi.fn(),
        isEnabled: () => true,
      };
    }

    it('undefined when nothing is wired up', () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: null,
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.effectiveDemandHitRate).toBeUndefined();
    });

    it('with demand counters but zero traffic: 0 (not undefined)', () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: providerWithDemand({
          l1Hits: 0,
          l2Hits: 0,
          networkRequests: 0,
        }),
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.effectiveDemandHitRate).toBe(0);
    });

    it('demand 90 L1 hits + 10 network = 0.9 hit rate', () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: providerWithDemand({
          l1Hits: 90,
          l2Hits: 0,
          networkRequests: 10,
        }),
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.effectiveDemandHitRate).toBeCloseTo(0.9, 5);
    });

    it('demand counts L0 hits when L0 provider is present', () => {
      const l0Provider: L0Provider = {
        getStats: () => ({
          size: 0,
          count: 0,
          hits: 50,
          misses: 0,
          evictions: 0,
          hitRate: 1,
        }),
      };
      const result = aggregateCacheMetrics({
        l0Provider,
        cacheStatsProvider: providerWithDemand({
          l1Hits: 30,
          l2Hits: 10,
          networkRequests: 10,
        }),
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      // (50 + 30 + 10) / (50 + 30 + 10 + 10) = 90/100
      expect(result.effectiveDemandHitRate).toBeCloseTo(0.9, 5);
    });

    it('falls back to L0.hitRate when only L0 is wired (no demand counters)', () => {
      const l0Provider: L0Provider = {
        getStats: () => ({
          size: 0,
          count: 0,
          hits: 0,
          misses: 0,
          evictions: 0,
          hitRate: 0.42,
        }),
      };
      const result = aggregateCacheMetrics({
        l0Provider,
        cacheStatsProvider: null,
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.effectiveDemandHitRate).toBe(0.42);
    });

    it('older provider without demand field: effectiveDemandHitRate undefined', () => {
      // Use the existing makeFullCacheProvider which DOES NOT include `demand`.
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: makeFullCacheProvider(),
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.effectiveDemandHitRate).toBeUndefined();
    });
  });

  describe('status badges (commits 7.2 + 7.4)', () => {
    function makeProviderWithHealth(overrides: {
      quotaWriteSkipped?: number;
      writeFailures?: number;
      corruptedEntries?: number;
      metadataParseFailures?: number;
      validationMode?: 'content-hash' | 'ttl' | 'none';
      unvalidatedExternalDataset?: boolean;
      enabled?: boolean;
    }) {
      return {
        getStats: () => ({
          l1: {
            metadataSize: 0,
            chunksSize: 0,
            metadataCount: 0,
            chunksCount: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
          },
          l2: {
            size: 0,
            count: 0,
            reads: 0,
            writes: 0,
            misses: 0,
            quotaWriteSkipped: overrides.quotaWriteSkipped ?? 0,
            writeFailures: overrides.writeFailures ?? 0,
            corruptedEntries: overrides.corruptedEntries ?? 0,
            metadataParseFailures: overrides.metadataParseFailures ?? 0,
          },
          network: { bytesTransferred: 0, requestCount: 0, bandwidth: 0 },
          health: {
            validationMode: overrides.validationMode ?? 'content-hash',
            lastValidatedAt: null,
            unvalidatedExternalDataset: overrides.unvalidatedExternalDataset ?? false,
          },
        }),
        clearL1: () => {},
        clearL2: async () => {},
        clearAll: async () => {},
        isEnabled: () => overrides.enabled ?? true,
      };
    }

    it("status includes 'cache-enabled' when telemetry is enabled", () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: makeProviderWithHealth({}),
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.status).toContain('cache-enabled');
    });

    it("status includes 'no-cache' when explicit telemetry says disabled-no-cache", () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: null,
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
        telemetryState: { kind: 'disabled-no-cache' },
      });
      expect(result.status).toContain('no-cache');
    });

    it("status includes 'quota-constrained' when L2.quotaWriteSkipped > 0", () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: makeProviderWithHealth({ quotaWriteSkipped: 3 }),
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.status).toContain('quota-constrained');
    });

    it("status includes 'cache-errors-detected' when any L2 error counter > 0", () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: makeProviderWithHealth({ writeFailures: 1 }),
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.status).toContain('cache-errors-detected');
    });

    it("status includes 'unvalidated-external-dataset' when health flag is set", () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: makeProviderWithHealth({
          validationMode: 'none',
          unvalidatedExternalDataset: true,
        }),
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.status).toContain('unvalidated-external-dataset');
      expect(result.health?.validationMode).toBe('none');
    });

    // S2: opfs-unavailable badge fires when the provider reports
    // health.opfsAvailable === false. Older providers omit the field —
    // in that case the badge stays unemitted.
    it("status includes 'opfs-unavailable' when health.opfsAvailable is false", () => {
      const provider = makeProviderWithHealth({});
      const baseStats = provider.getStats();
      const wrapped = {
        ...provider,
        getStats: () => ({
          ...baseStats,
          health: { ...baseStats.health, opfsAvailable: false },
        }),
      };
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: wrapped,
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.status).toContain('opfs-unavailable');
      expect(result.health?.opfsAvailable).toBe(false);
    });

    it("status does NOT include 'opfs-unavailable' when opfsAvailable is true", () => {
      const provider = makeProviderWithHealth({});
      const baseStats = provider.getStats();
      const wrapped = {
        ...provider,
        getStats: () => ({
          ...baseStats,
          health: { ...baseStats.health, opfsAvailable: true },
        }),
      };
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: wrapped,
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.status).not.toContain('opfs-unavailable');
    });

    it("status does NOT include 'opfs-unavailable' when provider omits opfsAvailable (older API)", () => {
      // makeProviderWithHealth does not include opfsAvailable in the
      // health object — older providers leave it undefined.
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: makeProviderWithHealth({}),
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.status).not.toContain('opfs-unavailable');
    });

    it("status includes 'provider-missing' when telemetry says enabled but no provider exists", () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: null,
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
        telemetryState: { kind: 'enabled' },
      });
      expect(result.status).toContain('cache-enabled');
      expect(result.status).toContain('provider-missing');
    });

    // R3: cache-tab now renders L2 error counters inline, so the
    // aggregator must surface them on `l2Stats` (not only into the
    // badge-computation scratch variables).
    it('l2 stats carry the four OPFS health counters when provider supplies them', () => {
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: makeProviderWithHealth({
          quotaWriteSkipped: 3,
          writeFailures: 1,
          corruptedEntries: 2,
          metadataParseFailures: 4,
        }),
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.l2?.quotaWriteSkipped).toBe(3);
      expect(result.l2?.writeFailures).toBe(1);
      expect(result.l2?.corruptedEntries).toBe(2);
      expect(result.l2?.metadataParseFailures).toBe(4);
    });

    it('l2 stats counters are undefined when provider omits them', () => {
      // Build a minimal provider that returns only the required L2
      // fields, exercising the older-provider compatibility path.
      const provider = {
        getStats: () => ({
          l1: {
            metadataSize: 0,
            chunksSize: 0,
            metadataCount: 0,
            chunksCount: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
          },
          l2: {
            size: 0,
            count: 0,
            reads: 0,
            writes: 0,
            misses: 0,
            // No quota/write/corrupt/metadata fields.
          },
          network: { bytesTransferred: 0, requestCount: 0, bandwidth: 0 },
        }),
        clearL1: () => {},
        clearL2: async () => {},
        clearAll: async () => {},
        isEnabled: () => true,
      };
      const result = aggregateCacheMetrics({
        l0Provider: null,
        cacheStatsProvider: provider,
        loaders: new Map(),
        metricsCache: new Map(),
        rates: ZERO_RATES,
      });
      expect(result.l2?.quotaWriteSkipped).toBeUndefined();
      expect(result.l2?.writeFailures).toBeUndefined();
      expect(result.l2?.corruptedEntries).toBeUndefined();
      expect(result.l2?.metadataParseFailures).toBeUndefined();
    });
  });
});
