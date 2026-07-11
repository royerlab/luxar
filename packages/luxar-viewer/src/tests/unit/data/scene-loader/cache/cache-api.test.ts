/**
 * Unit tests for the cache-api helpers.
 *
 * These verify the helpers do exactly what the SceneLoader inline
 * methods used to do — graceful no-ops on `null`, correct snapshot
 * shape, correct delegation to the underlying cache layers. We don't
 * construct real cache instances (they touch IndexedDB / OPFS / zarr);
 * stubs with the right shape are sufficient.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  getCacheStats,
  listCachedDatasets,
  clearL0Cache,
  clearL1Cache,
  clearL2Cache,
  clearAllCaches,
} from '../../../../../data/scene-loader/cache/cache-api';
import type { MultiLevelCachingStore } from '../../../../../cache/multi-level-caching-store';
import type { DecompressedChunkCache } from '../../../../../cache/decompressed-chunk-cache';

function makeL0Stub() {
  const stats = { entries: 3, sizeBytes: 1024 };
  const clear = vi.fn();
  const stub = {
    getStats: () => stats,
    clear,
  } as unknown as DecompressedChunkCache;
  return { stub, stats, clear };
}

function makeStoreStub() {
  const l1Stats = { hitRate: 0.5, sizeBytes: 2048 };
  const l2Stats = { hitRate: 0.2, sizeBytes: 8192 };
  const datasets = [{ url: 'a' }, { url: 'b' }];
  const clearL1 = vi.fn();
  const clearL2 = vi.fn(async () => {});
  const clearAll = vi.fn(async () => {});
  const listDatasets = vi.fn(async () => datasets);
  const stub = {
    getStats: () => ({ l1: l1Stats, l2: l2Stats }),
    clearL1,
    clearL2,
    clearAll,
    listDatasets,
  } as unknown as MultiLevelCachingStore;
  return { stub, l1Stats, l2Stats, datasets, clearL1, clearL2, clearAll, listDatasets };
}

describe('getCacheStats', () => {
  it('returns nulls when both caches are absent', () => {
    expect(getCacheStats(null, null)).toMatchObject({ l0: null, l1: null, l2: null });
  });

  it('reports L0 only when only the L0 cache is set', () => {
    const { stub, stats } = makeL0Stub();
    expect(getCacheStats(stub, null)).toMatchObject({ l0: stats, l1: null, l2: null });
  });

  it('reports L1 + L2 only when only the caching store is set', () => {
    const { stub, l1Stats, l2Stats } = makeStoreStub();
    expect(getCacheStats(null, stub)).toMatchObject({ l0: null, l1: l1Stats, l2: l2Stats });
  });

  it('reports all three levels when all are present', () => {
    const l0 = makeL0Stub();
    const store = makeStoreStub();
    expect(getCacheStats(l0.stub, store.stub)).toMatchObject({
      l0: l0.stats,
      l1: store.l1Stats,
      l2: store.l2Stats,
    });
  });

  it('surfaces the S-cache slice stats (incl. the heap-aware budget) when present, null otherwise', () => {
    const sliceStats = {
      size: 42,
      count: 3,
      hits: 10,
      misses: 4,
      evictions: 1,
      hitRate: 10 / 14,
      thrashMisses: 2,
      maxSize: 512 * 1024 * 1024,
    };
    const sliceStub = { getStats: () => sliceStats } as unknown as Parameters<
      typeof getCacheStats
    >[2];
    expect(getCacheStats(null, null, sliceStub)!.slice).toEqual(sliceStats);
    // Absent SliceCache → null (and the default 2-arg call keeps slice null).
    expect(getCacheStats(null, null).slice).toBeNull();
  });

  it('extends the snapshot with network/demand/prefetch/health fields (commit 7.1)', () => {
    // Stub a caching store that exposes everything the snapshot
    // surfaces — including the prefetcher and the new health field.
    const stubGetStats = vi.fn(() => ({
      l1: {
        metadataSize: 0,
        chunksSize: 0,
        metadataCount: 0,
        chunksCount: 0,
        hits: 0,
        misses: 0,
        evictions: 0,
      },
      l2: { size: 0, count: 0, reads: 0, writes: 0, misses: 0 },
      network: { bytesTransferred: 1234, requestCount: 5, bandwidth: 100 },
      demand: { l1Hits: 1, l2Hits: 2, networkRequests: 3 },
      health: {
        validationMode: 'content-hash' as const,
        lastValidatedAt: 99,
        unvalidatedExternalDataset: false,
      },
    }));
    const stubPrefetcher = {
      getStats: () => ({ queued: 4, inFlight: 1, enabled: true }),
    };
    const stub = {
      getStats: stubGetStats,
      getPrefetcher: () => stubPrefetcher,
    } as unknown as MultiLevelCachingStore;

    const snapshot = getCacheStats(null, stub);
    expect(snapshot.network).toEqual({ bytesTransferred: 1234, requestCount: 5, bandwidth: 100 });
    expect(snapshot.demand).toEqual({ l1Hits: 1, l2Hits: 2, networkRequests: 3 });
    expect(snapshot.prefetch).toEqual({ queued: 4, inFlight: 1, enabled: true });
    expect(snapshot.health).toEqual({
      validationMode: 'content-hash',
      lastValidatedAt: 99,
      unvalidatedExternalDataset: false,
    });
  });
});

describe('listCachedDatasets', () => {
  it('returns [] when no caching store is set', async () => {
    await expect(listCachedDatasets(null)).resolves.toEqual([]);
  });

  it('forwards to the store and returns its dataset list', async () => {
    const { stub, datasets, listDatasets } = makeStoreStub();
    await expect(listCachedDatasets(stub)).resolves.toEqual(datasets);
    expect(listDatasets).toHaveBeenCalledTimes(1);
  });
});

describe('clear* helpers', () => {
  it('clearL0Cache no-ops on null and forwards to stub', () => {
    expect(() => clearL0Cache(null)).not.toThrow();
    const { stub, clear } = makeL0Stub();
    clearL0Cache(stub);
    expect(clear).toHaveBeenCalledTimes(1);
  });

  it('clearL1Cache no-ops on null and forwards to stub', () => {
    expect(() => clearL1Cache(null)).not.toThrow();
    const { stub, clearL1 } = makeStoreStub();
    clearL1Cache(stub);
    expect(clearL1).toHaveBeenCalledTimes(1);
  });

  it('clearL2Cache no-ops on null and forwards to stub', async () => {
    await expect(clearL2Cache(null)).resolves.toBeUndefined();
    const { stub, clearL2 } = makeStoreStub();
    await clearL2Cache(stub);
    expect(clearL2).toHaveBeenCalledTimes(1);
  });

  it('clearAllCaches handles every (l0, store) combination', async () => {
    // both null
    await expect(clearAllCaches(null, null)).resolves.toBeUndefined();

    // only L0
    const l0Only = makeL0Stub();
    await clearAllCaches(l0Only.stub, null);
    expect(l0Only.clear).toHaveBeenCalledTimes(1);

    // only store
    const storeOnly = makeStoreStub();
    await clearAllCaches(null, storeOnly.stub);
    expect(storeOnly.clearAll).toHaveBeenCalledTimes(1);

    // both — L0 cleared synchronously, store.clearAll awaited
    const l0 = makeL0Stub();
    const store = makeStoreStub();
    await clearAllCaches(l0.stub, store.stub);
    expect(l0.clear).toHaveBeenCalledTimes(1);
    expect(store.clearAll).toHaveBeenCalledTimes(1);
  });

  it('clearAllCaches also clears the S-cache when provided (all four tiers)', async () => {
    const l0 = makeL0Stub();
    const store = makeStoreStub();
    const sliceClear = vi.fn();
    const sliceStub = { clear: sliceClear } as unknown as Parameters<typeof clearAllCaches>[2];

    await clearAllCaches(l0.stub, store.stub, sliceStub);
    expect(l0.clear).toHaveBeenCalledTimes(1);
    expect(store.clearAll).toHaveBeenCalledTimes(1);
    expect(sliceClear).toHaveBeenCalledTimes(1);

    // Null S-cache stays a no-op (pre-slice-cache callers).
    await expect(clearAllCaches(null, null, null)).resolves.toBeUndefined();
  });
});
