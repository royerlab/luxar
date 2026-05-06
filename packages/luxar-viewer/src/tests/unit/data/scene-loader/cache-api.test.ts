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
} from '../../../../data/scene-loader/cache-api';
import type {
  MultiLevelCachingStore,
  DecompressedChunkCache,
} from '../../../../cache';

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
    expect(getCacheStats(null, null)).toEqual({ l0: null, l1: null, l2: null });
  });

  it('reports L0 only when only the L0 cache is set', () => {
    const { stub, stats } = makeL0Stub();
    expect(getCacheStats(stub, null)).toEqual({ l0: stats, l1: null, l2: null });
  });

  it('reports L1 + L2 only when only the caching store is set', () => {
    const { stub, l1Stats, l2Stats } = makeStoreStub();
    expect(getCacheStats(null, stub)).toEqual({ l0: null, l1: l1Stats, l2: l2Stats });
  });

  it('reports all three levels when all are present', () => {
    const l0 = makeL0Stub();
    const store = makeStoreStub();
    expect(getCacheStats(l0.stub, store.stub)).toEqual({
      l0: l0.stats,
      l1: store.l1Stats,
      l2: store.l2Stats,
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
});
