/**
 * Tests for SceneLoader's typed cache accessors.
 *
 * These methods replaced 7+ `(loader as any).cachingStore` casts in
 * core/app.ts:setupDebugInterface. They give debug tools and embedders
 * a typed surface and behave gracefully when the cache layer is absent
 * (the `?noCache` code path that embedders are most likely to hit).
 */

import { describe, expect, it } from 'vitest';
import { SceneLoader } from '../../../../data/scene-loader';

describe('SceneLoader cache accessors', () => {
  describe('with no cache layer (fresh loader, before loadScene)', () => {
    const loader = new SceneLoader();

    it('reports hasCachingStore=false', () => {
      expect(loader.hasCachingStore).toBe(false);
    });

    it('returns null for every cache level in getCacheStats()', () => {
      const stats = loader.getCacheStats();
      expect(stats.l0).toBeNull();
      expect(stats.l1).toBeNull();
      expect(stats.l2).toBeNull();
    });

    it('returns an empty array from listCachedDatasets()', async () => {
      await expect(loader.listCachedDatasets()).resolves.toEqual([]);
    });

    it('treats clear methods as no-ops without throwing', async () => {
      expect(() => loader.clearL0Cache()).not.toThrow();
      expect(() => loader.clearL1Cache()).not.toThrow();
      await expect(loader.clearL2Cache()).resolves.toBeUndefined();
      await expect(loader.clearAllCaches()).resolves.toBeUndefined();
    });
  });

  describe('with stub cache instances injected (smoke)', () => {
    /**
     * We don't construct real MultiLevelCachingStore / DecompressedChunkCache
     * here — both touch IndexedDB, OPFS, or zarr. Instead, we inject minimal
     * shape-compatible stubs by reaching into the private fields through a
     * narrow interface. This validates that the public methods forward to the
     * right underlying calls.
     */
    function withStubs() {
      const loader = new SceneLoader();
      const calls: string[] = [];

      const stubL0 = {
        getStats: () => ({
          size: 100,
          count: 1,
          hits: 0,
          misses: 0,
          evictions: 0,
          hitRate: 0,
        }),
        clear: () => {
          calls.push('l0.clear');
        },
      };

      const stubCachingStore = {
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
          l2: { size: 0, count: 0, reads: 0, writes: 0 },
          network: { bytesTransferred: 0, requestCount: 0, bandwidth: 0 },
        }),
        listDatasets: async () => [{ url: 'x', hash: 'h', size: 1, count: 1 }],
        clearL1: () => {
          calls.push('store.clearL1');
        },
        clearL2: async () => {
          calls.push('store.clearL2');
        },
        clearAll: async () => {
          calls.push('store.clearAll');
        },
      };

      // Inject via a private-field cast rather than widening the loader's
      // public surface for testing — these tests live alongside the private
      // members they exercise.
      const loaderAny = loader as unknown as { l0Cache: unknown; cachingStore: unknown };
      loaderAny.l0Cache = stubL0;
      loaderAny.cachingStore = stubCachingStore;

      return { loader, calls };
    }

    it('hasCachingStore reflects an injected store', () => {
      const { loader } = withStubs();
      expect(loader.hasCachingStore).toBe(true);
    });

    it('getCacheStats merges l0 + l1 + l2 from underlying stores', () => {
      const { loader } = withStubs();
      const stats = loader.getCacheStats();
      expect(stats.l0).not.toBeNull();
      expect(stats.l1).not.toBeNull();
      expect(stats.l2).not.toBeNull();
    });

    it('listCachedDatasets forwards to the store', async () => {
      const { loader } = withStubs();
      await expect(loader.listCachedDatasets()).resolves.toEqual([
        { url: 'x', hash: 'h', size: 1, count: 1 },
      ]);
    });

    it('clear methods forward to the right level', async () => {
      const { loader, calls } = withStubs();
      loader.clearL0Cache();
      loader.clearL1Cache();
      await loader.clearL2Cache();
      expect(calls).toEqual(['l0.clear', 'store.clearL1', 'store.clearL2']);
    });

    it('clearAllCaches clears L0 then asks the store to clear L1+L2', async () => {
      const { loader, calls } = withStubs();
      await loader.clearAllCaches();
      expect(calls).toEqual(['l0.clear', 'store.clearAll']);
    });
  });
});
