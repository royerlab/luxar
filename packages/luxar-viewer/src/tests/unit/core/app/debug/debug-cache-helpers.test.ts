/**
 * Unit tests for core/debug-cache-helpers.ts.
 *
 * Pure factory + tiny dispatch wrappers. Tests pass a stub loader
 * provider to exercise the not-found / no-cache / success paths
 * for each of the six cache-tier helpers.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  buildDebugCacheHelpers,
  type CacheCapableLoader,
} from '../../../../../core/app/debug/debug-cache-helpers';

// vitest's Mock type doesn't structurally satisfy concrete method signatures
// (e.g. `() => CacheStatsSnapshot`), so the stub keeps the shape inline and we
// cast through unknown at the call site.
interface LoaderStub {
  getCacheStats: ReturnType<typeof vi.fn>;
  listCachedDatasets: ReturnType<typeof vi.fn>;
  clearL0Cache: ReturnType<typeof vi.fn>;
  clearL1Cache: ReturnType<typeof vi.fn>;
  clearL2Cache: ReturnType<typeof vi.fn>;
  clearAllCaches: ReturnType<typeof vi.fn>;
  hasCachingStore: boolean;
}

function makeLoader(hasCachingStore: boolean): LoaderStub {
  return {
    getCacheStats: vi.fn().mockReturnValue({ l0: { entries: 1 } }),
    listCachedDatasets: vi.fn().mockResolvedValue(['a', 'b']),
    clearL0Cache: vi.fn(),
    clearL1Cache: vi.fn(),
    clearL2Cache: vi.fn().mockResolvedValue(undefined),
    clearAllCaches: vi.fn().mockResolvedValue(undefined),
    hasCachingStore,
  };
}

const asLoader = (l: LoaderStub) => l as unknown as CacheCapableLoader;

describe('buildDebugCacheHelpers', () => {
  describe('getStats', () => {
    it('returns the loader stats when one is available', () => {
      const loader = makeLoader(true);
      const helpers = buildDebugCacheHelpers(() => asLoader(loader));
      expect(helpers.getStats()).toEqual({ l0: { entries: 1 } });
      expect(loader.getCacheStats).toHaveBeenCalled();
    });

    it('returns an error envelope when no loader is available', () => {
      const helpers = buildDebugCacheHelpers(() => null);
      expect(helpers.getStats()).toEqual({ error: 'No active loader found' });
    });
  });

  describe('listDatasets', () => {
    it('forwards to listCachedDatasets when loader has a caching store', async () => {
      const loader = makeLoader(true);
      const helpers = buildDebugCacheHelpers(() => asLoader(loader));
      await expect(helpers.listDatasets()).resolves.toEqual(['a', 'b']);
    });

    it('returns an error envelope when no loader is available', async () => {
      const helpers = buildDebugCacheHelpers(() => null);
      await expect(helpers.listDatasets()).resolves.toEqual({
        error: 'No active cache found',
      });
    });

    it('returns an error envelope when loader has no caching store', async () => {
      const loader = makeLoader(false);
      const helpers = buildDebugCacheHelpers(() => asLoader(loader));
      await expect(helpers.listDatasets()).resolves.toEqual({
        error: 'No active cache found',
      });
      expect(loader.listCachedDatasets).not.toHaveBeenCalled();
    });
  });

  describe('clearL0', () => {
    it('forwards to clearL0Cache when loader is available', () => {
      const loader = makeLoader(false); // L0 doesn't need a caching store
      const helpers = buildDebugCacheHelpers(() => asLoader(loader));
      helpers.clearL0();
      expect(loader.clearL0Cache).toHaveBeenCalled();
    });

    it('no-ops when no loader is available', () => {
      const helpers = buildDebugCacheHelpers(() => null);
      // Must not throw.
      expect(() => helpers.clearL0()).not.toThrow();
    });
  });

  describe('clearL1', () => {
    it('forwards to clearL1Cache when loader has a caching store', () => {
      const loader = makeLoader(true);
      const helpers = buildDebugCacheHelpers(() => asLoader(loader));
      helpers.clearL1();
      expect(loader.clearL1Cache).toHaveBeenCalled();
    });

    it('no-ops when loader has no caching store', () => {
      const loader = makeLoader(false);
      const helpers = buildDebugCacheHelpers(() => asLoader(loader));
      helpers.clearL1();
      expect(loader.clearL1Cache).not.toHaveBeenCalled();
    });

    it('no-ops when no loader is available', () => {
      const helpers = buildDebugCacheHelpers(() => null);
      expect(() => helpers.clearL1()).not.toThrow();
    });
  });

  describe('clearL2', () => {
    let loader: LoaderStub;

    beforeEach(() => {
      loader = makeLoader(true);
    });

    it('awaits clearL2Cache when loader has a caching store', async () => {
      const helpers = buildDebugCacheHelpers(() => asLoader(loader));
      await helpers.clearL2();
      expect(loader.clearL2Cache).toHaveBeenCalled();
    });

    it('no-ops when loader has no caching store', async () => {
      const noStoreLoader = makeLoader(false);
      const helpers = buildDebugCacheHelpers(() => asLoader(noStoreLoader));
      await helpers.clearL2();
      expect(noStoreLoader.clearL2Cache).not.toHaveBeenCalled();
    });

    it('no-ops when no loader is available', async () => {
      const helpers = buildDebugCacheHelpers(() => null);
      await expect(helpers.clearL2()).resolves.toBeUndefined();
    });
  });

  describe('clearAll', () => {
    it('awaits clearAllCaches when loader is available (regardless of caching store)', async () => {
      const loader = makeLoader(false); // clearAll doesn't gate on hasCachingStore
      const helpers = buildDebugCacheHelpers(() => asLoader(loader));
      await helpers.clearAll();
      expect(loader.clearAllCaches).toHaveBeenCalled();
    });

    it('no-ops when no loader is available', async () => {
      const helpers = buildDebugCacheHelpers(() => null);
      await expect(helpers.clearAll()).resolves.toBeUndefined();
    });
  });

  describe('loader-getter is called per invocation (not cached)', () => {
    it('looks up the loader each time so dataset switches are picked up', () => {
      let current: LoaderStub | null = null;
      const helpers = buildDebugCacheHelpers(() => (current ? asLoader(current) : null));

      // Initially no loader.
      expect(helpers.getStats()).toEqual({ error: 'No active loader found' });

      // Loader becomes available later.
      current = makeLoader(true);
      const stats = helpers.getStats();
      expect(stats).not.toHaveProperty('error');
      expect(current.getCacheStats).toHaveBeenCalled();
    });
  });
});
