/**
 * Cache-tier helpers exposed under `window.__luxarDebug.cache`.
 *
 * Six thin wrappers over the SceneLoader cache API. Each follows the
 * same pattern: resolve the active loader, check whether it's
 * available (and for L1/L2 + listDatasets, whether it has a caching
 * store), then call into the loader's cache method.
 *
 * Extracted from `core/app.ts::setupDebugInterface` so the
 * "no loader / no cache / call through" branches can be unit-tested
 * directly without spinning up a full SceneLoaderManager singleton.
 *
 * @module core/app/debug/debug-cache-helpers
 */

import type { CacheStatsSnapshot } from '../../../data/scene-loader/cache/cache-api';
import { log, Modules } from '../../../utils/log';

/** Subset of `SceneLoader` the cache helpers call into. */
export interface CacheCapableLoader {
  getCacheStats(): CacheStatsSnapshot;
  listCachedDatasets(): Promise<unknown>;
  clearL0Cache(): void;
  clearL1Cache(): void;
  clearL2Cache(): Promise<void>;
  clearAllCaches(): Promise<void>;
  /** Whether the loader has an L1 / L2 caching store backing it. */
  readonly hasCachingStore: boolean;
}

/**
 * Returns the loader to operate on, or `null` if no loader is active
 * (e.g. before the first dataset is loaded).
 */
export type LoaderProvider = () => CacheCapableLoader | null;

/** Shape of the cache-helpers object exposed under `window.__luxarDebug.cache`. */
export interface DebugCacheHelpers {
  getStats: () => CacheStatsSnapshot | { error: string };
  listDatasets: () => Promise<unknown | { error: string }>;
  clearL0: () => void;
  clearL1: () => void;
  clearL2: () => Promise<void>;
  clearAll: () => Promise<void>;
}

/**
 * Build the cache-helpers bag using a loader-getter port. The
 * port-based shape lets tests inject a stub loader without needing
 * the SceneLoaderManager singleton.
 *
 * Notes:
 *   - `getStats` / `listDatasets` return a `{ error }` envelope when
 *     no loader is available, mirroring the pre-existing public
 *     contract of `__luxarDebug.cache.getStats()`.
 *   - The `clear*` helpers log a warning and silently no-op rather
 *     than returning an error envelope — this matches the original
 *     in-app behavior.
 */
export function buildDebugCacheHelpers(getLoader: LoaderProvider): DebugCacheHelpers {
  return {
    getStats: () => {
      const loader = getLoader();
      if (!loader) return { error: 'No active loader found' };
      return loader.getCacheStats();
    },

    listDatasets: async () => {
      const loader = getLoader();
      if (!loader || !loader.hasCachingStore) {
        return { error: 'No active cache found' };
      }
      return loader.listCachedDatasets();
    },

    clearL0: () => {
      const loader = getLoader();
      if (!loader) {
        log.warning(Modules.CACHE, 'No L0 cache found');
        return;
      }
      loader.clearL0Cache();
      log.info(Modules.CACHE, 'L0 cache cleared');
    },

    clearL1: () => {
      const loader = getLoader();
      if (!loader || !loader.hasCachingStore) {
        log.warning(Modules.CACHE, 'No active cache found');
        return;
      }
      loader.clearL1Cache();
      log.info(Modules.CACHE, 'L1 cache cleared');
    },

    clearL2: async () => {
      const loader = getLoader();
      if (!loader || !loader.hasCachingStore) {
        log.warning(Modules.CACHE, 'No active cache found');
        return;
      }
      await loader.clearL2Cache();
      log.info(Modules.CACHE, 'L2 cache cleared');
    },

    clearAll: async () => {
      const loader = getLoader();
      if (!loader) {
        log.warning(Modules.CACHE, 'No active loader found');
        return;
      }
      await loader.clearAllCaches();
      log.info(Modules.CACHE, 'All caches cleared (L0, L1, L2)');
    },
  };
}
