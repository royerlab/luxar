/**
 * Cache surface helpers for the scene loader.
 *
 * The SceneLoader owns three cache layers:
 *   - **L0**: in-memory `DecompressedChunkCache` of decoded zarr chunks
 *     (skips the Blosc decompression on reuse).
 *   - **L1**: in-memory metadata/raw-chunk cache inside
 *     `MultiLevelCachingStore`.
 *   - **L2**: persistent OPFS cache inside the same store.
 *
 * These pure functions implement the typed get/clear/list API that the
 * SceneLoader exposes (`getCacheStats`, `listCachedDatasets`,
 * `clearL0Cache`, `clearL1Cache`, `clearL2Cache`, `clearAllCaches`).
 * Pulling them out of the SceneLoader class lets the cache surface be
 * tested in isolation and gives a single home for any future cache-API
 * additions (eviction tuning, hit-rate telemetry, etc.).
 *
 * Each helper gracefully no-ops when the corresponding cache layer is
 * `null` — matching the behavior the SceneLoader had inline, where
 * caches are absent under `?no-cache` or when the global config disables
 * them.
 *
 * @module data/scene-loader/cache-api
 */

import type { MultiLevelCachingStore, DecompressedChunkCache } from '../../cache';

/**
 * Snapshot of all three cache levels in the form historically exposed
 * by `__luxarDebug.cache.getStats()`. Levels that are absent return
 * `null` for that field; the caller can render them as "disabled".
 */
export interface CacheStatsSnapshot {
  l0: ReturnType<DecompressedChunkCache['getStats']> | null;
  l1: ReturnType<MultiLevelCachingStore['getStats']>['l1'] | null;
  l2: ReturnType<MultiLevelCachingStore['getStats']>['l2'] | null;
}

/**
 * Build the `{ l0, l1, l2 }` snapshot for the debug interface and the
 * data-monitor "Cache" tab.
 */
export function getCacheStats(
  l0Cache: DecompressedChunkCache | null,
  cachingStore: MultiLevelCachingStore | null
): CacheStatsSnapshot {
  const l1l2 = cachingStore?.getStats();
  return {
    l0: l0Cache?.getStats() ?? null,
    l1: l1l2?.l1 ?? null,
    l2: l1l2?.l2 ?? null,
  };
}

/**
 * List datasets currently held by the L1/L2 caching store. Returns an
 * empty array if no caching store is configured.
 */
export async function listCachedDatasets(
  cachingStore: MultiLevelCachingStore | null
): Promise<Awaited<ReturnType<MultiLevelCachingStore['listDatasets']>>> {
  if (!cachingStore) return [];
  return cachingStore.listDatasets();
}

/** Clear the in-memory L0 decompressed-chunk cache. No-op if absent. */
export function clearL0Cache(l0Cache: DecompressedChunkCache | null): void {
  l0Cache?.clear();
}

/** Clear the in-memory L1 metadata/chunk cache. No-op if absent. */
export function clearL1Cache(cachingStore: MultiLevelCachingStore | null): void {
  cachingStore?.clearL1();
}

/** Clear the persistent L2 OPFS cache. No-op if absent. */
export async function clearL2Cache(
  cachingStore: MultiLevelCachingStore | null
): Promise<void> {
  await cachingStore?.clearL2();
}

/**
 * Clear all three cache levels in one call. L1+L2 are cleared together
 * via the caching store's `clearAll`; L0 is cleared independently.
 */
export async function clearAllCaches(
  l0Cache: DecompressedChunkCache | null,
  cachingStore: MultiLevelCachingStore | null
): Promise<void> {
  l0Cache?.clear();
  if (cachingStore) {
    await cachingStore.clearAll();
  }
}
