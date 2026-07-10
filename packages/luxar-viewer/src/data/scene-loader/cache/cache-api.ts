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
 * tested in isolation and gives a single home for cache-API additions
 * (eviction tuning, hit-rate telemetry, etc.).
 *
 * Each helper gracefully no-ops when the corresponding cache layer is
 * `null` — matching the behavior the SceneLoader had inline, where
 * caches are absent under `?no-cache` or when the global config disables
 * them.
 *
 * @module data/scene-loader/cache/cache-api
 */

import type { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';
import type { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import type { SliceCache } from '../../../cache/slice-cache';

/**
 * Snapshot of all three cache levels exposed by
 * `__luxarDebug.cache.getStats()`. Levels that are absent return
 * `null` for that field; the caller can render them as "disabled".
 *
 * `network`, `demand`, `prefetch`, and `health` are optional so
 * callers keying off `{l0, l1, l2}` keep working without changes.
 */
export interface CacheStatsSnapshot {
  l0: ReturnType<DecompressedChunkCache['getStats']> | null;
  /** S-cache (decoded per-slice geometry) — null when the SliceCache is off. */
  slice: ReturnType<SliceCache['getStats']> | null;
  l1: ReturnType<MultiLevelCachingStore['getStats']>['l1'] | null;
  l2: ReturnType<MultiLevelCachingStore['getStats']>['l2'] | null;
  network?: ReturnType<MultiLevelCachingStore['getStats']>['network'] | null;
  demand?: ReturnType<MultiLevelCachingStore['getStats']>['demand'] | null;
  prefetch?: { queued: number; inFlight: number; enabled: boolean } | null;
  health?: ReturnType<MultiLevelCachingStore['getStats']>['health'] | null;
  /**
   * S4: number of times `?clear-cache` triggered a clearAll on init
   * for the active store. Surfaced so E2E tests can assert that a
   * clear actually ran rather than only checking that stats survived.
   */
  clearOnInitCount?: number;
}

/**
 * Build the cache snapshot for the debug interface and the data-monitor
 * "Cache" tab. Includes per-tier stats plus network, demand,
 * prefetch, and health diagnostics so a single `getStats()` call
 * answers every diagnostic question without a follow-up roundtrip.
 */
export function getCacheStats(
  l0Cache: DecompressedChunkCache | null,
  cachingStore: MultiLevelCachingStore | null,
  sliceCache: SliceCache | null = null
): CacheStatsSnapshot {
  const stats = cachingStore?.getStats();
  // Defensive: cache-api is sometimes called with a stub cachingStore
  // in tests that doesn't implement getPrefetcher. Treat missing method
  // as "no prefetcher attached".
  const prefetcher =
    cachingStore && typeof cachingStore.getPrefetcher === 'function'
      ? cachingStore.getPrefetcher()
      : null;
  return {
    l0: l0Cache?.getStats() ?? null,
    slice: sliceCache?.getStats() ?? null,
    l1: stats?.l1 ?? null,
    l2: stats?.l2 ?? null,
    network: stats?.network ?? null,
    demand: stats?.demand ?? null,
    prefetch: prefetcher
      ? {
          queued: prefetcher.getStats().queued,
          inFlight: prefetcher.getStats().inFlight,
          enabled: prefetcher.getStats().enabled,
        }
      : null,
    health: stats?.health ?? null,
    clearOnInitCount: stats?.clearOnInitCount,
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
export async function clearL2Cache(cachingStore: MultiLevelCachingStore | null): Promise<void> {
  await cachingStore?.clearL2();
}

/**
 * Clear every cache tier in one call. L1+L2 are cleared together via the
 * caching store's `clearAll`; L0 and the decoded-slice S-cache are cleared
 * independently (the S-cache sits above L0 and holds decoded geometry, so a
 * "clear all" that skipped it would keep serving slice revisits from memory).
 */
export async function clearAllCaches(
  l0Cache: DecompressedChunkCache | null,
  cachingStore: MultiLevelCachingStore | null,
  sliceCache: SliceCache | null = null
): Promise<void> {
  l0Cache?.clear();
  sliceCache?.clear();
  if (cachingStore) {
    await cachingStore.clearAll();
  }
}
