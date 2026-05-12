/**
 * Cache initialization helpers for `SceneLoader.loadScene()`.
 *
 * Wires the L0/L1/L2 cache stack:
 *   - L0 (DecompressedChunkCache, in-memory decoded chunks)
 *   - L1 + L2 (MultiLevelCachingStore: in-memory + OPFS)
 *   - ChunkPrefetcher attached to L1/L2
 *   - L0 invalidation on L1/L2 clear
 *
 * Splitting this out keeps `loadScene()` readable as a high-level
 * "open cache → open zarr → enumerate → build scene" sequence.
 */

import * as zarr from '../zarr';
import { MultiLevelCachingStore, ChunkPrefetcher, DecompressedChunkCache } from '../../cache';
import { config as appConfig } from '../../config';
import { log, Modules } from '../../utils/log';
import type { CacheTelemetryState } from '../../types/data-monitor-types';

/** Subset of LoaderConfig the cache setup needs. Mirrors the fields
 *  the original inline code consulted. */
export interface CacheSetupFlags {
  noCache?: boolean;
  cacheDebug?: boolean;
  clearCache?: boolean;
  noPrefetch?: boolean;
  prefetchDebug?: boolean;
}

/** Result of cache setup: the three layers and a ready-to-open store. */
export interface CacheSetupResult {
  l0Cache: DecompressedChunkCache | null;
  cachingStore: MultiLevelCachingStore | null;
  rawStore: zarr.AsyncReadable;
  /**
   * Resolved cache telemetry state for the UI monitor. Reflects the
   * actual policy decision the cache stack made:
   *   - `disabled-no-cache`: URL `?no-cache` flag.
   *   - `disabled-config` : `appConfig.cache.enabled === false`.
   *   - `enabled`         : at least one tier (L0 or L1/L2) is active.
   * Pushed to the monitor via `setCacheTelemetryState()` in
   * `wireMonitorAfterLoad`.
   */
  telemetryState: CacheTelemetryState;
}

/**
 * Build the L0/L1/L2 cache stack and return the raw store the caller
 * should pass to the Luxar Zarr facade's `openStore()`.
 *
 * Behavior matches the previous inline block in `SceneLoader.loadScene()`:
 *   - L0 is enabled iff `appConfig.cache.l0Enabled && !flags.noCache`.
 *   - L1/L2 are enabled iff `appConfig.cache.enabled && !flags.noCache`.
 *   - When L1/L2 + L0 are both active, L0 is registered as an
 *     invalidation listener so a content-hash bump clears all three.
 *   - When caching is fully disabled, falls back to a vanilla fetch store
 *     created by the Luxar Zarr facade.
 */
export async function setupCaches(url: string, flags: CacheSetupFlags): Promise<CacheSetupResult> {
  const noCache = flags.noCache ?? false;
  const cacheDebug = flags.cacheDebug ?? false;
  const clearCache = flags.clearCache ?? false;
  const noPrefetch = flags.noPrefetch ?? false;
  const prefetchDebug = flags.prefetchDebug ?? false;

  let l0Cache: DecompressedChunkCache | null = null;

  if (appConfig.cache.l0Enabled && !noCache) {
    l0Cache = new DecompressedChunkCache({
      maxSize: appConfig.cache.l0MaxSizeMB * 1024 * 1024,
      debug: cacheDebug || appConfig.cache.debug,
    });

    if (clearCache) {
      l0Cache.clear();
      log.info(Modules.SCENE_LOADER, 'L0 cache cleared via ?clear-cache URL parameter');
    }

    log.info(
      Modules.SCENE_LOADER,
      `L0 decompressed chunk cache enabled (max size: ${appConfig.cache.l0MaxSizeMB}MB)`
    );
  } else if (noCache) {
    log.info(Modules.SCENE_LOADER, 'L0 cache disabled via ?no-cache URL parameter');
  }

  let rawStore: zarr.AsyncReadable;
  let cachingStore: MultiLevelCachingStore | null = null;

  if (appConfig.cache.enabled && !noCache) {
    cachingStore = new MultiLevelCachingStore(url, {
      l1MaxSize: appConfig.cache.l1MaxSizeMB * 1024 * 1024,
      l2MaxSize: appConfig.cache.l2MaxSizeMB * 1024 * 1024,
      debug: cacheDebug || appConfig.cache.debug,
      noCache,
      clearCache,
    });
    await cachingStore.init();

    const prefetcher = new ChunkPrefetcher(cachingStore, {
      maxConcurrent: 4,
      enabled: !noPrefetch,
      debug: prefetchDebug,
    });
    cachingStore.setPrefetcher(prefetcher);

    if (l0Cache) {
      const l0 = l0Cache;
      cachingStore.onInvalidate(() => {
        l0.clear();
        log.info(Modules.SCENE_LOADER, 'L0 cache cleared due to L1/L2 invalidation');
      });
    }

    rawStore = cachingStore;
  } else {
    rawStore = zarr.createFetchStore(url);
  }

  // Resolve the telemetry state. URL flag wins (most user-visible);
  // app-config disable comes next; otherwise enabled if any tier is
  // active. (L0-only counts as `enabled` — the cache tab keys off the
  // top-level state and renders L0 stats independently.)
  let telemetryState: CacheTelemetryState;
  if (noCache) {
    telemetryState = { kind: 'disabled-no-cache' };
  } else if (!appConfig.cache.enabled && !appConfig.cache.l0Enabled) {
    telemetryState = { kind: 'disabled-config' };
  } else {
    telemetryState = { kind: 'enabled' };
  }

  return { l0Cache, cachingStore, rawStore, telemetryState };
}
