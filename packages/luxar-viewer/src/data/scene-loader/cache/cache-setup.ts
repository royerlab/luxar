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

import * as zarr from '../../zarr';
import { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';
import { ChunkPrefetcher } from '../../../cache/chunk-prefetcher';
import { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import { SliceCache } from '../../../cache/slice-cache';
import {
  cachePoolOverrideBytes,
  computeCacheBudgets,
  computeOpfsWriteQueueBudgetBytes,
  deviceClassPoolBytes,
  type CacheBudgets,
} from '../../../cache/heap-budget';
import { ZipChunkSource } from '../../../cache/chunk-source/zip-chunk-source';
import { isZippedStoreUrl } from '../../zip/entries';
import { LuxarZipStore } from '../../zip/store';
import { config as appConfig } from '../../../config';
import { log, Modules } from '../../../utils/log';
import type { CacheTelemetryState } from '../../../types/data-monitor-types';

/** Subset of LoaderConfig the cache setup needs. Mirrors the fields
 *  the original inline code consulted. */
export interface CacheSetupFlags {
  noCache?: boolean;
  /** Disable ONLY the SliceCache (`?no-slice-cache`); L0/L1/L2 stay on. */
  noSliceCache?: boolean;
  /** Disable ONLY the L2 OPFS tier (`?no-opfs`); L0/L1/S-cache stay on. */
  noOpfs?: boolean;
  cacheDebug?: boolean;
  clearCache?: boolean;
  noPrefetch?: boolean;
  prefetchDebug?: boolean;
  /**
   * Explicit total cache pool (L0+L1+S-cache) in MB (`?cacheBudgetMB=` / native
   * launcher). Overrides heap detection — the path that gives the WKWebView app
   * / Safari (no `performance.memory`) a real budget.
   */
  cacheBudgetMB?: number | null;
}

/** Result of cache setup: the three layers and a ready-to-open store. */
export interface CacheSetupResult {
  l0Cache: DecompressedChunkCache | null;
  /** Shared SliceCache ("S-cache") for per-slice decoded-geometry reuse. */
  sliceCache: SliceCache | null;
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
  /**
   * The resolved per-tier budgets (and their source: heap / explicit /
   * device-class / fixed) — surfaced so the Settings popover can show the
   * user what their budget actually resolved to.
   */
  budgets: CacheBudgets;
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
  const noSliceCache = flags.noSliceCache ?? false;
  const noOpfs = flags.noOpfs ?? false;
  const cacheDebug = flags.cacheDebug ?? false;
  const clearCache = flags.clearCache ?? false;
  const noPrefetch = flags.noPrefetch ?? false;
  const prefetchDebug = flags.prefetchDebug ?? false;

  // Two-sided heap-aware budgets for the in-memory tiers (L0/L1/S-cache):
  // scale up on a large heap so a fits-in-RAM timelapse stays resident, and
  // down on a small heap to avoid an OOM from the previously-fixed 428 MB.
  // An explicit `cacheBudgetMB` (URL / native launcher) takes precedence — the
  // path that gives WKWebView/Safari (no `performance.memory`) a real budget
  // instead of the fixed fallback. L2 (OPFS/disk) is unaffected.
  const poolOverrideBytes = cachePoolOverrideBytes(flags.cacheBudgetMB);
  // Which tiers are actually active this session (single source of truth, reused
  // both to size the budgets and to gate construction below) — so a disabled
  // tier doesn't reserve pool it can't use.
  const sliceEnabled = appConfig.cache.sliceCacheEnabled && !noCache && !noSliceCache;
  const l0Enabled = appConfig.cache.l0Enabled && !noCache;
  // Zipped stores are cached like any other now: MultiLevelCachingStore takes a
  // ChunkSource rather than a base URL, so an archive member is reachable
  // without a per-chunk URL. Caching earns MORE here than for a directory
  // store — an archive costs ~2 requests per member (unzipit reads each local
  // file header in its own round trip) and a repeat read cannot fall back to
  // the browser's HTTP cache, because every member read is a `Range` request
  // against one URL.
  const zipped = isZippedStoreUrl(url);
  const l1Enabled = appConfig.cache.enabled && !noCache;
  // Device-class fallback pool (mobile/laptop/desktop) for WebKit without an
  // override — where the heap can't be measured. undefined in non-browser envs.
  const fallbackPoolBytes = deviceClassPoolBytes();
  const budgets = computeCacheBudgets(undefined, poolOverrideBytes, fallbackPoolBytes, {
    l0: l0Enabled,
    l1: l1Enabled,
    slice: sliceEnabled,
  });
  const toMB = (bytes: number) => (bytes / 1024 / 1024).toFixed(0);

  let l0Cache: DecompressedChunkCache | null = null;

  // SliceCache ("S-cache"): shared per-slice decoded-geometry cache. Gated by
  // its own config flag + `?no-slice-cache`, and also off when `?no-cache`
  // disables all tiers. It is cleared on content-hash invalidation alongside L0
  // (see the onInvalidate registration below).
  let sliceCache: SliceCache | null = null;
  if (sliceEnabled) {
    // No `?clear-cache` handling here (unlike L0/L1/L2 below): the SliceCache
    // is in-memory only and constructed fresh for every loadScene, so there is
    // never a prior session's state to clear.
    sliceCache = new SliceCache({
      maxSize: budgets.sliceBytes,
      debug: cacheDebug || appConfig.cache.debug,
    });
    log.info(
      Modules.SCENE_LOADER,
      `SliceCache (S-cache) enabled (max size: ${toMB(budgets.sliceBytes)}MB, ${budgets.source})`
    );
  } else if (noSliceCache) {
    log.info(Modules.SCENE_LOADER, 'SliceCache disabled via ?no-slice-cache URL parameter');
  }

  if (l0Enabled) {
    l0Cache = new DecompressedChunkCache({
      maxSize: budgets.l0Bytes,
      debug: cacheDebug || appConfig.cache.debug,
    });

    if (clearCache) {
      l0Cache.clear();
      log.info(Modules.SCENE_LOADER, 'L0 cache cleared via ?clear-cache URL parameter');
    }

    log.info(
      Modules.SCENE_LOADER,
      `L0 decompressed chunk cache enabled (max size: ${toMB(budgets.l0Bytes)}MB)`
    );
  } else if (noCache) {
    log.info(Modules.SCENE_LOADER, 'L0 cache disabled via ?no-cache URL parameter');
  }

  let rawStore: zarr.AsyncReadable;
  let cachingStore: MultiLevelCachingStore | null = null;

  if (l1Enabled) {
    cachingStore = new MultiLevelCachingStore(
      zipped ? new ZipChunkSource(url, new LuxarZipStore(url)) : url,
      {
        l1MaxSize: budgets.l1Bytes,
        l2MaxSize: appConfig.cache.l2MaxSizeMB * 1024 * 1024,
        // The L2 write queue's retained-byte ceiling comes from the same
        // memory model, so the `?cacheBudgetMB=` / native-launcher override and
        // the device-class fallback feed it exactly like the tiers above.
        opfsWriteQueueMaxBytes: computeOpfsWriteQueueBudgetBytes(
          undefined,
          poolOverrideBytes,
          fallbackPoolBytes
        ),
        debug: cacheDebug || appConfig.cache.debug,
        noCache,
        noOpfs,
        clearCache,
      }
    );
    await cachingStore.init();
    if (noOpfs) {
      log.info(Modules.SCENE_LOADER, 'L2 OPFS tier disabled via ?no-opfs URL parameter');
    }

    const prefetcher = new ChunkPrefetcher(cachingStore, {
      maxConcurrent: appConfig.dataLoading.network.maxConcurrent,
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

    // SliceCache holds decoded geometry derived from the dataset's content, so a
    // content-hash bump must evict it too — otherwise a revisit would serve
    // geometry from the stale dataset (cf. the past stale-cache black screen).
    // NOTE: this registration lives inside the `cache.enabled` block (same as
    // L0's), so with L1/L2 disabled there is no content-hash listener. That's an
    // acceptably narrow gap: in-session dataset switches clear the SliceCache via
    // dispose(), and with no persistent chunk cache there's nothing else to be
    // stale against.
    if (sliceCache) {
      const sc = sliceCache;
      cachingStore.onInvalidate(() => {
        sc.clear();
        log.info(Modules.SCENE_LOADER, 'SliceCache cleared due to L1/L2 invalidation');
      });
    }

    rawStore = cachingStore;
  } else {
    rawStore = zarr.createStoreForUrl(url);
  }

  // Resolve the telemetry state. URL flag wins (most user-visible);
  // app-config disable comes next; otherwise enabled if any tier is
  // active. (L0-only counts as `enabled` — the cache tab keys off the
  // top-level state and renders L0 stats independently.)
  let telemetryState: CacheTelemetryState;
  if (noCache) {
    telemetryState = { kind: 'disabled-no-cache' };
  } else if (sliceCache === null && !appConfig.cache.enabled && !appConfig.cache.l0Enabled) {
    telemetryState = { kind: 'disabled-config' };
  } else {
    // At least one tier is active (S-cache counts: an S-cache-only
    // configuration still serves slice revisits and reports live stats).
    telemetryState = { kind: 'enabled' };
  }

  return { l0Cache, sliceCache, cachingStore, rawStore, telemetryState, budgets };
}
