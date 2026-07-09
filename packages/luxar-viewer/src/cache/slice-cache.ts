/**
 * SliceCache ("S-cache") — a per-(node, view) LRU of decoded per-slice geometry.
 *
 * Sits ABOVE the L0/L1/L2 chunk caches. Where those are keyed by chunk and skip
 * only decompression, the SliceCache is keyed by the *view* (a node path + the
 * query's slice/tolerance/displayDims signature) and caches the fully decoded
 * per-slice geometry a progressive loader assembled. That lets revisiting a
 * slice — e.g. scrubbing back to a timepoint in a 4D dataset — skip the whole
 * query + fetch + dequant + gather pipeline (~100 ms on a large timelapse) and
 * only re-run the cheap nD→3D projection downstream.
 *
 * The payload is OPAQUE to this cache: the owning loader stores a cloned snapshot
 * of its decoded per-LOD data and casts it back on retrieval. The clone is the
 * loader's responsibility because its decoded arrays are views into a reused
 * accumulator buffer (see the spatial-index loaders) — a cross-view cache must
 * not alias buffers the next load overwrites. The cache only tracks bytes for
 * its byte-budget LRU eviction, so the loader passes the retained byte count.
 *
 * Invalidation: content-hash change / dataset switch clears the whole cache
 * (wired next to L0 in cache-setup.ts). The view signature in the key covers
 * slice/tolerance/displayDims; nothing projection-affecting (truncate, colormap,
 * opacity) enters the key because projection re-runs on every hit.
 *
 * @module cache/slice-cache
 */

import { LRUCache } from './lru-cache';
import { log, Modules, LogEmoji } from '../utils/log';
import { config } from '../config';

/**
 * One cached per-slice entry.
 *
 * `payload` is opaque (the owning loader's cloned decoded-slice snapshot);
 * `bytes` is the retained size the loader measured, used for LRU accounting.
 */
export interface SliceCacheEntry {
  /** Opaque cloned per-slice payload; the owning loader casts it back. */
  payload: unknown;
  /** Total retained bytes (sum of the snapshot's typed-array byteLengths). */
  bytes: number;
}

/**
 * Cache statistics for the data-loading monitor. Mirrors
 * {@link DecompressedChunkCacheStats} so the aggregator/UI can treat both
 * tiers uniformly.
 */
export interface SliceCacheStats {
  /** Current cache size in bytes. */
  size: number;
  /** Number of cached slices. */
  count: number;
  /** Total cache hits. */
  hits: number;
  /** Total cache misses. */
  misses: number;
  /** Total evicted slices. */
  evictions: number;
  /** Hit rate (0–1). */
  hitRate: number;
}

/** Configuration options for the SliceCache. */
export interface SliceCacheOptions {
  /** Maximum cache size in bytes (default: config.cache.sliceCacheMaxSizeMB). */
  maxSize?: number;
  /** Enable debug logging (default: false). */
  debug?: boolean;
}

/**
 * SliceCache — byte-budget LRU of decoded per-slice geometry.
 *
 * Key format: `${nodePath}\u0000${viewSig}` (see {@link SliceCache.makeKey}),
 * where `viewSig` is a stable serialization of the query's displayDims /
 * slicePosition / tolerance / dimensions — the loader's `viewStatesEqual`
 * fields, ALL of them (a key narrower than that equality can restore a
 * snapshot the loader itself would have reloaded; see
 * `data/loaders/progressive/slice-cache-helper.ts::buildSliceViewSig`).
 */
export class SliceCache {
  /** Default cache size derived from config.cache.sliceCacheMaxSizeMB. */
  private static readonly DEFAULT_MAX_SIZE = config.cache.sliceCacheMaxSizeMB * 1024 * 1024;

  private cache: LRUCache<SliceCacheEntry>;
  private debug: boolean;
  private readonly maxSize: number;

  constructor(options?: SliceCacheOptions) {
    const maxSize = options?.maxSize ?? SliceCache.DEFAULT_MAX_SIZE;
    this.maxSize = maxSize;
    this.debug = options?.debug ?? false;

    // Byte-budget LRU keyed on the caller-measured retained size.
    this.cache = new LRUCache<SliceCacheEntry>(maxSize, (entry) => entry.bytes);

    if (this.debug) {
      log.custom(
        LogEmoji.CACHE,
        Modules.CACHE,
        `SliceCache initialized with max size: ${(maxSize / 1024 / 1024).toFixed(1)}MB`
      );
    }
  }

  /**
   * Get a cached per-slice entry (LRU-promotes it).
   *
   * @param key - Cache key (use {@link SliceCache.makeKey}).
   * @returns Cached entry or undefined on miss.
   */
  get(key: string): SliceCacheEntry | undefined {
    const entry = this.cache.get(key);
    if (this.debug) {
      log.custom(
        LogEmoji.CACHE,
        Modules.CACHE,
        entry ? `SliceCache HIT: ${key} (${entry.bytes} bytes)` : `SliceCache MISS: ${key}`
      );
    }
    return entry;
  }

  /**
   * Cache a per-slice entry (may evict LRU entries to stay within budget).
   *
   * @param key - Cache key (use {@link SliceCache.makeKey}).
   * @param entry - Cloned payload + its retained byte size.
   */
  set(key: string, entry: SliceCacheEntry): void {
    this.cache.set(key, entry);
    if (this.debug) {
      log.custom(
        LogEmoji.CACHE,
        Modules.CACHE,
        `SliceCache SET: ${key} (${entry.bytes} bytes, ` +
          `total: ${(this.cache.size / 1024 / 1024).toFixed(1)}MB)`
      );
    }
  }

  /** Check whether a slice is cached (without LRU promotion). */
  has(key: string): boolean {
    return this.cache.has(key);
  }

  /**
   * Read an entry WITHOUT LRU promotion and WITHOUT counting a hit/miss.
   * Used by `storeLadder`'s upgrade-if-longer check so bookkeeping reads
   * don't perturb the hit-rate statistic the monitor reports.
   */
  peek(key: string): SliceCacheEntry | undefined {
    return this.cache.peek(key);
  }

  /**
   * Whether an entry of `bytes` could ever be stored (i.e. it does not exceed
   * the whole budget). The LRU silently rejects oversized entries, so callers
   * check this BEFORE doing the (potentially large) clone to avoid a wasted
   * copy that would just be dropped — see `storeLadder`.
   */
  willFit(bytes: number): boolean {
    return bytes <= this.maxSize;
  }

  /** Clear all cached slices (invoked on content-hash / dataset invalidation). */
  clear(): void {
    this.cache.clear();
    if (this.debug) {
      log.custom(LogEmoji.CACHE, Modules.CACHE, 'SliceCache cleared');
    }
  }

  /** Dispose the cache, releasing all cached slices. */
  dispose(): void {
    this.clear();
  }

  /** Cache statistics for the monitor. */
  getStats(): SliceCacheStats {
    const hits = this.cache.hitCount;
    const misses = this.cache.missCount;
    const total = hits + misses;
    return {
      size: this.cache.size,
      count: this.cache.count,
      hits,
      misses,
      evictions: this.cache.evictionCount,
      hitRate: total > 0 ? hits / total : 0,
    };
  }

  /**
   * Build a cache key from a node path and a view signature.
   *
   * The NUL separator can't appear in a node path or the JSON-ish view
   * signature, so the two are always unambiguously recoverable.
   *
   * @param nodePath - Scene-graph path of the geometry node (per-node namespace).
   * @param viewSig - Stable serialization of the query view (slice/tolerance/displayDims).
   */
  static makeKey(nodePath: string, viewSig: string): string {
    return `${nodePath}\u0000${viewSig}`;
  }
}
