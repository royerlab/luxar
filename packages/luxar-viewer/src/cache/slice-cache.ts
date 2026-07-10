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
 * PRE- vs POST-projection payloads (a deliberate, inherent asymmetry):
 * Lines/GSplats cache PRE-projection decoded data — their projection is a
 * downstream worker step owned by the scene-loader's data processors, and a
 * same-reference revisit already skips it via the handlers'
 * `isAlreadyCommitted` fast path, so caching its output here would only
 * duplicate memory and couple this cache to the processors. Points caches
 * POST-projection data because its projection is folded into the loader
 * itself and consumes only key fields + node-static context — so a hit
 * safely skips the WASM projection too.
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
  /**
   * Misses on keys that were previously stored and then evicted (eviction-
   * induced misses, vs. cold misses). A high thrashMisses/misses ratio is the
   * signature of a working set larger than the budget — e.g. cyclic playback
   * thrash. Optional so stat mirrors that predate it keep compiling.
   */
  thrashMisses?: number;
  /**
   * Resolved byte budget (heap-aware; see `heap-budget.ts`). Surfaced so the
   * monitor / debug API can show the live budget — critical when it varies by
   * device heap rather than being the fixed config value. Mirrors
   * {@link DecompressedChunkCacheStats.maxSize}.
   */
  maxSize?: number;
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
  /**
   * Default cache size derived from config.cache.sliceCacheMaxSizeMB.
   * A method, not a static initializer: reading config at module-load time
   * couples every transitive importer (e.g. the shared facade helpers) to a
   * fully-populated config mock in tests.
   */
  private static defaultMaxSize(): number {
    return config.cache.sliceCacheMaxSizeMB * 1024 * 1024;
  }

  /** Tombstone cap — bounded so long sessions can't grow it unboundedly. */
  private static readonly MAX_TOMBSTONES = 4096;

  private cache: LRUCache<SliceCacheEntry>;
  private debug: boolean;
  private readonly maxSize: number;

  // Evicted-key tombstones (bounded FIFO): lets getStats() distinguish an
  // eviction-induced miss ("was cached, got evicted, asked for again" —
  // thrash) from a cold miss. A JS Set iterates in insertion order, so FIFO
  // trimming is just "delete the first key".
  private readonly tombstones = new Set<string>();
  private thrashMisses = 0;

  // Keys already warned about as oversized (a full ladder exceeding the whole
  // budget → only a coarse prefix is cached; see `storeLadder`). Instance-
  // scoped and cleared on clear() so a dataset switch can warn afresh, and
  // bounded FIFO so a long session can't grow it unboundedly. Same discipline
  // as `tombstones`.
  private readonly oversizedWarned = new Set<string>();

  constructor(options?: SliceCacheOptions) {
    const maxSize = options?.maxSize ?? SliceCache.defaultMaxSize();
    this.maxSize = maxSize;
    this.debug = options?.debug ?? false;

    // Byte-budget LRU keyed on the caller-measured retained size.
    // The onEvict hook records tombstones for thrash detection.
    this.cache = new LRUCache<SliceCacheEntry>(
      maxSize,
      (entry) => entry.bytes,
      (key) => this.recordTombstone(key)
    );

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
    if (entry !== undefined) {
      // A hit means the consumer (e.g. the foreground tick restoring a
      // prefetched slice) has taken the entry — release any prefetch pin so it
      // rejoins normal eviction. Harmless when the key was never pinned.
      this.cache.unpin(key);
    } else if (this.tombstones.has(key)) {
      // Was cached earlier, got evicted, asked for again: thrash, not cold.
      this.thrashMisses++;
    }
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
   * Cache a per-slice entry (may evict entries to stay within budget).
   *
   * @param key - Cache key (use {@link SliceCache.makeKey}).
   * @param entry - Cloned payload + its retained byte size.
   * @param opts.scan - The caller KNOWS it is storing inside a sequential
   *   scan (dimension playback: the loaders pass `frameBudgetMs !== null`).
   *   Evicts from the MRU end instead of LRU — scan-resistant eviction that
   *   keeps the loop-head prefix resident across cyclic loops (see
   *   `LRUCache.set`). Known trade-off: under `bounce` loop mode near a
   *   turnaround this is locally worse than LRU (the loaders only see the
   *   budget directive, not the loop mode) — accepted; a loop-mode-aware
   *   per-pass hint is a possible follow-up.
   * @param opts.pin - Protect this entry from eviction until it is next read
   *   (unpinned on the first {@link get} hit). The SlicePrefetcher sets it so a
   *   projected next-frame slice — which lands as the MRU entry and would be
   *   the FIRST victim of a subsequent scan store under budget pressure —
   *   survives until the foreground tick restores it. Best-effort: if the whole
   *   working set is pinned and over budget, pinned entries are still evicted.
   */
  set(key: string, entry: SliceCacheEntry, opts?: { scan?: boolean; pin?: boolean }): void {
    this.cache.set(key, entry, { evictMostRecent: opts?.scan });
    if (opts?.pin) this.cache.pin(key);
    // Freshly cached: a later miss on this key is only thrash if it gets
    // evicted AGAIN (recordTombstone re-adds it then).
    this.tombstones.delete(key);
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

  /**
   * Warn-once gate for oversized ladders: returns true the FIRST time `key` is
   * reported oversized (a full ladder exceeding the whole budget, so only a
   * coarse prefix is cached — see `storeLadder`), false thereafter. Bounded
   * FIFO + cleared on {@link clear}, so warnings can't leak across a long
   * session or a dataset switch (mirrors the tombstone discipline).
   */
  markOversizedWarned(key: string): boolean {
    if (this.oversizedWarned.has(key)) return false;
    this.oversizedWarned.add(key);
    if (this.oversizedWarned.size > SliceCache.MAX_TOMBSTONES) {
      const oldest = this.oversizedWarned.values().next().value;
      if (oldest !== undefined) this.oversizedWarned.delete(oldest);
    }
    return true;
  }

  /** Clear all cached slices (invoked on content-hash / dataset invalidation). */
  clear(): void {
    this.cache.clear(); // fires onEvict per entry — reset tombstones AFTER
    this.tombstones.clear();
    this.oversizedWarned.clear();
    this.thrashMisses = 0;
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
      thrashMisses: this.thrashMisses,
      maxSize: this.maxSize,
    };
  }

  /** Record an evicted key as a tombstone (bounded FIFO). */
  private recordTombstone(key: string): void {
    this.tombstones.delete(key); // re-insert at FIFO tail if already present
    this.tombstones.add(key);
    if (this.tombstones.size > SliceCache.MAX_TOMBSTONES) {
      const oldest = this.tombstones.values().next().value;
      if (oldest !== undefined) this.tombstones.delete(oldest);
    }
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
