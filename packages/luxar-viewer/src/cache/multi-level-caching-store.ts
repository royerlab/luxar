import type { AsyncReadable } from '../data/zarr';
import { SegmentedLRUCache } from './multi-level-caching-store/segmented-lru-cache';
import { OPFSStore, type CachedDatasetSummary } from './multi-level-caching-store/opfs-store';
import { BandwidthWindow } from './multi-level-caching-store/bandwidth-window';
import {
  buildUrl,
  fetchWithRetry,
  hashUrl,
  mergeAbortSignals,
} from './multi-level-caching-store/fetch-retry';
import {
  ValidationQueue,
  getRemoteContentHash,
} from './multi-level-caching-store/validation-queue';
import type { ChunkPrefetcher } from './chunk-prefetcher';
import { OpfsWriteQueue } from './multi-level-caching-store/opfs-write-queue';
import { log, Modules } from '../utils/log';
import { config } from '../config';
import { type Result, ok, err, isErr } from '../utils/result';
import type { MultiLevelCacheStats } from './types';

/**
 * Structured failure modes from {@link MultiLevelCachingStore.getResult}.
 * Distinguishing them lets callers (prefetcher, retry policies, debug
 * overlays) act on the underlying cause instead of treating every
 * absence the same way.
 */
export type CacheError =
  | { readonly kind: 'Missing' }
  | { readonly kind: 'NetworkError'; readonly cause: Error }
  | { readonly kind: 'Aborted' };

export interface MultiLevelCachingStoreOptions {
  /** L1 memory cache size in bytes (default: 100MB) */
  l1MaxSize?: number;
  /** L2 OPFS cache size in bytes (default: 2GB) */
  l2MaxSize?: number;
  /** Enable debug logging (default: false) */
  debug?: boolean;
  /** Disable both cache tiers (e.g. driven by `?no-cache`). Default false. */
  noCache?: boolean;
  /** Clear caches on init (e.g. driven by `?clear-cache`). Default false. */
  clearCache?: boolean;
  /**
   * Background L2 write-queue concurrency cap. Defaults to
   * `config.cache.opfsWriteConcurrency`. Exposed mainly so tests can inject a
   * tiny cap.
   */
  opfsWriteConcurrency?: number;
  /**
   * Background L2 write-queue max pending depth. Defaults to
   * `config.cache.opfsWriteQueueMax`.
   */
  opfsWriteQueueMax?: number;
}

/**
 * Multi-level caching store that implements zarrita's AsyncReadable
 * interface. Orchestrates three tiers — L0 (decompressed in-memory chunk
 * cache, owned by the zarrita layer), L1 (memory, in-process), L2
 * (OPFS, cross-tab) — plus an HTTP fallback for zarr chunks.
 */
export class MultiLevelCachingStore implements AsyncReadable {
  private l1Cache: SegmentedLRUCache;
  private l2Store: OPFSStore | null = null;
  private prefetcher: ChunkPrefetcher | null = null;
  private baseUrl: string;
  private l2MaxSize: number;
  private enabled: boolean;
  private debug: boolean;
  private shouldClearOnInit: boolean;
  // S4: increments each time `?clear-cache` triggers a clearAll on
  // init. Surfaced through getStats() so the E2E suite can prove the
  // clear actually ran rather than only checking that stats survived.
  private clearOnInitCount = 0;

  private static readonly DEFAULT_L1_SIZE = config.cache.l1MaxSizeMB * 1024 * 1024;
  private static readonly DEFAULT_L2_SIZE = config.cache.l2MaxSizeMB * 1024 * 1024;
  // Captured during init() so dispose() can find this instance's queue entry.
  private datasetId?: string;

  // Lifecycle: set by dispose(). Synchronously short-circuits getResult and
  // gets propagated to fetchWithRetry via dataAbort below so any in-flight
  // network/prefetch fetch unwinds rather than continuing to write into a
  // disposed L2 store.
  private disposed = false;

  // Aborted by dispose(). Merged with each caller's optional signal in
  // getResult so a dataset switch cancels every in-flight data/prefetch
  // fetch this store kicked off, not only the validation request.
  private readonly dataAbort = new AbortController();

  // Background L2 (OPFS) write queue — moves the durable write off the fetch
  // critical path (see opfs-write-queue.ts). Assigned in the constructor.
  private readonly l2WriteQueue: OpfsWriteQueue;

  // MLC-level epoch, bumped on every cache clear/invalidation. A queued L2
  // write captures the epoch at enqueue and self-drops at drain if it changed
  // — the fire-and-forget analogue of OPFSStore's `generation` guard. Needed
  // because `dataAbort` is aborted only on dispose, NOT on a content-hash/TTL
  // clear, so it can't distinguish a clear from normal running state.
  private l2Epoch = 0;

  // Same-key in-flight coalescing: concurrent getResult callers for the
  // same key share a single L2/network fetch. Each caller still does
  // its own L1 check (synchronous; the L1 fast path stays direct) and
  // tracks its own demand counter — only the underlying network
  // request and L1/L2 writes are deduplicated. Cleared on settle.
  //
  // CRIT-5 fix: each entry carries an AbortController so a
  // content-hash-mismatch invalidation can cancel in-flight fetches
  // before they write stale data back into L1/L2 after a clear.
  private pendingGets = new Map<
    string,
    {
      promise: Promise<{
        result: Result<Uint8Array, CacheError>;
        source: 'l2' | 'network' | 'missing';
      }>;
      controller: AbortController;
    }
  >();

  // Invalidation callbacks (e.g., L0 DecompressedChunkCache clearing on L1/L2 invalidation)
  private invalidationCallbacks: (() => void)[] = [];

  // Network I/O tracking
  private networkBytesTransferred = 0;
  private networkRequestCount = 0;

  // Cumulative bytes delivered to demand callers across ALL tiers
  // (L1 + L2 + network). Drives the monitor's "data loaded" figure so
  // it reflects real I/O work even when every chunk is cache-served and
  // `networkBytesTransferred` legitimately stays 0 (e.g. warm reload of
  // an OPFS-cached dataset). Prefetch traffic is excluded — those bytes
  // are already counted under the network totals and are counted here
  // when a demand read later pulls them from cache.
  private totalBytesServed = 0;
  private totalRequestsServed = 0;

  // Per-tier demand-hit counters. Each user-demand call to
  // getResult() increments exactly one of l1HitCount / l2HitCount /
  // demandNetworkRequestCount. Prefetch-originated calls
  // (suppressPrefetch: true) are excluded so the hit-rate reflects
  // user demand only — a prefetch that hits L2 must not inflate
  // the apparent hit-rate. The aggregate `networkRequestCount` and
  // `networkBytesTransferred` above stay unconditional and count
  // ALL traffic.
  private l1HitCount = 0;
  private l2HitCount = 0;
  private demandNetworkRequestCount = 0;

  // Sliding-window bandwidth tracking (last ~10 seconds).
  private static readonly BANDWIDTH_WINDOW_MS = 10_000;
  private bandwidth = new BandwidthWindow(MultiLevelCachingStore.BANDWIDTH_WINDOW_MS);

  constructor(baseUrl: string, options?: MultiLevelCachingStoreOptions) {
    this.baseUrl = baseUrl;

    this.enabled = !(options?.noCache ?? false);
    this.debug = options?.debug ?? false;
    this.shouldClearOnInit = options?.clearCache ?? false;

    // Initialize L1 (always, even if disabled)
    const l1Size = options?.l1MaxSize ?? MultiLevelCachingStore.DEFAULT_L1_SIZE;
    this.l1Cache = new SegmentedLRUCache(l1Size);

    // Save L2 max size for later initialization
    this.l2MaxSize = options?.l2MaxSize ?? MultiLevelCachingStore.DEFAULT_L2_SIZE;

    // Background L2 write queue (defaults from config; options override for tests).
    this.l2WriteQueue = new OpfsWriteQueue({
      concurrency: options?.opfsWriteConcurrency ?? config.cache.opfsWriteConcurrency,
      maxDepth: options?.opfsWriteQueueMax ?? config.cache.opfsWriteQueueMax,
    });
  }

  /**
   * Register a callback to be invoked when caches are invalidated (e.g., clearAll, validateCache).
   * Used by L0 DecompressedChunkCache to clear itself when L1/L2 are invalidated.
   */
  onInvalidate(callback: () => void): void {
    this.invalidationCallbacks.push(callback);
  }

  /**
   * Attach a prefetcher to receive access notifications.
   */
  setPrefetcher(prefetcher: ChunkPrefetcher | null): void {
    this.prefetcher = prefetcher;
  }

  /**
   * Get the attached prefetcher (if any).
   */
  getPrefetcher(): ChunkPrefetcher | null {
    return this.prefetcher;
  }

  /**
   * Initialize the two-level cache and validate against remote dataset.
   *
   * Performs complete cache setup including OPFS initialization and content
   * hash validation. This MUST be called before any get() operations.
   *
   * Initialization steps:
   * 1. Generate dataset ID from URL hash (for OPFS directory isolation)
   * 2. Initialize L2 OPFS store (creates directory structure)
   * 3. Clear cache if ?clear-cache URL parameter present
   * 4. Validate cache by comparing content_hash with remote .zattrs
   *
   * Cache validation is CRITICAL: fetches .zattrs directly from HTTP
   * (bypassing cache) to detect dataset changes. If content_hash differs,
   * clears L2 completely and re-initializes. This prevents stale data.
   *
   * @returns Promise that resolves when cache is fully initialized and validated.
   *          If caching is disabled (?no-cache), resolves immediately without setup.
   *
   * @throws {Error} If OPFS is not supported (Safari < 15.2, Firefox < 111)
   * @throws {Error} If HTTP fetch of .zattrs fails (network error, 404)
   * @throws {QuotaExceededError} If OPFS storage quota exceeded
   *
   * @example
   * ```typescript
   * const store = new MultiLevelCachingStore(url, {
   *   l1MaxSize: 100 * 1024 * 1024,  // 100MB
   *   l2MaxSize: 2 * 1024 * 1024 * 1024  // 2GB
   * });
   *
   * await store.init();
   * console.log('Cache ready');
   * // Now safe to call get()
   * ```
   *
   * @example
   * ```typescript
   * // Handle initialization errors gracefully
   * try {
   *   await store.init();
   * } catch (error) {
   *   console.error('Cache init failed:', error);
   *   // Fall back to direct HTTP (no caching) by reloading with ?no-cache
   *   // (caching is controlled via URL parameters, not constructor options)
   * }
   * ```
   *
   * @see {@link validateCache} for cache validation algorithm
   * @see README.md for OPFS architecture
   * @remarks Performance: First init: ~100ms (OPFS setup + validation), Subsequent: ~10ms (validation only)
   */
  async init(): Promise<void> {
    if (!this.enabled) {
      this.log('Caching disabled');
      return;
    }

    // Generate dataset ID from URL and create L2 store
    const datasetId = await hashUrl(this.baseUrl);
    this.datasetId = datasetId;
    this.l2Store = new OPFSStore(datasetId, this.baseUrl, this.l2MaxSize);

    await this.l2Store.init();

    // Clear cache if requested
    if (this.shouldClearOnInit) {
      this.log('Clearing cache due to ?clear-cache parameter');
      await this.clearAll();
      // S4: observable signal for the E2E suite — proves clearAll
      // actually ran in response to `?clear-cache` on init.
      this.clearOnInitCount++;
    }

    // Validate cache using content hash. Validation is serialized per dataset
    // ID so rapid same-URL dataset switches cannot let an older validation
    // finish after a newer one and restore stale content_hash metadata.
    await this.validateCache(datasetId);
  }

  /**
   * Get a Zarr chunk with three-level cascade: L1 memory → L2 OPFS → L3 HTTP.
   *
   * Implements zarrita's AsyncReadable interface for seamless Zarr integration.
   * This is the primary data access method called by zarrita for ALL chunk reads.
   *
   * Cache cascade behavior:
   * 1. **L1 Memory Cache** (~1μs): Check in-memory LRU cache first
   * 2. **L2 OPFS Cache** (~1ms): Check persistent Origin Private File System if L1 miss
   * 3. **L3 HTTP Fetch** (~100ms): Fetch from remote server if both caches miss
   *
   * On successful fetch:
   * - Populates L1 cache immediately
   * - Populates L2 cache asynchronously (fire-and-forget)
   * - Triggers prefetcher to load adjacent chunks
   * - Tracks network I/O statistics
   *
   * Performance characteristics:
   * - L1 hit: ~1μs (hash table lookup)
   * - L2 hit: ~1ms (OPFS file read) + promotes to L1
   * - L3 fetch: ~100ms (network) + populates L1 and L2
   *
   * @param key - Zarr chunk key relative to store root.
   *              Examples:
   *              - Array chunk: 'positions/0.1.2'
   *              - Metadata: '.zarray', '.zmetadata', '.zattrs'
   *              - Nested: 'group1/subgroup/array/0.0'
   *
   * @param _options - Reserved for zarrita compatibility (currently unused).
   *                   Zarrita may pass options in future versions.
   *
   * @returns Promise resolving to chunk data as Uint8Array, or undefined if:
   *          - Chunk doesn't exist (404 Not Found)
   *          - Network error occurs
   *          - Fetch fails for any reason
   *          undefined is NOT an error - zarrita handles it gracefully
   *
   * @throws Never throws - all errors caught and returned as undefined.
   *         Errors are logged to console.warn for debugging.
   *
   * @example
   * ```typescript
   * // Get a chunk (called by zarrita internally)
   * const chunk = await store.get('positions/0.1.2');
   * if (chunk) {
   *   console.log(`Loaded ${chunk.byteLength} bytes`);
   *   // Process chunk data...
   * } else {
   *   console.log('Chunk not found or network error');
   * }
   * ```
   *
   * @example
   * ```typescript
   * // Load metadata (also goes through cache)
   * const zattrs = await store.get('.zattrs');
   * if (zattrs) {
   *   const attrs = JSON.parse(new TextDecoder().decode(zattrs));
   *   console.log('Dataset metadata:', attrs);
   * }
   * ```
   *
   * @example
   * ```typescript
   * // Cache cascade demonstration
   * // First access: L3 fetch (~100ms)
   * console.time('first');
   * await store.get('positions/0.0.0');
   * console.timeEnd('first'); // ~100ms
   *
   * // Second access: L1 hit (~1μs)
   * console.time('second');
   * await store.get('positions/0.0.0');
   * console.timeEnd('second'); // ~0.001ms
   * ```
   *
   * @remarks Performance: Typical access pattern:
   *              - First load: 90% L3 fetches (cold cache)
   *              - Subsequent loads: 95% L1 hits, 4% L2 hits, 1% L3 fetches
   *              - Memory usage: L1 ~100MB, L2 ~2GB (configurable)
   *
   * @see {@link init} for cache initialization and validation
   * @see {@link setPrefetcher} for enabling automatic adjacent chunk loading
   */
  async get(key: string, _options?: unknown): Promise<Uint8Array | undefined> {
    // zarrita's AsyncReadable contract is `Uint8Array | undefined`; both
    // Missing and transient NetworkError collapse to undefined here. Internal
    // callers that need to distinguish the cases use `getResult` directly.
    const result = await this.getResult(key);
    if (isErr(result)) {
      if (result.error.kind === 'NetworkError') {
        log.warning(Modules.CACHE, `Network error fetching ${key}: ${result.error.cause.message}`);
      }
      return undefined;
    }
    return result.value;
  }

  /**
   * Get a chunk and report the failure mode structurally.
   *
   * Same L1 → L2 → L3 cascade as {@link get}, but distinguishes:
   * - `ok(data)` — present in some tier or fetched successfully.
   * - `err({ kind: 'Missing' })` — server returned non-2xx (e.g. 404)
   *   or a non-retryable client error. Caller may treat as "not yet
   *   stored" without alarm.
   * - `err({ kind: 'NetworkError', cause })` — transient network/DNS
   *   error or 5xx after retries exhausted. Caller may back off.
   * - `err({ kind: 'Aborted' })` — caller-supplied AbortSignal fired.
   */
  async getResult(
    key: string,
    options?: { signal?: AbortSignal; suppressPrefetch?: boolean }
  ): Promise<Result<Uint8Array, CacheError>> {
    // Disposed-store fast path: bail before touching any tier. Avoids
    // late writes against a torn-down L2 and lets a dataset switch
    // unwind in-flight prefetch.onAccess cascades cleanly.
    if (this.disposed || this.dataAbort.signal.aborted) {
      return err({ kind: 'Aborted' });
    }

    // Only count user-demand requests in the demand hit-rate. The
    // prefetcher calls back through getResult with
    // `suppressPrefetch: true` after observing demand on key K to
    // pre-load K+1, K+2, …; counting those would distort the hit-rate
    // (a successful prefetch would look identical to a successful
    // user demand).
    const isDemand = !options?.suppressPrefetch;

    // L1: Memory check (fastest, ~1μs). Stays direct (no coalescing
    // needed — synchronous, no I/O cost to share).
    const l1Hit = this.l1Cache.get(key);
    if (l1Hit) {
      this.log(`L1 hit: ${key}`, 'info');
      if (isDemand) {
        this.l1HitCount++;
        this.totalBytesServed += l1Hit.byteLength;
        this.totalRequestsServed++;
      }
      return ok(l1Hit);
    }

    // Per-caller abort: a caller-supplied signal that fired between
    // the L1 miss and the coalesced wait surfaces as Aborted to that
    // caller without affecting any shared chain.
    if (options?.signal?.aborted) return err({ kind: 'Aborted' });

    // Coalesce same-key L2/network cascade. The first concurrent
    // caller creates the chain; subsequent callers await it. Bytes,
    // L1, and L2 are populated exactly once.
    //
    // Bypass coalescing when the caller provides its own signal: the
    // shared chain only respects dataAbort (so its lifetime can outlive
    // any single caller), but a caller passing a signal expects that
    // aborting it actually cancels the underlying fetch resource.
    // Falling back to a non-coalesced fetchKeyChain preserves the
    // per-caller abort contract for those (rare) callers.
    let inflight: Promise<{
      result: Result<Uint8Array, CacheError>;
      source: 'l2' | 'network' | 'missing';
    }>;
    // MED-2: only the originator of a coalesced inflight chain should
    // run prefetcher.onAccess(key) — subsequent waiters early-return
    // inside the prefetcher anyway, but each call still walks the
    // neighbour set / seen-set. Tracking originator here dedupes that
    // fan-out to exactly one onAccess call per logical access.
    let isPrefetchOriginator = true;
    if (options?.signal !== undefined) {
      inflight = this.fetchKeyChain(key, options.signal);
    } else {
      const cached = this.pendingGets.get(key);
      if (cached) {
        inflight = cached.promise;
        isPrefetchOriginator = false;
      } else {
        // CRIT-5: per-entry AbortController lets validateCache cancel
        // in-flight gets on a content-hash mismatch so the post-fetch
        // L1/L2 populate cannot resurrect stale data after a clear.
        const controller = new AbortController();
        let fresh!: Promise<{
          result: Result<Uint8Array, CacheError>;
          source: 'l2' | 'network' | 'missing';
        }>;
        fresh = this.fetchKeyChain(key, controller.signal).finally(() => {
          // An invalidation clears pendingGets before an aborted chain settles.
          // A new request may install a replacement for the same key during
          // that window, so the old chain must only remove its own entry.
          if (this.pendingGets.get(key)?.promise === fresh) {
            this.pendingGets.delete(key);
          }
        });
        this.pendingGets.set(key, { promise: fresh, controller });
        inflight = fresh;
      }
    }

    let outcome: { result: Result<Uint8Array, CacheError>; source: 'l2' | 'network' | 'missing' };
    try {
      outcome = await inflight;
    } catch (error) {
      const cause = error instanceof Error ? error : new Error(String(error));
      return err({ kind: 'NetworkError', cause });
    }

    // Per-caller demand counters: which tier "served" this caller.
    // All callers waiting on a shared network fetch count as demand
    // network requests; the underlying network counter (incremented
    // inside fetchKeyChain) only bumped once per actual fetch.
    if (isDemand) {
      if (outcome.source === 'l2') this.l2HitCount++;
      else if (outcome.source === 'network') this.demandNetworkRequestCount++;
      // Count delivered bytes for any tier that actually returned data
      // (L2 or network). L1 hits are counted on their fast-path return
      // above; Missing/Aborted outcomes carry no bytes.
      if (outcome.result.ok) {
        this.totalBytesServed += outcome.result.value.byteLength;
        this.totalRequestsServed++;
      }
    }

    // MED-2: per-key prefetch trigger. Only the originator of the
    // coalesced inflight chain fans out neighbour onAccess work — the
    // prefetcher's own seen-set would early-return for waiters, but
    // each call still touches that seen-set N times for N waiters.
    // Restrict to the originator (or non-coalesced signal'd callers)
    // so we do exactly one onAccess per logical access.
    if (!options?.suppressPrefetch && outcome.result.ok && isPrefetchOriginator) {
      this.prefetcher?.onAccess(key);
    }

    return outcome.result;
  }

  /**
   * Run the L2 → network cascade for a single key. Called at most once
   * per key per concurrent-getter wave by getResult's pendingGets
   * coalescer. Increments aggregate network counters once; per-caller
   * demand counters are incremented in getResult after this resolves.
   */
  private async fetchKeyChain(
    key: string,
    callerSignal?: AbortSignal
  ): Promise<{ result: Result<Uint8Array, CacheError>; source: 'l2' | 'network' | 'missing' }> {
    // L2: OPFS check (~1ms)
    if (this.enabled && this.l2Store) {
      const l2Hit = await this.l2Store.get(key);
      if (l2Hit) {
        this.log(`L2 hit: ${key}`, 'info');
        // CRIT-5: if validateCache aborted this in-flight get during the
        // L2 read (content-hash mismatch), do NOT promote stale bytes to
        // a just-cleared L1.
        if (callerSignal?.aborted || this.disposed || this.dataAbort.signal.aborted) {
          return { result: err({ kind: 'Aborted' }), source: 'l2' };
        }
        // Promote to L1
        this.l1Cache.set(key, l2Hit);
        return { result: ok(l2Hit), source: 'l2' };
      }
    }

    // L3: Remote fetch (~100ms)
    this.log(`HTTP fetch: ${key}`, 'info');
    // Compose the store-level dispose signal so dataset disposal
    // aborts in-flight prefetch/demand fetches without each call site
    // plumbing its own controller. When the caller passed its own
    // signal (and bypassed coalescing in getResult), forward that too
    // so per-caller cancellation actually aborts the resource.
    const fetchSignal = mergeAbortSignals(this.dataAbort.signal, callerSignal);
    let response: Response | undefined;
    try {
      response = await fetchWithRetry(buildUrl(this.baseUrl, key), { signal: fetchSignal });
    } catch (error) {
      const cause = error instanceof Error ? error : new Error(String(error));
      return { result: err({ kind: 'NetworkError', cause }), source: 'network' };
    }

    if (this.disposed || this.dataAbort.signal.aborted || callerSignal?.aborted) {
      return { result: err({ kind: 'Aborted' }), source: 'network' };
    }
    if (!response) {
      if (this.disposed || this.dataAbort.signal.aborted || callerSignal?.aborted) {
        return { result: err({ kind: 'Aborted' }), source: 'network' };
      }
      return {
        result: err({
          kind: 'NetworkError',
          cause: new Error(`fetch exhausted retries for ${key}`),
        }),
        source: 'network',
      };
    }
    if (!response.ok) {
      return { result: err({ kind: 'Missing' }), source: 'missing' };
    }

    const data = new Uint8Array(await response.arrayBuffer());

    // Aggregate network counters (one per actual fetch — pendingGets
    // ensures this body runs at most once per key per concurrent wave).
    this.networkRequestCount++;
    this.networkBytesTransferred += data.byteLength;
    this.bandwidth.record(data.byteLength);

    // CRIT-5: if validateCache aborted this in-flight get between the
    // arrayBuffer() resolve and now (content-hash mismatch raced an
    // in-flight fetch), do NOT write stale bytes back into a
    // just-cleared L1/L2 — that would silently undo the invalidation.
    if (callerSignal?.aborted || this.disposed || this.dataAbort.signal.aborted) {
      return { result: err({ kind: 'Aborted' }), source: 'network' };
    }

    // Populate caches once. L1 synchronously (the caller may read it back
    // immediately); L2 (OPFS) is DEFERRED to the background write queue so the
    // durable disk write never blocks this fetch — profiling showed the awaited
    // OPFS write dominated the cold-load critical path (~6× the network fetch).
    // The bytes are already in hand + promoted to L1, so a queued write carries
    // no correctness weight for THIS session; it only persists for the next.
    if (this.enabled) {
      this.l1Cache.set(key, data);
      if (this.l2Store) {
        const l2Store = this.l2Store;
        // Capture the epoch NOW; re-check at drain time so a clear/dispose that
        // interleaves between enqueue and the actual write drops the stale write
        // (the enqueue→drain window that the inline await used to make atomic).
        const epoch = this.l2Epoch;
        this.l2WriteQueue.enqueue(key, async () => {
          if (this.disposed || this.dataAbort.signal.aborted || this.l2Epoch !== epoch) {
            return; // superseded by dispose or a cache clear — do not persist
          }
          try {
            await l2Store.set(key, data);
          } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            log.warning(Modules.CACHE, `L2 write failed for ${key}: ${msg}`);
          }
        });
      }
    }

    return { result: ok(data), source: 'network' };
  }

  /**
   * Validate cache using content hash. Clears cache if content changed.
   *
   * Validation is serialized per dataset via a static queue keyed on
   * `datasetId`, which is `SHA-256(baseUrl)` (see {@link hashUrl}). All
   * `MultiLevelCachingStore` instances pointing at the same URL share the
   * same id and therefore the same queue, so rapid same-URL switches
   * cannot let an older validation finish after a newer one and restore
   * stale metadata. Each entry registers an `AbortController` so a
   * subsequent `dispose()` on this instance can cancel both the
   * in-flight HTTP fetch and any waiting follow-up validation that
   * captured `this`.
   */
  private async validateCache(datasetId: string): Promise<void> {
    await ValidationQueue.serialize(datasetId, (signal) => this.doValidateCache(signal));
  }

  private async doValidateCache(signal: AbortSignal): Promise<void> {
    try {
      const remoteToken = await getRemoteContentHash(this.baseUrl, {
        signal,
        timeoutMsOverride: config.dataLoading.network.validationTimeoutMs,
      });

      if (signal.aborted) {
        this.log('Validation aborted (caller disposed)');
        return;
      }

      if (!remoteToken) {
        // No `.zattrs` reachable at all (offline / headerless store) —
        // no token to compare. Apply optional TTL — if the cache is older
        // than `cache.externalDatasetTtlMs` we invalidate to avoid serving
        // stale data indefinitely. Tracked as `validationMode: ttl`
        // (or `none` when no TTL is configured).
        const ttlMs = config.cache.externalDatasetTtlMs;
        const state = this.l2Store?.getValidationState();
        if (ttlMs != null && state?.lastValidatedAt != null) {
          const age = Date.now() - state.lastValidatedAt;
          if (age > ttlMs) {
            this.log(`External dataset TTL expired (${age}ms > ${ttlMs}ms), clearing cache`);
            await this.invalidateAllTiers();
          }
        }
        this.l2Store?.setValidationMode(ttlMs != null ? 'ttl' : 'none');
        return;
      }

      const remoteHash = remoteToken.hash;
      const cachedHash = this.l2Store?.getContentHash();

      // Compare tokens. Implicit (`zattrs:`-prefixed) and stamped tokens
      // can never collide, so a producer ADDING content_hash to a dataset
      // previously validated implicitly also reads as a change — which is
      // correct (the dataset was rewritten).
      if (cachedHash && remoteHash !== cachedHash) {
        this.log('Dataset content changed, clearing cache');
        this.log(`Old: ${cachedHash.slice(0, 16)}...`);
        this.log(`New: ${remoteHash.slice(0, 16)}...`);
        // Defensive: clear L1 alongside L2 even though init() builds a
        // fresh L1 before validation runs. Cheap, makes the invariant
        // ("hash mismatch ⇒ every tier dropped") explicit, and protects
        // future call paths that might revalidate against a populated
        // L1 (e.g. content-hash refresh during a long session).
        await this.invalidateAllTiers();
      }

      this.l2Store?.setContentHash(remoteHash);
      this.l2Store?.setValidationMode(remoteToken.mode);
    } catch {
      // Offline or error - use cached data
      this.log('Cannot validate (offline?), using cached data');
    }
  }

  /**
   * CRIT-5: cancel every in-flight coalesced get and forget them, so a fetch
   * that started before an invalidation cannot resurrect stale bytes by writing
   * back into the just-cleared L1/L2 after it resolves. Each pending entry's
   * controller signal is composed into `fetchKeyChain`, so `abort()` trips the
   * post-arrayBuffer populate guard. Shared by the content-hash-mismatch and
   * TTL-expiry clear paths so the two stay in lockstep.
   */
  private abortPendingGets(): void {
    for (const [, pending] of this.pendingGets) {
      pending.controller.abort();
    }
    this.pendingGets.clear();
  }

  /**
   * Drop every cache tier for an invalidation (content-hash mismatch, TTL
   * expiry, or a user-triggered clearAll). Ordering is load-bearing:
   * `abortPendingGets()` runs FIRST so no in-flight coalesced fetch can pass
   * `fetchKeyChain`'s populate guard and repopulate a tier DURING the async
   * `clearL2()` wipe — the residual window a clear-then-abort order leaves open.
   * L2's epoch is bumped inside `clearL2()`; L0 is dropped via the invalidation
   * callbacks. Shared by all three invalidation paths so they stay in lockstep.
   */
  private async invalidateAllTiers(): Promise<void> {
    this.abortPendingGets();
    this.clearL1();
    await this.clearL2();
    this.invalidationCallbacks.forEach((cb) => cb());
  }

  /**
   * List all cached datasets in OPFS. Thin wrapper around
   * {@link OPFSStore.listAll} — kept on the orchestrator so external
   * callers keep importing through the package's public API.
   */
  async listDatasets(): Promise<CachedDatasetSummary[]> {
    return OPFSStore.listAll();
  }

  /**
   * Get cache statistics. Returns the aggregated multi-tier snapshot
   * (`MultiLevelCacheStats`) consumed by the data-loading monitor,
   * debug overlay, and cache E2E suite.
   */
  getStats(): MultiLevelCacheStats {
    const bandwidth = this.bandwidth.rate();

    const validationState = this.l2Store?.getValidationState() ?? {
      mode: 'none' as const,
      lastValidatedAt: null,
    };

    // S2: opfsAvailable signals whether L2 storage is operational.
    // When caching is disabled (config / ?no-cache), L2 wasn't
    // expected, so report `true` so the UI doesn't surface a
    // misleading badge.
    const l2Stats = this.l2Store?.getStats();
    const opfsAvailable = this.enabled ? (l2Stats?.available ?? false) : true;

    return {
      l1: this.l1Cache.getStats(),
      l2: {
        ...(l2Stats ?? {
          size: 0,
          count: 0,
          reads: 0,
          writes: 0,
          misses: 0,
        }),
        // Fixed OPFS/disk budget — surfaced so the monitor's memory gauge has a
        // real per-tier limit to sum (L2 is disk, not heap, but it is a tier).
        maxSize: this.l2MaxSize,
      },
      l2WriteQueue: this.l2WriteQueue.stats(),
      network: {
        bytesTransferred: this.networkBytesTransferred,
        requestCount: this.networkRequestCount,
        bandwidth,
        totalBytesServed: this.totalBytesServed,
        totalRequestsServed: this.totalRequestsServed,
      },
      demand: {
        l1Hits: this.l1HitCount,
        l2Hits: this.l2HitCount,
        networkRequests: this.demandNetworkRequestCount,
      },
      health: {
        validationMode: validationState.mode,
        lastValidatedAt: validationState.lastValidatedAt,
        unvalidatedExternalDataset: validationState.mode === 'none',
        opfsAvailable,
      },
      clearOnInitCount: this.clearOnInitCount,
    };
  }

  /**
   * Check if caching is enabled.
   * Used by CacheStatsProvider interface.
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Clear L1 memory cache only.
   */
  clearL1(): void {
    this.l1Cache.clear();
  }

  /**
   * Clear L2 OPFS cache only.
   *
   * Bumps `l2Epoch` and drops queued background writes FIRST (synchronously),
   * so a write enqueued before this clear can never land after it and resurrect
   * stale bytes: not-yet-started tasks are dropped here, and an already-running
   * task self-drops on its epoch re-check (with the OPFS `generation` counter as
   * a final backstop). This is the single chokepoint for every L2-clearing path
   * (`clearAll`, content-hash mismatch, TTL expiry, `?clear-cache`).
   */
  async clearL2(): Promise<void> {
    this.l2Epoch++;
    this.l2WriteQueue.clear();
    if (this.l2Store) {
      await this.l2Store.clear();
    }
  }

  /**
   * Clear all caches (L1 + L2). Reachable mid-session from the monitor /
   * settings "clear caches" controls and `__luxarDebug`, so it aborts in-flight
   * gets FIRST (see invalidateAllTiers) to prevent stale repopulation.
   */
  async clearAll(): Promise<void> {
    await this.invalidateAllTiers();
  }

  /**
   * Dispose the cache store. Flushes pending writes, cancels any in-flight
   * cache-validation fetch, removes the dataset from the static validation
   * queue, and clears L1.
   *
   * The validation cancellation matters because two stores against the same
   * URL share the static queue; without it, a closure that captured `this`
   * could run `setContentHash()` against a disposed l2Store.
   */
  async dispose(): Promise<void> {
    // Mark disposed first so concurrent getResult calls bail synchronously
    // and any racing fetch-error path returns Aborted. Aborting dataAbort
    // unwinds in-flight prefetch/demand fetches that were composed with
    // it via mergeAbortSignals in getResult.
    this.disposed = true;
    this.dataAbort.abort();

    // Tear down the prefetcher: clears queues/seen/parsed/bounds and
    // sets its own isDisposed flag so the in-flight `.finally()` path
    // cannot re-enter processQueue with new fetches.
    this.prefetcher?.dispose();
    this.prefetcher = null;

    // Cancel any in-flight or queued validation belonging to this instance.
    if (this.datasetId !== undefined) {
      ValidationQueue.cancel(this.datasetId);
    }

    // Drop not-yet-started background L2 writes (best-effort tier; keeps
    // dataset-switch teardown fast). Writes that already entered `l2Store.set`
    // are registered in OPFSStore.pendingWrites and drained by its dispose()
    // below; the `disposed` flag set above also makes any dequeued-but-unstarted
    // task self-drop on its epoch/disposed re-check.
    this.l2WriteQueue.clear();

    if (this.l2Store) {
      await this.l2Store.dispose();
    }
    this.clearL1();
  }

  /**
   * Conditional logging based on debug mode. Errors always emit; info/warn
   * only when `debug` is on. All output is routed through the shared
   * `log` utility so it shows up consistently in the debug-console overlay.
   */
  private log(message: string, level: 'info' | 'warn' | 'error' = 'info'): void {
    if (level === 'error') {
      log.error(Modules.CACHE, message);
    } else if (this.debug) {
      if (level === 'warn') log.warning(Modules.CACHE, message);
      else log.info(Modules.CACHE, message);
    }
  }
}
