import type { AsyncReadable } from '@zarrita/storage';
import { SegmentedLRUCache } from './segmented-lru-cache';
import { OPFSStore } from './opfs-store';
import type { ChunkPrefetcher } from './chunk-prefetcher';
import { log, Modules } from '../utils/log';
import { config } from '../config';

type IterableFileSystemDirectoryHandle = FileSystemDirectoryHandle & {
  entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
};

/**
 * Merge a primary `AbortSignal` (e.g. per-attempt timeout) with an optional
 * caller signal (e.g. dispose-cancellation) so abort wins immediately on
 * either path.
 *
 * Uses native `AbortSignal.any` when available (Node 22+, modern browsers);
 * falls back to a hand-rolled relay otherwise.
 */
function mergeAbortSignals(primary: AbortSignal, caller?: AbortSignal): AbortSignal {
  if (!caller) return primary;
  type StaticAny = { any?: (signals: AbortSignal[]) => AbortSignal };
  const anyImpl = (AbortSignal as unknown as StaticAny).any;
  if (typeof anyImpl === 'function') {
    return anyImpl([primary, caller]);
  }
  // Fallback: relay aborts onto a fresh controller.
  const relay = new AbortController();
  const onAbort = (): void => relay.abort();
  if (primary.aborted || caller.aborted) {
    relay.abort();
  } else {
    primary.addEventListener('abort', onAbort, { once: true });
    caller.addEventListener('abort', onAbort, { once: true });
  }
  return relay.signal;
}

interface CacheMetadataFile {
  baseUrl?: string;
  contentHash?: string;
  totalSize?: number;
  entries?: unknown[];
}

export interface TwoLevelCachingStoreOptions {
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
}

/**
 * Two-level caching store that implements zarrita's AsyncReadable interface.
 * Orchestrates L1 (memory), L2 (OPFS), and HTTP fallback for zarr chunks.
 */
export class TwoLevelCachingStore implements AsyncReadable {
  private l1Cache: SegmentedLRUCache;
  private l2Store: OPFSStore | null = null;
  private prefetcher: ChunkPrefetcher | null = null;
  private baseUrl: string;
  private l2MaxSize: number;
  private enabled: boolean;
  private debug: boolean;
  private shouldClearOnInit: boolean;

  private static readonly DEFAULT_L1_SIZE = config.cache.l1MaxSizeMB * 1024 * 1024;
  private static readonly DEFAULT_L2_SIZE = config.cache.l2MaxSizeMB * 1024 * 1024;
  /**
   * Per-dataset validation queue keyed by `datasetId` (URL hash). Two stores
   * pointing at the same URL serialize their validations across instances so
   * a slower-finishing older validation cannot overwrite a newer
   * content-hash. Each entry carries an `AbortController` so `dispose()`
   * can both cancel the in-flight fetch AND remove the queue entry,
   * preventing a closure that captured `this` from running
   * `setContentHash()` against a disposed L2 store.
   */
  private static readonly validationQueues = new Map<
    string,
    { promise: Promise<void>; abort: AbortController }
  >();
  private static readonly INITIAL_RETRY_DELAY_MS = 50;
  private static readonly MAX_RETRY_DELAY_MS = 500;

  // Captured during init() so dispose() can find this instance's queue entry.
  private datasetId?: string;

  // Invalidation callbacks (e.g., L0 DecompressedChunkCache clearing on L1/L2 invalidation)
  private invalidationCallbacks: (() => void)[] = [];

  // Network I/O tracking
  private networkBytesTransferred = 0;
  private networkRequestCount = 0;

  // Sliding window bandwidth tracking (last ~10 seconds)
  private bandwidthWindow: { timestamp: number; bytes: number }[] = [];
  private static readonly BANDWIDTH_WINDOW_MS = 10_000;

  constructor(baseUrl: string, options?: TwoLevelCachingStoreOptions) {
    this.baseUrl = baseUrl;

    this.enabled = !(options?.noCache ?? false);
    this.debug = options?.debug ?? false;
    this.shouldClearOnInit = options?.clearCache ?? false;

    // Initialize L1 (always, even if disabled)
    const l1Size = options?.l1MaxSize ?? TwoLevelCachingStore.DEFAULT_L1_SIZE;
    this.l1Cache = new SegmentedLRUCache(l1Size);

    // Save L2 max size for later initialization
    this.l2MaxSize = options?.l2MaxSize ?? TwoLevelCachingStore.DEFAULT_L2_SIZE;
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
   * const store = new TwoLevelCachingStore(url, {
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
   * @see SPECIFICATIONS.md - Section 4 for OPFS architecture
   * @remarks Performance: First init: ~100ms (OPFS setup + validation), Subsequent: ~10ms (validation only)
   */
  async init(): Promise<void> {
    if (!this.enabled) {
      this.log('Caching disabled');
      return;
    }

    // Generate dataset ID from URL and create L2 store
    const datasetId = await this.hashUrl(this.baseUrl);
    this.datasetId = datasetId;
    this.l2Store = new OPFSStore(datasetId, this.baseUrl, this.l2MaxSize);

    await this.l2Store.init();

    // Clear cache if requested
    if (this.shouldClearOnInit) {
      this.log('Clearing cache due to ?clear-cache parameter');
      await this.clearAll();
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
   * @see SPECIFICATIONS.md - Section 3 for complete cache algorithm
   */
  async get(key: string, _options?: unknown): Promise<Uint8Array | undefined> {
    // L1: Memory check (fastest, ~1μs)
    const l1Hit = this.l1Cache.get(key);
    if (l1Hit) {
      this.log(`L1 hit: ${key}`, 'info');
      return l1Hit;
    }

    // L2: OPFS check (~1ms)
    if (this.enabled && this.l2Store) {
      const l2Hit = await this.l2Store.get(key);
      if (l2Hit) {
        this.log(`L2 hit: ${key}`, 'info');
        // Promote to L1
        this.l1Cache.set(key, l2Hit);
        // Trigger prefetch on L2 hit
        this.prefetcher?.onAccess(key);
        return l2Hit;
      }
    }

    // L3: Remote fetch (~100ms)
    try {
      this.log(`HTTP fetch: ${key}`, 'info');
      const response = await this.fetchWithRetry(this.buildUrl(key));
      if (!response?.ok) return undefined;

      const data = new Uint8Array(await response.arrayBuffer());

      // Track network I/O
      this.networkRequestCount++;
      this.networkBytesTransferred += data.byteLength;
      this.bandwidthWindow.push({ timestamp: Date.now(), bytes: data.byteLength });

      // Populate caches (only if caching is enabled via URL params)
      if (this.enabled) {
        this.l1Cache.set(key, data);
        if (this.l2Store) {
          this.l2Store.set(key, data).catch((e) => {
            log.warning(Modules.CACHE, `L2 write failed for ${key}: ${e?.message || e}`);
          });
        }
      }

      // Trigger prefetch on L3 fetch
      this.prefetcher?.onAccess(key);

      return data;
    } catch (error) {
      // Log error with message (error objects don't serialize well in console)
      const errorMsg = error instanceof Error ? error.message : String(error);
      log.warning(Modules.CACHE, `Network error fetching ${key}: ${errorMsg}`);
      return undefined;
    }
  }

  /**
   * Validate cache using content hash. Clears cache if content changed.
   *
   * Validation is serialized per dataset via a static queue keyed on
   * `datasetId`, which is `SHA-256(baseUrl)` (see {@link hashUrl}). All
   * `TwoLevelCachingStore` instances pointing at the same URL share the
   * same id and therefore the same queue, so rapid same-URL switches
   * cannot let an older validation finish after a newer one and restore
   * stale metadata. Each entry registers an `AbortController` so a
   * subsequent `dispose()` on this instance can cancel both the
   * in-flight HTTP fetch and any waiting follow-up validation that
   * captured `this`.
   */
  private async validateCache(datasetId: string): Promise<void> {
    const abort = new AbortController();
    const previous = TwoLevelCachingStore.validationQueues.get(datasetId);
    const validation = (previous?.promise ?? Promise.resolve())
      .catch(() => undefined)
      .then(() => {
        // Skip the validation entirely if dispose() aborted while we were
        // waiting in line. The closure captured `this`, so running
        // doValidateCache() now would write into a disposed l2Store.
        if (abort.signal.aborted) return;
        return this.doValidateCache(abort.signal);
      });

    const entry = { promise: validation, abort };
    TwoLevelCachingStore.validationQueues.set(datasetId, entry);
    try {
      await validation;
    } finally {
      // Only delete the entry if it's still ours — a newer validation may
      // have replaced it after we started.
      if (TwoLevelCachingStore.validationQueues.get(datasetId) === entry) {
        TwoLevelCachingStore.validationQueues.delete(datasetId);
      }
    }
  }

  private async doValidateCache(signal: AbortSignal): Promise<void> {
    try {
      // Fetch current content_hash directly from server (bypass cache)
      const remoteHash = await this.getRemoteContentHash(signal);

      // No hash → skip validation (external dataset handling)
      if (!remoteHash) {
        this.log('No content_hash, skipping validation');
        return;
      }

      // If dispose() aborted between the fetch and the L2 write, bail out
      // before touching the (possibly already-disposed) l2Store.
      if (signal.aborted) {
        this.log('Validation aborted (caller disposed)');
        return;
      }

      const cachedHash = this.l2Store?.getContentHash();

      // Compare hashes
      if (cachedHash && remoteHash !== cachedHash) {
        this.log('Dataset content changed, clearing cache');
        this.log(`Old: ${cachedHash.slice(0, 16)}...`);
        this.log(`New: ${remoteHash.slice(0, 16)}...`);
        await this.clearL2();
        this.invalidationCallbacks.forEach((cb) => cb());
      }

      this.l2Store?.setContentHash(remoteHash);
    } catch {
      // Offline or error - use cached data
      this.log('Cannot validate (offline?), using cached data');
    }
  }

  /**
   * Get content_hash directly from remote server, bypassing cache.
   * Used for cache validation to detect dataset changes.
   * This ensures we always check the TRUE current hash, not a cached one.
   *
   * Uses the dedicated `validationTimeoutMs` budget (default 5 s) so a flaky
   * network never blocks scene loading for the full data-fetch timeout.
   */
  private async getRemoteContentHash(signal?: AbortSignal): Promise<string | null> {
    try {
      // Direct HTTP fetch, no cache lookup
      const response = await this.fetchWithRetry(this.buildUrl('.zattrs'), {
        timeoutMsOverride: config.dataLoading.network.validationTimeoutMs,
        signal,
      });
      if (!response?.ok) return null;

      const data = await response.arrayBuffer();
      const attrs = JSON.parse(new TextDecoder().decode(data));
      return attrs?.content_hash ?? null;
    } catch (error) {
      // Network error or parse error
      const errorMsg = error instanceof Error ? error.message : String(error);
      this.log(`Failed to fetch remote content_hash: ${errorMsg}`, 'warn');
      return null;
    }
  }

  /**
   * Build a remote URL without producing duplicate slashes.
   */
  private buildUrl(key: string): string {
    const cleanBase = this.baseUrl.replace(/\/+$/, '');
    const cleanKey = key.replace(/^\/+/, '');
    return `${cleanBase}/${cleanKey}`;
  }

  /**
   * Fetch with retries for transient failures.
   *
   * 4xx responses are returned immediately because retrying cannot fix a
   * missing zarr key. Network errors, timeouts, 429, and 5xx responses are
   * retried using the configured retry budget. The configured timeout is
   * treated as a total budget across attempts so retries do not multiply
   * worst-case load time.
   *
   * @param url - URL to fetch.
   * @param options - Optional `timeoutMsOverride` (e.g. for cache-validation
   *   probes that want a shorter budget than the data-fetch timeout) and a
   *   caller `signal` for dispose-cancel propagation. The caller signal is
   *   merged with the per-attempt timeout signal so either abort source
   *   wins immediately. A caller-aborted call exits without consuming
   *   retry budget.
   */
  private async fetchWithRetry(
    url: string,
    options?: { timeoutMsOverride?: number; signal?: AbortSignal }
  ): Promise<Response | undefined> {
    const maxAttempts = Math.max(1, config.dataLoading.network.retryAttempts + 1);
    const totalTimeoutMs = options?.timeoutMsOverride ?? config.dataLoading.network.timeoutMs;
    const timeoutPerAttemptMs = Math.max(1, Math.ceil(totalTimeoutMs / maxAttempts));
    let lastError: unknown;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      // Caller-aborted requests must not be retried; bail out before the
      // next attempt.
      if (options?.signal?.aborted) {
        return undefined;
      }

      const timeoutController = new AbortController();
      const timeoutId = setTimeout(() => timeoutController.abort(), timeoutPerAttemptMs);
      const signal = mergeAbortSignals(timeoutController.signal, options?.signal);

      try {
        const response = await fetch(url, { signal });
        if (response.ok || (response.status < 500 && response.status !== 429)) {
          return response;
        }
        lastError = new Error(`HTTP ${response.status} for ${url}`);
      } catch (error) {
        lastError = error;
        // Caller-aborted: exit immediately rather than retrying.
        if (options?.signal?.aborted) {
          return undefined;
        }
      } finally {
        clearTimeout(timeoutId);
      }

      if (attempt < maxAttempts - 1) {
        const delayMs = Math.min(
          TwoLevelCachingStore.INITIAL_RETRY_DELAY_MS * 2 ** attempt,
          TwoLevelCachingStore.MAX_RETRY_DELAY_MS
        );
        await this.sleep(delayMs);
      }
    }

    if (lastError) {
      const message = lastError instanceof Error ? lastError.message : String(lastError);
      log.warning(Modules.CACHE, `Fetch failed after ${maxAttempts} attempt(s): ${message}`);
    }
    return undefined;
  }

  private sleep(delayMs: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, delayMs));
  }

  /**
   * Generate a unique, collision-resistant hash for the dataset URL.
   */
  private async hashUrl(url: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(url);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
    return `zarr-cache-${hashHex.slice(0, 16)}`; // First 16 chars = 64 bits
  }

  /**
   * List all cached datasets in OPFS.
   */
  async listDatasets(): Promise<Array<{ url: string; hash: string; size: number; count: number }>> {
    const datasets = [];

    try {
      const opfsRoot = await navigator.storage.getDirectory();

      // Iterate all zarr-cache-* directories
      const iterableRoot = opfsRoot as IterableFileSystemDirectoryHandle;
      for await (const [name, handle] of iterableRoot.entries()) {
        if (name.startsWith('zarr-cache-') && handle.kind === 'directory') {
          try {
            // Read _cache_meta.json from this dataset
            const directoryHandle = handle as FileSystemDirectoryHandle;
            const metaHandle = await directoryHandle.getFileHandle('_cache_meta.json');
            const file = await metaHandle.getFile();
            const meta = JSON.parse(await file.text()) as CacheMetadataFile;

            datasets.push({
              url: meta.baseUrl || 'unknown',
              hash: meta.contentHash?.slice(0, 16) || 'none',
              size: meta.totalSize || 0,
              count: meta.entries?.length || 0,
            });
          } catch {
            // Skip corrupted/invalid cache directories
          }
        }
      }
    } catch {
      // OPFS not available
    }

    return datasets;
  }

  /**
   * Get cache statistics.
   * Returns extended stats compatible with CacheStatsProvider interface.
   */
  getStats(): {
    l1: {
      metadataSize: number;
      chunksSize: number;
      metadataCount: number;
      chunksCount: number;
      hits: number;
      misses: number;
      evictions: number;
    };
    l2: { size: number; count: number; reads: number; writes: number };
    network: { bytesTransferred: number; requestCount: number; bandwidth: number };
  } {
    // Calculate bandwidth using sliding window (last ~10 seconds)
    const now = Date.now();
    const windowStart = now - TwoLevelCachingStore.BANDWIDTH_WINDOW_MS;

    // Prune entries older than the window
    while (this.bandwidthWindow.length > 0 && this.bandwidthWindow[0].timestamp < windowStart) {
      this.bandwidthWindow.shift();
    }

    let bandwidth: number;
    if (this.bandwidthWindow.length === 0) {
      bandwidth = 0;
    } else {
      const windowBytes = this.bandwidthWindow.reduce((sum, e) => sum + e.bytes, 0);
      const windowSpan = Math.max(1, (now - this.bandwidthWindow[0].timestamp) / 1000);
      bandwidth = windowBytes / windowSpan;
    }

    return {
      l1: this.l1Cache.getStats(),
      l2: this.l2Store?.getStats() ?? { size: 0, count: 0, reads: 0, writes: 0 },
      network: {
        bytesTransferred: this.networkBytesTransferred,
        requestCount: this.networkRequestCount,
        bandwidth,
      },
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
   */
  async clearL2(): Promise<void> {
    if (this.l2Store) {
      await this.l2Store.clear();
    }
  }

  /**
   * Clear all caches (L1 + L2).
   */
  async clearAll(): Promise<void> {
    this.clearL1();
    await this.clearL2();
    this.invalidationCallbacks.forEach((cb) => cb());
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
    // Clear prefetcher reference (in-flight requests will complete harmlessly)
    this.prefetcher = null;

    // Cancel any in-flight or queued validation belonging to this instance.
    if (this.datasetId !== undefined) {
      const queued = TwoLevelCachingStore.validationQueues.get(this.datasetId);
      if (queued) {
        queued.abort.abort();
        TwoLevelCachingStore.validationQueues.delete(this.datasetId);
      }
    }

    if (this.l2Store) {
      await this.l2Store.dispose();
    }
    this.clearL1();
  }

  /**
   * Conditional logging based on debug mode.
   */
  private log(message: string, level: 'info' | 'warn' | 'error' = 'info'): void {
    if (level === 'error' || this.debug) {
      console[level](`[Cache] ${message}`);
    }
  }
}
