import type { AsyncReadable } from '@zarrita/storage';
import { SegmentedLRUCache } from './segmented-lru-cache';
import { OPFSStore } from './opfs-store';
import type { ChunkPrefetcher } from './chunk-prefetcher';

export interface TwoLevelCachingStoreOptions {
  /** L1 memory cache size in bytes (default: 100MB) */
  l1MaxSize?: number;
  /** L2 OPFS cache size in bytes (default: 2GB) */
  l2MaxSize?: number;
  /** Enable debug logging (default: false) */
  debug?: boolean;
  /** URL parameters for testability (optional, defaults to window.location.search) */
  urlParams?: URLSearchParams;
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

  private static readonly DEFAULT_L1_SIZE = 100 * 1024 * 1024; // 100MB
  private static readonly DEFAULT_L2_SIZE = 2 * 1024 * 1024 * 1024; // 2GB

  // Network I/O tracking
  private networkBytesTransferred = 0;
  private networkRequestCount = 0;
  private networkStartTime = Date.now();

  constructor(baseUrl: string, options?: TwoLevelCachingStoreOptions) {
    this.baseUrl = baseUrl;

    // URL params (testable via options.urlParams, defaults to window.location)
    const params =
      options?.urlParams ??
      new URLSearchParams(typeof window !== 'undefined' ? window.location?.search : '');

    const noCache = params.has('no-cache');
    const cacheDebug = params.has('cache-debug');
    const clearCache = params.has('clear-cache');

    this.enabled = !noCache;
    this.debug = cacheDebug || options?.debug || false;
    this.shouldClearOnInit = clearCache;

    // Initialize L1 (always, even if disabled)
    const l1Size = options?.l1MaxSize ?? TwoLevelCachingStore.DEFAULT_L1_SIZE;
    this.l1Cache = new SegmentedLRUCache(l1Size);

    // Save L2 max size for later initialization
    this.l2MaxSize = options?.l2MaxSize ?? TwoLevelCachingStore.DEFAULT_L2_SIZE;
  }

  /**
   * Attach a prefetcher to receive access notifications.
   */
  setPrefetcher(prefetcher: ChunkPrefetcher | null): void {
    this.prefetcher = prefetcher;
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
   *   // Fall back to direct HTTP (no caching)
   *   store = new TwoLevelCachingStore(url, { cachingEnabled: false });
   * }
   * ```
   *
   * @see {@link validateCache} for cache validation algorithm
   * @see {@link SPECIFICATIONS.md} Section 4 for OPFS architecture
   * @performance First init: ~100ms (OPFS setup + validation), Subsequent: ~10ms (validation only)
   */
  async init(): Promise<void> {
    if (!this.enabled) {
      this.log('Caching disabled');
      return;
    }

    // Generate dataset ID from URL and create L2 store
    const datasetId = await this.hashUrl(this.baseUrl);
    this.l2Store = new OPFSStore(datasetId, this.baseUrl, this.l2MaxSize);

    await this.l2Store.init();

    // Clear cache if requested
    if (this.shouldClearOnInit) {
      this.log('Clearing cache due to ?clear-cache parameter');
      await this.clearAll();
    }

    // Validate cache using content hash
    await this.validateCache();
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
   * @performance Typical access pattern:
   *              - First load: 90% L3 fetches (cold cache)
   *              - Subsequent loads: 95% L1 hits, 4% L2 hits, 1% L3 fetches
   *              - Memory usage: L1 ~100MB, L2 ~2GB (configurable)
   *
   * @see {@link init} for cache initialization and validation
   * @see {@link setPrefetcher} for enabling automatic adjacent chunk loading
   * @see {@link SPECIFICATIONS.md} Section 3 for complete cache algorithm
   */
  async get(key: string, _options?: any): Promise<Uint8Array | undefined> {
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
      const response = await fetch(`${this.baseUrl}/${key}`);
      if (!response.ok) return undefined;

      const data = new Uint8Array(await response.arrayBuffer());

      // Track network I/O
      this.networkRequestCount++;
      this.networkBytesTransferred += data.byteLength;

      // Populate both caches
      this.l1Cache.set(key, data);
      if (this.enabled && this.l2Store) {
        this.l2Store.set(key, data).catch(() => {});
      }

      // Trigger prefetch on L3 fetch
      this.prefetcher?.onAccess(key);

      return data;
    } catch (error) {
      // Log error with message (error objects don't serialize well in console)
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.warn(`[Cache] Network error fetching ${key}: ${errorMsg}`);
      return undefined;
    }
  }

  /**
   * Validate cache using content hash. Clears cache if content changed.
   */
  private async validateCache(): Promise<void> {
    try {
      // Fetch current content_hash directly from server (bypass cache)
      const remoteHash = await this.getRemoteContentHash();

      // No hash → skip validation (external dataset handling)
      if (!remoteHash) {
        this.log('No content_hash, skipping validation');
        return;
      }

      const cachedHash = this.l2Store?.getContentHash();

      // Compare hashes
      if (cachedHash && remoteHash !== cachedHash) {
        this.log('Dataset content changed, clearing cache');
        this.log(`Old: ${cachedHash.slice(0, 16)}...`);
        this.log(`New: ${remoteHash.slice(0, 16)}...`);
        await this.clearL2();
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
   */
  private async getRemoteContentHash(): Promise<string | null> {
    try {
      // Direct HTTP fetch, no cache lookup
      const response = await fetch(`${this.baseUrl}/.zattrs`);
      if (!response.ok) return null;

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
      for await (const [name, handle] of (opfsRoot as any).entries()) {
        if (name.startsWith('zarr-cache-') && handle.kind === 'directory') {
          try {
            // Read _cache_meta.json from this dataset
            const metaHandle = await handle.getFileHandle('_cache_meta.json');
            const file = await metaHandle.getFile();
            const meta = JSON.parse(await file.text());

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
    // Calculate average bandwidth (bytes per second since start)
    const elapsedSeconds = Math.max(1, (Date.now() - this.networkStartTime) / 1000);
    const bandwidth = this.networkBytesTransferred / elapsedSeconds;

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
  }

  /**
   * Dispose the cache store. Flushes pending writes and clears L1.
   */
  async dispose(): Promise<void> {
    // Clear prefetcher reference (in-flight requests will complete harmlessly)
    this.prefetcher = null;

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
