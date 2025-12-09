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
   * Initialize OPFS store and validate cache.
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
   * Get a zarr chunk with L1 → L2 → HTTP cascade.
   * Implements zarrita's AsyncReadable interface.
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
        console.log('[Cache] Dataset content changed, clearing cache');
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
   */
  getStats(): {
    l1: { metadataSize: number; chunksSize: number; metadataCount: number; chunksCount: number };
    l2: { size: number; count: number };
    } {
    return {
      l1: this.l1Cache.getStats(),
      l2: this.l2Store?.getStats() ?? { size: 0, count: 0 },
    };
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
