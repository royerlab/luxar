import type { TwoLevelCachingStore } from './two-level-caching-store';
import { log, Modules, LogEmoji } from '../utils/log';

export interface ChunkPrefetcherOptions {
  /** Maximum concurrent prefetch requests (default: 4) */
  maxConcurrent?: number;

  /** Enable/disable prefetching (default: true) */
  enabled?: boolean;

  /** Use Fetch Priority API hint (default: true) */
  useFetchPriority?: boolean;

  /** Enable debug logging (default: false) */
  debug?: boolean;

  /** URL parameters for testability (optional, defaults to window.location.search) */
  urlParams?: URLSearchParams;
}

/**
 * Chunk prefetcher that proactively fetches adjacent zarr chunks.
 *
 * Triggered on L2 hits and L3 fetches to hide network latency by prefetching
 * adjacent chunks (±1 in each dimension) before they are explicitly requested.
 *
 * Key features:
 * - Symmetric ±1 adjacency in all dimensions
 * - Concurrency limiting (default: 4 concurrent)
 * - Full deduplication (queue + in-flight)
 * - Fire-and-forget pattern (non-blocking)
 * - Race-condition safe queue processing
 *
 * See: docs/CACHE_PREFETCHING_SPEC.md
 */
export class ChunkPrefetcher {
  private store: TwoLevelCachingStore;
  private maxConcurrent: number;
  private enabled: boolean;
  private debug: boolean;

  // Queue management
  private inFlight = new Set<string>();
  private queue = new Set<string>();
  private processing = false;

  // Tracks all keys whose neighbors have already been enqueued, preventing
  // cascading prefetch amplification: without this, prefetched chunks trigger
  // their own neighbor prefetches, which cascade across the entire dataset.
  // Bounded to prevent unbounded memory growth during long browsing sessions.
  private static readonly MAX_SEEN_SIZE = 10000;
  private seen = new Set<string>();

  /** Upper bounds per array path for suppressing out-of-range prefetch requests */
  private maxChunkIndices = new Map<string, number[]>();

  constructor(store: TwoLevelCachingStore, options?: ChunkPrefetcherOptions) {
    this.store = store;

    // URL params (testable via options.urlParams, defaults to window.location)
    const params =
      options?.urlParams ??
      new URLSearchParams(typeof window !== 'undefined' ? window.location?.search : '');

    const noPrefetch = params.has('no-prefetch');
    const prefetchDebug = params.has('prefetch-debug');

    this.maxConcurrent = options?.maxConcurrent ?? 4;
    this.enabled = !noPrefetch && (options?.enabled ?? true);
    this.debug = prefetchDebug || options?.debug || false;

    if (!this.enabled) {
      this.log('Prefetching disabled');
    }
  }

  /**
   * Called by store after L2 hit or L3 fetch.
   * Enqueues adjacent chunks for prefetching.
   */
  onAccess(key: string): void {
    if (!this.enabled) return;

    // Prevent cascading prefetch amplification: if we've already expanded
    // this key's neighbors, don't do it again. Without this guard, a prefetched
    // chunk triggers its own neighbor expansion, which cascades until the entire
    // dataset is fetched (O(N^D) for D-dimensional data with N chunks/dim).
    if (this.seen.has(key)) return;
    this.seen.add(key);

    // Prevent unbounded memory growth — clear when limit reached.
    // Queue dedup (below) prevents actual duplicate fetches.
    if (this.seen.size > ChunkPrefetcher.MAX_SEEN_SIZE) {
      this.seen.clear();
      this.seen.add(key);
      this.maxChunkIndices.clear();
    }

    const adjacent = this.getAdjacentChunks(key);
    if (adjacent.length === 0) {
      // Not a chunk file (metadata), skip prefetching
      return;
    }

    this.log(`Access: ${key} → ${adjacent.length} adjacent chunks`);

    for (const adjKey of adjacent) {
      // Full deduplication: not in queue, not in-flight
      if (!this.queue.has(adjKey) && !this.inFlight.has(adjKey)) {
        this.queue.add(adjKey);
        this.log(`  Enqueued: ${adjKey}`);
      }
    }

    // Fire-and-forget (not awaited)
    this.processQueue();
  }

  /**
   * Process the prefetch queue with concurrency limiting.
   * Race-condition safe via processing flag and queueMicrotask re-trigger.
   */
  private async processQueue(): Promise<void> {
    // Prevent concurrent processing
    if (this.processing) return;
    this.processing = true;

    try {
      while (this.queue.size > 0 && this.inFlight.size < this.maxConcurrent) {
        const key = this.queue.values().next().value;
        if (!key) break; // Safety check (should never happen due to while condition)

        this.queue.delete(key);
        this.inFlight.add(key);

        this.log(`Prefetching: ${key} (${this.inFlight.size}/${this.maxConcurrent} slots)`);

        // Fire-and-forget with cleanup
        this.store
          .get(key)
          .catch(() => {
            // Ignore errors (404s, network failures)
          })
          .finally(() => {
            this.inFlight.delete(key);
            // Trigger queue processing when slot frees up
            if (this.queue.size > 0) {
              queueMicrotask(() => this.processQueue());
            }
          });
      }
    } finally {
      this.processing = false;
    }
  }

  /**
   * Parse chunk indices from a zarr chunk key.
   *
   * @example
   * parseChunkIndices('points/positions/0.1.2') → [0, 1, 2]
   * parseChunkIndices('points/positions/c/0/1/2') → [0, 1, 2]
   * parseChunkIndices('.zattrs') → null (not a chunk)
   */
  private parseChunkIndices(key: string): number[] | null {
    // CRITICAL: Check v3 FIRST before v2!
    // v2 regex can match the trailing digits of v3 paths (e.g., /2 in /c/0/1/2)

    // Check for v3 path notation: /c/ followed by path segments
    const v3Match = key.match(/\/c\/(\d+(?:\/\d+)*)$/);
    if (v3Match) {
      return v3Match[1].split('/').map(Number);
    }

    // Check for v2 dot notation: ends with digits separated by dots
    const v2Match = key.match(/\/(\d+(?:\.\d+)*)$/);
    if (v2Match) {
      return v2Match[1].split('.').map(Number);
    }

    return null; // Not a chunk key (metadata file)
  }

  /**
   * Register array shape and chunk sizes for bounds checking during prefetch.
   * When registered, getAdjacentChunks will skip indices beyond valid bounds.
   *
   * @param arrayPath - Base path of the array (e.g., 'gsplats_t0023/centers')
   * @param shape - Array shape (e.g., [2096, 4])
   * @param chunks - Chunk sizes (e.g., [1024, 4])
   */
  registerArrayBounds(arrayPath: string, shape: number[], chunks: number[]): void {
    const maxIndices = shape.map((s, i) => Math.ceil(s / chunks[i]));
    // Normalize: register both with and without leading slash for robust lookup
    // (zarrita keys have leading '/', loader paths may not)
    const normalized = arrayPath.replace(/^\/+/, '');
    this.maxChunkIndices.set(normalized, maxIndices);
    if (normalized !== arrayPath) {
      this.maxChunkIndices.set(arrayPath, maxIndices);
    }
  }

  /**
   * Generate adjacent chunk keys (±1 in each dimension).
   *
   * @example
   * getAdjacentChunks('points/positions/1.2.3')
   * → ['points/positions/0.2.3', 'points/positions/2.2.3',
   *    'points/positions/1.1.3', 'points/positions/1.3.3',
   *    'points/positions/1.2.2', 'points/positions/1.2.4']
   */
  private getAdjacentChunks(key: string): string[] {
    const indices = this.parseChunkIndices(key);
    if (!indices) return [];

    // Extract base path (everything before the indices)
    const isV3 = key.includes('/c/');
    const basePath = isV3 ? key.replace(/\/c\/[\d/]+$/, '') : key.replace(/\/[\d.]+$/, '');

    // Normalize: strip leading slash for consistent lookup
    // (zarrita resolves paths with leading '/', but registerArrayBounds strips it)
    const normalizedBasePath = basePath.replace(/^\/+/, '');

    // Look up upper bounds for this array (if registered)
    // Try both with and without leading slash for robustness
    const maxIndices =
      this.maxChunkIndices.get(normalizedBasePath) ?? this.maxChunkIndices.get(basePath);

    // Debug logging for troubleshooting
    if (this.debug) {
      this.log(`Debug - key: ${key}`);
      this.log(`Debug - indices: [${indices.join(', ')}]`);
      this.log(`Debug - isV3: ${isV3}, basePath: ${basePath}`);
    }

    const adjacent: string[] = [];

    // Without registered bounds we cannot safely determine which neighbors exist.
    // Prefetching blindly would produce out-of-bounds 404s for small arrays
    // (e.g. chunk_bounds with a single chunk in every dimension).
    if (!maxIndices) return adjacent;

    for (let dim = 0; dim < indices.length; dim++) {
      // Skip dimensions with only 1 chunk — no neighbors to prefetch
      if (dim < maxIndices.length && maxIndices[dim] <= 1) continue;

      for (const delta of [-1, 1]) {
        const newIndices = [...indices];
        newIndices[dim] += delta;

        // Skip negative indices (lower bounds)
        if (newIndices[dim] < 0) continue;

        // Skip indices beyond array bounds (upper bounds)
        if (dim < maxIndices.length && newIndices[dim] >= maxIndices[dim]) continue;

        // Generate key in same format as input
        const indexStr = isV3 ? 'c/' + newIndices.join('/') : newIndices.join('.');
        const adjacentKey = `${basePath}/${indexStr}`;

        if (this.debug) {
          this.log(`Debug - dim ${dim}, delta ${delta}: ${adjacentKey}`);
        }

        adjacent.push(adjacentKey);
      }
    }

    return adjacent;
  }

  /**
   * Get prefetch statistics.
   */
  getStats(): { queued: number; inFlight: number; enabled: boolean } {
    return {
      queued: this.queue.size,
      inFlight: this.inFlight.size,
      enabled: this.enabled,
    };
  }

  /**
   * Conditional logging based on debug mode.
   */
  private log(message: string): void {
    if (this.debug) {
      log.custom(LogEmoji.NETWORK, Modules.CACHE, `Prefetch: ${message}`);
    }
  }
}
