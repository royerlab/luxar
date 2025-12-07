import type { TwoLevelCachingStore } from './two-level-caching-store';

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
    // IMPORTANT: Check v3 FIRST before v2!
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

    // Debug logging for troubleshooting
    if (this.debug) {
      console.log(`[Prefetch Debug] key: ${key}`);
      console.log(`[Prefetch Debug] indices: [${indices.join(', ')}]`);
      console.log(`[Prefetch Debug] isV3: ${isV3}, basePath: ${basePath}`);
    }

    const adjacent: string[] = [];

    for (let dim = 0; dim < indices.length; dim++) {
      for (const delta of [-1, 1]) {
        const newIndices = [...indices];
        newIndices[dim] += delta;

        // Skip negative indices
        if (newIndices[dim] < 0) continue;

        // Generate key in same format as input
        const indexStr = isV3 ? 'c/' + newIndices.join('/') : newIndices.join('.');
        const adjacentKey = `${basePath}/${indexStr}`;

        if (this.debug) {
          console.log(`[Prefetch Debug] dim ${dim}, delta ${delta}: ${adjacentKey}`);
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
      console.log(`[Prefetch] ${message}`);
    }
  }
}
