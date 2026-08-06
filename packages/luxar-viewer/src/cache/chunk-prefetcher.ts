import type { MultiLevelCachingStore } from './multi-level-caching-store';
import { log, Modules, LogEmoji } from '../utils/log';

export interface ChunkPrefetcherOptions {
  /** Maximum concurrent prefetch requests (default: 4) */
  maxConcurrent?: number;

  /** Enable/disable prefetching (default: true) */
  enabled?: boolean;

  /** Enable debug logging (default: false) */
  debug?: boolean;
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
  private store: MultiLevelCachingStore;
  private maxConcurrent: number;
  private enabled: boolean;
  private debug: boolean;

  // Queue management. A single FIFO queue (the Set preserves insertion
  // order) drained under the concurrency limit each processQueue cycle.
  private inFlight = new Set<string>();
  private normalQueue = new Set<string>();
  private processing = false;
  // Lifecycle: set by dispose(). Distinct from `enabled` (a feature toggle)
  // so we can short-circuit the in-flight `.finally()` continuation
  // without claiming the feature itself is disabled.
  private isDisposed = false;

  // Tracks all keys whose neighbors have already been enqueued, preventing
  // cascading prefetch amplification: without this, prefetched chunks trigger
  // their own neighbor prefetches, which cascade across the entire dataset.
  // Bounded to prevent unbounded memory growth during long browsing sessions.
  private static readonly MAX_SEEN_SIZE = 10000;
  private seen = new Set<string>();

  /** Cache for parsed chunk indices (avoids repeated regex parsing per access) */
  private parsedCache = new Map<string, number[] | null>();

  /** Upper bounds per array path for suppressing out-of-range prefetch requests */
  private maxChunkIndices = new Map<string, number[]>();

  constructor(store: MultiLevelCachingStore, options?: ChunkPrefetcherOptions) {
    this.store = store;
    this.maxConcurrent = options?.maxConcurrent ?? 4;
    this.enabled = options?.enabled ?? true;
    this.debug = options?.debug ?? false;

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

    // Prevent unbounded memory growth — evict oldest half when limit reached.
    // Do NOT clear maxChunkIndices: they're registered once per array and losing
    // them causes out-of-range prefetch requests (404s) until arrays re-register.
    if (this.seen.size > ChunkPrefetcher.MAX_SEEN_SIZE) {
      const evictCount = Math.floor(ChunkPrefetcher.MAX_SEEN_SIZE / 2);
      let count = 0;
      for (const k of this.seen) {
        if (count++ >= evictCount) break;
        this.seen.delete(k);
      }
      // Trim parsed cache alongside seen eviction. Half-trim (FIFO,
      // matches Map insertion order) instead of clear() so a single
      // navigation burst doesn't evict every parsed key — the regex
      // parse cost recurs for whichever keys come back hot.
      if (this.parsedCache.size > ChunkPrefetcher.MAX_SEEN_SIZE * 2) {
        const trimCount = Math.floor(this.parsedCache.size / 2);
        let trimmed = 0;
        for (const k of this.parsedCache.keys()) {
          if (trimmed++ >= trimCount) break;
          this.parsedCache.delete(k);
        }
      }
    }

    const adjacent = this.getAdjacentChunks(key);
    if (adjacent.length === 0) {
      // Not a chunk file (metadata), skip prefetching
      return;
    }

    this.log(`Access: ${key} → ${adjacent.length} adjacent chunks`);

    for (const adjKey of adjacent) {
      this.addToQueue(adjKey);
    }

    // Fire-and-forget (not awaited)
    this.processQueue();
  }

  /**
   * Add a key to the queue. Returns true if the key was added (i.e.
   * not already queued or in-flight).
   */
  private addToQueue(key: string): boolean {
    if (this.inFlight.has(key)) return false;
    if (this.normalQueue.has(key)) return false;
    this.normalQueue.add(key);
    this.log(`  Enqueued: ${key}`);
    return true;
  }

  /**
   * Process the prefetch queue with concurrency limiting.
   * Race-condition safe via processing flag and queueMicrotask re-trigger.
   */
  private async processQueue(): Promise<void> {
    // Disposed-prefetcher short-circuit: bail before any further fetch
    // dispatch so a dispose() during in-flight processing cannot enqueue
    // additional store.getResult calls.
    if (this.isDisposed) return;
    // Prevent concurrent processing
    if (this.processing) return;
    this.processing = true;

    try {
      while (this.normalQueue.size > 0 && this.inFlight.size < this.maxConcurrent) {
        if (this.isDisposed) break;
        // Set insertion order gives FIFO semantics.
        const key = this.normalQueue.values().next().value;
        if (!key) break;

        this.normalQueue.delete(key);
        this.inFlight.add(key);

        this.log(`Prefetching: ${key} (${this.inFlight.size}/${this.maxConcurrent} slots)`);

        // Fire-and-forget with cleanup. Use getResult so a transient
        // network failure surfaces in the prefetch log instead of being
        // silently indistinguishable from a 404. Suppress
        // prefetcher.onAccess() inside getResult — without this flag,
        // a prefetch of K+1 would call onAccess(K+1) and enqueue K+2,
        // K+3, ... cascading until MAX_SEEN_SIZE bounds it.
        this.store
          .getResult(key, { suppressPrefetch: true })
          .then((r) => {
            if (!r.ok && r.error.kind === 'NetworkError') {
              this.log(`Prefetch network error: ${key} (${r.error.cause.message})`);
            }
          })
          .finally(() => {
            this.inFlight.delete(key);
            // Trigger queue processing when slot frees up — but only if
            // the prefetcher is still alive. Without this guard, a fetch
            // that started before dispose() can keep re-entering
            // processQueue and dispatching additional fetches against
            // the (now-disposed) store.
            if (!this.isDisposed && this.normalQueue.size > 0) {
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
    const cached = this.parsedCache.get(key);
    if (cached !== undefined) return cached;

    // CRITICAL: Check v3 FIRST before v2!
    // v2 regex can match the trailing digits of v3 paths (e.g., /2 in /c/0/1/2)
    let result: number[] | null = null;

    // Check for v3 path notation: /c/ followed by path segments
    const v3Match = key.match(/\/c\/(\d+(?:\/\d+)*)$/);
    if (v3Match) {
      result = v3Match[1].split('/').map(Number);
    } else {
      // Check for v2 dot notation: ends with digits separated by dots
      const v2Match = key.match(/\/(\d+(?:\.\d+)*)$/);
      if (v2Match) {
        result = v2Match[1].split('.').map(Number);
      }
    }

    this.parsedCache.set(key, result);
    return result;
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
    // Bounds registration is a BEST-EFFORT prefetch optimization: when present
    // it lets getAdjacentChunks skip out-of-range indices (avoiding spurious
    // 404s). It must NEVER abort a load. If shape/chunks are absent or malformed
    // (a partial/streaming array, an unusual store, mismatched ranks, or a zero
    // chunk size that would divide to Infinity), skip registration silently —
    // the prefetcher simply runs without bounds.
    if (
      !Array.isArray(shape) ||
      !Array.isArray(chunks) ||
      shape.length === 0 ||
      shape.length !== chunks.length ||
      chunks.some((c) => !Number.isFinite(c) || c <= 0)
    ) {
      return;
    }
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

    // Extract base path (everything before the indices).
    // [cache OOS] V3 detection uses an anchored regex so a key whose
    // base path itself contains `/c/` (e.g.
    // `dataset/scenes/cleanup/positions/1.2.3`) is not misclassified
    // as v3. Zarr v3 chunk keys end with `/c/<index>(/<index>)*`, so
    // anchor the pattern to end-of-key. Production keys don't currently
    // hit the pathological case, but the anchored regex defends against
    // future naming conventions.
    const v3ChunkPattern = /\/c\/\d+(\/\d+)*$/;
    const isV3 = v3ChunkPattern.test(key);
    const basePath = isV3 ? key.replace(v3ChunkPattern, '') : key.replace(/\/[\d.]+$/, '');

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
   * Dispose the prefetcher, clearing all internal state and stopping processing.
   */
  dispose(): void {
    this.isDisposed = true;
    this.enabled = false;
    this.seen.clear();
    this.parsedCache.clear();
    this.maxChunkIndices.clear();
    this.normalQueue.clear();
    this.inFlight.clear();
    this.processing = false;
  }

  /**
   * Get prefetch statistics.
   */
  getStats(): {
    queued: number;
    inFlight: number;
    enabled: boolean;
  } {
    return {
      queued: this.normalQueue.size,
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
