/**
 * L0 Decompressed Chunk Cache - Caches decoded zarr chunks to avoid repeated Blosc decompression.
 *
 * This cache sits above the L1/L2 compressed caches and stores the OUTPUT of zarrita's
 * codec.decode() - fully decompressed TypedArrays ready for use.
 *
 * Performance impact:
 * - L1 hit without L0: ~1μs lookup + ~2ms decompression = ~2ms
 * - L0 hit: ~1μs lookup + 0ms decompression = ~1μs (2000x faster)
 *
 * For 4 attributes (positions, colors, radii, sharpness):
 * - Without L0: ~8ms decompression overhead per view update
 * - With L0: ~0.004ms (essentially zero)
 *
 * @module cache/decompressed-chunk-cache
 */

import { LRUCache } from './lru-cache';
import { log, Modules, LogEmoji } from '../utils/log';
import { config } from '../config';

/**
 * Cached decompressed chunk structure.
 * Mirrors zarrita's Chunk<D> interface.
 */
export interface DecompressedChunk {
  /** The decompressed data (Float32Array, Uint8Array, etc.) */
  data: ArrayBufferView;
  /** Chunk dimensions */
  shape: number[];
  /** Memory layout strides */
  stride: number[];
}

/**
 * Configuration options for the decompressed chunk cache.
 */
export interface DecompressedChunkCacheOptions {
  /** Maximum cache size in bytes (default: 200MB) */
  maxSize?: number;
  /** Enable debug logging (default: false) */
  debug?: boolean;
}

/**
 * Cache statistics for monitoring.
 */
export interface DecompressedChunkCacheStats {
  /** Current cache size in bytes */
  size: number;
  /** Number of cached chunks */
  count: number;
  /** Total cache hits */
  hits: number;
  /** Total cache misses */
  misses: number;
  /** Total evicted chunks */
  evictions: number;
  /** Hit rate (0-1) */
  hitRate: number;
  /** Resolved byte budget (heap-aware; see `heap-budget.ts`). */
  maxSize?: number;
}

/**
 * L0 Decompressed Chunk Cache.
 *
 * Uses LRU eviction with byte-size tracking to cache decompressed zarr chunks.
 * Key format: `${arrayPath}:${chunkCoords.join(',')}` (e.g., "/points/positions:0,1,2")
 *
 * @example
 * ```typescript
 * const cache = new DecompressedChunkCache({ maxSize: 200 * 1024 * 1024 });
 *
 * // Cache a decompressed chunk
 * const key = DecompressedChunkCache.makeKey('/points/positions', [0, 1, 2]);
 * cache.set(key, { data: float32Array, shape: [1000, 3], stride: [3, 1] });
 *
 * // Retrieve cached chunk
 * const cached = cache.get(key);
 * if (cached) {
 *   console.log('L0 hit!', cached.data.byteLength, 'bytes');
 * }
 * ```
 */
export class DecompressedChunkCache {
  /** Default cache size derived from config.cache.l0MaxSizeMB */
  private static readonly DEFAULT_MAX_SIZE = config.cache.l0MaxSizeMB * 1024 * 1024;

  /** Metadata overhead estimate per chunk (shape array, stride array, object wrapper) */
  private static readonly METADATA_OVERHEAD = 64;

  private cache: LRUCache<DecompressedChunk>;
  private debug: boolean;
  /** Resolved byte budget — surfaced in getStats() for runtime introspection. */
  private readonly maxSize: number;

  constructor(options?: DecompressedChunkCacheOptions) {
    const maxSize = options?.maxSize ?? DecompressedChunkCache.DEFAULT_MAX_SIZE;
    this.maxSize = maxSize;
    this.debug = options?.debug ?? false;

    // Create LRU cache with byte-size tracking
    this.cache = new LRUCache<DecompressedChunk>(maxSize, (chunk) => {
      return chunk.data.byteLength + DecompressedChunkCache.METADATA_OVERHEAD;
    });

    if (this.debug) {
      log.custom(
        LogEmoji.CACHE,
        Modules.CACHE,
        `L0 initialized with max size: ${(maxSize / 1024 / 1024).toFixed(1)}MB`
      );
    }
  }

  /**
   * Get a cached decompressed chunk.
   *
   * @param key - Cache key (use `makeKey()` to generate)
   * @returns Cached chunk or undefined if not found
   */
  get(key: string): DecompressedChunk | undefined {
    const chunk = this.cache.get(key);

    if (this.debug) {
      if (chunk) {
        log.custom(
          LogEmoji.CACHE,
          Modules.CACHE,
          `L0 HIT: ${key} (${chunk.data.byteLength} bytes)`
        );
      } else {
        log.custom(LogEmoji.CACHE, Modules.CACHE, `L0 MISS: ${key}`);
      }
    }

    return chunk;
  }

  /**
   * Cache a decompressed chunk.
   *
   * @param key - Cache key (use `makeKey()` to generate)
   * @param chunk - Decompressed chunk to cache
   */
  set(key: string, chunk: DecompressedChunk): void {
    this.cache.set(key, chunk);

    if (this.debug) {
      log.custom(
        LogEmoji.CACHE,
        Modules.CACHE,
        `L0 SET: ${key} (${chunk.data.byteLength} bytes, ` +
          `total: ${(this.cache.size / 1024 / 1024).toFixed(1)}MB)`
      );
    }
  }

  /**
   * Check if a chunk is cached.
   *
   * @param key - Cache key to check
   * @returns true if chunk is in cache
   */
  has(key: string): boolean {
    return this.cache.has(key);
  }

  /**
   * Clear all cached chunks.
   */
  clear(): void {
    this.cache.clear();

    if (this.debug) {
      log.custom(LogEmoji.CACHE, Modules.CACHE, 'L0 cleared');
    }
  }

  /**
   * Dispose the cache, releasing all cached chunks.
   */
  dispose(): void {
    this.clear();
  }

  /**
   * Get cache statistics for monitoring.
   *
   * @returns Cache statistics including size, count, hits, misses, and hit rate
   */
  getStats(): DecompressedChunkCacheStats {
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
      maxSize: this.maxSize,
    };
  }

  /**
   * Generate a cache key from array path and chunk coordinates.
   *
   * @param arrayPath - Full path to the zarr array (e.g., "/points/positions")
   * @param chunkCoords - Chunk coordinates (e.g., [0, 1, 2])
   * @returns Cache key string
   *
   * @example
   * ```typescript
   * const key = DecompressedChunkCache.makeKey('/scene/points/positions', [0, 1, 2]);
   * // Returns: "/scene/points/positions:0,1,2"
   * ```
   */
  static makeKey(arrayPath: string, chunkCoords: number[]): string {
    return `${arrayPath}:${chunkCoords.join(',')}`;
  }

  /**
   * Parse a cache key back into array path and chunk coordinates.
   *
   * @param key - Cache key to parse
   * @returns Parsed components or null if invalid key format
   */
  static parseKey(key: string): { arrayPath: string; chunkCoords: number[] } | null {
    const colonIndex = key.lastIndexOf(':');
    if (colonIndex === -1) return null;

    const arrayPath = key.substring(0, colonIndex);
    const coordsStr = key.substring(colonIndex + 1);
    const chunkCoords = coordsStr.split(',').map(Number);

    if (chunkCoords.some(isNaN)) return null;

    return { arrayPath, chunkCoords };
  }
}
