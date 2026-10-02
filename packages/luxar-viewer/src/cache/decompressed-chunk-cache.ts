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
 * A decoded chunk as returned by zarrita's `getChunk` (the value an in-flight
 * decode settles with). Typed loosely here because the cache is not generic
 * over the zarr dtype; the proxy narrows it back.
 */
export interface DecodedChunkResult {
  data: unknown;
  shape: number[];
  stride: number[];
}

/**
 * One fetch + decode in flight for an L0 key, shared by EVERY proxy wrapping an
 * array over this cache (see `cached-zarr-array.ts`).
 *
 * The decode runs under its own {@link controller}, never under any single
 * caller's signal: it is aborted only once every waiter has abandoned it, so a
 * superseded caller cannot cancel work a live caller still awaits.
 */
export interface InflightDecode {
  /** Settles with the decoded chunk (or the underlying error). */
  readonly promise: Promise<DecodedChunkResult>;
  /** Owns the underlying fetch/decode; aborted when {@link waiters} hits 0. */
  readonly controller: AbortController;
  /** Callers still waiting on {@link promise} (not yet aborted). */
  waiters: number;
  /**
   * Every waiter left and {@link controller} was aborted. The entry STAYS
   * registered until it settles: an abort that lands after the chunk's bytes
   * arrived cannot stop the decode, which then completes and commits, and a
   * caller arriving meanwhile waits for that outcome instead of starting a
   * second fetch + decode of the same chunk (it retries only if the abandoned
   * decode really was cancelled).
   */
  abandoned?: boolean;
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
  /**
   * In-flight decodes keyed by the full L0 key (`makeKey(arrayPath, coords)`,
   * so the array identity is part of the key). Lives on the CACHE — not on a
   * proxy — so the foreground loaders and the SlicePrefetcher's shadow loaders,
   * which wrap the same arrays through different proxies, share one decode.
   */
  private readonly inflight = new Map<string, InflightDecode>();
  /** Bumped by every `clear()`; see {@link generation}. */
  private _generation = 0;
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

  /** The decode in flight for `key`, if any. */
  getInflight(key: string): InflightDecode | undefined {
    return this.inflight.get(key);
  }

  /** Register `entry` as THE in-flight decode for `key`. */
  setInflight(key: string, entry: InflightDecode): void {
    this.inflight.set(key, entry);
  }

  /**
   * Forget the in-flight decode for `key` — only if it is still `entry` (a
   * later decode for the same key may already have replaced an abandoned one).
   */
  deleteInflight(key: string, entry: InflightDecode): void {
    if (this.inflight.get(key) === entry) this.inflight.delete(key);
  }

  /**
   * Incremented by every {@link clear} (the invalidation / dispose path).
   * Holders of state derived from this cache's contents — e.g. the range
   * loader's memoised, L0-wrapped array_ref targets — compare it to know when
   * to re-derive.
   */
  get generation(): number {
    return this._generation;
  }

  /** Number of decodes currently in flight (diagnostics / tests). */
  get inflightCount(): number {
    return this.inflight.size;
  }

  /**
   * Clear all cached chunks. Also forgets the in-flight decodes: a decode
   * started before the clear still settles for its waiters, but no longer
   * commits its result (only a still-registered entry may `set`), and a new
   * request starts afresh rather than joining pre-clear work.
   */
  clear(): void {
    this.cache.clear();
    this.inflight.clear();
    this._generation++;

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
