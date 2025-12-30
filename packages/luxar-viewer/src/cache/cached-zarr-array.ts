/**
 * Proxy wrapper for zarr.Array that adds L0 decompressed chunk caching.
 *
 * Uses ES6 Proxy to intercept `getChunk()` calls and check the L0 cache before
 * triggering Blosc decompression. This approach:
 * - Preserves full TypeScript type compatibility
 * - Works with any zarr.Array without modification
 * - Requires no changes to zarrita source code
 *
 * @module cache/cached-zarr-array
 */

import type * as zarr from 'zarrita';
import { DecompressedChunkCache, type DecompressedChunk } from './decompressed-chunk-cache';

/** Symbol to mark proxied arrays (for detection) */
const CACHE_MARKER = Symbol('luxar.l0cache');

/** Symbol to store original array reference */
const ORIGINAL_ARRAY = Symbol('luxar.originalArray');

/**
 * Wrap a zarr.Array with L0 decompressed chunk caching.
 *
 * The returned proxy intercepts `getChunk()` calls:
 * 1. Checks L0 cache for the requested chunk
 * 2. On hit: returns cached decompressed data immediately (~1μs)
 * 3. On miss: calls original `getChunk()`, caches result, returns data
 *
 * All other zarr.Array properties and methods pass through unchanged.
 *
 * @param array - Original zarr.Array to wrap
 * @param cache - L0 decompressed chunk cache instance
 * @param arrayPath - Full path to the array (used for cache key generation)
 * @returns Proxied zarr.Array with L0 caching enabled
 *
 * @example
 * ```typescript
 * const cache = new DecompressedChunkCache({ maxSize: 200 * 1024 * 1024 });
 *
 * // Wrap array after opening
 * const rawArray = await zarr.open(location.resolve('positions'), { kind: 'array' });
 * const cachedArray = wrapWithCache(rawArray, cache, '/scene/points/positions');
 *
 * // Use normally - getChunk() now checks L0 cache
 * const chunk = await cachedArray.getChunk([0, 1, 2]);
 * // First call: decompresses and caches
 * // Subsequent calls: returns from L0 cache instantly
 * ```
 */
export function wrapWithCache<D extends zarr.DataType>(
  array: zarr.Array<D, zarr.Readable>,
  cache: DecompressedChunkCache,
  arrayPath: string
): zarr.Array<D, zarr.Readable> {
  // Don't double-wrap
  if (isCachedArray(array)) {
    return array;
  }

  return new Proxy(array, {
    get(target, prop, _receiver) {
      // Handle cache marker check
      if (prop === CACHE_MARKER) {
        return true;
      }

      // Handle original array access
      if (prop === ORIGINAL_ARRAY) {
        return target;
      }

      // Intercept getChunk() to add caching
      if (prop === 'getChunk') {
        return async function (
          chunkCoords: number[],
          options?: Parameters<typeof target.getChunk>[1]
        ): Promise<{ data: zarr.TypedArray<D>; shape: number[]; stride: number[] }> {
          const key = DecompressedChunkCache.makeKey(arrayPath, chunkCoords);

          // Check L0 cache first
          const cached = cache.get(key);
          if (cached) {
            // Return cached decompressed chunk
            return {
              data: cached.data as zarr.TypedArray<D>,
              shape: cached.shape,
              stride: cached.stride,
            };
          }

          // Cache miss - call original getChunk (triggers Blosc decompression)
          const chunk = await target.getChunk(chunkCoords, options);

          // Cache the decompressed result
          const cacheEntry: DecompressedChunk = {
            data: chunk.data as ArrayBufferView,
            shape: chunk.shape,
            stride: chunk.stride,
          };
          cache.set(key, cacheEntry);

          return chunk;
        };
      }

      // Pass through all other property access
      // CRITICAL: Use `target` as receiver, NOT `receiver` (the proxy)!
      // zarrita uses private class fields (e.g., #e, #store). When getters
      // access private fields, `this` must be the original object, not the proxy.
      // Using `receiver` (the proxy) causes: "Cannot read private member #e"
      return Reflect.get(target, prop, target);
    },
  }) as zarr.Array<D, zarr.Readable>;
}

/**
 * Check if an array is already wrapped with L0 caching.
 *
 * @param array - Array to check
 * @returns true if array is wrapped with cache proxy
 */
export function isCachedArray(array: unknown): boolean {
  if (array === null || typeof array !== 'object') {
    return false;
  }
  return (array as Record<symbol, unknown>)[CACHE_MARKER] === true;
}

/**
 * Get the underlying unwrapped zarr.Array from a cached proxy.
 *
 * Useful when you need direct access to the original array without caching.
 *
 * @param array - Possibly cached zarr.Array
 * @returns Original unwrapped array, or the input if not cached
 */
export function unwrapCachedArray<D extends zarr.DataType>(
  array: zarr.Array<D, zarr.Readable>
): zarr.Array<D, zarr.Readable> {
  if (!isCachedArray(array)) {
    return array;
  }
  return (array as unknown as Record<symbol, unknown>)[ORIGINAL_ARRAY] as zarr.Array<
    D,
    zarr.Readable
  >;
}
