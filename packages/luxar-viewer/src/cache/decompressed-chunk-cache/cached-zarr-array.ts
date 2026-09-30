/**
 * Proxy wrapper for zarr.Array that adds L0 decompressed chunk caching.
 *
 * Uses ES6 Proxy to intercept `getChunk()` calls and check the L0 cache before
 * triggering Blosc decompression. This approach:
 * - Preserves full TypeScript type compatibility
 * - Works with any zarr.Array without modification
 * - Requires no changes to zarrita source code
 *
 * @module cache/decompressed-chunk-cache/cached-zarr-array
 */

import type * as zarr from '../../data/zarr';
import {
  DecompressedChunkCache,
  type DecompressedChunk,
  type InflightDecode,
} from '../decompressed-chunk-cache';
import type { ResidencyProbe } from '../residency-probe';

/**
 * Clone an ArrayBufferView by allocating a fresh underlying buffer.
 * Handles both TypedArray (Float32Array, Uint8Array, Uint16Array,
 * BigInt64Array, etc.) and DataView. TypedArrays expose `.slice()`;
 * DataView needs an explicit buffer slice to preserve byte offsets.
 *
 * Exported only for unit testing; not part of the public package
 * surface.
 *
 * @internal
 */
export function cloneArrayBufferView(view: ArrayBufferView): ArrayBufferView {
  if (view instanceof DataView) {
    const buffer = view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
    return new DataView(buffer);
  }
  // All TypedArray subtypes implement slice() returning their own subtype.
  return (view as ArrayBufferView & { slice(): ArrayBufferView }).slice();
}

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
 * 3. On in-flight hit: joins the cache-wide decode as an independent waiter
 * 4. On miss: calls original `getChunk()`, caches result, returns data
 *
 * All other zarr.Array properties and methods pass through unchanged.
 *
 * @param array - Original zarr.Array to wrap
 * @param cache - L0 decompressed chunk cache instance
 * @param arrayPath - Full path to the array (used for cache key generation)
 * @param getProbe - Optional accessor for the currently-active residency
 *   probe. Called on every `getChunk` to report hit/miss. Returning `null`
 *   (the default / when no load is in flight) disables reporting — this is
 *   how prefetch traffic is kept from contaminating a demand load's signal.
 * @param getSignal - Optional accessor for the currently-active per-update
 *   `AbortSignal`. Called on every `getChunk`; if it returns an aborted
 *   signal the call throws (`AbortError`) BEFORE any L0 lookup, cache fetch,
 *   or Blosc decode — so a superseded `updateView` bails on the warm-cache
 *   hit path too (zarrita's own `throwIfAborted` only fires between chunks
 *   of a multi-chunk selection). Mirrors the `getProbe` thunk lifetime: it
 *   reads the owning loader's transient per-update field, so it is naturally
 *   per-caller and never aborts a coalesced chunk another live caller awaits.
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
  arrayPath: string,
  getProbe?: () => ResidencyProbe | null,
  getSignal?: () => AbortSignal | null
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

      // CRITICAL: Direct access for properties that use private fields internally.
      // zarrita's getters (attrs, shape, dtype, etc.) access private fields like #metadata.
      // Even with Reflect.get(target, prop, target), the getter can fail because the
      // property descriptor is retrieved from the proxy, not the original object.
      // Solution: Access these properties directly on target, bypassing Reflect entirely.
      if (
        prop === 'attrs' ||
        prop === 'shape' ||
        prop === 'dtype' ||
        prop === 'chunks' ||
        prop === 'order' ||
        prop === 'fill_value' ||
        prop === 'fillValue' ||
        prop === 'dimensionNames' ||
        prop === 'compressor' ||
        prop === 'filters' ||
        prop === 'codec' ||
        prop === 'codecs'
      ) {
        return (target as unknown as Record<string, unknown>)[prop as string];
      }

      // Intercept getChunk() to add caching
      if (prop === 'getChunk') {
        return async function (
          ...args: Parameters<typeof target.getChunk>
        ): Promise<{ data: zarr.TypedArray<D>; shape: number[]; stride: number[] }> {
          const [chunkCoords, callOptions, opts] = args;

          // Per-update abort chokepoint: bail BEFORE the L0 lookup / fetch /
          // Blosc decode if the owning load was superseded. This covers the
          // warm-cache hit and coalesced-pending paths below, which
          // short-circuit before zarrita's between-chunk throwIfAborted would
          // run. `getSignal` reads the loader's transient per-update field, so
          // it is per-caller — it never aborts the shared `pendingChunks`
          // promise that a different, still-live caller may be awaiting.
          const activeSignal = getSignal?.() ?? undefined;
          activeSignal?.throwIfAborted();
          const waiterSignal = callOptions?.signal ?? activeSignal;
          waiterSignal?.throwIfAborted();

          const key = DecompressedChunkCache.makeKey(arrayPath, chunkCoords);

          // Check L0 cache first
          const cached = cache.get(key);
          if (cached) {
            // Resident: served from L0 with no fresh fetch/decode.
            getProbe?.()?.record(true);
            // Return cached decompressed chunk
            return {
              data: cached.data as zarr.TypedArray<D>,
              shape: cached.shape,
              stride: cached.stride,
            };
          }

          // Same-chunk coalescing: if another caller is already
          // decompressing this chunk, await their promise instead of
          // re-running Blosc. No fresh Blosc work is triggered for this
          // caller, so treat the coalesced wait as a hit for residency.
          const pending = cache.getInflight(key);
          if (pending) {
            getProbe?.()?.record(true);
            pending.waiters++;
            return (await waitForDecode(cache, key, pending, waiterSignal)) as {
              data: zarr.TypedArray<D>;
              shape: number[];
              stride: number[];
            };
          }

          // Genuine miss: a fetch + Blosc decode is about to run.
          getProbe?.()?.record(false);

          const entry: InflightDecode = {
            controller: new AbortController(),
            waiters: 1,
            promise: undefined as unknown as InflightDecode['promise'],
          };
          entry.promise = (async () => {
            // The decode belongs to the shared entry, never to its first caller.
            const chunk = await target.getChunk(
              chunkCoords,
              { ...callOptions, signal: entry.controller.signal },
              opts
            );
            if (cache.getInflight(key) === entry) {
              // Clone before storage: a caller may mutate its returned view.
              const cacheEntry: DecompressedChunk = {
                data: cloneArrayBufferView(chunk.data as unknown as ArrayBufferView),
                shape: chunk.shape.slice(),
                stride: chunk.stride.slice(),
              };
              cache.set(key, cacheEntry);
            }
            return chunk;
          })();
          cache.setInflight(key, entry);
          const forget = (): void => cache.deleteInflight(key, entry);
          void entry.promise.then(forget, forget);
          return (await waitForDecode(cache, key, entry, waiterSignal)) as {
            data: zarr.TypedArray<D>;
            shape: number[];
            stride: number[];
          };
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

/** Leave a shared decode on abort; cancel its source when the last waiter leaves. */
function waitForDecode(
  cache: DecompressedChunkCache,
  key: string,
  entry: InflightDecode,
  signal?: AbortSignal
): Promise<Awaited<InflightDecode['promise']>> {
  if (!signal) return entry.promise;
  return new Promise((resolve, reject) => {
    const onAbort = (): void => {
      entry.waiters--;
      if (entry.waiters === 0) {
        cache.deleteInflight(key, entry);
        entry.controller.abort(signal.reason);
      }
      reject(signal.reason as Error);
    };
    signal.addEventListener('abort', onAbort, { once: true });
    void entry.promise.then(
      (chunk) => {
        signal.removeEventListener('abort', onAbort);
        resolve(chunk);
      },
      (error: unknown) => {
        signal.removeEventListener('abort', onAbort);
        reject(error);
      }
    );
  });
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
