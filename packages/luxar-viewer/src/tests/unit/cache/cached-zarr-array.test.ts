/**
 * Unit tests for cached-zarr-array proxy wrapper
 *
 * Tests the ES6 Proxy that intercepts zarr.Array.getChunk() calls
 * to add L0 decompressed chunk caching.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import {
  wrapWithCache,
  isCachedArray,
  unwrapCachedArray,
  cloneArrayBufferView,
} from '../../../cache/decompressed-chunk-cache/cached-zarr-array';
import { ResidencyAccumulator } from '../../../cache/residency-probe';

// Mock zarr types for testing
type MockChunk = {
  data: Float32Array;
  shape: number[];
  stride: number[];
};

// Create a mock zarr.Array-like object
function createMockZarrArray(getChunkImpl?: () => Promise<MockChunk>) {
  const defaultChunk: MockChunk = {
    data: new Float32Array([1, 2, 3, 4, 5, 6]),
    shape: [2, 3],
    stride: [3, 1],
  };

  return {
    dtype: 'float32',
    shape: [100, 3],
    chunks: [50, 3],
    attrs: {},
    getChunk: getChunkImpl || vi.fn().mockResolvedValue(defaultChunk),
    // Simulate other zarr.Array properties
    store: {},
    path: '/test',
  } as any;
}

describe('cached-zarr-array', () => {
  let cache: DecompressedChunkCache;

  beforeEach(() => {
    cache = new DecompressedChunkCache({ maxSize: 1024 * 1024 });
  });

  describe('wrapWithCache', () => {
    it('should return a proxy that intercepts getChunk calls', async () => {
      const mockArray = createMockZarrArray();
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions');

      // Call getChunk
      const result = await wrapped.getChunk([0, 1, 2]);

      // Should have called original getChunk. The proxy forwards args
      // via rest-spread (`target.getChunk(...args)`), so a no-options
      // call shows up as a single positional arg, not `(coords, undefined)`.
      expect(mockArray.getChunk).toHaveBeenCalledWith([0, 1, 2]);

      // Should return the chunk data
      expect(result.data).toBeInstanceOf(Float32Array);
      expect(result.shape).toEqual([2, 3]);
    });

    it('should cache chunks on first access', async () => {
      const mockArray = createMockZarrArray();
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions');

      // First call - cache miss
      await wrapped.getChunk([0]);
      expect(mockArray.getChunk).toHaveBeenCalledTimes(1);

      // Second call - cache hit
      await wrapped.getChunk([0]);
      expect(mockArray.getChunk).toHaveBeenCalledTimes(1); // Still 1 - didn't call original

      const stats = cache.getStats();
      expect(stats.hits).toBe(1);
      expect(stats.misses).toBe(1);
    });

    it('should pass through other properties unchanged', () => {
      const mockArray = createMockZarrArray();
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions');

      // These should pass through to the original array
      expect(wrapped.dtype).toBe('float32');
      expect(wrapped.shape).toEqual([100, 3]);
      expect(wrapped.chunks).toEqual([50, 3]);
    });

    // workers.md O3 / cache.md G13 [P5]: audit-id moved to comment per Phase E54.
    it('passes through every direct-access property in the proxy short-list', () => {
      // [cache.md/G13][P5] The proxy lists 12 properties for direct access
      // (cached-zarr-array.ts:109-124): attrs, shape, dtype, chunks, order,
      // fill_value, fillValue, dimensionNames, compressor, filters, codec,
      // codecs. Pre-audit only 4 were exercised; the other 8 were silently
      // uncovered — a regression dropping any of them from the list would
      // fall through to the generic Reflect.get path and crash on zarrita's
      // private-field getters. Pin all 12 here.
      const sentinel = {
        attrs: { foo: 'bar' },
        shape: [10, 20],
        dtype: 'int32',
        chunks: [5, 10],
        order: 'C',
        fill_value: 0,
        fillValue: 0,
        dimensionNames: ['z', 'y'],
        compressor: { id: 'blosc' },
        filters: [{ id: 'delta' }],
        codec: { name: 'zstd' },
        codecs: [{ name: 'zstd' }],
        getChunk: vi.fn(),
      };
      const wrapped = wrapWithCache(sentinel as any, cache, '/sentinel') as unknown as Record<
        string,
        unknown
      >;

      // Each property must round-trip *by reference* (proxy reads the same
      // value object directly off `target`, not via Reflect.get).
      expect(wrapped.attrs).toBe(sentinel.attrs);
      expect(wrapped.shape).toBe(sentinel.shape);
      expect(wrapped.dtype).toBe(sentinel.dtype);
      expect(wrapped.chunks).toBe(sentinel.chunks);
      expect(wrapped.order).toBe(sentinel.order);
      expect(wrapped.fill_value).toBe(sentinel.fill_value);
      expect(wrapped.fillValue).toBe(sentinel.fillValue);
      expect(wrapped.dimensionNames).toBe(sentinel.dimensionNames);
      expect(wrapped.compressor).toBe(sentinel.compressor);
      expect(wrapped.filters).toBe(sentinel.filters);
      expect(wrapped.codec).toBe(sentinel.codec);
      expect(wrapped.codecs).toBe(sentinel.codecs);
    });

    it('should handle different chunk coordinates as separate cache entries', async () => {
      const mockArray = createMockZarrArray();
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions');

      // Access different chunks
      await wrapped.getChunk([0, 0]);
      await wrapped.getChunk([0, 1]);
      await wrapped.getChunk([1, 0]);

      // All should be cache misses (original called 3 times)
      expect(mockArray.getChunk).toHaveBeenCalledTimes(3);

      // Now access same chunks again - all should be hits
      await wrapped.getChunk([0, 0]);
      await wrapped.getChunk([0, 1]);
      await wrapped.getChunk([1, 0]);

      // Should still be 3 (all hits)
      expect(mockArray.getChunk).toHaveBeenCalledTimes(3);

      const stats = cache.getStats();
      expect(stats.hits).toBe(3);
      expect(stats.misses).toBe(3);
    });

    it('should not double-wrap an already wrapped array', () => {
      const mockArray = createMockZarrArray();
      const wrapped1 = wrapWithCache(mockArray, cache, '/points/positions');
      const wrapped2 = wrapWithCache(wrapped1, cache, '/points/positions');

      // Should be the same proxy (not double-wrapped)
      expect(wrapped1).toBe(wrapped2);
    });
  });

  describe('isCachedArray', () => {
    it('should return true for wrapped arrays', () => {
      const mockArray = createMockZarrArray();
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions');

      expect(isCachedArray(wrapped)).toBe(true);
    });

    it('should return false for unwrapped arrays', () => {
      const mockArray = createMockZarrArray();
      expect(isCachedArray(mockArray)).toBe(false);
    });

    it('should return false for null/undefined', () => {
      expect(isCachedArray(null)).toBe(false);
      expect(isCachedArray(undefined)).toBe(false);
    });

    it('should return false for non-objects', () => {
      expect(isCachedArray(42)).toBe(false);
      expect(isCachedArray('string')).toBe(false);
    });
  });

  describe('unwrapCachedArray', () => {
    it('should return the original array from a wrapped proxy', () => {
      const mockArray = createMockZarrArray();
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions');
      const unwrapped = unwrapCachedArray(wrapped);

      // Should be the same reference as the original
      expect(unwrapped).toBe(mockArray);
    });

    it('should return the same array if not wrapped', () => {
      const mockArray = createMockZarrArray();
      const result = unwrapCachedArray(mockArray);

      expect(result).toBe(mockArray);
    });
  });

  describe('Cache Key Generation', () => {
    it('should use correct path-based cache keys', async () => {
      const mockArray = createMockZarrArray();
      const wrapped = wrapWithCache(mockArray, cache, '/scene/points/positions');

      await wrapped.getChunk([1, 2, 3]);

      // Check the cache directly for the expected key
      const expectedKey = '/scene/points/positions:1,2,3';
      expect(cache.has(expectedKey)).toBe(true);
    });

    it('should isolate different arrays in the same cache', async () => {
      const mockArray1 = createMockZarrArray();
      const mockArray2 = createMockZarrArray();

      const wrapped1 = wrapWithCache(mockArray1, cache, '/points/positions');
      const wrapped2 = wrapWithCache(mockArray2, cache, '/points/colors');

      // Access same chunk coords but from different arrays
      await wrapped1.getChunk([0]);
      await wrapped2.getChunk([0]);

      // Both should be cache misses (different keys)
      expect(mockArray1.getChunk).toHaveBeenCalledTimes(1);
      expect(mockArray2.getChunk).toHaveBeenCalledTimes(1);

      const stats = cache.getStats();
      expect(stats.misses).toBe(2);
      expect(stats.count).toBe(2);
    });
  });

  describe('cloneArrayBufferView (commit 5.3)', () => {
    it('clones a Float32Array into a fresh buffer', () => {
      const src = new Float32Array([1.5, 2.5, 3.5]);
      const cloned = cloneArrayBufferView(src) as Float32Array;
      expect(cloned).toBeInstanceOf(Float32Array);
      expect(Array.from(cloned)).toEqual([1.5, 2.5, 3.5]);
      // Mutating the clone must not affect the source.
      cloned[0] = 99;
      expect(src[0]).toBe(1.5);
    });

    it('clones a Uint8Array into a fresh buffer', () => {
      const src = new Uint8Array([10, 20, 30]);
      const cloned = cloneArrayBufferView(src) as Uint8Array;
      expect(cloned).toBeInstanceOf(Uint8Array);
      expect(Array.from(cloned)).toEqual([10, 20, 30]);
      cloned[0] = 200;
      expect(src[0]).toBe(10);
    });

    it('clones a Uint16Array into a fresh buffer', () => {
      const src = new Uint16Array([1000, 2000]);
      const cloned = cloneArrayBufferView(src) as Uint16Array;
      expect(cloned).toBeInstanceOf(Uint16Array);
      expect(Array.from(cloned)).toEqual([1000, 2000]);
      cloned[1] = 60_000;
      expect(src[1]).toBe(2000);
    });

    it('clones a DataView into a fresh buffer', () => {
      const buffer = new ArrayBuffer(8);
      const src = new DataView(buffer);
      src.setUint32(0, 0xdeadbeef, true);
      const cloned = cloneArrayBufferView(src) as DataView;
      expect(cloned).toBeInstanceOf(DataView);
      expect(cloned.getUint32(0, true)).toBe(0xdeadbeef);
      // Mutating the clone must not affect the source.
      cloned.setUint32(0, 0x00000000, true);
      expect(src.getUint32(0, true)).toBe(0xdeadbeef);
    });
  });

  describe('Decode coalescing (commit 5.2)', () => {
    it('two concurrent same-key getChunk calls invoke underlying decode once', async () => {
      let resolveDecode: (chunk: MockChunk) => void = () => {};
      const decodeFn = vi.fn(
        () =>
          new Promise<MockChunk>((resolve) => {
            resolveDecode = resolve;
          })
      );
      const slow = createMockZarrArray(decodeFn as any);
      const wrapped = wrapWithCache(slow, cache, '/points/positions');

      const p1 = wrapped.getChunk([0]);
      const p2 = wrapped.getChunk([0]);
      // Yield once so both callers reach the pending-chunk check.
      await new Promise((r) => setTimeout(r, 0));
      // Underlying getChunk has been called at most once at this point.
      expect(decodeFn.mock.calls.length).toBe(1);

      resolveDecode({
        data: new Float32Array([1, 2, 3]),
        shape: [3],
        stride: [1],
      });
      const [r1, r2] = await Promise.all([p1, p2]);
      // Both callers see the same data; underlying decode ran exactly once.
      expect(r1.data).toBe(r2.data);
      expect(decodeFn.mock.calls.length).toBe(1);
    });

    it('errors clear the pending entry; subsequent call retries', async () => {
      let firstAttempt = true;
      const failingThenOk = createMockZarrArray(async () => {
        if (firstAttempt) {
          firstAttempt = false;
          throw new Error('transient decode error');
        }
        return {
          data: new Float32Array([9, 9, 9]),
          shape: [3],
          stride: [1],
        };
      });
      const wrapped = wrapWithCache(failingThenOk, cache, '/points/values');

      await expect(wrapped.getChunk([0])).rejects.toThrow('transient decode error');
      // Subsequent call must NOT see the rejected pending entry — it
      // should re-call target.getChunk and succeed.
      const result = await wrapped.getChunk([0]);
      expect(Array.from(result.data)).toEqual([9, 9, 9]);
    });
  });

  describe('Error Handling', () => {
    it('should propagate errors from original getChunk', async () => {
      const error = new Error('Network error');
      const mockArray = createMockZarrArray(() => Promise.reject(error));
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions');

      await expect(wrapped.getChunk([0])).rejects.toThrow('Network error');

      // Error case should not be cached
      const stats = cache.getStats();
      expect(stats.count).toBe(0);
    });
  });

  describe('Chunk Data Integrity', () => {
    it('should return identical data from cache as from original', async () => {
      const originalData = new Float32Array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
      const mockArray = createMockZarrArray(() =>
        Promise.resolve({
          data: originalData,
          shape: [2, 3],
          stride: [3, 1],
        })
      );

      const wrapped = wrapWithCache(mockArray, cache, '/points/positions');

      // First call (cache miss)
      const result1 = await wrapped.getChunk([0]);

      // Second call (cache hit)
      const result2 = await wrapped.getChunk([0]);

      // Data should be identical
      expect(result1.data).toEqual(originalData);
      expect(result2.data).toEqual(originalData);
      expect(result1.shape).toEqual(result2.shape);
      expect(result1.stride).toEqual(result2.stride);
    });
  });

  describe('Private Field Compatibility', () => {
    it('should work with objects that have getters accessing internal state', () => {
      // This tests the fix for: "Cannot read private member #e from an object whose class did not declare it"
      // zarrita uses private fields (#e, #store, etc.) and getters that access them.
      // The proxy must use `target` as receiver in Reflect.get, not `receiver` (the proxy).

      // Create a mock that simulates zarrita's internal structure with a getter
      // that relies on `this` being the original object
      const internalState = { value: 42 };
      const mockArray = {
        dtype: 'float32',
        shape: [100, 3],
        chunks: [50, 3],
        // Getter that relies on correct `this` binding
        get attrs() {
          // In real zarrita, this would access private fields like `this.#e`
          // If `this` is the proxy instead of the original object, it would fail
          return { internalValue: internalState.value };
        },
        getChunk: vi.fn().mockResolvedValue({
          data: new Float32Array([1, 2, 3]),
          shape: [3],
          stride: [1],
        }),
        store: {},
        path: '/test',
      } as any;

      const wrapped = wrapWithCache(mockArray, cache, '/points/positions');

      // This should NOT throw "Cannot read private member" error
      // If the proxy used `receiver` instead of `target`, this would fail
      expect(() => wrapped.attrs).not.toThrow();
      expect(wrapped.attrs).toEqual({ internalValue: 42 });
    });

    it('should correctly proxy shape getter (simulates zarrita #e access)', () => {
      // zarrita's shape getter accesses private field #e
      const mockArray = {
        dtype: 'float32',
        _internalShape: [100, 3], // Simulates private state
        get shape() {
          return this._internalShape; // Relies on correct `this`
        },
        chunks: [50, 3],
        attrs: {},
        getChunk: vi.fn().mockResolvedValue({
          data: new Float32Array([1, 2, 3]),
          shape: [3],
          stride: [1],
        }),
        store: {},
        path: '/test',
      } as any;

      const wrapped = wrapWithCache(mockArray, cache, '/points/positions');

      // This tests that the getter works correctly through the proxy
      expect(wrapped.shape).toEqual([100, 3]);
    });
  });

  describe('residency probe', () => {
    it('records a miss on cold access then a hit when warm', async () => {
      const mockArray = createMockZarrArray();
      const probe = new ResidencyAccumulator();
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions', () => probe);

      // Cold: triggers a real getChunk → miss.
      await wrapped.getChunk([0, 0, 0]);
      expect(probe.misses).toBe(1);
      expect(probe.hits).toBe(0);
      expect(probe.allResident).toBe(false);

      // Warm: same chunk now resident → hit.
      await wrapped.getChunk([0, 0, 0]);
      expect(probe.hits).toBe(1);
      expect(probe.misses).toBe(1);
    });

    it('treats a coalesced concurrent access as a hit', async () => {
      // A slow getChunk so the second call coalesces onto the first's promise.
      let resolveChunk: (c: MockChunk) => void = () => {};
      const slow = vi
        .fn()
        .mockImplementation(() => new Promise<MockChunk>((res) => (resolveChunk = res)));
      const mockArray = createMockZarrArray(slow);
      const probe = new ResidencyAccumulator();
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions', () => probe);

      const p1 = wrapped.getChunk([0, 0, 0]); // miss → starts decode
      const p2 = wrapped.getChunk([0, 0, 0]); // coalesces → hit
      resolveChunk({ data: new Float32Array([1, 2, 3]), shape: [3], stride: [1] });
      await Promise.all([p1, p2]);

      expect(slow).toHaveBeenCalledTimes(1); // single underlying decode
      expect(probe.misses).toBe(1);
      expect(probe.hits).toBe(1); // the coalesced caller
    });

    it('does not record when the probe accessor returns null', async () => {
      const mockArray = createMockZarrArray();
      const probe = new ResidencyAccumulator();
      // Accessor returns null (e.g. prefetch traffic / no active load).
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions', () => null);

      await wrapped.getChunk([0, 0, 0]);
      expect(probe.touched).toBe(false);
    });
  });

  describe('per-update abort chokepoint (getSignal)', () => {
    it('throws (and never fetches) when the active signal is already aborted — miss path', async () => {
      const mockArray = createMockZarrArray();
      const ac = new AbortController();
      ac.abort();
      const wrapped = wrapWithCache(
        mockArray,
        cache,
        '/points/positions',
        undefined,
        () => ac.signal
      );

      await expect(wrapped.getChunk([0, 0, 0])).rejects.toThrow();
      // Bailed at the proxy entry — the underlying (Blosc) getChunk never ran.
      expect(mockArray.getChunk).not.toHaveBeenCalled();
    });

    it('throws on the warm-cache HIT path too (chokepoint precedes the L0 lookup)', async () => {
      const mockArray = createMockZarrArray();
      let signal: AbortSignal | null = null;
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions', undefined, () => signal);

      // Prime L0 with a non-aborted call (same coords → same key).
      await wrapped.getChunk([0, 0, 0]);

      // Now abort and re-request the cached chunk: must still throw.
      const ac = new AbortController();
      ac.abort();
      signal = ac.signal;
      await expect(wrapped.getChunk([0, 0, 0])).rejects.toThrow();
    });

    it('does NOT throw when a signal is present but not aborted', async () => {
      const mockArray = createMockZarrArray();
      const ac = new AbortController(); // live
      const wrapped = wrapWithCache(
        mockArray,
        cache,
        '/points/positions',
        undefined,
        () => ac.signal
      );

      await expect(wrapped.getChunk([0, 0, 0])).resolves.toBeDefined();
      expect(mockArray.getChunk).toHaveBeenCalledTimes(1);
    });

    it('ignores a missing getSignal thunk (back-compat with 4-arg callers)', async () => {
      const mockArray = createMockZarrArray();
      const wrapped = wrapWithCache(mockArray, cache, '/points/positions');
      await expect(wrapped.getChunk([0, 0, 0])).resolves.toBeDefined();
    });
  });
});

describe('ResidencyAccumulator', () => {
  it('reports allResident only when no misses recorded', () => {
    const acc = new ResidencyAccumulator();
    expect(acc.allResident).toBe(true); // nothing touched → resident
    expect(acc.touched).toBe(false);
    acc.record(true);
    expect(acc.allResident).toBe(true);
    expect(acc.touched).toBe(true);
    acc.record(false);
    expect(acc.allResident).toBe(false);
  });
});
