/**
 * Unit tests for the shared prefetch cache-warming helper.
 *
 * Contract: `prefetchRangesIntoCache` computes the DISTINCT chunk coordinates
 * each (array × range) touches and warms each chunk once through the L0 proxy's
 * `warmChunk` — decode + cache only. It never runs zarrita `get()` (which would
 * allocate and fill a full output selection only to discard it), never records
 * into the demand load's residency probe, and never reads the demand load's
 * abort signal (it uses its own, and counts its decodes as `prefetch`).
 *
 * This replaces the earlier "one get() per (array × range)" contract: that was
 * exactly the output-assembly waste (and the probe/signal contamination, since
 * `get()` routed through the foreground proxy's `getChunk` hooks) being fixed.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { prefetchRangesIntoCache } from '../../../../../data/loaders';
import { DecompressedChunkCache } from '../../../../../cache/decompressed-chunk-cache';
import {
  wrapWithCache,
  resetDecodeHistory,
} from '../../../../../cache/decompressed-chunk-cache/cached-zarr-array';
import { ResidencyAccumulator } from '../../../../../cache/residency-probe';
import { perfCounters } from '../../../../../profiling/perf-counters';

vi.mock('zarrita', async () => {
  const actual = await vi.importActual('zarrita');
  return {
    ...actual,
    get: vi.fn(),
    slice: (start: number | null, end?: number | null) => ({ start, end: end ?? null }),
  };
});
import { get as zarrGet } from 'zarrita';
const mockGet = vi.mocked(zarrGet);

/** A raw zarrita-array stand-in: shape/chunks + a counting getChunk. */
function rawArray(shape: number[], chunks: number[]) {
  const getChunk = vi.fn((coords: number[], _options?: { signal?: AbortSignal }) =>
    Promise.resolve({
      data: new Float32Array(chunks.reduce((a, b) => a * b, 1)).fill(coords[0]),
      shape: chunks,
      stride: [chunks[1] ?? 1, 1].slice(-chunks.length),
    })
  );
  return { shape, chunks, dtype: 'float32', attrs: {}, getChunk };
}

const coordsOf = (fn: ReturnType<typeof rawArray>['getChunk']): string[] =>
  fn.mock.calls.map((c) => c[0].join(',')).sort();

describe('prefetchRangesIntoCache', () => {
  let cache: DecompressedChunkCache;

  beforeEach(() => {
    mockGet.mockReset();
    mockGet.mockResolvedValue({ data: new Uint8Array() } as never);
    cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    perfCounters.reset();
    resetDecodeHistory();
  });

  it('warms L0 for each distinct chunk without calling get()', async () => {
    const raw = rawArray([10, 3], [4, 3]);
    const wrapped = wrapWithCache(raw as never, cache, '/n/centers');
    // Rows [0,2) and [1,3) are both in chunk 0; [5,6) is in chunk 1.
    await prefetchRangesIntoCache(
      [wrapped],
      [
        { start: 0, end: 2 },
        { start: 5, end: 6 },
        { start: 1, end: 3 },
      ]
    );
    expect(mockGet).not.toHaveBeenCalled();
    expect(coordsOf(raw.getChunk)).toEqual(['0,0', '1,0']);
    expect(cache.has(DecompressedChunkCache.makeKey('/n/centers', [0, 0]))).toBe(true);
    expect(cache.has(DecompressedChunkCache.makeKey('/n/centers', [1, 0]))).toBe(true);
    expect(cache.has(DecompressedChunkCache.makeKey('/n/centers', [2, 0]))).toBe(false);
  });

  it('covers every trailing-axis chunk and clamps the range to the array', async () => {
    const raw = rawArray([10, 6], [4, 2]);
    const wrapped = wrapWithCache(raw as never, cache, '/n/colors');
    await prefetchRangesIntoCache([wrapped], [{ start: 7, end: 50 }]);
    expect(coordsOf(raw.getChunk)).toEqual(['1,0', '1,1', '1,2', '2,0', '2,1', '2,2']);
  });

  it('records nothing in the demand residency probe and ignores the demand signal', async () => {
    const raw = rawArray([10, 3], [4, 3]);
    const probe = new ResidencyAccumulator();
    const demand = new AbortController();
    demand.abort(); // a superseded demand load must not cancel the warm-up
    const wrapped = wrapWithCache(raw as never, cache, '/n/centers', {
      getProbe: () => probe,
      getSignal: () => demand.signal,
    });
    await prefetchRangesIntoCache([wrapped], [{ start: 0, end: 10 }]);
    expect(probe.hits + probe.misses).toBe(0);
    expect(raw.getChunk).toHaveBeenCalledTimes(3);
    // Warm again: all hits, still nothing recorded in the probe.
    await prefetchRangesIntoCache([wrapped], [{ start: 0, end: 10 }]);
    expect(probe.hits + probe.misses).toBe(0);
    expect(raw.getChunk).toHaveBeenCalledTimes(3);
    expect(perfCounters.get('l0.hits')).toBe(3);
  });

  it('counts its decodes under decode.count.prefetch', async () => {
    const wrapped = wrapWithCache(rawArray([10, 3], [4, 3]) as never, cache, '/n/centers');
    await prefetchRangesIntoCache([wrapped], [{ start: 0, end: 4 }]);
    expect(perfCounters.get('decode.count.prefetch')).toBe(1);
    expect(perfCounters.get('decode.count.foreground')).toBe(0);
  });

  it('honours its own abort signal', async () => {
    const wrapped = wrapWithCache(rawArray([10, 3], [4, 3]) as never, cache, '/n/centers');
    const own = new AbortController();
    own.abort();
    await expect(
      prefetchRangesIntoCache([wrapped], [{ start: 0, end: 4 }], own.signal)
    ).rejects.toThrow();
    expect(cache.has(DecompressedChunkCache.makeKey('/n/centers', [0, 0]))).toBe(false);
  });

  it('falls back to a raw getChunk (no get()) for an array without L0', async () => {
    const raw = rawArray([10], [4]);
    await prefetchRangesIntoCache([raw as never], [{ start: 3, end: 5 }]);
    expect(mockGet).not.toHaveBeenCalled();
    expect(coordsOf(raw.getChunk)).toEqual(['0', '1']);
  });

  it('issues no reads when there are no arrays or no ranges', async () => {
    const raw = rawArray([10], [4]);
    await prefetchRangesIntoCache([], [{ start: 0, end: 1 }]);
    await prefetchRangesIntoCache([raw as never], []);
    await prefetchRangesIntoCache([raw as never], [{ start: 4, end: 4 }]);
    expect(raw.getChunk).not.toHaveBeenCalled();
    expect(mockGet).not.toHaveBeenCalled();
  });
});
