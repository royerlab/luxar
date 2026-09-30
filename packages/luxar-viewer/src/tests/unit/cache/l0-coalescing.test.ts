/**
 * L0 same-chunk decode coalescing — cache-wide, abort-isolated.
 *
 * Ported from the perf audit's reproduction:
 *
 * 1. Coalescing must be per L0 CACHE, not per proxy. The SlicePrefetcher's
 *    shadow loaders and the foreground loaders wrap the same arrays through
 *    different `wrapWithCache` proxies over one `DecompressedChunkCache`; a
 *    per-proxy in-flight map let both decode the same chunk concurrently
 *    (42-49% of all L0 decodes on time-partitioned playback were duplicates).
 * 2. A coalesced waiter must not inherit the FIRST caller's abort signal. The
 *    shared decode runs under its own controller, aborted only when every
 *    waiter has abandoned it; each waiter races the result against its own
 *    signal.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import {
  wrapWithCache,
  resetDecodeHistory,
} from '../../../cache/decompressed-chunk-cache/cached-zarr-array';
import { perfCounters } from '../../../profiling/perf-counters';

interface Counter {
  decodes: number;
  starts: number;
  aborted: number;
}

/** Fake zarrita array: getChunk honours options.signal like the MLC store does. */
function fakeArray(counter: Counter, delayMs = 20) {
  return {
    shape: [1000, 3],
    chunks: [100, 3],
    async getChunk(_coords: number[], options?: { signal?: AbortSignal }) {
      counter.starts++;
      await new Promise<void>((resolve, reject) => {
        const t = setTimeout(resolve, delayMs);
        options?.signal?.addEventListener('abort', () => {
          clearTimeout(t);
          counter.aborted++;
          reject(new DOMException('aborted', 'AbortError'));
        });
      });
      counter.decodes++;
      return { data: new Float32Array(300), shape: [100, 3], stride: [3, 1] };
    },
  };
}

/**
 * Fake array whose getChunk IGNORES options.signal and returns data filled
 * with call + 1, so a late result is distinguishable from a fresh one.
 *
 * The FIRST call does not finish until the test calls `releaseFirst()`, so
 * a test can order "later call lands" before "first call lands" explicitly
 * rather than through a race between two wall-clock timers. Later calls
 * finish on their own after a macrotask.
 */
function signalDeafArray(counter: Counter) {
  let finishFirst: (() => void) | undefined;
  const array = {
    shape: [1000, 3],
    chunks: [100, 3],
    async getChunk(_coords: number[], _options?: { signal?: AbortSignal }) {
      const call = counter.starts++;
      await new Promise<void>((resolve) => {
        if (call === 0) finishFirst = resolve;
        else setTimeout(resolve, 0);
      });
      counter.decodes++;
      return { data: new Float32Array(300).fill(call + 1), shape: [100, 3], stride: [3, 1] };
    },
  };
  const releaseFirst = () => {
    if (!finishFirst) throw new Error('releaseFirst() called before the first getChunk');
    finishFirst();
  };
  return { array, releaseFirst };
}

const outcome = (p: Promise<unknown>): Promise<string> =>
  p.then(
    () => 'ok',
    (e: Error) => e.name
  );

describe('L0 coalescing is per cache, not per proxy', () => {
  beforeEach(() => {
    perfCounters.reset();
    resetDecodeHistory();
  });

  it('two proxies over the SAME L0 cache share one in-flight decode', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const arr = fakeArray(counter);
    const a = wrapWithCache(arr as never, cache, '/n/centers');
    const b = wrapWithCache(arr as never, cache, '/n/centers'); // e.g. a shadow loader
    await Promise.all([a.getChunk([0, 0]), b.getChunk([0, 0])]);
    expect(counter.decodes).toBe(1);
    expect(perfCounters.get('l0.misses')).toBe(1);
    expect(perfCounters.get('l0.coalesced')).toBe(1);
    expect(perfCounters.get('decode.count')).toBe(1);
    expect(perfCounters.get('decode.duplicates')).toBe(0);
  });

  it('proxies over two separately-opened arrays of the same path also share it', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const a = wrapWithCache(fakeArray(counter) as never, cache, '/n/centers');
    const b = wrapWithCache(fakeArray(counter) as never, cache, '/n/centers');
    await Promise.all([a.getChunk([0, 0]), b.getChunk([0, 0])]);
    expect(counter.decodes).toBe(1);
  });

  it('different L0 caches (scenes) never coalesce with each other', async () => {
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const arr = fakeArray(counter);
    const a = wrapWithCache(arr as never, new DecompressedChunkCache(), '/n/centers');
    const b = wrapWithCache(arr as never, new DecompressedChunkCache(), '/n/centers');
    await Promise.all([a.getChunk([0, 0]), b.getChunk([0, 0])]);
    expect(counter.decodes).toBe(2);
  });
});

describe('L0 coalesced waiters are abort-isolated', () => {
  beforeEach(() => {
    perfCounters.reset();
    resetDecodeHistory();
  });

  it('a live coalesced waiter survives the FIRST caller aborting', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const a = wrapWithCache(fakeArray(counter) as never, cache, '/n/centers');
    const first = new AbortController();
    const second = new AbortController();
    const p1 = a.getChunk([0, 0], { signal: first.signal } as never);
    const p2 = a.getChunk([0, 0], { signal: second.signal } as never); // live caller
    first.abort();
    const r1 = await outcome(p1);
    const r2 = await outcome(p2);
    expect(second.signal.aborted).toBe(false);
    expect(r1).toBe('AbortError');
    expect(r2).toBe('ok');
    // The decode was not abandoned, so it completed once and is cached.
    expect(counter.decodes).toBe(1);
    expect(cache.has(DecompressedChunkCache.makeKey('/n/centers', [0, 0]))).toBe(true);
  });

  it('a waiter on a different proxy survives the first proxy caller aborting', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const arr = fakeArray(counter);
    const first = new AbortController();
    const a = wrapWithCache(arr as never, cache, '/n/centers', { getSignal: () => first.signal });
    const b = wrapWithCache(arr as never, cache, '/n/centers'); // no signal at all
    const p1 = a.getChunk([0, 0]);
    const p2 = b.getChunk([0, 0]);
    first.abort();
    expect(await outcome(p1)).toBe('AbortError');
    expect(await outcome(p2)).toBe('ok');
    expect(counter.decodes).toBe(1);
  });

  it('aborts the underlying decode only when EVERY waiter has aborted', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const a = wrapWithCache(fakeArray(counter) as never, cache, '/n/centers');
    const c1 = new AbortController();
    const c2 = new AbortController();
    const p1 = a.getChunk([0, 0], { signal: c1.signal } as never);
    const p2 = a.getChunk([0, 0], { signal: c2.signal } as never);
    c1.abort();
    await Promise.resolve();
    expect(counter.aborted).toBe(0); // c2 still waits
    c2.abort();
    expect(await outcome(p1)).toBe('AbortError');
    expect(await outcome(p2)).toBe('AbortError');
    // Wait past the fake decode's latency: the shared decode was cancelled.
    await new Promise((r) => setTimeout(r, 40));
    expect(counter.aborted).toBe(1);
    expect(counter.decodes).toBe(0);
    expect(cache.has(DecompressedChunkCache.makeKey('/n/centers', [0, 0]))).toBe(false);
  });

  it('a caller arriving after a fully-abandoned decode starts a fresh one', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const a = wrapWithCache(fakeArray(counter) as never, cache, '/n/centers');
    const c1 = new AbortController();
    const p1 = a.getChunk([0, 0], { signal: c1.signal } as never);
    c1.abort();
    expect(await outcome(p1)).toBe('AbortError');
    const p2 = a.getChunk([0, 0]);
    expect(await outcome(p2)).toBe('ok');
    expect(counter.starts).toBe(2);
    expect(counter.decodes).toBe(1);
  });

  it('an adopting proxy keeps the decode alive and aborts it when it leaves last', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const inner = fakeArray(counter);
    const decodeSignals: (AbortSignal | undefined)[] = [];
    const arr = {
      ...inner,
      getChunk(coords: number[], options?: { signal?: AbortSignal }) {
        decodeSignals.push(options?.signal);
        return inner.getChunk(coords, options);
      },
    };
    const shadowAc = new AbortController();
    const foregroundAc = new AbortController();
    const shadow = wrapWithCache(arr as never, cache, '/n/centers', {
      getSignal: () => shadowAc.signal,
    });
    const foreground = wrapWithCache(arr as never, cache, '/n/centers', {
      getSignal: () => foregroundAc.signal,
    });
    const key = DecompressedChunkCache.makeKey('/n/centers', [0, 0]);
    const p1 = shadow.getChunk([0, 0]);
    const p2 = foreground.getChunk([0, 0]); // adopts the shadow's decode
    expect(counter.starts).toBe(1);
    expect(decodeSignals).toHaveLength(1);
    expect(decodeSignals[0]).toBeDefined();
    expect(decodeSignals[0]).not.toBe(shadowAc.signal);
    expect(decodeSignals[0]).not.toBe(foregroundAc.signal);

    shadowAc.abort();
    expect(await outcome(p1)).toBe('AbortError');
    expect(counter.aborted).toBe(0);
    expect(decodeSignals[0]?.aborted).toBe(false);
    expect(cache.getInflight(key)).toBeDefined();

    foregroundAc.abort();
    expect(await outcome(p2)).toBe('AbortError');
    expect(decodeSignals[0]?.aborted).toBe(true);
    await new Promise((r) => setTimeout(r, 40));
    expect(counter.aborted).toBe(1);
    expect(counter.decodes).toBe(0);
    expect(cache.getInflight(key)).toBeUndefined();
    expect(cache.has(key)).toBe(false);
  });

  it('a fully-abandoned decode that finishes late does not overwrite its replacement', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const { array, releaseFirst } = signalDeafArray(counter);
    const a = wrapWithCache(array as never, cache, '/n/centers');
    const key = DecompressedChunkCache.makeKey('/n/centers', [0, 0]);
    const c1 = new AbortController();
    const p1 = a.getChunk([0, 0], { signal: c1.signal } as never);
    c1.abort();
    expect(await outcome(p1)).toBe('AbortError');
    expect(cache.getInflight(key)).toBeUndefined();

    // The abandoned decode is still pending (only releaseFirst() can finish
    // it), so the fresh decode lands and is cached strictly first.
    const fresh = await a.getChunk([0, 0]);
    expect(counter.starts).toBe(2);
    expect(counter.decodes).toBe(1);
    const firstValue = (data: unknown) => (data as Float32Array)[0];
    expect(firstValue(fresh.data)).toBe(2);
    expect(firstValue(cache.get(key)?.data)).toBe(2);

    // Now let the stale decode finish, and drain its settle handlers.
    releaseFirst();
    await new Promise((r) => setTimeout(r, 0));
    expect(counter.decodes).toBe(2);
    expect(firstValue(cache.get(key)?.data)).toBe(2);
    expect(firstValue((await a.getChunk([0, 0])).data)).toBe(2);
    expect(counter.starts).toBe(2);
  });
});

describe('L0 in-flight map and cache.clear()', () => {
  it('a decode orphaned by clear() still settles but does not commit', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const a = wrapWithCache(fakeArray(counter) as never, cache, '/n/centers');
    const p1 = a.getChunk([0, 0]);
    expect(cache.inflightCount).toBe(1);
    cache.clear();
    expect(cache.inflightCount).toBe(0);
    expect(await outcome(p1)).toBe('ok');
    expect(cache.has(DecompressedChunkCache.makeKey('/n/centers', [0, 0]))).toBe(false);
  });
});
