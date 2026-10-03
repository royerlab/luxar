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
import { warmChunk } from '../../../cache/decompressed-chunk-cache/warm-chunk';
import { perfCounters } from '../../../profiling/perf-counters';
import { ResidencyAccumulator } from '../../../cache/residency-probe';

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

  it('a caller arriving while a cancelled decode settles retries once it is really cancelled', async () => {
    // A fully-abandoned decode stays registered until it settles (it may be
    // past its fetch and complete anyway — see the read-ahead suite below). One
    // that the abort DID stop rejects with AbortError; the caller that waited
    // on it then starts a fresh decode rather than inheriting the cancellation.
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const a = wrapWithCache(fakeArray(counter) as never, cache, '/n/centers');
    const key = DecompressedChunkCache.makeKey('/n/centers', [0, 0]);
    const c1 = new AbortController();
    const p1 = a.getChunk([0, 0], { signal: c1.signal } as never);
    c1.abort();
    // Before the cancelled decode has settled: the entry is still registered.
    expect(cache.getInflight(key)?.abandoned).toBe(true);
    const p2 = a.getChunk([0, 0]);
    expect(await outcome(p1)).toBe('AbortError');
    expect(await outcome(p2)).toBe('ok');
    expect(counter.starts).toBe(2);
    expect(counter.aborted).toBe(1);
    expect(counter.decodes).toBe(1);
    expect(cache.has(key)).toBe(true);
    expect(perfCounters.get('l0.coalesced')).toBe(0);
    expect(perfCounters.get('l0.misses')).toBe(2);
  });

  it.fails('the retry keys on the decode being cancelled, not on the abort reason being an AbortError', async () => {
    // fetch() rejects with the signal's REASON, and a caller may abort with any
    // reason (here a plain Error): the shared decode then rejects with that
    // error, whose name is not 'AbortError'. The waiting caller must still retry.
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    let starts = 0;
    const array = {
      shape: [1000, 3],
      chunks: [100, 3],
      async getChunk(_coords: number[], options?: { signal?: AbortSignal }) {
        starts++;
        await new Promise<void>((resolve, reject) => {
          const t = setTimeout(resolve, 20);
          options?.signal?.addEventListener('abort', () => {
            clearTimeout(t);
            reject(options.signal!.reason as Error);
          });
        });
        return { data: new Float32Array(300), shape: [100, 3], stride: [3, 1] };
      },
    };
    const a = wrapWithCache(array as never, cache, '/n/centers');
    const key = DecompressedChunkCache.makeKey('/n/centers', [0, 0]);
    const c1 = new AbortController();
    const p1 = a.getChunk([0, 0], { signal: c1.signal } as never);
    c1.abort(new Error('superseded'));
    expect(cache.getInflight(key)?.abandoned).toBe(true);
    const p2 = a.getChunk([0, 0]);
    expect(await outcome(p1)).toBe('Error');
    expect(await outcome(p2)).toBe('ok');
    expect(starts).toBe(2);
    expect(cache.has(key)).toBe(true);
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

describe('L0 — a cancelled read-ahead whose decode is already past its fetch', () => {
  beforeEach(() => {
    perfCounters.reset();
    resetDecodeHistory();
  });

  // A speculative warm (B5 read-ahead, lookahead, predicted view) is aborted on
  // the next view change. When that lands after the chunk's bytes arrived, the
  // decode cannot stop and runs to completion anyway. It used to be
  // DEREGISTERED at the abort and its result DISCARDED, so the foreground read
  // of the same chunk a moment later fetched and decoded it again — counted as
  // `decode.duplicates` (WebGPU scrub_pl_drag: 1 -> 6, decodes +28%).
  it('the foreground read joins the cancelled warm instead of decoding again', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const { array, releaseFirst } = signalDeafArray(counter);
    const a = wrapWithCache(array as never, cache, '/n/centers');
    const probe = new ResidencyAccumulator();
    const demand = wrapWithCache(array as never, cache, '/n/centers', {
      getProbe: () => probe,
    });
    const warm = new AbortController();
    const warming = warmChunk(a as never, [0, 0], { signal: warm.signal });
    warm.abort(); // the next drag step supersedes the read-ahead
    expect(await outcome(warming)).toBe('AbortError');

    const foreground = demand.getChunk([0, 0]);
    releaseFirst(); // the cancelled warm's decode completes regardless
    const chunk = await foreground;

    expect((chunk.data as Float32Array)[0]).toBe(1); // the warm's own bytes
    expect(counter.starts).toBe(1);
    expect(counter.decodes).toBe(1);
    expect(perfCounters.get('decode.duplicates')).toBe(0);
    expect(perfCounters.get('l0.coalesced')).toBe(1);
    expect(probe.hits).toBe(1);
    expect(probe.misses).toBe(0);
  });

  it('a cancelled warm that completes with no joiner still lands in L0', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const counter: Counter = { decodes: 0, starts: 0, aborted: 0 };
    const { array, releaseFirst } = signalDeafArray(counter);
    const a = wrapWithCache(array as never, cache, '/n/centers');
    const warm = new AbortController();
    const warming = warmChunk(a as never, [0, 0], { signal: warm.signal });
    warm.abort();
    expect(await outcome(warming)).toBe('AbortError');
    releaseFirst();
    await new Promise((r) => setTimeout(r, 0));

    await a.getChunk([0, 0]); // an L0 hit: the paid-for decode was kept
    expect(counter.starts).toBe(1);
    expect(perfCounters.get('l0.hits')).toBe(1);
  });
});
