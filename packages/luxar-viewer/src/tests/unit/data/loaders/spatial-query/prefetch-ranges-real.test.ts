/**
 * `prefetchRangesIntoCache` against a REAL (unmocked) zarrita array wrapped by
 * the L0 proxy — the demand-load hooks must not see prefetch traffic.
 *
 * Before the fix the warm-up ran zarrita `get()` through the foreground
 * proxy's `getChunk`, so it recorded into the demand load's residency probe
 * (turning a "cold" demand load "resident", or vice versa) and bailed on the
 * demand load's abort signal.
 */

import { describe, it, expect } from 'vitest';
import * as zarr from '../../../../../data/zarr';
import { prefetchRangesIntoCache } from '../../../../../data/loaders';
import { DecompressedChunkCache } from '../../../../../cache/decompressed-chunk-cache';
import { wrapWithCache } from '../../../../../cache/decompressed-chunk-cache/cached-zarr-array';
import { ResidencyAccumulator } from '../../../../../cache/residency-probe';

const encoder = new TextEncoder();

class MemoryReadable {
  readonly reads: string[] = [];
  constructor(private readonly entries: Map<string, Uint8Array>) {}
  get(key: string): Promise<Uint8Array | undefined> {
    this.reads.push(key);
    return Promise.resolve(this.entries.get(key));
  }
}

function float32Bytes(values: number[]): Uint8Array {
  const data = new Float32Array(values);
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

/** A format-3 [6, 2] float32 array chunked [2, 2] (three chunks) at `/values`. */
function makeStore(): MemoryReadable {
  const meta = {
    zarr_format: 3,
    node_type: 'array',
    shape: [6, 2],
    data_type: 'float32',
    chunk_grid: { name: 'regular', configuration: { chunk_shape: [2, 2] } },
    chunk_key_encoding: { name: 'default', configuration: { separator: '/' } },
    fill_value: 0,
    codecs: [{ name: 'bytes', configuration: { endian: 'little' } }],
    attributes: {},
  };
  return new MemoryReadable(
    new Map([
      ['/values/zarr.json', encoder.encode(JSON.stringify(meta))],
      ['/values/c/0/0', float32Bytes([0, 1, 2, 3])],
      ['/values/c/1/0', float32Bytes([4, 5, 6, 7])],
      ['/values/c/2/0', float32Bytes([8, 9, 10, 11])],
    ])
  );
}

describe('prefetchRangesIntoCache (real zarrita array)', () => {
  it('warms L0 without touching the demand probe or the demand signal', async () => {
    const store = makeStore();
    const raw = await zarr.openArray(zarr.root(store as never).resolve('values'));
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const probe = new ResidencyAccumulator();
    const demand = new AbortController();
    demand.abort(); // the demand load was superseded; the warm-up is independent
    const wrapped = wrapWithCache(raw, cache, '/values', {
      getProbe: () => probe,
      getSignal: () => demand.signal,
    });

    await prefetchRangesIntoCache([wrapped], [{ start: 1, end: 3 }]);

    expect(probe.hits + probe.misses).toBe(0);
    expect(cache.has(DecompressedChunkCache.makeKey('/values', [0, 0]))).toBe(true);
    expect(cache.has(DecompressedChunkCache.makeKey('/values', [1, 0]))).toBe(true);
    expect(cache.has(DecompressedChunkCache.makeKey('/values', [2, 0]))).toBe(false);
  });

  it('a later demand read of the warmed rows is a pure L0 hit', async () => {
    const store = makeStore();
    const raw = await zarr.openArray(zarr.root(store as never).resolve('values'));
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const probe = new ResidencyAccumulator();
    const wrapped = wrapWithCache(raw, cache, '/values', { getProbe: () => probe });

    await prefetchRangesIntoCache([wrapped], [{ start: 0, end: 4 }]);
    const readsAfterWarm = store.reads.length;
    const out = await zarr.readArray(wrapped, [zarr.slice(0, 4), zarr.slice(null)]);

    expect(Array.from(out.data as Float32Array)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(store.reads.length).toBe(readsAfterWarm);
    expect(probe.misses).toBe(0);
    expect(probe.hits).toBe(2);
  });
});
