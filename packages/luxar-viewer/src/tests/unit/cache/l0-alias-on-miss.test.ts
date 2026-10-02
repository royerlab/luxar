/**
 * L0 stores the decoded chunk WITHOUT a defensive clone — on every array.
 *
 * A clone would protect only the caller that decoded the chunk: every hit and
 * every coalesced waiter shares the stored buffer anyway, so the contract is
 * "L0 chunks are read-only" (`l0-immutability.test.ts`). zarrita `get()` keeps
 * it by construction — it copies each chunk into its own freshly allocated
 * output (`setter.setFromChunk`). The deprecated `aliasOnMiss` hook no longer
 * changes anything.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as zarr from '../../../data/zarr';
import { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import {
  wrapWithCache,
  resetDecodeHistory,
} from '../../../cache/decompressed-chunk-cache/cached-zarr-array';
import { perfCounters } from '../../../profiling/perf-counters';

const encoder = new TextEncoder();

class MemoryReadable {
  constructor(private readonly entries: Map<string, Uint8Array>) {}
  get(key: string): Promise<Uint8Array | undefined> {
    return Promise.resolve(this.entries.get(key));
  }
}

/** A format-3 [4, 2] float32 array in one [4, 2] chunk at `/values`. */
function makeStore(): MemoryReadable {
  const meta = {
    zarr_format: 3,
    node_type: 'array',
    shape: [4, 2],
    data_type: 'float32',
    chunk_grid: { name: 'regular', configuration: { chunk_shape: [4, 2] } },
    chunk_key_encoding: { name: 'default', configuration: { separator: '/' } },
    fill_value: 0,
    codecs: [{ name: 'bytes', configuration: { endian: 'little' } }],
    attributes: {},
  };
  const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
  return new MemoryReadable(
    new Map([
      ['/values/zarr.json', encoder.encode(JSON.stringify(meta))],
      ['/values/c/0/0', new Uint8Array(data.buffer)],
    ])
  );
}

function fakeArray(decoded: Float32Array) {
  return {
    dtype: 'float32',
    shape: [decoded.length],
    chunks: [decoded.length],
    attrs: {},
    getChunk: () => Promise.resolve({ data: decoded, shape: [decoded.length], stride: [1] }),
  };
}

const KEY = DecompressedChunkCache.makeKey('/n/values', [0]);

describe('L0 stores decoded chunks without a clone', () => {
  let cache: DecompressedChunkCache;

  beforeEach(() => {
    cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    perfCounters.reset();
    resetDecodeHistory();
  });

  it('the miss path stores the decoded buffer itself (no clone), hook or not', async () => {
    const decoded = new Float32Array([1, 2, 3]);
    const wrapped = wrapWithCache(fakeArray(decoded) as never, cache, '/n/values');
    await wrapped.getChunk([0]);
    expect(cache.get(KEY)!.data).toBe(decoded);
    expect(perfCounters.get('l0.cloneBytes')).toBe(0);
  });

  it('aliasOnMiss: the miss path stores the decoded buffer itself (no clone)', async () => {
    const decoded = new Float32Array([1, 2, 3]);
    const wrapped = wrapWithCache(fakeArray(decoded) as never, cache, '/n/values', {
      aliasOnMiss: true,
    });
    await wrapped.getChunk([0]);
    expect(cache.get(KEY)!.data).toBe(decoded);
    expect(perfCounters.get('l0.cloneBytes')).toBe(0);
  });

  it('aliasOnMiss still clones a view into a LARGER buffer (no hidden retention)', async () => {
    const shard = new Float32Array(1024);
    const decoded = shard.subarray(8, 11); // e.g. an uncompressed chunk inside a shard
    const wrapped = wrapWithCache(fakeArray(decoded) as never, cache, '/n/values', {
      aliasOnMiss: true,
    });
    await wrapped.getChunk([0]);
    const stored = cache.get(KEY)!.data;
    expect(stored).not.toBe(decoded);
    expect(stored.buffer.byteLength).toBe(decoded.byteLength);
    expect(perfCounters.get('l0.cloneBytes')).toBe(decoded.byteLength);
  });

  it('aliasOnMiss also applies to a warmChunk miss', async () => {
    const decoded = new Float32Array([1, 2, 3]);
    const wrapped = wrapWithCache(fakeArray(decoded) as never, cache, '/n/values', {
      aliasOnMiss: true,
    });
    await (wrapped as unknown as { warmChunk(c: number[]): Promise<void> }).warmChunk([0]);
    expect(cache.get(KEY)!.data).toBe(decoded);
    expect(perfCounters.get('l0.cloneBytes')).toBe(0);
  });

  it('aliasOnMiss: mutating a get() result never corrupts the cache', async () => {
    const raw = await zarr.openArray(zarr.root(makeStore() as never).resolve('values'));
    const wrapped = wrapWithCache(raw, cache, '/values', { aliasOnMiss: true });

    const first = await zarr.readArray(wrapped); // miss: decoded buffer aliased into L0
    (first.data as Float32Array).fill(-1);
    const second = await zarr.readArray(wrapped); // hit
    (second.data as Float32Array).fill(-2);
    const third = await zarr.readArray(wrapped, [zarr.slice(1, 3), zarr.slice(null)]);

    expect(Array.from(third.data as Float32Array)).toEqual([3, 4, 5, 6]);
    const entry = cache.get(DecompressedChunkCache.makeKey('/values', [0, 0]))!;
    expect(Array.from(entry.data as Float32Array)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(perfCounters.get('l0.cloneBytes')).toBe(0);
  });
});
