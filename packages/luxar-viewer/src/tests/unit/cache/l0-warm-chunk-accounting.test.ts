/**
 * A cache warm-up (`warmChunk`) is prefetch traffic, not demand: it must not
 * move the L0 hit/miss statistics the monitor reports, and its store read must
 * reach the multi-level store as a prefetch (`suppressPrefetch`), so that store
 * neither counts it as demand nor fans the prefetcher out from it.
 */
import { describe, it, expect } from 'vitest';
import { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import { wrapWithCache } from '../../../cache/decompressed-chunk-cache/cached-zarr-array';
import { warmChunk } from '../../../cache/decompressed-chunk-cache/warm-chunk';

/** Fake zarrita array recording the options each store read was issued with. */
function recordingArray(reads: Array<Record<string, unknown> | undefined>) {
  return {
    shape: [1000, 3],
    chunks: [100, 3],
    async getChunk(_coords: number[], options?: Record<string, unknown>) {
      reads.push(options);
      await Promise.resolve();
      return { data: new Float32Array(300), shape: [100, 3], stride: [3, 1] };
    },
  };
}

describe('L0 warm-ups are not demand', () => {
  it('a warm-up of a resident chunk counts no L0 hit', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const a = wrapWithCache(recordingArray([]) as never, cache, '/n/centers');
    await a.getChunk([0, 0]); // demand: one miss
    const before = cache.getStats();

    await warmChunk(a as never, [0, 0]);

    expect(cache.getStats().hits).toBe(before.hits);
    expect(cache.getStats().misses).toBe(before.misses);
  });

  it('a warm-up miss counts no L0 miss and reads the store as a prefetch', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const reads: Array<Record<string, unknown> | undefined> = [];
    const a = wrapWithCache(recordingArray(reads) as never, cache, '/n/centers');

    await warmChunk(a as never, [0, 0]);

    expect(cache.getStats().misses).toBe(0);
    expect(reads).toEqual([expect.objectContaining({ suppressPrefetch: true })]);
    // The warmed chunk is resident: the demand read that follows is a hit.
    await a.getChunk([0, 0]);
    expect(cache.getStats().hits).toBe(1);
  });

  it('an unwrapped array is warmed with a prefetch read too', async () => {
    const reads: Array<Record<string, unknown> | undefined> = [];

    await warmChunk(recordingArray(reads), [0, 0]);

    expect(reads).toEqual([expect.objectContaining({ suppressPrefetch: true })]);
  });

  it('a demand read still counts and reads the store as demand', async () => {
    const cache = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const reads: Array<Record<string, unknown> | undefined> = [];
    const a = wrapWithCache(recordingArray(reads) as never, cache, '/n/centers');

    await a.getChunk([0, 0]);

    expect(cache.getStats().misses).toBe(1);
    expect(reads[0]?.suppressPrefetch).toBeUndefined();
  });
});
