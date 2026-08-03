/**
 * Lock-in tests for the L0 read-only chunk contract.
 *
 * The L0 cache returns its stored `ArrayBufferView` directly on hit.
 * Loaders downstream of `wrapWithCache` must therefore treat the
 * chunk data as immutable and copy into accumulator/output buffers
 * before mutating. These tests prove that contract holds:
 *
 * 1. Two consecutive getChunk() calls observe the same data —
 *    nothing mutates the cached entry between hits.
 * 2. The L0 cache entry's underlying bytes are unchanged after a
 *    "consumer" loop reads through them.
 *
 * Production loaders never call getChunk() directly (they go through
 * zarr.readArray()), so this is a tier-boundary smoke test rather than an
 * exhaustive sweep. If a future loader is wired against getChunk(), it
 * should add its own immutability assertion here.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import { wrapWithCache } from '../../../cache/decompressed-chunk-cache/cached-zarr-array';

describe('L0 read-only chunk contract (commit 6.1)', () => {
  let cache: DecompressedChunkCache;

  beforeEach(() => {
    cache = new DecompressedChunkCache({ maxSize: 1024 * 1024 });
  });

  // Renamed: the previous name ("no in-cache mutation") was misleading because
  // the assertion `expect(r2.data).toBe(r1.data)` actually proves the OPPOSITE
  // — the cache hands back reference-equal views, so a mutation through r1
  // WOULD corrupt r2. The contract this test pins is reference-stability of
  // cache hits + that the byte payload survives unchanged across hits when no
  // consumer has mutated. The downstream "a consumer loop … does not corrupt
  // later hits" test is what actually guards against mutation.
  it('cache hits return the same reference (load-bearing assumption: consumers must not mutate)', async () => {
    const sourceData = new Float32Array([1.5, 2.5, 3.5, 4.5]);
    let getChunkCalls = 0;
    const fakeArray: any = {
      dtype: 'float32',
      shape: [4],
      chunks: [4],
      attrs: {},
      async getChunk() {
        getChunkCalls++;
        // Return a fresh view each call so the cache's clone-on-miss
        // path stores its own copy (current behavior); the assertion
        // here is about post-hit immutability.
        return {
          data: new Float32Array(sourceData),
          shape: [4],
          stride: [1],
        };
      },
    };
    const wrapped = wrapWithCache(fakeArray, cache, '/points/positions');

    // Warm the cache. r0 came from the miss path so its data view is
    // distinct from the cached clone (current behavior).
    const r0 = await wrapped.getChunk([0]);
    expect(getChunkCalls).toBe(1);
    expect(Array.from(r0.data as Float32Array)).toEqual([1.5, 2.5, 3.5, 4.5]);

    // Two subsequent hits both come from the cache. Their data views
    // are reference-equal — that's what makes the read-only contract
    // load-bearing: a mutation through r1 would be visible through r2.
    const r1 = await wrapped.getChunk([0]);
    const r2 = await wrapped.getChunk([0]);
    expect(getChunkCalls).toBe(1);
    expect(r2.data).toBe(r1.data);
    expect(Array.from(r2.data as Float32Array)).toEqual([1.5, 2.5, 3.5, 4.5]);
  });

  it('a consumer loop that reads the chunk does not corrupt later hits', async () => {
    const sourceData = new Float32Array([10, 20, 30]);
    const fakeArray: any = {
      dtype: 'float32',
      shape: [3],
      chunks: [3],
      attrs: {},
      async getChunk() {
        return {
          data: new Float32Array(sourceData),
          shape: [3],
          stride: [1],
        };
      },
    };
    const wrapped = wrapWithCache(fakeArray, cache, '/lines/values');

    const first = await wrapped.getChunk([0]);
    // Read-only consumer pattern: copy into an accumulator before any
    // arithmetic. This is what production loaders do.
    const accumulator = new Float32Array(first.data as Float32Array);
    for (let i = 0; i < accumulator.length; i++) accumulator[i] *= 2;
    expect(Array.from(accumulator)).toEqual([20, 40, 60]);

    // The cache entry must be unchanged.
    const second = await wrapped.getChunk([0]);
    expect(Array.from(second.data as Float32Array)).toEqual([10, 20, 30]);
  });
});
