/**
 * array_ref resolution through a REAL L0 cache and a real zarrita store.
 *
 * Before the fix `resolveArrayRef` re-opened the target on every update and
 * returned an UNWRAPPED array, so every revisit re-fetched and re-decoded the
 * target's chunks (17 re-decodes per revisit step were measured on an
 * array_ref store). With the loader's L0 wrapper installed, the second read of
 * the same ref must be served from L0: no store reads, one L0 hit per chunk.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { RangeLoader } from '../../../../../data/loaders';
import { ArrayRefRegistry, type ArrayMetadata } from '../../../../../data/array-decoder/decoder';
import { DecompressedChunkCache } from '../../../../../cache/decompressed-chunk-cache';
import {
  wrapWithCache,
  resetDecodeHistory,
} from '../../../../../cache/decompressed-chunk-cache/cached-zarr-array';
import { perfCounters } from '../../../../../profiling/perf-counters';

const encoder = new TextEncoder();

class MemoryReadable {
  readonly reads: string[] = [];
  constructor(private readonly entries: Map<string, Uint8Array>) {}
  get(key: string): Promise<Uint8Array | undefined> {
    this.reads.push(key);
    return Promise.resolve(this.entries.get(key));
  }
}

/** A format-3 store with a [4, 3] float32 target at `/Shared/colors`, one chunk. */
function makeStore(): MemoryReadable {
  const meta = {
    zarr_format: 3,
    node_type: 'array',
    shape: [4, 3],
    data_type: 'float32',
    chunk_grid: { name: 'regular', configuration: { chunk_shape: [4, 3] } },
    chunk_key_encoding: { name: 'default', configuration: { separator: '/' } },
    fill_value: 0,
    codecs: [{ name: 'bytes', configuration: { endian: 'little' } }],
    attributes: {},
  };
  const data = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
  return new MemoryReadable(
    new Map([
      ['/zarr.json', encoder.encode(JSON.stringify({ zarr_format: 3, node_type: 'group' }))],
      ['/Shared/colors/zarr.json', encoder.encode(JSON.stringify(meta))],
      ['/Shared/colors/c/0/0', new Uint8Array(data.buffer)],
    ])
  );
}

const refAttrs: ArrayMetadata = {
  encoding: { name: 'array_ref', target: '/Shared/colors', hash: 'sha-test' },
};

describe('array_ref targets are read through L0', () => {
  beforeEach(() => {
    perfCounters.reset();
    resetDecodeHistory();
  });

  it('the second load of the same ref is a pure L0 hit (no re-open, no re-decode)', async () => {
    const store = makeStore();
    const l0 = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const loader = new RangeLoader(new ArrayRefRegistry(), { workerThreshold: Infinity });
    loader.setVerbose(false);
    loader.setRefTargetWrapper({
      wrap: (array, path) => wrapWithCache(array, l0, path),
      epoch: () => l0.generation,
    });

    const load = async () => {
      const out = new Float32Array(6);
      await loader.loadRangesResolvingRef(
        {} as never,
        refAttrs,
        [{ start: 1, end: 3 }],
        out,
        2,
        3,
        store as never
      );
      return Array.from(out);
    };

    expect(await load()).toEqual([3, 4, 5, 6, 7, 8]);
    const readsAfterFirst = store.reads.length;
    expect(perfCounters.get('decode.count')).toBe(1);

    expect(await load()).toEqual([3, 4, 5, 6, 7, 8]);
    expect(store.reads.length).toBe(readsAfterFirst); // no zarr.json re-open, no chunk fetch
    expect(perfCounters.get('decode.count')).toBe(1);
    expect(perfCounters.get('l0.hits')).toBe(1);
  });

  it('re-opens the target after the L0 cache is cleared', async () => {
    const store = makeStore();
    const l0 = new DecompressedChunkCache({ maxSize: 1 << 20 });
    const loader = new RangeLoader(new ArrayRefRegistry(), { workerThreshold: Infinity });
    loader.setVerbose(false);
    loader.setRefTargetWrapper({
      wrap: (array, path) => wrapWithCache(array, l0, path),
      epoch: () => l0.generation,
    });
    const load = () =>
      loader.loadRangesResolvingRef(
        {} as never,
        refAttrs,
        [{ start: 0, end: 1 }],
        new Float32Array(3),
        1,
        3,
        store as never
      );

    await load();
    const metaReads = () => store.reads.filter((k) => k.endsWith('zarr.json')).length;
    const before = metaReads();
    l0.clear();
    await load();
    expect(metaReads()).toBeGreaterThan(before);
  });
});
