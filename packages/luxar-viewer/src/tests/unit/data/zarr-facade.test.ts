/**
 * Contract tests for Luxar's Zarr facade.
 *
 * These tests deliberately avoid mocking the Zarr backend. They use tiny
 * in-memory Zarr v2 stores and exercise the Luxar-owned API in `data/zarr`.
 * The implementation currently delegates to zarrita, but callers should rely
 * on this facade rather than importing the backend directly.
 */

import { describe, expect, it } from 'vitest';
import * as zarr from '../../../data/zarr';
import { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import { wrapWithCache } from '../../../cache/decompressed-chunk-cache/cached-zarr-array';

const encoder = new TextEncoder();

type MemoryCall = { key: string; options: unknown };

class MemoryReadable implements zarr.AsyncReadable {
  readonly chunks = new Map<string, Uint8Array>();
  readonly calls: MemoryCall[] = [];

  constructor(entries: Array<[string, Uint8Array]>) {
    for (const [key, value] of entries) {
      this.chunks.set(key, value);
    }
  }

  async get(key: zarr.AbsolutePath, options?: zarr.GetOptions): Promise<Uint8Array | undefined> {
    this.calls.push({ key, options });
    return this.chunks.get(key);
  }

  callsFor(key: string): MemoryCall[] {
    return this.calls.filter((call) => call.key === key);
  }
}

function jsonBytes(value: unknown): Uint8Array {
  return encoder.encode(JSON.stringify(value));
}

function typedBytes<T extends ArrayBufferView>(data: T): Uint8Array {
  return new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
}

function float32Bytes(values: number[]): Uint8Array {
  return typedBytes(new Float32Array(values));
}

function uint16Bytes(values: number[]): Uint8Array {
  return typedBytes(new Uint16Array(values));
}

const rootAttrs = { content_hash: 'zarr-facade-test', scene_dimensions: { dimensions: [] } };
const pointsAttrs = { encoding: { name: 'direct' } };
const pointsArrayMetadata = {
  zarr_format: 2,
  shape: [4, 2],
  chunks: [2, 2],
  dtype: '<f4',
  compressor: null,
  fill_value: 0,
  order: 'C',
  filters: null,
};

function makeUnconsolidatedStore(): MemoryReadable {
  return new MemoryReadable([
    ['/.zgroup', jsonBytes({ zarr_format: 2 })],
    ['/.zattrs', jsonBytes(rootAttrs)],
    ['/points/.zarray', jsonBytes(pointsArrayMetadata)],
    ['/points/.zattrs', jsonBytes(pointsAttrs)],
    ['/points/0.0', float32Bytes([1, 2, 3, 4])],
    ['/points/1.0', float32Bytes([5, 6, 7, 8])],
  ]);
}

function makeConsolidatedStore(): MemoryReadable {
  return new MemoryReadable([
    [
      '/.zmetadata',
      jsonBytes({
        zarr_consolidated_format: 1,
        metadata: {
          '.zgroup': { zarr_format: 2 },
          '.zattrs': rootAttrs,
          'points/.zarray': { ...pointsArrayMetadata, shape: [2, 2] },
          'points/.zattrs': pointsAttrs,
        },
      }),
    ],
    ['/points/0.0', float32Bytes([9, 8, 7, 6])],
  ]);
}

function makeFillValueStore(): MemoryReadable {
  return new MemoryReadable([
    ['/.zgroup', jsonBytes({ zarr_format: 2 })],
    ['/.zattrs', jsonBytes(rootAttrs)],
    [
      '/values/.zarray',
      jsonBytes({
        zarr_format: 2,
        shape: [4],
        chunks: [2],
        dtype: '<u2',
        compressor: null,
        fill_value: 42,
        order: 'C',
        filters: null,
      }),
    ],
    ['/values/.zattrs', jsonBytes({ semantic: 'fill-value-contract' })],
    ['/values/0', uint16Bytes([10, 11])],
    // Deliberately omit /values/1. The facade must preserve backend fill-value behavior.
  ]);
}

function hasSignalOption(options: unknown, signal?: AbortSignal): boolean {
  if (typeof options !== 'object' || options === null || !('signal' in options)) return false;
  return signal === undefined || (options as { signal?: AbortSignal }).signal === signal;
}

describe('Zarr facade contract', () => {
  it('exposes the Luxar reader API surface', () => {
    expect(zarr.createFetchStore).toEqual(expect.any(Function));
    expect(zarr.openStore).toEqual(expect.any(Function));
    expect(zarr.root).toEqual(expect.any(Function));
    expect(zarr.open).toEqual(expect.any(Function));
    expect(zarr.openGroup).toEqual(expect.any(Function));
    expect(zarr.openArray).toEqual(expect.any(Function));
    expect(zarr.readArray).toEqual(expect.any(Function));
    expect(zarr.slice).toEqual(expect.any(Function));
    expect(zarr.isNotFoundError).toEqual(expect.any(Function));
    expect(zarr.codecRegistry).toBeDefined();

    const fetchStore = zarr.createFetchStore('https://example.test/data.zarr');
    expect(fetchStore).toHaveProperty('get');
  });

  it('resolves dataset-relative locations predictably', () => {
    const rawStore = makeUnconsolidatedStore();
    const rootLoc = zarr.root(rawStore);

    expect(rootLoc.path).toBe('/');
    expect(rootLoc.resolve('points').path).toBe('/points');
    expect(rootLoc.resolve('/points').path).toBe('/points');
    expect(rootLoc.resolve('nested/../points').path).toBe('/points');
    expect(rootLoc.resolve('points').resolve('../points').path).toBe('/points');
  });

  it('opens an unconsolidated v2 group/array and reads full + sliced data', async () => {
    const rawStore = makeUnconsolidatedStore();
    const store = await zarr.openStore(rawStore);
    expect(store.contents).toBeUndefined();

    const rootLoc = zarr.root(store);
    const group = await zarr.openGroup(rootLoc);
    expect(group.attrs).toEqual(rootAttrs);

    const genericOpened = await zarr.open(rootLoc.resolve('points'));
    expect(genericOpened.kind).toBe('array');

    const array = await zarr.openArray(rootLoc.resolve('points'));
    expect(array.shape).toEqual([4, 2]);
    expect(array.chunks).toEqual([2, 2]);
    expect(array.dtype).toBe('float32');
    expect(array.attrs).toEqual(pointsAttrs);

    const full = await zarr.readArray(array);
    expect(full.shape).toEqual([4, 2]);
    expect(full.stride).toEqual([2, 1]);
    expect(Array.from(full.data as Float32Array)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);

    const sliced = await zarr.readArray(array, [zarr.slice(1, 3), zarr.slice(null)]);
    expect(sliced.shape).toEqual([2, 2]);
    expect(Array.from(sliced.data as Float32Array)).toEqual([3, 4, 5, 6]);

    const openEnded = await zarr.readArray(array, [zarr.slice(null, 2), zarr.slice(1, null)]);
    expect(openEnded.shape).toEqual([2, 1]);
    expect(Array.from(openEnded.data as Float32Array)).toEqual([2, 4]);

    expect(rawStore.calls.every((call) => call.key.startsWith('/'))).toBe(true);
    expect(rawStore.calls.some((call) => call.key === '/points/.zarray')).toBe(true);
    expect(rawStore.calls.some((call) => call.key === '/points/0.0')).toBe(true);
  });

  it('forwards AbortSignal options for metadata and chunk reads', async () => {
    const rawStore = makeUnconsolidatedStore();
    const store = await zarr.openStore(rawStore);
    const rootLoc = zarr.root(store);
    const signal = new AbortController().signal;

    rawStore.calls.length = 0;
    const array = await zarr.openArray(rootLoc.resolve('points'), { signal });
    expect(rawStore.calls.some((call) => call.key === '/points/.zarray')).toBe(true);
    expect(rawStore.calls.some((call) => hasSignalOption(call.options, signal))).toBe(true);

    rawStore.calls.length = 0;
    await zarr.readArray(array, undefined, { signal });
    expect(rawStore.calls.some((call) => call.key === '/points/0.0')).toBe(true);
    expect(rawStore.calls.some((call) => hasSignalOption(call.options, signal))).toBe(true);
  });

  it('uses consolidated metadata when present and exposes contents()', async () => {
    const rawStore = makeConsolidatedStore();
    const store = await zarr.openStore(rawStore);

    expect(rawStore.calls.map((call) => call.key)).toEqual(['/.zmetadata']);
    expect(store.contents?.()).toEqual([
      { path: '/', kind: 'group' },
      { path: '/points', kind: 'array' },
    ]);

    rawStore.calls.length = 0;
    const rootLoc = zarr.root(store);
    const group = await zarr.openGroup(rootLoc);
    const array = await zarr.openArray(rootLoc.resolve('points'));
    const full = await zarr.readArray(array);

    expect(group.attrs).toEqual(rootAttrs);
    expect(array.attrs).toEqual(pointsAttrs);
    expect(Array.from(full.data as Float32Array)).toEqual([9, 8, 7, 6]);

    const underlyingKeys = rawStore.calls.map((call) => call.key);
    expect(underlyingKeys).toEqual(['/points/0.0']);
    expect(underlyingKeys).not.toContain('/.zgroup');
    expect(underlyingKeys).not.toContain('/points/.zarray');
  });

  it('preserves dtype conversion and fill-value behavior for missing chunks', async () => {
    const rawStore = makeFillValueStore();
    const rootLoc = zarr.root(await zarr.openStore(rawStore));
    const array = await zarr.openArray(rootLoc.resolve('values'));

    expect(array.dtype).toBe('uint16');
    expect(array.shape).toEqual([4]);
    expect(array.attrs).toEqual({ semantic: 'fill-value-contract' });

    const full = await zarr.readArray(array);
    expect(full.data).toBeInstanceOf(Uint16Array);
    expect(Array.from(full.data as Uint16Array)).toEqual([10, 11, 42, 42]);

    const missingChunk = await array.getChunk([1]);
    expect(Array.from(missingChunk.data as Uint16Array)).toEqual([42, 42]);
  });

  it('propagates store AbortError instead of substituting fill values', async () => {
    const rawStore = makeFillValueStore();
    const originalGet = rawStore.get.bind(rawStore);
    rawStore.get = async (key, options) => {
      if (key === '/values/1') {
        throw new DOMException('Cache read aborted during invalidation', 'AbortError');
      }
      return originalGet(key, options);
    };

    const rootLoc = zarr.root(await zarr.openStore(rawStore));
    const array = await zarr.openArray(rootLoc.resolve('values'));

    await expect(zarr.readArray(array)).rejects.toMatchObject({
      name: 'AbortError',
      message: expect.stringContaining('invalidation'),
    });
  });

  it('normalizes backend missing-node errors', async () => {
    const rawStore = makeUnconsolidatedStore();
    const rootLoc = zarr.root(await zarr.openStore(rawStore));

    try {
      await zarr.openArray(rootLoc.resolve('missing-array'));
      throw new Error('Expected openArray to reject');
    } catch (error) {
      expect(zarr.isNotFoundError(error)).toBe(true);
    }

    expect(zarr.isNotFoundError(new Error('Node not found: /foo'))).toBe(true);
    expect(zarr.isNotFoundError(new Error('HTTP 404'))).toBe(true);
    expect(zarr.isNotFoundError(new Error('decode failed'))).toBe(false);
  });

  it('keeps real array getters and getChunk() usable through the L0 cache proxy', async () => {
    const rawStore = makeUnconsolidatedStore();
    const rootLoc = zarr.root(rawStore);
    const array = await zarr.openArray(rootLoc.resolve('points'));
    const cache = new DecompressedChunkCache({ maxSize: 1024 * 1024 });
    const wrapped = wrapWithCache(array, cache, '/points');

    expect(wrapped.shape).toEqual([4, 2]);
    expect(wrapped.chunks).toEqual([2, 2]);
    expect(wrapped.dtype).toBe('float32');
    expect(wrapped.attrs).toEqual(pointsAttrs);
    expect((wrapped as typeof wrapped & { fillValue: unknown }).fillValue).toBe(0);
    expect(
      (wrapped as typeof wrapped & { dimensionNames?: string[] }).dimensionNames
    ).toBeUndefined();

    const first = await wrapped.getChunk([0, 0]);
    const readsAfterFirstChunk = rawStore.callsFor('/points/0.0').length;
    const second = await wrapped.getChunk([0, 0]);

    expect(Array.from(first.data as Float32Array)).toEqual([1, 2, 3, 4]);
    expect(Array.from(second.data as Float32Array)).toEqual([1, 2, 3, 4]);
    expect(rawStore.callsFor('/points/0.0')).toHaveLength(readsAfterFirstChunk);
    expect(cache.getStats().hits).toBeGreaterThanOrEqual(1);
  });
});
