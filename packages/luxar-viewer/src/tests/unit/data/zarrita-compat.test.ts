/**
 * Guard tests for the small zarrita surface Luxar depends on.
 *
 * These intentionally use real zarrita reads against tiny in-memory Zarr v2
 * stores instead of mocking the library. They should keep passing across the
 * supported zarrita 0.5 -> 0.7 migration window and fail loudly if a future
 * zarrita release changes the reader, slicing, consolidated metadata, or
 * zarr.Array proxy assumptions used by the viewer.
 */

import { describe, expect, it } from 'vitest';
import * as zarr from 'zarrita';
import { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import { wrapWithCache } from '../../../cache/cached-zarr-array';
import { withMaybeConsolidatedMetadata } from '../../../data/zarrita-compat';

const encoder = new TextEncoder();

type MemoryCall = { key: string; options: unknown };

class MemoryReadable {
  readonly chunks = new Map<string, Uint8Array>();
  readonly calls: MemoryCall[] = [];

  constructor(entries: Array<[string, Uint8Array]>) {
    for (const [key, value] of entries) {
      this.chunks.set(key, value);
    }
  }

  get(key: `/${string}`, options?: unknown): Uint8Array | undefined {
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

function float32Bytes(values: number[]): Uint8Array {
  const data = new Float32Array(values);
  return new Uint8Array(data.buffer.slice(0));
}

const rootAttrs = { content_hash: 'zarrita-compat-test', scene_dimensions: { dimensions: [] } };
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

function hasSignalOption(options: unknown): boolean {
  return typeof options === 'object' && options !== null && 'signal' in options;
}

describe('zarrita compatibility guards', () => {
  it('exposes the public reader APIs Luxar uses', () => {
    const api = zarr as Record<string, unknown>;

    expect(api.FetchStore).toEqual(expect.any(Function));
    expect(api.root).toEqual(expect.any(Function));
    expect(api.open).toEqual(expect.any(Function));
    expect(api.get).toEqual(expect.any(Function));
    expect(api.slice).toEqual(expect.any(Function));
    expect(api.registry).toBeDefined();

    // zarrita 0.5 exported tryWithConsolidated(); zarrita 0.7 renamed the
    // behavior to withMaybeConsolidatedMetadata(). Luxar must support one of
    // them via data/zarrita-compat.ts.
    expect(
      typeof api.withMaybeConsolidatedMetadata === 'function' ||
        typeof api.tryWithConsolidated === 'function'
    ).toBe(true);
  });

  it('opens and slices a v2 array through a custom Readable store', async () => {
    const rawStore = makeUnconsolidatedStore();
    const store = await withMaybeConsolidatedMetadata(rawStore);
    const rootLoc = zarr.root(store);

    const group = await zarr.open(rootLoc, { kind: 'group' });
    expect(group.attrs).toEqual(rootAttrs);

    const array = await zarr.open(rootLoc.resolve('points'), { kind: 'array' });
    expect(array.shape).toEqual([4, 2]);
    expect(array.chunks).toEqual([2, 2]);
    expect(array.dtype).toBe('float32');
    expect(array.attrs).toEqual(pointsAttrs);

    const full = await zarr.get(array);
    expect(full.shape).toEqual([4, 2]);
    expect(Array.from(full.data as Float32Array)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);

    const sliced = await zarr.get(array, [zarr.slice(1, 3), zarr.slice(null)]);
    expect(sliced.shape).toEqual([2, 2]);
    expect(Array.from(sliced.data as Float32Array)).toEqual([3, 4, 5, 6]);

    expect(rawStore.calls.every((call) => call.key.startsWith('/'))).toBe(true);
    expect(rawStore.calls.some((call) => call.key === '/points/.zarray')).toBe(true);
    expect(rawStore.calls.some((call) => call.key === '/points/0.0')).toBe(true);

    // zarrita 0.7 forwards AbortSignal-bearing read options to stores. 0.5
    // does not, so only require this when the modern helper is present.
    if ('withMaybeConsolidatedMetadata' in zarr) {
      expect(rawStore.calls.some((call) => hasSignalOption(call.options))).toBe(true);
    }
  });

  it('uses consolidated metadata when present and exposes contents()', async () => {
    const rawStore = makeConsolidatedStore();
    const store = await withMaybeConsolidatedMetadata(rawStore);

    expect(store.contents?.()).toEqual([
      { path: '/', kind: 'group' },
      { path: '/points', kind: 'array' },
    ]);

    rawStore.calls.length = 0;
    const rootLoc = zarr.root(store);
    const group = await zarr.open(rootLoc, { kind: 'group' });
    const array = await zarr.open(rootLoc.resolve('points'), { kind: 'array' });
    const full = await zarr.get(array);

    expect(group.attrs).toEqual(rootAttrs);
    expect(array.attrs).toEqual(pointsAttrs);
    expect(Array.from(full.data as Float32Array)).toEqual([9, 8, 7, 6]);

    const underlyingKeys = rawStore.calls.map((call) => call.key);
    expect(underlyingKeys).toEqual(['/points/0.0']);
    expect(underlyingKeys).not.toContain('/.zgroup');
    expect(underlyingKeys).not.toContain('/points/.zarray');
  });

  it('keeps real zarrita.Array getters and getChunk() usable through the L0 cache proxy', async () => {
    const rawStore = makeUnconsolidatedStore();
    const rootLoc = zarr.root(rawStore);
    const array = await zarr.open(rootLoc.resolve('points'), { kind: 'array' });
    const cache = new DecompressedChunkCache({ maxSize: 1024 * 1024 });
    const wrapped = wrapWithCache(array, cache, '/points');

    expect(wrapped.shape).toEqual([4, 2]);
    expect(wrapped.chunks).toEqual([2, 2]);
    expect(wrapped.dtype).toBe('float32');
    expect(wrapped.attrs).toEqual(pointsAttrs);

    if ('fillValue' in array) {
      expect((wrapped as typeof wrapped & { fillValue: unknown }).fillValue).toBe(0);
    }
    if ('dimensionNames' in array) {
      expect((wrapped as typeof wrapped & { dimensionNames?: string[] }).dimensionNames).toBeUndefined();
    }

    const first = await wrapped.getChunk([0, 0]);
    const readsAfterFirstChunk = rawStore.callsFor('/points/0.0').length;
    const second = await wrapped.getChunk([0, 0]);

    expect(Array.from(first.data as Float32Array)).toEqual([1, 2, 3, 4]);
    expect(Array.from(second.data as Float32Array)).toEqual([1, 2, 3, 4]);
    expect(rawStore.callsFor('/points/0.0')).toHaveLength(readsAfterFirstChunk);
    expect(cache.getStats().hits).toBeGreaterThanOrEqual(1);
  });
});
