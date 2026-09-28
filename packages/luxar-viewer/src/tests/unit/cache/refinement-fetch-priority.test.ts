/**
 * Refinement reads queue in the `refinement` fetch class (B9c).
 *
 * A refinement run tags its abort signal (`tagSignalPriority`), and every read
 * made under it — through the L0 chunk proxy's shared decode and the caching
 * store — reaches the chunk source at that class instead of `demand`. A demand
 * caller joining the same decode lifts it back to `demand`, so a frame never
 * waits behind the refinement class. Untagged reads keep their old default.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';
import type { ChunkFetchOutcome, ChunkSource } from '../../../cache/chunk-source';
import type { RemoteValidationToken } from '../../../cache/multi-level-caching-store/validation-queue';
import { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import { wrapWithCache } from '../../../cache/decompressed-chunk-cache/cached-zarr-array';
import {
  FetchPriorityCell,
  signalPriority,
  tagSignalPriority,
  type FetchPriority,
} from '../../../utils/fetch-concurrency';

/** A source that records the fetch class each read arrives with. */
function recordingSource(): { source: ChunkSource; seen: FetchPriority[] } {
  const seen: FetchPriority[] = [];
  const source: ChunkSource = {
    identity: 'fake://identity',
    describe: 'fake://describe',
    async get(_key, _signal, options): Promise<ChunkFetchOutcome> {
      seen.push(options?.priority?.value ?? 'demand');
      return { kind: 'ok', data: new Uint8Array([1, 2, 3, 4]), bytesOverWire: 4 };
    },
    async probeIdentityToken(): Promise<RemoteValidationToken | null> {
      return null;
    },
    dispose() {},
  };
  return { source, seen };
}

beforeEach(() => {
  vi.stubGlobal('navigator', {});
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('tagSignalPriority', () => {
  it('attaches one cell per signal that only ever rises', () => {
    const signal = new AbortController().signal;
    expect(signalPriority(signal)).toBeUndefined();
    const cell = tagSignalPriority(signal, 'speculative');
    expect(signalPriority(signal)).toBe(cell);
    expect(tagSignalPriority(signal, 'refinement')).toBe(cell);
    expect(cell.value).toBe('refinement');
    tagSignalPriority(signal, 'speculative'); // never demotes
    expect(cell.value).toBe('refinement');
    expect(signalPriority(null)).toBeUndefined();
  });

  it('adopts a passed cell when the signal has none', () => {
    const signal = new AbortController().signal;
    const cell = new FetchPriorityCell('refinement');
    expect(tagSignalPriority(signal, cell)).toBe(cell);
  });
});

describe('MultiLevelCachingStore — signal-carried priority', () => {
  it('fetches a tagged read at its class, an untagged one at demand', async () => {
    const { source, seen } = recordingSource();
    const store = new MultiLevelCachingStore(source, { noOpfs: true });
    const refinement = new AbortController().signal;
    tagSignalPriority(refinement, 'refinement');

    await store.get('a/c/0', { signal: refinement });
    await store.get('b/c/0', { signal: new AbortController().signal });
    await store.get('c/c/0', { signal: refinement, priority: 'demand' }); // explicit wins

    expect(seen).toEqual(['refinement', 'demand', 'demand']);
    await store.dispose();
  });
});

describe('L0 chunk proxy — the shared decode keeps the caller class', () => {
  function storeBackedArray(store: MultiLevelCachingStore) {
    return {
      dtype: 'float32',
      shape: [4],
      chunks: [4],
      attrs: {},
      getChunk: async (coords: number[], options?: { signal?: AbortSignal }) => {
        await store.get(`arr/c/${coords.join('/')}`, { signal: options?.signal });
        return { data: new Float32Array(4), shape: [4], stride: [1] };
      },
    } as never;
  }

  it('reaches the source at refinement for a refinement caller', async () => {
    const { source, seen } = recordingSource();
    const store = new MultiLevelCachingStore(source, { noOpfs: true });
    const array = wrapWithCache(storeBackedArray(store), new DecompressedChunkCache(), '/arr');
    const refinement = new AbortController().signal;
    tagSignalPriority(refinement, 'refinement');

    await (array as { getChunk: (c: number[], o: object) => Promise<unknown> }).getChunk([0], {
      signal: refinement,
    });

    expect(seen).toEqual(['refinement']);
    await store.dispose();
  });

  it('a demand caller joining a refinement decode lifts it to demand', async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const seen: FetchPriority[] = [];
    let decodeSignal: AbortSignal | undefined;
    const array = wrapWithCache(
      {
        dtype: 'float32',
        shape: [4],
        chunks: [4],
        attrs: {},
        getChunk: async (_coords: number[], options?: { signal?: AbortSignal }) => {
          decodeSignal = options?.signal;
          await gate;
          seen.push(signalPriority(decodeSignal)?.value ?? 'demand');
          return { data: new Float32Array(4), shape: [4], stride: [1] };
        },
      } as never,
      new DecompressedChunkCache(),
      '/arr'
    );
    const get = (array as { getChunk: (c: number[], o: object) => Promise<unknown> }).getChunk;
    const refinement = new AbortController().signal;
    tagSignalPriority(refinement, 'refinement');

    const first = get([0], { signal: refinement });
    await Promise.resolve();
    expect(signalPriority(decodeSignal)?.value).toBe('refinement');
    const joined = get([0], { signal: new AbortController().signal }); // untagged = demand
    release();
    await Promise.all([first, joined]);

    expect(seen).toEqual(['demand']);
    // The run's own cell is untouched: only this decode was lifted.
    expect(signalPriority(refinement)?.value).toBe('refinement');
  });
});
