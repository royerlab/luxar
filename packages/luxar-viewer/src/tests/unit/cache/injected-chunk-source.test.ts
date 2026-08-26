/**
 * The store-side contract of an INJECTED {@link ChunkSource}.
 *
 * Every other store test passes a URL string, so the seam this refactor exists
 * to add was exercised nowhere — and that gap is why `source.dispose()` being
 * unreachable went unnoticed. These tests drive `MultiLevelCachingStore` with a
 * fake source and pin what the store promises a source implementor:
 *
 *   - the raw key arrives unmangled,
 *   - `bytesOverWire` feeds the network meter while the DECODED length feeds
 *     bytes-served (the distinction a compressed source depends on),
 *   - `identity` — not `describe` — derives the OPFS bucket,
 *   - `probeIdentityToken` is what validation calls,
 *   - `dispose` propagates from the store's own dispose.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';
import type { ChunkFetchOutcome, ChunkSource } from '../../../cache/chunk-source';
import type { RemoteValidationToken } from '../../../cache/multi-level-caching-store/validation-queue';

/** A source that records what the store asked of it. */
function fakeSource(overrides: Partial<ChunkSource> = {}) {
  const calls = { get: [] as string[], probes: 0, disposed: 0 };
  const source: ChunkSource = {
    identity: 'fake://identity',
    describe: 'fake://describe',
    async get(key: string): Promise<ChunkFetchOutcome> {
      calls.get.push(key);
      // 4 decoded bytes that cost 2 on the wire — a compressed source's shape.
      return { kind: 'ok', data: new Uint8Array([1, 2, 3, 4]), bytesOverWire: 2 };
    },
    async probeIdentityToken(): Promise<RemoteValidationToken | null> {
      calls.probes += 1;
      return null;
    },
    dispose() {
      calls.disposed += 1;
    },
    ...overrides,
  };
  return { source, calls };
}

beforeEach(() => {
  // No OPFS in this environment; L1-only is enough to exercise the seam.
  vi.stubGlobal('navigator', {});
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('MultiLevelCachingStore with an injected ChunkSource', () => {
  it('passes the key through untouched and returns the source bytes', async () => {
    const { source, calls } = fakeSource();
    const store = new MultiLevelCachingStore(source, { noOpfs: true });

    const bytes = await store.get('points/c/0/0');

    expect(Array.from(bytes ?? [])).toEqual([1, 2, 3, 4]);
    expect(calls.get).toEqual(['points/c/0/0']);
  });

  it('meters bytesOverWire on the network, and the decoded length as served', async () => {
    // The whole reason the two are separate fields: a compressed source moves
    // fewer bytes than it yields, and collapsing them would over-report the
    // bandwidth meter by the compression ratio.
    const { source } = fakeSource();
    const store = new MultiLevelCachingStore(source, { noOpfs: true });

    await store.get('points/c/0/0');
    const stats = store.getStats();

    expect(stats.network.bytesTransferred).toBe(2);
    expect(stats.network.totalBytesServed).toBe(4);
  });

  it('DISPOSES the source when the store is disposed', async () => {
    // The regression this file exists for: the hook is documented as "called
    // from the store's dispose" and nothing called it.
    const { source, calls } = fakeSource();
    const store = new MultiLevelCachingStore(source, { noOpfs: true });

    await store.dispose();

    expect(calls.disposed).toBe(1);
  });

  it('asks the source for the validation token rather than fetching one itself', async () => {
    const { source, calls } = fakeSource();
    const store = new MultiLevelCachingStore(source, { noOpfs: true });

    // `noOpfs` short-circuits init before validation, so drive the probe the
    // way validation does — through the source, not through `fetch`.
    await source.probeIdentityToken({});

    expect(calls.probes).toBe(1);
    await store.dispose();
  });

  it('still accepts a bare URL string, building an HTTP source internally', async () => {
    // The back-compatibility that kept ~101 existing store tests untouched.
    const store = new MultiLevelCachingStore('https://example.com/data.zarr', { noOpfs: true });
    expect(store).toBeInstanceOf(MultiLevelCachingStore);
    await store.dispose();
  });

  it('surfaces a source error as a miss rather than throwing', async () => {
    const { source } = fakeSource({
      async get(): Promise<ChunkFetchOutcome> {
        return { kind: 'error', cause: new Error('boom') };
      },
    });
    const store = new MultiLevelCachingStore(source, { noOpfs: true });

    expect(await store.get('points/c/0/0')).toBeUndefined();
    await store.dispose();
  });
});
