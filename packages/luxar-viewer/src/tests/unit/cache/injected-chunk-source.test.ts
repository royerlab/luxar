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
import { hashUrl } from '../../../cache/multi-level-caching-store/fetch-retry';
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

  it('asks the SOURCE for the validation token during init, not the network', async () => {
    // Previously this called `source.probeIdentityToken` itself and counted its
    // own call — `noOpfs` returns from init before validation ever runs, so the
    // store was not involved at all. Give it a fake OPFS so init proceeds.
    const dir = {
      getFileHandle: vi.fn(async () => {
        throw new DOMException('not found', 'NotFoundError');
      }),
      getDirectoryHandle: vi.fn(async () => dir),
      removeEntry: vi.fn(async () => undefined),
      keys: async function* () {},
    };
    vi.stubGlobal('navigator', { storage: { getDirectory: async () => dir } });

    const { source, calls } = fakeSource();
    const store = new MultiLevelCachingStore(source, {});
    await store.init();

    // Exactly one: ValidationQueue serializes per dataset, and `>= 1` would
    // sail past a regression that probed twice.
    expect(calls.probes).toBe(1);

    // The OPFS bucket must come from `identity`, never `describe` — the two
    // differ for a container whose identity is not its label, and nothing else
    // in the repo would notice the swap.
    expect(dir.getDirectoryHandle).toHaveBeenCalledWith(await hashUrl('fake://identity'), {
      create: true,
    });
    expect(dir.getDirectoryHandle).not.toHaveBeenCalledWith(await hashUrl('fake://describe'), {
      create: true,
    });
    await store.dispose();
  });

  it('still accepts a bare URL string, building an HTTP source internally', async () => {
    // The back-compatibility that kept ~101 existing store tests untouched.
    // Asserting the instance type would pass for an argument that was thrown
    // away — what "built an HTTP source" means is that the key is fetched
    // under the base URL.
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => ({
      ok: true,
      status: 200,
      statusText: '',
      headers: new Headers(),
      body: null,
      arrayBuffer: async () => new ArrayBuffer(4),
    }));
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    const store = new MultiLevelCachingStore('https://example.com/data.zarr', { noOpfs: true });
    await store.get('c/0/0');

    expect(fetchMock.mock.calls[0]?.[0]).toBe('https://example.com/data.zarr/c/0/0');
    await store.dispose();
  });

  it('RETHROWS a fatal outcome instead of degrading it to a miss', async () => {
    // The distinction that matters: `error` means one chunk failed and a
    // fill-valued read is acceptable; `fatal` means the whole container is
    // unreadable. Reporting the second as a miss renders an empty scene and
    // swallows the diagnosis the error was written to deliver.
    const boom = new Error('archive is not range-readable');
    const { source } = fakeSource({
      async get(): Promise<ChunkFetchOutcome> {
        return { kind: 'fatal', cause: boom };
      },
    });
    const store = new MultiLevelCachingStore(source, { noOpfs: true });

    await expect(store.get('points/c/0/0')).rejects.toBe(boom);
    await store.dispose();
  });

  it('REJECTS on an aborted outcome, so an invalidated read cannot become fill values', async () => {
    // An invalidation fired precisely because those bytes must not be trusted,
    // so an aborted read must reject rather than becoming fill values.
    const { source } = fakeSource({
      async get(): Promise<ChunkFetchOutcome> {
        return { kind: 'aborted' };
      },
    });
    const store = new MultiLevelCachingStore(source, { noOpfs: true });

    await expect(store.get('points/c/0/0')).rejects.toThrow(/aborted during invalidation/);
    await store.dispose();
  });

  it('throws a source error so zarrita cannot synthesize fill values', async () => {
    const { source } = fakeSource({
      async get(): Promise<ChunkFetchOutcome> {
        return { kind: 'error', cause: new Error('boom') };
      },
    });
    const store = new MultiLevelCachingStore(source, { noOpfs: true });

    await expect(store.get('points/c/0/0')).rejects.toThrow('boom');
    await store.dispose();
  });
});
