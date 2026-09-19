/**
 * The OPFS circuit breaker (#1645): after `opfsTimeoutTripThreshold`
 * CONSECUTIVE per-op timeouts the store disables itself for the session
 * (sticky `opfsRoot = null`), so a systemically stalled OPFS backend
 * (automated Chromium) costs at most N timeouts instead of one full
 * `opfsOperationTimeoutMs` burn per chunk forever. Any non-timeout
 * settlement — success OR a fast rejection — resets the count.
 *
 * Harness conventions follow `opfs-store-correctness.test.ts`: the live
 * config timeout is mutated to a tiny value and restored in `finally`,
 * and the static `pendingDeletesByDataset` registry must end empty
 * (a leaked entry taxes every downstream dispose).
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OPFSStore } from '../../../cache/multi-level-caching-store/opfs-store';
import { createFakeOpfsRoot } from '../../mocks/opfs.mock';
import { config } from '../../../config';
import { log } from '../../../utils/log';

type FsMode = 'ok' | 'hang' | 'reject';

/**
 * Mock OPFS whose behavior is switchable PER CALL via `state.mode`:
 * 'ok' = in-memory FS, 'hang' = the I/O promise never settles (until
 * `releaseHung()`), 'reject' = fast NotFoundError. The mode is read at
 * I/O time (getFile/createWritable/removeEntry), not at handle time, so
 * a test can interleave stalled and responsive operations.
 */
function createSwitchableFS() {
  const files = new Map<string, Uint8Array>();
  const metaFiles = new Map<string, string>();
  const state: { mode: FsMode; hungResolvers: Array<() => void> } = {
    mode: 'ok',
    hungResolvers: [],
  };
  const hangForever = <T>(): Promise<T> =>
    new Promise<T>((resolve) => {
      state.hungResolvers.push(() => resolve(undefined as T));
    });
  const gate = async (): Promise<void> => {
    if (state.mode === 'hang') await hangForever();
    if (state.mode === 'reject') throw new DOMException('Entry not found', 'NotFoundError');
  };

  const mockFileHandle = (path: string) => ({
    async getFile() {
      await gate();
      const data = files.get(path) || new Uint8Array(0);
      return {
        async arrayBuffer() {
          return data.buffer;
        },
        async text() {
          return metaFiles.get(path) || '{}';
        },
      };
    },
    async createWritable() {
      await gate();
      return {
        async write(data: ArrayBuffer | string) {
          if (typeof data === 'string') metaFiles.set(path, data);
          else files.set(path, new Uint8Array(data));
        },
        async close() {},
      };
    },
  });

  const mockDirHandle: {
    getFileHandle: (name: string, opts?: { create?: boolean }) => Promise<unknown>;
    getDirectoryHandle: (name: string, opts?: { create?: boolean }) => Promise<unknown>;
    removeEntry: (name: string) => Promise<void>;
    keys: () => AsyncGenerator<string>;
  } = {
    async getFileHandle(name: string, opts?: { create?: boolean }) {
      if (!files.has(name) && !metaFiles.has(name) && !opts?.create) {
        throw new DOMException(`Entry not found: ${name}`, 'NotFoundError');
      }
      return mockFileHandle(name);
    },
    async getDirectoryHandle() {
      return mockDirHandle;
    },
    async removeEntry(name: string) {
      await gate();
      files.delete(name);
      metaFiles.delete(name);
    },
    async *keys() {},
  };

  const releaseHung = () => {
    for (const r of state.hungResolvers) r();
    state.hungResolvers.length = 0;
  };

  return { mockDirHandle, files, metaFiles, state, releaseHung };
}

describe('OPFSStore circuit breaker', () => {
  let fs: ReturnType<typeof createSwitchableFS>;
  let originalTimeout: number;
  let originalThreshold: number;
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fs = createSwitchableFS();
    // Origin root → `luxar/` → the switchable dataset dir (see opfs.mock.ts).
    createFakeOpfsRoot({
      datasetDir: fs.mockDirHandle,
      onRemoveDataset: () => {
        fs.files.clear();
        fs.metaFiles.clear();
      },
    }).install();
    vi.stubGlobal('crypto', {
      subtle: {
        async digest() {
          return new Uint8Array(32).fill(0xab).buffer;
        },
      },
    });
    originalTimeout = config.cache.opfsOperationTimeoutMs;
    originalThreshold = config.cache.opfsTimeoutTripThreshold;
    config.cache.opfsOperationTimeoutMs = 30;
    config.cache.opfsTimeoutTripThreshold = 3;
    warnSpy = vi.spyOn(log, 'warning');
  });

  afterEach(async () => {
    config.cache.opfsOperationTimeoutMs = originalTimeout;
    config.cache.opfsTimeoutTripThreshold = originalThreshold;
    fs.releaseHung();
    warnSpy.mockRestore();
    vi.unstubAllGlobals();

    // #1390 leaked-delete guard (see opfs-store.test.ts): the static
    // registry must end empty or every downstream dispose pays a tax.
    const registry = (
      OPFSStore as unknown as {
        pendingDeletesByDataset: Map<string, Map<string, Set<Promise<void>>>>;
      }
    ).pendingDeletesByDataset;
    expect(
      registry,
      'OPFSStore.pendingDeletesByDataset was renamed — update this guard'
    ).toBeInstanceOf(Map);
    await vi.waitFor(() => {
      expect(registry.size).toBe(0);
    });
  });

  async function makeStore(name: string): Promise<OPFSStore> {
    const store = new OPFSStore(name, `https://example.com/${name}.zarr`, 100 * 1024 * 1024);
    await store.init();
    return store;
  }

  const breakerWarnings = () =>
    warnSpy.mock.calls.filter(
      (c: unknown[]) => typeof c[1] === 'string' && c[1].includes('circuit breaker tripped')
    ).length;

  it('trips exactly once after N consecutive timeouts, then every op is an instant no-op', async () => {
    const store = await makeStore('breaker-a');
    fs.state.mode = 'hang';
    for (let i = 0; i < 3; i++) {
      await store.set(`k${i}`, new Uint8Array([1]));
    }
    expect(store.getStats().breakerTripped).toBe(true);
    expect(store.getStats().available).toBe(false);
    expect(breakerWarnings()).toBe(1);

    // Post-trip ops must not touch the (hung) backend at all: 10 mixed
    // ops complete in far less than one 30 ms timeout each.
    const t0 = performance.now();
    for (let i = 0; i < 5; i++) {
      await store.set(`post${i}`, new Uint8Array([2]));
      expect(await store.get(`post${i}`)).toBeUndefined();
    }
    expect(performance.now() - t0).toBeLessThan(50);
    expect(breakerWarnings()).toBe(1);
  });

  it('any non-timeout settlement resets the consecutive count (threshold is exact)', async () => {
    const store = await makeStore('breaker-b');

    // Two timeouts — one short of the threshold.
    fs.state.mode = 'hang';
    await store.set('t1', new Uint8Array([1]));
    await store.set('t2', new Uint8Array([1]));
    expect(store.getStats().breakerTripped).toBe(false);

    // A SUCCESS resets the count...
    fs.state.mode = 'ok';
    await store.set('ok1', new Uint8Array([1]));

    // ...so two more timeouts still do not trip...
    fs.state.mode = 'hang';
    await store.set('t3', new Uint8Array([1]));
    await store.set('t4', new Uint8Array([1]));
    expect(store.getStats().breakerTripped).toBe(false);

    // ...and the third consecutive one does.
    await store.set('t5', new Uint8Array([1]));
    expect(store.getStats().breakerTripped).toBe(true);
    expect(breakerWarnings()).toBe(1);
  });

  it('a fast rejection (NotFoundError) also resets the count — the breaker targets stalls, not errors', async () => {
    const store = await makeStore('breaker-b2');
    // Seed an index entry whose read we can gate.
    fs.state.mode = 'ok';
    await store.set('seeded', new Uint8Array([1, 2, 3]));

    fs.state.mode = 'hang';
    await store.set('t1', new Uint8Array([1]));
    await store.set('t2', new Uint8Array([1]));
    expect(store.getStats().breakerTripped).toBe(false);

    // Fast rejection: the read gate throws NotFoundError immediately.
    fs.state.mode = 'reject';
    expect(await store.get('seeded')).toBeUndefined();

    fs.state.mode = 'hang';
    await store.set('t3', new Uint8Array([1]));
    await store.set('t4', new Uint8Array([1]));
    expect(store.getStats().breakerTripped).toBe(false);
    await store.set('t5', new Uint8Array([1]));
    expect(store.getStats().breakerTripped).toBe(true);
  });

  it('tripping with a hung delete in flight strands no pendingDeletes registry entry', async () => {
    const store = await makeStore('breaker-c');
    fs.state.mode = 'ok';
    await store.set('victim', new Uint8Array([1]));

    // delete() registers the real removeEntry in the static registry and
    // returns on its own timeout; the hung op stays outstanding.
    fs.state.mode = 'hang';
    await store.delete('victim');

    // Two more timeouts reach the threshold (the delete's counted too).
    await store.set('t1', new Uint8Array([1]));
    await store.set('t2', new Uint8Array([1]));
    expect(store.getStats().breakerTripped).toBe(true);

    // The registry entry self-cleans when the underlying op finally
    // settles, tripped or not (locked by the afterEach guard as well).
    fs.releaseHung();
    const registry = (
      OPFSStore as unknown as {
        pendingDeletesByDataset: Map<string, unknown>;
      }
    ).pendingDeletesByDataset;
    await vi.waitFor(() => {
      expect(registry.size).toBe(0);
    });
  });

  it('an init-canary timeout degrades the store WITHOUT tripping the breaker', async () => {
    // Hung from the very start: probeWritability() times out inside
    // doInit's catch — unavailable, but that is init-failure degradation,
    // not a breaker trip (the canary site is deliberately not counted).
    fs.state.mode = 'hang';
    const store = new OPFSStore('breaker-d', 'https://example.com/d.zarr', 100 * 1024 * 1024);
    await store.init();
    expect(store.getStats().available).toBe(false);
    expect(store.getStats().breakerTripped).toBe(false);
    expect(breakerWarnings()).toBe(0);
  });

  it('dispose() stays bounded when the backend stalls during its metadata flush', async () => {
    // The teardown flush (awaitInFlight + final save) was the last unbounded
    // OPFS await: a backend that stalls before dispose() would leave it
    // unresolved forever, and the dataset switch waiting to take over the
    // directory with it. It is deadline-bounded, and deliberately NOT counted
    // by the breaker (the store is already dying).
    const store = await makeStore('breaker-f');
    fs.state.mode = 'ok';
    await store.set('k', new Uint8Array([1, 2, 3]));

    fs.state.mode = 'hang';
    const t0 = performance.now();
    await store.dispose();
    // One 30 ms deadline, not forever (an unbounded await fails this by
    // timing the whole test out).
    expect(performance.now() - t0).toBeLessThan(1000);
    expect(store.getStats().available).toBe(false);
    expect(store.getStats().breakerTripped).toBe(false);
  });

  it('clear() does not resurrect a tripped store and stats survive the counter reset', async () => {
    const store = await makeStore('breaker-e');
    fs.state.mode = 'hang';
    for (let i = 0; i < 3; i++) await store.set(`k${i}`, new Uint8Array([1]));
    expect(store.getStats().breakerTripped).toBe(true);

    fs.state.mode = 'ok';
    await store.clear();
    expect(store.getStats().breakerTripped).toBe(true);
    expect(store.getStats().available).toBe(false);
    // Still an instant no-op after clear().
    await store.set('after-clear', new Uint8Array([1]));
    expect(await store.get('after-clear')).toBeUndefined();
  });
});
