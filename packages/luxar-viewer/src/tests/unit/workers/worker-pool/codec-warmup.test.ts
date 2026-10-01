// @vitest-environment jsdom

/**
 * The data workers' blosc codec is warmed LAZILY, one worker first.
 *
 * Warming a worker's codec downloads the worker bundle's own blosc chunk
 * (~600 KB). Doing it for every worker the moment it came ready meant 15
 * concurrent downloads (~9.4 MB) competing with the scene metadata on a hosted
 * link: the sp64 gate store's cold first frame went from 3.6 s to 6.6 s — on a
 * store none of whose chunks is big enough to be offloaded at all.
 *
 * The contract pinned here: nothing is warmed when the pool comes up; the first
 * decode that clears the offload floor warms exactly ONE worker (and decodes on
 * the main thread meanwhile); the rest warm only once that one finished (so
 * they hit the HTTP cache); `?mainThreadCodecs` never warms; and a warm-up is
 * not a query, so it never makes a worker look busy to dispatch.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

function deferred<T = void>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => (resolve = res));
  return { promise, resolve };
}

/** A blosc-framed stand-in: only the header's decoded size (bytes 4..7) matters. */
function frame(decodedBytes: number): Uint8Array {
  const bytes = new Uint8Array(32);
  new DataView(bytes.buffer).setUint32(4, decodedBytes, true);
  return bytes;
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

async function loadPool(workerCount: number) {
  vi.resetModules();
  vi.stubGlobal('navigator', { hardwareConcurrency: 32 });
  vi.doMock('../../../../wasm/shared-module', () => ({
    getSharedWasmModule: vi.fn(async () => null),
    resetSharedWasmModule: vi.fn(),
  }));
  vi.doMock('../../../../config', () => ({
    config: {
      dataLoading: {
        performance: {
          workerCount,
          workerProjectionTimeoutMs: 0,
          workerInitTimeoutMs: 0,
        },
      },
    },
  }));
  vi.doMock('../../../../utils/log', () => ({
    log: { info: vi.fn(), warning: vi.fn(), error: vi.fn(), success: vi.fn(), update: vi.fn() },
    Modules: { WORKER_POOL: 'WorkerPool' },
  }));

  /** Per-worker warm-up gates, settled by the test. */
  const warmGates: ReturnType<typeof deferred>[] = [];
  const warmCalls: number[] = [];
  const decodeCalls: number[] = [];
  let nextWrap = 0;
  vi.doMock('comlink', () => ({
    transfer: <T>(value: T) => value,
    wrap: vi.fn(() => {
      const index = nextWrap++;
      warmGates[index] = deferred();
      return {
        initialize: async () => ({ wasmFallback: false }),
        warmCodecs: () => {
          warmCalls.push(index);
          return warmGates[index].promise;
        },
        decodeBloscBatch: async (items: { bytes: Uint8Array }[]) => {
          decodeCalls.push(index);
          return items.map((item) => ({ data: item.bytes }));
        },
      };
    }),
  }));
  vi.doMock('../../../../workers/data-worker?worker', () => ({
    default: class MockDataWorker {
      terminate = vi.fn();
      onerror = null;
      onmessageerror = null;
    },
  }));

  const poolModule = await import('../../../../workers/worker-pool');
  const dispatch = await import('../../../../workers/worker-pool/codec-dispatch');
  const blosc = await import('../../../../data/codecs/worker-blosc');
  return { ...poolModule, ...dispatch, blosc, warmGates, warmCalls, decodeCalls };
}

describe('WorkerPool — lazy, one-worker-first codec warm-up', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('sends no codec warm-up to any worker when the pool comes up', async () => {
    const { WorkerPool, warmCalls } = await loadPool(3);
    const pool = new WorkerPool();
    await pool.initialize();
    await flush();
    expect(pool.getWorkerCount()).toBe(3);
    expect(warmCalls).toEqual([]);
    pool.dispose();
  });

  it('the first above-floor decode warms ONE worker; the rest only after it is warm', async () => {
    const { WorkerPool, BloscDecodeDispatcher, MIN_OFFLOAD_DECODED_BYTES, warmGates, warmCalls } =
      await loadPool(3);
    const pool = new WorkerPool();
    await pool.initialize();
    const dispatcher = new BloscDecodeDispatcher(pool);

    // Below the floor: never offloaded, never warms anything.
    expect(dispatcher.decode({ bytes: frame(1024), delta: null })).toBeNull();
    expect(warmCalls).toEqual([]);

    // Above the floor, nothing warm yet: decode on the main thread, warm one.
    const big = () => dispatcher.decode({ bytes: frame(MIN_OFFLOAD_DECODED_BYTES), delta: null });
    expect(big()).toBeNull();
    expect(big()).toBeNull();
    await flush();
    expect(warmCalls).toHaveLength(1);

    warmGates[warmCalls[0]].resolve(undefined);
    await flush();
    expect(warmCalls).toHaveLength(3);
    expect(new Set(warmCalls).size).toBe(3);

    // A warm worker exists: decodes are offloaded now.
    const job = big();
    expect(job).not.toBeNull();
    await job;
    pool.dispose();
  });

  it('offloads only to WARM workers: one still fetching its codec gets no decode', async () => {
    const {
      WorkerPool,
      BloscDecodeDispatcher,
      MIN_OFFLOAD_DECODED_BYTES,
      warmGates,
      warmCalls,
      decodeCalls,
    } = await loadPool(3);
    const pool = new WorkerPool();
    await pool.initialize();
    const dispatcher = new BloscDecodeDispatcher(pool);
    const big = () => dispatcher.decode({ bytes: frame(MIN_OFFLOAD_DECODED_BYTES), delta: null });

    expect(big()).toBeNull(); // starts warming ONE worker
    await flush();
    const first = warmCalls[0];
    warmGates[first].resolve(undefined);
    await flush();
    // The other two are warming now; their gates stay open (codec still downloading).
    expect(warmCalls).toHaveLength(3);

    // A burst the dispatcher spreads over "idle" workers: every batch must still
    // land on the one warm worker, not queue behind a codec download.
    const jobs = Array.from({ length: 6 }, big);
    expect(jobs.every((job) => job !== null)).toBe(true);
    await Promise.all(jobs.filter((job) => job !== null));
    expect(decodeCalls.length).toBeGreaterThan(0);
    expect(new Set(decodeCalls)).toEqual(new Set([first]));
    pool.dispose();
  });

  it('keeps concurrent decodes on the warm worker while the others warm', async () => {
    const { WorkerPool, warmGates, warmCalls, decodeCalls } = await loadPool(3);
    const pool = new WorkerPool();
    await pool.initialize();
    pool.warmCodecs();
    await flush();
    const first = warmCalls[0];
    warmGates[first].resolve(undefined);
    await flush();
    expect(warmCalls).toHaveLength(3);

    const run = () =>
      pool.runDecode('decodeBloscBatch', (api) =>
        api.decodeBloscBatch([{ bytes: new Uint8Array(1), delta: null }])
      );
    await Promise.all([run(), run(), run()]);
    expect(decodeCalls).toHaveLength(3);
    expect(new Set(decodeCalls)).toEqual(new Set([first]));
    pool.dispose();
  });

  it('never warms a worker codec under the ?mainThreadCodecs kill switch', async () => {
    const { WorkerPool, BloscDecodeDispatcher, MIN_OFFLOAD_DECODED_BYTES, blosc, warmCalls } =
      await loadPool(2);
    blosc.setWorkerCodecsEnabled(false);
    try {
      const pool = new WorkerPool();
      await pool.initialize();
      pool.warmCodecs();
      expect(
        new BloscDecodeDispatcher(pool).decode({
          bytes: frame(MIN_OFFLOAD_DECODED_BYTES),
          delta: null,
        })
      ).toBeNull();
      await flush();
      expect(warmCalls).toEqual([]);
      pool.dispose();
    } finally {
      blosc.setWorkerCodecsEnabled(true);
    }
  });

  it('a warm-up in flight does not count as a query (workers stay idle)', async () => {
    const { WorkerPool, warmCalls } = await loadPool(2);
    const pool = new WorkerPool();
    await pool.initialize();
    pool.warmCodecs();
    await flush();
    expect(warmCalls.length).toBeGreaterThan(0);
    expect(pool.getIdleWorkerCount()).toBe(2);
    expect(pool.getQueueDepth()).toBe(0);
    pool.dispose();
  });
});
