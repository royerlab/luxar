/**
 * The WorkerPool's two-gate startup contract.
 *
 * `initialize()` resolves only once every spawn has settled — that is what
 * callers who want a full pool (or an honest final worker count) rely on. But
 * the HOT path does not need a full pool: the first LOD rung of a scene is
 * ~1024 splats and needs one worker. Measured on a hosted demo, waiting for
 * all 15 cost 2.23 s of an 8.0 s first paint, because each worker compiles its
 * own copy of the WASM module and they come ready ~160 ms apart.
 *
 * So `getWorkerWithTracking()` / `getWorker()` await a second gate that settles
 * on the FIRST published worker. These tests pin both halves of that split, and
 * the failure modes incremental publishing introduces.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

type MockWorker = {
  index: number;
  terminate: ReturnType<typeof vi.fn>;
  onerror: ((e: { message?: string; preventDefault?: () => void }) => void) | null;
  onmessageerror: (() => void) | null;
};

/** A promise plus the handles to settle it from the test body. */
function deferred<T = void>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/**
 * Load a fresh pool whose per-worker `initialize()` is controlled by the test.
 * `initByIndex(i)` is called once per spawned worker, in construction order.
 */
async function loadWorkerPool(
  workerCount: number,
  initByIndex: (index: number) => Promise<unknown>,
  sharedModule: WebAssembly.Module | null = null
) {
  vi.resetModules();

  // The pool resolves the shared, main-thread-compiled WASM module before each
  // worker's init guard starts. Stubbed here so the tests neither touch the
  // network nor depend on a built artifact.
  vi.doMock('../../../../../wasm/shared-module', () => ({
    getSharedWasmModule: vi.fn(async () => sharedModule),
    resetSharedWasmModule: vi.fn(),
  }));

  const log = {
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    success: vi.fn(),
    update: vi.fn(),
  };
  const workers: MockWorker[] = [];
  let nextWorkerIndex = 0;
  // Comlink's `wrap` is called once per worker, in the same order the workers
  // are constructed, so the wrap counter identifies which worker it belongs to.
  let nextWrapIndex = 0;

  vi.doMock('../../../../../config', () => ({
    config: {
      dataLoading: {
        performance: {
          workerCount,
          workerProjectionTimeoutMs: 60000,
          workerInitTimeoutMs: 0, // disabled — tests control settlement directly
        },
      },
    },
  }));

  vi.doMock('../../../../../utils/log', () => ({
    log,
    Modules: { WORKER_POOL: 'WorkerPool' },
  }));

  const initCalls: unknown[][] = [];
  vi.doMock('comlink', () => ({
    wrap: vi.fn(() => {
      const index = nextWrapIndex++;
      return {
        initialize: (...args: unknown[]) => {
          initCalls.push(args);
          return initByIndex(index);
        },
      };
    }),
  }));

  vi.doMock('../../../../../workers/data-worker?worker', () => ({
    default: class MockDataWorker {
      index = nextWorkerIndex++;
      terminate = vi.fn();
      onerror: MockWorker['onerror'] = null;
      onmessageerror: MockWorker['onmessageerror'] = null;
      constructor() {
        workers.push(this as unknown as MockWorker);
      }
    },
  }));

  const module = await import('../../../../../workers/worker-pool');
  return { ...module, log, workers, initCalls };
}

/** Let queued microtasks drain without advancing time. */
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('WorkerPool — first-usable-worker gate', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('hands out a worker as soon as the FIRST one is ready', async () => {
    const stragglers = deferred();
    const { WorkerPool } = await loadWorkerPool(4, async (index) => {
      if (index === 0) return { wasmFallback: false };
      await stragglers.promise;
      return { wasmFallback: false };
    });
    const pool = new WorkerPool();

    const handle = await pool.getWorkerWithTracking();

    // The whole point: usable while 3 of 4 spawns are still parked.
    expect(handle).toBeTruthy();
    expect(pool.getWorkerCount()).toBe(1);

    stragglers.resolve();
    await pool.initialize();
    expect(pool.getWorkerCount()).toBe(4);
    pool.dispose();
  });

  it('initialize() still resolves only after every spawn has settled', async () => {
    const stragglers = deferred();
    const { WorkerPool } = await loadWorkerPool(4, async (index) => {
      if (index === 0) return { wasmFallback: false };
      await stragglers.promise;
      return { wasmFallback: false };
    });
    const pool = new WorkerPool();

    const sentinel = Symbol('pending');
    const raced = await Promise.race([
      pool.initialize().then(() => 'settled'),
      flush().then(() => sentinel),
    ]);
    // The public contract did NOT move: one ready worker must not resolve it.
    expect(raced).toBe(sentinel);

    stragglers.resolve();
    await pool.initialize();
    expect(pool.getWorkerCount()).toBe(4);
    pool.dispose();
  });

  it('never publishes a worker that becomes ready after dispose()', async () => {
    const stragglers = deferred();
    const { WorkerPool } = await loadWorkerPool(3, async () => {
      await stragglers.promise;
      return { wasmFallback: false };
    });
    const pool = new WorkerPool();

    void pool.initialize().catch(() => {});
    await flush();
    pool.dispose();

    // Every worker resolves AFTER the generation was bumped.
    stragglers.resolve();
    await flush();
    await flush();

    expect(pool.getWorkerCount()).toBe(0);
    pool.dispose();
  });

  it('unblocks a caller parked on the gate when the pool is disposed mid-init', async () => {
    const stragglers = deferred();
    const { WorkerPool } = await loadWorkerPool(2, async () => {
      await stragglers.promise;
      return { wasmFallback: false };
    });
    const pool = new WorkerPool();

    const pending = pool.getWorkerWithTracking();
    await flush();
    pool.dispose();

    // Without a settle on the dispose path this rejects never — the caller
    // simply hangs for the lifetime of the tab.
    await expect(pending).rejects.toThrow(/disposed|No workers available/);
    stragglers.resolve();
  });

  it('rejects fast, and stays rejected, when every spawn fails', async () => {
    const { WorkerPool, workers } = await loadWorkerPool(3, async () => {
      throw new Error('boom');
    });
    const pool = new WorkerPool();

    await expect(pool.getWorkerWithTracking()).rejects.toThrow(/Failed to initialize any data/);
    const spawnedAfterFirst = workers.length;

    // The sticky-rejection contract: a second call must not re-spawn.
    await expect(pool.getWorkerWithTracking()).rejects.toThrow();
    expect(workers.length).toBe(spawnedAfterFirst);
    pool.dispose();
  });
});

describe('WorkerPool — shared WASM module', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('forwards the shared module to every worker init', async () => {
    const shared = { __fake: 'module' } as unknown as WebAssembly.Module;
    const { WorkerPool, initCalls } = await loadWorkerPool(
      3,
      async () => ({ wasmFallback: false }),
      shared
    );
    const pool = new WorkerPool();

    await pool.initialize();

    // Every worker instantiates from the SAME module the main thread compiled
    // once, instead of compiling its own copy.
    expect(initCalls).toHaveLength(3);
    for (const args of initCalls) {
      expect(args[1]).toBe(shared);
    }
    pool.dispose();
  });

  it('passes undefined when no shared module could be compiled', async () => {
    const { WorkerPool, initCalls } = await loadWorkerPool(
      2,
      async () => ({ wasmFallback: false }),
      null
    );
    const pool = new WorkerPool();

    await pool.initialize();

    // The degradation path: each worker self-initializes exactly as before.
    expect(initCalls).toHaveLength(2);
    for (const args of initCalls) {
      expect(args[1]).toBeUndefined();
    }
    pool.dispose();
  });
});

describe('WorkerPool — incremental publishing', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('keeps filling when the first worker dies while spawns are in flight', async () => {
    const stragglers = deferred();
    const { WorkerPool, workers } = await loadWorkerPool(3, async (index) => {
      if (index === 0) return { wasmFallback: false };
      await stragglers.promise;
      return { wasmFallback: false };
    });
    const pool = new WorkerPool();

    const init = pool.initialize();
    await flush();
    expect(pool.getWorkerCount()).toBe(1);

    // Kill the only published worker. With batch publishing this state was
    // unreachable, so `'pool-empty'` unambiguously meant "attempt over". It no
    // longer does — and clearing the init promise here would let the next
    // initialize() bump the generation and orphan the two workers in flight.
    workers[0].onerror?.({ message: 'died', preventDefault: () => {} });
    expect(pool.getWorkerCount()).toBe(0);

    // A later hot-path call arriving in this window must JOIN the running
    // attempt, not start a rival one. If `'pool-empty'` had cleared the init
    // promise, this would bump the generation and the two in-flight workers
    // would self-terminate on publish — the pool would end up with 0 of 3.
    void pool.initialize().catch(() => {});
    expect(workers.length).toBe(3);

    stragglers.resolve();
    await init;

    expect(pool.getWorkerCount()).toBe(2);
    // Exactly the original three were ever constructed: no second generation.
    expect(workers.length).toBe(3);
    pool.dispose();
  });

  it('a caller whose first worker died waits out the rest of the attempt', async () => {
    const stragglers = deferred();
    const { WorkerPool, workers } = await loadWorkerPool(3, async (index) => {
      if (index === 0) return { wasmFallback: false };
      await stragglers.promise;
      return { wasmFallback: false };
    });
    const pool = new WorkerPool();

    void pool.initialize().catch(() => {});
    await flush();
    workers[0].onerror?.({ message: 'died', preventDefault: () => {} });

    // Gate already settled (worker 0 published), but the pool is empty right
    // now. Falling straight through to the empty-pool throw would abandon two
    // workers that are microtasks away from ready.
    const pending = pool.getWorkerWithTracking();
    stragglers.resolve();

    await expect(pending).resolves.toBeTruthy();
    expect(pool.getWorkerCount()).toBe(2);
    pool.dispose();
  });
});
