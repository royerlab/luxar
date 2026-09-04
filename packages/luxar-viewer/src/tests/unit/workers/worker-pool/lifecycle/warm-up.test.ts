/**
 * `warmUpDataWorkerPool()` — the fire-and-forget early start.
 *
 * Called from `loadScene` so worker startup overlaps the scene-metadata fetch
 * instead of being paid for by the first chunk decode. Two properties matter
 * and neither is obvious: it must not run where there is no `Worker` global
 * (or it would latch the pool's deliberately-sticky rejected `initPromise`),
 * and a failed warm-up must not soften that stickiness for the real caller.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

type MockWorker = { terminate: ReturnType<typeof vi.fn> };

async function loadWorkerPool(initializeImpl: () => Promise<unknown>) {
  vi.resetModules();

  const log = {
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    success: vi.fn(),
    update: vi.fn(),
  };
  const workers: MockWorker[] = [];

  vi.doMock('../../../../../config', () => ({
    config: {
      dataLoading: {
        performance: {
          workerCount: 2,
          workerProjectionTimeoutMs: 60000,
          workerInitTimeoutMs: 0,
        },
      },
    },
  }));

  vi.doMock('../../../../../utils/log', () => ({
    log,
    Modules: { WORKER_POOL: 'WorkerPool' },
  }));

  vi.doMock('comlink', () => ({
    wrap: vi.fn(() => ({ initialize: initializeImpl })),
  }));

  vi.doMock('../../../../../workers/data-worker?worker', () => ({
    default: class MockDataWorker {
      terminate = vi.fn();
      onerror: unknown = null;
      onmessageerror: unknown = null;
      constructor() {
        workers.push(this as unknown as MockWorker);
      }
    },
  }));

  const module = await import('../../../../../workers/worker-pool');
  return { ...module, log, workers };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('warmUpDataWorkerPool', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it('is a no-op when there is no Worker global', async () => {
    const { warmUpDataWorkerPool, getWorkerPool, disposeWorkerPool, workers } =
      await loadWorkerPool(async () => ({ wasmFallback: false }));
    vi.stubGlobal('Worker', undefined);

    warmUpDataWorkerPool();
    await flush();

    // Nothing spawned — and crucially the pool was not poisoned: a real
    // initialize() afterwards still runs a fresh attempt.
    expect(workers).toHaveLength(0);
    vi.unstubAllGlobals();
    await getWorkerPool().initialize();
    expect(workers.length).toBeGreaterThan(0);
    disposeWorkerPool();
  });

  it('starts the pool without being awaited', async () => {
    const { warmUpDataWorkerPool, getWorkerPool, disposeWorkerPool, workers } =
      await loadWorkerPool(async () => ({ wasmFallback: false }));
    vi.stubGlobal('Worker', class {});

    // Returns void, synchronously.
    expect(warmUpDataWorkerPool()).toBeUndefined();
    await flush();
    expect(workers.length).toBeGreaterThan(0);

    // Idempotent: the later real call joins the same attempt.
    const spawnedByWarmUp = workers.length;
    await getWorkerPool().initialize();
    expect(workers.length).toBe(spawnedByWarmUp);
    disposeWorkerPool();
  });

  it('swallows a failed warm-up without softening the sticky rejection', async () => {
    const { warmUpDataWorkerPool, getWorkerPool, disposeWorkerPool } = await loadWorkerPool(
      async () => {
        throw new Error('boom');
      }
    );
    vi.stubGlobal('Worker', class {});

    // No unhandled rejection may escape (vitest fails the run on one).
    warmUpDataWorkerPool();
    await flush();
    await flush();

    // The real caller still sees the failure — warm-up must not mask it.
    await expect(getWorkerPool().initialize()).rejects.toThrow(/Failed to initialize any data/);
    disposeWorkerPool();
  });
});
