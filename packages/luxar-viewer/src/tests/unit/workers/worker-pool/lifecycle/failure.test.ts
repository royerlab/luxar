import { beforeEach, describe, expect, it, vi } from 'vitest';

type MockWorker = {
  index: number;
  terminate: ReturnType<typeof vi.fn>;
};

async function loadWorkerPoolWithInitializationResults(results: Array<'success' | 'fail'>) {
  vi.resetModules();

  const log = {
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    success: vi.fn(),
    update: vi.fn(),
  };
  const workers: MockWorker[] = [];
  let nextWorkerIndex = 0;

  vi.doMock('../../../../../config', () => ({
    config: {
      dataLoading: {
        performance: {
          workerCount: results.length,
        },
      },
    },
  }));

  vi.doMock('../../../../../utils/log', () => ({
    log,
    Modules: { WORKER_POOL: 'WorkerPool' },
  }));

  vi.doMock('comlink', () => ({
    wrap: vi.fn((worker: MockWorker) => ({
      initialize: vi.fn(async () => {
        if (results[worker.index] === 'fail') {
          throw new Error(`worker ${worker.index} failed`);
        }
      }),
    })),
  }));

  vi.doMock('../../../../../workers/data-worker?worker', () => ({
    default: class MockDataWorker {
      index = nextWorkerIndex++;
      terminate = vi.fn();

      constructor() {
        workers.push(this);
      }
    },
  }));

  const module = await import('../../../../../workers/worker-pool');
  return { ...module, log, workers };
}

describe('WorkerPool initialization failure handling', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.stubGlobal('navigator', { hardwareConcurrency: 16 });
  });

  it('throws, terminates workers, and remains retryable when all workers fail', async () => {
    const first = await loadWorkerPoolWithInitializationResults(['fail', 'fail', 'fail']);
    const pool = new first.WorkerPool();

    await expect(pool.initialize()).rejects.toThrow('Failed to initialize any data workers');

    expect(pool.isInitialized()).toBe(false);
    expect(first.workers).toHaveLength(3);
    expect(first.workers.every((worker) => worker.terminate.mock.calls.length === 1)).toBe(true);

    const second = await loadWorkerPoolWithInitializationResults(['success', 'success']);
    const retryPool = new second.WorkerPool();

    await expect(retryPool.initialize()).resolves.toBeUndefined();
    expect(retryPool.getWorkerCount()).toBe(2);
  });

  it('MED-22: empty-pool throw leaves no pending workers behind', async () => {
    // spawnWorker.delete(pendingWorkers, worker) runs on BOTH success and
    // failure paths, so by the time `Promise.allSettled(workerPromises)`
    // returns, `pendingWorkers` is guaranteed empty. The empty-pool
    // throw on line 171 of worker-pool.ts must not leak workers — the
    // catch handler below it walks `pendingWorkers` AFTER the throw, so
    // a defensive sweep / verified invariant in the throw branch is the
    // only guarantee that an in-flight worker cannot escape the guard.
    const { WorkerPool, workers } = await loadWorkerPoolWithInitializationResults(['fail', 'fail']);
    const pool = new WorkerPool();

    await expect(pool.initialize()).rejects.toThrow('Failed to initialize any data workers');

    // All spawned workers terminated — none leaked through the empty
    // guard. Use the internal pendingWorkers Set to assert directly.
    expect(workers).toHaveLength(2);
    expect(workers.every((worker) => worker.terminate.mock.calls.length === 1)).toBe(true);
    expect((pool as unknown as { pendingWorkers: Set<unknown> }).pendingWorkers.size).toBe(0);
  });

  it('continues with reduced capacity when some workers fail', async () => {
    const { WorkerPool, log, workers } = await loadWorkerPoolWithInitializationResults([
      'success',
      'fail',
      'success',
    ]);
    const pool = new WorkerPool();

    await pool.initialize();

    expect(pool.getWorkerCount()).toBe(2);
    expect(workers).toHaveLength(3);
    expect(workers[1].terminate).toHaveBeenCalledTimes(1);
    expect(workers[0].terminate).not.toHaveBeenCalled();
    expect(workers[2].terminate).not.toHaveBeenCalled();
    expect(log.warning).toHaveBeenCalledWith(
      'WorkerPool',
      'Only 2/3 workers initialized successfully'
    );
  });
});
