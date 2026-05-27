/**
 * Unit tests for the worker-pool load balancing path.
 *
 * Verifies that runWithTimeout routes to the least-loaded worker via
 * getWorkerWithTracking. The previous round-robin via nextWorkerInstance
 * had no load awareness, so a slow worker received every Nth call until
 * each call timed out — head-of-line blocking.
 *
 * The tests bypass real Worker construction by injecting fakes into
 * the pool's `workers` array and forcing `initPromise` to resolved,
 * so we can run in jsdom without a Worker API.
 *
 * AUDIT NOTE (workers.md C3): the `makePool` helper mutates private fields
 * of a real WorkerPool via `as any`. The tests exercise the public
 * `runWithTimeout`/`getStats`/`getQueueDepth` paths but couple to the
 * class internals (`pool.workers`, `pool.initPromise`,
 * `pool.nextWorkerIndex`). The pure helpers `selectLeastBusy`,
 * `nextRoundRobin`, `computeStats`, `computeQueueDepth` would be cleaner
 * to call directly (they live in workers/worker-pool/selection/ +
 * workers/worker-pool/stats.ts). Follow-up: extract direct unit tests
 * for those pure helpers, then keep this file as the integration smoke
 * for the WorkerPool wrapper.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { WorkerPool } from '../../../../../workers/worker-pool';
interface FakeWorkerInstance {
  worker: { terminate: () => void };
  api: { handle: ReturnType<typeof vi.fn> };
  activeQueries: number;
}
function makePool(workers: FakeWorkerInstance[]): WorkerPool {
  const pool = new WorkerPool() as any;
  pool.workers = workers;
  // Skip real initialization; pretend we're already done.
  pool.initPromise = Promise.resolve();
  pool.nextWorkerIndex = 0;
  return pool;
}
function makeFakeWorker(label: string, activeQueries = 0): FakeWorkerInstance {
  return {
    worker: { terminate: vi.fn() },
    api: { handle: vi.fn().mockResolvedValue(label) },
    activeQueries,
  };
}
// workers.md O6 / Phase E37: the `WorkerPool — AbortSignal` describe
// block moved to `workers/worker-pool/abort/abort-signal.test.ts`. Its
// subject is abort plumbing, not least-loaded routing.

describe('WorkerPool — getStats / getQueueDepth', () => {
  it('getQueueDepth sums activeQueries across workers', () => {
    const w0 = makeFakeWorker('A', 3);
    const w1 = makeFakeWorker('B', 5);
    const w2 = makeFakeWorker('C', 0);
    const pool = makePool([w0, w1, w2]);
    expect(pool.getQueueDepth()).toBe(8);
  });

  it('getQueueDepth returns 0 when the pool is uninitialized', () => {
    const pool = makePool([]);
    expect(pool.getQueueDepth()).toBe(0);
  });

  it('getStats returns workerCount, activeQueries, totalActive, peakActive', () => {
    const w0 = makeFakeWorker('A', 3);
    const w1 = makeFakeWorker('B', 5);
    const w2 = makeFakeWorker('C', 0);
    const pool = makePool([w0, w1, w2]);
    const stats = pool.getStats();
    expect(stats.workerCount).toBe(3);
    expect(stats.activeQueries).toEqual([3, 5, 0]);
    expect(stats.totalActive).toBe(8);
    expect(stats.peakActive).toBe(5);
  });
});

describe('WorkerPool — load balancing', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });
  it('runWithTimeout routes to the least-loaded worker', async () => {
    // Two workers; worker[0] has 5 in-flight queries, worker[1] is idle.
    const w0 = makeFakeWorker('A', 5);
    const w1 = makeFakeWorker('B', 0);
    const pool = makePool([w0, w1]);
    const result = await pool.runWithTimeout('test-op', 'visibility', (api: any) => api.handle());
    expect(result).toBe('B');
    expect(w0.api.handle).not.toHaveBeenCalled();
    expect(w1.api.handle).toHaveBeenCalledTimes(1);
  });
  it('runWithTimeout increments activeQueries during the call and decrements after', async () => {
    const w0 = makeFakeWorker('A', 0);
    const pool = makePool([w0]);
    let activeDuring = -1;
    w0.api.handle.mockImplementation(() => {
      activeDuring = w0.activeQueries;
      return Promise.resolve('A');
    });
    await pool.runWithTimeout('test-op', 'visibility', (api: any) => api.handle());
    expect(activeDuring).toBe(1);
    expect(w0.activeQueries).toBe(0);
  });
  it('runWithTimeout decrements activeQueries even when the call rejects', async () => {
    const w0 = makeFakeWorker('A', 0);
    const pool = makePool([w0]);
    w0.api.handle.mockRejectedValue(new Error('worker boom'));
    await expect(
      pool.runWithTimeout('test-op', 'visibility', (api: any) => api.handle())
    ).rejects.toThrow('worker boom');
    expect(w0.activeQueries).toBe(0);
  });
  it('with three workers, picks the strictly minimum activeQueries', async () => {
    const w0 = makeFakeWorker('A', 3);
    const w1 = makeFakeWorker('B', 1); // least
    const w2 = makeFakeWorker('C', 2);
    const pool = makePool([w0, w1, w2]);
    const result = await pool.runWithTimeout('test-op', 'visibility', (api: any) => api.handle());
    expect(result).toBe('B');
  });
  it('with all workers tied, picks index 0 (deterministic tie-break)', async () => {
    const w0 = makeFakeWorker('A', 2);
    const w1 = makeFakeWorker('B', 2);
    const pool = makePool([w0, w1]);
    const result = await pool.runWithTimeout('test-op', 'visibility', (api: any) => api.handle());
    expect(result).toBe('A');
  });
});
