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
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { WorkerPool } from '../../../workers/worker-pool';
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
describe('WorkerPool — load balancing', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });
  it('runWithTimeout routes to the least-loaded worker', async () => {
    // Two workers; worker[0] has 5 in-flight queries, worker[1] is idle.
    const w0 = makeFakeWorker('A', 5);
    const w1 = makeFakeWorker('B', 0);
    const pool = makePool([w0, w1]);
    const result = await pool.runWithTimeout(
      'test-op',
      'visibility',
      (api: any) => api.handle()
    );
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
    await pool.runWithTimeout(
      'test-op',
      'visibility',
      (api: any) => api.handle()
    );
    expect(activeDuring).toBe(1);
    expect(w0.activeQueries).toBe(0);
  });
  it('runWithTimeout decrements activeQueries even when the call rejects', async () => {
    const w0 = makeFakeWorker('A', 0);
    const pool = makePool([w0]);
    w0.api.handle.mockRejectedValue(new Error('worker boom'));
    await expect(
      pool.runWithTimeout(
        'test-op',
        'visibility',
        (api: any) => api.handle()
      )
    ).rejects.toThrow('worker boom');
    expect(w0.activeQueries).toBe(0);
  });
  it('with three workers, picks the strictly minimum activeQueries', async () => {
    const w0 = makeFakeWorker('A', 3);
    const w1 = makeFakeWorker('B', 1); // least
    const w2 = makeFakeWorker('C', 2);
    const pool = makePool([w0, w1, w2]);
    const result = await pool.runWithTimeout(
      'test-op',
      'visibility',
      (api: any) => api.handle()
    );
    expect(result).toBe('B');
  });
  it('with all workers tied, picks index 0 (deterministic tie-break)', async () => {
    const w0 = makeFakeWorker('A', 2);
    const w1 = makeFakeWorker('B', 2);
    const pool = makePool([w0, w1]);
    const result = await pool.runWithTimeout(
      'test-op',
      'visibility',
      (api: any) => api.handle()
    );
    expect(result).toBe('A');
  });
});
