/**
 * Worker slot accounting (B7): a worker's `activeQueries` slot must stay busy
 * until the WORKER'S OWN promise settles, not until the caller's race settles.
 *
 * Measured on real playback: an aborted call freed its slot while the worker
 * was still running the abandoned task, and concurrent dispatches all saw
 * index 0 as idle, so 35-79% of dispatches went to a busy worker while
 * others idled. Real WorkerPool with fake workers (same private-field
 * injection as the sibling load-balancing / abort tests).
 */
import { describe, it, expect, vi } from 'vitest';
import { WorkerPool } from '../../../../../workers/worker-pool';
import { deferred } from '../../../../helpers/deferred';

interface FakeWorkerInstance {
  worker: { terminate: () => void };
  api: { handle: ReturnType<typeof vi.fn> };
  activeQueries: number;
}

function makePool(workers: FakeWorkerInstance[]): WorkerPool {
  const pool = new WorkerPool() as any;
  pool.workers = workers;
  pool.initPromise = Promise.resolve();
  return pool;
}

function makeFakeWorker(label: string): FakeWorkerInstance {
  return {
    worker: { terminate: vi.fn() },
    api: { handle: vi.fn().mockResolvedValue(label) },
    activeQueries: 0,
  };
}

async function flush(): Promise<void> {
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

describe('WorkerPool — slot accounting follows the worker, not the caller', () => {
  it('an aborted call keeps its slot busy until the worker settles', async () => {
    const w0 = makeFakeWorker('A');
    const task = deferred<string>();
    w0.api.handle.mockReturnValue(task.promise);
    const pool = makePool([w0]);
    const controller = new AbortController();

    const call = pool.runWithTimeout(
      'op',
      'projection',
      (api: any) => api.handle(),
      controller.signal
    );
    await flush();
    controller.abort();
    await expect(call).rejects.toMatchObject({ name: 'WorkerAbortError' });

    // The worker is still running the abandoned task.
    expect(w0.activeQueries).toBe(1);

    task.resolve('A');
    await flush();
    expect(w0.activeQueries).toBe(0);
  });

  it('a worker that REJECTS after the caller aborted still frees its slot (no unhandled rejection)', async () => {
    const w0 = makeFakeWorker('A');
    const task = deferred<string>();
    w0.api.handle.mockReturnValue(task.promise);
    const pool = makePool([w0]);
    const controller = new AbortController();

    const call = pool.runWithTimeout(
      'op',
      'projection',
      (api: any) => api.handle(),
      controller.signal
    );
    await flush();
    controller.abort();
    await expect(call).rejects.toMatchObject({ name: 'WorkerAbortError' });

    task.reject(new Error('late worker failure'));
    await flush();
    expect(w0.activeQueries).toBe(0);
  });

  it('the dispatch after an aborted call goes to an idle worker, not the one still running', async () => {
    const w0 = makeFakeWorker('A');
    const w1 = makeFakeWorker('B');
    w0.api.handle.mockReturnValue(new Promise(() => {})); // abandoned task keeps running
    const pool = makePool([w0, w1]);
    const controller = new AbortController();

    const first = pool.runWithTimeout(
      'op',
      'projection',
      (api: any) => api.handle(),
      controller.signal
    );
    await flush();
    controller.abort();
    await expect(first).rejects.toMatchObject({ name: 'WorkerAbortError' });

    const second = await pool.runWithTimeout('op', 'projection', (api: any) => api.handle());
    expect(second).toBe('B');
    expect(w0.api.handle).toHaveBeenCalledTimes(1);
  });

  it('concurrent dispatches spread across idle workers (the slot is taken at selection)', async () => {
    const w0 = makeFakeWorker('A');
    const w1 = makeFakeWorker('B');
    const pool = makePool([w0, w1]);

    const results = await Promise.all([
      pool.runWithTimeout('op', 'projection', (api: any) => api.handle()),
      pool.runWithTimeout('op', 'projection', (api: any) => api.handle()),
    ]);
    expect(results.sort()).toEqual(['A', 'B']);
  });

  it('a call aborted between selection and dispatch releases the slot it took', async () => {
    const w0 = makeFakeWorker('A');
    const pool = makePool([w0]);
    const controller = new AbortController();
    const call = pool.runWithTimeout(
      'op',
      'projection',
      (api: any) => api.handle(),
      controller.signal
    );
    controller.abort(); // before the whenUsable await resumes
    await expect(call).rejects.toMatchObject({ name: 'WorkerAbortError' });
    expect(w0.api.handle).not.toHaveBeenCalled();
    expect(w0.activeQueries).toBe(0);
  });
});
