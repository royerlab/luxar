/**
 * Unit tests for the worker-pool AbortSignal handling.
 *
 * workers.md O6 / Phase E37: moved here from
 * `workers/worker-pool/selection/load-balancing.test.ts` where it was
 * mis-filed (the load-balancing file's subject is least-loaded routing,
 * not abort plumbing).
 *
 * The tests bypass real Worker construction by injecting fakes into
 * the pool's `workers` array and forcing `initPromise` to resolved,
 * so we can run in jsdom without a Worker API.
 *
 * AUDIT NOTE (workers.md C3): the `makePool` helper mutates private
 * fields of a real WorkerPool via `as any`. Same caveat as the
 * load-balancing file — the helper is duped here intentionally rather
 * than extracted to a shared module; both files are slim and the
 * shape is unlikely to change.
 */
import { getEventListeners } from 'node:events';
import { describe, it, expect, vi } from 'vitest';
import { WorkerPool } from '../../../../../workers/worker-pool';

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

function makeFakeWorker(label: string, activeQueries = 0): FakeWorkerInstance {
  return {
    worker: { terminate: vi.fn() },
    api: { handle: vi.fn().mockResolvedValue(label) },
    activeQueries,
  };
}

describe('WorkerPool — AbortSignal', () => {
  it('rejects immediately when called with an already-aborted signal', async () => {
    const w0 = makeFakeWorker('A');
    const pool = makePool([w0]);
    const controller = new AbortController();
    controller.abort();
    await expect(
      (pool as any).runWithTimeout(
        'aborted-op',
        'projection',
        (api: any) => api.handle(),
        controller.signal
      )
    ).rejects.toMatchObject({ name: 'WorkerAbortError', operation: 'aborted-op' });
    // Worker was never reached.
    expect(w0.api.handle).not.toHaveBeenCalled();
  });

  it('rejects when the signal aborts after dispatch', async () => {
    // Worker call that never resolves on its own — the abort must
    // settle the promise.
    const w0 = makeFakeWorker('A');
    w0.api.handle = vi.fn(() => new Promise(() => {}));
    const pool = makePool([w0]);
    const controller = new AbortController();
    const promise = (pool as any).runWithTimeout(
      'mid-flight-abort',
      'projection',
      (api: any) => api.handle(),
      controller.signal
    );
    // Give microtasks a tick to start the call, then abort.
    await Promise.resolve();
    controller.abort();
    await expect(promise).rejects.toMatchObject({ name: 'WorkerAbortError' });
  });

  it('releases its listener on the caller signal after a completed call', async () => {
    const pool = makePool([makeFakeWorker('A')]);
    const callerController = new AbortController();

    const result = await (pool as any).runWithTimeout(
      'listener-cleanup',
      'projection',
      (api: any) => api.handle(),
      callerController.signal
    );

    expect(result).toBe('A');
    expect(getEventListeners(callerController.signal, 'abort')).toHaveLength(0);
  });
});
