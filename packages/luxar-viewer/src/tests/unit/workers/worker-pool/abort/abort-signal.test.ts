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

  it('pool-wide setAbortSignal applies to every subsequent runWithTimeout', async () => {
    const w0 = makeFakeWorker('A');
    const pool = makePool([w0]);
    const controller = new AbortController();
    pool.setAbortSignal(controller.signal);
    controller.abort();
    await expect(
      (pool as any).runWithTimeout('pool-signal-op', 'projection', (api: any) => api.handle())
    ).rejects.toMatchObject({ name: 'WorkerAbortError' });
    // Clearing the pool signal restores normal behavior.
    pool.setAbortSignal(undefined);
    const result = await (pool as any).runWithTimeout('after-clear-op', 'projection', (api: any) =>
      api.handle()
    );
    expect(result).toBe('A');
  });

  it('dispose clears a stale pool-wide abort signal before reuse', async () => {
    const pool = makePool([makeFakeWorker('before-dispose')]);
    const controller = new AbortController();
    pool.setAbortSignal(controller.signal);
    controller.abort();

    pool.dispose();

    const w0 = makeFakeWorker('after-dispose');
    (pool as any).workers = [w0];
    (pool as any).initPromise = Promise.resolve();
    (pool as any).nextWorkerIndex = 0;

    const result = await (pool as any).runWithTimeout(
      'after-dispose-op',
      'projection',
      (api: any) => api.handle()
    );
    expect(result).toBe('after-dispose');
  });
});
