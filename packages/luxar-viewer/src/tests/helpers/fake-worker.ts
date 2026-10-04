/**
 * Fake data-pool workers: the `{ worker, api, activeQueries }` slot a
 * `WorkerPool` routes to, without spawning a real Worker. `api` is whatever
 * the test needs the worker to answer.
 */

import { vi, type Mock } from 'vitest';
import { WorkerPool } from '../../workers/worker-pool';

/** The shape of one entry of `WorkerPool.workers`. */
export interface FakeWorkerInstance<Api = unknown> {
  worker: { terminate: Mock };
  api: Api;
  activeQueries: number;
}

/** A worker slot answering through `api`. */
export function fakeWorkerInstance<Api>(api: Api, activeQueries = 0): FakeWorkerInstance<Api> {
  return { worker: { terminate: vi.fn() }, api, activeQueries };
}

/** A worker whose generic `handle()` resolves with `label` (routing tests). */
export function makeFakeWorker(
  label: string,
  activeQueries = 0
): FakeWorkerInstance<{ handle: Mock }> {
  return fakeWorkerInstance({ handle: vi.fn().mockResolvedValue(label) }, activeQueries);
}

/**
 * Publish `workers` on `pool` (a fresh one by default) as if it had already
 * initialised them: the routing and dispatch paths then run for real.
 */
export function poolWithWorkers(
  workers: ReadonlyArray<{ activeQueries: number }>,
  pool: WorkerPool = new WorkerPool()
): WorkerPool {
  const internals = pool as unknown as { workers: unknown[]; initPromise: Promise<void> };
  internals.workers = workers as unknown as unknown[];
  internals.initPromise = Promise.resolve();
  return pool;
}
