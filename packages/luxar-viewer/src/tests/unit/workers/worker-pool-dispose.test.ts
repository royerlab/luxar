/**
 * Phase 15.1 regression tests for the WorkerPool dispose-mid-init
 * race.
 *
 * Pre-fix: `WorkerPool.initialize()` spawned workers via
 * `Array.from(...).map(async (_, i) => …)` and pushed each successful
 * worker into `this.workers` only on the fulfilled branch of
 * `Promise.allSettled`. A `dispose()` between `new DataWorker()` and
 * the push walked an empty `this.workers` and exited; the in-flight
 * factories then resolved and re-populated `this.workers`
 * post-dispose, leaving live Worker globals the pool no longer
 * referenced.
 *
 * Post-fix:
 *   - `initialize()` captures an `initGeneration` token; factories
 *     check it before pushing into `this.workers`.
 *   - Every constructed Worker is added to `pendingWorkers` until it
 *     either succeeds (removed) or fails (also removed).
 *   - `dispose()` bumps `initGeneration` and terminates everything in
 *     `pendingWorkers`.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

type MockWorker = {
  index: number;
  terminate: ReturnType<typeof vi.fn>;
  onerror: ((e: { message?: string; preventDefault?: () => void }) => void) | null;
  onmessageerror: (() => void) | null;
};

async function loadWorkerPool(
  workerCount = 2,
  initializeImpl: () => Promise<void> = async () => {}
) {
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

  vi.doMock('../../../config', () => ({
    config: {
      dataLoading: {
        performance: {
          workerCount,
          workerVisibilityTimeoutMs: 30000,
          workerProjectionTimeoutMs: 60000,
          workerInitTimeoutMs: 0, // disabled — tests control settlement directly
        },
      },
    },
  }));

  vi.doMock('../../../utils/log', () => ({
    log,
    Modules: { WORKER_POOL: 'WorkerPool' },
  }));

  vi.doMock('comlink', () => ({
    wrap: vi.fn(() => ({ initialize: initializeImpl })),
  }));

  vi.doMock('../../../workers/data-worker?worker', () => ({
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

  const module = await import('../../../workers/worker-pool');
  return { ...module, log, workers };
}

describe('WorkerPool.dispose — mid-init race (Phase 15.1)', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.stubGlobal('navigator', { hardwareConcurrency: 16 });
    vi.useRealTimers();
  });

  it('dispose mid-init terminates every pending worker', async () => {
    // Hold every factory's `api.initialize()` promise open so each
    // factory is still parked on `await initializeWithGuard(...)`
    // when dispose runs. Each factory call to initImpl returns a
    // fresh Promise, so collect all resolvers and release them after
    // dispose.
    const releases: Array<() => void> = [];
    const initImpl = (): Promise<void> =>
      new Promise<void>((resolve) => {
        releases.push(resolve);
      });

    const { WorkerPool, workers } = await loadWorkerPool(3, initImpl);
    const pool = new WorkerPool();

    // Start initialize() but don't await — let factories construct
    // their workers (synchronous) and enter the await on init.
    const initP = pool.initialize();
    // Yield once so the IIFE runs to its first await and the
    // workerPromises array is built; this populates pendingWorkers.
    await Promise.resolve();

    expect(workers.length).toBe(3);
    expect(workers.every((w) => w.terminate.mock.calls.length === 0)).toBe(true);

    // Dispose mid-init.
    pool.dispose();

    // Each pending worker must have been terminated by dispose (via
    // the pendingWorkers walk).
    expect(workers.every((w) => w.terminate.mock.calls.length >= 1)).toBe(true);

    // Release every factory's init promise so they proceed past the
    // await. They detect the generation mismatch and self-terminate
    // (the per-factory catch fires terminate again — idempotent).
    releases.forEach((r) => r());
    await initP;

    expect(pool.getWorkerCount()).toBe(0);
    expect(pool.isInitialized()).toBe(false);
  });

  it('initialize() resolves cleanly when dispose races with successful init', async () => {
    // api.initialize resolves immediately; dispose runs synchronously
    // before the pool's IIFE finishes its post-allSettled push loop.
    const { WorkerPool, workers } = await loadWorkerPool(2);
    const pool = new WorkerPool();

    const initP = pool.initialize();
    pool.dispose();

    // Should NOT throw — the post-dispose generation check inside the
    // IIFE early-returns instead of throwing.
    await expect(initP).resolves.toBeUndefined();

    expect(pool.getWorkerCount()).toBe(0);
    expect(pool.isInitialized()).toBe(false);
    // Each constructed worker must have been terminated (either by
    // dispose's pendingWorkers walk or by the per-factory catch
    // fired by the generation check).
    expect(workers.length).toBe(2);
    expect(workers.every((w) => w.terminate.mock.calls.length >= 1)).toBe(true);
  });

  it('dispose() bumps generation so subsequent initialize() runs fresh', async () => {
    const { WorkerPool } = await loadWorkerPool(1);
    const pool = new WorkerPool();

    // Access the private field via cast — only safe inside this test.
    type Internals = { initGeneration: number };
    const before = (pool as unknown as Internals).initGeneration;

    pool.dispose();
    pool.dispose();

    const after = (pool as unknown as Internals).initGeneration;
    expect(after).toBe(before + 2);
  });

  it('dispose AFTER successful init terminates the live workers (regression for existing path)', async () => {
    const { WorkerPool, workers } = await loadWorkerPool(2);
    const pool = new WorkerPool();

    await pool.initialize();
    expect(pool.getWorkerCount()).toBe(2);
    expect(workers.every((w) => w.terminate.mock.calls.length === 0)).toBe(true);

    pool.dispose();

    expect(pool.getWorkerCount()).toBe(0);
    expect(workers.every((w) => w.terminate.mock.calls.length === 1)).toBe(true);
  });
});
