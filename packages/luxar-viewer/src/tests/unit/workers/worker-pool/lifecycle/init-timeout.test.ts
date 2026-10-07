/**
 * Tests for WorkerPool.initializeWithGuard() — the per-worker init-time
 * guard that races api.initialize() against a hard timeout, worker.onerror,
 * and worker.onmessageerror.
 *
 * coverage: `workerInitTimeoutMs: 0` is documented in
 * `config/validation.ts` as "disables the guard". Pre-fix, the pool called
 * `setTimeout(..., 0)` unconditionally, which fired on the next macrotask
 * and rejected real async inits immediately. The fix mirrors the
 * `withTimeout()` convention (`<= 0 || !isFinite` ⇒ no timer).
 *
 * Also covers `dispose()` on a zero-worker pool — used to leak `initPromise`
 * because the reset was inside `if (workers.length > 0)`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { WorkerInitTimeoutError } from '../../../../../workers/worker-pool/errors';
import { initializeWithGuard } from '../../../../../workers/worker-pool/lifecycle/init-with-guard';

interface MockWorker {
  index: number;
  terminate: ReturnType<typeof vi.fn>;
  onerror: ((e: { message?: string; preventDefault?: () => void }) => void) | null;
  onmessageerror: (() => void) | null;
}

async function loadWorkerPool(
  workerCount: number,
  perf: Record<string, number> = {},
  initBehavior:
    | 'immediate'
    | 'delayed'
    | 'never-settles'
    | 'rejects'
    // Per worker, by construction order (a respawned worker gets the next index).
    | ((workerIndex: number) => Promise<void>) = 'immediate',
  delayMs = 0
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

  vi.doMock('../../../../../config', () => ({
    config: {
      dataLoading: {
        performance: {
          workerCount,
          workerInitTimeoutMs: perf.workerInitTimeoutMs ?? 30000,
          workerProjectionTimeoutMs: 60000,
          ...perf,
        },
      },
    },
  }));
  vi.doMock('../../../../../utils/log', () => ({
    log,
    Modules: { WORKER_POOL: 'WorkerPool' },
  }));

  // `initialize()` shape varies per test: immediate-resolve, delayed-resolve,
  // never-settle (for timeout tests), or reject (for onerror parity).
  vi.doMock('comlink', () => ({
    wrap: vi.fn((worker: MockWorker) => ({
      initialize: vi.fn(() => {
        if (typeof initBehavior === 'function') return initBehavior(worker.index);
        if (initBehavior === 'immediate') return Promise.resolve();
        if (initBehavior === 'delayed') {
          return new Promise<void>((resolve) => setTimeout(resolve, delayMs));
        }
        if (initBehavior === 'rejects') {
          return Promise.reject(new Error('init throws'));
        }
        // never-settles
        return new Promise<void>(() => {});
      }),
    })),
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
  return { ...module, log, workers };
}

describe('WorkerPool.initializeWithGuard — workerInitTimeoutMs:0 disables guard', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.stubGlobal('navigator', { hardwareConcurrency: 16 });
    vi.useRealTimers();
  });

  it('does not reject a delayed-but-resolving init when timeout is 0', async () => {
    // Pre-fix: setTimeout(..., 0) fires on next macrotask, and a 50ms
    // delayed init lost the race → rejection. Post-fix: no timer is set.
    const { WorkerPool } = await loadWorkerPool(1, { workerInitTimeoutMs: 0 }, 'delayed', 50);
    const pool = new WorkerPool();
    await expect(pool.initialize()).resolves.toBeUndefined();
    expect(pool.getWorkerCount()).toBe(1);
  });

  it('does not reject a delayed-but-resolving init when timeout is negative', async () => {
    const { WorkerPool } = await loadWorkerPool(1, { workerInitTimeoutMs: -100 }, 'delayed', 50);
    const pool = new WorkerPool();
    await expect(pool.initialize()).resolves.toBeUndefined();
  });

  it('does not reject a delayed-but-resolving init when timeout is non-finite', async () => {
    const { WorkerPool } = await loadWorkerPool(
      1,
      { workerInitTimeoutMs: Infinity },
      'delayed',
      50
    );
    const pool = new WorkerPool();
    await expect(pool.initialize()).resolves.toBeUndefined();
  });

  it('still surfaces api.initialize() rejection when timeout is disabled', async () => {
    // The timeout is the belt-and-suspenders fallback; the primary error
    // path is api.initialize() rejecting. Disabling the timeout must not
    // mask that.
    const { WorkerPool } = await loadWorkerPool(1, { workerInitTimeoutMs: 0 }, 'rejects');
    const pool = new WorkerPool();
    await expect(pool.initialize()).rejects.toThrow('Failed to initialize any data workers');
    expect(pool.isInitialized()).toBe(false);
  });

  it('rejects with a positive timeout when init never settles', async () => {
    vi.useFakeTimers();
    const { WorkerPool } = await loadWorkerPool(1, { workerInitTimeoutMs: 100 }, 'never-settles');
    const pool = new WorkerPool();
    // Capture the rejection at the call site so vitest doesn't see it as
    // an unhandled rejection while the test is awaiting the timer drain.
    const inited = pool.initialize().catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(150);
    const err = await inited;
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).toContain('Failed to initialize any data workers');
  });
});

describe('initializeWithGuard — delayed main-thread reply', () => {
  beforeEach(() => vi.useFakeTimers());

  it('accepts an init reply queued for the same deadline as the guard', async () => {
    const worker = { onerror: null, onmessageerror: null } as unknown as Worker;
    const restoreHandlers = vi.fn();
    const api = {
      initialize: () => new Promise<void>((resolve) => setTimeout(resolve, 100)),
    };
    const outcome = initializeWithGuard(worker, api, 'Worker 1', 100, restoreHandlers).then(
      () => 'ready',
      (error: unknown) => error
    );

    await vi.advanceTimersByTimeAsync(101);

    expect(await outcome).toBe('ready');
    expect(restoreHandlers).toHaveBeenCalledOnce();
  });

  it('still rejects a worker that never replies', async () => {
    const worker = { onerror: null, onmessageerror: null } as unknown as Worker;
    const restoreHandlers = vi.fn();
    const api = { initialize: () => new Promise<void>(() => {}) };
    const outcome = initializeWithGuard(worker, api, 'Worker 2', 100, restoreHandlers).catch(
      (error: unknown) => error
    );

    await vi.advanceTimersByTimeAsync(101);

    expect(await outcome).toBeInstanceOf(WorkerInitTimeoutError);
    expect(restoreHandlers).toHaveBeenCalledOnce();
  });
});

describe('WorkerPool — a slot whose init missed its deadline is respawned', () => {
  // A deadline miss does not prove the worker broken (see
  // `WorkerInitTimeoutError`), so the slot gets the sort worker's bounded
  // retry instead of being written off for the tab's lifetime.
  const never = (): Promise<void> => new Promise<void>(() => {});
  const ready = (): Promise<void> => Promise.resolve();

  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.stubGlobal('navigator', { hardwareConcurrency: 16 });
    vi.useFakeTimers();
  });
  afterEach(() => vi.useRealTimers());

  it('respawns the starved slot after a backoff and publishes it', async () => {
    const { WorkerPool, workers } = await loadWorkerPool(2, { workerInitTimeoutMs: 100 }, (i) =>
      i === 1 ? never() : ready()
    );
    const pool = new WorkerPool();
    const inited = pool.initialize();
    await vi.advanceTimersByTimeAsync(150);
    await inited;
    expect(pool.getWorkerCount()).toBe(1);

    await vi.advanceTimersByTimeAsync(10_000);

    expect(workers).toHaveLength(3);
    expect(pool.getWorkerCount()).toBe(2);
  });

  it('recovers a pool whose every slot missed its deadline', async () => {
    const { WorkerPool } = await loadWorkerPool(1, { workerInitTimeoutMs: 100 }, (i) =>
      i === 0 ? never() : ready()
    );
    const pool = new WorkerPool();
    const inited = pool.initialize().catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(150);
    expect(await inited).toBeInstanceOf(Error);

    await vi.advanceTimersByTimeAsync(10_000);

    expect(pool.getWorkerCount()).toBe(1);
  });

  it('gives a slot that keeps missing its deadline three attempts in all', async () => {
    const { WorkerPool, workers } = await loadWorkerPool(2, { workerInitTimeoutMs: 100 }, (i) =>
      i === 0 ? ready() : never()
    );
    const pool = new WorkerPool();
    void pool.initialize();

    await vi.advanceTimersByTimeAsync(60_000);

    // Slot 0 once, slot 1 three times.
    expect(workers).toHaveLength(4);
    expect(pool.getWorkerCount()).toBe(1);
  });

  it('does not respawn a slot whose init failed outright', async () => {
    const { WorkerPool, workers } = await loadWorkerPool(2, { workerInitTimeoutMs: 100 }, (i) =>
      i === 1 ? Promise.reject(new Error('init throws')) : ready()
    );
    const pool = new WorkerPool();
    await pool.initialize();

    await vi.advanceTimersByTimeAsync(60_000);

    expect(workers).toHaveLength(2);
    expect(pool.getWorkerCount()).toBe(1);
  });

  it('does not respawn into a disposed pool', async () => {
    const { WorkerPool, workers } = await loadWorkerPool(2, { workerInitTimeoutMs: 100 }, (i) =>
      i === 1 ? never() : ready()
    );
    const pool = new WorkerPool();
    const inited = pool.initialize();
    await vi.advanceTimersByTimeAsync(150);
    await inited;
    pool.dispose();

    await vi.advanceTimersByTimeAsync(60_000);

    expect(workers).toHaveLength(2);
    expect(pool.getWorkerCount()).toBe(0);
  });
});

// workers.md O5 / Phase E13: the `WorkerPool.dispose — clears
// initPromise even with no workers` describe block was moved from here
// to `dispose.test.ts` — its subject ("dispose clears initPromise") is
// the dispose-themed file's concern, not the workerInitTimeoutMs:0
// file's. See dispose.test.ts for the test body.
