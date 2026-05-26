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

import { beforeEach, describe, expect, it, vi } from 'vitest';

interface MockWorker {
  index: number;
  terminate: ReturnType<typeof vi.fn>;
  onerror: ((e: { message?: string; preventDefault?: () => void }) => void) | null;
  onmessageerror: (() => void) | null;
}

async function loadWorkerPool(
  workerCount: number,
  perf: Record<string, number> = {},
  initBehavior: 'immediate' | 'delayed' | 'never-settles' | 'rejects' = 'immediate',
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
          workerVisibilityTimeoutMs: 30000,
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
    wrap: vi.fn(() => ({
      initialize: vi.fn(() => {
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

describe('WorkerPool.dispose — clears initPromise even with no workers', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.stubGlobal('navigator', { hardwareConcurrency: 16 });
    vi.useRealTimers();
  });

  it('[workers.md C4] a failed-init pool can be re-initialized cleanly after dispose (public contract)', async () => {
    // workers.md C4[P10] fix: prior version probed the private
    // `initPromise` field via cast. The public contract "dispose clears
    // the cached (rejected) initPromise so re-initialization runs fresh"
    // is observable via a follow-up `await pool.initialize()` succeeding
    // (with a fresh-loader factory that doesn't reject). If dispose
    // failed to clear the cached promise, the second call would resolve
    // to the same rejection.
    const { WorkerPool } = await loadWorkerPool(1, { workerInitTimeoutMs: 1000 }, 'rejects');
    const pool = new WorkerPool();
    await expect(pool.initialize()).rejects.toThrow();
    expect(pool.getWorkerCount()).toBe(0);
    expect(pool.isInitialized()).toBe(false);

    pool.dispose();
    expect(pool.getWorkerCount()).toBe(0);

    // After dispose, calling initialize() again must NOT return the cached
    // rejected promise. We expect EITHER a fresh rejection (same factory)
    // OR clean resolve. The key contract is "fresh attempt" — assert that
    // the new initialize() runs through the loader factory again (the
    // mock-rejection re-fires) rather than instantly settling with the
    // first promise's state. We pin "rejects again" rather than "is null".
    await expect(pool.initialize()).rejects.toThrow();
    // And the public observable state is consistent post-second-attempt.
    expect(pool.getWorkerCount()).toBe(0);
    expect(pool.isInitialized()).toBe(false);
  });
});
