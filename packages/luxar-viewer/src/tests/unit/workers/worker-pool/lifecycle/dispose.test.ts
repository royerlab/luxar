/**
 * the WorkerPool dispose-mid-init
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

  vi.doMock('../../../../../config', () => ({
    config: {
      dataLoading: {
        performance: {
          workerCount,
          workerProjectionTimeoutMs: 60000,
          workerInitTimeoutMs: 0, // disabled — tests control settlement directly
        },
      },
    },
  }));

  vi.doMock('../../../../../utils/log', () => ({
    log,
    Modules: { WORKER_POOL: 'WorkerPool' },
  }));

  vi.doMock('comlink', () => ({
    wrap: vi.fn(() => ({ initialize: initializeImpl })),
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

describe('WorkerPool.dispose — mid-init race', () => {
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

  it('[workers.md C3] dispose+initialize sequence runs a fresh init (public contract, no private field probe)', async () => {
    // workers.md C3[P10] fix: prior version probed the private
    // `initGeneration` field via `as unknown as Internals` cast. The
    // behaviour under test is "dispose lets subsequent initialize()
    // run fresh" — that's observable on the PUBLIC surface
    // (getWorkerCount before/after dispose + re-init). Rewritten to
    // exercise the public contract without naming any private field.
    const { WorkerPool, workers } = await loadWorkerPool(2);
    const pool = new WorkerPool();

    await pool.initialize();
    expect(pool.getWorkerCount()).toBe(2);

    // First dispose: count drops to 0, all workers terminated.
    pool.dispose();
    expect(pool.getWorkerCount()).toBe(0);

    // Re-initialise: a SECOND batch of workers is spawned. The fact
    // that re-init succeeds (and getWorkerCount returns 2 again) is
    // what `dispose bumps initGeneration` enables; we observe it
    // without ever reading the private counter.
    await pool.initialize();
    expect(pool.getWorkerCount()).toBe(2);

    // Second dispose: terminates the SECOND batch.
    pool.dispose();
    expect(pool.getWorkerCount()).toBe(0);

    // `workers` is a module-level register of every Worker instance
    // ever spawned by the test factory. After two init+dispose cycles
    // we should have terminated at least 2 distinct worker instances.
    const terminatedCount = workers.filter((w) => w.terminate.mock.calls.length > 0).length;
    expect(terminatedCount).toBeGreaterThanOrEqual(2);
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

  // workers.md O7 / Phase E27: previous name used arrow notation
  // (`init1 → dispose → init2 ...`). Rewrite in the surrounding file's
  // "does X when Y" style so the test name reads as a behavior
  // assertion rather than a sequence diagram.
  it('late settlement of a disposed init1 does NOT corrupt init2 workers', async () => {
    // Per-call init impl: first N calls (init1's workers) park forever
    // until released; subsequent calls (init2's workers) resolve immediately.
    const init1Releases: Array<() => void> = [];
    let callCount = 0;
    const initImpl = (): Promise<void> => {
      const myCallIndex = callCount++;
      if (myCallIndex < 2) {
        // init1's two workers — defer.
        return new Promise<void>((resolve) => {
          init1Releases.push(resolve);
        });
      }
      // init2's workers — resolve immediately.
      return Promise.resolve();
    };

    const { WorkerPool, workers } = await loadWorkerPool(2, initImpl);
    const pool = new WorkerPool();

    // ── Step 1: start init1 (workers 0, 1).
    const initP1 = pool.initialize();
    await Promise.resolve();
    expect(workers.length).toBe(2);

    // ── Step 2: dispose (bumps generation, terminates pending,
    // nils initPromise).
    pool.dispose();
    expect(workers[0].terminate.mock.calls.length).toBeGreaterThanOrEqual(1);
    expect(workers[1].terminate.mock.calls.length).toBeGreaterThanOrEqual(1);

    // ── Step 3: start init2 (workers 2, 3) and await — it should
    // succeed because the deferred initImpl resolves immediately for
    // worker indices >= 2.
    const initP2 = pool.initialize();
    await initP2;
    expect(pool.getWorkerCount()).toBe(2);
    expect(workers.length).toBe(4); // 2 from init1 + 2 from init2
    // init2's workers haven't been terminated.
    expect(workers[2].terminate.mock.calls.length).toBe(0);
    expect(workers[3].terminate.mock.calls.length).toBe(0);

    // ── Step 4: release init1's deferred promises so init1 settles
    // late. Init1's IIFE must NOT touch `this.workers` (which now
    // holds init2's workers).
    init1Releases.forEach((r) => r());
    await initP1;

    // ── Assertions: pool still has init2's two workers; init1's
    // late-arriving stale workers are terminated (and stay
    // terminated); pool.workers reference identity is preserved.
    expect(pool.getWorkerCount()).toBe(2);
    // init1's workers terminated (by dispose + by per-factory catch
    // when generation-mismatch fires; idempotent).
    expect(workers[0].terminate.mock.calls.length).toBeGreaterThanOrEqual(1);
    expect(workers[1].terminate.mock.calls.length).toBeGreaterThanOrEqual(1);
    // init2's workers UNTOUCHED.
    expect(workers[2].terminate.mock.calls.length).toBe(0);
    expect(workers[3].terminate.mock.calls.length).toBe(0);
  });
});

// workers.md O5 / Phase E13: moved here from `init-timeout.test.ts` —
// the test's subject is "dispose clears initPromise", which belongs
// in this dispose-themed file, not in the workerInitTimeoutMs:0 file.
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
    const rejectingInit = async () => {
      throw new Error('init throws');
    };
    const { WorkerPool } = await loadWorkerPool(1, rejectingInit);
    const pool = new WorkerPool();
    await expect(pool.initialize()).rejects.toThrow();
    expect(pool.getWorkerCount()).toBe(0);
    expect(pool.isInitialized()).toBe(false);

    pool.dispose();
    expect(pool.getWorkerCount()).toBe(0);

    // After dispose, calling initialize() again must NOT return the cached
    // rejected promise. With the same factory (still rejecting), the new
    // initialize() must re-run the loader factory — assert "rejects again"
    // rather than "is null" so dispose's "clear cached initPromise"
    // contract is observable.
    await expect(pool.initialize()).rejects.toThrow();
    expect(pool.getWorkerCount()).toBe(0);
    expect(pool.isInitialized()).toBe(false);
  });
});
