// @vitest-environment jsdom
/**
 * Tests for WorkerPool's runtime-error handling: a worker that
 * crashes after init via `onerror` / `onmessageerror` should be
 * removed from the pool, and dropping to zero workers should clear
 * the init promise so a subsequent initialize() can recover.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

interface MockWorker {
  index: number;
  terminate: ReturnType<typeof vi.fn>;
  onerror: ((event: { message?: string; preventDefault?: () => void }) => void) | null;
  onmessageerror: (() => void) | null;
}

async function loadWorkerPool(workerCount: number, perf: Record<string, number> = {}) {
  vi.resetModules();

  const log = {
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    success: vi.fn(),
    update: vi.fn(),
  };
  const workers: MockWorker[] = [];
  /** `<worker index>:<label>` for every `task` call, in dispatch order. */
  const taskCalls: string[] = [];
  let nextWorkerIndex = 0;

  vi.doMock('../../../../../config', () => ({
    config: { dataLoading: { performance: { workerCount, ...perf } } },
  }));

  vi.doMock('../../../../../utils/log', () => ({
    log,
    Modules: { WORKER_POOL: 'WorkerPool' },
  }));

  vi.doMock('comlink', () => ({
    // `hang` stands for any task the worker never answers: Comlink settles a
    // call only from the worker's reply, which a terminated worker never sends.
    wrap: vi.fn(({ index }: MockWorker) => ({
      initialize: vi.fn(async () => {}),
      hang: vi.fn(() => new Promise<never>(() => {})),
      // Worker 0 is wedged and never answers; every other worker answers
      // after a short delay, with its own index so a test can see who ran it.
      task: vi.fn((label: string) => {
        taskCalls.push(`${index}:${label}`);
        if (index === 0) return new Promise<never>(() => {});
        return new Promise<string>((resolve) => setTimeout(() => resolve(`w${index}`), 20));
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
  return { ...module, log, workers, taskCalls };
}

describe('WorkerPool runtime-error handling', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.stubGlobal('navigator', { hardwareConcurrency: 16 });
  });

  it('attaches onerror and onmessageerror handlers that actually trigger eviction when fired', async () => {
    // workers.md W7 fix: previous version asserted only `typeof === 'function'`,
    // which doesn't prove the handler invokes evictFailedWorker. Strengthen by
    // firing one of the handlers and checking the pool count drops.
    const { WorkerPool, workers } = await loadWorkerPool(3);
    const pool = new WorkerPool();
    await pool.initialize();

    expect(workers).toHaveLength(3);
    for (const w of workers) {
      expect(typeof w.onerror).toBe('function');
      expect(typeof w.onmessageerror).toBe('function');
    }

    // Fire onerror on the first worker; pool count must drop by 1.
    const beforeCount = pool.getWorkerCount();
    const preventDefault = vi.fn();
    workers[0].onerror?.({ preventDefault, message: 'test' } as unknown as ErrorEvent);
    expect(pool.getWorkerCount()).toBe(beforeCount - 1);
  });

  it('removes a worker from the pool on runtime error and continues with reduced capacity', async () => {
    const { WorkerPool, workers, log } = await loadWorkerPool(3);
    const pool = new WorkerPool();
    await pool.initialize();
    expect(pool.getWorkerCount()).toBe(3);

    // Simulate a runtime crash on worker 1.
    const preventDefault = vi.fn();
    workers[1].onerror?.({ message: 'OOM', preventDefault });

    expect(pool.getWorkerCount()).toBe(2);
    expect(workers[1].terminate).toHaveBeenCalledTimes(1);
    expect(preventDefault).toHaveBeenCalledTimes(1);
    expect(log.error).toHaveBeenCalled();
    expect(log.warning).toHaveBeenCalledWith(
      'WorkerPool',
      expect.stringContaining('2 worker(s) remaining')
    );
  });

  it('removes a worker on unserializable message error', async () => {
    const { WorkerPool, workers } = await loadWorkerPool(2);
    const pool = new WorkerPool();
    await pool.initialize();
    expect(pool.getWorkerCount()).toBe(2);

    workers[0].onmessageerror?.();

    expect(pool.getWorkerCount()).toBe(1);
    expect(workers[0].terminate).toHaveBeenCalledTimes(1);
  });

  it('is idempotent: a second error event for the same worker is a no-op', async () => {
    const { WorkerPool, workers } = await loadWorkerPool(2);
    const pool = new WorkerPool();
    await pool.initialize();

    workers[0].onerror?.({ message: 'bang', preventDefault: vi.fn() });
    workers[0].onerror?.({ message: 'bang', preventDefault: vi.fn() });

    // Still only one terminate, still one worker remaining.
    expect(workers[0].terminate).toHaveBeenCalledTimes(1);
    expect(pool.getWorkerCount()).toBe(1);
  });

  it('clears initPromise when the last worker dies so initialize() can retry', async () => {
    const { WorkerPool, workers, log } = await loadWorkerPool(1);
    const pool = new WorkerPool();
    await pool.initialize();
    expect(pool.getWorkerCount()).toBe(1);

    workers[0].onerror?.({ message: 'bang', preventDefault: vi.fn() });

    expect(pool.getWorkerCount()).toBe(0);
    expect(log.error).toHaveBeenCalledWith(
      'WorkerPool',
      expect.stringContaining('All data workers failed')
    );

    // After the failure, initialize() should be able to spin up a fresh worker.
    await pool.initialize();
    expect(pool.getWorkerCount()).toBe(1);
    expect(workers).toHaveLength(2); // first crashed, second freshly created
  });
});

/** What `call` settled with, or `'pending'` if it had not settled by the deadline. */
async function settledWithin(call: Promise<unknown>, ms = 20): Promise<unknown> {
  return Promise.race([
    call.then(
      (value: unknown) => ({ value }),
      (error: unknown) => error
    ),
    new Promise((resolve) => setTimeout(() => resolve('pending'), ms)),
  ]);
}

type HangApi = { hang: () => Promise<never> };
type TaskApi = { task: (label: string) => Promise<string> };

/** `{ value }` on success, the error on rejection — observed immediately. */
const outcome = (call: Promise<unknown>): Promise<unknown> =>
  call.then(
    (value: unknown) => ({ value }),
    (error: unknown) => error
  );

/** Let `runWithTimeout`'s worker selection run, so the calls reach a worker. */
const dispatched = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

describe('WorkerPool in-flight calls on an evicted worker', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.stubGlobal('navigator', { hardwareConcurrency: 16 });
  });

  it('rejects every call in flight on a worker evicted by a runtime error', async () => {
    // No per-call timeout is configured, so nothing but the eviction can
    // settle these calls: a terminated worker never replies.
    const { WorkerPool, WorkerUnavailableError, workers } = await loadWorkerPool(1);
    const pool = new WorkerPool();
    await pool.initialize();

    const run = (): Promise<unknown> =>
      pool.runWithTimeout('hang', 'projection', (api) => (api as unknown as HangApi).hang());
    const first = run();
    const second = run();
    await dispatched();

    workers[0].onerror?.({ message: 'OOM', preventDefault: vi.fn() });

    expect(await settledWithin(first)).toBeInstanceOf(WorkerUnavailableError);
    expect(await settledWithin(second)).toBeInstanceOf(WorkerUnavailableError);
  });

  it.fails('leaves calls on the surviving workers in flight, and moves the evicted ones there', async () => {
    const { WorkerPool, workers } = await loadWorkerPool(2);
    const pool = new WorkerPool();
    await pool.initialize();

    // Least-busy routing puts one call on each worker.
    const calls = [0, 1].map(() =>
      pool.runWithTimeout('hang', 'projection', (api) => (api as unknown as HangApi).hang())
    );
    await dispatched();
    workers[0].onerror?.({ message: 'OOM', preventDefault: vi.fn() });

    // The survivor's call is untouched; w0's was re-dispatched onto the
    // survivor rather than rejected, so neither has settled.
    const outcomes = await Promise.all(calls.map((c) => settledWithin(c)));
    expect(outcomes).toEqual(['pending', 'pending']);
    expect(pool.getWorkerCount()).toBe(1);
  });

  it('rejects calls in flight when the pool is disposed', async () => {
    const { WorkerPool, WorkerUnavailableError } = await loadWorkerPool(1);
    const pool = new WorkerPool();
    await pool.initialize();

    const call = pool.runWithTimeout('hang', 'projection', (api) =>
      (api as unknown as HangApi).hang()
    );
    await dispatched();
    pool.dispose();

    expect(await settledWithin(call)).toBeInstanceOf(WorkerUnavailableError);
  });
});

describe('WorkerPool re-dispatch of calls abandoned by an eviction', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.stubGlobal('navigator', { hardwareConcurrency: 16 });
  });

  const runTask = (
    pool: { runWithTimeout: (...args: never[]) => Promise<unknown> },
    label: string,
    signal?: AbortSignal
  ): Promise<unknown> =>
    (pool.runWithTimeout as (...args: unknown[]) => Promise<unknown>)(
      'task',
      'projection',
      (api: TaskApi) => api.task(label),
      signal
    );

  it.fails('re-runs a call queued behind a timed-out task on a surviving worker', async () => {
    // A hangs on w0; B runs on w1; C ties and queues on w0. A's timeout evicts
    // w0, which abandons C. C never reached a kernel, so it must run on w1
    // rather than reject into the in-process (main-thread) fallback, and A —
    // the timed-out task itself — must NOT be re-run anywhere.
    const { WorkerPool, WorkerTimeoutError, taskCalls } = await loadWorkerPool(2, {
      workerProjectionTimeoutMs: 100,
    });
    const pool = new WorkerPool();
    await pool.initialize();

    // Outcomes are captured at once so no rejection sits unobserved.
    const [a, b, c] = ['A', 'B', 'C'].map((label) => outcome(runTask(pool, label)));
    await dispatched();
    expect(taskCalls).toEqual(['0:A', '1:B', '0:C']);

    expect(await a).toBeInstanceOf(WorkerTimeoutError);
    expect(await b).toEqual({ value: 'w1' });
    expect(await c).toEqual({ value: 'w1' });
    expect(taskCalls).toEqual(['0:A', '1:B', '0:C', '1:C']);
    expect(pool.getWorkerCount()).toBe(1);
  });

  it.fails('re-dispatches an abandoned call at most once', async () => {
    const { WorkerPool, WorkerUnavailableError, workers, taskCalls } = await loadWorkerPool(3);
    const pool = new WorkerPool();
    await pool.initialize();

    // The call lands on w0 (all idle; first wins). Crash w0, then the worker
    // the call was re-dispatched to.
    const hanging = (pool.runWithTimeout as (...args: unknown[]) => Promise<unknown>)(
      'hang',
      'projection',
      (api: HangApi) => api.hang()
    );
    hanging.catch(() => {}); // observed below, via settledWithin
    await dispatched();
    workers[0].onerror?.({ message: 'OOM', preventDefault: vi.fn() });
    await dispatched();
    const stillPending = await settledWithin(hanging);
    expect(stillPending).toBe('pending');
    // It moved to w1 (least busy, first in order). Crashing that one too must
    // reject it — not move it again — even though w2 is still alive.
    workers[1].onerror?.({ message: 'OOM', preventDefault: vi.fn() });
    expect(await settledWithin(hanging)).toBeInstanceOf(WorkerUnavailableError);
    expect(pool.getWorkerCount()).toBe(1);
    expect(taskCalls).toEqual([]);
  });

  it('does not re-dispatch a call whose caller already aborted', async () => {
    const { WorkerPool, WorkerAbortError, workers, taskCalls } = await loadWorkerPool(2);
    const pool = new WorkerPool();
    await pool.initialize();

    const controller = new AbortController();
    const aborted = runTask(pool, 'A', controller.signal);
    await dispatched();
    controller.abort();
    await expect(aborted).rejects.toBeInstanceOf(WorkerAbortError);

    workers[0].onerror?.({ message: 'OOM', preventDefault: vi.fn() });
    await dispatched();
    expect(taskCalls).toEqual(['0:A']);
  });
});
