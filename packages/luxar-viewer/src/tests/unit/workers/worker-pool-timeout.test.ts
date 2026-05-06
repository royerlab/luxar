/**
 * Unit tests for WorkerPool.withTimeout and WorkerTimeoutError.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

interface MockWorker {
  index: number;
  terminate: ReturnType<typeof vi.fn>;
  onerror: ((e: { message?: string; preventDefault?: () => void }) => void) | null;
  onmessageerror: (() => void) | null;
}

async function loadWorkerPool(workerCount = 2) {
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
    config: { dataLoading: { performance: { workerCount } } },
  }));
  vi.doMock('../../../utils/log', () => ({
    log,
    Modules: { WORKER_POOL: 'WorkerPool' },
  }));
  vi.doMock('comlink', () => ({
    wrap: vi.fn(() => ({ initialize: vi.fn(async () => {}) })),
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

describe('WorkerPool.withTimeout', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.stubGlobal('navigator', { hardwareConcurrency: 16 });
    vi.useRealTimers();
  });

  it('resolves with the call result when it completes before the timeout', async () => {
    const { WorkerPool } = await loadWorkerPool();
    const pool = new WorkerPool();
    const call = Promise.resolve(42);
    const result = await pool.withTimeout('test', call, 1000);
    expect(result).toBe(42);
  });

  it('rejects with WorkerTimeoutError when the call exceeds the timeout', async () => {
    vi.useFakeTimers();
    const { WorkerPool, WorkerTimeoutError } = await loadWorkerPool();
    const pool = new WorkerPool();

    const neverResolves = new Promise<number>(() => {});
    const raced = pool.withTimeout('slow-op', neverResolves, 50);

    vi.advanceTimersByTime(60);
    await expect(raced).rejects.toBeInstanceOf(WorkerTimeoutError);
    await expect(raced).rejects.toMatchObject({
      operation: 'slow-op',
      timeoutMs: 50,
    });
  });

  it('passes through unchanged when timeoutMs is 0 (disabled)', async () => {
    const { WorkerPool } = await loadWorkerPool();
    const pool = new WorkerPool();
    const call = Promise.resolve('ok');
    expect(await pool.withTimeout('disabled', call, 0)).toBe('ok');
  });

  it('passes through unchanged when timeoutMs is negative', async () => {
    const { WorkerPool } = await loadWorkerPool();
    const pool = new WorkerPool();
    expect(await pool.withTimeout('neg', Promise.resolve(1), -100)).toBe(1);
  });

  it('passes through unchanged when timeoutMs is non-finite', async () => {
    const { WorkerPool } = await loadWorkerPool();
    const pool = new WorkerPool();
    expect(await pool.withTimeout('inf', Promise.resolve(1), Infinity)).toBe(1);
    expect(await pool.withTimeout('nan', Promise.resolve(2), NaN)).toBe(2);
  });

  it('evicts the responsible worker when given one and the timeout fires', async () => {
    vi.useFakeTimers();
    const { WorkerPool, workers } = await loadWorkerPool(2);
    const pool = new WorkerPool();
    await pool.initialize();
    expect(pool.getWorkerCount()).toBe(2);

    const neverResolves = new Promise<number>(() => {});
    const raced = pool.withTimeout('project', neverResolves, 100, workers[1] as unknown as Worker);
    vi.advanceTimersByTime(110);
    await expect(raced).rejects.toThrow();

    expect(pool.getWorkerCount()).toBe(1);
    expect(workers[1].terminate).toHaveBeenCalledTimes(1);
  });

  it('clears the timeout when the call resolves first (no late firing)', async () => {
    vi.useFakeTimers();
    const { WorkerPool } = await loadWorkerPool();
    const pool = new WorkerPool();

    let resolveCall!: (v: number) => void;
    const call = new Promise<number>((r) => {
      resolveCall = r;
    });
    const raced = pool.withTimeout('quick', call, 1000);
    resolveCall(7);
    expect(await raced).toBe(7);

    // Advance past the original timeout — nothing should reject.
    vi.advanceTimersByTime(2000);
  });
});

describe('WorkerTimeoutError', () => {
  it('carries the operation name and timeoutMs', async () => {
    const { WorkerTimeoutError } = await loadWorkerPool();
    const err = new WorkerTimeoutError('foo', 500);
    expect(err.operation).toBe('foo');
    expect(err.timeoutMs).toBe(500);
    expect(err.name).toBe('WorkerTimeoutError');
    expect(err.message).toContain('foo');
    expect(err.message).toContain('500ms');
  });

  it('is an instance of Error', async () => {
    const { WorkerTimeoutError } = await loadWorkerPool();
    expect(new WorkerTimeoutError('x', 1)).toBeInstanceOf(Error);
  });
});
