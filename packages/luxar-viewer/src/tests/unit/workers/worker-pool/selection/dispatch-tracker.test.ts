/**
 * Worker dispatch perf counters (`worker.dispatches`, `worker.busyMs`,
 * `worker.misroutes`): the pure {@link DispatchTracker}, plus one WorkerPool
 * integration case showing the misroute the counter exists to expose — a
 * caller abort settles `activeQueries` while the worker is still busy, so the
 * least-busy picker re-selects it although another worker is idle.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DispatchTracker } from '../../../../../workers/worker-pool/selection/dispatch-tracker';
import type { WorkerInstance } from '../../../../../workers/worker-pool/types';
import { WorkerPool, WorkerAbortError } from '../../../../../workers/worker-pool';
import { perfCounters } from '../../../../../profiling/perf-counters';

function makeInstance(label: string): WorkerInstance {
  return {
    worker: { _label: label, terminate: vi.fn() } as unknown as Worker,
    api: { _label: label } as unknown as WorkerInstance['api'],
    activeQueries: 0,
    wasmFallback: false,
  };
}

/** A promise plus its settle handles. */
function deferred<T = void>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const flush = async (): Promise<void> => {
  for (let i = 0; i < 5; i++) await Promise.resolve();
};

describe('DispatchTracker', () => {
  let nowMs: number;

  beforeEach(() => {
    perfCounters.reset();
    nowMs = 0;
    vi.spyOn(performance, 'now').mockImplementation(() => nowMs);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('counts dispatches and sums worker-side busy time', async () => {
    const tracker = new DispatchTracker();
    const a = makeInstance('A');
    const t1 = deferred();
    const t2 = deferred();
    tracker.dispatch(a.worker, [a], t1.promise);
    nowMs = 5;
    tracker.dispatch(a.worker, [a], t2.promise);
    expect(tracker.inFlight(a.worker)).toBe(2);
    nowMs = 12;
    t1.resolve();
    t2.reject(new Error('worker failed')); // a rejection still ends the task
    await flush();
    expect(tracker.inFlight(a.worker)).toBe(0);
    expect(perfCounters.get('worker.dispatches')).toBe(2);
    expect(perfCounters.get('worker.busyMs')).toBe(12 + 7);
  });

  it('flags a dispatch to a busy worker only while another worker is idle', async () => {
    const tracker = new DispatchTracker();
    const a = makeInstance('A');
    const b = makeInstance('B');
    const workers = [a, b];
    const ta = deferred();
    tracker.dispatch(a.worker, workers, ta.promise); // A idle: fine
    tracker.dispatch(a.worker, workers, deferred().promise); // A busy, B idle: misroute
    expect(perfCounters.get('worker.misroutes')).toBe(1);

    tracker.dispatch(b.worker, workers, deferred().promise); // B idle: fine
    tracker.dispatch(a.worker, workers, deferred().promise); // both busy: not a misroute
    expect(perfCounters.get('worker.misroutes')).toBe(1);
    expect(perfCounters.get('worker.dispatches')).toBe(4);

    ta.resolve();
    await flush();
    expect(tracker.inFlight(a.worker)).toBe(2);
  });

  it('a single-worker pool never misroutes', () => {
    const tracker = new DispatchTracker();
    const a = makeInstance('A');
    tracker.dispatch(a.worker, [a], deferred().promise);
    tracker.dispatch(a.worker, [a], deferred().promise);
    expect(perfCounters.get('worker.misroutes')).toBe(0);
  });
});

describe('WorkerPool dispatch counters', () => {
  beforeEach(() => {
    perfCounters.reset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('counts a misroute when a caller abort frees activeQueries but not the worker', async () => {
    const a = makeInstance('A');
    const b = makeInstance('B');
    const pool = new WorkerPool() as unknown as Record<string, unknown>;
    pool.workers = [a, b];
    pool.initPromise = Promise.resolve();
    const typed = pool as unknown as WorkerPool;

    const stuck = deferred<string>();
    const controller = new AbortController();
    const first = typed.runWithTimeout('op', 'decode', () => stuck.promise, controller.signal);
    await flush();
    controller.abort();
    await expect(first).rejects.toBeInstanceOf(WorkerAbortError);
    expect(a.activeQueries).toBe(0); // caller-side accounting released

    // Selection still ties at A (index 0) although A's task is running.
    const second = typed.runWithTimeout('op', 'decode', () => Promise.resolve('ok'));
    await expect(second).resolves.toBe('ok');
    expect(perfCounters.get('worker.dispatches')).toBe(2);
    expect(perfCounters.get('worker.misroutes')).toBe(1);

    stuck.resolve('late');
    await flush();
  });
});
