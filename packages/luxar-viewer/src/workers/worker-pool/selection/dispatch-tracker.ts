/**
 * Real in-flight accounting for worker dispatches — instrumentation only.
 *
 * `activeQueries` (see `least-busy.ts`) holds a slot until the worker's
 * timeout-guarded promise settles, even if the caller has already aborted.
 * This tracker counts from dispatch until the worker's own task promise
 * settles, per worker, alongside (not replacing) `activeQueries`. It never
 * influences selection; it only feeds the perf counters:
 *
 * - `worker.dispatches` — every task sent to a worker.
 * - `worker.busyMs` — summed dispatch→worker-settle time of every task.
 * - `worker.misroutes` — a dispatch to a worker that already has a real
 *   in-flight task while at least one OTHER live worker has none (the task
 *   will queue behind unfinished work although an idle worker existed).
 *
 * @module workers/worker-pool/selection/dispatch-tracker
 */

import { perfCounters } from '../../../profiling/perf-counters';
import type { WorkerInstance } from '../types';

const S_DISPATCHES = perfCounters.slot('worker.dispatches');
const S_BUSY_MS = perfCounters.slot('worker.busyMs');
const S_MISROUTES = perfCounters.slot('worker.misroutes');

/** Counts worker-side in-flight tasks and records dispatch performance counters. */
export class DispatchTracker {
  /** Real (worker-side) in-flight task count per worker; absent means 0. */
  private readonly realInFlight = new Map<Worker, number>();

  /** Current real in-flight count for `worker` (0 when none). */
  inFlight(worker: Worker): number {
    return this.realInFlight.get(worker) ?? 0;
  }

  /**
   * Record a dispatch of `task` to `target`, chosen from the live `workers`.
   * `task` must be the worker call's own promise (not a timeout/abort race),
   * so the count drops only when the worker has actually finished.
   */
  dispatch(target: Worker, workers: readonly WorkerInstance[], task: Promise<unknown>): void {
    perfCounters.add(S_DISPATCHES);
    const current = this.inFlight(target);
    if (current > 0 && this.hasIdleOther(target, workers)) perfCounters.add(S_MISROUTES);
    this.realInFlight.set(target, current + 1);

    const startedAt = performance.now();
    const settle = (): void => {
      perfCounters.add(S_BUSY_MS, performance.now() - startedAt);
      const remaining = this.inFlight(target) - 1;
      if (remaining > 0) this.realInFlight.set(target, remaining);
      else this.realInFlight.delete(target);
    };
    // Both arms return undefined, so the derived promise never rejects.
    void task.then(settle, settle);
  }

  private hasIdleOther(target: Worker, workers: readonly WorkerInstance[]): boolean {
    for (const instance of workers) {
      if (instance.worker !== target && this.inFlight(instance.worker) === 0) return true;
    }
    return false;
  }
}
