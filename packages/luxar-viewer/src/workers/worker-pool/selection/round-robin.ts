/**
 * Round-robin advance over a worker pool. Given the current cursor
 * and the workers list, returns the next instance + the new cursor
 * (caller stores it back on the pool).
 *
 * Lifted from `WorkerPool.nextWorkerInstance` (the body after the
 * initialize/empty-pool guard). Untracked callers route here via
 * `WorkerPool.getWorker()`; load-aware callers should use
 * `selectLeastBusy` instead.
 */

import type { WorkerInstance } from '../types';

export interface RoundRobinResult {
  instance: WorkerInstance;
  nextIndex: number;
}

export function nextRoundRobin(workers: WorkerInstance[], cursor: number): RoundRobinResult {
  const instance = workers[cursor];
  const nextIndex = (cursor + 1) % workers.length;
  return { instance, nextIndex };
}
