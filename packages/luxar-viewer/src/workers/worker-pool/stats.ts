/**
 * Pool monitoring helpers. Pure on a {@link WorkerInstance} list.
 *
 * - `computeStats` aggregates the per-worker `activeQueries` into a
 *   shape consumed by diagnostic dashboards and unit tests.
 * - `computeQueueDepth` is the cheap sum used by
 *   `__luxarDebug.workers.queueDepth` to spot prefetch backpressure
 *   or dataset-switch task accumulation in live sessions.
 */

import type { WorkerInstance } from './types';

export interface PoolStats {
  workerCount: number;
  activeQueries: number[];
  /** Aggregate across all workers — sum of per-worker `activeQueries`. */
  totalActive: number;
  /** Per-worker peak since init; useful for spotting one hot worker. */
  peakActive: number;
}

export function computeStats(workers: WorkerInstance[]): PoolStats {
  const activeQueries = workers.map((w) => w.activeQueries);
  let totalActive = 0;
  let peakActive = 0;
  for (const n of activeQueries) {
    totalActive += n;
    if (n > peakActive) peakActive = n;
  }
  return {
    workerCount: workers.length,
    activeQueries,
    totalActive,
    peakActive,
  };
}

export function computeQueueDepth(workers: WorkerInstance[]): number {
  let total = 0;
  for (const w of workers) total += w.activeQueries;
  return total;
}
