/**
 * Pick the least-loaded worker in a {@link WorkerInstance} list, and
 * return tracking callbacks the caller uses to mark a query in flight.
 * Lifted from `WorkerPool.getWorkerWithTracking` (the body after the
 * initialize/empty-pool guard).
 *
 * A stalled worker stops being selected once its `activeQueries` grows
 * past the others — without this, round-robin selection would queue
 * new calls on the stalled worker until it timed out individually
 * (head-of-line blocking).
 */

import type { Remote } from 'comlink';
import type { DataWorkerAPI } from '../../data-worker';
import type { WorkerInstance } from '../types';

export interface TrackedWorkerHandle {
  api: Remote<DataWorkerAPI>;
  worker: Worker;
  markQueryStart: () => void;
  markQueryEnd: () => void;
}

export function selectLeastBusy(workers: WorkerInstance[]): TrackedWorkerHandle {
  // Find worker with least active queries
  let leastBusyIndex = 0;
  let minQueries = workers[0].activeQueries;

  for (let i = 1; i < workers.length; i++) {
    if (workers[i].activeQueries < minQueries) {
      minQueries = workers[i].activeQueries;
      leastBusyIndex = i;
    }
  }

  const workerInstance = workers[leastBusyIndex];

  return {
    api: workerInstance.api,
    worker: workerInstance.worker,
    markQueryStart: () => {
      workerInstance.activeQueries++;
    },
    markQueryEnd: () => {
      workerInstance.activeQueries = Math.max(0, workerInstance.activeQueries - 1);
    },
  };
}
