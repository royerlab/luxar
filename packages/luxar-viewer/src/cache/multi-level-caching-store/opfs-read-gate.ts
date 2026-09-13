import { config } from '../../config';

/**
 * Maximum OPFS chunk reads running concurrently across the page.
 *
 * Deep progressive passes can request several hundred cached chunks at once.
 * Reads were the last unbounded path to the browser filesystem after writes
 * gained their own concurrency cap. The reported multi-second L2 stall remains
 * unattributed; this gate provides a bounded, observable point for diagnosis.
 */
let active = 0;
let epoch = 0;
const queue: Array<{ resolve: () => void; reject: (error: Error) => void }> = [];

/** Return the page-wide OPFS read gate occupancy. */
export function getOpfsReadGateStats(): { active: number; queued: number } {
  return { active, queued: queue.length };
}

/** Reset gate state and reject queued readers. Intended for test isolation. */
export function resetOpfsReadGate(): void {
  epoch += 1;
  active = 0;
  const error = new Error('OPFS read gate reset');
  for (const waiter of queue.splice(0)) waiter.reject(error);
}

/** Run one OPFS read under the page-wide FIFO concurrency cap. */
export function withOpfsReadGate<T>(run: () => Promise<T>): Promise<T> {
  const acquiredEpoch = epoch;
  const acquire =
    active < config.cache.opfsReadConcurrency
      ? ((active += 1), Promise.resolve())
      : new Promise<void>((resolve, reject) =>
          queue.push({
            resolve: () => {
              active += 1;
              resolve();
            },
            reject,
          })
        );

  return acquire.then(async () => {
    try {
      return await run();
    } finally {
      if (acquiredEpoch === epoch) {
        active -= 1;
        queue.shift()?.resolve();
      }
    }
  });
}
