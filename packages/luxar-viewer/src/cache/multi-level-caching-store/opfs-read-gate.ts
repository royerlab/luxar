/**
 * Maximum OPFS chunk reads running concurrently across the page.
 *
 * Deep progressive passes can request several hundred cached chunks at once.
 * Reads were the last unbounded path to the browser filesystem after writes
 * gained their own concurrency cap. The reported multi-second L2 stall remains
 * unattributed; this gate provides a bounded, observable point for diagnosis.
 */
let active = 0;
const queue: Array<() => void> = [];

/** Run one OPFS read under the page-wide FIFO concurrency cap. */
export function withOpfsReadGate<T>(run: () => Promise<T>): Promise<T> {
  const acquire =
    active < config.cache.opfsReadConcurrency
      ? ((active += 1), Promise.resolve())
      : new Promise<void>((resolve) =>
          queue.push(() => {
            active += 1;
            resolve();
          })
        );

  return acquire.then(async () => {
    try {
      return await run();
    } finally {
      active -= 1;
      queue.shift()?.();
    }
  });
}
import { config } from '../../config';
