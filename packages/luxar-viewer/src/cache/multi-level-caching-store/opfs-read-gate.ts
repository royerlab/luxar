/**
 * Maximum OPFS chunk reads running concurrently across the page.
 *
 * Deep progressive passes can request several hundred cached chunks at once.
 * Chromium's main-thread OPFS promises become pathologically slow under that
 * fan-out even though the same files are fast in smaller batches. Sixty-four
 * keeps the local tier well above the network lane's concurrency while
 * preventing the browser-level stampede.
 */
export const MAX_CONCURRENT_OPFS_READS = 64;

let active = 0;
const queue: Array<() => void> = [];

/** Run one OPFS read under the page-wide FIFO concurrency cap. */
export function withOpfsReadGate<T>(run: () => Promise<T>): Promise<T> {
  const acquire =
    active < MAX_CONCURRENT_OPFS_READS
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
