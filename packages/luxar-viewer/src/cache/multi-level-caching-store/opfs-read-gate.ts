import { config } from '../../config';
import { AsyncGate } from '../../utils/async-gate';

/**
 * Maximum OPFS chunk reads running concurrently across the page.
 *
 * Deep progressive passes can request several hundred cached chunks at once.
 * Reads were the last unbounded path to the browser filesystem after writes
 * gained their own concurrency cap. The reported multi-second L2 stall remains
 * unattributed; this gate provides a bounded, observable point for diagnosis.
 *
 * A lease lasts until the read's FILE I/O settles, not merely until `run()`
 * does: a caller that races its read against a timeout gives up while the
 * browser keeps reading, and releasing the slot then would let more reads start
 * than the cap allows (and under-report occupancy). `run()` hands the real I/O
 * to `hold` for that. A queued read whose signal aborts leaves the queue at once
 * (the shared {@link AsyncGate}'s contract).
 */
const gate = new AsyncGate(() => config.cache.opfsReadConcurrency);

/** Return the page-wide OPFS read gate occupancy. */
export function getOpfsReadGateStats(): { active: number; queued: number } {
  return gate.stats();
}

/** Reset gate state and reject queued readers. Intended for test isolation. */
export function resetOpfsReadGate(): void {
  gate.reset(new Error('OPFS read gate reset'));
}

/**
 * Run one OPFS read under the page-wide FIFO concurrency cap.
 *
 * @param run - The read. Pass the real file I/O through `hold` (it returns the
 *   same promise): the slot is held until `run()` AND every held promise have
 *   settled, so a read abandoned by a timeout still counts while it runs.
 * @param signal - Aborting it while the read still waits for a slot rejects
 *   with the signal's reason (or an `AbortError`) and frees its queue place.
 *   Once started, `run` owns the signal.
 */
export function withOpfsReadGate<T>(
  run: (hold: <R>(io: Promise<R>) => Promise<R>) => Promise<T>,
  signal?: AbortSignal
): Promise<T> {
  return gate.acquire(signal).then(async (release) => {
    let pending = 1;
    let released = false;
    const settle = (): void => {
      pending -= 1;
      // The lease ends once: a hold arriving after it ended passes its I/O
      // straight through instead of re-opening the count.
      if (pending > 0 || released) return;
      released = true;
      release();
    };
    const hold = <R>(io: Promise<R>): Promise<R> => {
      if (released) return io;
      pending += 1;
      void io.then(settle, settle);
      return io;
    };
    try {
      return await run(hold);
    } finally {
      settle();
    }
  });
}
