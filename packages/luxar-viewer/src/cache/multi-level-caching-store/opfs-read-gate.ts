import { config } from '../../config';

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
 * to `hold` for that. A queued read whose signal aborts leaves the queue at once.
 */
let active = 0;
let epoch = 0;

interface Waiter {
  readonly start: () => void;
  readonly reject: (error: unknown) => void;
  live: boolean;
}

const queue: Waiter[] = [];
let queued = 0;

/** Return the page-wide OPFS read gate occupancy. */
export function getOpfsReadGateStats(): { active: number; queued: number } {
  return { active, queued };
}

/** Reset gate state and reject queued readers. Intended for test isolation. */
export function resetOpfsReadGate(): void {
  epoch += 1;
  active = 0;
  queued = 0;
  const error = new Error('OPFS read gate reset');
  for (const waiter of queue.splice(0)) if (waiter.live) waiter.reject(error);
}

/** Start the oldest live waiter, if any. */
function startNext(): void {
  for (let waiter = queue.shift(); waiter; waiter = queue.shift()) {
    if (!waiter.live) continue;
    waiter.live = false;
    queued -= 1;
    waiter.start();
    return;
  }
}

/** Wait for a slot; an abort of `signal` while waiting rejects and frees the place. */
function acquire(signal: AbortSignal | undefined): Promise<void> {
  if (signal?.aborted) return Promise.reject(abortError(signal));
  if (active < config.cache.opfsReadConcurrency) {
    active += 1;
    return Promise.resolve();
  }
  return new Promise<void>((resolve, reject) => {
    const onAbort = (): void => {
      if (!waiter.live) return;
      waiter.live = false;
      queued -= 1;
      reject(abortError(signal!));
    };
    const waiter: Waiter = {
      live: true,
      start: () => {
        signal?.removeEventListener('abort', onAbort);
        active += 1;
        resolve();
      },
      reject: (error) => {
        signal?.removeEventListener('abort', onAbort);
        reject(error);
      },
    };
    queue.push(waiter);
    queued += 1;
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

function abortError(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException('OPFS read aborted while queued', 'AbortError');
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
  const acquiredEpoch = epoch;
  return acquire(signal).then(async () => {
    let pending = 1;
    const settle = (): void => {
      pending -= 1;
      if (pending > 0 || acquiredEpoch !== epoch) return;
      active -= 1;
      startNext();
    };
    const hold = <R>(io: Promise<R>): Promise<R> => {
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
