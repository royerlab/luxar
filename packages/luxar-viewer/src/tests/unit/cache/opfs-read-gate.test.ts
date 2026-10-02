/**
 * The page-wide OPFS read gate: a queued read can leave on abort, and a lease
 * is held for as long as the real file I/O runs.
 *
 * The gate was a FIFO with no abort exit, and it released a slot when `run()`
 * settled — which, for a read whose timeout fired, is long before the browser
 * finishes the read it is still doing. The occupancy it reported (and the cap
 * it enforced) then under-counted the reads actually in progress.
 */

import { afterEach, describe, expect, it } from 'vitest';
import { config } from '../../../config';
import {
  getOpfsReadGateStats,
  resetOpfsReadGate,
  withOpfsReadGate,
} from '../../../cache/multi-level-caching-store/opfs-read-gate';

async function flush(): Promise<void> {
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => (resolve = r));
  return { promise, resolve };
}

afterEach(() => resetOpfsReadGate());

describe('OPFS read gate', () => {
  it('a queued read whose signal aborts leaves the queue without running', async () => {
    const cap = config.cache.opfsReadConcurrency;
    const blocker = deferred();
    const busy = Array.from({ length: cap }, () => withOpfsReadGate(() => blocker.promise));
    await flush();

    const controller = new AbortController();
    let ran = false;
    let failure: unknown;
    const read = withOpfsReadGate(async () => {
      ran = true;
    }, controller.signal).catch((error: unknown) => (failure = error));
    await flush();
    const queuedBefore = getOpfsReadGateStats().queued;
    controller.abort();
    await flush();
    const afterAbort = { ...getOpfsReadGateStats(), name: (failure as Error | undefined)?.name };

    blocker.resolve();
    await Promise.all(busy);
    await read;

    expect(queuedBefore).toBe(1);
    expect(afterAbort).toEqual({ active: cap, queued: 0, name: 'AbortError' });
    expect(ran).toBe(false);
  });

  it('holds the slot until held file I/O settles, even after run() gave up on it', async () => {
    const cap = config.cache.opfsReadConcurrency;
    const io = Array.from({ length: cap }, deferred);
    // Each run() hands the real I/O to `hold`, then rejects early — what a
    // timeout race does while the browser keeps reading.
    const timedOut = io.map((pending) =>
      withOpfsReadGate(async (hold) => {
        void hold(pending.promise);
        throw new Error('OPFS timeout: get exceeded 1ms');
      }).catch(() => undefined)
    );
    await Promise.all(timedOut);

    let started = false;
    const next = withOpfsReadGate(async () => {
      started = true;
    });
    await flush();
    const whileReading = { ...getOpfsReadGateStats(), started };

    for (const pending of io) pending.resolve();
    await next;

    expect(whileReading).toEqual({ active: cap, queued: 1, started: false });
    expect(getOpfsReadGateStats()).toEqual({ active: 0, queued: 0 });
  });

  it('a late hold cannot release the same slot twice', async () => {
    let hold!: <T>(io: Promise<T>) => Promise<T>;
    await withOpfsReadGate(async (lease) => {
      hold = lease;
    });
    await hold(Promise.resolve());
    expect(getOpfsReadGateStats()).toEqual({ active: 0, queued: 0 });
  });
});
