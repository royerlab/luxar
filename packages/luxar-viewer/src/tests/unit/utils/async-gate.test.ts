/**
 * `AsyncGate` — the counting gate behind the OPFS read gate: a capacity cap,
 * rank-then-FIFO service, abort exit while queued, idempotent releases, and a
 * reset that later releases cannot corrupt.
 */

import { describe, expect, it } from 'vitest';
import { AsyncGate, type AsyncGateRelease } from '../../../utils/async-gate';

async function flush(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

describe('AsyncGate', () => {
  it('admits up to capacity, then serves waiters FIFO as places free', async () => {
    const gate = new AsyncGate(() => 2);
    const first = await gate.acquire();
    const second = await gate.acquire();
    const order: number[] = [];
    const waiting = [3, 4].map((n) => gate.acquire().then((release) => (order.push(n), release)));
    await flush();
    expect(gate.stats()).toEqual({ active: 2, queued: 2 });

    first();
    await flush();
    expect(order).toEqual([3]);
    second();
    await flush();
    expect(order).toEqual([3, 4]);
    for (const release of await Promise.all(waiting)) release();
    expect(gate.stats()).toEqual({ active: 0, queued: 0 });
  });

  it('serves a more urgent rank first, FIFO within a rank', async () => {
    const gate = new AsyncGate(() => 1, 2);
    const held = await gate.acquire();
    const order: string[] = [];
    const take = (name: string, rank: number): Promise<AsyncGateRelease> =>
      gate.acquire(undefined, rank).then((release) => {
        order.push(name);
        release();
        return release;
      });
    const all = [take('low-a', 1), take('high', 0), take('low-b', 1)];
    held();
    await Promise.all(all);
    expect(order).toEqual(['high', 'low-a', 'low-b']);
  });

  it('a waiter whose signal aborts leaves the queue with the reason', async () => {
    const gate = new AsyncGate(() => 1);
    const held = await gate.acquire();
    const controller = new AbortController();
    const reason = new Error('superseded');
    const aborted = gate.acquire(controller.signal).catch((error: unknown) => error);
    const next = gate.acquire();
    controller.abort(reason);
    expect(await aborted).toBe(reason);
    expect(gate.stats()).toEqual({ active: 1, queued: 1 });

    held();
    (await next)();
    expect(gate.stats()).toEqual({ active: 0, queued: 0 });
  });

  it('rejects an already-aborted acquire at once with an AbortError', async () => {
    const gate = new AsyncGate(() => 1);
    const controller = new AbortController();
    controller.abort();
    const error = await gate.acquire(controller.signal).catch((e: unknown) => e);
    expect((error as { name?: string }).name).toBe('AbortError');
    expect(gate.stats()).toEqual({ active: 0, queued: 0 });
  });

  it('a release is idempotent', async () => {
    const gate = new AsyncGate(() => 2);
    const release = await gate.acquire();
    await gate.acquire();
    release();
    release();
    expect(gate.stats().active).toBe(1);
  });

  it('grants a queued waiter as soon as a raised capacity has room', async () => {
    let capacity = 1;
    const gate = new AsyncGate(() => capacity);
    await gate.acquire();
    const queued = gate.acquire();
    await flush();
    expect(gate.stats().queued).toBe(1);
    capacity = 3;
    await gate.acquire(); // arrives behind the queued waiter, and wakes it
    await queued;
    expect(gate.stats()).toEqual({ active: 3, queued: 0 });
  });

  it('reset rejects waiters and ignores releases from before it', async () => {
    const gate = new AsyncGate(() => 1);
    const stale = await gate.acquire();
    const waiter = gate.acquire().catch((error: unknown) => error);
    const reason = new Error('reset');
    gate.reset(reason);
    expect(await waiter).toBe(reason);
    expect(gate.stats()).toEqual({ active: 0, queued: 0 });

    const fresh = await gate.acquire();
    stale();
    expect(gate.stats().active).toBe(1);
    fresh();
    expect(gate.stats().active).toBe(0);
  });
});
