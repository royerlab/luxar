import { describe, it, expect } from 'vitest';
import { OpfsWriteQueue } from '../../../cache/multi-level-caching-store/opfs-write-queue';

/** A manually-resolvable gate for controlling task completion in tests. */
function makeGate(): { promise: Promise<void>; release: () => void } {
  let release!: () => void;
  const promise = new Promise<void>((r) => {
    release = r;
  });
  return { promise, release };
}

describe('OpfsWriteQueue', () => {
  it('never runs more than `concurrency` tasks at once', async () => {
    const q = new OpfsWriteQueue({ concurrency: 2, maxDepth: 100, maxBytes: 100 });
    const gate = makeGate();
    let started = 0;
    let finished = 0;

    for (let i = 0; i < 5; i++) {
      q.enqueue(
        `k${i}`,
        async () => {
          started++;
          await gate.promise;
          finished++;
        },
        1
      );
    }

    // pump() runs synchronously inside enqueue, so exactly `concurrency`
    // tasks have started; the rest wait in the pending queue.
    expect(started).toBe(2);
    expect(finished).toBe(0);
    expect(q.stats().inFlight).toBe(2);
    expect(q.stats().depth).toBe(3);

    gate.release();
    await q.drain();

    expect(finished).toBe(5);
    expect(q.stats().inFlight).toBe(0);
    expect(q.stats().depth).toBe(0);
  });

  it('coalesces repeat enqueues of the same key, keeping the latest task', async () => {
    const q = new OpfsWriteQueue({ concurrency: 1, maxDepth: 100, maxBytes: 100 });
    const gate = makeGate();
    const order: string[] = [];

    // Occupy the single slot so the coalescing target stays pending.
    q.enqueue(
      'busy',
      async () => {
        await gate.promise;
        order.push('busy');
      },
      1
    );
    q.enqueue(
      'k',
      async () => {
        order.push('k-v1');
      },
      3
    );
    q.enqueue(
      'k',
      async () => {
        order.push('k-v2');
      },
      5
    );

    // 'busy' in flight, a single coalesced 'k' entry pending (not two).
    expect(q.stats().inFlight).toBe(1);
    expect(q.stats().depth).toBe(1);
    expect(q.stats().pendingBytes).toBe(5);

    gate.release();
    await q.drain();

    // v1 was replaced by v2 and never ran.
    expect(order).toEqual(['busy', 'k-v2']);
  });

  it('drops the oldest pending task past maxDepth (best-effort overflow)', async () => {
    const q = new OpfsWriteQueue({ concurrency: 1, maxDepth: 2, maxBytes: 100 });
    const gate = makeGate();
    const ran: string[] = [];

    q.enqueue(
      'busy',
      async () => {
        await gate.promise;
        ran.push('busy');
      },
      1
    );
    q.enqueue(
      'a',
      async () => {
        ran.push('a');
      },
      1
    );
    q.enqueue(
      'b',
      async () => {
        ran.push('b');
      },
      1
    );
    // pending = [a, b] (depth 2 == maxDepth). Enqueue a third → drop oldest (a).
    q.enqueue(
      'c',
      async () => {
        ran.push('c');
      },
      1
    );

    expect(q.stats().depth).toBe(2);
    expect(q.stats().dropped).toBe(1);

    gate.release();
    await q.drain();

    // 'a' was dropped; 'busy' (in flight) + the two newest survive.
    expect(ran).toEqual(['busy', 'b', 'c']);
  });

  it('drops the oldest pending task when retained bytes exceed maxBytes', async () => {
    const q = new OpfsWriteQueue({ concurrency: 1, maxDepth: 100, maxBytes: 5 });
    const gate = makeGate();
    const ran: string[] = [];

    q.enqueue(
      'busy',
      async () => {
        await gate.promise;
        ran.push('busy');
      },
      1
    );
    q.enqueue(
      'a',
      async () => {
        ran.push('a');
      },
      3
    );
    q.enqueue(
      'b',
      async () => {
        ran.push('b');
      },
      3
    );

    expect(q.stats()).toMatchObject({ depth: 1, pendingBytes: 3, dropped: 1, maxBytes: 5 });

    gate.release();
    await q.drain();

    expect(ran).toEqual(['busy', 'b']);
    expect(q.stats().pendingBytes).toBe(0);
  });

  it('clear() drops pending tasks but lets in-flight tasks finish', async () => {
    const q = new OpfsWriteQueue({ concurrency: 1, maxDepth: 100, maxBytes: 100 });
    const gate = makeGate();
    const ran: string[] = [];

    q.enqueue(
      'busy',
      async () => {
        await gate.promise;
        ran.push('busy');
      },
      1
    );
    q.enqueue(
      'a',
      async () => {
        ran.push('a');
      },
      2
    );
    q.enqueue(
      'b',
      async () => {
        ran.push('b');
      },
      3
    );
    expect(q.stats().depth).toBe(2);
    expect(q.stats().pendingBytes).toBe(5);

    q.clear();
    expect(q.stats().depth).toBe(0);
    expect(q.stats().pendingBytes).toBe(0);

    gate.release();
    await q.drain();

    // In-flight 'busy' completed; pending a/b were dropped by clear().
    expect(ran).toEqual(['busy']);
  });

  it('drain() resolves only after every task settles, even if some throw', async () => {
    const q = new OpfsWriteQueue({ concurrency: 2, maxDepth: 100, maxBytes: 100 });
    let finished = 0;

    for (let i = 0; i < 6; i++) {
      q.enqueue(
        `k${i}`,
        async () => {
          await Promise.resolve();
          if (i === 3) throw new Error('boom'); // a throwing task must not stall the pump
          finished++;
        },
        1
      );
    }

    await q.drain();

    expect(finished).toBe(5); // the 5 non-throwing tasks all ran
    expect(q.stats().inFlight).toBe(0);
    expect(q.stats().depth).toBe(0);
  });

  it('clamps non-positive config to a floor of 1', () => {
    const q = new OpfsWriteQueue({ concurrency: 0, maxDepth: 0, maxBytes: 0 });
    expect(q.stats().concurrency).toBe(1);
    expect(q.stats().maxDepth).toBe(1);
    expect(q.stats().maxBytes).toBe(1);
  });

  it('holds the concurrency cap even if a task re-enters enqueue() during its own run', async () => {
    // A task whose run() synchronously enqueues more work must not let the
    // re-entrant pump start extra tasks past the cap (the re-entrant call sees
    // the current task before it is counted in inFlightPromises).
    const concurrency = 1;
    const q = new OpfsWriteQueue({ concurrency, maxDepth: 100, maxBytes: 100 });
    const gate = makeGate();
    let maxInFlight = 0;
    let reentered = false;

    q.enqueue(
      'a',
      async () => {
        // Re-enter synchronously (before the first await) — the danger window.
        if (!reentered) {
          reentered = true;
          q.enqueue(
            'b',
            async () => {
              await Promise.resolve();
            },
            1
          );
        }
        maxInFlight = Math.max(maxInFlight, q.stats().inFlight);
        await gate.promise;
      },
      1
    );

    // With concurrency 1, 'b' must NOT be running while 'a' is in flight.
    expect(q.stats().inFlight).toBe(1);
    gate.release();
    await q.drain();
    expect(maxInFlight).toBeLessThanOrEqual(concurrency);
    expect(q.stats().inFlight).toBe(0);
    expect(q.stats().depth).toBe(0);
  });

  it('stress: concurrency cap holds and accounting is exact under out-of-order completion', async () => {
    const concurrency = 3;
    const q = new OpfsWriteQueue({ concurrency, maxDepth: 10_000, maxBytes: 10_000 });
    const N = 60;
    const gates: Array<() => void> = [];
    let completed = 0;
    let maxInFlight = 0;

    for (let i = 0; i < N; i++) {
      q.enqueue(
        `k${i}`,
        async () => {
          await new Promise<void>((r) => gates.push(r));
          completed++;
        },
        1
      );
    }
    // Right after the synchronous enqueue burst: exactly `concurrency` running,
    // the rest pending (distinct keys → none coalesced/dropped).
    expect(q.stats().inFlight).toBe(concurrency);
    expect(q.stats().depth).toBe(N - concurrency);

    // Release gates in a deliberately scrambled order; sample the cap each step.
    let idx = 0;
    while (gates.length > 0 || q.stats().inFlight > 0) {
      maxInFlight = Math.max(maxInFlight, q.stats().inFlight);
      // scrambled pick (deterministic, no Math.random): jump by 7 through the queue
      const pick = gates.length > 0 ? (idx * 7) % gates.length : 0;
      idx++;
      const g = gates.splice(pick, 1)[0];
      if (g) g();
      await Promise.resolve();
    }
    await q.drain();

    expect(maxInFlight).toBeLessThanOrEqual(concurrency);
    expect(completed).toBe(N); // every distinct-key task ran (no loss)
    expect(q.stats().dropped).toBe(0);
    expect(q.stats().inFlight).toBe(0);
    expect(q.stats().depth).toBe(0);
  });
});
