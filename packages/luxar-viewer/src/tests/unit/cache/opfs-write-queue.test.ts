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
    const q = new OpfsWriteQueue({ concurrency: 2, maxDepth: 100 });
    const gate = makeGate();
    let started = 0;
    let finished = 0;

    for (let i = 0; i < 5; i++) {
      q.enqueue(`k${i}`, async () => {
        started++;
        await gate.promise;
        finished++;
      });
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
    const q = new OpfsWriteQueue({ concurrency: 1, maxDepth: 100 });
    const gate = makeGate();
    const order: string[] = [];

    // Occupy the single slot so the coalescing target stays pending.
    q.enqueue('busy', async () => {
      await gate.promise;
      order.push('busy');
    });
    q.enqueue('k', async () => {
      order.push('k-v1');
    });
    q.enqueue('k', async () => {
      order.push('k-v2');
    });

    // 'busy' in flight, a single coalesced 'k' entry pending (not two).
    expect(q.stats().inFlight).toBe(1);
    expect(q.stats().depth).toBe(1);

    gate.release();
    await q.drain();

    // v1 was replaced by v2 and never ran.
    expect(order).toEqual(['busy', 'k-v2']);
  });

  it('drops the oldest pending task past maxDepth (best-effort overflow)', async () => {
    const q = new OpfsWriteQueue({ concurrency: 1, maxDepth: 2 });
    const gate = makeGate();
    const ran: string[] = [];

    q.enqueue('busy', async () => {
      await gate.promise;
      ran.push('busy');
    });
    q.enqueue('a', async () => {
      ran.push('a');
    });
    q.enqueue('b', async () => {
      ran.push('b');
    });
    // pending = [a, b] (depth 2 == maxDepth). Enqueue a third → drop oldest (a).
    q.enqueue('c', async () => {
      ran.push('c');
    });

    expect(q.stats().depth).toBe(2);
    expect(q.stats().dropped).toBe(1);

    gate.release();
    await q.drain();

    // 'a' was dropped; 'busy' (in flight) + the two newest survive.
    expect(ran).toEqual(['busy', 'b', 'c']);
  });

  it('clear() drops pending tasks but lets in-flight tasks finish', async () => {
    const q = new OpfsWriteQueue({ concurrency: 1, maxDepth: 100 });
    const gate = makeGate();
    const ran: string[] = [];

    q.enqueue('busy', async () => {
      await gate.promise;
      ran.push('busy');
    });
    q.enqueue('a', async () => {
      ran.push('a');
    });
    q.enqueue('b', async () => {
      ran.push('b');
    });
    expect(q.stats().depth).toBe(2);

    q.clear();
    expect(q.stats().depth).toBe(0);

    gate.release();
    await q.drain();

    // In-flight 'busy' completed; pending a/b were dropped by clear().
    expect(ran).toEqual(['busy']);
  });

  it('drain() resolves only after every task settles, even if some throw', async () => {
    const q = new OpfsWriteQueue({ concurrency: 2, maxDepth: 100 });
    let finished = 0;

    for (let i = 0; i < 6; i++) {
      q.enqueue(`k${i}`, async () => {
        await Promise.resolve();
        if (i === 3) throw new Error('boom'); // a throwing task must not stall the pump
        finished++;
      });
    }

    await q.drain();

    expect(finished).toBe(5); // the 5 non-throwing tasks all ran
    expect(q.stats().inFlight).toBe(0);
    expect(q.stats().depth).toBe(0);
  });

  it('clamps non-positive config to a floor of 1', () => {
    const q = new OpfsWriteQueue({ concurrency: 0, maxDepth: 0 });
    expect(q.stats().concurrency).toBe(1);
    expect(q.stats().maxDepth).toBe(1);
  });
});
