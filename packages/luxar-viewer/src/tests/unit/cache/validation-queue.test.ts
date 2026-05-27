import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  ValidationQueue,
  getRemoteContentHash,
} from '../../../cache/multi-level-caching-store/validation-queue';

/**
 * The static `queues` map is global state — every test cancels the
 * datasetIds it used in afterEach to keep tests independent.
 */

function mockResponse(status: number, body: ArrayBuffer | string = ''): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    async arrayBuffer() {
      return typeof body === 'string' ? new TextEncoder().encode(body).buffer : body;
    },
  } as unknown as Response;
}

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/**
 * Flush several microtask ticks. ValidationQueue.serialize chains
 * Promise.resolve().catch().then(task) — task only runs after 2-3
 * microtask ticks. Use 10 to be safely beyond any chain.
 */
async function flush() {
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

describe('ValidationQueue.serialize', () => {
  const activeIds = new Set<string>();
  function freshId(label: string) {
    const id = `${label}-${Math.random().toString(36).slice(2)}`;
    activeIds.add(id);
    return id;
  }

  afterEach(() => {
    for (const id of activeIds) ValidationQueue.cancel(id);
    activeIds.clear();
  });

  it('invokes the task with an AbortSignal', async () => {
    const id = freshId('a');
    let receivedSignal: AbortSignal | null = null;
    await ValidationQueue.serialize(id, async (signal) => {
      receivedSignal = signal;
    });
    expect(receivedSignal).not.toBeNull();
    expect(receivedSignal!.aborted).toBe(false);
  });

  it('serializes sequential calls for the same datasetId', async () => {
    const id = freshId('seq');
    const order: number[] = [];
    const d1 = deferred<void>();
    const d2 = deferred<void>();

    const p1 = ValidationQueue.serialize(id, async () => {
      order.push(1);
      await d1.promise;
      order.push(2);
    });
    const p2 = ValidationQueue.serialize(id, async () => {
      order.push(3);
      await d2.promise;
      order.push(4);
    });

    // Let p1 begin.
    await flush();
    expect(order).toEqual([1]);
    d1.resolve();
    await p1;
    await flush();
    expect(order).toEqual([1, 2, 3]); // p2 enters only after p1 finishes
    d2.resolve();
    await p2;
    expect(order).toEqual([1, 2, 3, 4]);
  });

  it('runs different datasetIds in parallel', async () => {
    const a = freshId('a');
    const b = freshId('b');
    const order: string[] = [];
    const dA = deferred<void>();
    const dB = deferred<void>();

    const pA = ValidationQueue.serialize(a, async () => {
      order.push('a-start');
      await dA.promise;
      order.push('a-end');
    });
    const pB = ValidationQueue.serialize(b, async () => {
      order.push('b-start');
      await dB.promise;
      order.push('b-end');
    });

    await flush();
    // Both should have started independently.
    expect(order).toContain('a-start');
    expect(order).toContain('b-start');

    dB.resolve();
    await pB;
    expect(order).toContain('b-end');
    // 'a' has not finished yet.
    expect(order).not.toContain('a-end');

    dA.resolve();
    await pA;
    expect(order).toContain('a-end');
  });

  it('cancel(datasetId) fires the AbortSignal mid-task', async () => {
    const id = freshId('cancel-mid');
    let sawAbort = false;
    const taskGate = deferred<void>();

    const p = ValidationQueue.serialize(id, async (signal) => {
      signal.addEventListener('abort', () => {
        sawAbort = true;
        taskGate.resolve();
      });
      await taskGate.promise;
    });

    // Let the task subscribe to the signal.
    await flush();
    ValidationQueue.cancel(id);
    await p;
    expect(sawAbort).toBe(true);
  });

  it('cancel(datasetId) removes the queue entry', async () => {
    const id = freshId('cancel-rm');
    const taskGate = deferred<void>();
    const p = ValidationQueue.serialize(id, async () => {
      await taskGate.promise;
    });
    await flush();
    ValidationQueue.cancel(id);
    // After cancel, a fresh serialize() should not queue behind the old one —
    // it should start immediately (no prior entry).
    let secondStartedImmediately = false;
    const p2 = ValidationQueue.serialize(id, async () => {
      secondStartedImmediately = true;
    });
    await flush();
    expect(secondStartedImmediately).toBe(true);

    taskGate.resolve();
    await p;
    await p2;
  });

  it('skips the task when abort fires while still waiting in line', async () => {
    const id = freshId('skip-while-waiting');
    const d1 = deferred<void>();
    let task2Ran = false;

    // Task 1 holds the queue.
    const p1 = ValidationQueue.serialize(id, async () => {
      await d1.promise;
    });

    // Task 2 enqueued behind task 1. By the time of registration, queues[id]
    // points at task 2's entry (last-writer wins on the map.set).
    const p2 = ValidationQueue.serialize(id, async () => {
      task2Ran = true;
    });

    // Cancel fires task 2's abort controller (queues[id] is now task 2's
    // entry). When task 1 finishes, task 2's chain runs the
    // `if (abort.signal.aborted) return` guard and skips the task body.
    await flush();
    ValidationQueue.cancel(id);
    d1.resolve();
    await p1;
    await p2;
    expect(task2Ran).toBe(false);
  });

  it('delete-only-if-still-head: a newer entry is not removed by an older one finishing', async () => {
    const id = freshId('not-head');
    const d1 = deferred<void>();
    const p1 = ValidationQueue.serialize(id, async () => {
      await d1.promise;
    });

    // p2 replaces p1 in the queues map (last-writer-wins on the set()
    // call inside serialize).
    const d2 = deferred<void>();
    const p2 = ValidationQueue.serialize(id, async () => {
      await d2.promise;
    });

    // Now resolve p1. Its finally block should NOT delete p2's entry.
    d1.resolve();
    await p1;
    // If p2's entry was clobbered, cancel(id) would be a no-op; instead
    // we can observe it via the side-effect of cancel still firing p2's
    // abort.
    let cancelled = false;
    const dCancel = deferred<void>();
    const p3probe = ValidationQueue.serialize(id, async () => {
      // never runs — p2 still holds the queue.
    }).then(() => {
      cancelled = true;
      dCancel.resolve();
    });
    await flush();
    // p2 is still in flight (its abort is the head's abort).
    ValidationQueue.cancel(id);
    d2.resolve();
    await p2;
    await dCancel.promise;
    await p3probe;
    expect(cancelled).toBe(true);
  });

  // workers.md O3 / cache.md G11 [P5]: audit-id moved to comment per Phase E53.
  it('cancel(unknownId) is a safe no-op', async () => {
    // [cache.md/G11][P5] cancel on an id with no entry must not throw and
    // must not perturb any unrelated queue. The comment in the source
    // implies safety but no test pinned it.
    const id = freshId('unrelated');
    const taskGate = deferred<void>();
    const p = ValidationQueue.serialize(id, async () => {
      await taskGate.promise;
    });
    await flush();

    // Cancel a totally separate, never-registered id — must not throw.
    expect(() => ValidationQueue.cancel('never-registered-id')).not.toThrow();
    expect(() => ValidationQueue.cancel('')).not.toThrow();

    // Unrelated cancels do not interfere with the live queue: the original
    // task is still pending and resolves normally when we release it.
    taskGate.resolve();
    await p; // No timeout — would hang if the unrelated cancel had aborted it.
  });
});

describe('getRemoteContentHash', () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it('returns the content_hash on a 200 response with the attr', async () => {
    global.fetch = vi.fn(async () =>
      mockResponse(200, JSON.stringify({ content_hash: 'abc123' }))
    ) as unknown as typeof fetch;
    expect(await getRemoteContentHash('https://example.com/d.zarr', {})).toBe('abc123');
  });

  it('returns null when the attr is absent', async () => {
    global.fetch = vi.fn(async () =>
      mockResponse(200, JSON.stringify({ other: 'attr' }))
    ) as unknown as typeof fetch;
    expect(await getRemoteContentHash('https://example.com/d.zarr', {})).toBeNull();
  });

  it('returns null when the response is not ok', async () => {
    global.fetch = vi.fn(async () => mockResponse(404)) as unknown as typeof fetch;
    expect(await getRemoteContentHash('https://example.com/d.zarr', {})).toBeNull();
  });

  it('returns null when fetch returns undefined (retry budget exhausted)', async () => {
    global.fetch = vi.fn(async () => {
      throw new Error('network');
    }) as unknown as typeof fetch;
    expect(
      await getRemoteContentHash('https://example.com/d.zarr', { timeoutMsOverride: 1 })
    ).toBeNull();
  });

  it('returns null on JSON parse failure', async () => {
    global.fetch = vi.fn(async () =>
      mockResponse(200, '{ not valid json')
    ) as unknown as typeof fetch;
    expect(await getRemoteContentHash('https://example.com/d.zarr', {})).toBeNull();
  });
});
