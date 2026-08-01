import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  ValidationQueue,
  getRemoteContentHash,
  type QueueEntry,
} from '../../../cache/multi-level-caching-store/validation-queue';

/**
 * The static `queues` map is global state — every test cancels the
 * entries it created in afterEach to keep tests independent.
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
  // Cancellation is identity-scoped: we track every entry serialize() hands
  // us (via onStart) and cancel them all in afterEach to keep the static
  // `queues` map clean between tests.
  const activeEntries = new Set<QueueEntry>();
  let idCounter = 0;
  function freshId(label: string) {
    return `${label}-${idCounter++}-${Math.random().toString(36).slice(2)}`;
  }

  /**
   * serialize() wrapper that captures the created entry synchronously so the
   * test (and afterEach) can cancel exactly that entry.
   */
  function serialize(
    id: string,
    task: (signal: AbortSignal) => Promise<void>
  ): { promise: Promise<void>; entry: QueueEntry } {
    let entry!: QueueEntry;
    const promise = ValidationQueue.serialize(id, task, (e) => {
      entry = e;
      activeEntries.add(e);
    });
    return { promise, entry };
  }

  afterEach(() => {
    for (const e of activeEntries) ValidationQueue.cancel(e);
    activeEntries.clear();
  });

  it('invokes the task with an AbortSignal', async () => {
    const id = freshId('a');
    let receivedSignal: AbortSignal | null = null;
    await serialize(id, async (signal) => {
      receivedSignal = signal;
    }).promise;
    expect(receivedSignal).not.toBeNull();
    expect(receivedSignal!.aborted).toBe(false);
  });

  it('onStart receives the entry synchronously, before the first await', () => {
    const id = freshId('sync');
    let entry: QueueEntry | null = null;
    // Do NOT await — the entry must be available the instant serialize returns.
    void ValidationQueue.serialize(
      id,
      async () => {},
      (e) => {
        entry = e;
        activeEntries.add(e);
      }
    );
    expect(entry).not.toBeNull();
    expect(entry!.abort.signal.aborted).toBe(false);
  });

  it('serializes sequential calls for the same datasetId', async () => {
    const id = freshId('seq');
    const order: number[] = [];
    const d1 = deferred<void>();
    const d2 = deferred<void>();

    const p1 = serialize(id, async () => {
      order.push(1);
      await d1.promise;
      order.push(2);
    }).promise;
    const p2 = serialize(id, async () => {
      order.push(3);
      await d2.promise;
      order.push(4);
    }).promise;

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

    const pA = serialize(a, async () => {
      order.push('a-start');
      await dA.promise;
      order.push('a-end');
    }).promise;
    const pB = serialize(b, async () => {
      order.push('b-start');
      await dB.promise;
      order.push('b-end');
    }).promise;

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

  it('cancel(entry) fires the AbortSignal mid-task (owner dispose)', async () => {
    // Models the owning store's dispose(): aborting ITS OWN in-flight entry
    // must unwind the running task promptly.
    const id = freshId('cancel-mid');
    let sawAbort = false;
    const taskGate = deferred<void>();

    const { promise, entry } = serialize(id, async (signal) => {
      signal.addEventListener('abort', () => {
        sawAbort = true;
        taskGate.resolve();
      });
      await taskGate.promise;
    });

    // Let the task subscribe to the signal.
    await flush();
    ValidationQueue.cancel(entry);
    await promise;
    expect(sawAbort).toBe(true);
  });

  it('cancel(entry) is abort-only: map cleanup happens on settle, not in cancel', async () => {
    const id = freshId('cancel-settle');
    const gate = deferred<void>();
    // Running head that ignores its abort signal — it settles only when the
    // gate resolves, so we can observe the map still holding it post-cancel.
    const { promise, entry } = serialize(id, async () => {
      await gate.promise;
    });
    await flush();
    ValidationQueue.cancel(entry);

    // cancel does NOT remove the entry: it is still the head, so a fresh
    // serialize() must queue BEHIND it rather than start immediately.
    let secondStarted = false;
    const p2 = serialize(id, async () => {
      secondStarted = true;
    }).promise;
    await flush();
    expect(secondStarted).toBe(false);

    // Once the head settles, serialize's `finally` head-guard evicts it and
    // the queued successor runs.
    gate.resolve();
    await promise;
    await p2;
    expect(secondStarted).toBe(true);
  });

  it('cancel(entry) on a waiting head skips it AND keeps the FIFO chain intact', async () => {
    const id = freshId('skip-while-waiting');
    const d1 = deferred<void>();
    let task2Ran = false;
    let task3Started = false;

    // Task 1: running head, blocked on d1.
    const p1 = serialize(id, async () => {
      await d1.promise;
    }).promise;

    // Task 2: queued behind task 1; it becomes the map head.
    const { promise: p2, entry: entry2 } = serialize(id, async () => {
      task2Ran = true;
    });

    await flush();
    // Cancel the WAITING head (task 2's own entry). abort-only cancel leaves it
    // in the map, so successors stay chained behind it.
    ValidationQueue.cancel(entry2);

    // Task 3 enqueued AFTER the cancel. If cancel had DELETED task 2's (head)
    // entry, task 3 would see `previous === undefined` and run CONCURRENTLY
    // with the still-running task 1 — severing FIFO. It must instead stay
    // chained behind the running task 1.
    const p3 = serialize(id, async () => {
      task3Started = true;
    }).promise;
    await flush();
    expect(task3Started).toBe(false); // still chained behind the running task 1

    // Release task 1: task 2's chain hits the `if (abort.signal.aborted)
    // return` guard and is skipped; task 3 then runs.
    d1.resolve();
    await p1;
    await p2;
    await p3;
    expect(task2Ran).toBe(false); // aborted waiting head skipped
    expect(task3Started).toBe(true); // ran only after task 1 released
  });

  it('delete-only-if-still-head: an older entry finishing does not remove the newer head', async () => {
    const id = freshId('not-head');
    const d1 = deferred<void>();
    const p1 = serialize(id, async () => {
      await d1.promise;
    }).promise;

    // p2 replaces p1 as the map head (last-writer-wins on set()).
    const d2 = deferred<void>();
    const p2 = serialize(id, async () => {
      await d2.promise;
    }).promise;

    // Resolve p1. Its finally block must NOT delete p2's (still-head) entry.
    d1.resolve();
    await p1;

    // If p2's entry survived as head, a fresh serialize() must queue BEHIND it
    // (not start immediately).
    let p3Started = false;
    const p3 = serialize(id, async () => {
      p3Started = true;
    }).promise;
    await flush();
    expect(p3Started).toBe(false); // p2 still holds the queue

    d2.resolve();
    await p2;
    await p3;
    expect(p3Started).toBe(true); // p3 ran only after p2 released
  });

  it('two owners: an older owner cancelling cannot skip a newer owner (issue #682)', async () => {
    // Reproduces the report's interleaving:
    //   1. Store A serializes taskA -> entryA becomes head, taskA starts.
    //   2. Store B serializes taskB -> entryB replaces entryA as head; taskB
    //      waits for taskA to finish.
    //   3. Store A disposes -> cancel(entryA). Identity-scoped, so it aborts
    //      ONLY entryA (already replaced as head) and must NOT touch entryB.
    //   4. A is released -> taskB must still run (newer validation preserved),
    //      while taskA observes its own abort.
    const id = freshId('two-owner');
    const aGate = deferred<void>();
    const bGate = deferred<void>();
    let aSawAbort = false;
    let bRan = false;
    let cStarted = false;

    // Owner A: running head, blocks on aGate.
    const { promise: pA, entry: entryA } = serialize(id, async (signal) => {
      await aGate.promise;
      aSawAbort = signal.aborted;
    });

    // Owner B: queued behind A, becomes the map head. Gated so we can observe
    // a later entry queue behind it.
    const { promise: pB, entry: entryB } = serialize(id, async (signal) => {
      // B's own signal must be clean — the older owner's cancel is not ours.
      expect(signal.aborted).toBe(false);
      bRan = true;
      await bGate.promise;
    });

    // Let A start and B enqueue.
    await flush();
    expect(entryA).not.toBe(entryB);

    // Older owner A disposes: aborts entryA only.
    ValidationQueue.cancel(entryA);
    // entryB is still the head; cancel of the stale entryA must not evict it.
    expect(entryB.abort.signal.aborted).toBe(false);

    // Owner C enqueues after the cancel. entryB must still be the head, so C
    // chains behind B (it would run immediately if cancel had evicted B).
    const pC = serialize(id, async () => {
      cStarted = true;
    }).promise;

    // Release A; B's chain now runs and blocks on bGate. C stays queued.
    aGate.resolve();
    await pA;
    await flush();
    expect(bRan).toBe(true); // newer validation NOT skipped
    expect(aSawAbort).toBe(true); // older owner's own task was aborted
    expect(cStarted).toBe(false); // C is chained behind the surviving head B

    // Release B; only now may C run.
    bGate.resolve();
    await pB;
    await pC;
    expect(cStarted).toBe(true);
  });

  it('cancel(entry) on an already-settled entry is a safe no-op', async () => {
    // The owner's dispose() may fire after its validation already completed —
    // cancelling a settled (map-evicted) entry must not throw and must not
    // perturb an unrelated live queue.
    const doneId = freshId('settled');
    const { promise: donePromise, entry: settled } = serialize(doneId, async () => {});
    await donePromise; // settles and self-evicts from the map

    const liveId = freshId('live');
    const taskGate = deferred<void>();
    const { promise: live, entry: liveEntry } = serialize(liveId, async () => {
      await taskGate.promise;
    });
    await flush();

    // Cancel the already-settled entry twice — must not throw and must not
    // touch the unrelated live queue.
    expect(() => ValidationQueue.cancel(settled)).not.toThrow();
    expect(() => ValidationQueue.cancel(settled)).not.toThrow();

    // The unrelated live entry's signal is untouched by the settled cancels.
    expect(liveEntry.abort.signal.aborted).toBe(false);

    taskGate.resolve();
    await live; // Would hang/reject if the settled cancel had aborted it.
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

  it('returns the content_hash (content-hash mode) on a 200 response with the attr', async () => {
    global.fetch = vi.fn(async () =>
      mockResponse(200, JSON.stringify({ content_hash: 'abc123' }))
    ) as unknown as typeof fetch;
    expect(await getRemoteContentHash('https://example.com/d.zarr', {})).toEqual({
      hash: 'abc123',
      mode: 'content-hash',
    });
  });

  it('falls back to an implicit zattrs-hash token when the attr is absent', async () => {
    // Standalone .gsplats.zarr / external datasets carry no content_hash;
    // the SHA-256 of the raw .zattrs bytes serves as the validation token
    // so a dataset regenerated in place still invalidates the OPFS cache.
    global.fetch = vi.fn(async () =>
      mockResponse(200, JSON.stringify({ other: 'attr' }))
    ) as unknown as typeof fetch;
    const token = await getRemoteContentHash('https://example.com/d.zarr', {});
    expect(token).toEqual({
      hash: expect.stringMatching(/^zattrs:[0-9a-f]{64}$/),
      mode: 'zattrs-hash',
    });
  });

  it('implicit token is stable for identical bytes and differs for changed bytes', async () => {
    const fetchBody = (body: string) =>
      vi.fn(async () => mockResponse(200, body)) as unknown as typeof fetch;
    global.fetch = fetchBody(JSON.stringify({ timestamp: 't1' }));
    const a = await getRemoteContentHash('https://example.com/d.zarr', {});
    global.fetch = fetchBody(JSON.stringify({ timestamp: 't1' }));
    const b = await getRemoteContentHash('https://example.com/d.zarr', {});
    global.fetch = fetchBody(JSON.stringify({ timestamp: 't2' }));
    const c = await getRemoteContentHash('https://example.com/d.zarr', {});
    expect(a!.hash).toBe(b!.hash);
    expect(a!.hash).not.toBe(c!.hash);
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
