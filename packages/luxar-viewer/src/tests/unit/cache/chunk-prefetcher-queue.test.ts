/**
 * The neighbour prefetch queue is bounded, newest first, and cancellable.
 *
 * It used to be an unbounded FIFO `Set` with no signal: a fast pan queued the
 * neighbours of every chunk it passed, the prefetcher kept fetching the
 * OLDEST of them (the region the view had already left) ahead of the current
 * one, and a dispose left its in-flight reads running.
 */

import { describe, it, expect, vi } from 'vitest';
import { ChunkPrefetcher } from '../../../cache/chunk-prefetcher';

/** A store whose prefetch reads stay pending until released, recording their options. */
function controlledStore() {
  const calls: Array<{ key: string; signal?: AbortSignal; release: () => void }> = [];
  const store = {
    getResult: vi.fn(
      (key: string, options?: { signal?: AbortSignal }) =>
        new Promise((resolve) =>
          calls.push({
            key,
            signal: options?.signal,
            release: () => resolve({ ok: true, value: new Uint8Array(1) }),
          })
        )
    ),
    setPrefetcher: vi.fn(),
  };
  return { store, calls };
}

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

describe('ChunkPrefetcher queue', () => {
  it('bounds the queue and serves the newest neighbours first', async () => {
    const { store, calls } = controlledStore();
    const prefetcher = new ChunkPrefetcher(store as never, { maxConcurrent: 1 });
    prefetcher.registerArrayBounds('data', [100_000], [1]);

    // 500 accesses, 3 apart: 1,000 distinct neighbours.
    for (let i = 1; i <= 500; i++) prefetcher.onAccess(`data/${i * 3}`);
    await settle();
    const queued = prefetcher.getStats().queued;

    calls[0].release();
    await settle();
    const next = calls[1]?.key;
    prefetcher.dispose();

    expect(queued).toBeLessThanOrEqual(256);
    // The newest access was data/1500; its neighbours are data/1499 and data/1501.
    expect(['data/1499', 'data/1501']).toContain(next);
  });

  it('aborts its in-flight reads on dispose', async () => {
    const { store, calls } = controlledStore();
    const prefetcher = new ChunkPrefetcher(store as never, { maxConcurrent: 2 });
    prefetcher.registerArrayBounds('data', [100], [1]);
    prefetcher.onAccess('data/50');
    await settle();
    expect(calls).toHaveLength(2);

    prefetcher.dispose();

    expect(calls.map((call) => call.signal?.aborted)).toEqual([true, true]);
    for (const call of calls) call.release();
  });
});
