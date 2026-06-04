/**
 * Tests for the generic progressive-LOD refinement helper.
 *
 * Asserts the rAF yield + cancellable-via-queue contract independently
 * of any leaf-type specifics.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { runProgressiveRefinement } from '../../../../../data/scene-loader/progressive/refinement';
import type { ViewState } from '../../../../../data/data-loader-types';

interface FakeLoader {
  hasMoreLODs: boolean;
  loadedLevels: number;
}

function makeQueue(pendingSequence: (Partial<ViewState> | null)[]): {
  takePending: () => Partial<ViewState> | null;
} {
  let i = 0;
  return {
    takePending: () => {
      const v = pendingSequence[i] ?? null;
      i++;
      return v;
    },
  };
}

beforeEach(() => {
  // No-op rAF stub for deterministic test timing.
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    queueMicrotask(() => cb(performance.now()));
    return 0 as unknown as number;
  });
});

describe('runProgressiveRefinement', () => {
  it('runs to completion when no loader has more LODs', async () => {
    const releaseLock = vi.fn();
    const updateCounts = vi.fn();
    const retrigger = vi.fn();

    const loaders = new Map<string, FakeLoader>();
    loaders.set('/a', { hasMoreLODs: false, loadedLevels: 1 });

    let processed = 0;
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([]) as never,
      processLoader: async () => {
        processed++;
      },
      anyHasMoreLODs: () => false, // immediate exit
      updateVisibleCountsInMonitor: updateCounts,
      releaseLock,
      retriggerUpdate: retrigger,
    });

    expect(processed).toBe(1); // one pass executed
    expect(updateCounts).toHaveBeenCalledTimes(1);
    expect(releaseLock).toHaveBeenCalledTimes(1);
    expect(retrigger).not.toHaveBeenCalled();
  });

  it('iterates until anyHasMoreLODs returns false', async () => {
    const releaseLock = vi.fn();
    const retrigger = vi.fn();

    const loader: FakeLoader = { hasMoreLODs: true, loadedLevels: 0 };
    const loaders = new Map([['/a', loader]]);

    let processed = 0;
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([]) as never,
      processLoader: async (_path, l) => {
        l.loadedLevels++;
        if (l.loadedLevels >= 3) l.hasMoreLODs = false;
        processed++;
      },
      anyHasMoreLODs: () => loader.hasMoreLODs,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock,
      retriggerUpdate: retrigger,
    });

    expect(processed).toBe(3); // three passes
    expect(loader.loadedLevels).toBe(3);
    expect(releaseLock).toHaveBeenCalledTimes(1);
    expect(retrigger).not.toHaveBeenCalled();
  });

  it('aborts when isActive() returns false (disposed mid-flight)', async () => {
    const retrigger = vi.fn();
    const loader: FakeLoader = { hasMoreLODs: true, loadedLevels: 0 };
    const loaders = new Map([['/a', loader]]);

    let active = true;
    let processed = 0;
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([]) as never,
      isActive: () => active,
      processLoader: async () => {
        processed++;
        active = false; // owner disposed after the first pass
      },
      // Would loop forever if the isActive abort did not fire.
      anyHasMoreLODs: () => true,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: retrigger,
    });

    expect(processed).toBe(1); // aborted before a second pass
    expect(retrigger).not.toHaveBeenCalled();
  });

  it('cancels on pending view state and hands off to retriggerUpdate', async () => {
    const releaseLock = vi.fn();
    const retrigger = vi.fn();

    const loader: FakeLoader = { hasMoreLODs: true, loadedLevels: 0 };
    const loaders = new Map([['/a', loader]]);

    // Queue says pending on the SECOND rAF tick.
    const pendingState = { displayDims: [0, 1, 2] } as Partial<ViewState>;
    const queue = makeQueue([null, pendingState]);

    let processed = 0;
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: queue as never,
      processLoader: async () => {
        processed++;
      },
      anyHasMoreLODs: () => true, // would loop forever
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock,
      retriggerUpdate: retrigger,
    });

    // One iteration runs before the pending check fires on tick 2.
    expect(processed).toBe(1);
    expect(retrigger).toHaveBeenCalledWith(pendingState);
    // releaseLock is NOT called on cancellation — retriggerUpdate owns it.
    expect(releaseLock).not.toHaveBeenCalled();
  });

  it('calls processLoader for every loader in the map per pass', async () => {
    const loaders = new Map<string, FakeLoader>([
      ['/a', { hasMoreLODs: true, loadedLevels: 0 }],
      ['/b', { hasMoreLODs: true, loadedLevels: 0 }],
      ['/c', { hasMoreLODs: true, loadedLevels: 0 }],
    ]);

    const visited = new Map<string, number>();
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([]) as never,
      processLoader: async (path, l) => {
        visited.set(path, (visited.get(path) ?? 0) + 1);
        l.loadedLevels++;
        if (l.loadedLevels >= 2) l.hasMoreLODs = false;
      },
      anyHasMoreLODs: () => [...loaders.values()].some((l) => l.hasMoreLODs),
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
    });

    expect(visited.get('/a')).toBe(2);
    expect(visited.get('/b')).toBe(2);
    expect(visited.get('/c')).toBe(2);
  });
});
