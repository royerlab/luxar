/**
 * Tests for the generic progressive-LOD refinement helper.
 *
 * Asserts the rAF yield + cancellable-via-queue contract independently
 * of any leaf-type specifics.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import {
  RefinementFailureTracker,
  runProgressiveRefinement,
} from '../../../../../data/scene-loader/progressive/refinement';
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

function getFakeLoaderProgress(_path: string, loader: FakeLoader) {
  return loader.hasMoreLODs ? { loaded: loader.loadedLevels, total: 3 } : null;
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
      getLoaderProgress: getFakeLoaderProgress,
      processLoader: async () => {
        processed++;
        return true;
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
      getLoaderProgress: getFakeLoaderProgress,
      processLoader: async (_path, l) => {
        l.loadedLevels++;
        if (l.loadedLevels >= 3) l.hasMoreLODs = false;
        processed++;
        return true;
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

  it('stops after a pass that leaves every pending loader at the same LOD count', async () => {
    const releaseLock = vi.fn();
    const onNoProgress = vi.fn();
    const loader: FakeLoader = { hasMoreLODs: true, loadedLevels: 2 };
    const loaders = new Map([['/a', loader]]);

    let processed = 0;
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([]) as never,
      getLoaderProgress: getFakeLoaderProgress,
      isActive: () => processed < 5,
      processLoader: async () => {
        processed++;
        return true;
      },
      anyHasMoreLODs: () => true,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock,
      retriggerUpdate: vi.fn(),
      onNoProgress,
    });

    expect(processed).toBe(1);
    expect(releaseLock).toHaveBeenCalledTimes(1);
    expect(onNoProgress).toHaveBeenCalledWith([{ path: '/a', loaded: 2, total: 3 }]);
  });

  it('continues when one loader advances, then reports only the stalled loader', async () => {
    const stalled: FakeLoader = { hasMoreLODs: true, loadedLevels: 1 };
    const progressing: FakeLoader = { hasMoreLODs: true, loadedLevels: 1 };
    const loaders = new Map([
      ['/stalled', stalled],
      ['/progressing', progressing],
    ]);
    const onNoProgress = vi.fn();
    let passes = 0;

    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([]) as never,
      getLoaderProgress: getFakeLoaderProgress,
      processLoader: async (path, loader) => {
        if (path === '/stalled') passes++;
        if (path === '/progressing' && loader.hasMoreLODs) {
          loader.loadedLevels++;
          loader.hasMoreLODs = false;
        }
        return true;
      },
      anyHasMoreLODs: () => [...loaders.values()].some((loader) => loader.hasMoreLODs),
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
      onNoProgress,
    });

    expect(passes).toBe(2);
    expect(onNoProgress).toHaveBeenCalledWith([{ path: '/stalled', loaded: 1, total: 3 }]);
  });

  it('aborts when isActive() returns false (disposed mid-flight)', async () => {
    const retrigger = vi.fn();
    const loader: FakeLoader = { hasMoreLODs: true, loadedLevels: 0 };
    const loaders = new Map([['/a', loader]]);

    let active = true;
    let processed = 0;
    const isActive = vi.fn(() => active);
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([]) as never,
      getLoaderProgress: getFakeLoaderProgress,
      isActive,
      processLoader: async (_path, currentLoader) => {
        processed++;
        currentLoader.loadedLevels++;
        active = false; // owner disposed after the first pass
        return true;
      },
      // Would loop forever if the isActive abort did not fire.
      anyHasMoreLODs: () => true,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: retrigger,
    });

    expect(processed).toBe(1); // aborted before a second pass
    expect(isActive).toHaveBeenCalledTimes(2);
    expect(retrigger).not.toHaveBeenCalled();
  });

  it('reports a loop-body throw via onError and breaks (no unhandled rejection)', async () => {
    const onError = vi.fn();
    const loader: FakeLoader = { hasMoreLODs: true, loadedLevels: 0 };
    const loaders = new Map([['/a', loader]]);

    let passes = 0;
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([]) as never,
      getLoaderProgress: getFakeLoaderProgress,
      processLoader: async () => {
        passes++;
        return true;
      },
      // Would loop forever if the throw were not caught + broken on.
      anyHasMoreLODs: () => true,
      updateVisibleCountsInMonitor: () => {
        throw new Error('monitor boom');
      },
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
      onError,
    });

    expect(passes).toBe(1);
    expect(onError).toHaveBeenCalledTimes(1);
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
      getLoaderProgress: getFakeLoaderProgress,
      processLoader: async (_path, currentLoader) => {
        processed++;
        currentLoader.loadedLevels++;
        return true;
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
      getLoaderProgress: getFakeLoaderProgress,
      processLoader: async (path, l) => {
        visited.set(path, (visited.get(path) ?? 0) + 1);
        l.loadedLevels++;
        if (l.loadedLevels >= 2) l.hasMoreLODs = false;
        return true;
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

describe('RefinementFailureTracker', () => {
  it('exhausts a path only after the configured consecutive failures', () => {
    const t = new RefinementFailureTracker(3);
    expect(t.recordFailure('/p')).toBe(false);
    expect(t.recordFailure('/p')).toBe(false);
    expect(t.isExhausted('/p')).toBe(false);
    expect(t.recordFailure('/p')).toBe(true); // crossing the cap reports once
    expect(t.isExhausted('/p')).toBe(true);
    expect(t.recordFailure('/p')).toBe(false); // already exhausted → no re-report
  });

  it('a success resets the consecutive-failure count', () => {
    const t = new RefinementFailureTracker(3);
    t.recordFailure('/p');
    t.recordFailure('/p');
    t.recordSuccess('/p'); // healthy step — counter resets
    expect(t.recordFailure('/p')).toBe(false);
    expect(t.recordFailure('/p')).toBe(false);
    expect(t.isExhausted('/p')).toBe(false);
    expect(t.recordFailure('/p')).toBe(true);
  });

  it('tracks paths independently', () => {
    const t = new RefinementFailureTracker(2);
    t.recordFailure('/a');
    expect(t.recordFailure('/b')).toBe(false);
    expect(t.recordFailure('/a')).toBe(true);
    expect(t.isExhausted('/b')).toBe(false);
  });
});

describe('runProgressiveRefinement — paint-yield budget', () => {
  /** Count animation-frame yields; the stub fires the callback on a microtask. */
  function countFrames(): { count: () => number } {
    let n = 0;
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      n++;
      queueMicrotask(() => cb(performance.now()));
      return 0 as unknown as number;
    });
    return { count: () => n };
  }

  /** A loader that needs `rungs` passes to finish. */
  function ladder(rungs: number) {
    const loader: FakeLoader = { hasMoreLODs: true, loadedLevels: 0 };
    return {
      loader,
      loaders: new Map([['/a', loader]]),
      processLoader: async (_path: string, l: FakeLoader) => {
        l.loadedLevels++;
        if (l.loadedLevels >= rungs) l.hasMoreLODs = false;
        return true;
      },
    };
  }

  it('budget 0 yields a frame before every pass (the historical cadence)', async () => {
    const frames = countFrames();
    const { loader, loaders, processLoader } = ladder(3);
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([]) as never,
      getLoaderProgress: getFakeLoaderProgress,
      processLoader,
      anyHasMoreLODs: () => loader.hasMoreLODs,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
      passBudgetMs: () => 0,
    });
    expect(loader.loadedLevels).toBe(3);
    expect(frames.count()).toBe(3);
  });

  it('a generous budget yields once before the first pass and then batches the rest', async () => {
    const frames = countFrames();
    const { loader, loaders, processLoader } = ladder(3);
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([]) as never,
      getLoaderProgress: getFakeLoaderProgress,
      processLoader,
      anyHasMoreLODs: () => loader.hasMoreLODs,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
      passBudgetMs: () => 1_000_000,
    });
    expect(loader.loadedLevels).toBe(3);
    expect(frames.count()).toBe(1);
  });

  it('yields again once the passes since the last yield have cost the budget', async () => {
    const frames = countFrames();
    // The loop reads the clock once at pass start and once at pass end, so
    // each pass "costs" 10 ms of main-thread time; budget 15 ms → a yield
    // every second pass: passes 1-2, yield, 3-4, yield, 5.
    let clock = 0;
    vi.spyOn(performance, 'now').mockImplementation(() => (clock += 10));
    try {
      const { loader, loaders, processLoader } = ladder(5);
      await runProgressiveRefinement<FakeLoader>({
        loaders,
        viewStateQueue: makeQueue([]) as never,
        getLoaderProgress: getFakeLoaderProgress,
        processLoader,
        anyHasMoreLODs: () => loader.hasMoreLODs,
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock: vi.fn(),
        retriggerUpdate: vi.fn(),
        passBudgetMs: () => 15,
      });
      expect(loader.loadedLevels).toBe(5);
      expect(frames.count()).toBe(3);
    } finally {
      vi.restoreAllMocks();
    }
  });

  it('still checks for a pending view state between batched passes and hands off the lock', async () => {
    countFrames();
    const releaseLock = vi.fn();
    const retrigger = vi.fn();
    const { loader, loaders, processLoader } = ladder(3);
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([null, { slicePosition: [0, 0, 0, 1] }]) as never,
      getLoaderProgress: getFakeLoaderProgress,
      processLoader,
      anyHasMoreLODs: () => loader.hasMoreLODs,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock,
      retriggerUpdate: retrigger,
      passBudgetMs: () => 1_000_000,
    });
    // One pass ran; the second loop-top check saw the pending state without
    // any frame yield in between.
    expect(loader.loadedLevels).toBe(1);
    expect(retrigger).toHaveBeenCalledWith({ slicePosition: [0, 0, 0, 1] });
    expect(releaseLock).not.toHaveBeenCalled();
  });
});
