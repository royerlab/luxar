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

describe('runProgressiveRefinement — bounded concurrency (B9c)', () => {
  /** Resolve after `ms` real milliseconds (so steps genuinely overlap). */
  const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

  /**
   * Ten laddered loaders with uneven step latencies. Each step "commits" its
   * next level into `committed`, the loader's settled state. Returns what the
   * loop left committed plus the peak number of steps in flight at once.
   */
  async function runLadders(): Promise<{
    committed: Map<string, number[]>;
    peak: number;
  }> {
    const levels = [3, 1, 4, 2, 5, 3, 2, 4, 1, 3];
    const loaders = new Map<string, FakeLoader>(
      levels.map((_, i) => [`/p/${i}`, { hasMoreLODs: true, loadedLevels: 0 }])
    );
    const committed = new Map<string, number[]>([...loaders.keys()].map((p) => [p, []]));
    let inFlight = 0;
    let peak = 0;
    let step = 0;
    await runProgressiveRefinement<FakeLoader>({
      loaders,
      viewStateQueue: makeQueue([]) as never,
      getLoaderProgress: (path, l) =>
        l.hasMoreLODs ? { loaded: l.loadedLevels, total: levels[Number(path.slice(3))] } : null,
      processLoader: async (path, l) => {
        if (!l.hasMoreLODs) return false;
        inFlight++;
        peak = Math.max(peak, inFlight);
        try {
          await sleep((step++ * 7) % 5);
          l.loadedLevels++;
          committed.get(path)!.push(l.loadedLevels);
          l.hasMoreLODs = l.loadedLevels < levels[Number(path.slice(3))];
          return true;
        } finally {
          inFlight--;
        }
      },
      anyHasMoreLODs: () => [...loaders.values()].some((l) => l.hasMoreLODs),
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
    });
    return { committed, peak };
  }

  it('keeps up to 4 loader steps in flight, never more', async () => {
    const { peak } = await runLadders();
    expect(peak).toBe(4);
  });

  it('settles to exactly the serial result: every ladder complete, each in level order', async () => {
    const { committed } = await runLadders();
    const levels = [3, 1, 4, 2, 5, 3, 2, 4, 1, 3];
    levels.forEach((n, i) => {
      expect(committed.get(`/p/${i}`)).toEqual(Array.from({ length: n }, (_, k) => k + 1));
    });
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

  it('a success clears an exhausted path', () => {
    const t = new RefinementFailureTracker(2);
    t.recordFailure('/p');
    t.recordFailure('/p');
    expect(t.isExhausted('/p')).toBe(true);

    t.recordSuccess('/p');

    expect(t.isExhausted('/p')).toBe(false);
    expect(t.recordFailure('/p')).toBe(false);
  });

  it('reset re-opens every exhausted path', () => {
    const t = new RefinementFailureTracker(1);
    t.recordFailure('/a');
    t.recordFailure('/b');

    expect(t.reset()).toBe(true);
    expect(t.isExhausted('/a')).toBe(false);
    expect(t.isExhausted('/b')).toBe(false);
    expect(t.reset()).toBe(false);
  });

  it('preserves toast state across retries and view changes until the path succeeds', () => {
    const t = new RefinementFailureTracker(1);
    expect(t.recordFailure('/a')).toBe(true);
    expect(t.shouldNotify('/a')).toBe(true);
    expect(t.shouldNotify('/a')).toBe(false);

    t.reopen();
    expect(t.isExhausted('/a')).toBe(false);
    expect(t.recordFailure('/a')).toBe(true);
    expect(t.shouldNotify('/a')).toBe(false);
    expect(t.shouldNotify('/b')).toBe(true);

    expect(t.reset()).toBe(true);
    expect(t.shouldNotify('/a')).toBe(false);
    t.recordSuccess('/a');
    expect(t.shouldNotify('/a')).toBe(true);
  });

  it('tracks paths independently', () => {
    const t = new RefinementFailureTracker(2);
    t.recordFailure('/a');
    expect(t.recordFailure('/b')).toBe(false);
    expect(t.recordFailure('/a')).toBe(true);
    expect(t.isExhausted('/b')).toBe(false);
  });
});
