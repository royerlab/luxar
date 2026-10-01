/**
 * B5: scrubbing behaviour of the progressive gsplats loader.
 *
 * - (a) a PINNED pass (`ladderDepth`, playback) fetches its rungs concurrently,
 *   started coarsest first, and still commits them in ladder order;
 * - (b) after a hidden-dimension slice change, the coarse rung (rung 0) of the
 *   neighbouring slices (±1 step along the scrubbed axis) is read ahead.
 */

import { describe, expect, it, vi } from 'vitest';

import { GSplatsProgressiveLoader } from '../../../data/gsplats/gsplats-progressive-loader';
import type { GSplatsSpatialIndexLoader } from '../../../data/gsplats/gsplats-spatial-index-loader';
import type { GSplatsViewState, LoadedGSplatsData } from '../../../types/gsplats';
import { signalPriority } from '../../../utils/fetch-concurrency';

const DIMENSIONS = [
  { name: 'X', unit: 'um', scale: 1 },
  { name: 'Y', unit: 'um', scale: 1 },
  { name: 'Z', unit: 'um', scale: 1 },
  {
    name: 'Time',
    unit: 'frame',
    scale: 1,
    discrete: true,
    step: 1,
    range: [0, 9] as [number, number],
  },
];

function viewAt(t: number, extra: Partial<GSplatsViewState> = {}): GSplatsViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, t],
    tolerance: [0, 0, 0, 0],
    dimensions: DIMENSIONS,
    ...extra,
  };
}

function lodData(splatCount: number): LoadedGSplatsData {
  return {
    positions: new Float32Array(splatCount * 4),
    amplitudes: new Float32Array(splatCount),
    choleskyFactors: new Float32Array(splatCount * 10),
    colors: null,
    splatCount,
    ndim: 4,
  } as LoadedGSplatsData;
}

/** A sub-loader whose level loads resolve when `release()` is called (or at once). */
function subLoader(splatCount: number, gated: boolean) {
  const releases: Array<() => void> = [];
  const stub = {
    ensureInitialized: vi.fn().mockResolvedValue(undefined),
    updateViewWithResidency: vi.fn(async (..._args: unknown[]) => {
      if (gated) await new Promise<void>((resolve) => releases.push(resolve));
      return { data: lodData(splatCount), allResident: true };
    }),
    updateView: vi.fn(),
    prefetchChunks: vi.fn().mockResolvedValue(undefined),
    planPrefetch: vi.fn().mockResolvedValue({ bytes: 0, ranges: [] }),
    getPrefetchCacheStats: vi.fn(() => null),
    releaseAccumulator: vi.fn(),
    dispose: vi.fn(),
    getMetrics: vi.fn(() => ({})),
    getActiveQueries: vi.fn(() => []),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  };
  return { stub, release: () => releases.splice(0).forEach((resolve) => resolve()) };
}

function makeLoader(rungs: ReturnType<typeof subLoader>[]): GSplatsProgressiveLoader {
  return new GSplatsProgressiveLoader(
    rungs.map((r) => r.stub) as unknown as GSplatsSpatialIndexLoader[],
    rungs.length,
    '/splats'
  );
}

describe('GSplatsProgressiveLoader — scrubbing (B5)', () => {
  it('pinned rungs are fetched concurrently, coarsest first', async () => {
    const rungs = [subLoader(1, true), subLoader(2, true), subLoader(4, true)];
    const loader = makeLoader(rungs);
    const started: number[] = [];
    let returned = 0;
    let returnedWhenLastStarted = -1;
    rungs.forEach((r, level) =>
      r.stub.updateViewWithResidency.mockImplementationOnce(async () => {
        started.push(level);
        if (started.length === 3) returnedWhenLastStarted = returned;
        await new Promise<void>((resolve) => setTimeout(resolve, 5 * (3 - level)));
        returned++;
        return { data: lodData(2 ** level), allResident: true };
      })
    );

    const pass = loader.updateView(viewAt(0, { ladderDepth: 3 }));
    await vi.waitFor(() => expect(started).toHaveLength(3));
    // All three rungs in flight before any has returned, coarsest started first.
    expect(started).toEqual([0, 1, 2]);
    expect(returnedWhenLastStarted).toBe(0);

    const result = await pass;
    expect(result.splatCount).toBe(7);
    expect(loader.hasMoreLODs).toBe(false);
  });

  it('a progressive loader gets predictive read-ahead of its coarse rung (±1 slice)', async () => {
    const rungs = [subLoader(3, false), subLoader(3, false)];
    const loader = makeLoader(rungs);
    await loader.updateView(viewAt(4));
    rungs[0].stub.prefetchChunks.mockClear();

    await loader.updateView(viewAt(5));

    await vi.waitFor(() => expect(rungs[0].stub.prefetchChunks).toHaveBeenCalledTimes(2));
    const slices = rungs[0].stub.prefetchChunks.mock.calls
      .map((call) => (call[0] as GSplatsViewState).slicePosition[3])
      .sort();
    expect(slices).toEqual([4, 6]);
  });

  it('stepping onto a slice being read ahead keeps that read alive for the pass to join', async () => {
    // The warm is the ONLY waiter on its chunk fetch until the new pass's
    // level-0 read joins it, so aborting it at the start of that pass cancelled
    // the shared fetch and the demand read refetched: read-ahead helped only a
    // user who paused. Only the neighbour the view did NOT step onto is stale.
    const rungs = [subLoader(3, false), subLoader(3, false)];
    const loader = makeLoader(rungs);
    const warmSignals = new Map<number, AbortSignal>();
    rungs[0].stub.prefetchChunks.mockImplementation(
      (view: GSplatsViewState, signal: AbortSignal) => {
        warmSignals.set(view.slicePosition[3], signal);
        return new Promise<void>(() => undefined);
      }
    );
    await loader.updateView(viewAt(3));
    await loader.updateView(viewAt(4));
    await vi.waitFor(() => expect([...warmSignals.keys()].sort()).toEqual([3, 5]));

    let signalsAtDemandRead: { t3: boolean; t5: boolean } | null = null;
    rungs[0].stub.updateViewWithResidency.mockImplementationOnce(async () => {
      signalsAtDemandRead = {
        t3: warmSignals.get(3)!.aborted,
        t5: warmSignals.get(5)!.aborted,
      };
      return { data: lodData(3), allResident: true };
    });
    await loader.updateView(viewAt(5));

    expect(signalsAtDemandRead).toEqual({ t3: true, t5: false });
  });
});

describe('GSplatsProgressiveLoader — a braked pinned pass cancels its remaining rungs', () => {
  /** Rungs whose loads record their signal and park until released. */
  function parkedRungs(count: number) {
    const rungs = Array.from({ length: count }, (_, i) => subLoader(2 ** i, false));
    const signals: Array<AbortSignal | undefined> = [];
    const releases: Array<() => void> = [];
    rungs.forEach((r, level) =>
      r.stub.updateViewWithResidency.mockImplementation(async (...args: unknown[]) => {
        signals[level] = args[2] as AbortSignal | undefined;
        if (level > 0) await new Promise<void>((resolve) => releases.push(resolve));
        return { data: lodData(2 ** level), allResident: true };
      })
    );
    return { rungs, signals, release: () => releases.splice(0).forEach((r) => r()) };
  }

  it('the residency brake aborts and releases the rungs it will never commit', async () => {
    const { rungs, signals, release } = parkedRungs(3);
    const loader = makeLoader(rungs);
    // A 1-byte allowance: rung 0 spends it, so the loop stops right after it.
    const result = await loader.updateView(viewAt(0, { ladderDepth: 3 }), undefined, undefined, 1);
    expect(result.splatCount).toBe(1);

    // Rungs 1 and 2 were started up front; the brake must cancel their reads…
    expect(signals[1]?.aborted).toBe(true);
    expect(signals[2]?.aborted).toBe(true);
    // …and drop whatever their accumulators already decoded.
    expect(rungs[1].stub.releaseAccumulator).toHaveBeenCalled();
    expect(rungs[2].stub.releaseAccumulator).toHaveBeenCalled();
    release();
  });

  it('a committed rung keeps its read: only the discarded remainder is aborted', async () => {
    const { rungs, signals, release } = parkedRungs(3);
    const loader = makeLoader(rungs);
    const pass = new AbortController();
    const promise = loader.updateView(viewAt(0, { ladderDepth: 3 }), undefined, pass.signal, 1);
    await promise;
    expect(signals[0]?.aborted).toBe(false);
    expect(pass.signal.aborted).toBe(false);
    release();
  });

  it('a pass abort still reaches every started rung', async () => {
    const { rungs, signals } = parkedRungs(3);
    const loader = makeLoader(rungs);
    const pass = new AbortController();
    const promise = loader.updateView(viewAt(0, { ladderDepth: 3 }), undefined, pass.signal);
    await vi.waitFor(() => expect(signals).toHaveLength(3));
    pass.abort();
    expect(signals.every((s) => s?.aborted)).toBe(true);
    void promise.catch(() => {});
  });
});

describe('GSplatsProgressiveLoader — lookahead honours the update signal', () => {
  /** Three rungs; rungs 0 and 1 come back COLD, so a refine pass stops after rung 1. */
  function coldFirstRung() {
    const rungs = [subLoader(1, false), subLoader(2, false), subLoader(4, false)];
    for (const level of [0, 1]) {
      rungs[level].stub.updateViewWithResidency.mockImplementation(async () => ({
        data: lodData(2 ** level),
        allResident: false,
      }));
    }
    return rungs;
  }

  it('the next-rung lookahead is speculative and aborts with the update that scheduled it', async () => {
    const rungs = coldFirstRung();
    const loader = makeLoader(rungs);
    const pass = new AbortController();
    await loader.updateView(viewAt(0), undefined, pass.signal);

    await vi.waitFor(() => expect(rungs[2].stub.prefetchChunks).toHaveBeenCalled());
    const lookahead = rungs[2].stub.prefetchChunks.mock.calls[0][1] as AbortSignal;
    expect(signalPriority(lookahead)?.value).toBe('speculative');
    expect(lookahead.aborted).toBe(false);
    pass.abort();
    expect(lookahead.aborted).toBe(true);
  });

  it('a pass aborted by the time its rungs land schedules no lookahead and no read-ahead', async () => {
    const rungs = coldFirstRung();
    const loader = makeLoader(rungs);
    await loader.updateView(viewAt(4));
    rungs[0].stub.prefetchChunks.mockClear();
    rungs[2].stub.prefetchChunks.mockClear();

    const pass = new AbortController();
    rungs[1].stub.updateViewWithResidency.mockImplementationOnce(async () => {
      pass.abort(); // superseded while its last rung was in flight
      return { data: lodData(2), allResident: false };
    });
    await loader.updateView(viewAt(5), undefined, pass.signal).catch(() => undefined);
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(rungs[2].stub.prefetchChunks).not.toHaveBeenCalled();
    expect(rungs[0].stub.prefetchChunks).not.toHaveBeenCalled();
  });
});
