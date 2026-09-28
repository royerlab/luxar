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
    updateViewWithResidency: vi.fn(async () => {
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
});
