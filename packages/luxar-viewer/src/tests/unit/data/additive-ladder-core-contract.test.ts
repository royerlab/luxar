/**
 * The additive-ladder behaviours the performance programme first landed in
 * GSplats, pinned for ALL THREE progressive loaders now that they share
 * `AdditiveLadderCore`. Each case mirrors a GSplats test
 * (`gsplats-progressive-scrub.test.ts`, `gsplats-progressive-loader.test.ts`);
 * for Points and Lines every one of them was a behaviour change:
 *
 * - B6: rung k+1's spatial index is warmed at refinement priority while rung k loads;
 * - a pinned (`ladderDepth`) pass starts its rungs concurrently, and a braked
 *   pass cancels and releases the ones it will not commit (A20);
 * - the next-rung lookahead is speculative, linked to the scheduling update,
 *   and not scheduled by an already-aborted pass;
 * - B5: after a hidden-dim slice step, rung 0 of the neighbouring slices is read ahead;
 * - the progressive loader answers `prefetchChunks` / `prefetchChunkBoundary`
 *   (it used to lack them, so `dispatchPredictivePrefetch` silently no-op'd for
 *   every laddered node).
 */

import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';

import { GSplatsProgressiveLoader } from '../../../data/gsplats/gsplats-progressive-loader';
import type { GSplatsSpatialIndexLoader } from '../../../data/gsplats/gsplats-spatial-index-loader';
import { LinesProgressiveLoader } from '../../../data/lines/lines-progressive-loader';
import type { LinesSpatialIndexLoader } from '../../../data/lines/lines-spatial-index-loader';
import { PointsProgressiveLoader } from '../../../data/points/points-progressive-loader';
import type { PointsSpatialIndexLoader } from '../../../data/points/points-spatial-index-loader';
import type { ViewState } from '../../../data/data-loader-types';
import type { LoadedGSplatsData } from '../../../types/gsplats';
import type { LoadedLinesData } from '../../../types/lines';
import type { LoadedPointsData } from '../../../types/points';
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

function viewAt(t: number, extra: Partial<ViewState> = {}): ViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, t],
    tolerance: [0, 0, 0, 0],
    dimensions: DIMENSIONS,
    ...extra,
  };
}

function pointsData(n: number): LoadedPointsData {
  return {
    positions: new Float32Array(n * 3),
    pointCount: n,
    ndim: 4,
    metadata: { totalPoints: n, loadedPoints: n, bounds: new THREE.Box3(), usedSpatialIndex: true },
  };
}

function linesData(n: number): LoadedLinesData {
  return {
    positions: new Float32Array(n * 2 * 4),
    segments: Uint32Array.from({ length: n * 2 }, (_, i) => i),
    widths: new Float32Array(n * 2).fill(1),
    colors: null,
    sharpness: null,
    segmentCount: n,
    vertexCount: n * 2,
    ndim: 4,
  };
}

function gsplatsData(n: number): LoadedGSplatsData {
  return {
    positions: new Float32Array(n * 4),
    amplitudes: new Float32Array(n),
    choleskyFactors: new Float32Array(n * 10),
    colors: null,
    splatCount: n,
    ndim: 4,
  } as LoadedGSplatsData;
}

type Data = LoadedPointsData | LoadedLinesData | LoadedGSplatsData;

/** A rung whose loads resolve at once (resident unless `cold`). */
function rung(make: (n: number) => Data, n: number) {
  return {
    updateViewWithResidency: vi.fn(async (..._args: unknown[]) => ({
      data: make(n),
      allResident: true,
    })),
    updateView: vi.fn(),
    prefetchChunks: vi.fn().mockResolvedValue(undefined),
    prefetchChunkBoundary: vi.fn().mockResolvedValue(undefined),
    planPrefetch: vi.fn().mockResolvedValue({ bytes: 0, ranges: [] }),
    getPrefetchCacheStats: vi.fn(() => null),
    ensureInitialized: vi.fn().mockResolvedValue(undefined),
    releaseAccumulator: vi.fn(),
    dispose: vi.fn(),
    getMetrics: vi.fn(() => ({})),
    getActiveQueries: vi.fn(() => []),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  };
}
type Rung = ReturnType<typeof rung>;

interface Ladder {
  updateView(
    view: ViewState,
    session?: undefined,
    signal?: AbortSignal,
    allowance?: number
  ): Promise<Data>;
  prefetchChunks(view: ViewState, signal?: AbortSignal): Promise<void>;
  prefetchChunkBoundary(
    current: ViewState,
    predicted: ViewState,
    signal?: AbortSignal
  ): Promise<void>;
  ladderResidency(): { loadedRungs: number; elementCount: number; residentBytes: number };
  dispose(): void;
}

const GEOMETRIES: Array<{
  name: string;
  make: (n: number) => Data;
  build: (rungs: Rung[]) => Ladder;
}> = [
  {
    name: 'Points',
    make: pointsData,
    build: (rungs) =>
      new PointsProgressiveLoader(
        rungs as unknown as PointsSpatialIndexLoader[],
        rungs.length,
        '/p'
      ) as unknown as Ladder,
  },
  {
    name: 'Lines',
    make: linesData,
    build: (rungs) =>
      new LinesProgressiveLoader(
        rungs as unknown as LinesSpatialIndexLoader[],
        rungs.length,
        '/l'
      ) as unknown as Ladder,
  },
  {
    name: 'GSplats',
    make: gsplatsData,
    build: (rungs) =>
      new GSplatsProgressiveLoader(
        rungs as unknown as GSplatsSpatialIndexLoader[],
        rungs.length,
        '/g'
      ) as unknown as Ladder,
  },
];

/** Rungs whose loads come back COLD, so a refine pass stops after `stopAfter`. */
function coldRungs(make: (n: number) => Data, count: number, coldUpTo: number): Rung[] {
  const rungs = Array.from({ length: count }, (_, i) => rung(make, i + 1));
  for (let level = 0; level <= coldUpTo; level++) {
    rungs[level].updateViewWithResidency.mockImplementation(async () => ({
      data: make(level + 1),
      allResident: false,
    }));
  }
  return rungs;
}

describe.each(GEOMETRIES)('$name progressive loader — shared ladder engine', ({ make, build }) => {
  it("B6: warms rung k+1's index at refinement priority while rung k loads", async () => {
    const rungs = Array.from({ length: 3 }, (_, i) => rung(make, i + 1));
    let warmedWhileRung0Loaded = false;
    rungs[0].updateViewWithResidency.mockImplementationOnce(async () => {
      warmedWhileRung0Loaded = rungs[1].ensureInitialized.mock.calls.some(
        (call) => call[0] === 'refinement'
      );
      return { data: make(1), allResident: true };
    });
    await build(rungs).updateView(viewAt(0));
    expect(warmedWhileRung0Loaded).toBe(true);
  });

  it('starts a pinned pass’s rungs concurrently, coarsest first', async () => {
    const rungs = Array.from({ length: 3 }, (_, i) => rung(make, i + 1));
    const started: number[] = [];
    let returnedWhenLastStarted = -1;
    let returned = 0;
    rungs.forEach((r, level) =>
      r.updateViewWithResidency.mockImplementationOnce(async () => {
        started.push(level);
        if (started.length === 3) returnedWhenLastStarted = returned;
        await new Promise<void>((resolve) => setTimeout(resolve, 5 * (3 - level)));
        returned++;
        return { data: make(level + 1), allResident: true };
      })
    );
    await build(rungs).updateView(viewAt(0, { ladderDepth: 3 }));
    expect(started).toEqual([0, 1, 2]);
    expect(returnedWhenLastStarted).toBe(0);
  });

  it('a braked pinned pass aborts and releases the rungs it will never commit', async () => {
    const rungs = Array.from({ length: 3 }, (_, i) => rung(make, i + 1));
    const signals: Array<AbortSignal | undefined> = [];
    const releases: Array<() => void> = [];
    rungs.forEach((r, level) =>
      r.updateViewWithResidency.mockImplementation(async (...args: unknown[]) => {
        signals[level] = args[2] as AbortSignal | undefined;
        if (level > 0) await new Promise<void>((resolve) => releases.push(resolve));
        return { data: make(level + 1), allResident: true };
      })
    );
    await build(rungs).updateView(viewAt(0, { ladderDepth: 3 }), undefined, undefined, 1);
    expect(signals[1]?.aborted).toBe(true);
    expect(signals[2]?.aborted).toBe(true);
    expect(rungs[1].releaseAccumulator).toHaveBeenCalled();
    expect(rungs[2].releaseAccumulator).toHaveBeenCalled();
    releases.forEach((release) => release());
  });

  it('the next-rung lookahead is speculative and aborts with the scheduling update', async () => {
    const rungs = coldRungs(make, 3, 1);
    const pass = new AbortController();
    await build(rungs).updateView(viewAt(0), undefined, pass.signal);
    await vi.waitFor(() => expect(rungs[2].prefetchChunks).toHaveBeenCalled());
    const lookahead = rungs[2].prefetchChunks.mock.calls[0][1] as AbortSignal;
    expect(signalPriority(lookahead)?.value).toBe('speculative');
    pass.abort();
    expect(lookahead.aborted).toBe(true);
  });

  it('a pass aborted by the time its rungs land schedules no lookahead and no read-ahead', async () => {
    const rungs = coldRungs(make, 3, 1);
    const loader = build(rungs);
    await loader.updateView(viewAt(4));
    rungs[0].prefetchChunks.mockClear();
    rungs[2].prefetchChunks.mockClear();
    const pass = new AbortController();
    rungs[1].updateViewWithResidency.mockImplementationOnce(async () => {
      pass.abort();
      return { data: make(2), allResident: false };
    });
    await loader.updateView(viewAt(5), undefined, pass.signal).catch(() => undefined);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(rungs[2].prefetchChunks).not.toHaveBeenCalled();
    expect(rungs[0].prefetchChunks).not.toHaveBeenCalled();
  });

  it('B5: reads ahead rung 0 of the neighbouring slices after a slice step', async () => {
    const rungs = Array.from({ length: 2 }, (_, i) => rung(make, i + 1));
    const loader = build(rungs);
    await loader.updateView(viewAt(4));
    rungs[0].prefetchChunks.mockClear();
    await loader.updateView(viewAt(5));
    await vi.waitFor(() => expect(rungs[0].prefetchChunks).toHaveBeenCalledTimes(2));
    const slices = rungs[0].prefetchChunks.mock.calls
      .map((call) => (call[0] as ViewState).slicePosition[3])
      .sort();
    expect(slices).toEqual([4, 6]);
    const signal = rungs[0].prefetchChunks.mock.calls[0][1] as AbortSignal;
    expect(signalPriority(signal)?.value).toBe('speculative');
  });

  it.fails('a rung landing after dispose() is dropped, not appended to the dead ladder', async () => {
    const rungs = Array.from({ length: 2 }, (_, i) => rung(make, i + 1));
    let land!: () => void;
    rungs[0].updateViewWithResidency.mockImplementationOnce(async () => {
      await new Promise<void>((resolve) => (land = resolve));
      return { data: make(1), allResident: true };
    });
    const loader = build(rungs);
    const pass = loader.updateView(viewAt(0));
    await vi.waitFor(() => expect(land).toBeDefined());
    loader.dispose();
    land();
    await pass.catch(() => undefined);
    const residency = loader.ladderResidency();
    expect(residency.loadedRungs).toBe(0);
    expect(residency.elementCount).toBe(0);
    expect(residency.residentBytes).toBe(0);
  });

  it('answers the predicted-view prefetch by warming its coarse rung', async () => {
    const rungs = Array.from({ length: 2 }, (_, i) => rung(make, i + 1));
    const loader = build(rungs);
    const signal = new AbortController().signal;
    await loader.prefetchChunkBoundary(viewAt(1), viewAt(2), signal);
    expect(rungs[0].prefetchChunkBoundary).toHaveBeenCalledWith(viewAt(1), viewAt(2), signal);
    await loader.prefetchChunks(viewAt(3), signal);
    expect(rungs[0].prefetchChunks).toHaveBeenCalledWith(viewAt(3), signal);
    expect(rungs[1].prefetchChunks).not.toHaveBeenCalled();
    loader.dispose();
    await loader.prefetchChunks(viewAt(4), signal);
    expect(rungs[0].prefetchChunks).toHaveBeenCalledTimes(1);
  });
});
