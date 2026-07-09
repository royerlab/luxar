/**
 * Unit tests for data/lines/lines-progressive-loader.ts.
 *
 * Mirrors `gsplats-progressive-loader.test.ts` and
 * `points-progressive-loader.test.ts`. Pure orchestration: the composite
 * loader wraps N LinesSpatialIndexLoader sub-loaders and decides how
 * many LODs to load per frame based on cache-hit timing. Tests stub
 * the sub-loaders and inspect the merged result.
 *
 * The line-specific wrinkle (vs. points/gsplats): `segments` indices are
 * LOCAL per LOD; on concatenation the loader rebases them by the
 * cumulative vertex count. The "segment-index offset" test pins that
 * contract.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { LinesProgressiveLoader } from '../../../data/lines/lines-progressive-loader';
import type { LinesSpatialIndexLoader } from '../../../data/lines/lines-spatial-index-loader';
import type { LinesViewState, LoadedLinesData } from '../../../types/lines';
import { CACHE_HIT_THRESHOLD_MS } from '../../../data/loaders/progressive/constants';
import { SliceCache } from '../../../cache/slice-cache';
import { buildSliceViewSig } from '../../../data/loaders/progressive/slice-cache-helper';

interface SubLoaderStub {
  updateView: ReturnType<typeof vi.fn>;
  updateViewWithResidency: ReturnType<typeof vi.fn>;
  prefetchChunks: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn>;
  getMetrics: ReturnType<typeof vi.fn>;
  getActiveQueries: ReturnType<typeof vi.fn>;
  addEventListener: ReturnType<typeof vi.fn>;
  removeEventListener: ReturnType<typeof vi.fn>;
}

/** Minimal LoaderMetrics stub with the fields the aggregator reads. */
function stubMetrics(over: Partial<Record<string, number>> = {}) {
  return {
    type: 'lines-spatial-index' as const,
    path: '/lines/additive_x',
    queries: 0,
    loads: 0,
    evictions: 0,
    errors: 0,
    pointsLoaded: 0,
    bytesLoaded: 0,
    visiblePoints: 0,
    avgQueryTime: 0,
    avgLoadTime: 0,
    memoryUsed: 0,
    memoryLimit: 0,
    ...over,
  };
}

function makeLodData(
  vertexCount: number,
  segmentCount: number,
  ndim = 3,
  options: { color?: 'none' | 'uint8' | 'uint16' | 'float32'; sharpness?: boolean } = {}
): LoadedLinesData {
  const positions = new Float32Array(vertexCount * ndim);
  positions.fill(0.5);
  // Segments: local indices [i, i+1] for segment i (clamped if too short).
  const segments = new Uint32Array(segmentCount * 2);
  for (let i = 0; i < segmentCount; i++) {
    segments[i * 2] = Math.min(i * 2, vertexCount - 2);
    segments[i * 2 + 1] = Math.min(i * 2 + 1, vertexCount - 1);
  }
  const widths = new Float32Array(vertexCount).fill(1);
  let colors: Float32Array | Uint8Array | Uint16Array | null = null;
  if (options.color === 'uint8') {
    colors = new Uint8Array(vertexCount * 3).fill(128);
  } else if (options.color === 'uint16') {
    colors = new Uint16Array(vertexCount * 3).fill(32000);
  } else if (options.color === 'float32') {
    colors = new Float32Array(vertexCount * 3).fill(0.5);
  }
  const sharpness = options.sharpness ? new Float32Array(vertexCount).fill(1) : null;
  return {
    positions,
    segments,
    widths,
    colors,
    sharpness,
    segmentCount,
    vertexCount,
    ndim,
  };
}

function makeSubLoader(
  initialData: LoadedLinesData,
  metrics: Record<string, number> = {}
): SubLoaderStub {
  const updateView = vi.fn().mockResolvedValue(initialData);
  // Progressive loader calls updateViewWithResidency; delegate to updateView
  // (default allResident=true) so existing assertions / timing tests hold.
  const updateViewWithResidency = vi.fn(async (vs: LinesViewState, s?: unknown) => ({
    data: await updateView(vs, s),
    allResident: true,
  }));
  return {
    updateView,
    updateViewWithResidency,
    prefetchChunks: vi.fn().mockResolvedValue(undefined),
    dispose: vi.fn(),
    getMetrics: vi.fn(() => stubMetrics(metrics)),
    getActiveQueries: vi.fn(() => []),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  };
}

const baseViewState: LinesViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0],
  tolerance: [0, 0, 0],
};

describe('LinesProgressiveLoader', () => {
  let lodA: SubLoaderStub;
  let lodB: SubLoaderStub;
  let lodC: SubLoaderStub;
  let loader: LinesProgressiveLoader;

  beforeEach(() => {
    vi.useFakeTimers();
    // 20 verts × 10 segs, 10 × 5, 4 × 2 — segment counts shrink with LOD.
    lodA = makeSubLoader(makeLodData(20, 10, 3, { color: 'uint8' }));
    lodB = makeSubLoader(makeLodData(10, 5, 3, { color: 'uint8' }));
    lodC = makeSubLoader(makeLodData(4, 2, 3, { color: 'uint8' }));
    loader = new LinesProgressiveLoader(
      [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
      3,
      '/lines'
    );
  });

  describe('SliceCache integration', () => {
    const viewA: LinesViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [0, 0, 0],
    };
    const viewB: LinesViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 1],
      tolerance: [0, 0, 0],
    };

    it('restores a revisited view from the SliceCache without re-streaming sub-LODs', async () => {
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const a = makeSubLoader(makeLodData(20, 10, 3, { color: 'uint8' }));
      const b = makeSubLoader(makeLodData(10, 5, 3, { color: 'uint8' }));
      const l = new LinesProgressiveLoader(
        [a, b] as unknown as LinesSpatialIndexLoader[],
        2,
        '/l',
        undefined,
        sc
      );
      await l.loadLines(viewA);
      await l.loadLines(viewB);
      a.updateViewWithResidency.mockClear();
      b.updateViewWithResidency.mockClear();
      const restored = await l.loadLines(viewA);
      expect(a.updateViewWithResidency).not.toHaveBeenCalled();
      expect(b.updateViewWithResidency).not.toHaveBeenCalled();
      expect(l.hasMoreLODs).toBe(false);
      expect(restored.vertexCount).toBe(30);
      expect(sc.getStats().hits).toBeGreaterThanOrEqual(1);
    });

    it('clones on store so a later accumulator overwrite cannot corrupt a cached slice', async () => {
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const dataA = makeLodData(20, 10, 3, { color: 'uint8' });
      const a = makeSubLoader(dataA);
      const b = makeSubLoader(makeLodData(10, 5, 3, { color: 'uint8' }));
      const l = new LinesProgressiveLoader(
        [a, b] as unknown as LinesSpatialIndexLoader[],
        2,
        '/l',
        undefined,
        sc
      );
      await l.loadLines(viewA);
      dataA.positions.fill(999);
      await l.loadLines(viewB);
      const restored = await l.loadLines(viewA);
      expect(restored.positions[0]).toBeCloseTo(0.5);
    });
  });

  describe('initial load', () => {
    it('loads LOD 0 on first call', async () => {
      const result = await loader.loadLines(baseViewState);
      expect(lodA.updateView).toHaveBeenCalled();
      expect(result.segmentCount).toBeGreaterThanOrEqual(10);
    });

    it('returns concatenated data from all loaded LODs', async () => {
      const result = await loader.loadLines(baseViewState);
      // All cache-hit fast → all 3 loaded → 10+5+2 = 17 segments, 20+10+4 = 34 verts.
      expect(result.segmentCount).toBe(17);
      expect(result.vertexCount).toBe(34);
    });

    it('exposes totalLODCount and loadedLODCount', async () => {
      expect(loader.totalLODCount).toBe(3);
      expect(loader.loadedLODCount).toBe(0);
      await loader.loadLines(baseViewState);
      expect(loader.loadedLODCount).toBe(3);
    });

    it('hasMoreLODs is false after all LODs loaded', async () => {
      expect(loader.hasMoreLODs).toBe(true);
      await loader.loadLines(baseViewState);
      expect(loader.hasMoreLODs).toBe(false);
    });
  });

  describe('view-state change handling', () => {
    it('resets loaded LODs when displayDims changes', async () => {
      await loader.loadLines(baseViewState);
      expect(loader.loadedLODCount).toBe(3);

      lodA.updateView.mockClear();
      await loader.loadLines({ ...baseViewState, displayDims: [1, 2, 3] });
      expect(lodA.updateView).toHaveBeenCalled();
    });

    it('resets when slicePosition changes', async () => {
      await loader.loadLines(baseViewState);
      lodA.updateView.mockClear();
      await loader.loadLines({ ...baseViewState, slicePosition: [1, 1, 1] });
      expect(lodA.updateView).toHaveBeenCalled();
    });

    it('resets when tolerance changes', async () => {
      await loader.loadLines(baseViewState);
      lodA.updateView.mockClear();
      await loader.loadLines({ ...baseViewState, tolerance: [0.1, 0.1, 0.1] });
      expect(lodA.updateView).toHaveBeenCalled();
    });

    it('does NOT reload when view state is unchanged + all LODs loaded', async () => {
      await loader.loadLines(baseViewState);
      lodA.updateView.mockClear();
      lodB.updateView.mockClear();
      lodC.updateView.mockClear();

      await loader.loadLines(baseViewState);
      expect(lodA.updateView).not.toHaveBeenCalled();
      expect(lodB.updateView).not.toHaveBeenCalled();
      expect(lodC.updateView).not.toHaveBeenCalled();
    });

    it('treats different dimensions metadata as a state change', async () => {
      const stateA: LinesViewState = {
        ...baseViewState,
        dimensions: [{ name: 'x', unit: 'um', display: true, scale: 1 }],
      };
      const stateB: LinesViewState = {
        ...baseViewState,
        dimensions: [{ name: 'y', unit: 'um', display: true, scale: 1 }],
      };
      await loader.loadLines(stateA);
      lodA.updateView.mockClear();
      await loader.loadLines(stateB);
      expect(lodA.updateView).toHaveBeenCalled();
    });
  });

  describe('LOD 0 short-circuit', () => {
    it('stops after LOD 0 if it returns 0 segments', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(0, 0));
      await loader.loadLines(baseViewState);
      expect(lodA.updateView).toHaveBeenCalled();
      expect(lodB.updateView).not.toHaveBeenCalled();
      expect(lodC.updateView).not.toHaveBeenCalled();
    });
  });

  describe('cache-hit timing short-circuit', () => {
    it(`stops loading further LODs when one takes > ${CACHE_HIT_THRESHOLD_MS}ms`, async () => {
      let now = 0;
      const performanceNowSpy = vi.spyOn(performance, 'now');
      performanceNowSpy.mockImplementation(() => {
        now += 5;
        return now;
      });

      lodB.updateView.mockImplementation(async () => {
        now += CACHE_HIT_THRESHOLD_MS + 10; // simulated work, over threshold
        return makeLodData(10, 5);
      });

      await loader.loadLines(baseViewState);

      expect(lodA.updateView).toHaveBeenCalled();
      expect(lodB.updateView).toHaveBeenCalled();
      expect(lodC.updateView).not.toHaveBeenCalled();

      performanceNowSpy.mockRestore();
    });

    it('stops loading further LODs after a cache miss (fast but not resident)', async () => {
      // LOD B is fast (no timing break) but reports a cache miss → the loop
      // must still stop so the frame renders and refinement continues.
      lodB.updateViewWithResidency.mockImplementation(async () => ({
        data: makeLodData(10, 5, 3, { color: 'uint8' }),
        allResident: false,
      }));

      await loader.loadLines(baseViewState);

      expect(lodA.updateViewWithResidency).toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
    });

    it('keeps loading while levels are resident', async () => {
      // All resident + fast → the loop loads every level in one call.
      await loader.loadLines(baseViewState);
      expect(lodA.updateViewWithResidency).toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).toHaveBeenCalled();
    });
  });

  describe('playback frame budget (frameBudgetMs)', () => {
    // Mirrors gsplats-progressive-loader.test.ts (three-geometry symmetry).
    let now: number;
    let nowSpy: ReturnType<typeof vi.spyOn>;

    beforeEach(() => {
      now = 0;
      nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => now);
      // Every level "costs" 30ms of simulated work.
      for (const lod of [lodA, lodB, lodC]) {
        lod.updateViewWithResidency.mockImplementation(async () => {
          now += 30;
          return { data: makeLodData(10, 5, 3, { color: 'uint8' }), allResident: true };
        });
      }
    });

    afterEach(() => {
      nowSpy.mockRestore();
    });

    it('stops streaming when the budget runs out and reports hasMoreLODs=false', async () => {
      await loader.updateView({ ...baseViewState, frameBudgetMs: 10 });

      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodB.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
      expect(loader.loadedLODCount).toBe(1);
      expect(loader.hasMoreLODs).toBe(false);
    });

    it('loads as many levels as fit the budget', async () => {
      await loader.updateView({ ...baseViewState, frameBudgetMs: 70 });

      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodB.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
      expect(loader.loadedLODCount).toBe(2);
    });

    it('always loads at least one level under a tiny budget (first-paint floor)', async () => {
      await loader.updateView({ ...baseViewState, frameBudgetMs: 0.001 });
      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(loader.loadedLODCount).toBe(1);
    });

    it('a budget-free call with the SAME view resumes from the prefix and completes', async () => {
      await loader.updateView({ ...baseViewState, frameBudgetMs: 10 });
      expect(loader.loadedLODCount).toBe(1);
      expect(loader.hasMoreLODs).toBe(false); // capped

      await loader.updateView(baseViewState);
      expect(loader.loadedLODCount).toBe(3);
      expect(loader.hasMoreLODs).toBe(false); // genuinely complete now
      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1); // no reset
    });

    it('a differing budget with an identical view does NOT reset the ladder', async () => {
      await loader.updateView({ ...baseViewState, frameBudgetMs: 10 });
      expect(loader.loadedLODCount).toBe(1);

      await loader.updateView({ ...baseViewState, frameBudgetMs: 70 });
      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(loader.loadedLODCount).toBeGreaterThanOrEqual(2);
    });

    it('playback prefix caching: capped ladders are stored, restored, and deepened loop-over-loop', async () => {
      // Mirrors gsplats-progressive-loader.test.ts (three-geometry symmetry).
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const l = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/l',
        undefined,
        sc
      );
      const viewA = baseViewState;
      const viewB = { ...baseViewState, slicePosition: [0, 0, 1] };

      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      expect(l.loadedLODCount).toBe(1);
      expect(sc.getStats().count).toBe(1);

      await l.updateView({ ...viewB, frameBudgetMs: 10 });

      lodA.updateViewWithResidency.mockClear();
      lodB.updateViewWithResidency.mockClear();
      lodC.updateViewWithResidency.mockClear();
      await l.updateView({ ...viewA, frameBudgetMs: 20 });
      expect(lodA.updateViewWithResidency).not.toHaveBeenCalled(); // from cache
      expect(lodB.updateViewWithResidency).toHaveBeenCalledTimes(1); // deepened
      expect(l.loadedLODCount).toBe(2);

      await l.updateView({ ...viewB, frameBudgetMs: 10 });
      lodA.updateViewWithResidency.mockClear();
      lodB.updateViewWithResidency.mockClear();
      lodC.updateViewWithResidency.mockClear();
      await l.updateView(viewA);
      expect(lodA.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).toHaveBeenCalledTimes(1); // only the tail
      expect(l.loadedLODCount).toBe(3);
      expect(l.hasMoreLODs).toBe(false);

      await l.updateView({ ...viewB, frameBudgetMs: 10 });
      lodA.updateViewWithResidency.mockClear();
      lodB.updateViewWithResidency.mockClear();
      lodC.updateViewWithResidency.mockClear();
      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      expect(lodA.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
      expect(l.loadedLODCount).toBe(3);
    });

    it('partial restore copies the container: resume never mutates the cached payload', async () => {
      // Mirrors gsplats-progressive-loader.test.ts (three-geometry symmetry).
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const l = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/l',
        undefined,
        sc
      );
      const viewA = baseViewState;
      const viewB = { ...baseViewState, slicePosition: [0, 0, 1] };

      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      await l.updateView({ ...viewB, frameBudgetMs: 10 });

      const key = SliceCache.makeKey('/l', buildSliceViewSig(viewA));
      const cachedPayload = sc.peek(key)!.payload as unknown[];
      expect(cachedPayload.length).toBe(1);

      await l.updateView({ ...viewA, frameBudgetMs: 20 });
      expect(cachedPayload.length).toBe(1);
    });
  });

  describe('prefetch scheduling', () => {
    it('fires prefetchChunks on the next unloaded LOD after a partial load', async () => {
      let now = 0;
      const spy = vi.spyOn(performance, 'now');
      spy.mockImplementation(() => {
        now += 5;
        return now;
      });
      lodB.updateView.mockImplementation(async () => {
        now += 25;
        return makeLodData(10, 5);
      });

      await loader.loadLines(baseViewState);
      await Promise.resolve();

      expect(lodC.prefetchChunks).toHaveBeenCalledWith(baseViewState);
      spy.mockRestore();
    });

    it('does NOT prefetch if all LODs already loaded', async () => {
      await loader.loadLines(baseViewState);
      expect(lodA.prefetchChunks).not.toHaveBeenCalled();
      expect(lodB.prefetchChunks).not.toHaveBeenCalled();
      expect(lodC.prefetchChunks).not.toHaveBeenCalled();
    });

    it('swallows prefetch failures (fire and forget)', async () => {
      let now = 0;
      const spy = vi.spyOn(performance, 'now');
      spy.mockImplementation(() => {
        now += 5;
        return now;
      });
      lodB.updateView.mockImplementation(async () => {
        now += 25;
        return makeLodData(10, 5);
      });
      lodC.prefetchChunks.mockRejectedValue(new Error('Network down'));

      await loader.loadLines(baseViewState);
      await Promise.resolve();

      expect(lodC.prefetchChunks).toHaveBeenCalled();
      spy.mockRestore();
    });
  });

  describe('concatenation', () => {
    it('returns a single part as-is when only one LOD is loaded', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(0, 0));
      const result = await loader.loadLines(baseViewState);
      expect(result.segmentCount).toBe(0);
    });

    it('concatenates positions / segments / widths across multiple LODs', async () => {
      const result = await loader.loadLines(baseViewState);
      expect(result.positions.length).toBe(34 * 3);
      expect(result.widths.length).toBe(34);
      expect(result.segments.length).toBe(17 * 2);
    });

    it('offset-adjusts segment indices into the concatenated vertex array', async () => {
      // LOD A: 20 verts → segs reference local indices 0..19
      // LOD B: 10 verts → local 0..9, rebased to 20..29
      // LOD C: 4 verts → local 0..3, rebased to 30..33
      const result = await loader.loadLines(baseViewState);
      // LOD A segs occupy result.segments[0..19] (10 segs × 2 indices). All
      // indices must be < 20 (unchanged from local).
      for (let i = 0; i < 20; i++) {
        expect(result.segments[i]).toBeLessThan(20);
      }
      // LOD B starts at index 20 in result.segments (10 × 2 = 20). All
      // indices in [20, 30) must be >= 20 (rebased) and < 30.
      for (let i = 20; i < 30; i++) {
        expect(result.segments[i]).toBeGreaterThanOrEqual(20);
        expect(result.segments[i]).toBeLessThan(30);
      }
      // LOD C starts at index 30 in result.segments (4 indices = 2 segs ×
      // 2). All indices in [30, 34) must be >= 30 (rebased) and < 34.
      for (let i = 30; i < 34; i++) {
        expect(result.segments[i]).toBeGreaterThanOrEqual(30);
        expect(result.segments[i]).toBeLessThan(34);
      }
    });

    it('preserves Uint8Array color type across LODs', async () => {
      const result = await loader.loadLines(baseViewState);
      expect(result.colors).toBeInstanceOf(Uint8Array);
      expect(result.colors?.length).toBe(34 * 3);
    });

    it('uses Uint16Array fill (65535) when first LOD colors are uint16 but a level lacks them', async () => {
      lodA = makeSubLoader(makeLodData(20, 10, 3, { color: 'uint16' }));
      lodB = makeSubLoader(makeLodData(10, 5, 3)); // no colors
      lodC = makeSubLoader(makeLodData(4, 2, 3, { color: 'uint16' }));
      loader = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/lines'
      );
      const result = await loader.loadLines(baseViewState);
      expect(result.colors).toBeInstanceOf(Uint16Array);
      // LOD B is verts 20..29 → indices 60..89; should be filled with 65535.
      expect(result.colors?.[60]).toBe(65535);
    });

    it('uses Float32 fill (1.0) when first LOD colors are float32 but a level lacks them', async () => {
      lodA = makeSubLoader(makeLodData(20, 10, 3, { color: 'float32' }));
      lodB = makeSubLoader(makeLodData(10, 5, 3)); // no colors
      lodC = makeSubLoader(makeLodData(4, 2, 3, { color: 'float32' }));
      loader = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/lines'
      );
      const result = await loader.loadLines(baseViewState);
      expect(result.colors).toBeInstanceOf(Float32Array);
      expect(result.colors?.[60]).toBeCloseTo(1.0, 5);
    });

    it('uses Uint8 fill (255) when first LOD colors are uint8 but a level lacks them', async () => {
      lodB = makeSubLoader(makeLodData(10, 5, 3)); // no colors
      loader = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/lines'
      );
      const result = await loader.loadLines(baseViewState);
      expect(result.colors).toBeInstanceOf(Uint8Array);
      // LOD B verts 20..29 → color indices 60..89.
      expect(result.colors?.[60]).toBe(255);
    });

    it('keeps colors null when no LOD has colors', async () => {
      lodA = makeSubLoader(makeLodData(20, 10, 3));
      lodB = makeSubLoader(makeLodData(10, 5, 3));
      lodC = makeSubLoader(makeLodData(4, 2, 3));
      loader = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/lines'
      );
      const result = await loader.loadLines(baseViewState);
      expect(result.colors).toBeNull();
    });

    it('concatenates sharpness when any LOD carries it', async () => {
      lodA = makeSubLoader(makeLodData(20, 10, 3, { sharpness: true }));
      loader = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/lines'
      );
      const result = await loader.loadLines(baseViewState);
      expect(result.sharpness).toBeInstanceOf(Float32Array);
      expect(result.sharpness?.length).toBe(34);
    });
  });

  describe('dispose', () => {
    it('reports hasMoreLODs=false after dispose (stale refinement-loop guard)', () => {
      // Mirrors GSplatsProgressiveLoader: a disposed loader has loadedLODs
      // cleared but nLods kept, so without the guard hasMoreLODs flipped
      // BACK to true and a refinement loop holding a stale reference would
      // index into the emptied lodLoaders forever (TypeError every pass).
      expect(loader.hasMoreLODs).toBe(true);
      loader.dispose();
      expect(loader.hasMoreLODs).toBe(false);
    });

    it('disposes all sub-loaders', () => {
      loader.dispose();
      expect(lodA.dispose).toHaveBeenCalled();
      expect(lodB.dispose).toHaveBeenCalled();
      expect(lodC.dispose).toHaveBeenCalled();
    });

    it('clears internal state', async () => {
      await loader.loadLines(baseViewState);
      loader.dispose();
      expect(loader.loadedLODCount).toBe(0);
    });
  });

  describe('LoaderMonitor surface', () => {
    it('exposes the four monitor methods (so connectLoaderToMonitor wires it)', () => {
      expect(typeof loader.addEventListener).toBe('function');
      expect(typeof loader.removeEventListener).toBe('function');
      expect(typeof loader.getMetrics).toBe('function');
      expect(typeof loader.getActiveQueries).toBe('function');
    });

    it('getMetrics aggregates inner-loader metrics under the node path', () => {
      lodA = makeSubLoader(makeLodData(20, 10), { queries: 2, pointsLoaded: 100, memoryUsed: 10 });
      lodB = makeSubLoader(makeLodData(10, 5), { queries: 3, pointsLoaded: 50, memoryUsed: 20 });
      loader = new LinesProgressiveLoader(
        [lodA, lodB] as unknown as LinesSpatialIndexLoader[],
        2,
        '/lines'
      );

      const metrics = loader.getMetrics();
      expect(metrics.path).toBe('/lines');
      expect(metrics.type).toBe('lines-spatial-index');
      expect(metrics.queries).toBe(5); // 2 + 3
      expect(metrics.pointsLoaded).toBe(150); // 100 + 50
      expect(metrics.memoryUsed).toBe(30); // 10 + 20
    });

    it('addEventListener / removeEventListener fan out to every inner loader', () => {
      const listener = vi.fn();
      loader.addEventListener(listener);
      expect(lodA.addEventListener).toHaveBeenCalledTimes(1);
      expect(lodB.addEventListener).toHaveBeenCalledTimes(1);
      expect(lodC.addEventListener).toHaveBeenCalledTimes(1);

      loader.removeEventListener(listener);
      expect(lodA.removeEventListener).toHaveBeenCalledTimes(1);
      expect(lodC.removeEventListener).toHaveBeenCalledTimes(1);
    });

    it('getActiveQueries merges inner active-query lists', () => {
      lodA.getActiveQueries.mockReturnValue([{ id: 'a' }]);
      lodB.getActiveQueries.mockReturnValue([{ id: 'b' }, { id: 'c' }]);
      lodC.getActiveQueries.mockReturnValue([]);
      expect(loader.getActiveQueries()).toHaveLength(3);
    });
  });
});

describe('LinesProgressiveLoader — concat memoization (no-op commit skip)', () => {
  let lodA: SubLoaderStub;
  let lodB: SubLoaderStub;
  let loader: LinesProgressiveLoader;

  beforeEach(() => {
    vi.useFakeTimers();
    lodA = makeSubLoader(makeLodData(20, 10));
    lodB = makeSubLoader(makeLodData(10, 5));
    loader = new LinesProgressiveLoader(
      [lodA, lodB] as unknown as LinesSpatialIndexLoader[],
      2,
      '/lines'
    );
  });

  it('returns the IDENTICAL reference for a repeat call with unchanged view state', async () => {
    const first = await loader.updateView(baseViewState);
    const second = await loader.updateView(baseViewState);
    // Same reference — the commit pipeline uses this identity to skip
    // no-op re-commits (mesh.userData.committedData === data).
    expect(second).toBe(first);
  });

  it('returns a NEW reference after a view-state change back to the same LOD count (resetGeneration)', async () => {
    const first = await loader.updateView(baseViewState);
    const away = await loader.updateView({ ...baseViewState, slicePosition: [1, 1, 1] });
    expect(away).not.toBe(first);
    const back = await loader.updateView(baseViewState);
    expect(back).not.toBe(first);
    const backAgain = await loader.updateView(baseViewState);
    expect(backAgain).toBe(back);
  });
});

describe('LinesProgressiveLoader — committedEnergyFraction (quality stamps)', () => {
  function make(table?: Array<number | null>) {
    const lodA = makeSubLoader(makeLodData(100, 50));
    const lodB = makeSubLoader(makeLodData(50, 25));
    return new LinesProgressiveLoader(
      [lodA, lodB] as unknown as LinesSpatialIndexLoader[],
      2,
      '/lines',
      table
    );
  }

  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('reports 0 before any LOD loads and the table e(k) as the ladder streams', async () => {
    const loader = make([0.7, 1.0]);
    expect(loader.committedEnergyFraction).toBe(0);
    await loader.updateView(baseViewState);
    // Sync mocks are cache-hit fast, so both LODs load in the first pass.
    expect(loader.committedEnergyFraction).toBe([0.7, 1.0][loader.loadedLODCount - 1]);
  });

  it('reads as unstamped (null) without a table, with a wrong-length table, or with any missing entry', () => {
    expect(make().committedEnergyFraction).toBeNull();
    expect(make([0.7]).committedEnergyFraction).toBeNull();
    expect(make([0.7, null]).committedEnergyFraction).toBeNull();
  });
});
