/**
 * Unit tests for data/gsplats/gsplats-progressive-loader.ts.
 *
 * Pure orchestration: the composite loader wraps N
 * GSplatsSpatialIndexLoader sub-loaders and decides how many LODs
 * to load per frame based on cache-hit timing. Tests stub the
 * sub-loaders and inspect the merged result.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { GSplatsProgressiveLoader } from '../../../data/gsplats/gsplats-progressive-loader';
import type { GSplatsSpatialIndexLoader } from '../../../data/gsplats/gsplats-spatial-index-loader';
import type { GSplatsViewState, LoadedGSplatsData } from '../../../types/gsplats';
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
    type: 'gsplats-spatial-index' as const,
    path: '/test_gsplats/additive_x',
    queries: 0,
    loads: 0,
    evictions: 0,
    errors: 0,
    elementsLoaded: 0,
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
  splatCount: number,
  ndim = 3,
  options: { color?: 'none' | 'uint8' | 'uint16' | 'float32' } = {}
): LoadedGSplatsData {
  const cholSize = (ndim * (ndim + 1)) / 2;
  const positions = new Float32Array(splatCount * ndim);
  positions.fill(0.5);
  const amplitudes = new Float32Array(splatCount);
  amplitudes.fill(1.0);
  const choleskyFactors = new Float32Array(splatCount * cholSize);
  choleskyFactors.fill(0.1);
  let colors: Float32Array | Uint8Array | Uint16Array | null = null;
  if (options.color === 'uint8') {
    colors = new Uint8Array(splatCount * 3).fill(128);
  } else if (options.color === 'uint16') {
    colors = new Uint16Array(splatCount * 3).fill(32000);
  } else if (options.color === 'float32') {
    colors = new Float32Array(splatCount * 3).fill(0.5);
  }
  return {
    positions,
    amplitudes,
    choleskyFactors,
    colors,
    splatCount,
    ndim,
  };
}

function makeSubLoader(
  initialData: LoadedGSplatsData,
  metrics: Record<string, number> = {}
): SubLoaderStub {
  const updateView = vi.fn().mockResolvedValue(initialData);
  // Progressive loader calls updateViewWithResidency; delegate to updateView
  // (default allResident=true) so existing assertions / timing tests hold.
  const updateViewWithResidency = vi.fn(async (vs: unknown, s?: unknown) => ({
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

const baseViewState: GSplatsViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 0],
};

describe('GSplatsProgressiveLoader', () => {
  let lodA: SubLoaderStub;
  let lodB: SubLoaderStub;
  let lodC: SubLoaderStub;
  let loader: GSplatsProgressiveLoader;

  beforeEach(() => {
    vi.useFakeTimers();
    lodA = makeSubLoader(makeLodData(100, 3, { color: 'uint8' }));
    lodB = makeSubLoader(makeLodData(50, 3, { color: 'uint8' }));
    lodC = makeSubLoader(makeLodData(25, 3, { color: 'uint8' }));
    loader = new GSplatsProgressiveLoader(
      [lodA, lodB, lodC] as unknown as GSplatsSpatialIndexLoader[],
      3,
      '/test_gsplats'
    );
  });

  describe('initial load', () => {
    it('loads LOD 0 on first call', async () => {
      const result = await loader.loadGSplats(baseViewState);
      expect(lodA.updateView).toHaveBeenCalled();
      expect(result.splatCount).toBeGreaterThanOrEqual(100);
    });

    it('returns concatenated data from all loaded LODs', async () => {
      const result = await loader.loadGSplats(baseViewState);
      // All 3 LODs are cache-hit fast (sync mock) → all loaded → 100+50+25 = 175.
      expect(result.splatCount).toBe(175);
    });

    it('exposes totalLODCount and loadedLODCount', async () => {
      expect(loader.totalLODCount).toBe(3);
      expect(loader.loadedLODCount).toBe(0);
      await loader.loadGSplats(baseViewState);
      expect(loader.loadedLODCount).toBe(3);
    });

    it('hasMoreLODs is false after all LODs loaded', async () => {
      expect(loader.hasMoreLODs).toBe(true);
      await loader.loadGSplats(baseViewState);
      expect(loader.hasMoreLODs).toBe(false);
    });

    it('hasMoreLODs is false after dispose (M6)', () => {
      // Fresh loader reports work to do; once disposed it must report none,
      // so a refinement loop holding a stale reference stops instead of
      // indexing into the cleared lodLoaders.
      expect(loader.hasMoreLODs).toBe(true);
      loader.dispose();
      expect(loader.hasMoreLODs).toBe(false);
    });
  });

  describe('SliceCache integration', () => {
    const viewA: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    };
    const viewB: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 1],
      tolerance: [0, 0, 0, 0],
    };

    it('restores a revisited view from the SliceCache without re-streaming sub-LODs', async () => {
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const a = makeSubLoader(makeLodData(100, 3, { color: 'uint8' }));
      const b = makeSubLoader(makeLodData(50, 3, { color: 'uint8' }));
      const l = new GSplatsProgressiveLoader(
        [a, b] as unknown as GSplatsSpatialIndexLoader[],
        2,
        '/g',
        undefined,
        sc
      );

      await l.loadGSplats(viewA); // full ladder for A → stored
      expect(l.loadedLODCount).toBe(2);
      await l.loadGSplats(viewB); // different view → loads B, stores B

      // Clear call history, then revisit A: it must be served entirely from the
      // SliceCache — no sub-loader load at all.
      a.updateViewWithResidency.mockClear();
      b.updateViewWithResidency.mockClear();
      const restored = await l.loadGSplats(viewA);

      expect(a.updateViewWithResidency).not.toHaveBeenCalled();
      expect(b.updateViewWithResidency).not.toHaveBeenCalled();
      expect(l.hasMoreLODs).toBe(false); // full ladder → refinement won't re-stream
      expect(l.loadedLODCount).toBe(2);
      expect(restored.splatCount).toBe(150);
      expect(sc.getStats().hits).toBeGreaterThanOrEqual(1);
    });

    it('clones on store so a later accumulator overwrite cannot corrupt a cached slice', async () => {
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const dataA = makeLodData(100, 3, { color: 'uint8' }); // positions filled 0.5
      const a = makeSubLoader(dataA);
      const b = makeSubLoader(makeLodData(50, 3, { color: 'uint8' }));
      const l = new GSplatsProgressiveLoader(
        [a, b] as unknown as GSplatsSpatialIndexLoader[],
        2,
        '/g',
        undefined,
        sc
      );

      await l.loadGSplats(viewA); // stores a CLONE of [dataA, dataB]
      // Simulate the spatial-index loader reusing its accumulator buffer for the
      // next load: overwrite dataA's decoded arrays in place.
      dataA.positions.fill(999);
      dataA.amplitudes.fill(999);

      await l.loadGSplats(viewB); // change view away (evicts loadedLODs)
      const restored = await l.loadGSplats(viewA); // restore A from the cache

      // The cached clone must retain the original values, not the 999 overwrite.
      expect(restored.positions[0]).toBeCloseTo(0.5);
      expect(restored.amplitudes[0]).toBeCloseTo(1.0);
    });

    it('is a no-op (no restore, always re-streams) when no SliceCache is supplied', async () => {
      const a = makeSubLoader(makeLodData(100, 3, { color: 'uint8' }));
      const b = makeSubLoader(makeLodData(50, 3, { color: 'uint8' }));
      const l = new GSplatsProgressiveLoader(
        [a, b] as unknown as GSplatsSpatialIndexLoader[],
        2,
        '/g'
        // no energyTable, no sliceCache
      );
      await l.loadGSplats(viewA);
      await l.loadGSplats(viewB);
      a.updateViewWithResidency.mockClear();
      await l.loadGSplats(viewA); // revisit → must re-stream (no cache)
      expect(a.updateViewWithResidency).toHaveBeenCalled();
    });
  });

  describe('view-state change handling', () => {
    it('resets loaded LODs when displayDims changes', async () => {
      await loader.loadGSplats(baseViewState);
      expect(loader.loadedLODCount).toBe(3);

      lodA.updateView.mockClear();
      await loader.loadGSplats({ ...baseViewState, displayDims: [1, 2, 3] });
      // New state → reset + reload from LOD 0.
      expect(lodA.updateView).toHaveBeenCalled();
    });

    it('resets when slicePosition changes', async () => {
      await loader.loadGSplats(baseViewState);
      lodA.updateView.mockClear();
      await loader.loadGSplats({ ...baseViewState, slicePosition: [1, 1, 1] });
      expect(lodA.updateView).toHaveBeenCalled();
    });

    it('resets when tolerance changes', async () => {
      await loader.loadGSplats(baseViewState);
      lodA.updateView.mockClear();
      await loader.loadGSplats({ ...baseViewState, tolerance: [0.1, 0.1, 0.1] });
      expect(lodA.updateView).toHaveBeenCalled();
    });

    it('does NOT reload when view state is unchanged + all LODs loaded', async () => {
      await loader.loadGSplats(baseViewState);
      lodA.updateView.mockClear();
      lodB.updateView.mockClear();
      lodC.updateView.mockClear();

      await loader.loadGSplats(baseViewState);
      // All 3 already loaded → no further updateView calls.
      expect(lodA.updateView).not.toHaveBeenCalled();
      expect(lodB.updateView).not.toHaveBeenCalled();
      expect(lodC.updateView).not.toHaveBeenCalled();
    });

    it('treats different dimensions metadata as a state change', async () => {
      const stateA: GSplatsViewState = {
        ...baseViewState,
        dimensions: [{ name: 'x', unit: 'um', display: true, scale: 1 }],
      };
      const stateB: GSplatsViewState = {
        ...baseViewState,
        dimensions: [{ name: 'y', unit: 'um', display: true, scale: 1 }],
      };
      await loader.loadGSplats(stateA);
      lodA.updateView.mockClear();
      await loader.loadGSplats(stateB);
      expect(lodA.updateView).toHaveBeenCalled();
    });
  });

  describe('LOD 0 short-circuit', () => {
    it('stops after LOD 0 if it returns 0 splats', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(0));
      await loader.loadGSplats(baseViewState);
      expect(lodA.updateView).toHaveBeenCalled();
      expect(lodB.updateView).not.toHaveBeenCalled();
      expect(lodC.updateView).not.toHaveBeenCalled();
    });
  });

  describe('cache-hit timing short-circuit', () => {
    it(`stops loading further LODs when one takes > ${CACHE_HIT_THRESHOLD_MS}ms`, async () => {
      // Make LOD B slow (over the threshold) so LOD C is deferred.
      let now = 0;
      const performanceNowSpy = vi.spyOn(performance, 'now');
      // sequence: t0_lodA, t1_lodA, t0_lodB, t1_lodB, t0_lodC, t1_lodC
      performanceNowSpy.mockImplementation(() => {
        now += 5;
        return now;
      });

      // LOD B's updateView "takes" longer than the threshold — bump the stride.
      lodB.updateView.mockImplementation(async () => {
        now += CACHE_HIT_THRESHOLD_MS + 10; // simulated work, over threshold
        return makeLodData(50);
      });

      await loader.loadGSplats(baseViewState);

      // LOD A loaded (always); LOD B loaded but slow → LOD C deferred.
      expect(lodA.updateView).toHaveBeenCalled();
      expect(lodB.updateView).toHaveBeenCalled();
      expect(lodC.updateView).not.toHaveBeenCalled();

      performanceNowSpy.mockRestore();
    });

    it('stops loading further LODs after a cache miss (fast but not resident)', async () => {
      // LOD B is fast (no timing break) but reports a cache miss → the loop
      // must still stop so the frame renders and refinement continues.
      lodB.updateViewWithResidency.mockImplementation(async () => ({
        data: makeLodData(50, 3, { color: 'uint8' }),
        allResident: false,
      }));

      await loader.loadGSplats(baseViewState);

      expect(lodA.updateViewWithResidency).toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
    });

    it('keeps loading while levels are resident', async () => {
      // All resident + fast → the loop loads every level in one call.
      await loader.loadGSplats(baseViewState);
      expect(lodA.updateViewWithResidency).toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).toHaveBeenCalled();
    });
  });

  describe('playback frame budget (frameBudgetMs)', () => {
    // Mirrored in points-progressive-loader.test.ts and
    // lines-progressive-loader.test.ts (three-geometry symmetry).
    let now: number;
    let nowSpy: ReturnType<typeof vi.spyOn>;

    beforeEach(() => {
      now = 0;
      nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => now);
      // Every level "costs" 30ms of simulated work.
      for (const lod of [lodA, lodB, lodC]) {
        lod.updateViewWithResidency.mockImplementation(async () => {
          now += 30;
          return { data: makeLodData(10, 3, { color: 'uint8' }), allResident: true };
        });
      }
    });

    afterEach(() => {
      nowSpy.mockRestore();
    });

    it('stops streaming when the budget runs out and reports hasMoreLODs=false', async () => {
      // Budget 10ms < one 30ms level: level 0 loads (first-paint floor),
      // level 1's loop-top check sees the budget spent → stop.
      await loader.updateView({ ...baseViewState, frameBudgetMs: 10 });

      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodB.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
      expect(loader.loadedLODCount).toBe(1);
      // Budget active → the prefix IS the target: no refinement scheduling.
      expect(loader.hasMoreLODs).toBe(false);
    });

    it('loads as many levels as fit the budget', async () => {
      // Budget 70ms fits two 30ms levels; the third's loop-top check fails.
      await loader.updateView({ ...baseViewState, frameBudgetMs: 70 });

      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodB.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
      expect(loader.loadedLODCount).toBe(2);
    });

    it('always loads at least one level under a tiny budget (first-paint floor)', async () => {
      // Advance the clock on EVERY performance.now() call so the deadline is
      // already past by level 0's loop-top check — without the
      // `level > startLevel` guard, level 0 would be skipped entirely
      // (pins the first-paint floor against guard removal).
      nowSpy.mockImplementation(() => (now += 5));
      await loader.updateView({ ...baseViewState, frameBudgetMs: 0.001 });
      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(loader.loadedLODCount).toBe(1);
    });

    it('a budget-free call with the SAME view resumes from the prefix and completes', async () => {
      await loader.updateView({ ...baseViewState, frameBudgetMs: 10 });
      expect(loader.loadedLODCount).toBe(1);
      expect(loader.hasMoreLODs).toBe(false); // capped

      // Pause re-trigger: same view, no budget → resume from level 1.
      await loader.updateView(baseViewState);
      expect(loader.loadedLODCount).toBe(3);
      expect(loader.hasMoreLODs).toBe(false); // genuinely complete now
      // Level 0 was NOT reloaded — the ladder survived (no reset).
      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
    });

    it('a differing budget with an identical view does NOT reset the ladder', async () => {
      await loader.updateView({ ...baseViewState, frameBudgetMs: 10 });
      expect(loader.loadedLODCount).toBe(1);

      await loader.updateView({ ...baseViewState, frameBudgetMs: 70 });
      // No reset: level 0 loaded once; the second pass continued at level 1.
      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(loader.loadedLODCount).toBeGreaterThanOrEqual(2);
    });

    it('playback prefix caching: capped ladders are stored, restored, and deepened loop-over-loop', async () => {
      // Mirrored in points/lines loader tests (three-geometry symmetry).
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const l = new GSplatsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as GSplatsSpatialIndexLoader[],
        3,
        '/g',
        undefined,
        sc
      );
      const viewA = baseViewState;
      const viewB = { ...baseViewState, slicePosition: [0, 0, 0, 1] };

      // Loop-1 tick at view A: budget 10 < one 30ms level → prefix(1) STORED.
      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      expect(l.loadedLODCount).toBe(1);
      expect(sc.getStats().count).toBe(1);

      await l.updateView({ ...viewB, frameBudgetMs: 10 }); // move on (stores B's prefix)

      // Loop-2 tick at view A (budget 20): the prefix restores WITHOUT
      // re-streaming level 0, and the budget deepens the ladder by one level.
      lodA.updateViewWithResidency.mockClear();
      lodB.updateViewWithResidency.mockClear();
      lodC.updateViewWithResidency.mockClear();
      await l.updateView({ ...viewA, frameBudgetMs: 20 });
      expect(lodA.updateViewWithResidency).not.toHaveBeenCalled(); // from cache
      expect(lodB.updateViewWithResidency).toHaveBeenCalledTimes(1); // deepened
      expect(l.loadedLODCount).toBe(2);

      // Idle revisit (pause → no budget): prefix(2) restores, ladder completes,
      // the FULL ladder upgrades the cache entry.
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

      // Loop-3 tick at view A: FULL restore — zero streaming even under budget.
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
      // Mirrored in points/lines loader tests (three-geometry symmetry).
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const l = new GSplatsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as GSplatsSpatialIndexLoader[],
        3,
        '/g',
        undefined,
        sc
      );
      const viewA = baseViewState;
      const viewB = { ...baseViewState, slicePosition: [0, 0, 0, 1] };

      await l.updateView({ ...viewA, frameBudgetMs: 10 }); // prefix(1) stored
      await l.updateView({ ...viewB, frameBudgetMs: 10 });

      const key = SliceCache.makeKey('/g', buildSliceViewSig(viewA));
      const cachedPayload = sc.peek(key)!.payload as unknown[];
      expect(cachedPayload.length).toBe(1);

      // Restore + deepen: the loader must push into a COPIED container, so
      // the previously cached prefix array stays untouched (the upgrade
      // replaces the ENTRY, never mutates the old payload in place).
      await l.updateView({ ...viewA, frameBudgetMs: 20 });
      expect(cachedPayload.length).toBe(1);
    });

    it('stores with the scan hint while a frame budget is active, and without it when budget-free', async () => {
      // Mirrored across the three progressive loader tests (symmetry). The
      // hint selects scan-resistant (MRU-victim) eviction in the S-cache so
      // cyclic playback loops keep their loop-head prefix resident.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const setSpy = vi.spyOn(sc, 'set');
      const l = new GSplatsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as GSplatsSpatialIndexLoader[],
        3,
        '/g',
        undefined,
        sc
      );
      const viewA = baseViewState;
      const viewB = { ...baseViewState, slicePosition: [0, 0, 0, 1] };

      // Playback tick (budget active): the store carries scan: true.
      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      expect(setSpy).toHaveBeenCalled();
      expect(setSpy.mock.calls.at(-1)![2]).toEqual({ scan: true, pin: false });

      // Pause re-trigger (same view, budget-free): the ladder completes and
      // the full-ladder upgrade store is scan-free.
      setSpy.mockClear();
      await l.updateView(viewA);
      expect(setSpy).toHaveBeenCalled();
      for (const call of setSpy.mock.calls) {
        expect(call[2]).toEqual({ scan: false, pin: false });
      }

      // Next playback tick at a NEW view: the departure store for A (skipped
      // here only if not longer) and B's prefix store are scan-hinted again.
      setSpy.mockClear();
      await l.updateView({ ...viewB, frameBudgetMs: 10 });
      expect(setSpy).toHaveBeenCalled();
      for (const call of setSpy.mock.calls) {
        expect(call[2]).toEqual({ scan: true, pin: false });
      }
    });

    it('pins the stored ladder on a prefetch pass (prefetch → pin: true)', async () => {
      // Mirrored across the three progressive loader tests (symmetry). The
      // SlicePrefetcher marks its shadow pass `prefetch: true`; the loader
      // forwards pin so the projected t+1 slice survives eviction until the
      // foreground tick restores it.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const setSpy = vi.spyOn(sc, 'set');
      const l = new GSplatsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as GSplatsSpatialIndexLoader[],
        3,
        '/g',
        undefined,
        sc
      );
      await l.updateView({ ...baseViewState, frameBudgetMs: 10, prefetch: true });
      expect(setSpy).toHaveBeenCalled();
      expect(setSpy.mock.calls.at(-1)![2]).toEqual({ scan: true, pin: true });
    });

    it('shadow-prefetch handoff: a prefix stored by ANOTHER instance restores here and deepens', async () => {
      // Mirrored across the three progressive loader tests (symmetry). The
      // t+1 SlicePrefetcher runs SHADOW loader instances whose only handoff
      // to the foreground is the shared S-cache: the shadow stores view B's
      // prefix while the foreground displays A; the real tick at B then
      // restores that prefix (no level-0 re-stream) and deepens with its
      // own budget.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const shadow = new GSplatsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as GSplatsSpatialIndexLoader[],
        3,
        '/g',
        undefined,
        sc
      );
      const foreground = new GSplatsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as GSplatsSpatialIndexLoader[],
        3,
        '/g',
        undefined,
        sc
      );
      const viewA = baseViewState;
      const viewB = { ...baseViewState, slicePosition: [0, 0, 0, 1] };

      await foreground.updateView({ ...viewA, frameBudgetMs: 10 }); // real tick at A
      await shadow.updateView({ ...viewB, frameBudgetMs: 10 }); // shadow prefetches B: prefix(1)

      lodA.updateViewWithResidency.mockClear();
      lodB.updateViewWithResidency.mockClear();
      lodC.updateViewWithResidency.mockClear();

      // Real tick at B (budget 20 = one more 30ms level): level 0 comes from
      // the SHADOW's cache entry; the budget deepens from startLevel = 1.
      await foreground.updateView({ ...viewB, frameBudgetMs: 20 });
      expect(lodA.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(foreground.loadedLODCount).toBe(2);
    });
  });

  describe('prefetch scheduling', () => {
    it('fires prefetchChunks on the next unloaded LOD after a partial load', async () => {
      // Mock LOD B slow → C deferred, but C's prefetch fires.
      let now = 0;
      const spy = vi.spyOn(performance, 'now');
      spy.mockImplementation(() => {
        now += 5;
        return now;
      });
      lodB.updateView.mockImplementation(async () => {
        now += 25;
        return makeLodData(50);
      });

      await loader.loadGSplats(baseViewState);
      // Drain microtasks / fire-and-forget promise.
      await Promise.resolve();

      expect(lodC.prefetchChunks).toHaveBeenCalledWith(baseViewState);
      spy.mockRestore();
    });

    it('does NOT prefetch if all LODs already loaded', async () => {
      await loader.loadGSplats(baseViewState);
      // All 3 done → no prefetch should fire.
      expect(lodA.prefetchChunks).not.toHaveBeenCalled();
      expect(lodB.prefetchChunks).not.toHaveBeenCalled();
      expect(lodC.prefetchChunks).not.toHaveBeenCalled();
    });

    it('swallows prefetch failures (fire and forget)', async () => {
      // Force a state where prefetch fires, then reject.
      let now = 0;
      const spy = vi.spyOn(performance, 'now');
      spy.mockImplementation(() => {
        now += 5;
        return now;
      });
      lodB.updateView.mockImplementation(async () => {
        now += 25;
        return makeLodData(50);
      });
      lodC.prefetchChunks.mockRejectedValue(new Error('Network down'));

      // Must not throw.
      await loader.loadGSplats(baseViewState);
      await Promise.resolve();

      expect(lodC.prefetchChunks).toHaveBeenCalled();
      spy.mockRestore();
    });
  });

  describe('concatenation', () => {
    it('returns a single part as-is when only one LOD is loaded', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(0)); // short-circuit after LOD 0
      const result = await loader.loadGSplats(baseViewState);
      expect(result.splatCount).toBe(0);
    });

    it('concatenates positions / amplitudes / cholesky from multiple LODs', async () => {
      const result = await loader.loadGSplats(baseViewState);
      expect(result.positions.length).toBe(175 * 3);
      expect(result.amplitudes.length).toBe(175);
      // 3D cholesky packed = 6 → 175 × 6.
      expect(result.choleskyFactors.length).toBe(175 * 6);
    });

    it('preserves Uint8Array color type across LODs', async () => {
      const result = await loader.loadGSplats(baseViewState);
      expect(result.colors).toBeInstanceOf(Uint8Array);
      expect(result.colors?.length).toBe(175 * 3);
    });

    it('fills missing LOD colors with 255 (Uint8) when first LOD has colors', async () => {
      // LOD A has colors (uint8), LOD B has none, LOD C has uint8 colors.
      lodB.updateView.mockResolvedValue(makeLodData(50, 3)); // no colors
      const result = await loader.loadGSplats(baseViewState);
      expect(result.colors).toBeInstanceOf(Uint8Array);
      // LOD A: 100 splats × 3 = indices 0–299 (color 128)
      // LOD B: 50 splats × 3 = indices 300–449 (filled with 255)
      // LOD C: 25 splats × 3 = indices 450–524 (color 128)
      expect(result.colors?.[0]).toBe(128);
      expect(result.colors?.[300]).toBe(255);
      expect(result.colors?.[450]).toBe(128);
    });

    it('uses Uint16Array fill (65535) when first LOD colors are uint16', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(100, 3, { color: 'uint16' }));
      lodB.updateView.mockResolvedValue(makeLodData(50, 3)); // no colors
      lodC.updateView.mockResolvedValue(makeLodData(25, 3, { color: 'uint16' }));
      const result = await loader.loadGSplats(baseViewState);
      expect(result.colors).toBeInstanceOf(Uint16Array);
      expect(result.colors?.[300]).toBe(65535);
    });

    it('uses Float32Array fill (1.0) when first LOD colors are float32', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(100, 3, { color: 'float32' }));
      lodB.updateView.mockResolvedValue(makeLodData(50, 3));
      lodC.updateView.mockResolvedValue(makeLodData(25, 3, { color: 'float32' }));
      const result = await loader.loadGSplats(baseViewState);
      expect(result.colors).toBeInstanceOf(Float32Array);
      expect(result.colors?.[300]).toBeCloseTo(1.0, 5);
    });

    it('keeps colors null when no LOD has colors', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(100, 3));
      lodB.updateView.mockResolvedValue(makeLodData(50, 3));
      lodC.updateView.mockResolvedValue(makeLodData(25, 3));
      const result = await loader.loadGSplats(baseViewState);
      expect(result.colors).toBeNull();
    });
  });

  describe('dispose', () => {
    it('disposes all sub-loaders', () => {
      loader.dispose();
      expect(lodA.dispose).toHaveBeenCalled();
      expect(lodB.dispose).toHaveBeenCalled();
      expect(lodC.dispose).toHaveBeenCalled();
    });

    it('clears internal state', async () => {
      await loader.loadGSplats(baseViewState);
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
      lodA = makeSubLoader(makeLodData(100), { queries: 2, elementsLoaded: 100, memoryUsed: 10 });
      lodB = makeSubLoader(makeLodData(50), { queries: 3, elementsLoaded: 50, memoryUsed: 20 });
      loader = new GSplatsProgressiveLoader(
        [lodA, lodB] as unknown as GSplatsSpatialIndexLoader[],
        2,
        '/test_gsplats'
      );

      const metrics = loader.getMetrics();
      expect(metrics.path).toBe('/test_gsplats');
      expect(metrics.type).toBe('gsplats-spatial-index');
      expect(metrics.queries).toBe(5); // 2 + 3
      expect(metrics.elementsLoaded).toBe(150); // 100 + 50
      expect(metrics.memoryUsed).toBe(30); // 10 + 20
    });

    it('addEventListener / removeEventListener fan out to every inner loader', () => {
      // The wrapper registers a re-pathing wrapper (not the raw listener) on
      // each inner loader — the re-path itself is asserted in the
      // ProgressiveMonitorAdapter unit test. Here we just confirm fan-out.
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

describe('GSplatsProgressiveLoader — concat memoization (no-op commit skip)', () => {
  let lodA: SubLoaderStub;
  let lodB: SubLoaderStub;
  let loader: GSplatsProgressiveLoader;

  beforeEach(() => {
    vi.useFakeTimers();
    lodA = makeSubLoader(makeLodData(100));
    lodB = makeSubLoader(makeLodData(50));
    loader = new GSplatsProgressiveLoader(
      [lodA, lodB] as unknown as GSplatsSpatialIndexLoader[],
      2,
      '/test_gsplats'
    );
  });

  it('returns the IDENTICAL reference for a repeat call with unchanged view state', async () => {
    const first = await loader.updateView(baseViewState);
    expect(first.splatCount).toBe(150);
    const second = await loader.updateView(baseViewState);
    // Same reference — the commit pipeline uses this identity to skip
    // no-op re-commits (mesh.userData.committedData === data).
    expect(second).toBe(first);
  });

  it('returns a NEW reference after a view-state change back to the same LOD count (resetGeneration)', async () => {
    const first = await loader.updateView(baseViewState);
    expect(first.splatCount).toBe(150);

    // Scrub away…
    const away = await loader.updateView({
      ...baseViewState,
      slicePosition: [1, 1, 1],
    });
    expect(away).not.toBe(first);

    // …and back: same slicing state, same LOD count as `first` — but the
    // loaders re-fetched, so contents may differ. Must NOT be the memo.
    const back = await loader.updateView(baseViewState);
    expect(back).not.toBe(first);

    // Repeat at the restored state memoizes again.
    const backAgain = await loader.updateView(baseViewState);
    expect(backAgain).toBe(back);
  });

  it('returns a NEW reference when a refinement pass adds a LOD level', async () => {
    // 3 LODs with a cache miss at level 1: the first call loads LOD 0 and
    // the missing LOD 1, then stops - LOD 2 is left for refinement.
    const lodC = makeSubLoader(makeLodData(25));
    loader = new GSplatsProgressiveLoader(
      [lodA, lodB, lodC] as unknown as GSplatsSpatialIndexLoader[],
      3,
      '/test_gsplats'
    );
    lodB.updateViewWithResidency.mockImplementation(async () => ({
      data: makeLodData(50),
      allResident: false,
    }));

    const first = await loader.updateView(baseViewState);
    expect(first.splatCount).toBe(150); // LOD 0 + LOD 1 (the miss breaks the loop)
    expect(loader.hasMoreLODs).toBe(true);

    const second = await loader.updateView(baseViewState);
    expect(second.splatCount).toBe(175); // LOD 2 added
    expect(second).not.toBe(first);
  });
});

describe('GSplatsProgressiveLoader — committedEnergyFraction (quality stamps)', () => {
  function make(table?: Array<number | null>) {
    const lodA = makeSubLoader(makeLodData(100, 3, { color: 'uint8' }));
    const lodB = makeSubLoader(makeLodData(50, 3, { color: 'uint8' }));
    return new GSplatsProgressiveLoader(
      [lodA, lodB] as unknown as GSplatsSpatialIndexLoader[],
      2,
      '/test_gsplats',
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
