/**
 * Unit tests for data/gsplats/gsplats-progressive-loader.ts.
 *
 * Pure orchestration: the composite loader wraps N
 * GSplatsSpatialIndexLoader sub-loaders and decides how many LODs
 * to load per frame based on cache-hit timing. Tests stub the
 * sub-loaders and inspect the merged result.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  GSplatsProgressiveLoader,
  concatenateGSplatsData,
} from '../../../data/gsplats/gsplats-progressive-loader';
import type { GSplatsSpatialIndexLoader } from '../../../data/gsplats/gsplats-spatial-index-loader';
import type { GSplatsViewState, LoadedGSplatsData } from '../../../types/gsplats';
import { CACHE_HIT_THRESHOLD_MS } from '../../../data/loaders/progressive/constants';
import { SliceCache } from '../../../cache/slice-cache';
import { buildSliceViewSig } from '../../../data/loaders/progressive/slice-cache-helper';
import { getPrefixParent } from '../../../types/prefix-lineage';

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
    visibleElements: 0,
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

  describe('prefix lineage (Phase 4 Stage 2 append)', () => {
    it('forward-chains each concat result to the previous same-generation result', async () => {
      // Call 1: LOD 1 not resident → the loop stops after LOD 0 (partial ladder).
      lodB.updateViewWithResidency.mockResolvedValueOnce({
        data: makeLodData(50, 3, { color: 'uint8' }),
        allResident: false,
      });
      const r1 = await loader.updateView(baseViewState);
      // First concat of the generation extends nothing.
      expect(getPrefixParent(r1)).toBeUndefined();
      expect(loader.hasMoreLODs).toBe(true);

      // Call 2 (SAME view): refinement loads the remaining levels → a new,
      // longer concat that forward-chains to r1.
      const r2 = await loader.updateView(baseViewState);
      expect(r2).not.toBe(r1);
      expect(r2.splatCount).toBeGreaterThan(r1.splatCount);
      expect(getPrefixParent(r2)).toBe(r1);
    });

    it('drops lineage across a view change (new generation → full rewrite)', async () => {
      await loader.updateView(baseViewState);
      // A slicePosition change resets the ladder + bumps the generation, so the
      // first concat of the new generation has no parent → append gate rejects.
      const r2 = await loader.updateView({ ...baseViewState, slicePosition: [0, 0, 0, 1] });
      expect(getPrefixParent(r2)).toBeUndefined();
    });

    it('a memoized no-op re-commit keeps the SAME reference (its lineage is unchanged)', async () => {
      const r1 = await loader.updateView(baseViewState); // full ladder in one call
      expect(loader.hasMoreLODs).toBe(false);
      const r2 = await loader.updateView(baseViewState); // no new LODs → memoized
      expect(r2).toBe(r1); // same reference → commit takes the stamp-only no-op
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

    it('reports hasMoreLODs=false after an empty LOD 0 (no wasted refinement passes)', async () => {
      // An empty slice has no content at any level. Leaving hasMoreLODs=true
      // (loadedLODs.length 1 < nLods) makes queueNext schedule refinement,
      // which then fetches+decodes every higher (also-empty) LOD one pass at a
      // time — pure waste, repeated on every revisit. The empty LOD 0 must mark
      // the ladder terminal.
      lodA.updateView.mockResolvedValue(makeLodData(0));
      await loader.loadGSplats(baseViewState);
      expect(loader.hasMoreLODs).toBe(false);
    });

    it('does NOT load higher LODs on a same-view re-invoke after an empty LOD 0', async () => {
      // refine-on-pause fires setDimensionValue(current) → a SAME-view,
      // budget-free updateView. hasMoreLODs gates *refinement*, not a direct
      // re-invoke: without the terminal-ladder short-circuit the loop would
      // re-enter at startLevel=1 and fetch the higher (empty) LODs once.
      lodA.updateView.mockResolvedValue(makeLodData(0));
      await loader.loadGSplats(baseViewState);
      lodB.updateViewWithResidency.mockClear();
      lodC.updateViewWithResidency.mockClear();
      await loader.loadGSplats(baseViewState); // same view again
      expect(lodB.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
    });
  });

  describe('S-cache restored EMPTY prefix (terminal re-derivation)', () => {
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

    it('does not re-stream higher LODs when a restored prefix has an empty LOD 0', async () => {
      // Scrub away from an empty slice (the departure store snapshots the
      // [empty-LOD0] 1-level ladder), then scrub back: the restore path
      // must RE-DERIVE the terminal flag from the restored prefix — the
      // level===0 empty check only fires for freshly LOADED levels, so
      // without re-derivation the loop resumes at startLevel=1 and
      // re-fetches every higher (equally empty) LOD on every revisit.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const a = makeSubLoader(makeLodData(0));
      const b = makeSubLoader(makeLodData(50, 3, { color: 'uint8' }));
      const l = new GSplatsProgressiveLoader(
        [a, b] as unknown as GSplatsSpatialIndexLoader[],
        2,
        '/g-empty',
        undefined,
        sc
      );

      await l.loadGSplats(viewA); // empty LOD 0 → terminal 1-level ladder
      expect(l.hasMoreLODs).toBe(false);
      // The DISCOVERY pass itself must not prefetch the (empty) next LOD —
      // prefetch goes through prefetchChunks, a separate surface from
      // updateViewWithResidency, so pin it explicitly.
      expect(b.prefetchChunks).not.toHaveBeenCalled();
      await l.loadGSplats(viewB); // departure: stores viewA's ladder

      a.updateViewWithResidency.mockClear();
      b.updateViewWithResidency.mockClear();
      await l.loadGSplats(viewA); // revisit the empty slice

      // Restored from the S-cache (LOD 0 not re-fetched)…
      expect(a.updateViewWithResidency).not.toHaveBeenCalled();
      // …and the restored empty prefix is TERMINAL: no higher-LOD fetch,
      // and refinement stays off.
      expect(b.updateViewWithResidency).not.toHaveBeenCalled();
      expect(b.prefetchChunks).not.toHaveBeenCalled();
      expect(l.hasMoreLODs).toBe(false);
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

    it('a foreground PLAYBACK pass loads ONLY the LOD-0 first-paint floor (never blocks on fine levels)', async () => {
      // Even with a budget that fits two 30ms levels, a budgeted foreground
      // (non-prefetch) pass commits just the floor and stays responsive —
      // deepening is the background prefetch's job.
      await loader.updateView({ ...baseViewState, frameBudgetMs: 70 });

      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodB.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
      expect(loader.loadedLODCount).toBe(1);
    });

    it('a background PREFETCH pass deepens as many levels as fit the budget', async () => {
      // Budget 50ms fits two 30ms levels; the third's loop-top check fails.
      // Prefetch has no first-paint floor — it deepens toward the full ladder.
      await loader.updateView({ ...baseViewState, frameBudgetMs: 50, prefetch: true });

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
      // No reset: level 0 is NOT reloaded (same-view, ladder survives). Playback
      // stays at the floor — deepening is the background prefetch's job, not the
      // foreground tick's.
      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(loader.loadedLODCount).toBe(1);
    });

    it('background prefetch deepens a capped ladder loop-over-loop; playback restores it responsively', async () => {
      // Mirrored in points/lines loader tests (three-geometry symmetry).
      // The responsiveness contract: foreground PLAYBACK ticks commit only the
      // cached prefix (never block on fine levels), while background PREFETCH
      // passes deepen the SAME slice's cached ladder +1 level at a time. So a
      // later playback tick restores a DEEPER prefix — higher quality, still
      // instant.
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

      // Loop-1 playback tick at A: budget 10 < one 30ms level → floor(1) STORED.
      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      expect(l.loadedLODCount).toBe(1);
      expect(sc.getStats().count).toBe(1);

      await l.updateView({ ...viewB, frameBudgetMs: 10 }); // move on (stores B's floor)

      // Background prefetch at A (budget 20): restores the floor WITHOUT
      // re-streaming level 0 and deepens the cached ladder by one level.
      lodA.updateViewWithResidency.mockClear();
      lodB.updateViewWithResidency.mockClear();
      lodC.updateViewWithResidency.mockClear();
      await l.updateView({ ...viewA, frameBudgetMs: 20, prefetch: true });
      expect(lodA.updateViewWithResidency).not.toHaveBeenCalled(); // level 0 from cache
      expect(lodB.updateViewWithResidency).toHaveBeenCalledTimes(1); // deepened +1
      expect(l.loadedLODCount).toBe(2);

      await l.updateView({ ...viewB, frameBudgetMs: 10 }); // move on

      // Loop-2 PLAYBACK tick at A: restores the DEEPER prefix(2) — higher
      // quality than loop 1 — and commits it with ZERO streaming (responsive:
      // the floor gate blocks any further foreground decode).
      lodA.updateViewWithResidency.mockClear();
      lodB.updateViewWithResidency.mockClear();
      lodC.updateViewWithResidency.mockClear();
      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      expect(lodA.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
      expect(l.loadedLODCount).toBe(2); // shows the deepened quality, no re-decode
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

    it('shadow-prefetch handoff: a prefix stored by ANOTHER (prefetch) instance restores here', async () => {
      // Mirrored across the three progressive loader tests (symmetry). The
      // t+1 SlicePrefetcher runs SHADOW loader instances (prefetch: true) whose
      // only handoff to the foreground is the shared S-cache: the shadow
      // deepens view B's ladder while the foreground displays A; the real
      // PLAYBACK tick at B then restores that cached prefix (no level-0
      // re-stream) and commits it responsively (the floor gate blocks any
      // foreground decode).
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

      await foreground.updateView({ ...viewA, frameBudgetMs: 10 }); // real tick at A: floor
      await shadow.updateView({ ...viewB, frameBudgetMs: 10, prefetch: true }); // shadow caches B

      lodA.updateViewWithResidency.mockClear();
      lodB.updateViewWithResidency.mockClear();
      lodC.updateViewWithResidency.mockClear();

      // Real PLAYBACK tick at B: level 0 comes from the SHADOW's cache entry —
      // no LOD loader runs (restored, not re-streamed).
      await foreground.updateView({ ...viewB, frameBudgetMs: 20 });
      expect(lodA.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
      expect(foreground.loadedLODCount).toBe(1);
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

    it('rejects mixed color dtypes across LOD levels (ladder-dtype contract)', async () => {
      // TypedArray.set converts by VALUE, not semantics — a Float32 level
      // (0..1) merged into a Uint8 (0..255) output truncates to garbage.
      // Malformed ladders fail fast instead of rendering corruption.
      lodA.updateView.mockResolvedValue(makeLodData(100, 3, { color: 'uint8' }));
      lodB.updateView.mockResolvedValue(makeLodData(50, 3, { color: 'float32' }));
      lodC.updateView.mockResolvedValue(makeLodData(25, 3, { color: 'uint8' }));
      await expect(loader.loadGSplats(baseViewState)).rejects.toThrow(/mixed color dtypes/);
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

describe('concatenateGSplatsData — RGBA color layout (per-element opacity)', () => {
  // Regression for a bug found in double-check: the additive-ladder color
  // concat hardcoded a stride of 3, so an RGBA ladder (what imported 3DGS
  // scenes become after `gsplat lod`) was truncated + misaligned and its
  // colorComponents defaulted to 3 downstream. Fail-first: against the
  // pre-fix stride the alpha column is dropped and the buffer is 3-strided.
  function makeRgbaLod(splatCount: number, alpha: number): LoadedGSplatsData {
    const colors = new Float32Array(splatCount * 4);
    for (let i = 0; i < splatCount; i++) {
      colors[i * 4] = 0.2;
      colors[i * 4 + 1] = 0.4;
      colors[i * 4 + 2] = 0.6;
      colors[i * 4 + 3] = alpha;
    }
    return {
      positions: new Float32Array(splatCount * 3).fill(0.5),
      amplitudes: new Float32Array(splatCount).fill(1),
      choleskyFactors: new Float32Array(splatCount * 6).fill(0.1),
      colors,
      colorComponents: 4,
      splatCount,
      ndim: 3,
    };
  }

  it('preserves the 4th channel and reports colorComponents=4 (RGBA ladder)', () => {
    const merged = concatenateGSplatsData([makeRgbaLod(3, 0.9), makeRgbaLod(2, 0.3)]);
    expect(merged.colorComponents).toBe(4);
    expect(merged.colors).not.toBeNull();
    expect(merged.colors!.length).toBe(5 * 4); // NOT 5 * 3 (the pre-fix bug)
    // Alpha of each splat lands intact and stride-aligned.
    for (let i = 0; i < 3; i++) expect(merged.colors![i * 4 + 3]).toBeCloseTo(0.9, 6);
    for (let i = 3; i < 5; i++) expect(merged.colors![i * 4 + 3]).toBeCloseTo(0.3, 6);
    // RGB survives (stride-aligned, not smeared by a wrong stride).
    expect(merged.colors![4 * 4]).toBeCloseTo(0.2, 6);
    expect(merged.colors![4 * 4 + 2]).toBeCloseTo(0.6, 6);
  });

  it('RGB ladder stays colorComponents=3', () => {
    const merged = concatenateGSplatsData([
      makeLodData(3, 3, { color: 'float32' }),
      makeLodData(2, 3, { color: 'float32' }),
    ]);
    expect(merged.colorComponents).toBe(3);
    expect(merged.colors!.length).toBe(5 * 3);
  });

  it('opaque white-fills a colorless LOD at the RGBA stride (alpha=1)', () => {
    const merged = concatenateGSplatsData([
      makeRgbaLod(2, 0.5),
      makeLodData(2, 3, { color: 'none' }),
    ]);
    expect(merged.colorComponents).toBe(4);
    expect(merged.colors!.length).toBe(4 * 4);
    // The colorless part fills opaque white (1,1,1,1) at stride 4.
    for (let i = 2; i < 4; i++) {
      expect(merged.colors![i * 4]).toBeCloseTo(1, 6);
      expect(merged.colors![i * 4 + 3]).toBeCloseTo(1, 6);
    }
  });

  it('rejects mixed color LAYOUTS across LOD levels (RGB level inside an RGBA ladder)', () => {
    // The dtype check cannot catch this — an RGB and an RGBA level can share
    // Float32Array — but `colorK` strides every copy, so the RGB level would
    // land at the wrong stride and silently corrupt every splat after it.
    expect(() =>
      concatenateGSplatsData([makeRgbaLod(3, 0.9), makeLodData(2, 3, { color: 'float32' })])
    ).toThrow(/mixed color layouts .*3 vs 4 components/);
    // And the reverse: an RGBA level inside an RGB ladder.
    expect(() =>
      concatenateGSplatsData([makeLodData(3, 3, { color: 'float32' }), makeRgbaLod(2, 0.9)])
    ).toThrow(/mixed color layouts .*4 vs 3 components/);
  });

  it('rejects mixed dimensionality across LOD levels (ndim strides the concat)', () => {
    // Same fail-fast family as the dtype/layout checks: ndim strides the
    // position concat AND sizes the Cholesky blocks, so sub-LODs disagreeing
    // on it would silently mis-stride every splat after the first part.
    expect(() =>
      concatenateGSplatsData([
        makeLodData(3, 3, { color: 'none' }),
        makeLodData(2, 4, { color: 'none' }),
      ])
    ).toThrow(/mixed dimensionality .*ndim 4 vs 3/);
  });
});

describe('dispose during an in-flight level (teardown race)', () => {
  it('stops streaming instead of indexing into the cleared lodLoaders', async () => {
    // A dispose() racing the awaited level used to make the NEXT iteration
    // read `this.lodLoaders[level]` as undefined — a TypeError that the
    // refinement loop then mis-counted as a real failure. The loop now
    // checks `_disposed` and breaks.
    let release!: () => void;
    const gate = new Promise<void>((r) => {
      release = r;
    });
    const slowA = makeSubLoader(makeLodData(10, 3, { color: 'uint8' }));
    const origA = slowA.updateViewWithResidency as unknown as (
      vs: unknown,
      s?: unknown
    ) => Promise<unknown>;
    slowA.updateViewWithResidency = vi.fn(async (vs: unknown, s?: unknown) => {
      const out = await origA(vs, s);
      await gate; // hold level 0 in flight
      return out;
    });
    const fastB = makeSubLoader(makeLodData(5, 3, { color: 'uint8' }));
    const l = new GSplatsProgressiveLoader(
      [slowA, fastB] as unknown as GSplatsSpatialIndexLoader[],
      2,
      '/dispose-race'
    );
    const pending = l.loadGSplats(baseViewState);
    l.dispose(); // clears lodLoaders while level 0 is awaited
    release();
    await expect(pending).resolves.toBeDefined(); // pre-fix: TypeError
    // Level 1 was never touched after the dispose.
    expect(fastB.updateViewWithResidency).not.toHaveBeenCalled();
  });
});

describe('determinant-equal dimensions refresh (three-geometry twin of the points test)', () => {
  it('does NOT reload on a determinant-equal dimensions refresh, then adopts the new reference', async () => {
    // The scene rebuilds the dimensions objects right after the first data
    // load (range fill, displayed-dim step derivation, key reorder). The
    // loader must neither reset on it (view-state-equal's determinant
    // projection) nor keep sig-comparing forever afterwards — it adopts the
    // fresh reference so later passes ref-short-circuit. A REAL query change
    // after the refresh must still reset. Ports the points-loader test so a
    // regression in the GSplatsProgressiveLoader wiring can't hide behind the shared helper's
    // own unit tests.
    const dimsV1 = [
      { name: 'x', unit: 'units', scale: 1, range: null, display: true, step: null },
      { name: 'y', unit: 'units', scale: 1, range: null, display: true, step: null },
      { name: 'z', unit: 'units', scale: 1, range: null, display: true, step: null },
    ] as unknown as GSplatsViewState['dimensions'];
    const dimsV2 = [
      { name: 'x', unit: 'units', scale: 1, display: true, step: 1, range: [0, 9] },
      { name: 'y', unit: 'units', scale: 1, display: true, step: 1, range: [0, 9] },
      { name: 'z', unit: 'units', scale: 1, display: true, step: 1, range: [0, 9] },
    ] as unknown as GSplatsViewState['dimensions'];
    const lod0 = makeSubLoader(makeLodData(20, 3));
    const l = new GSplatsProgressiveLoader(
      [lod0] as unknown as GSplatsSpatialIndexLoader[],
      1,
      '/dims-refresh'
    );
    await l.loadGSplats({ ...baseViewState, dimensions: dimsV1 });
    lod0.updateView.mockClear();

    // Refresh: determinant-equal, different reference → no reload.
    await l.loadGSplats({ ...baseViewState, dimensions: dimsV2 });
    expect(lod0.updateView).not.toHaveBeenCalled();
    // Same refreshed reference again → still no reload (ref fast path).
    await l.loadGSplats({ ...baseViewState, dimensions: dimsV2 });
    expect(lod0.updateView).not.toHaveBeenCalled();

    // A genuine query change still resets.
    await l.loadGSplats({
      ...baseViewState,
      slicePosition: [1, 1, 1, 0],
      dimensions: dimsV2,
    });
    expect(lod0.updateView).toHaveBeenCalled();
  });
});
