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
    errors: 0,
    elementsLoaded: 0,
    bytesLoaded: 0,
    visibleElements: 0,
    avgQueryTime: 0,
    avgLoadTime: 0,
    memoryUsed: 0,
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

  describe('empty LOD levels (#1456)', () => {
    it('streams PAST an empty LOD 0 to the levels that DO have splats', async () => {
      // THE regression. This loader exists only for ADDITIVE ladders, whose
      // levels are disjoint increments of one permutation — LOD 0 is a small
      // SUBSET of the node (a few thousand splats under `-b stream:C` /
      // `--target-ms`), so a hidden-dimension slice that none of ITS members
      // lands on is ordinary and says nothing about levels 1..n-1. The loader
      // used to latch a terminal "empty ladder" on an empty LOD 0 and break,
      // so such a slice rendered NOTHING even though the higher levels held
      // plenty of splats right there.
      lodA.updateView.mockResolvedValue(makeLodData(0));
      const result = await loader.loadGSplats(baseViewState);
      expect(lodB.updateView).toHaveBeenCalled();
      expect(lodC.updateView).toHaveBeenCalled();
      expect(result.splatCount).toBe(75); // 0 + 50 + 25
    });

    it('keeps hasMoreLODs true after an empty LOD 0 while levels remain unloaded', async () => {
      // Level 1 is a cache miss, so the refine pass stops with 2 of 3 levels
      // loaded. An empty LOD 0 must not be mistaken for a finished ladder:
      // refinement has to stay scheduled, and the next pass resumes at level 2.
      lodA.updateView.mockResolvedValue(makeLodData(0));
      lodB.updateViewWithResidency.mockImplementation(async () => ({
        data: makeLodData(50, 3, { color: 'uint8' }),
        allResident: false,
      }));
      await loader.loadGSplats(baseViewState);
      expect(loader.loadedLODCount).toBe(2);
      expect(loader.hasMoreLODs).toBe(true);

      await loader.loadGSplats(baseViewState);
      expect(lodC.updateView).toHaveBeenCalled();
      expect(loader.loadedLODCount).toBe(3);
      expect(loader.hasMoreLODs).toBe(false);
    });

    it('still prefetches the next level after an empty LOD 0', async () => {
      // Budget-expiry needs a clock that MOVES: with the real one a 10ms budget
      // never expires against these instant stubs, and playback now streams
      // resident levels rather than stopping at level 0 (#2374). Advance on
      // every read so the deadline is past by level 1's loop-top check.
      let nowMs = 0;
      const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => (nowMs += 100));
      try {
        // The moving clock exhausts the playback budget after LOD 0. The level
        // that actually holds this slice's splats is the one still to come, so
        // skipping prefetch would leave it cold.
        lodA.updateView.mockResolvedValue(makeLodData(0));
        await loader.updateView({ ...baseViewState, frameBudgetMs: 10 });
        expect(loader.loadedLODCount).toBe(1);
        expect(lodB.prefetchChunks).toHaveBeenCalled();
      } finally {
        nowSpy.mockRestore();
      }
    });

    it('walks EVERY level even when they all come back empty', async () => {
      // The wholly empty slice. The loop must still visit all three levels
      // rather than infer their emptiness from level 0 — in an additive ladder
      // that inference is unsound (#1456), and the only way to know a slice is
      // empty is to look. Afterwards the ladder is complete, so nothing is left
      // to refine.
      lodA.updateView.mockResolvedValue(makeLodData(0));
      lodB.updateView.mockResolvedValue(makeLodData(0));
      lodC.updateView.mockResolvedValue(makeLodData(0));

      const result = await loader.loadGSplats(baseViewState);

      expect(lodA.updateView).toHaveBeenCalled();
      expect(lodB.updateView).toHaveBeenCalled();
      expect(lodC.updateView).toHaveBeenCalled();
      expect(result.splatCount).toBe(0);
      expect(loader.loadedLODCount).toBe(3);
      expect(loader.hasMoreLODs).toBe(false);
    });
  });

  describe('S-cache restored ladders', () => {
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

    it('streams on from a restored PREFIX whose LOD 0 is empty', async () => {
      // Budget-expiry needs a clock that MOVES (see the sibling tests): a 10ms
      // budget never expires against instant stubs on the real clock, and
      // playback now streams resident levels instead of stopping at level 0
      // (#2374). Restored before the budget-free scrub-back so the refine pass
      // below is timed honestly.
      let nowMs = 0;
      const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => (nowMs += 100));
      // The moving clock expires after LOD 0 and stores that 1-level prefix.
      // Scrubbing back restores it — and an empty restored LOD 0 is
      // exactly as uninformative as a freshly loaded one (#1456): the levels
      // the prefix never reached are disjoint increments that may well
      // intersect this slice, so the loop must resume at level 1. Concluding
      // "terminal" from `restored[0]` alone stranded the slice blank.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const a = makeSubLoader(makeLodData(0));
      const b = makeSubLoader(makeLodData(50, 3, { color: 'uint8' }));
      const l = new GSplatsProgressiveLoader(
        [a, b] as unknown as GSplatsSpatialIndexLoader[],
        2,
        '/g-empty-prefix',
        undefined,
        sc
      );

      try {
        await l.updateView({ ...viewA, frameBudgetMs: 10 }); // LOD 0 only, prefix stored
        expect(l.loadedLODCount).toBe(1);
        await l.updateView({ ...viewB, frameBudgetMs: 10 }); // move away
      } finally {
        nowSpy.mockRestore();
      }

      a.updateViewWithResidency.mockClear();
      b.updateViewWithResidency.mockClear();
      const restored = await l.loadGSplats(viewA); // scrub back, no budget

      expect(a.updateViewWithResidency).not.toHaveBeenCalled(); // served from the S-cache
      expect(b.updateViewWithResidency).toHaveBeenCalled(); // …and streamed on
      expect(restored.splatCount).toBe(50);
    });

    it('does not re-walk a restored FULL empty ladder', async () => {
      // An all-empty ladder is still a ladder: it must be STORED (it measures 0
      // bytes, which the cache accepts) and RESTORED, so the full-restore
      // short-circuit fires on the revisit and no level is queried again. If a
      // future cache change ever refused the 0-byte snapshot, `restored` would
      // come back null and this revisit would re-walk all of it — which is what
      // these assertions catch.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const a = makeSubLoader(makeLodData(0));
      const b = makeSubLoader(makeLodData(0));
      const l = new GSplatsProgressiveLoader(
        [a, b] as unknown as GSplatsSpatialIndexLoader[],
        2,
        '/g-empty',
        undefined,
        sc
      );

      await l.loadGSplats(viewA); // both levels empty → complete ladder, stored
      expect(l.loadedLODCount).toBe(2);
      await l.loadGSplats(viewB); // move away

      a.updateViewWithResidency.mockClear();
      b.updateViewWithResidency.mockClear();
      const revisited = await l.loadGSplats(viewA); // revisit the empty slice

      expect(a.updateViewWithResidency).not.toHaveBeenCalled();
      expect(b.updateViewWithResidency).not.toHaveBeenCalled();
      expect(revisited.splatCount).toBe(0);
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

    it('a foreground PLAYBACK pass STREAMS RESIDENT levels within its budget', async () => {
      // Budget 50ms fits two 30ms levels; the third's loop-top check fails.
      // Playback used to load level 0 and stop, which assumed LOD 0 is a usable
      // picture. On a node the viewer SLICES it is not — an additive ladder's
      // rungs are sized against the WHOLE node, so a 500-timepoint gsplat leaf
      // put ~42 splats of a 166,443-splat frame in LOD 0 and playback rendered
      // an empty screen (#2374, #2376). Resident levels are cheap, so spend the
      // budget on them.
      await loader.updateView({ ...baseViewState, frameBudgetMs: 50 });

      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodB.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
      expect(loader.loadedLODCount).toBe(2);
    });

    it('a foreground PLAYBACK pass STOPS at the first COLD level, however much budget is left', async () => {
      // The safety property that replaces the level-0 gate: a cold level costs
      // hundreds of ms and would blow the tick, so residency (not level index)
      // is the brake. Worst case is one cold load per pass — exactly what the
      // old floor-only behaviour cost.
      lodB.updateViewWithResidency.mockImplementation(async () => {
        now += 30;
        return { data: makeLodData(10, 3, { color: 'uint8' }), allResident: false };
      });

      await loader.updateView({ ...baseViewState, frameBudgetMs: 100_000 });

      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodB.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
      expect(loader.loadedLODCount).toBe(2);
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

    it('a differing budget deepens the existing ladder without resetting it', async () => {
      await loader.updateView({ ...baseViewState, frameBudgetMs: 10 });
      expect(loader.loadedLODCount).toBe(1);

      await loader.updateView({ ...baseViewState, frameBudgetMs: 70 });
      // No reset: level 0 is NOT reloaded, and the remaining playback budget
      // deepens the existing prefix instead of treating it worse than an empty
      // ladder (#2379).
      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodB.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(loader.loadedLODCount).toBe(2);
    });

    it('a restored prefix does not receive an unconditional level at zero budget', async () => {
      await loader.updateView({ ...baseViewState, frameBudgetMs: 10 });
      expect(loader.loadedLODCount).toBe(1);

      nowSpy.mockImplementation(() => (now += 5));
      await loader.updateView({ ...baseViewState, frameBudgetMs: 0 });

      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodB.updateViewWithResidency).not.toHaveBeenCalled();
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

      // Loop-2 PLAYBACK tick at A: restores the DEEPER prefix(2), then spends
      // the remaining foreground budget on the resident final level.
      lodA.updateViewWithResidency.mockClear();
      lodB.updateViewWithResidency.mockClear();
      lodC.updateViewWithResidency.mockClear();
      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      expect(lodA.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).toHaveBeenCalledTimes(1);
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

    it('shadow-prefetch handoff: a prefix stored by ANOTHER (prefetch) instance restores here', async () => {
      // Mirrored across the three progressive loader tests (symmetry). The
      // t+1 SlicePrefetcher runs SHADOW loader instances (prefetch: true) whose
      // only handoff to the foreground is the shared S-cache: the shadow
      // deepens view B's ladder while the foreground displays A; the real
      // PLAYBACK tick at B then restores that cached prefix (no level-0
      // re-stream) and spends its remaining budget on the next resident level.
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

      // Real PLAYBACK tick at B: level 0 comes from the SHADOW's cache entry,
      // then the foreground budget deepens the restored prefix by one level.
      await foreground.updateView({ ...viewB, frameBudgetMs: 20 });
      expect(lodA.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
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
      // Budget-expiry needs a clock that MOVES: with the real one a 10ms budget
      // never expires against these instant stubs, and playback now streams
      // resident levels rather than stopping at level 0 (#2374). Advance on
      // every read so the deadline is past by level 1's loop-top check.
      let nowMs = 0;
      const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => (nowMs += 100));
      try {
        // The moving clock expires after the first level, preserving the
        // single-part state of the ladder.
        const result = await loader.updateView({ ...baseViewState, frameBudgetMs: 10 });
        expect(loader.loadedLODCount).toBe(1);
        expect(result.splatCount).toBe(100);
      } finally {
        nowSpy.mockRestore();
      }
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

describe('concatenateGSplatsData — picking index-space fields (issue #1423)', () => {
  // A ladder payload must never publish `ranges`: they live in the SUB-LOD's
  // on-disk index space, while the pick path resolves labels against the
  // PARENT node. `parts.length === 1` is the first-paint state of EVERY
  // ladder, not an "unladdered" node, so the single-part passthrough is
  // exactly where a per-level map would leak. (A Points / Lines ladder's labels
  // live in one union CSR on the parent — #1422; gsplat ladders carry none.)
  it('strips ranges from a single-part passthrough', () => {
    const only: LoadedGSplatsData = {
      ...makeLodData(4),
      ranges: [{ start: 2048, end: 2052 }],
    };

    const merged = concatenateGSplatsData([only]);

    expect(merged.ranges).toBeUndefined();
    // The strip is a shallow copy — the caller's payload is left intact.
    expect(merged).not.toBe(only);
    expect(only.ranges).toBeDefined();
    expect(merged.splatCount).toBe(4);
    expect(merged.positions).toBe(only.positions);
  });

  it('returns the single part by REFERENCE when it carries no ranges', () => {
    // The memoized-concat no-op commit keys on reference identity, so the
    // common (unlabelled) case must not start minting new objects.
    const only = makeLodData(4);
    expect(concatenateGSplatsData([only])).toBe(only);
  });

  it('does not concatenate ranges across multiple parts', () => {
    const a: LoadedGSplatsData = { ...makeLodData(3), ranges: [{ start: 0, end: 3 }] };
    const b: LoadedGSplatsData = { ...makeLodData(2), ranges: [{ start: 10, end: 12 }] };

    const merged = concatenateGSplatsData([a, b]);

    expect(merged.splatCount).toBe(5);
    expect(merged.ranges).toBeUndefined();
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

  it('rejects a single part whose RGBA buffer omits colorComponents (layout omission)', () => {
    // Distinct from the cross-level mismatch above: here every part AGREES on
    // colorComponents (all default to 3) but one part carries a 4-wide (RGBA)
    // buffer without declaring it. That satisfies the downstream count·3
    // minimum yet mis-strides every splat after the first — silently-wrong
    // colors/alpha. The per-part layout check must throw at concat time.
    const rgbaUndeclared: LoadedGSplatsData = {
      positions: new Float32Array(2 * 3).fill(0.5),
      amplitudes: new Float32Array(2).fill(1),
      choleskyFactors: new Float32Array(2 * 6).fill(0.1),
      colors: new Float32Array(2 * 4).fill(0.5), // RGBA length, but…
      // …colorComponents OMITTED → defaults to 3 → 2×3=6 ≠ 8.
      splatCount: 2,
      ndim: 3,
    };
    expect(() =>
      concatenateGSplatsData([rgbaUndeclared, makeLodData(2, 3, { color: 'float32' })])
    ).toThrow(/LOD level 0\): colors length 8 does not match count 2/);
  });

  it('accepts a correctly-declared RGBA part (colorComponents: 4, no false positive)', () => {
    // The per-part check is a no-op when the declared layout matches the
    // buffer length — a properly-declared RGBA ladder still concatenates.
    const merged = concatenateGSplatsData([makeRgbaLod(2, 0.9), makeRgbaLod(2, 0.3)]);
    expect(merged.colorComponents).toBe(4);
    expect(merged.colors!.length).toBe(4 * 4);
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
