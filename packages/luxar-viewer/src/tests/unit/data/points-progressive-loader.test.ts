/**
 * Unit tests for data/points/points-progressive-loader.ts.
 *
 * Mirrors `gsplats-progressive-loader.test.ts`. Pure orchestration:
 * the composite loader wraps N PointsSpatialIndexLoader sub-loaders
 * and decides how many LODs to load per frame based on cache-hit
 * timing. Tests stub the sub-loaders and inspect the merged result.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import { PointsProgressiveLoader } from '../../../data/points/points-progressive-loader';
import type { PointsSpatialIndexLoader } from '../../../data/points/points-spatial-index-loader';
import type { LoadedPointsData, PointsViewState } from '../../../types/points';
import { CACHE_HIT_THRESHOLD_MS } from '../../../data/loaders/progressive/constants';
import { SliceCache } from '../../../cache/slice-cache';
import { buildSliceViewSig } from '../../../data/loaders/progressive/slice-cache-helper';
import { getPrefixParent } from '../../../types/prefix-lineage';
import { log } from '../../../utils/log';
import {
  resetLodLoadStats,
  setLodLoadStatsEnabled,
  snapshotLodLoadStats,
} from '../../../data/scene-loader/lod-load-stats';

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
    type: 'point-spatial-index' as const,
    path: '/points/additive_x',
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
  pointCount: number,
  ndim = 3,
  options: {
    color?: 'none' | 'uint8' | 'uint16' | 'float32';
    withRadii?: boolean;
    withSharpness?: boolean;
    withScalars?: boolean;
    /**
     * Per-level marker written into colors/radii/sharpness/scalars so a merged
     * ladder's rows can be traced back to the level they came from. Omitted
     * (the default) keeps each field's historical fill value, so every test
     * predating the marker is byte-unaffected.
     */
    mark?: number;
  } = {}
): LoadedPointsData {
  // Positions are ALWAYS 3D-projected (stride 3) regardless of `ndim` — the
  // points facade folds nD→3D projection into loadPoints() itself, and `ndim`
  // reports the ORIGINAL dimensionality. Mirrors the real loaders.
  const positions = new Float32Array(pointCount * 3);
  positions.fill(0.5);
  const mark = options.mark;
  let colors: Float32Array | Uint8Array | Uint16Array | undefined;
  if (options.color === 'uint8') {
    colors = new Uint8Array(pointCount * 3).fill(mark ?? 128);
  } else if (options.color === 'uint16') {
    colors = new Uint16Array(pointCount * 3).fill(mark ?? 32000);
  } else if (options.color === 'float32') {
    colors = new Float32Array(pointCount * 3).fill(mark ?? 0.5);
  }
  const result: LoadedPointsData = {
    positions,
    pointCount,
    ndim,
    metadata: {
      totalPoints: pointCount,
      loadedPoints: pointCount,
      bounds: new THREE.Box3(),
      usedSpatialIndex: false,
    },
  };
  if (colors !== undefined) {
    result.colors = colors;
  }
  if (options.withRadii) {
    result.radii = new Float32Array(pointCount).fill(mark ?? 0.1);
  }
  if (options.withSharpness) {
    result.sharpness = new Float32Array(pointCount).fill(mark ?? 0.5);
  }
  if (options.withScalars) {
    result.scalars = new Float32Array(pointCount).fill(mark ?? 1);
  }
  return result;
}

/**
 * A level the current slice culls to zero, shaped like the real thing:
 * `createEmptyPointsData` (points/projection.ts) returns ONLY
 * `{positions, pointCount, ndim, metadata}` — colors/radii/sharpness/scalars
 * are ABSENT, not zero-length. That asymmetry is what made an empty level
 * strip the whole merged ladder's optional attributes (#1456).
 */
function makeEmptyLodData(): LoadedPointsData {
  return makeLodData(0);
}

function makeSubLoader(
  initialData: LoadedPointsData,
  metrics: Record<string, number> = {}
): SubLoaderStub {
  const updateView = vi.fn().mockResolvedValue(initialData);
  // The progressive loader now calls updateViewWithResidency; delegate to
  // updateView so existing `.updateView` assertions still hold. Default
  // allResident=true so timing-based break tests are unaffected; tests that
  // exercise the residency break override this mock per-case.
  const updateViewWithResidency = vi.fn(async (vs: PointsViewState, s?: unknown) => ({
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

const baseViewState: PointsViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 0],
};

describe('PointsProgressiveLoader', () => {
  let lodA: SubLoaderStub;
  let lodB: SubLoaderStub;
  let lodC: SubLoaderStub;
  let loader: PointsProgressiveLoader;

  beforeEach(() => {
    vi.useFakeTimers();
    lodA = makeSubLoader(makeLodData(100, 3, { color: 'uint8' }));
    lodB = makeSubLoader(makeLodData(50, 3, { color: 'uint8' }));
    lodC = makeSubLoader(makeLodData(25, 3, { color: 'uint8' }));
    loader = new PointsProgressiveLoader(
      [lodA, lodB, lodC] as unknown as PointsSpatialIndexLoader[],
      3,
      '/points'
    );
  });

  describe('SliceCache integration', () => {
    const viewA: PointsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    };
    const viewB: PointsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 1],
      tolerance: [0, 0, 0, 0],
    };

    // Regression (deep-double-check round 5, Playwright-measured): scrubbing
    // faster than the ladder completes NEVER stored anything — the reset
    // branch discarded the partial ladder, so scrub-back was always cold
    // (measured 1/7 hits pre-fix vs 7/7 post-fix under 4G emulation). The
    // DEPARTURE store snapshots the outgoing view's prefix under the
    // OUTGOING key when the view changes.
    it('stores the outgoing PARTIAL ladder on view change (departure store)', async () => {
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const a = makeSubLoader(makeLodData(100, 3, { color: 'uint8' }));
      const b = makeSubLoader(makeLodData(50, 3, { color: 'uint8' }));
      const c = makeSubLoader(makeLodData(25, 3, { color: 'uint8' }));
      // Level 1 is a network miss: the loop pushes it then BREAKS, leaving a
      // 2-of-3 PREFIX that never completes (level 2 is never streamed).
      b.updateViewWithResidency.mockImplementation(async () => ({
        data: makeLodData(50, 3, { color: 'uint8' }),
        allResident: false,
      }));
      const l = new PointsProgressiveLoader(
        [a, b, c] as unknown as PointsSpatialIndexLoader[],
        3,
        '/p',
        undefined,
        sc
      );
      await l.loadPoints(viewA); // levels 0-1 only (miss break)
      expect(c.updateViewWithResidency).not.toHaveBeenCalled();
      expect(sc.getStats().count).toBe(0); // incomplete: no completion store
      await l.loadPoints(viewB); // leaving A -> departure store of A's prefix
      expect(sc.getStats().count).toBeGreaterThanOrEqual(1);

      // Scrub back to A: the prefix must be served from the cache (no
      // level-0 re-stream) and streaming resumes at level 1.
      a.updateViewWithResidency.mockClear();
      await l.loadPoints(viewA);
      expect(sc.getStats().hits).toBe(1);
      expect(a.updateViewWithResidency).not.toHaveBeenCalled();
    });

    it('restores a revisited view from the SliceCache without re-streaming sub-LODs', async () => {
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const a = makeSubLoader(makeLodData(100, 3, { color: 'uint8' }));
      const b = makeSubLoader(makeLodData(50, 3, { color: 'uint8' }));
      const l = new PointsProgressiveLoader(
        [a, b] as unknown as PointsSpatialIndexLoader[],
        2,
        '/p',
        undefined,
        sc
      );
      await l.loadPoints(viewA);
      await l.loadPoints(viewB);
      a.updateViewWithResidency.mockClear();
      b.updateViewWithResidency.mockClear();
      const restored = await l.loadPoints(viewA);
      expect(a.updateViewWithResidency).not.toHaveBeenCalled();
      expect(b.updateViewWithResidency).not.toHaveBeenCalled();
      expect(l.hasMoreLODs).toBe(false);
      expect(restored.pointCount).toBe(150);
      expect(sc.getStats().hits).toBeGreaterThanOrEqual(1);
    });

    it('clones on store so a later accumulator overwrite cannot corrupt a cached slice', async () => {
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const dataA = makeLodData(100, 3, { color: 'uint8' });
      const a = makeSubLoader(dataA);
      const b = makeSubLoader(makeLodData(50, 3, { color: 'uint8' }));
      const l = new PointsProgressiveLoader(
        [a, b] as unknown as PointsSpatialIndexLoader[],
        2,
        '/p',
        undefined,
        sc
      );
      await l.loadPoints(viewA);
      dataA.positions.fill(999);
      await l.loadPoints(viewB);
      const restored = await l.loadPoints(viewA);
      expect(restored.positions[0]).toBeCloseTo(0.5);
    });
  });

  describe('initial load', () => {
    it('loads LOD 0 on first call', async () => {
      const result = await loader.loadPoints(baseViewState);
      expect(lodA.updateView).toHaveBeenCalled();
      expect(result.pointCount).toBeGreaterThanOrEqual(100);
    });

    it('returns concatenated data from all loaded LODs', async () => {
      const result = await loader.loadPoints(baseViewState);
      // All 3 LODs are cache-hit fast (sync mock) → all loaded → 100+50+25 = 175.
      expect(result.pointCount).toBe(175);
    });

    it('records points additive load timing keys', async () => {
      resetLodLoadStats();
      setLodLoadStatsEnabled(true);
      try {
        await loader.loadPoints(baseViewState);
        expect(Object.keys(snapshotLodLoadStats())).toEqual([
          'additive:points:level:0:resident',
          'additive:points:level:1:resident',
          'additive:points:level:2:resident',
        ]);
      } finally {
        setLodLoadStatsEnabled(false);
        resetLodLoadStats();
      }
    });

    it('exposes totalLODCount and loadedLODCount', async () => {
      expect(loader.totalLODCount).toBe(3);
      expect(loader.loadedLODCount).toBe(0);
      await loader.loadPoints(baseViewState);
      expect(loader.loadedLODCount).toBe(3);
    });

    it('hasMoreLODs is false after all LODs loaded', async () => {
      expect(loader.hasMoreLODs).toBe(true);
      await loader.loadPoints(baseViewState);
      expect(loader.hasMoreLODs).toBe(false);
    });
  });

  describe('view-state change handling', () => {
    it('resets loaded LODs when displayDims changes', async () => {
      await loader.loadPoints(baseViewState);
      expect(loader.loadedLODCount).toBe(3);

      lodA.updateView.mockClear();
      await loader.loadPoints({ ...baseViewState, displayDims: [1, 2, 3] });
      expect(lodA.updateView).toHaveBeenCalled();
    });

    it('resets when slicePosition changes', async () => {
      await loader.loadPoints(baseViewState);
      lodA.updateView.mockClear();
      await loader.loadPoints({ ...baseViewState, slicePosition: [1, 1, 1] });
      expect(lodA.updateView).toHaveBeenCalled();
    });

    it('resets when tolerance changes', async () => {
      await loader.loadPoints(baseViewState);
      lodA.updateView.mockClear();
      await loader.loadPoints({ ...baseViewState, tolerance: [0.1, 0.1, 0.1] });
      expect(lodA.updateView).toHaveBeenCalled();
    });

    it('does NOT reload when view state is unchanged + all LODs loaded', async () => {
      await loader.loadPoints(baseViewState);
      lodA.updateView.mockClear();
      lodB.updateView.mockClear();
      lodC.updateView.mockClear();

      await loader.loadPoints(baseViewState);
      expect(lodA.updateView).not.toHaveBeenCalled();
      expect(lodB.updateView).not.toHaveBeenCalled();
      expect(lodC.updateView).not.toHaveBeenCalled();
    });

    it('does NOT reload on a determinant-equal dimensions refresh, then adopts the new reference', async () => {
      // The scene rebuilds the dimensions objects right after the first data
      // load (range fill, displayed-dim step derivation, key reorder). The
      // loader must neither reset on it (view-state-equal's determinant
      // projection) nor keep sig-comparing forever afterwards — it adopts
      // the fresh reference so later passes ref-short-circuit. A REAL query
      // change after the refresh must still reset.
      const dimsV1 = [
        { name: 'x', unit: 'units', scale: 1, range: null, display: true, step: null },
        { name: 'y', unit: 'units', scale: 1, range: null, display: true, step: null },
        { name: 'z', unit: 'units', scale: 1, range: null, display: true, step: null },
      ] as unknown as PointsViewState['dimensions'];
      const dimsV2 = [
        { name: 'x', unit: 'units', scale: 1, display: true, step: 1, range: [0, 9] },
        { name: 'y', unit: 'units', scale: 1, display: true, step: 1, range: [0, 9] },
        { name: 'z', unit: 'units', scale: 1, display: true, step: 1, range: [0, 9] },
      ] as unknown as PointsViewState['dimensions'];
      await loader.loadPoints({ ...baseViewState, dimensions: dimsV1 });
      lodA.updateView.mockClear();

      // Refresh: determinant-equal, different reference → no reload.
      await loader.loadPoints({ ...baseViewState, dimensions: dimsV2 });
      expect(lodA.updateView).not.toHaveBeenCalled();
      // Same refreshed reference again → still no reload (ref fast path).
      await loader.loadPoints({ ...baseViewState, dimensions: dimsV2 });
      expect(lodA.updateView).not.toHaveBeenCalled();

      // A genuine query change still resets.
      await loader.loadPoints({
        ...baseViewState,
        slicePosition: [1, 1, 1, 0],
        dimensions: dimsV2,
      });
      expect(lodA.updateView).toHaveBeenCalled();
    });

    it('treats different dimensions metadata as a state change', async () => {
      const stateA: PointsViewState = {
        ...baseViewState,
        dimensions: [{ name: 'x', unit: 'um', display: true, scale: 1 }],
      };
      const stateB: PointsViewState = {
        ...baseViewState,
        dimensions: [{ name: 'y', unit: 'um', display: true, scale: 1 }],
      };
      await loader.loadPoints(stateA);
      lodA.updateView.mockClear();
      await loader.loadPoints(stateB);
      expect(lodA.updateView).toHaveBeenCalled();
    });
  });

  describe('empty LOD levels (#1456)', () => {
    it('streams PAST an empty LOD 0 to the levels that DO have points', async () => {
      // THE regression. This loader exists only for ADDITIVE ladders, whose
      // levels are disjoint increments of one permutation — LOD 0 is a small
      // SUBSET of the node (a few thousand points under `-b stream:C` /
      // `--target-ms`), so a hidden-dimension slice that none of ITS members
      // lands on is ordinary and says nothing about levels 1..n-1. The loader
      // used to latch a terminal "empty ladder" on an empty LOD 0 and break,
      // so such a slice rendered NOTHING even though the higher levels held
      // plenty of points right there.
      lodA.updateView.mockResolvedValue(makeEmptyLodData());
      const result = await loader.loadPoints(baseViewState);
      expect(lodB.updateView).toHaveBeenCalled();
      expect(lodC.updateView).toHaveBeenCalled();
      expect(result.pointCount).toBe(75); // 0 + 50 + 25
    });

    it("keeps the surviving levels' attributes across an empty LOD 0", async () => {
      // An empty level is an ABSTAINER, not a veto. `createEmptyPointsData`
      // OMITS colors/radii/sharpness/scalars (absent, not zero-length) and
      // `concatOptionalField` is all-or-nothing, so without the abstainer rule
      // one empty level stripped EVERY optional attribute from the merged
      // ladder: `create-points-node.ts` then substitutes white / radius 0.5 /
      // sharpness 0.5 and `stampPointPresenceFlags` clears `hasScalars`,
      // killing the colormap. The #1456 slice would have gone from blank to 75
      // white, uniformly-sized, colormap-less points — a quieter wrong answer
      // than the blank one it replaced.
      const attrs = {
        color: 'uint8' as const,
        withRadii: true,
        withSharpness: true,
        withScalars: true,
      };
      lodA.updateView.mockResolvedValue(makeEmptyLodData());
      lodB.updateView.mockResolvedValue(makeLodData(50, 3, { ...attrs, mark: 11 }));
      lodC.updateView.mockResolvedValue(makeLodData(25, 3, { ...attrs, mark: 22 }));

      const r = await loader.loadPoints(baseViewState);

      expect(r.pointCount).toBe(75);
      // Lengths track the SURVIVING rows only — the empty level adds none.
      expect(r.colors).toBeInstanceOf(Uint8Array);
      expect(r.colors).toHaveLength(75 * 3);
      expect(r.colorComponents ?? 3).toBe(3);
      expect(r.radii).toHaveLength(75);
      expect(r.sharpness).toHaveLength(75);
      expect(r.scalars).toHaveLength(75);
      // …and every row carries the value its own level supplied, in order.
      expect(Array.from(r.colors!.slice(0, 3))).toEqual([11, 11, 11]);
      expect(Array.from(r.colors!.slice(50 * 3, 50 * 3 + 3))).toEqual([22, 22, 22]);
      expect(r.radii![0]).toBeCloseTo(11);
      expect(r.radii![49]).toBeCloseTo(11);
      expect(r.radii![50]).toBeCloseTo(22);
      expect(r.sharpness![0]).toBeCloseTo(11);
      expect(r.sharpness![50]).toBeCloseTo(22);
      expect(r.scalars![0]).toBeCloseTo(11);
      expect(r.scalars![74]).toBeCloseTo(22);
    });

    it('keeps attributes when the empty level lands in the MIDDLE of the ladder', async () => {
      // Same defect, and this half of it predates #1456: the old short-circuit
      // only fired at level 0, so a ladder whose level 1 was culled to zero
      // already lost every optional attribute.
      const attrs = {
        color: 'uint8' as const,
        withRadii: true,
        withSharpness: true,
        withScalars: true,
      };
      lodA.updateView.mockResolvedValue(makeLodData(100, 3, { ...attrs, mark: 11 }));
      lodB.updateView.mockResolvedValue(makeEmptyLodData());
      lodC.updateView.mockResolvedValue(makeLodData(25, 3, { ...attrs, mark: 22 }));

      const r = await loader.loadPoints(baseViewState);

      expect(r.pointCount).toBe(125); // 100 + 0 + 25
      expect(r.colors).toHaveLength(125 * 3);
      expect(r.scalars).toHaveLength(125);
      expect(Array.from(r.colors!.slice(0, 3))).toEqual([11, 11, 11]);
      // The empty level occupies no rows, so level 2 starts right after level 0.
      expect(Array.from(r.colors!.slice(100 * 3, 100 * 3 + 3))).toEqual([22, 22, 22]);
      expect(r.radii![99]).toBeCloseTo(11);
      expect(r.radii![100]).toBeCloseTo(22);
      expect(r.scalars![124]).toBeCloseTo(22);
    });

    it('keeps hasMoreLODs true after an empty LOD 0 while levels remain unloaded', async () => {
      // Level 1 is a cache miss, so the refine pass stops with 2 of 3 levels
      // loaded. An empty LOD 0 must not be mistaken for a finished ladder:
      // refinement has to stay scheduled, and the next pass must resume at
      // level 2 rather than re-walk what is already resident.
      lodA.updateView.mockResolvedValue(makeLodData(0));
      lodB.updateViewWithResidency.mockImplementation(async () => ({
        data: makeLodData(50, 3, { color: 'uint8' }),
        allResident: false,
      }));
      await loader.loadPoints(baseViewState);
      expect(loader.loadedLODCount).toBe(2);
      expect(loader.hasMoreLODs).toBe(true);

      await loader.loadPoints(baseViewState);
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
        // that actually holds this slice's points is the one still to come, so
        // skipping prefetch would leave it cold — the flip side of the
        // same wrong assumption.
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
      lodA.updateView.mockResolvedValue(makeEmptyLodData());
      lodB.updateView.mockResolvedValue(makeEmptyLodData());
      lodC.updateView.mockResolvedValue(makeEmptyLodData());

      const result = await loader.loadPoints(baseViewState);

      expect(lodA.updateView).toHaveBeenCalled();
      expect(lodB.updateView).toHaveBeenCalled();
      expect(lodC.updateView).toHaveBeenCalled();
      expect(result.pointCount).toBe(0);
      expect(loader.loadedLODCount).toBe(3);
      expect(loader.hasMoreLODs).toBe(false);
    });
  });

  describe('S-cache restored ladders', () => {
    const viewA: PointsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    };
    const viewB: PointsViewState = {
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
      const l = new PointsProgressiveLoader(
        [a, b] as unknown as PointsSpatialIndexLoader[],
        2,
        '/p-empty-prefix',
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
      const restored = await l.loadPoints(viewA); // scrub back, no budget

      expect(a.updateViewWithResidency).not.toHaveBeenCalled(); // served from the S-cache
      expect(b.updateViewWithResidency).toHaveBeenCalled(); // …and streamed on
      expect(restored.pointCount).toBe(50);
    });

    it('does not re-walk a restored FULL empty ladder', async () => {
      // An all-empty ladder is still a ladder: it must be STORED (it measures 0
      // bytes, which the cache accepts) and RESTORED, so the full-restore
      // short-circuit fires on the revisit and no level is queried again. If a
      // future cache change ever refused the 0-byte snapshot, `restored` would
      // come back null and this revisit would re-walk all of it — which is what
      // these assertions catch.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const a = makeSubLoader(makeEmptyLodData());
      const b = makeSubLoader(makeEmptyLodData());
      const l = new PointsProgressiveLoader(
        [a, b] as unknown as PointsSpatialIndexLoader[],
        2,
        '/p-empty',
        undefined,
        sc
      );

      await l.loadPoints(viewA); // both levels empty → complete ladder, stored
      expect(l.loadedLODCount).toBe(2);
      await l.loadPoints(viewB); // move away

      a.updateViewWithResidency.mockClear();
      b.updateViewWithResidency.mockClear();
      const revisited = await l.loadPoints(viewA); // revisit the empty slice

      expect(a.updateViewWithResidency).not.toHaveBeenCalled();
      expect(b.updateViewWithResidency).not.toHaveBeenCalled();
      expect(revisited.pointCount).toBe(0);
      expect(l.hasMoreLODs).toBe(false);
    });
  });

  describe('rollbackToPassStart (failed-commit recovery, #2426)', () => {
    it('unwinds only the levels the failing pass appended', async () => {
      lodB.updateViewWithResidency.mockImplementationOnce(async () => ({
        data: makeLodData(50, 3, { color: 'uint8' }),
        allResident: false,
      }));
      await loader.loadPoints(baseViewState);
      expect(loader.loadedLODCount).toBe(2);

      await loader.loadPoints(baseViewState);
      expect(loader.loadedLODCount).toBe(3);
      expect(loader.rollbackToPassStart()).toBe(1);
      expect(loader.loadedLODCount).toBe(2);
      expect(loader.hasMoreLODs).toBe(true);
    });

    it('evicts an uncommitted full ladder snapshot before a later slice revisit', async () => {
      const sliceCache = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const first = makeSubLoader(makeLodData(20, 3, { color: 'uint8' }));
      const second = makeSubLoader(makeLodData(10, 3, { color: 'uint8' }));
      const recovering = new PointsProgressiveLoader(
        [first, second] as unknown as PointsSpatialIndexLoader[],
        2,
        '/rollback-cache',
        undefined,
        sliceCache
      );
      const viewA = baseViewState;
      const viewB = { ...baseViewState, slicePosition: [0, 0, 0, 1] };

      await recovering.loadPoints(viewA);
      expect(recovering.rollbackToPassStart()).toBe(2);
      await recovering.loadPoints(viewB);

      first.updateViewWithResidency.mockClear();
      second.updateViewWithResidency.mockClear();
      await recovering.loadPoints(viewA);

      expect(first.updateViewWithResidency).toHaveBeenCalled();
      expect(second.updateViewWithResidency).toHaveBeenCalled();
    });
  });

  describe('cache-hit timing short-circuit', () => {
    it(`stops loading further LODs when one takes > ${CACHE_HIT_THRESHOLD_MS}ms`, async () => {
      // Make LOD B slow (over the threshold) so LOD C is deferred.
      let now = 0;
      const performanceNowSpy = vi.spyOn(performance, 'now');
      performanceNowSpy.mockImplementation(() => {
        now += 5;
        return now;
      });

      lodB.updateView.mockImplementation(async () => {
        now += CACHE_HIT_THRESHOLD_MS + 10; // simulated work, over threshold
        return makeLodData(50);
      });

      await loader.loadPoints(baseViewState);

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

      await loader.loadPoints(baseViewState);

      expect(lodA.updateViewWithResidency).toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).toHaveBeenCalled();
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
    });

    it('keeps loading while levels are resident', async () => {
      // All resident + fast → the loop loads every level in one call.
      await loader.loadPoints(baseViewState);
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
          return { data: makeLodData(10, 3, { color: 'uint8' }), allResident: true };
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
      // Budget 50ms fits two 30ms levels; prefetch has no first-paint floor.
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

      await loader.updateView(baseViewState);
      expect(loader.loadedLODCount).toBe(3);
      expect(loader.hasMoreLODs).toBe(false); // genuinely complete now
      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1); // no reset
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

    it('a restored prefetch prefix still deepens one level after its budget expires', async () => {
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const l = new PointsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as PointsSpatialIndexLoader[],
        3,
        '/p',
        undefined,
        sc
      );
      const viewA = baseViewState;
      const viewB = { ...baseViewState, slicePosition: [0, 0, 0, 1] };

      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      await l.updateView({ ...viewB, frameBudgetMs: 10 });

      lodA.updateViewWithResidency.mockClear();
      lodB.updateViewWithResidency.mockClear();
      lodC.updateViewWithResidency.mockClear();
      nowSpy.mockImplementation(() => (now += 5));

      await l.updateView({ ...viewA, frameBudgetMs: 0, prefetch: true });

      expect(lodA.updateViewWithResidency).not.toHaveBeenCalled();
      expect(lodB.updateViewWithResidency).toHaveBeenCalledTimes(1);
      expect(lodC.updateViewWithResidency).not.toHaveBeenCalled();
      expect(l.loadedLODCount).toBe(2);
    });

    it('background prefetch deepens a capped ladder loop-over-loop; playback restores it responsively', async () => {
      // Mirrors gsplats-progressive-loader.test.ts (three-geometry symmetry).
      // Background PREFETCH deepens the SAME slice's cached ladder +1 level at
      // a time. A later PLAYBACK tick restores that deeper prefix, then spends
      // only its remaining foreground budget on resident detail.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const l = new PointsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as PointsSpatialIndexLoader[],
        3,
        '/p',
        undefined,
        sc
      );
      const viewA = baseViewState;
      const viewB = { ...baseViewState, slicePosition: [0, 0, 0, 1] };

      // Loop-1 playback tick at A: floor(1) STORED.
      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      expect(l.loadedLODCount).toBe(1);
      expect(sc.getStats().count).toBe(1);

      await l.updateView({ ...viewB, frameBudgetMs: 10 }); // move on

      // Background prefetch at A (budget 20): restores the floor without
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
      // Mirrors gsplats-progressive-loader.test.ts (three-geometry symmetry).
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const l = new PointsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as PointsSpatialIndexLoader[],
        3,
        '/p',
        undefined,
        sc
      );
      const viewA = baseViewState;
      const viewB = { ...baseViewState, slicePosition: [0, 0, 0, 1] };

      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      await l.updateView({ ...viewB, frameBudgetMs: 10 });

      const key = SliceCache.makeKey('/p', buildSliceViewSig(viewA));
      const cachedPayload = sc.peek(key)!.payload as unknown[];
      expect(cachedPayload.length).toBe(1);

      await l.updateView({ ...viewA, frameBudgetMs: 20 });
      expect(cachedPayload.length).toBe(1);
    });

    it('stores with the scan hint while a frame budget is active, and without it when budget-free', async () => {
      // Mirrored across the three progressive loader tests (symmetry). The
      // hint selects scan-resistant (MRU-victim) eviction in the S-cache so
      // cyclic playback loops keep their loop-head prefix resident.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const setSpy = vi.spyOn(sc, 'set');
      const l = new PointsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as PointsSpatialIndexLoader[],
        3,
        '/p',
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
      const l = new PointsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as PointsSpatialIndexLoader[],
        3,
        '/p',
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
      // re-stream) and spends its remaining budget on resident detail.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const shadow = new PointsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as PointsSpatialIndexLoader[],
        3,
        '/p',
        undefined,
        sc
      );
      const foreground = new PointsProgressiveLoader(
        [lodA, lodB, lodC] as unknown as PointsSpatialIndexLoader[],
        3,
        '/p',
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

      await loader.loadPoints(baseViewState);
      await Promise.resolve();

      expect(lodC.prefetchChunks).toHaveBeenCalledWith(baseViewState);
      spy.mockRestore();
    });

    it('does NOT prefetch if all LODs already loaded', async () => {
      await loader.loadPoints(baseViewState);
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
        return makeLodData(50);
      });
      lodC.prefetchChunks.mockRejectedValue(new Error('Network down'));

      // Must not throw.
      await loader.loadPoints(baseViewState);
      await Promise.resolve();

      expect(lodC.prefetchChunks).toHaveBeenCalled();
      spy.mockRestore();
    });
  });

  describe('prefix lineage (Phase 4 Stage 2 append)', () => {
    it('forward-chains each concat result to the previous same-generation result', async () => {
      // Call 1: LOD 1 not resident → the loop stops after LOD 0 (partial ladder).
      lodB.updateViewWithResidency.mockResolvedValueOnce({
        data: makeLodData(50, 3, { color: 'uint8' }),
        allResident: false,
      });
      const r1 = await loader.loadPoints(baseViewState);
      // First concat of the generation extends nothing.
      expect(getPrefixParent(r1)).toBeUndefined();
      expect(loader.hasMoreLODs).toBe(true);

      // Call 2 (SAME view): refinement loads the remaining levels → a new,
      // longer concat that forward-chains to r1.
      const r2 = await loader.loadPoints(baseViewState);
      expect(r2).not.toBe(r1);
      expect(r2.pointCount).toBeGreaterThan(r1.pointCount);
      expect(getPrefixParent(r2)).toBe(r1);
    });

    it('drops lineage across a view change (new generation → full rewrite)', async () => {
      await loader.loadPoints(baseViewState);
      // A slicePosition change resets the ladder + bumps the generation, so the
      // first concat of the new generation has no parent → append gate rejects.
      const r2 = await loader.loadPoints({ ...baseViewState, slicePosition: [0, 0, 0, 1] });
      expect(getPrefixParent(r2)).toBeUndefined();
    });

    it('a memoized no-op re-commit keeps the SAME reference (its lineage is unchanged)', async () => {
      const r1 = await loader.loadPoints(baseViewState); // full ladder in one call
      expect(loader.hasMoreLODs).toBe(false);
      const r2 = await loader.loadPoints(baseViewState); // no new LODs → memoized
      expect(r2).toBe(r1); // same reference → commit takes the stamp-only no-op
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
        expect(result.pointCount).toBe(100);
      } finally {
        nowSpy.mockRestore();
      }
    });

    it('concatenates positions across multiple LODs', async () => {
      const result = await loader.loadPoints(baseViewState);
      expect(result.positions.length).toBe(175 * 3);
      expect(result.pointCount).toBe(175);
    });

    it('concatenates >3D positions at stride 3 (positions are 3D-projected)', async () => {
      // Regression: the concat used perItem=ndim, but points positions are
      // ALWAYS 3D-projected (stride 3) while ndim reports the original
      // dimensionality — for 4D data every level after the first landed at
      // wrong offsets (zeros/garbage on screen). Distinct per-level fill
      // values pin the contiguous stride-3 layout.
      const lod = (n: number, fill: number): LoadedPointsData => {
        const d = makeLodData(n, 4);
        (d.positions as Float32Array).fill(fill);
        return d;
      };
      lodA.updateView.mockResolvedValue(lod(100, 1));
      lodB.updateView.mockResolvedValue(lod(50, 2));
      lodC.updateView.mockResolvedValue(lod(25, 3));
      const result = await loader.loadPoints(baseViewState);
      expect(result.ndim).toBe(4); // original dimensionality preserved
      expect(result.pointCount).toBe(175);
      expect(result.positions.length).toBe(175 * 3); // stride 3, no ndim padding
      // Level boundaries land contiguously at stride 3.
      expect(result.positions[0]).toBe(1); // first LOD-A element
      expect(result.positions[100 * 3 - 1]).toBe(1); // last LOD-A element
      expect(result.positions[100 * 3]).toBe(2); // first LOD-B element
      expect(result.positions[150 * 3 - 1]).toBe(2); // last LOD-B element
      expect(result.positions[150 * 3]).toBe(3); // first LOD-C element
      expect(result.positions[175 * 3 - 1]).toBe(3); // last LOD-C element
    });

    it('preserves Uint8Array color type across LODs', async () => {
      const result = await loader.loadPoints(baseViewState);
      expect(result.colors).toBeInstanceOf(Uint8Array);
      expect(result.colors?.length).toBe(175 * 3);
    });

    it('preserves Uint16Array color type across LODs', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(100, 3, { color: 'uint16' }));
      lodB.updateView.mockResolvedValue(makeLodData(50, 3, { color: 'uint16' }));
      lodC.updateView.mockResolvedValue(makeLodData(25, 3, { color: 'uint16' }));
      const result = await loader.loadPoints(baseViewState);
      expect(result.colors).toBeInstanceOf(Uint16Array);
      expect(result.colors?.length).toBe(175 * 3);
    });

    it('preserves Float32Array color type across LODs', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(100, 3, { color: 'float32' }));
      lodB.updateView.mockResolvedValue(makeLodData(50, 3, { color: 'float32' }));
      lodC.updateView.mockResolvedValue(makeLodData(25, 3, { color: 'float32' }));
      const result = await loader.loadPoints(baseViewState);
      expect(result.colors).toBeInstanceOf(Float32Array);
      expect(result.colors?.length).toBe(175 * 3);
    });

    it('rejects mixed color dtypes across LOD levels (ladder-dtype contract)', async () => {
      // TypedArray.set converts by VALUE, not semantics — a Uint8 level
      // (0..255) merged into a Float32 (0..1) output would silently write
      // 255× values. Malformed ladders fail fast instead of rendering
      // corruption (see concat-helpers.ts).
      lodA.updateView.mockResolvedValue(makeLodData(100, 3, { color: 'float32' }));
      lodB.updateView.mockResolvedValue(makeLodData(50, 3, { color: 'uint8' }));
      lodC.updateView.mockResolvedValue(makeLodData(25, 3, { color: 'float32' }));
      await expect(loader.loadPoints(baseViewState)).rejects.toThrow(/'colors' as Uint8Array/);
    });

    it('drops colors entirely when at least one LOD lacks them (all-or-nothing policy)', async () => {
      // PointsProgressiveLoader's all-or-nothing per-attr concatenation
      // policy: any LOD missing an optional attribute → the merged result
      // drops that attribute (no fill-with-default like gsplats).
      lodB.updateView.mockResolvedValue(makeLodData(50, 3)); // no colors
      const result = await loader.loadPoints(baseViewState);
      expect(result.colors).toBeUndefined();
    });

    it('keeps colors absent when no LOD has colors', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(100, 3));
      lodB.updateView.mockResolvedValue(makeLodData(50, 3));
      lodC.updateView.mockResolvedValue(makeLodData(25, 3));
      const result = await loader.loadPoints(baseViewState);
      expect(result.colors).toBeUndefined();
    });

    it('concatenates radii when all LODs carry them', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(100, 3, { withRadii: true }));
      lodB.updateView.mockResolvedValue(makeLodData(50, 3, { withRadii: true }));
      lodC.updateView.mockResolvedValue(makeLodData(25, 3, { withRadii: true }));
      const result = await loader.loadPoints(baseViewState);
      expect(result.radii).toBeInstanceOf(Float32Array);
      expect(result.radii?.length).toBe(175);
    });

    it('aggregates bounds across loaded LODs', async () => {
      const result = await loader.loadPoints(baseViewState);
      expect(result.metadata.bounds).toBeInstanceOf(THREE.Box3);
      expect(result.metadata.loadedPoints).toBe(175);
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
      await loader.loadPoints(baseViewState);
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
      loader = new PointsProgressiveLoader(
        [lodA, lodB] as unknown as PointsSpatialIndexLoader[],
        2,
        '/points'
      );

      const metrics = loader.getMetrics();
      expect(metrics.path).toBe('/points');
      expect(metrics.type).toBe('point-spatial-index');
      expect(metrics.queries).toBe(5); // 2 + 3
      expect(metrics.elementsLoaded).toBe(150); // 100 + 50
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

describe('PointsProgressiveLoader — concat memoization (no-op commit skip)', () => {
  let lodA: SubLoaderStub;
  let lodB: SubLoaderStub;
  let loader: PointsProgressiveLoader;

  beforeEach(() => {
    vi.useFakeTimers();
    lodA = makeSubLoader(makeLodData(100));
    lodB = makeSubLoader(makeLodData(50));
    loader = new PointsProgressiveLoader(
      [lodA, lodB] as unknown as PointsSpatialIndexLoader[],
      2,
      '/points'
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

describe('PointsProgressiveLoader — committedEnergyFraction (quality stamps)', () => {
  function make(table?: Array<number | null>) {
    const lodA = makeSubLoader(makeLodData(100));
    const lodB = makeSubLoader(makeLodData(50));
    return new PointsProgressiveLoader(
      [lodA, lodB] as unknown as PointsSpatialIndexLoader[],
      2,
      '/points',
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
    const l = new PointsProgressiveLoader(
      [slowA, fastB] as unknown as PointsSpatialIndexLoader[],
      2,
      '/dispose-race'
    );
    const pending = l.loadPoints(baseViewState);
    l.dispose(); // clears lodLoaders while level 0 is awaited
    release();
    await expect(pending).resolves.toBeDefined(); // pre-fix: TypeError
    // Level 1 was never touched after the dispose.
    expect(fastB.updateViewWithResidency).not.toHaveBeenCalled();
  });
});

describe('PointsProgressiveLoader — RGBA color layout (per-point opacity)', () => {
  // The additive-ladder color concat is strided by `colorComponents`
  // (3 = RGB, 4 = RGBA) — a hardcoded 3 would truncate + misalign an RGBA
  // ladder and drop `colorComponents` downstream (the gsplat colorK
  // lesson). Layout is a property of the dataset, uniform across its LODs;
  // a mismatch is malformed data and fails fast, naming the level.
  function makeRgbaLodData(pointCount: number, alpha: number): LoadedPointsData {
    const data = makeLodData(pointCount);
    const colors = new Float32Array(pointCount * 4);
    for (let i = 0; i < pointCount; i++) {
      colors[i * 4] = 0.2;
      colors[i * 4 + 1] = 0.4;
      colors[i * 4 + 2] = 0.6;
      colors[i * 4 + 3] = alpha;
    }
    data.colors = colors;
    data.colorComponents = 4;
    return data;
  }

  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('concatenates two RGBA parts at STRIDE 4 and reports colorComponents=4', async () => {
    const lodA = makeSubLoader(makeRgbaLodData(3, 0.9));
    const lodB = makeSubLoader(makeRgbaLodData(2, 0.3));
    const loader = new PointsProgressiveLoader(
      [lodA, lodB] as unknown as PointsSpatialIndexLoader[],
      2,
      '/points'
    );
    const result = await loader.loadPoints(baseViewState);
    expect(result.pointCount).toBe(5);
    expect(result.colorComponents).toBe(4);
    expect(result.colors!.length).toBe(5 * 4); // NOT 5 * 3 (a stride-3 concat)
    // Per-part alphas land intact and stride-aligned across the boundary.
    for (let i = 0; i < 3; i++) expect(result.colors![i * 4 + 3]).toBeCloseTo(0.9, 6);
    for (let i = 3; i < 5; i++) expect(result.colors![i * 4 + 3]).toBeCloseTo(0.3, 6);
    // RGB survives at the right offsets (not smeared by a wrong stride).
    expect(result.colors![4 * 4]).toBeCloseTo(0.2, 6);
    expect(result.colors![4 * 4 + 2]).toBeCloseTo(0.6, 6);
  });

  it('rejects a mixed RGB + RGBA ladder, naming the offending level', async () => {
    // An RGB and an RGBA level can share Float32Array, so the dtype check
    // cannot catch this — the layout gate must fail fast instead of
    // corrupting every point after the mismatched level.
    const lodA = makeSubLoader(makeRgbaLodData(3, 0.9));
    const lodB = makeSubLoader(makeLodData(2, 3, { color: 'float32' })); // RGB
    const loader = new PointsProgressiveLoader(
      [lodA, lodB] as unknown as PointsSpatialIndexLoader[],
      2,
      '/points'
    );
    await expect(loader.loadPoints(baseViewState)).rejects.toThrow(
      /mixed color layouts .*level 1: 3 vs 4 components/
    );
  });

  it('single-part passthrough keeps colorComponents=4', async () => {
    const lodA = makeSubLoader(makeRgbaLodData(4, 0.7));
    const loader = new PointsProgressiveLoader(
      [lodA] as unknown as PointsSpatialIndexLoader[],
      1,
      '/points'
    );
    const result = await loader.loadPoints(baseViewState);
    expect(result.colorComponents).toBe(4);
    expect(result.colors!.length).toBe(4 * 4);
  });

  it('an RGB ladder stays colorComponents=3', async () => {
    const lodA = makeSubLoader(makeLodData(3, 3, { color: 'float32' }));
    const lodB = makeSubLoader(makeLodData(2, 3, { color: 'float32' }));
    const loader = new PointsProgressiveLoader(
      [lodA, lodB] as unknown as PointsSpatialIndexLoader[],
      2,
      '/points'
    );
    const result = await loader.loadPoints(baseViewState);
    expect(result.colorComponents).toBe(3);
    expect(result.colors!.length).toBe(5 * 3);
  });
});

describe('buildSliceViewSig extend_to_all membership (shared helper)', () => {
  const dims4 = () =>
    [
      { name: 'x' },
      { name: 'y' },
      { name: 'z' },
      { name: 't', discrete: true, spatial: false, step: 1 },
    ] as unknown as PointsViewState['dimensions'];
  const view = (tol3: number) => ({
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, 0],
    tolerance: [0, 0, 0, tol3],
    dimensions: dims4(),
  });

  it('collapses ordinary ride-along churn but keys extend_to_all membership', () => {
    // Ordinary builder churn (0 at init vs 0.5 from navigation) must collide
    // (cache hit) — that was the whole point of nulling the ride-along slot.
    expect(buildSliceViewSig(view(0))).toBe(buildSliceViewSig(view(0.5)));
    // …but the extend_to_all sentinel is the one ride-along value the
    // worker's discrete-dim membership reads: it must NOT collide.
    expect(buildSliceViewSig(view(0.5))).not.toBe(buildSliceViewSig(view(1e10)));
    // Two above-threshold sentinels are the same query.
    expect(buildSliceViewSig(view(1e10))).toBe(buildSliceViewSig(view(2e10)));
  });
});

describe('PointsProgressiveLoader — elementIds without levelOffsets (issue #1421)', () => {
  // WITHOUT `levelOffsets` the parent node declares no union label CSR, so no
  // reader can key by any index this ladder could publish: each part is a
  // sub-LOD (`additive_<i>`) whose arrays live in their own on-disk index
  // space, while the labels a map would serve are ONE union CSR on the parent
  // (#1422) keyed by the concatenation of those spaces. The
  // single-part branch is the FIRST-PAINT state of every ladder, not an
  // "unladdered node", so passing `parts[0]` through verbatim would make
  // hover report an additive_0 index until a second level lands and then
  // silently switch to the raw slot. (WITH `levelOffsets` — the parent DOES
  // carry the union CSR — the maps are composed instead; see the #1439 block.)
  function labelledLod(pointCount: number, firstOnDisk: number): LoadedPointsData {
    const data = makeLodData(pointCount, 3, { color: 'uint8' });
    const ids = new Uint32Array(pointCount);
    for (let i = 0; i < pointCount; i++) ids[i] = firstOnDisk + i;
    data.elementIds = ids;
    return data;
  }

  it('strips the sub-LOD map on the single-part passthrough', async () => {
    const lod0 = labelledLod(10, 2048);
    const a = makeSubLoader(lod0);
    const loader = new PointsProgressiveLoader(
      [a] as unknown as PointsSpatialIndexLoader[],
      1,
      '/points'
    );
    const result = await loader.loadPoints(baseViewState);
    expect(result.pointCount).toBe(10);
    expect(result.elementIds).toBeUndefined();
    // Non-destructive: the strip is a copy, so the sub-LOD's OWN payload must
    // still carry its intact map (a `delete only.elementIds` refactor would
    // corrupt the level the accumulator still owns).
    expect(lod0.elementIds).toBeInstanceOf(Uint32Array);
    expect(lod0.elementIds!.length).toBe(10);
    expect(lod0.elementIds![0]).toBe(2048);
  });

  // FENCE, not evidence of the fix: the multi-part branch always built a fresh
  // literal that never named `elementIds`, so this passed before the change
  // too. It exists so a future "helpfully concatenate it" edit breaks a test.
  it('does not concatenate maps across levels (regression fence)', async () => {
    const a = makeSubLoader(labelledLod(10, 2048));
    const b = makeSubLoader(labelledLod(5, 4096));
    const loader = new PointsProgressiveLoader(
      [a, b] as unknown as PointsSpatialIndexLoader[],
      2,
      '/points'
    );
    const result = await loader.loadPoints(baseViewState);
    expect(result.pointCount).toBe(15);
    expect(result.elementIds).toBeUndefined();
  });
});

describe('PointsProgressiveLoader — ladder elementIds composition (issue #1439)', () => {
  // WITH `levelOffsets` the PARENT node carries one union label CSR keyed by
  // `additive_0 || additive_1 || …` (each level in its stored order), so every
  // level's map is composed into that space by adding the preceding levels'
  // ON-DISK counts. Constructor arg order: (loaders, nLods, path, energyTable,
  // sliceCache, levelOffsets).
  function labelledLod(pointCount: number, firstOnDisk: number): LoadedPointsData {
    const data = makeLodData(pointCount, 3, { color: 'uint8' });
    const ids = new Uint32Array(pointCount);
    for (let i = 0; i < pointCount; i++) ids[i] = firstOnDisk + i;
    data.elementIds = ids;
    return data;
  }

  /**
   * The shape the feature actually sees: a GAPPED map. Range loading and
   * effective-radius compaction both produce non-contiguous on-disk indices, so
   * a contiguous run cannot distinguish `base + ids[k]` from `base + ids[0] + k`.
   */
  function gappedLod(ids: number[]): LoadedPointsData {
    const data = makeLodData(ids.length, 3, { color: 'uint8' });
    data.elementIds = Uint32Array.from(ids);
    return data;
  }

  function makeLoader(
    parts: LoadedPointsData[],
    levelOffsets: number[] | null,
    nLods = parts.length
  ): PointsProgressiveLoader {
    return new PointsProgressiveLoader(
      parts.map((p) => makeSubLoader(p)) as unknown as PointsSpatialIndexLoader[],
      nLods,
      '/points',
      undefined,
      null,
      levelOffsets
    );
  }

  it('passes the single-part payload through UNCHANGED (level 0 sits at offset 0)', async () => {
    // Level 0's index space IS the parent CSR's, shifted by levelOffsets[0]=0 —
    // no strip, no copy.
    const lod0 = labelledLod(10, 2048);
    const result = await makeLoader([lod0], [0, 4096]).loadPoints(baseViewState);
    expect(result).toBe(lod0);
    expect(result.elementIds).toBeInstanceOf(Uint32Array);
    expect(Array.from(result.elementIds!.slice(0, 3))).toEqual([2048, 2049, 2050]);
  });

  it('offsets each GAPPED level map into the parent union CSR index space', async () => {
    // additive_0 stores 100 rows on disk (only 4 of them visible here), so
    // level 1's map must be shifted by 100 — not by the 4 loaded points. Both
    // maps are gapped (culled chunks / compacted points), which is the only
    // shape that pins per-slot indexing rather than a per-level base.
    const result = await makeLoader(
      [gappedLod([3, 9, 40, 41]), gappedLod([0, 17, 18])],
      [0, 100, 200]
    ).loadPoints(baseViewState);
    expect(result.pointCount).toBe(7);
    expect(result.elementIds).toBeInstanceOf(Uint32Array);
    expect(Array.from(result.elementIds!)).toEqual([3, 9, 40, 41, 100, 117, 118]);
  });

  it('composes a PARTIALLY loaded ladder (levelOffsets longer than the parts)', async () => {
    // Refinement stopped after level 1 (cache miss), so 2 of 3 levels are
    // resident: the extra trailing offset must simply go unused.
    const a = makeSubLoader(gappedLod([3, 9]));
    const b = makeSubLoader(gappedLod([2, 5]));
    const bData = gappedLod([2, 5]);
    b.updateViewWithResidency = vi.fn(async () => ({ data: bData, allResident: false }));
    const c = makeSubLoader(gappedLod([1]));
    const loader = new PointsProgressiveLoader(
      [a, b, c] as unknown as PointsSpatialIndexLoader[],
      3,
      '/points',
      undefined,
      null,
      [0, 100, 350, 400]
    );
    const result = await loader.loadPoints(baseViewState);
    expect(result.pointCount).toBe(4);
    expect(c.updateViewWithResidency).not.toHaveBeenCalled();
    expect(Array.from(result.elementIds!)).toEqual([3, 9, 102, 105]);
  });

  it('tolerates a level that loaded ZERO points', async () => {
    const empty = makeLodData(0, 3, { color: 'uint8' });
    empty.elementIds = new Uint32Array(0);
    const result = await makeLoader(
      [gappedLod([3, 9]), empty, gappedLod([4])],
      [0, 100, 350, 400]
    ).loadPoints(baseViewState);
    expect(result.pointCount).toBe(3);
    expect(Array.from(result.elementIds!)).toEqual([3, 9, 354]);
  });

  it('fills a level that published NO map with identity + offset', async () => {
    // Level 1 took the projection's identity fast path (one range at 0, no
    // compaction), so its slot k IS its level-space index.
    const plain = makeLodData(3, 3, { color: 'uint8' });
    const result = await makeLoader([labelledLod(2, 40), plain], [0, 100, 200]).loadPoints(
      baseViewState
    );
    expect(Array.from(result.elementIds!)).toEqual([40, 41, 100, 101, 102]);
  });

  it('still emits a map when a MAP-LESS level is only partly loaded', async () => {
    // No level published a map — level 0 loaded a single head range [0, 10) of
    // its 100 on-disk rows, which IS the projection's identity fast path — but
    // slot 10 is on-disk row 100, not 10. The ladder identity therefore does
    // NOT hold, and dropping the running-sum check would silently return no
    // map at all.
    const result = await makeLoader(
      [makeLodData(10, 3, { color: 'uint8' }), makeLodData(3, 3, { color: 'uint8' })],
      [0, 100, 200]
    ).loadPoints(baseViewState);
    expect(result.pointCount).toBe(13);
    expect(result.elementIds).toBeInstanceOf(Uint32Array);
    expect(Array.from(result.elementIds!.slice(9))).toEqual([9, 100, 101, 102]);
  });

  it('emits NO map when every resident level is complete and unculled (identity)', async () => {
    // No level published a map and each offset equals the running sum of the
    // LOADED counts ⇒ slot === union on-disk index; stay allocation-free.
    const result = await makeLoader(
      [makeLodData(10, 3, { color: 'uint8' }), makeLodData(5, 3, { color: 'uint8' })],
      [0, 10, 15]
    ).loadPoints(baseViewState);
    expect(result.pointCount).toBe(15);
    expect(result.elementIds).toBeUndefined();
  });

  it('fails closed when levelOffsets is shorter than the loaded levels', async () => {
    const result = await makeLoader([labelledLod(10, 5), labelledLod(4, 7)], [0]).loadPoints(
      baseViewState
    );
    expect(result.pointCount).toBe(14);
    expect(result.elementIds).toBeUndefined();
  });

  it('fails closed when a level publishes the wrong number of element ids', async () => {
    const bad = labelledLod(4, 7);
    bad.elementIds = new Uint32Array(3); // shorter than pointCount
    const result = await makeLoader([labelledLod(10, 5), bad], [0, 100, 200]).loadPoints(
      baseViewState
    );
    expect(result.pointCount).toBe(14);
    expect(result.elementIds).toBeUndefined();
  });

  it('fails closed on a single part whose map length disagrees with pointCount', async () => {
    const lod0 = labelledLod(10, 2048);
    lod0.elementIds = new Uint32Array(9);
    const result = await makeLoader([lod0], [0, 4096]).loadPoints(baseViewState);
    expect(result.elementIds).toBeUndefined();
    // Non-destructive: the sub-LOD's own payload keeps its (bogus) map.
    expect(lod0.elementIds!.length).toBe(9);
  });

  // `elementIdsUnavailable` — the projection wanted a map and could NOT build
  // one, so its slots are not on-disk indices. That is the opposite of the
  // identity, which is the other reason `elementIds` can be missing; reading it
  // as identity would compose a confident WRONG id inside the right level.
  it('fails closed when a level maps a slot PAST its own on-disk rows', async () => {
    // Level 1 owns union rows [100, 350). An id of 260 in ITS index space
    // composes to 360 — a real row that belongs to additive_2. Shifting it
    // would report a confident label from the wrong level, so the ladder
    // publishes no map at all instead.
    const result = await makeLoader(
      [gappedLod([3, 9]), gappedLod([2, 260]), gappedLod([4])],
      [0, 100, 350, 400]
    ).loadPoints(baseViewState);
    expect(result.pointCount).toBe(5);
    expect(result.elementIds).toBeUndefined();
  });

  it('fails closed when the SINGLE part maps a slot past level 0’s rows', async () => {
    // Same check on the first-paint state: level 0 owns [0, 100), so an id of
    // 2048 names an additive_1 row in the parent's union CSR.
    const lod0 = labelledLod(10, 2048);
    const result = await makeLoader([lod0], [0, 100]).loadPoints(baseViewState);
    expect(result).not.toBe(lod0);
    expect(result.elementIds).toBeUndefined();
    // Non-destructive, as everywhere else: the sub-LOD keeps its own map.
    expect(lod0.elementIds!.length).toBe(10);
  });

  it('accepts the LAST loaded level up to its closing bound', async () => {
    // The closing entry is what makes the final level checkable at all: 99 is
    // the last row level 1 owns, so it must compose (to 199), not fail.
    const result = await makeLoader([gappedLod([3]), gappedLod([99])], [0, 100, 200]).loadPoints(
      baseViewState
    );
    expect(Array.from(result.elementIds!)).toEqual([3, 199]);
  });

  it('fails closed when a MULTI-part level flags elementIdsUnavailable', async () => {
    const bailed = makeLodData(3, 3, { color: 'uint8' });
    bailed.elementIdsUnavailable = true;
    const result = await makeLoader([gappedLod([3, 9]), bailed], [0, 100, 200]).loadPoints(
      baseViewState
    );
    expect(result.pointCount).toBe(5);
    expect(result.elementIds).toBeUndefined();
  });

  it('lets an EMPTY level flag elementIdsUnavailable without vetoing the map', async () => {
    // A level culled to zero by the current slice writes nothing into the union
    // map, so its missing map cannot corrupt a slot — the other levels' maps
    // must still be composed.
    const empty = makeLodData(0, 3, { color: 'uint8' });
    empty.elementIdsUnavailable = true;
    const result = await makeLoader(
      [gappedLod([3, 9]), empty, gappedLod([4])],
      [0, 100, 350, 400]
    ).loadPoints(baseViewState);
    expect(result.pointCount).toBe(3);
    expect(Array.from(result.elementIds!)).toEqual([3, 9, 354]);
  });

  it('fails closed when the SINGLE part flags elementIdsUnavailable', async () => {
    // The map it does carry is in nobody's index space; passing the payload
    // through would publish it as if it were the parent's.
    const bailed = labelledLod(4, 60);
    bailed.elementIdsUnavailable = true;
    const result = await makeLoader([bailed], [0, 100]).loadPoints(baseViewState);
    expect(result).not.toBe(bailed);
    expect(result.elementIds).toBeUndefined();
  });

  it('does not burn the warn latch on an EMPTY single-part first paint', async () => {
    // An empty LOD 0 is the terminal state of any slice that culls everything —
    // ordinary, not a defect. It publishes no slots, so the single-part branch
    // exempts it: its only observable effect would be spending the
    // once-per-loader `_composeWarned` latch on a benign payload, permanently
    // swallowing a later genuine warning.
    const spy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      const empty = makeLodData(0, 3, { color: 'uint8' });
      empty.elementIdsUnavailable = true;
      const result = await makeLoader([empty], [0, 100]).loadPoints(baseViewState);
      expect(result.pointCount).toBe(0);
      expect(result.elementIds).toBeUndefined();
      expect(spy).not.toHaveBeenCalled();
    } finally {
      spy.mockRestore();
    }
  });

  it('warns at most ONCE per loader about a fail-closed composition', async () => {
    // The concat re-runs on every (generation, lodCount) miss and a view change
    // bumps the generation, so an unlatched warning would spam tens of lines a
    // second during dimension-animation playback.
    const spy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      const loader = makeLoader([labelledLod(10, 5), labelledLod(4, 7)], [0]);
      await loader.loadPoints(baseViewState);
      await loader.loadPoints({ ...baseViewState, slicePosition: [1, 0, 0, 0] });
      await loader.loadPoints({ ...baseViewState, slicePosition: [2, 0, 0, 0] });
      expect(spy).toHaveBeenCalledTimes(1);
    } finally {
      spy.mockRestore();
    }
  });
});
