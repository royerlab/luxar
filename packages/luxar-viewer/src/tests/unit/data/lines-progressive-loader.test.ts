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
    type: 'lines-spatial-index' as const,
    path: '/lines/additive_x',
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
  vertexCount: number,
  segmentCount: number,
  ndim = 3,
  options: {
    color?: 'none' | 'uint8' | 'uint16' | 'float32';
    /** Widen colors to 4 channels (per-vertex alpha column). */
    rgba?: boolean;
    sharpness?: boolean;
    scalars?: boolean;
    /**
     * Per-level marker written into colors/scalars so a merged ladder's rows
     * can be traced back to the level they came from. Omitted (the default)
     * keeps each field's historical fill value.
     */
    mark?: number;
  } = {}
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
  const colorK = options.rgba ? 4 : 3;
  const mark = options.mark;
  let colors: Float32Array | Uint8Array | Uint16Array | null = null;
  if (options.color === 'uint8') {
    colors = new Uint8Array(vertexCount * colorK).fill(mark ?? 128);
  } else if (options.color === 'uint16') {
    colors = new Uint16Array(vertexCount * colorK).fill(mark ?? 32000);
  } else if (options.color === 'float32') {
    colors = new Float32Array(vertexCount * colorK).fill(mark ?? 0.5);
  }
  if (colors && options.rgba) {
    // Distinct per-vertex alphas so stride slips are detectable.
    for (let i = 0; i < vertexCount; i++) {
      colors[i * 4 + 3] = colors instanceof Float32Array ? 0.25 : 64;
    }
  }
  const sharpness = options.sharpness ? new Float32Array(vertexCount).fill(1) : null;
  return {
    positions,
    segments,
    widths,
    colors,
    ...(colors && options.rgba ? { colorComponents: 4 as const } : {}),
    sharpness,
    ...(options.scalars ? { scalars: new Float32Array(vertexCount).fill(mark ?? 1) } : {}),
    segmentCount,
    vertexCount,
    ndim,
  };
}

/**
 * A level the current slice culls to zero, shaped like the real thing:
 * `createEmptyLinesData` (lines/projection.ts) OMITS `scalars` entirely
 * (absent, not zero-length) — the asymmetry that made an empty level strip the
 * merged ladder's scalars and suppress the colormap node-wide (#1456).
 */
function makeEmptyLodData(): LoadedLinesData {
  return makeLodData(0, 0);
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
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 0],
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
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    };
    const viewB: LinesViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 1],
      tolerance: [0, 0, 0, 0],
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

  describe('empty LOD levels (#1456)', () => {
    it('streams PAST an empty LOD 0 to the levels that DO have segments', async () => {
      // THE regression. This loader exists only for ADDITIVE ladders, whose
      // levels are disjoint increments of one permutation — LOD 0 is a small
      // SUBSET of the node, so a hidden-dimension slice that none of ITS
      // vertices lands on is ordinary and says nothing about levels 1..n-1.
      // The loader used to latch a terminal "empty ladder" on an empty LOD 0
      // and break, so such a slice rendered NOTHING even though the higher
      // levels held plenty of geometry right there.
      lodA.updateView.mockResolvedValue(makeEmptyLodData());
      const result = await loader.loadLines(baseViewState);
      expect(lodB.updateView).toHaveBeenCalled();
      expect(lodC.updateView).toHaveBeenCalled();
      expect(result.segmentCount).toBeGreaterThan(0);
    });

    it("keeps the surviving levels' scalars across an empty LOD 0", async () => {
      // An empty level is an ABSTAINER, not a veto. `createEmptyLinesData`
      // OMITS `scalars`, and `concatOptionalField` is all-or-nothing, so
      // without the abstainer rule one empty level stripped the merged
      // ladder's scalars — `data-processor-lines.ts`'s
      // `hasScalars = !!data.scalars` then flips false and the colormap is
      // suppressed node-wide. (`colors` and `sharpness` were already safe:
      // they use the find-first + fill-for-missing pattern, not the
      // all-or-nothing helper. Pinned below so a white fill can never be
      // mistaken for the real thing.)
      const attrs = { color: 'uint8' as const, scalars: true };
      lodA.updateView.mockResolvedValue(makeEmptyLodData());
      lodB.updateView.mockResolvedValue(makeLodData(10, 5, 3, { ...attrs, mark: 11 }));
      lodC.updateView.mockResolvedValue(makeLodData(4, 2, 3, { ...attrs, mark: 22 }));

      const r = await loader.loadLines(baseViewState);

      expect(r.vertexCount).toBe(14); // 0 + 10 + 4
      expect(r.scalars).toHaveLength(14);
      expect(r.scalars![0]).toBeCloseTo(11);
      expect(r.scalars![9]).toBeCloseTo(11);
      expect(r.scalars![10]).toBeCloseTo(22);
      expect(r.colors).toBeInstanceOf(Uint8Array);
      expect(r.colors).toHaveLength(14 * 3);
      expect(Array.from(r.colors!.slice(0, 3))).toEqual([11, 11, 11]);
      expect(Array.from(r.colors!.slice(10 * 3, 10 * 3 + 3))).toEqual([22, 22, 22]);
    });

    it('keeps scalars when the empty level lands in the MIDDLE of the ladder', async () => {
      // Same defect, and this half of it predates #1456: the old short-circuit
      // only fired at level 0, so a ladder whose level 1 was culled to zero
      // already lost its scalars.
      const attrs = { color: 'uint8' as const, scalars: true };
      lodA.updateView.mockResolvedValue(makeLodData(20, 10, 3, { ...attrs, mark: 11 }));
      lodB.updateView.mockResolvedValue(makeEmptyLodData());
      lodC.updateView.mockResolvedValue(makeLodData(4, 2, 3, { ...attrs, mark: 22 }));

      const r = await loader.loadLines(baseViewState);

      expect(r.vertexCount).toBe(24); // 20 + 0 + 4
      expect(r.scalars).toHaveLength(24);
      expect(r.scalars![0]).toBeCloseTo(11);
      // The empty level occupies no vertices, so level 2 starts right after
      // level 0 — a stray zero-row offset would show up here.
      expect(r.scalars![20]).toBeCloseTo(22);
      expect(Array.from(r.colors!.slice(20 * 3, 20 * 3 + 3))).toEqual([22, 22, 22]);
    });

    it('keeps hasMoreLODs true after an empty LOD 0 while levels remain unloaded', async () => {
      // Level 1 is a cache miss, so the refine pass stops with 2 of 3 levels
      // loaded. An empty LOD 0 must not be mistaken for a finished ladder:
      // refinement has to stay scheduled, and the next pass resumes at level 2.
      lodA.updateView.mockResolvedValue(makeLodData(0, 0));
      lodB.updateViewWithResidency.mockImplementation(async () => ({
        data: makeLodData(10, 5, 3, { color: 'uint8' }),
        allResident: false,
      }));
      await loader.loadLines(baseViewState);
      expect(loader.loadedLODCount).toBe(2);
      expect(loader.hasMoreLODs).toBe(true);

      await loader.loadLines(baseViewState);
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
        // that actually holds this slice's geometry is the one still to come,
        // so skipping prefetch would leave it cold.
        lodA.updateView.mockResolvedValue(makeLodData(0, 0));
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

      const result = await loader.loadLines(baseViewState);

      expect(lodA.updateView).toHaveBeenCalled();
      expect(lodB.updateView).toHaveBeenCalled();
      expect(lodC.updateView).toHaveBeenCalled();
      expect(result.segmentCount).toBe(0);
      expect(loader.loadedLODCount).toBe(3);
      expect(loader.hasMoreLODs).toBe(false);
    });
  });

  describe('S-cache restored ladders', () => {
    const viewA: LinesViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    };
    const viewB: LinesViewState = {
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
      const a = makeSubLoader(makeLodData(0, 0));
      const b = makeSubLoader(makeLodData(10, 5, 3, { color: 'uint8' }));
      const l = new LinesProgressiveLoader(
        [a, b] as unknown as LinesSpatialIndexLoader[],
        2,
        '/l-empty-prefix',
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
      const restored = await l.loadLines(viewA); // scrub back, no budget

      expect(a.updateViewWithResidency).not.toHaveBeenCalled(); // served from the S-cache
      expect(b.updateViewWithResidency).toHaveBeenCalled(); // …and streamed on
      expect(restored.segmentCount).toBe(5);
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
      const l = new LinesProgressiveLoader(
        [a, b] as unknown as LinesSpatialIndexLoader[],
        2,
        '/l-empty',
        undefined,
        sc
      );

      await l.loadLines(viewA); // both levels empty → complete ladder, stored
      expect(l.loadedLODCount).toBe(2);
      await l.loadLines(viewB); // move away

      a.updateViewWithResidency.mockClear();
      b.updateViewWithResidency.mockClear();
      const revisited = await l.loadLines(viewA); // revisit the empty slice

      expect(a.updateViewWithResidency).not.toHaveBeenCalled();
      expect(b.updateViewWithResidency).not.toHaveBeenCalled();
      expect(revisited.segmentCount).toBe(0);
      expect(l.hasMoreLODs).toBe(false);
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
        return { data: makeLodData(10, 5, 3, { color: 'uint8' }), allResident: false };
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

    it('background prefetch deepens a capped ladder loop-over-loop; playback restores it responsively', async () => {
      // Mirrors gsplats-progressive-loader.test.ts (three-geometry symmetry).
      // Foreground PLAYBACK ticks commit only the cached prefix (never block on
      // fine levels); background PREFETCH passes deepen the SAME slice +1 level,
      // so a later playback tick restores a DEEPER prefix — higher quality,
      // still instant.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const l = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/l',
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
      const l = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/l',
        undefined,
        sc
      );
      const viewA = baseViewState;
      const viewB = { ...baseViewState, slicePosition: [0, 0, 0, 1] };

      await l.updateView({ ...viewA, frameBudgetMs: 10 });
      await l.updateView({ ...viewB, frameBudgetMs: 10 });

      const key = SliceCache.makeKey('/l', buildSliceViewSig(viewA));
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
      const l = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/l',
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
      const l = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/l',
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
      // re-stream) and commits it responsively.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const shadow = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/l',
        undefined,
        sc
      );
      const foreground = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/l',
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
        expect(result.segmentCount).toBe(10);
      } finally {
        nowSpy.mockRestore();
      }
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

    it('rejects mixed color dtypes across LOD levels (ladder-dtype contract)', async () => {
      // TypedArray.set converts by VALUE, not semantics — a Float32 level
      // (0..1) merged into a Uint8 (0..255) output truncates to garbage.
      // Malformed ladders fail fast instead of rendering corruption.
      lodA = makeSubLoader(makeLodData(20, 10, 3, { color: 'uint8' }));
      lodB = makeSubLoader(makeLodData(10, 5, 3, { color: 'float32' }));
      lodC = makeSubLoader(makeLodData(4, 2, 3, { color: 'uint8' }));
      loader = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/lines'
      );
      await expect(loader.loadLines(baseViewState)).rejects.toThrow(/mixed color dtypes/);
    });

    it('rejects mixed dimensionality across LOD levels (ndim strides the concat)', async () => {
      // Same fail-fast family as the dtype check: ndim strides the position
      // concat, so sub-LODs disagreeing on it would silently mis-stride
      // every vertex after the first part.
      lodA = makeSubLoader(makeLodData(20, 10, 3));
      lodB = makeSubLoader(makeLodData(10, 5, 4));
      loader = new LinesProgressiveLoader(
        [lodA, lodB] as unknown as LinesSpatialIndexLoader[],
        2,
        '/lines'
      );
      await expect(loader.loadLines(baseViewState)).rejects.toThrow(
        /mixed dimensionality .*ndim 4 vs 3/
      );
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

    it('fills missing-part sharpness with the 0.5 default (mixed-sharpness ladder)', async () => {
      // The worker projection substitutes 0.5 (beta=2, Gaussian) for a NULL
      // sharpness array — the concat must fill the same value for parts
      // lacking sharpness, not 0.0 (razor-sharp). This keeps the merged
      // prefix byte-identical to what each part renders standalone (the
      // append fast path's prefix-identity contract) and stops
      // sharpness-less parts from popping sharp when a sharpness-carrying
      // level joins the ladder. Mirrors the white color fill.
      lodA = makeSubLoader(makeLodData(20, 10, 3)); // no sharpness
      lodB = makeSubLoader(makeLodData(10, 5, 3, { sharpness: true })); // fill(1)
      lodC = makeSubLoader(makeLodData(4, 2, 3)); // no sharpness
      loader = new LinesProgressiveLoader(
        [lodA, lodB, lodC] as unknown as LinesSpatialIndexLoader[],
        3,
        '/lines'
      );
      const result = await loader.loadLines(baseViewState);
      expect(result.sharpness).toBeInstanceOf(Float32Array);
      // LOD A verts 0..19 → default fill; LOD B verts 20..29 → its own 1.0;
      // LOD C verts 30..33 → default fill.
      expect(result.sharpness?.[0]).toBeCloseTo(0.5, 6);
      expect(result.sharpness?.[19]).toBeCloseTo(0.5, 6);
      expect(result.sharpness?.[20]).toBeCloseTo(1.0, 6);
      expect(result.sharpness?.[30]).toBeCloseTo(0.5, 6);
      expect(result.sharpness?.[33]).toBeCloseTo(0.5, 6);
    });
  });

  describe('prefix lineage (Phase 4 Stage 2 append)', () => {
    it('forward-chains each concat result to the previous same-generation result', async () => {
      // Call 1: LOD 1 not resident → the loop stops after LOD 0 (partial ladder).
      lodB.updateViewWithResidency.mockResolvedValueOnce({
        data: makeLodData(10, 5, 3, { color: 'uint8' }),
        allResident: false,
      });
      const r1 = await loader.loadLines(baseViewState);
      // First concat of the generation extends nothing.
      expect(getPrefixParent(r1)).toBeUndefined();
      expect(loader.hasMoreLODs).toBe(true);

      // Call 2 (SAME view): refinement loads the remaining levels → a new,
      // longer concat that forward-chains to r1.
      const r2 = await loader.loadLines(baseViewState);
      expect(r2).not.toBe(r1);
      expect(r2.segmentCount).toBeGreaterThan(r1.segmentCount);
      expect(getPrefixParent(r2)).toBe(r1);
    });

    it('drops lineage across a view change (new generation → full rewrite)', async () => {
      await loader.loadLines(baseViewState);
      // A slicePosition change resets the ladder + bumps the generation, so the
      // first concat of the new generation has no parent → append gate rejects.
      const r2 = await loader.loadLines({ ...baseViewState, slicePosition: [0, 0, 0, 1] });
      expect(getPrefixParent(r2)).toBeUndefined();
    });

    it('a memoized no-op re-commit keeps the SAME reference (its lineage is unchanged)', async () => {
      const r1 = await loader.loadLines(baseViewState); // full ladder in one call
      expect(loader.hasMoreLODs).toBe(false);
      const r2 = await loader.loadLines(baseViewState); // no new LODs → memoized
      expect(r2).toBe(r1); // same reference → commit takes the stamp-only no-op
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
      lodA = makeSubLoader(makeLodData(20, 10), {
        queries: 2,
        elementsLoaded: 100,
        memoryUsed: 10,
      });
      lodB = makeSubLoader(makeLodData(10, 5), { queries: 3, elementsLoaded: 50, memoryUsed: 20 });
      loader = new LinesProgressiveLoader(
        [lodA, lodB] as unknown as LinesSpatialIndexLoader[],
        2,
        '/lines'
      );

      const metrics = loader.getMetrics();
      expect(metrics.path).toBe('/lines');
      expect(metrics.type).toBe('lines-spatial-index');
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
    const slowA = makeSubLoader(makeLodData(10, 5, 3));
    const origA = slowA.updateViewWithResidency as unknown as (
      vs: unknown,
      s?: unknown
    ) => Promise<unknown>;
    slowA.updateViewWithResidency = vi.fn(async (vs: unknown, s?: unknown) => {
      const out = await origA(vs, s);
      await gate; // hold level 0 in flight
      return out;
    });
    const fastB = makeSubLoader(makeLodData(6, 3, 3));
    const l = new LinesProgressiveLoader(
      [slowA, fastB] as unknown as LinesSpatialIndexLoader[],
      2,
      '/dispose-race'
    );
    const pending = l.loadLines(baseViewState);
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
    // regression in the LinesProgressiveLoader wiring can't hide behind the shared helper's
    // own unit tests.
    const dimsV1 = [
      { name: 'x', unit: 'units', scale: 1, range: null, display: true, step: null },
      { name: 'y', unit: 'units', scale: 1, range: null, display: true, step: null },
      { name: 'z', unit: 'units', scale: 1, range: null, display: true, step: null },
    ] as unknown as LinesViewState['dimensions'];
    const dimsV2 = [
      { name: 'x', unit: 'units', scale: 1, display: true, step: 1, range: [0, 9] },
      { name: 'y', unit: 'units', scale: 1, display: true, step: 1, range: [0, 9] },
      { name: 'z', unit: 'units', scale: 1, display: true, step: 1, range: [0, 9] },
    ] as unknown as LinesViewState['dimensions'];
    const lod0 = makeSubLoader(makeLodData(20, 10, 3));
    const l = new LinesProgressiveLoader(
      [lod0] as unknown as LinesSpatialIndexLoader[],
      1,
      '/dims-refresh'
    );
    await l.loadLines({ ...baseViewState, dimensions: dimsV1 });
    lod0.updateView.mockClear();

    // Refresh: determinant-equal, different reference → no reload.
    await l.loadLines({ ...baseViewState, dimensions: dimsV2 });
    expect(lod0.updateView).not.toHaveBeenCalled();
    // Same refreshed reference again → still no reload (ref fast path).
    await l.loadLines({ ...baseViewState, dimensions: dimsV2 });
    expect(lod0.updateView).not.toHaveBeenCalled();

    // A genuine query change still resets.
    await l.loadLines({
      ...baseViewState,
      slicePosition: [1, 1, 1, 0],
      dimensions: dimsV2,
    });
    expect(lod0.updateView).toHaveBeenCalled();
  });
});

describe('LinesProgressiveLoader — RGBA color layout (colorK stride, volumetric phase 4)', () => {
  it('concatenates an RGBA ladder at stride 4 and stamps colorComponents=4', async () => {
    const lodA = makeSubLoader(makeLodData(20, 10, 3, { color: 'float32', rgba: true }));
    const lodB = makeSubLoader(makeLodData(10, 5, 3, { color: 'float32', rgba: true }));
    const loader = new LinesProgressiveLoader(
      [lodA, lodB] as unknown as LinesSpatialIndexLoader[],
      2,
      '/lines'
    );
    const result = await loader.loadLines(baseViewState);
    expect(result.colorComponents).toBe(4);
    expect(result.colors?.length).toBe(30 * 4);
    // LOD B's first vertex is global vertex 20 — its alpha must land at
    // 20*4+3 (a stride-3 concat would smear every vertex after level A).
    expect(result.colors?.[20 * 4 + 3]).toBeCloseTo(0.25, 6);
    expect(result.colors?.[19 * 4 + 3]).toBeCloseTo(0.25, 6);
  });

  it('rejects mixed RGB/RGBA layouts across LOD levels (same dtype, different stride)', async () => {
    // The dtype guard alone would MISS this — both levels are Float32 —
    // while the concat stride would silently corrupt every vertex after
    // the offending level (the gsplat ladder shipped exactly this bug
    // before its colorK guard).
    const lodA = makeSubLoader(makeLodData(20, 10, 3, { color: 'float32', rgba: true }));
    const lodB = makeSubLoader(makeLodData(10, 5, 3, { color: 'float32' })); // RGB
    const loader = new LinesProgressiveLoader(
      [lodA, lodB] as unknown as LinesSpatialIndexLoader[],
      2,
      '/lines'
    );
    await expect(loader.loadLines(baseViewState)).rejects.toThrow(/mixed color layouts/);
  });

  it('white-fills a missing-colors level at the RGBA stride with opaque alpha', async () => {
    const lodA = makeSubLoader(makeLodData(20, 10, 3, { color: 'float32', rgba: true }));
    const lodB = makeSubLoader(makeLodData(10, 5, 3)); // no colors
    const loader = new LinesProgressiveLoader(
      [lodA, lodB] as unknown as LinesSpatialIndexLoader[],
      2,
      '/lines'
    );
    const result = await loader.loadLines(baseViewState);
    expect(result.colorComponents).toBe(4);
    // LOD B verts 20..29 → white RGB + the 1.0 opaque alpha identity.
    expect(result.colors?.[20 * 4]).toBeCloseTo(1.0, 6);
    expect(result.colors?.[20 * 4 + 3]).toBeCloseTo(1.0, 6);
  });

  it('rejects a single part whose RGBA buffer omits colorComponents (layout omission)', async () => {
    // Distinct from the cross-level mismatch above: here both parts AGREE on
    // colorComponents (all default to 3) but one carries a 4-wide (RGBA)
    // buffer without declaring it. That satisfies the downstream count·3
    // minimum yet mis-strides every vertex after the first — silently-wrong
    // colors/alpha. The per-part layout check must throw at concat time.
    const rgbaUndeclared: LoadedLinesData = {
      positions: new Float32Array(2 * 3).fill(0.5),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array(2).fill(1),
      colors: new Float32Array(2 * 4).fill(0.5), // RGBA length, but…
      // …colorComponents OMITTED → defaults to 3 → 2×3=6 ≠ 8.
      sharpness: null,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };
    const lodA = makeSubLoader(rgbaUndeclared);
    const lodB = makeSubLoader(makeLodData(10, 5, 3, { color: 'float32' }));
    const loader = new LinesProgressiveLoader(
      [lodA, lodB] as unknown as LinesSpatialIndexLoader[],
      2,
      '/lines'
    );
    await expect(loader.loadLines(baseViewState)).rejects.toThrow(
      /LOD level 0\): colors length 8 does not match count 2/
    );
  });

  it('accepts a correctly-declared RGBA ladder (colorComponents: 4, no false positive)', async () => {
    // The per-part check is a no-op when the declared layout matches the
    // buffer length — a properly-declared RGBA ladder still concatenates.
    const lodA = makeSubLoader(makeLodData(20, 10, 3, { color: 'float32', rgba: true }));
    const lodB = makeSubLoader(makeLodData(10, 5, 3, { color: 'float32', rgba: true }));
    const loader = new LinesProgressiveLoader(
      [lodA, lodB] as unknown as LinesSpatialIndexLoader[],
      2,
      '/lines'
    );
    const result = await loader.loadLines(baseViewState);
    expect(result.colorComponents).toBe(4);
    expect(result.colors?.length).toBe(30 * 4);
  });
});

describe('LinesProgressiveLoader — vertexRangeBounds are never published (issue #1424)', () => {
  // A ladder payload must NEVER carry the on-disk VERTEX range bounds: each part is a
  // sub-LOD (`additive_<i>`) whose vertices live in their own on-disk space, while the
  // labels those bounds would serve are ONE per-vertex union CSR on the parent
  // (#1422) keyed by the concatenation of those spaces. The single-part
  // branch is the FIRST-PAINT state of every ladder, not an "unladdered node",
  // so passing `parts[0]` through verbatim would make hover report an
  // additive_0 vertex row until a second level lands and then silently switch
  // to the raw slot. Mirrors the points twin.
  function labelledLod(vertexCount: number, segmentCount: number, firstOnDisk: number) {
    const data = makeLodData(vertexCount, segmentCount, 3, { color: 'uint8' });
    data.vertexRangeBounds = new Uint32Array([firstOnDisk, firstOnDisk + vertexCount]);
    return data;
  }

  it('strips the sub-LOD ranges on the single-part passthrough', async () => {
    const lod0 = labelledLod(10, 5, 2048);
    const a = makeSubLoader(lod0);
    const loader = new LinesProgressiveLoader(
      [a] as unknown as LinesSpatialIndexLoader[],
      1,
      '/lines'
    );
    const result = await loader.loadLines(baseViewState);
    expect(result.vertexCount).toBe(10);
    expect(result.vertexRangeBounds).toBeUndefined();
    // Non-destructive: the strip is a copy, so the sub-LOD's OWN payload must
    // still carry its intact ranges (a `delete only.vertexRangeBounds` refactor
    // would corrupt the level the accumulator still owns).
    expect(Array.from(lod0.vertexRangeBounds!)).toEqual([2048, 2058]);
  });

  // FENCE, not evidence of the fix: the multi-part branch always built a fresh
  // literal that never named `vertexRangeBounds`, so this passed before the change
  // too. It exists so a future "helpfully concatenate them" edit breaks a test.
  it('does not concatenate ranges across levels (regression fence)', async () => {
    const a = makeSubLoader(labelledLod(20, 10, 2048));
    const b = makeSubLoader(labelledLod(10, 5, 4096));
    const loader = new LinesProgressiveLoader(
      [a, b] as unknown as LinesSpatialIndexLoader[],
      2,
      '/lines'
    );
    const result = await loader.loadLines(baseViewState);
    expect(result.vertexCount).toBe(30);
    expect(result.vertexRangeBounds).toBeUndefined();
  });
});
