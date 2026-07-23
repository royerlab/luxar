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

  describe('LOD 0 short-circuit', () => {
    it('stops after LOD 0 if it returns 0 segments', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(0, 0));
      await loader.loadLines(baseViewState);
      expect(lodA.updateView).toHaveBeenCalled();
      expect(lodB.updateView).not.toHaveBeenCalled();
      expect(lodC.updateView).not.toHaveBeenCalled();
    });
  });

  describe('S-cache restored EMPTY prefix (terminal re-derivation)', () => {
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

    it('does not re-stream higher LODs when a restored prefix has an empty LOD 0', async () => {
      // Scrub away from an empty slice (the departure store snapshots the
      // [empty-LOD0] 1-level ladder), then scrub back: the restore path
      // must RE-DERIVE the terminal flag from the restored prefix — the
      // level===0 empty check only fires for freshly LOADED levels, so
      // without re-derivation the loop resumes at startLevel=1 and
      // re-fetches every higher (equally empty) LOD on every revisit.
      const sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
      const a = makeSubLoader(makeLodData(0, 0));
      const b = makeSubLoader(makeLodData(10, 5, 3, { color: 'uint8' }));
      const l = new LinesProgressiveLoader(
        [a, b] as unknown as LinesSpatialIndexLoader[],
        2,
        '/l-empty',
        undefined,
        sc
      );

      await l.loadLines(viewA); // empty LOD 0 → terminal 1-level ladder
      expect(l.hasMoreLODs).toBe(false);
      // The DISCOVERY pass itself must not prefetch the (empty) next LOD —
      // prefetch goes through prefetchChunks, a separate surface from
      // updateViewWithResidency, so pin it explicitly.
      expect(b.prefetchChunks).not.toHaveBeenCalled();
      await l.loadLines(viewB); // departure: stores viewA's ladder

      a.updateViewWithResidency.mockClear();
      b.updateViewWithResidency.mockClear();
      await l.loadLines(viewA); // revisit the empty slice

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

    it('a differing budget with an identical view does NOT reset the ladder', async () => {
      await loader.updateView({ ...baseViewState, frameBudgetMs: 10 });
      expect(loader.loadedLODCount).toBe(1);

      await loader.updateView({ ...baseViewState, frameBudgetMs: 70 });
      // No reset: level 0 not reloaded. Playback stays at the floor — deepening
      // is the background prefetch's job, not the foreground tick's.
      expect(lodA.updateViewWithResidency).toHaveBeenCalledTimes(1);
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

      // Loop-2 PLAYBACK tick at A: restores the DEEPER prefix(2) and commits it
      // with ZERO streaming (the floor gate blocks further foreground decode).
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
