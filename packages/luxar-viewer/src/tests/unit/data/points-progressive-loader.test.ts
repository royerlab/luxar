/**
 * Unit tests for data/points/points-progressive-loader.ts.
 *
 * Mirrors `gsplats-progressive-loader.test.ts`. Pure orchestration:
 * the composite loader wraps N PointsSpatialIndexLoader sub-loaders
 * and decides how many LODs to load per frame based on cache-hit
 * timing. Tests stub the sub-loaders and inspect the merged result.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import { PointsProgressiveLoader } from '../../../data/points/points-progressive-loader';
import type { PointsSpatialIndexLoader } from '../../../data/points/points-spatial-index-loader';
import type { LoadedPointsData, PointsViewState } from '../../../types/points';
import { CACHE_HIT_THRESHOLD_MS } from '../../../data/loaders/progressive/constants';

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
  pointCount: number,
  ndim = 3,
  options: { color?: 'none' | 'uint8' | 'uint16' | 'float32'; withRadii?: boolean } = {}
): LoadedPointsData {
  const positions = new Float32Array(pointCount * ndim);
  positions.fill(0.5);
  let colors: Float32Array | Uint8Array | Uint16Array | undefined;
  if (options.color === 'uint8') {
    colors = new Uint8Array(pointCount * 3).fill(128);
  } else if (options.color === 'uint16') {
    colors = new Uint16Array(pointCount * 3).fill(32000);
  } else if (options.color === 'float32') {
    colors = new Float32Array(pointCount * 3).fill(0.5);
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
    result.radii = new Float32Array(pointCount).fill(0.1);
  }
  return result;
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
  slicePosition: [0, 0, 0],
  tolerance: [0, 0, 0],
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

  describe('LOD 0 short-circuit', () => {
    it('stops after LOD 0 if it returns 0 points', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(0));
      await loader.loadPoints(baseViewState);
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

  describe('concatenation', () => {
    it('returns a single part as-is when only one LOD is loaded', async () => {
      lodA.updateView.mockResolvedValue(makeLodData(0));
      const result = await loader.loadPoints(baseViewState);
      expect(result.pointCount).toBe(0);
    });

    it('concatenates positions across multiple LODs', async () => {
      const result = await loader.loadPoints(baseViewState);
      expect(result.positions.length).toBe(175 * 3);
      expect(result.pointCount).toBe(175);
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
      lodA = makeSubLoader(makeLodData(100), { queries: 2, pointsLoaded: 100, memoryUsed: 10 });
      lodB = makeSubLoader(makeLodData(50), { queries: 3, pointsLoaded: 50, memoryUsed: 20 });
      loader = new PointsProgressiveLoader(
        [lodA, lodB] as unknown as PointsSpatialIndexLoader[],
        2,
        '/points'
      );

      const metrics = loader.getMetrics();
      expect(metrics.path).toBe('/points');
      expect(metrics.type).toBe('point-spatial-index');
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
