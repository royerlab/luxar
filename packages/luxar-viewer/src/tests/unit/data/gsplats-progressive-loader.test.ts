/**
 * Unit tests for data/gsplats/gsplats-progressive-loader.ts.
 *
 * Pure orchestration: the composite loader wraps N
 * GSplatsSpatialIndexLoader sub-loaders and decides how many LODs
 * to load per frame based on cache-hit timing. Tests stub the
 * sub-loaders and inspect the merged result.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { GSplatsProgressiveLoader } from '../../../data/gsplats/gsplats-progressive-loader';
import type { GSplatsSpatialIndexLoader } from '../../../data/gsplats/gsplats-spatial-index-loader';
import type { GSplatsViewState, LoadedGSplatsData } from '../../../types/gsplats';

interface SubLoaderStub {
  updateView: ReturnType<typeof vi.fn>;
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
  return {
    updateView: vi.fn().mockResolvedValue(initialData),
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
  slicePosition: [0, 0, 0],
  tolerance: [0, 0, 0],
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
    it('stops loading further LODs when one takes > 15ms', async () => {
      // Make LOD B slow (~25ms) so LOD C is deferred.
      let now = 0;
      const performanceNowSpy = vi.spyOn(performance, 'now');
      // sequence: t0_lodA, t1_lodA, t0_lodB, t1_lodB, t0_lodC, t1_lodC
      performanceNowSpy.mockImplementation(() => {
        now += 5;
        return now;
      });

      // LOD B's updateView "takes" 25ms — toggle the now stride.
      lodB.updateView.mockImplementation(async () => {
        now += 25; // simulated work
        return makeLodData(50);
      });

      await loader.loadGSplats(baseViewState);

      // LOD A loaded (always); LOD B loaded but slow → LOD C deferred.
      expect(lodA.updateView).toHaveBeenCalled();
      expect(lodB.updateView).toHaveBeenCalled();
      expect(lodC.updateView).not.toHaveBeenCalled();

      performanceNowSpy.mockRestore();
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
      lodA = makeSubLoader(makeLodData(100), { queries: 2, pointsLoaded: 100, memoryUsed: 10 });
      lodB = makeSubLoader(makeLodData(50), { queries: 3, pointsLoaded: 50, memoryUsed: 20 });
      loader = new GSplatsProgressiveLoader(
        [lodA, lodB] as unknown as GSplatsSpatialIndexLoader[],
        2,
        '/test_gsplats'
      );

      const metrics = loader.getMetrics();
      expect(metrics.path).toBe('/test_gsplats');
      expect(metrics.type).toBe('gsplats-spatial-index');
      expect(metrics.queries).toBe(5); // 2 + 3
      expect(metrics.pointsLoaded).toBe(150); // 100 + 50
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
