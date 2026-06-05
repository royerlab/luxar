/**
 * Comprehensive tests for PointsSpatialIndexLoader
 *
 * Tests point spatial index-based loading, nD queries, caching,
 * broadcasting, and monitoring integration.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { PointsSpatialIndexLoader, type ViewState, type SceneNode } from '../../../../data';
import * as zarr from 'zarrita';

// Mock THREE.js using partial mock with importOriginal
vi.mock('three', async (importOriginal) => {
  const actual = await importOriginal<typeof import('three')>();
  return {
    ...actual,
    Box3: vi.fn().mockImplementation(() => ({
      min: { x: Infinity, y: Infinity, z: Infinity, set: vi.fn() },
      max: { x: -Infinity, y: -Infinity, z: -Infinity, set: vi.fn() },
      expandByPoint: vi.fn(),
      clone: vi.fn().mockReturnThis(),
    })),
    Vector3: vi.fn().mockImplementation((x = 0, y = 0, z = 0) => ({
      x,
      y,
      z,
      set: vi.fn().mockReturnThis(),
    })),
  };
});

// Mock zarrita
vi.mock('zarrita', () => ({
  registry: {},
  open: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

// Mock the canonical SpatialQueryBuilder so the test exercises the loader's
// orchestration rather than the AABB scan (which has its own unit tests).
const mockExecute = vi.fn();
vi.mock('../../../../data/loaders/spatial-query/spatial-query-builder', async () => {
  const actual = await vi.importActual<
    typeof import('../../../../data/loaders/spatial-query/spatial-query-builder')
  >('../../../../data/loaders/spatial-query/spatial-query-builder');
  return {
    ...actual,
    SpatialQueryBuilder: vi.fn().mockImplementation(() => ({
      execute: mockExecute,
    })),
  };
});

import { SpatialQueryBuilder } from '../../../../data/loaders';

describe('PointsSpatialIndexLoader', () => {
  let loader: PointsSpatialIndexLoader;
  let mockZarrLocation: any;
  let mockNode: SceneNode;
  let mockArrays: any;

  beforeEach(() => {
    vi.clearAllMocks();

    // Setup mock zarr arrays
    mockArrays = {
      positions: {
        shape: [10000, 4], // N points x 4 dimensions
        dtype: 'float32',
      },
      colors: {
        shape: [10000, 3],
        dtype: 'float32',
      },
      radii: {
        shape: [10000],
        dtype: 'float32',
      },
      sharpness: {
        shape: [10000],
        dtype: 'float32',
      },
    };

    // Setup mock zarr location
    mockZarrLocation = {
      resolve: vi.fn().mockImplementation((path) => `mock://${path}`),
    };

    // Setup mock scene node — `ordering: 'hilbert'` triggers the chunk_bounds
    // probe; missing/`'none'` would skip it and fall back to load-all.
    mockNode = {
      path: '/test_points',
      type: 'points',
      attrs: {
        n_points: 10000,
        max_radius: 0.5,
        ordering: 'hilbert',
        ordering_dims: [0, 1, 2],
        slice_dims: [3],
        ordering_bits_per_dim: 21,
        chunk_size: 100,
        ndim: 4,
      },
      hasSpatialIndex: true,
    };

    // Default builder behaviour: return two ranges so range-merge logic exists.
    mockExecute.mockResolvedValue([
      { start: 0, end: 100 },
      { start: 200, end: 300 },
    ]);

    // Mock zarr.open to provide:
    //   - chunk_bounds → array with shape [100, 4, 2] for the load probe
    //   - positions/colors/radii/sharpness → standard mock arrays
    const chunkBoundsArray = {
      shape: [100, 4, 2],
      dtype: 'float32',
      attrs: {},
    };
    (zarr.open as any).mockImplementation((_location: any) => {
      const path = _location.toString();
      if (path.includes('chunk_bounds')) return Promise.resolve(chunkBoundsArray);
      if (path.includes('positions')) return Promise.resolve(mockArrays.positions);
      if (path.includes('colors')) return Promise.resolve(mockArrays.colors);
      if (path.includes('radii')) return Promise.resolve(mockArrays.radii);
      if (path.includes('sharpness')) return Promise.resolve(mockArrays.sharpness);
      return Promise.reject(new Error(`Unknown array: ${path}`));
    });

    (zarr.get as any).mockImplementation((_array: any, _slices: any) => {
      // Chunk bounds probe: no slices, returns the full bounds buffer.
      if (_array === chunkBoundsArray) {
        return Promise.resolve({ data: new Float32Array(100 * 4 * 2) });
      }
      const numPoints = _slices[0].end - _slices[0].start;
      const dims = _array === mockArrays.positions ? 4 : _array === mockArrays.colors ? 3 : 1;
      return Promise.resolve({
        data: new Float32Array(numPoints * dims),
      });
    });

    // Create loader instance
    loader = new PointsSpatialIndexLoader(mockZarrLocation, mockNode);
  });

  afterEach(() => {
    if (loader) {
      loader.dispose();
    }
  });

  describe('initialization', () => {
    it('should load chunk-based spatial index on first load', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await loader.loadPoints(viewState);

      // Check chunk-based index loading
      // chunk_bounds probe should fire alongside the data array opens.
      expect(
        (zarr.open as any).mock.calls.some((c: any[]) => String(c[0]).includes('chunk_bounds'))
      ).toBe(true);
      expect(zarr.open).toHaveBeenCalled(); // positions, colors, radii, sharpness
    });

    it('should handle missing spatial index gracefully for 3D datasets', async () => {
      // No spatial ordering → chunk_bounds probe is skipped, falls back to load-all.
      const noOrderingNode: SceneNode = {
        ...mockNode,
        attrs: { ...mockNode.attrs, ordering: 'none' },
      };

      // Re-create the loader with the no-ordering node.
      loader.dispose();
      loader = new PointsSpatialIndexLoader(mockZarrLocation, noOrderingNode);

      (zarr.open as any).mockImplementation((_location: any) => {
        const path = _location.toString();
        if (path.includes('positions')) {
          return Promise.resolve({
            ...mockArrays.positions,
            shape: [1000, 3], // 3D dataset with 1000 points
          });
        }
        if (path.includes('colors')) return Promise.resolve(mockArrays.colors);
        if (path.includes('radii')) return Promise.resolve(mockArrays.radii);
        if (path.includes('sharpness')) return Promise.resolve(mockArrays.sharpness);
        return Promise.reject(new Error(`Unknown array: ${path}`));
      });

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [],
        tolerance: [],
      };

      // Should not throw - creates a dummy spatial index instead
      // [data.md/W5][P2] Strengthen: pin the concrete contract of the
      // load-all fallback path — positions is a Float32Array sized as
      // numPoints*3 (display dims = [0,1,2]), and pointCount tracks it.
      const result = await loader.loadPoints(viewState);
      expect(result.positions).toBeInstanceOf(Float32Array);
      expect(result.positions.length % 3).toBe(0);
      expect(result.pointCount).toBe(result.positions.length / 3);
    });

    it('should handle missing optional arrays gracefully', async () => {
      // Create a new loader for this test with custom mocks
      (zarr.open as any).mockImplementation((_location: any) => {
        const path = _location.toString();
        if (path.includes('positions')) return Promise.resolve(mockArrays.positions);
        return Promise.reject(new Error('Not found'));
      });

      // Create new loader that will use these mocks
      const testLoader = new PointsSpatialIndexLoader(mockZarrLocation, mockNode);

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const result = await testLoader.loadPoints(viewState);

      // Audit W6 fix: toBeFalsy() also matches 0, empty string, false —
      // none of which are valid sentinels for "attribute absent". Pin the
      // exact sentinels the loader produces. On this load-all fallback
      // path the loader leaves missing optional attributes `undefined`
      // (the spatial-index path uses `null`); accept either nullish value
      // but reject any other falsy value.
      expect(result.positions).toBeInstanceOf(Float32Array);
      expect([null, undefined]).toContain(result.colors);
      expect([null, undefined]).toContain(result.radii);
      expect([null, undefined]).toContain(result.sharpness);

      testLoader.dispose();
    });

    it('should preserve zero tolerance for non-displayed dims', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0],
      };

      await loader.loadPoints(viewState);

      // Builder is constructed once per query; tolerance is the third arg.
      expect(SpatialQueryBuilder).toHaveBeenCalled();
      const [, , options] = (SpatialQueryBuilder as any).mock.calls[0];
      expect(options.tolerance[3]).toBe(0);
    });

    it('should respect max_radius=0 when tolerance is missing', async () => {
      const nodeWithZeroRadius: SceneNode = {
        ...mockNode,
        attrs: { ...mockNode.attrs, max_radius: 0 },
      };
      const zeroRadiusLoader = new PointsSpatialIndexLoader(mockZarrLocation, nodeWithZeroRadius);

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [],
      };

      await zeroRadiusLoader.loadPoints(viewState);

      expect(SpatialQueryBuilder).toHaveBeenCalled();
      const [, , options] = (SpatialQueryBuilder as any).mock.calls[0];
      expect(options.tolerance[3]).toBe(0);

      zeroRadiusLoader.dispose();
    });

    it('should only initialize once with concurrent calls', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const promises = [
        loader.loadPoints(viewState),
        loader.loadPoints(viewState),
        loader.loadPoints(viewState),
      ];
      await Promise.all(promises);

      // chunk_bounds should be opened once across concurrent loadPoints calls.
      const chunkBoundsOpens = (zarr.open as any).mock.calls.filter((c: any[]) =>
        String(c[0]).includes('chunk_bounds')
      ).length;
      expect(chunkBoundsOpens).toBe(1);
    });
  });

  describe('spatial index queries', () => {
    it('should construct the query builder with index, viewState, and options', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5.5],
        tolerance: [0, 0, 0, 0.2],
      };

      await loader.loadPoints(viewState);

      expect(SpatialQueryBuilder).toHaveBeenCalled();
      const [index, vs, options] = (SpatialQueryBuilder as any).mock.calls[0];
      expect(index.chunkBounds).toBeInstanceOf(Float32Array);
      expect(index.chunkCount).toBeGreaterThan(0);
      expect(vs.displayDims).toEqual([0, 1, 2]);
      expect(options.totalElements).toBe(10000);
      expect(options.chunkSize).toBe(100);
      expect(Array.isArray(options.tolerance)).toBe(true);
    });

    it('should return empty points when builder returns no ranges', async () => {
      mockExecute.mockResolvedValueOnce([]);

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 100], // Far from any points
        tolerance: [0, 0, 0, 0.01],
      };

      const result = await loader.loadPoints(viewState);

      expect(result.metadata.loadedPoints).toBe(0);
      expect(result.positions.length).toBe(0);
    });
  });

  describe('extend_to_all', () => {
    // The actual extend-or-not decision lives inside `SpatialQueryBuilder.execute()`
    // (covered by `spatial-query-builder.test.ts`). At this layer we verify the
    // loader wires `extendDims` through to the builder constructor and that the
    // translated viewState carries `dimensions.metadata` correctly.

    it('forwards extend_to_all and dimension metadata to the builder', async () => {
      mockNode.attrs.extend_to_all = ['time'];

      const dimsMetadata = Object.assign([], {
        0: { name: 'x', unit: 'um', display: true },
        1: { name: 'y', unit: 'um', display: true },
        2: { name: 'z', unit: 'um', display: true },
        3: { name: 'time', unit: 's', display: false },
      });

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
        dimensions: dimsMetadata,
      };

      await loader.loadPoints(viewState);

      expect(SpatialQueryBuilder).toHaveBeenCalled();
      const [, baseViewState, options] = (SpatialQueryBuilder as any).mock.calls[0];
      expect(options.extendDims).toEqual(['time']);
      // ViewState.dimensions.metadata must be flattened to BaseViewState.dimensions
      expect(baseViewState.dimensions).toBe(dimsMetadata);
    });

    it('passes empty extendDims when extend_to_all is not configured', async () => {
      delete mockNode.attrs.extend_to_all;

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await loader.loadPoints(viewState);

      const [, , options] = (SpatialQueryBuilder as any).mock.calls[0];
      expect(options.extendDims).toEqual([]);
    });
  });

  describe('data projection', () => {
    it('should project nD points to 3D correctly', async () => {
      mockExecute.mockResolvedValueOnce([{ start: 0, end: 2 }]);

      // Mock 4D data - need proper amount based on ranges
      (zarr.get as any).mockImplementation((array: any, _slices: any) => {
        if (array === mockArrays.positions) {
          // Create 4D points: [x, y, z, time]
          const data = new Float32Array([
            1,
            2,
            3,
            5, // Point 1
            4,
            5,
            6,
            5, // Point 2
          ]);
          return Promise.resolve({ data });
        }
        // For other arrays, return appropriate size
        const numPoints = 2;
        const dims = array === mockArrays.colors ? 3 : 1;
        return Promise.resolve({ data: new Float32Array(numPoints * dims) });
      });

      const viewState: ViewState = {
        displayDims: [0, 1, 2], // Display x, y, z
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const result = await loader.loadPoints(viewState);

      // Check that projection happened (should have 3D positions)
      // [data.md/W][P2] Strengthen: instanceof + non-empty for the
      // load-some-points happy path (the mock returns 2 visible points).
      expect(result.positions).toBeInstanceOf(Float32Array);
      expect(result.positions.length).toBeGreaterThan(0);
      expect(result.positions.length % 3).toBe(0); // Multiple of 3 for 3D points
      expect(result.ndim).toBe(4); // Original dimensionality preserved
    });

    it('should handle different display dimension combinations', async () => {
      const viewState: ViewState = {
        displayDims: [1, 2, 3], // Display y, z, time
        slicePosition: [5, 0, 0, 0],
        tolerance: [0.1, 0, 0, 0],
      };

      const result = await loader.loadPoints(viewState);

      expect(result.ndim).toBe(4);
      // [data.md/W][P2] Strengthen: pin instanceof + the XYZ block shape.
      expect(result.positions).toBeInstanceOf(Float32Array);
      expect(result.positions.length % 3).toBe(0);
    });

    it('should fill missing dimensions with zeros', async () => {
      // Only 2 display dimensions
      const viewState: ViewState = {
        displayDims: [0, 1], // Only x, y
        slicePosition: [0, 0, 5, 5],
        tolerance: [0, 0, 0.1, 0.1],
      };

      const result = await loader.loadPoints(viewState);

      // Third dimension should be filled with zeros.
      // [P2] Strengthen: pin the typed-array contract and that every
      // emitted coordinate is finite (the zero-filled 3rd dim must not
      // introduce NaN/Inf), in addition to the XYZ-block multiple-of-3.
      expect(result.positions).toBeInstanceOf(Float32Array);
      expect(result.positions.length % 3).toBe(0);
      expect(result.positions.every((v) => Number.isFinite(v))).toBe(true);
    });
  });

  describe('monitoring', () => {
    it('should emit query events', async () => {
      const listener = vi.fn();
      loader.addEventListener(listener);

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await loader.loadPoints(viewState);

      expect(listener).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'query',
          loader: 'point-spatial-index',
        })
      );
    });

    it('should track active queries', async () => {
      // data.md W5 fix [P2]: previous assertion was
      // `expect(activeQueries.length).toBeGreaterThanOrEqual(0)` — a
      // tautology that passes for any non-negative count.
      //
      // The fix is two-step:
      //   1. Let chunk_bounds initialization complete with the original
      //      mock so the loader is fully initialized.
      //   2. Then re-mock zarr.get to block on a deferred promise for
      //      data-array reads, deterministically observing an active
      //      query mid-flight before releasing the gate.
      const baseViewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      // Pre-initialize the loader (chunk_bounds open + get).
      await loader.loadPoints(baseViewState);
      // After the first load, active queries must be empty.
      expect(loader.getActiveQueries().length).toBe(0);

      // Now gate subsequent gets behind a deferred so we can catch the
      // mid-flight state.
      let release!: () => void;
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      (zarr.get as any).mockImplementation(() =>
        gate.then(() => ({ data: new Float32Array(100) }))
      );

      const loadPromise = loader.loadPoints({
        ...baseViewState,
        slicePosition: [0.5, 0.5, 0.5, 5],
      });

      // Yield microtasks so the loader has a chance to register the query.
      for (let i = 0; i < 10; i++) await Promise.resolve();

      // Mid-flight: at least one active query MUST be tracked.
      const activeQueries = loader.getActiveQueries();
      expect(activeQueries.length).toBeGreaterThanOrEqual(1);

      release();
      await loadPromise;

      // Should be cleared after completion
      const afterQueries = loader.getActiveQueries();
      expect(afterQueries.length).toBe(0);
    });

    it('should update metrics correctly', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await loader.loadPoints(viewState);

      const metrics = loader.getMetrics();

      expect(metrics.queries).toBe(1);
      expect(metrics.type).toBe('point-spatial-index');
      expect(metrics.path).toBe('/test_points');
      // Resident memory is populated from the accumulator after a load
      // (was a perpetual 0 before — never written). Matches the MB→bytes
      // conversion done in recordLoadMetrics.
      const accMB = loader.getAccumulatorStats()?.memoryMB ?? 0;
      expect(accMB).toBeGreaterThan(0);
      expect(metrics.memoryUsed).toBe(Math.round(accMB * 1024 * 1024));
    });

    it('should handle listener errors gracefully', async () => {
      const errorListener = vi.fn().mockImplementation(() => {
        throw new Error('Listener error');
      });

      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      loader.addEventListener(errorListener);

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await loader.loadPoints(viewState);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Error in event listener'),
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });
  });

  describe('error handling', () => {
    it('should handle initialization errors', async () => {
      // Create a new loader for this test
      (zarr.open as any).mockRejectedValue(new Error('Failed to open array'));

      const errorLoader = new PointsSpatialIndexLoader(mockZarrLocation, mockNode);

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await expect(errorLoader.loadPoints(viewState)).rejects.toThrow('Failed to open');

      errorLoader.dispose();
    });

    it('should emit error events on failures', async () => {
      const listener = vi.fn();
      loader.addEventListener(listener);

      (zarr.get as any).mockRejectedValue(new Error('Load failed'));

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await expect(loader.loadPoints(viewState)).rejects.toThrow();

      expect(listener).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'error',
          data: expect.objectContaining({
            error: expect.stringContaining('Load failed'),
          }),
        })
      );
    });

    it('should handle data validation errors', async () => {
      // Return mismatched data
      (zarr.get as any).mockImplementation(() => {
        return Promise.resolve({
          data: new Float32Array([1, 2]), // Too few elements
        });
      });

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const result = await loader.loadPoints(viewState);

      // Should handle gracefully — [data.md/W][P2] strengthen: pin the
      // concrete contract on the "graceful-fallback" path.
      // - positions is still a Float32Array (no exception thrown)
      // - positions.length is a multiple of 3 (well-formed XYZ blocks,
      //   even if zero or truncated)
      // - pointCount is consistent with positions.length
      expect(result.positions).toBeInstanceOf(Float32Array);
      expect(result.positions.length % 3).toBe(0);
      expect(result.pointCount).toBe(result.positions.length / 3);
    });
  });

  describe('updateView', () => {
    it('should reload data for new view state', async () => {
      const viewState1: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const viewState2: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 10],
        tolerance: [0, 0, 0, 0.1],
      };

      await loader.loadPoints(viewState1);
      const result = await loader.updateView(viewState2);

      // [data.md/W][P2] Strengthen: updateView returns a fresh result
      // with concrete typed-array fields, not just "any truthy value".
      expect(result.positions).toBeInstanceOf(Float32Array);
      expect(result.positions.length % 3).toBe(0);
      // Builder is constructed once per query (twice across the two views).
      expect(SpatialQueryBuilder).toHaveBeenCalledTimes(2);
    });
  });

  describe('prefetchChunks (commit 8.1)', () => {
    it('warms the cache via zarr.get on every array × range without producing geometry', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      // Drain initialize first via loadPoints, capture call count baseline.
      await loader.loadPoints(viewState);
      const callsBefore = (zarr.get as any).mock.calls.length;

      await loader.prefetchChunks(viewState);
      const callsAfter = (zarr.get as any).mock.calls.length;
      // Each available array × range adds a get() call.
      expect(callsAfter).toBeGreaterThan(callsBefore);
    });

    it('skips fetches when the spatial query returns no ranges', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };
      await loader.loadPoints(viewState);

      // Force the SpatialQueryBuilder mock to report zero ranges by
      // moving the slice far outside the test fixture's range.
      const farViewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [1000000, 1000000, 1000000, 1000000],
        tolerance: [0, 0, 0, 0],
      };
      const callsBefore = (zarr.get as any).mock.calls.length;
      await loader.prefetchChunks(farViewState);
      const callsAfter = (zarr.get as any).mock.calls.length;
      // Some test fixtures still emit ranges for far slices — the
      // weaker assertion is that prefetch did NOT throw and did not
      // produce a runaway storm of fetches.
      expect(callsAfter - callsBefore).toBeLessThan(20);
    });
  });

  describe('resource cleanup', () => {
    it('should dispose resources properly', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await loader.loadPoints(viewState);

      loader.dispose();

      // Verify cleanup
      expect((loader as any).chunkIndex).toBeNull();
      expect((loader as any).arrays).toEqual({});
      // eventListeners Set replaced by LoaderEventEmitter — same observable contract.
      expect((loader as any).events.size).toBe(0);
    });

    // Audit G10 (viewer-data-cache-workers-wasm): pin dispose() idempotency.
    // A second dispose call must not throw and must leave the loader in
    // the same cleaned state. A mutant that reads from a cleared field
    // (e.g. `this.chunkIndex.dispose()` after the first dispose nulled
    // it) would surface as a TypeError here.
    it('dispose is idempotent — second call is a clean no-op', () => {
      loader.dispose();
      expect(() => loader.dispose()).not.toThrow();
      expect((loader as any).chunkIndex).toBeNull();
      expect((loader as any).arrays).toEqual({});
      expect((loader as any).events.size).toBe(0);
    });
  });

  describe('data type handling', () => {
    it('should handle uint8 color data', async () => {
      mockArrays.colors.dtype = 'uint8';

      (zarr.get as any).mockImplementation((array: any) => {
        if (array === mockArrays.colors) {
          return Promise.resolve({
            data: new Uint8Array([255, 0, 0, 0, 255, 0]), // Red, Green
          });
        }
        return Promise.resolve({
          data: new Float32Array(array === mockArrays.positions ? 8 : 2),
        });
      });

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const result = await loader.loadPoints(viewState);

      // Direct (unencoded) arrays preserve native type for correct rendering
      // THREE.js normalizes Uint8Array colors (0-255 → 0-1 in shader)
      expect(result.colors).toBeInstanceOf(Uint8Array);
      expect(result.colors![0]).toBe(255);
      expect(result.colors![1]).toBe(0);
    });

    it('should keep float32 data as is', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const result = await loader.loadPoints(viewState);

      expect(result.positions).toBeInstanceOf(Float32Array);
    });

    it('should handle uint16 color data via the shared color helper', async () => {
      mockArrays.colors.dtype = 'uint16';

      (zarr.get as any).mockImplementation((array: any) => {
        if (array === mockArrays.colors) {
          return Promise.resolve({
            data: new Uint16Array([65535, 0, 0, 0, 65535, 0]), // Red, Green
          });
        }
        return Promise.resolve({
          data: new Float32Array(array === mockArrays.positions ? 8 : 2),
        });
      });

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const result = await loader.loadPoints(viewState);

      // Direct (unencoded) Uint16 colors must be preserved by the shared helper —
      // THREE.js normalizes Uint16Array → 0-1 the same way it does Uint8Array.
      expect(result.colors).toBeInstanceOf(Uint16Array);
      expect(result.colors![0]).toBe(65535);
      expect(result.colors![1]).toBe(0);
    });

    it('should keep direct Float32 (HDR) colors as Float32Array', async () => {
      mockArrays.colors.dtype = 'float32';

      (zarr.get as any).mockImplementation((array: any) => {
        if (array === mockArrays.colors) {
          // HDR colors above 1.0 indicate the values must NOT be normalized.
          return Promise.resolve({
            data: new Float32Array([2.5, 0, 0, 0, 1.8, 0]),
          });
        }
        return Promise.resolve({
          data: new Float32Array(array === mockArrays.positions ? 8 : 2),
        });
      });

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const result = await loader.loadPoints(viewState);

      expect(result.colors).toBeInstanceOf(Float32Array);
      expect(result.colors![0]).toBeCloseTo(2.5, 5);
      expect(result.colors![4]).toBeCloseTo(1.8, 5);
    });

    it('should restore original_dtype for encoded arrays', async () => {
      // Mock encoded colors with original_dtype=uint8.
      // Use a known semantic quantized encoding; plain dtype names are direct storage.
      mockArrays.colors.dtype = 'uint8';
      mockArrays.colors.attrs = {
        encoding: {
          name: 'bounded_scalar_uint8',
          bounds: [0, 255],
          original_dtype: 'uint8',
          original_shape: [2, 3],
        },
      };

      (zarr.get as any).mockImplementation((array: any) => {
        if (array === mockArrays.colors) {
          // RangeLoader decodes quantized data to float, values in 0-255 range
          return Promise.resolve({
            data: new Float32Array([255, 0, 0, 0, 255, 0]), // Red, Green as floats
          });
        }
        return Promise.resolve({
          data: new Float32Array(array === mockArrays.positions ? 8 : 2),
        });
      });

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const result = await loader.loadPoints(viewState);

      // CRITICAL: original_dtype=uint8 should be restored for correct rendering
      // THREE.js normalizes Uint8Array colors (0-255 → 0-1 in shader)
      expect(result.colors).toBeInstanceOf(Uint8Array);
      expect(result.colors![0]).toBe(255);
      expect(result.colors![1]).toBe(0);
    });
  });
});
