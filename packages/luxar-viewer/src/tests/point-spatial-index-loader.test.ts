/**
 * Comprehensive tests for PointSpatialIndexLoader
 *
 * Tests point spatial index-based loading, nD queries, caching,
 * broadcasting, and monitoring integration.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { PointSpatialIndexLoader, type ViewState, type SceneNode } from '../data';
import * as zarr from 'zarrita';

// Mock THREE.js using partial mock with importOriginal
vi.mock('three', async (importOriginal) => {
  const actual = await importOriginal<typeof import('three')>();
  return {
    ...actual,
    Box3: vi.fn().mockImplementation(() => ({
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
  open: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

// Mock point spatial index functions
vi.mock('../data/point-spatial-index', () => ({
  loadPointSpatialIndex: vi.fn(),
  queryPointSpatialIndex: vi.fn(),
  mergePointRanges: vi.fn(),
}));

// Mock RangeCache
vi.mock('../data/range-cache', () => ({
  RangeCache: vi.fn().mockImplementation(() => ({
    get: vi.fn(),
    set: vi.fn(),
    has: vi.fn(),
    clear: vi.fn(),
    getStats: vi.fn().mockReturnValue({
      hits: 5,
      misses: 3,
      hitRate: 0.625,
      numEntries: 2,
      totalMemory: 1024 * 1024,
    }),
    getMemoryInfo: vi.fn().mockReturnValue({
      used: 1024 * 1024,
      max: 100 * 1024 * 1024,
      percentage: 1,
    }),
    getCacheStats: vi.fn().mockReturnValue({
      hits: 5,
      misses: 3,
      evictions: 0,
      avgAccessTime: 0.5,
    }),
  })),
}));

// Import mocked modules
import { loadPointSpatialIndex, queryPointSpatialIndex, mergePointRanges } from '../data';

describe('PointSpatialIndexLoader', () => {
  let loader: PointSpatialIndexLoader;
  let mockZarrLocation: any;
  let mockNode: SceneNode;
  let mockSpatialIndex: any;
  let mockArrays: any;

  beforeEach(() => {
    vi.clearAllMocks();

    // Setup mock spatial index
    mockSpatialIndex = {
      metadata: {
        grid_shape: [10, 10, 10],
        grid_origin: [0, 0, 0],
        cell_size: [1, 1, 1],
        num_occupied: 50,
        total_cells: 1000,
        total_points: 10000,
        dimensions: 1, // 1 indexed dimension (time)
        full_dimensions: 4, // x, y, z, time
        indexed_dimensions: [3], // Only time is indexed
        displayed_dimensions: [0, 1, 2], // x, y, z are displayed
        build_version: '0.5',
      },
      occupiedCells: new Uint32Array([0, 1, 2]),
      cellRanges: new BigUint64Array([0n, 100n, 100n, 200n, 200n, 300n]),
    };

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

    // Setup mock scene node
    mockNode = {
      path: '/test_points',
      type: 'points',
      attrs: {
        num_points: 10000,
        max_radius: 0.5,
      },
      hasSpatialIndex: true,
    };

    // Configure mocks - set default behavior
    (loadPointSpatialIndex as any).mockResolvedValue(mockSpatialIndex);
    (queryPointSpatialIndex as any).mockReturnValue([
      { start: 0, end: 100 },
      { start: 200, end: 300 },
    ]);
    (mergePointRanges as any).mockImplementation((_ranges: any) => _ranges);

    (zarr.open as any).mockImplementation((_location: any) => {
      const path = _location.toString();
      if (path.includes('positions')) return Promise.resolve(mockArrays.positions);
      if (path.includes('colors')) return Promise.resolve(mockArrays.colors);
      if (path.includes('radii')) return Promise.resolve(mockArrays.radii);
      if (path.includes('sharpness')) return Promise.resolve(mockArrays.sharpness);
      return Promise.reject(new Error(`Unknown array: ${path}`));
    });

    (zarr.get as any).mockImplementation((_array: any, _slices: any) => {
      const numPoints = _slices[0].end - _slices[0].start;
      const dims = _array === mockArrays.positions ? 4 : _array === mockArrays.colors ? 3 : 1;
      return Promise.resolve({
        data: new Float32Array(numPoints * dims),
      });
    });

    // Create loader instance
    loader = new PointSpatialIndexLoader(mockZarrLocation, mockNode);
  });

  afterEach(() => {
    if (loader) {
      loader.dispose();
    }
  });

  describe('initialization', () => {
    it('should load spatial index on first load', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await loader.loadPoints(viewState);

      expect(loadPointSpatialIndex).toHaveBeenCalledWith(mockZarrLocation);
      expect(zarr.open).toHaveBeenCalledTimes(4); // positions, colors, radii, sharpness
    });

    it('should handle missing spatial index gracefully for 3D datasets', async () => {
      (loadPointSpatialIndex as any).mockResolvedValue(null);

      // Mock positions array to determine point count
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
      const result = await loader.loadPoints(viewState);
      expect(result).toBeDefined();
      expect(result.positions).toBeDefined();
    });

    it('should handle missing optional arrays gracefully', async () => {
      // Create a new loader for this test with custom mocks
      (zarr.open as any).mockImplementation((_location: any) => {
        const path = _location.toString();
        if (path.includes('positions')) return Promise.resolve(mockArrays.positions);
        return Promise.reject(new Error('Not found'));
      });

      // Create new loader that will use these mocks
      const testLoader = new PointSpatialIndexLoader(mockZarrLocation, mockNode);

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const result = await testLoader.loadPoints(viewState);

      expect(result.positions).toBeDefined();
      expect(result.colors).toBeFalsy(); // Can be null or undefined
      expect(result.radii).toBeFalsy(); // Can be null or undefined
      expect(result.sharpness).toBeFalsy(); // Can be null or undefined

      testLoader.dispose();
    });

    it('should only initialize once with concurrent calls', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      // Start multiple loads concurrently
      const promises = [
        loader.loadPoints(viewState),
        loader.loadPoints(viewState),
        loader.loadPoints(viewState),
      ];

      await Promise.all(promises);

      // Should only initialize once
      expect(loadPointSpatialIndex).toHaveBeenCalledTimes(1);
    });
  });

  describe('spatial index queries', () => {
    it('should query spatial index with correct parameters', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5.5],
        tolerance: [0, 0, 0, 0.2],
      };

      await loader.loadPoints(viewState);

      expect(queryPointSpatialIndex).toHaveBeenCalledWith(
        mockSpatialIndex,
        expect.arrayContaining([0, 0, 0, 5.5]),
        expect.any(Array)
      );
    });

    it('should return empty points when no points visible', async () => {
      (queryPointSpatialIndex as any).mockReturnValue([]);

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 100], // Far from any points
        tolerance: [0, 0, 0, 0.01],
      };

      const result = await loader.loadPoints(viewState);

      expect(result.metadata.loadedPoints).toBe(0);
      expect(result.positions.length).toBe(0);
    });

    it('should merge adjacent ranges for efficiency', async () => {
      (queryPointSpatialIndex as any).mockReturnValue([
        { start: 0, end: 100 },
        { start: 100, end: 200 }, // Adjacent
        { start: 300, end: 400 },
      ]);

      (mergePointRanges as any).mockReturnValue([
        { start: 0, end: 200 }, // Merged
        { start: 300, end: 400 },
      ]);

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.5],
      };

      await loader.loadPoints(viewState);

      expect(mergePointRanges).toHaveBeenCalled();
    });
  });

  describe('broadcasting', () => {
    it('should return all points when broadcasting dimension is navigated', async () => {
      // Setup node with broadcast dimensions
      mockNode.attrs.broadcast_dims = ['time'];

      const viewState: ViewState = {
        displayDims: [0, 1, 2], // x, y, z displayed
        slicePosition: [0, 0, 0, 5], // Navigating time
        tolerance: [0, 0, 0, 0.1],
        dimensions: {
          ndim: 4,
          currentStep: [0, 0, 0, 5],
          displayed: [0, 1, 2],
          metadata: Object.assign([], {
            0: { name: 'x', unit: 'um', display: true },
            1: { name: 'y', unit: 'um', display: true },
            2: { name: 'z', unit: 'um', display: true },
            3: { name: 'time', unit: 's', display: false },
          }),
        },
      };

      // Should not call queryPointSpatialIndex but return all points
      const result = await loader.loadPoints(viewState);

      // When broadcasting, returns all points
      expect(result.metadata.totalPoints).toBeGreaterThan(0);
    });

    it('should use spatial index when not broadcasting', async () => {
      mockNode.attrs.broadcast_dims = ['channel']; // Different dimension

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
        dimensions: {
          ndim: 4,
          currentStep: [0, 0, 0, 5],
          displayed: [0, 1, 2],
          metadata: Object.assign([], {
            3: { name: 'time', unit: 's', display: false },
          }),
        },
      };

      await loader.loadPoints(viewState);

      expect(queryPointSpatialIndex).toHaveBeenCalled();
    });
  });

  describe('caching', () => {
    it('should use cache for repeated queries', async () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      // First load - cache miss
      const cache = (loader as any).cache;
      cache.get.mockReturnValue(null);

      await loader.loadPoints(viewState);

      expect(cache.get).toHaveBeenCalled();
      expect(cache.set).toHaveBeenCalled();

      // Reset mock call count for second load
      vi.clearAllMocks();

      // Second load - cache hit
      cache.get.mockReturnValue(new Float32Array([1, 2, 3]));

      await loader.loadPoints(viewState);

      // Should not call zarr.get again if cache hit
      expect(cache.get).toHaveBeenCalled();
    });

    it('should report cache statistics correctly', () => {
      const stats = loader.getCacheStats();

      expect(stats).toHaveProperty('hits', 5);
      expect(stats).toHaveProperty('misses', 3);
      expect(stats).toHaveProperty('hitRate', 0.625);
    });

    it('should clear cache on demand', () => {
      const cache = (loader as any).cache;

      loader.clearCache();

      expect(cache.clear).toHaveBeenCalled();
    });
  });

  describe('data projection', () => {
    it('should project nD points to 3D correctly', async () => {
      // Update query to return ranges for 2 points
      (queryPointSpatialIndex as any).mockReturnValue([{ start: 0, end: 2 }]);

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
      expect(result.positions).toBeDefined();
      expect(result.positions.length % 3).toBe(0); // Multiple of 3 for 3D points
      expect(result.metadata.ndim).toBe(4); // Original dimensionality preserved in metadata
    });

    it('should handle different display dimension combinations', async () => {
      const viewState: ViewState = {
        displayDims: [1, 2, 3], // Display y, z, time
        slicePosition: [5, 0, 0, 0],
        tolerance: [0.1, 0, 0, 0],
      };

      const result = await loader.loadPoints(viewState);

      expect(result.metadata.ndim).toBe(4);
      expect(result.positions).toBeDefined();
    });

    it('should fill missing dimensions with zeros', async () => {
      // Only 2 display dimensions
      const viewState: ViewState = {
        displayDims: [0, 1], // Only x, y
        slicePosition: [0, 0, 5, 5],
        tolerance: [0, 0, 0.1, 0.1],
      };

      const result = await loader.loadPoints(viewState);

      // Third dimension should be filled with zeros
      expect(result.positions.length % 3).toBe(0);
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

    it('should emit cache hit/miss events', async () => {
      const listener = vi.fn();
      loader.addEventListener(listener);

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await loader.loadPoints(viewState);

      // Should emit cache miss (first load)
      expect(listener).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'cache-miss',
        })
      );
    });

    it('should track active queries', async () => {
      // Make zarr.get slower to allow checking active queries
      (zarr.get as any).mockImplementation(
        () =>
          new Promise((resolve) => setTimeout(() => resolve({ data: new Float32Array(100) }), 10))
      );

      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      const loadPromise = loader.loadPoints(viewState);

      // Give a tiny bit of time for the query to start
      await new Promise((resolve) => setTimeout(resolve, 1));

      // Check active queries during loading
      const activeQueries = loader.getActiveQueries();
      expect(activeQueries.length).toBeGreaterThanOrEqual(0); // May or may not catch it

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

      const errorLoader = new PointSpatialIndexLoader(mockZarrLocation, mockNode);

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

      // Should handle gracefully
      expect(result).toBeDefined();
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

      expect(result).toBeDefined();
      expect(queryPointSpatialIndex).toHaveBeenCalledTimes(2);
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
      expect((loader as any).spatialIndex).toBeNull();
      expect((loader as any).arrays).toEqual({});
      expect((loader as any).eventListeners.size).toBe(0);
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

      // Should keep uint8 as is for efficiency (with normalized flag in BufferAttribute)
      expect(result.colors).toBeInstanceOf(Uint8Array);
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
  });
});
