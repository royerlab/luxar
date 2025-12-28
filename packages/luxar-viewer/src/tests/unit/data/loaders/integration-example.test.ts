import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  PointsLoaderIntegrationExample,
  queryVisibleRangesExample,
} from '../../../../data/loaders/integration-example';
import type { BaseViewState } from '../../../../data/loaders/base-types';

// Mock worker pool
vi.mock('../../../../workers/worker-pool', () => ({
  getWorkerPool: () => ({
    getWorker: () =>
      Promise.resolve({
        projectPointsTo3D: vi.fn().mockResolvedValue({
          positions3D: new Float32Array([1, 2, 3]),
          pointCount: 1,
        }),
      }),
  }),
}));

// Mock config
vi.mock('../../../../config', () => ({
  config: {
    dataLoading: {
      performance: {
        useWebWorkers: false, // Use main thread for predictable testing
      },
    },
  },
}));

describe('PointsLoaderIntegrationExample', () => {
  let loader: PointsLoaderIntegrationExample;

  beforeEach(() => {
    loader = new PointsLoaderIntegrationExample(100);
  });

  afterEach(() => {
    loader.dispose();
  });

  describe('constructor', () => {
    it('should create with initial capacity', () => {
      const stats = loader.getStats();
      expect(stats.capacity).toBe(100);
      expect(stats.allocations).toBe(1);
    });
  });

  describe('loadAndProject', () => {
    it('should project 3D points correctly', async () => {
      // Create 4D source positions (10 points)
      const positions = new Float32Array(40); // 10 points * 4 dims
      for (let i = 0; i < 10; i++) {
        positions[i * 4 + 0] = i * 1.0; // x
        positions[i * 4 + 1] = i * 2.0; // y
        positions[i * 4 + 2] = i * 3.0; // z
        positions[i * 4 + 3] = i * 4.0; // t
      }

      const viewState: BaseViewState = {
        displayDims: [0, 1, 2], // Display x, y, z
        slicePosition: [0, 0, 0, 5],
        tolerance: [],
      };

      const result = await loader.loadAndProject(positions, viewState);

      expect(result.pointCount).toBe(10);
      expect(result.positions3D.length).toBe(30); // 10 * 3

      // Verify first point projected correctly
      expect(result.positions3D[0]).toBe(0); // x
      expect(result.positions3D[1]).toBe(0); // y
      expect(result.positions3D[2]).toBe(0); // z

      // Verify last point
      expect(result.positions3D[27]).toBe(9); // x
      expect(result.positions3D[28]).toBe(18); // y
      expect(result.positions3D[29]).toBe(27); // z
    });

    it('should handle colors', async () => {
      const positions = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]); // 2 4D points
      const colors = new Float32Array([1, 0, 0, 0, 1, 0]); // Red, Green

      const viewState: BaseViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [],
      };

      const result = await loader.loadAndProject(positions, viewState, { colors });

      expect(result.colors).toBeDefined();
      expect(result.colors!.length).toBe(6);
      expect(result.colors![0]).toBe(1); // Red x
      expect(result.colors![1]).toBe(0); // Red y
      expect(result.colors![2]).toBe(0); // Red z
    });

    it('should handle radii', async () => {
      const positions = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]); // 2 4D points
      const radii = new Float32Array([1.5, 2.5]);

      const viewState: BaseViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [],
      };

      const result = await loader.loadAndProject(positions, viewState, { radii });

      expect(result.radii).toBeDefined();
      expect(result.radii!.length).toBe(2);
      expect(result.radii![0]).toBe(1.5);
      expect(result.radii![1]).toBe(2.5);
    });

    it('should grow capacity when needed', async () => {
      const smallLoader = new PointsLoaderIntegrationExample(5);

      // Create 10 points (more than capacity)
      const positions = new Float32Array(30); // 10 3D points
      for (let i = 0; i < 10; i++) {
        positions[i * 3 + 0] = i;
        positions[i * 3 + 1] = i * 2;
        positions[i * 3 + 2] = i * 3;
      }

      const viewState: BaseViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [],
      };

      const result = await smallLoader.loadAndProject(positions, viewState);

      expect(result.pointCount).toBe(10);
      expect(smallLoader.getStats().capacity).toBeGreaterThanOrEqual(10);

      smallLoader.dispose();
    });

    it('should track reuse count', async () => {
      const positions = new Float32Array([1, 2, 3]); // 1 3D point

      const viewState: BaseViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [],
      };

      // Multiple load operations
      await loader.loadAndProject(positions, viewState);
      await loader.loadAndProject(positions, viewState);
      await loader.loadAndProject(positions, viewState);

      const stats = loader.getStats();
      expect(stats.reuseCount).toBe(3);
    });
  });

  describe('getStats', () => {
    it('should return correct statistics', async () => {
      const positions = new Float32Array([1, 2, 3, 4, 5, 6]); // 2 3D points

      const viewState: BaseViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [],
      };

      await loader.loadAndProject(positions, viewState);

      const stats = loader.getStats();
      expect(stats.capacity).toBe(100);
      expect(stats.lastPointCount).toBe(2);
      expect(stats.allocations).toBe(1);
      expect(stats.reuseCount).toBe(1);
      expect(stats.peakMemoryBytes).toBeGreaterThan(0);
    });
  });
});

describe('queryVisibleRangesExample', () => {
  const createMockIndex = (numChunks: number, ndim: number) => ({
    chunkBounds: new Float32Array(numChunks * ndim * 2),
    chunkCount: numChunks,
    metadata: { ndim, chunk_size: 1000 },
  });

  it('should return ranges for matching chunks', async () => {
    const index = createMockIndex(3, 3);
    // Set up chunks: [0,100], [100,200], [200,300] in each dim
    index.chunkBounds.set([0, 100, 0, 100, 0, 100], 0);
    index.chunkBounds.set([100, 200, 100, 200, 100, 200], 6);
    index.chunkBounds.set([200, 300, 200, 300, 200, 300], 12);

    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [150, 150, 150], // Should hit chunk 1
      tolerance: [],
    };

    const ranges = await queryVisibleRangesExample(index, viewState, 3000);

    // With infinite tolerance for displayed dims, all chunks should match
    expect(ranges.length).toBeGreaterThan(0);
  });

  it('should apply maxRadius option', async () => {
    const index = createMockIndex(2, 3);
    index.chunkBounds.set([0, 50, 0, 50, 0, 50], 0);
    index.chunkBounds.set([100, 150, 100, 150, 100, 150], 6);

    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [25, 25, 25],
      tolerance: [],
    };

    const ranges = await queryVisibleRangesExample(index, viewState, 2000, {
      maxRadius: 10,
    });

    expect(ranges.length).toBeGreaterThan(0);
  });

  it('should return empty array when no chunks match hidden dim filter', async () => {
    const index = createMockIndex(1, 4);
    // Chunk bounds: all at position 1000 in dim 3 (hidden)
    index.chunkBounds.set([0, 100, 0, 100, 0, 100, 1000, 1100], 0);

    const viewState: BaseViewState = {
      displayDims: [0, 1, 2], // dim 3 is hidden
      slicePosition: [50, 50, 50, 0], // Query at t=0, but chunk is at t=1000
      tolerance: [0, 0, 0, 1], // Very small tolerance in hidden dim
    };

    const ranges = await queryVisibleRangesExample(index, viewState, 1000);

    expect(ranges.length).toBe(0);
  });
});
