import { describe, it, expect, vi, beforeEach } from 'vitest';
import { LazyDataManager } from '../data/lazy-data-manager';
import * as zarr from 'zarrita';

// Mock zarrita
vi.mock('zarrita', () => ({
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, stop: end })),
}));

// Mock memory detector
vi.mock('../utils/memory-detector', () => ({
  detectMemory: vi.fn().mockReturnValue({
    totalGB: 16,
    recommendedCacheMB: 1000,
    safetyFactorUsed: 0.8,
  }),
  MemoryMonitor: vi.fn().mockImplementation(() => ({
    start: vi.fn(),
    stop: vi.fn(),
    dispose: vi.fn(),
  })),
}));

describe('LazyDataManager', () => {
  let manager: LazyDataManager;
  let mockArray: any;

  beforeEach(() => {
    vi.clearAllMocks();

    // Setup mock zarr array
    mockArray = {
      shape: [1000, 4],
      chunks: [100, 4],
      dtype: '<f4',
      get: vi.fn().mockResolvedValue({
        data: new Float32Array(100 * 4),
      }),
    };

    // Setup the zarr.get mock to return data
    (zarr.get as any).mockResolvedValue({
      data: new Float32Array(100 * 4),
    });

    manager = new LazyDataManager({
      enabled: true,
      maxMemoryMB: 100,
      preloadRadius: 1,
      evictionStrategy: 'lru',
      debug: false,
    });
  });

  describe('calculateRequiredChunks', () => {
    it('should calculate correct chunks for partial loading', () => {
      const chunks = (manager as any).calculateRequiredChunks(
        mockArray.shape,
        mockArray.chunks,
        [500, 0], // Position at array index 500 (chunk 5 since chunks are 100)
        [1], // Fully load second dimension
        1 // Preload radius
      );

      // Should load chunks 4, 5, 6 for first dimension (radius 1)
      // Should load all chunks (0) for second dimension
      expect(chunks).toHaveLength(3);
      expect(chunks).toContainEqual([4, 0]);
      expect(chunks).toContainEqual([5, 0]);
      expect(chunks).toContainEqual([6, 0]);
    });

    it('should respect array boundaries', () => {
      const chunks = (manager as any).calculateRequiredChunks(
        mockArray.shape,
        mockArray.chunks,
        [0, 0], // At the beginning
        [1], // Fully load second dimension
        2 // Preload radius
      );

      // Should load chunks 0, 1, 2 (can't go below 0)
      expect(chunks).toHaveLength(3);
      expect(chunks).toContainEqual([0, 0]);
      expect(chunks).toContainEqual([1, 0]);
      expect(chunks).toContainEqual([2, 0]);
    });

    it('should handle end boundary correctly', () => {
      const chunks = (manager as any).calculateRequiredChunks(
        mockArray.shape,
        mockArray.chunks,
        [900, 0], // Array index 900 is in chunk 9 (last chunk)
        [1],
        2
      );

      // Should load chunks 7, 8, 9 (can't go above 9)
      expect(chunks).toHaveLength(3);
      expect(chunks).toContainEqual([7, 0]);
      expect(chunks).toContainEqual([8, 0]);
      expect(chunks).toContainEqual([9, 0]);
    });

    it('should fully load specified dimensions', () => {
      // Array with shape [1000, 10, 5] and chunks [100, 5, 5]
      const multiDimArray = {
        shape: [1000, 10, 5],
        chunks: [100, 5, 5],
        dtype: '<f4',
      };

      const chunks = (manager as any).calculateRequiredChunks(
        multiDimArray.shape,
        multiDimArray.chunks,
        [50, 5, 0], // Position at array indices (chunk 0 for dim 0, chunk 1 for dim 1, chunk 0 for dim 2)
        [1, 2], // Fully load dimensions 1 and 2
        0 // No preload radius
      );

      // Should load chunk 0 for dim 0 (position 50 / chunk 100 = 0)
      // Should load all chunks (0,1) for dim 1 (10/5 = 2 chunks)
      // Should load all chunks (0) for dim 2 (5/5 = 1 chunk)
      expect(chunks).toHaveLength(2); // 1 * 2 * 1 = 2 combinations
      expect(chunks).toContainEqual([0, 0, 0]);
      expect(chunks).toContainEqual([0, 1, 0]);
    });
  });

  describe('cache management', () => {
    it('should respect memory limits', async () => {
      // Create manager with small memory limit
      const smallManager = new LazyDataManager({
        enabled: true,
        maxMemoryMB: 10, // 10MB limit
        preloadRadius: 0,
        evictionStrategy: 'lru',
        debug: false,
      });

      // Mock chunk data
      const chunkData = new Float32Array(100 * 4); // 1.6KB per chunk
      (zarr.get as any).mockResolvedValue({ data: chunkData });

      // Load multiple chunks
      const promises = [];
      for (let i = 0; i < 10; i++) {
        promises.push(smallManager.loadChunk(mockArray, 'test', [i, 0]));
      }
      await Promise.all(promises);

      // Cache size should stay within limit
      const stats = smallManager.getCacheStats();
      expect(stats.totalSizeMB).toBeLessThanOrEqual(10);
    });

    it('should evict oldest chunks when memory limit reached', async () => {
      const smallManager = new LazyDataManager({
        enabled: true,
        maxMemoryMB: 0.001, // Very small limit (1KB) to force eviction
        preloadRadius: 0,
        evictionStrategy: 'lru',
        debug: false,
      });

      // Mock chunk data - each chunk is about 1.6KB
      const chunkData = new Float32Array(100 * 4);
      (zarr.get as any).mockResolvedValue({ data: chunkData });

      // Load first chunk
      await smallManager.loadChunk(mockArray, 'test', [0, 0]);

      // Load second chunk - should evict first since 1.6KB > 1KB limit
      await smallManager.loadChunk(mockArray, 'test', [1, 0]);

      const stats = smallManager.getCacheStats();
      expect(stats.numChunks).toBe(1); // Only one chunk should remain
    });

    it('should handle concurrent chunk loads', async () => {
      // Mock delayed chunk loading
      let resolveCount = 0;
      (zarr.get as any).mockImplementation(() => {
        return new Promise((resolve) => {
          setTimeout(() => {
            resolveCount++;
            resolve({ data: new Float32Array(100 * 4) });
          }, 10);
        });
      });

      // Start multiple concurrent loads for the same chunk
      const promises = [];
      for (let i = 0; i < 5; i++) {
        promises.push(manager.loadChunk(mockArray, 'test', [0, 0]));
      }

      const results = await Promise.all(promises);

      // All should return the same data
      expect(results.every((r) => r === results[0])).toBe(true);

      // Should only fetch once despite multiple requests
      expect(resolveCount).toBe(1);
    });
  });

  describe('preloadChunks', () => {
    it('should preload chunks around current position', async () => {
      const chunkData = new Float32Array(100 * 4);
      (zarr.get as any).mockResolvedValue({ data: chunkData });

      await manager.preloadChunks(
        mockArray,
        'test',
        [500, 0], // Current position (in chunk 5)
        [1] // Fully load dimension 1
      );

      // Position 500 is in chunk 5, with preloadRadius 1+1=2 should load chunks 3, 4, 5, 6, 7 (5 chunks)
      const stats = manager.getCacheStats();
      expect(stats.numChunks).toBe(5);
    });

    it('should handle preload failures gracefully', async () => {
      // Make some chunks fail to load
      let callCount = 0;
      (zarr.get as any).mockImplementation(() => {
        callCount++;
        if (callCount % 2 === 0) {
          return Promise.reject(new Error('Failed to load chunk'));
        }
        return Promise.resolve({ data: new Float32Array(100 * 4) });
      });

      await manager.preloadChunks(mockArray, 'test', [50, 0], [1]);

      // Some chunks should still be loaded despite failures
      const stats = manager.getCacheStats();
      expect(stats.numChunks).toBeGreaterThan(0);
      expect(stats.numChunks).toBeLessThanOrEqual(3);
    });
  });

  describe('getCacheStats', () => {
    it('should return accurate cache statistics', async () => {
      const chunkData = new Float32Array(100 * 4);
      (zarr.get as any).mockResolvedValue({ data: chunkData });

      // Load some chunks
      await manager.loadChunk(mockArray, 'test1', [0, 0]);
      await manager.loadChunk(mockArray, 'test2', [1, 0]);

      const stats = manager.getCacheStats();

      expect(stats.numChunks).toBe(2);
      expect(stats.totalSizeMB).toBeCloseTo(0.003125, 3); // 2 * 1.6KB
      expect(stats.maxSizeMB).toBe(100);
      expect(stats.utilizationPercent).toBeLessThan(1);
    });
  });

  describe('clearCache', () => {
    it('should clear all cached chunks', async () => {
      const chunkData = new Float32Array(100 * 4);
      (zarr.get as any).mockResolvedValue({ data: chunkData });

      // Load some chunks
      await manager.loadChunk(mockArray, 'test', [0, 0]);
      await manager.loadChunk(mockArray, 'test', [1, 0]);

      let stats = manager.getCacheStats();
      expect(stats.numChunks).toBe(2);

      // Clear cache
      manager.clearCache();

      stats = manager.getCacheStats();
      expect(stats.numChunks).toBe(0);
      expect(stats.totalSizeMB).toBe(0);
    });
  });
});
