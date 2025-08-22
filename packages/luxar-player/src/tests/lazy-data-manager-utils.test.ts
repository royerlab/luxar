/**
 * Comprehensive tests for lazy data manager utility functions
 *
 * These tests verify the pure functions extracted for better testability.
 * Each function is tested with various inputs including edge cases.
 */

import { describe, it, expect } from 'vitest';
import {
  calculateRequiredChunks,
  getChunkId,
  calculateChunkSlice,
  calculateTotalCacheSize,
  selectChunksToEvict,
  calculateSliceParams,
  isValidChunkIndex,
  calculateDataSize,
  estimateArrayMemory,
  shouldEnableLazyLoading,
} from '../data/lazy-data-manager-utils';

describe('lazy-data-manager-utils', () => {
  describe('calculateRequiredChunks', () => {
    it('should calculate chunks for simple 2D array', () => {
      const arrayShape = [100, 10];
      const chunkShape = [20, 10];
      const slicePosition = [50, 5];
      const sliceDimensions = [1]; // Fully load dimension 1

      const chunks = calculateRequiredChunks(
        arrayShape,
        chunkShape,
        slicePosition,
        sliceDimensions,
        1
      );

      // Should load chunks around position 50 (chunk 2) with radius 1
      // That's chunks 1, 2, 3 for dim 0, and all chunks (0) for dim 1
      expect(chunks).toHaveLength(3);
      expect(chunks).toContainEqual([1, 0]);
      expect(chunks).toContainEqual([2, 0]);
      expect(chunks).toContainEqual([3, 0]);
    });

    it('should handle edge boundaries correctly', () => {
      const arrayShape = [100, 10];
      const chunkShape = [20, 10];
      const slicePosition = [0, 5]; // At the start
      const sliceDimensions: number[] = [];

      const chunks = calculateRequiredChunks(
        arrayShape,
        chunkShape,
        slicePosition,
        sliceDimensions,
        1
      );

      // Should load chunks 0 and 1 (can't go below 0)
      expect(chunks).toHaveLength(2);
      expect(chunks).toContainEqual([0, 0]);
      expect(chunks).toContainEqual([1, 0]);
    });

    it('should fully load specified dimensions', () => {
      const arrayShape = [100, 100, 10];
      const chunkShape = [50, 50, 10];
      const slicePosition = [25, 25, 5];
      const sliceDimensions = [2]; // Fully load dimension 2

      const chunks = calculateRequiredChunks(
        arrayShape,
        chunkShape,
        slicePosition,
        sliceDimensions,
        0 // No preload radius
      );

      // Should load only the center chunk for dims 0 and 1, but all for dim 2
      expect(chunks).toHaveLength(1);
      expect(chunks[0]).toEqual([0, 0, 0]);
    });

    it('should handle large preload radius', () => {
      const arrayShape = [100];
      const chunkShape = [10];
      const slicePosition = [50];
      const sliceDimensions: number[] = [];

      const chunks = calculateRequiredChunks(
        arrayShape,
        chunkShape,
        slicePosition,
        sliceDimensions,
        3 // Large radius
      );

      // Center chunk is 5, radius 3 gives chunks 2-8
      expect(chunks).toHaveLength(7);
      for (let i = 2; i <= 8; i++) {
        expect(chunks).toContainEqual([i]);
      }
    });

    it('should handle nD arrays', () => {
      const arrayShape = [10, 10, 10, 10];
      const chunkShape = [5, 5, 5, 5];
      const slicePosition = [2, 2, 7, 7];
      const sliceDimensions: number[] = [];

      const chunks = calculateRequiredChunks(
        arrayShape,
        chunkShape,
        slicePosition,
        sliceDimensions,
        0
      );

      // Should return single chunk at [0, 0, 1, 1]
      expect(chunks).toHaveLength(1);
      expect(chunks[0]).toEqual([0, 0, 1, 1]);
    });
  });

  describe('getChunkId', () => {
    it('should generate unique IDs for different chunks', () => {
      const id1 = getChunkId('positions', [0, 1, 2]);
      const id2 = getChunkId('positions', [0, 1, 3]);
      const id3 = getChunkId('colors', [0, 1, 2]);

      expect(id1).not.toBe(id2);
      expect(id1).not.toBe(id3);
      expect(id2).not.toBe(id3);
    });

    it('should generate consistent IDs for same input', () => {
      const id1 = getChunkId('data', [5, 10]);
      const id2 = getChunkId('data', [5, 10]);

      expect(id1).toBe(id2);
    });

    it('should handle empty chunk indices', () => {
      const id = getChunkId('array', []);
      expect(id).toBe('array:');
    });
  });

  describe('calculateChunkSlice', () => {
    it('should calculate overlap for fully contained slice', () => {
      const chunkIndices = [1, 0];
      const chunkShape = [100, 50];
      const arrayShape = [300, 50];
      const startIndex = [110, 10];
      const endIndex = [150, 30];

      const slice = calculateChunkSlice(chunkIndices, chunkShape, arrayShape, startIndex, endIndex);

      expect(slice).not.toBeNull();
      expect(slice!.chunkStart).toEqual([10, 10]);
      expect(slice!.chunkEnd).toEqual([50, 30]);
      expect(slice!.arrayStart).toEqual([110, 10]);
      expect(slice!.arrayEnd).toEqual([150, 30]);
    });

    it('should return null for non-overlapping ranges', () => {
      const chunkIndices = [0, 0];
      const chunkShape = [100, 50];
      const arrayShape = [300, 50];
      const startIndex = [200, 0];
      const endIndex = [250, 50];

      const slice = calculateChunkSlice(chunkIndices, chunkShape, arrayShape, startIndex, endIndex);

      expect(slice).toBeNull();
    });

    it('should handle partial overlap', () => {
      const chunkIndices = [1, 0];
      const chunkShape = [100, 50];
      const arrayShape = [300, 50];
      const startIndex = [50, 0];
      const endIndex = [150, 50];

      const slice = calculateChunkSlice(chunkIndices, chunkShape, arrayShape, startIndex, endIndex);

      expect(slice).not.toBeNull();
      expect(slice!.chunkStart).toEqual([0, 0]);
      expect(slice!.chunkEnd).toEqual([50, 50]);
    });

    it('should handle chunk at array boundary', () => {
      const chunkIndices = [2, 0];
      const chunkShape = [100, 50];
      const arrayShape = [250, 50]; // Last chunk is partial
      const startIndex = [200, 0];
      const endIndex = [250, 50];

      const slice = calculateChunkSlice(chunkIndices, chunkShape, arrayShape, startIndex, endIndex);

      expect(slice).not.toBeNull();
      expect(slice!.arrayEnd).toEqual([250, 50]);
    });
  });

  describe('calculateTotalCacheSize', () => {
    it('should sum sizes of all chunks', () => {
      const chunks = new Map([
        ['chunk1', { sizeBytes: 1000 }],
        ['chunk2', { sizeBytes: 2000 }],
        ['chunk3', { sizeBytes: 1500 }],
      ]);

      const total = calculateTotalCacheSize(chunks);
      expect(total).toBe(4500);
    });

    it('should return 0 for empty cache', () => {
      const chunks = new Map();
      const total = calculateTotalCacheSize(chunks);
      expect(total).toBe(0);
    });

    it('should handle single chunk', () => {
      const chunks = new Map([['only', { sizeBytes: 5000 }]]);

      const total = calculateTotalCacheSize(chunks);
      expect(total).toBe(5000);
    });
  });

  describe('selectChunksToEvict', () => {
    it('should evict oldest chunks first (LRU)', () => {
      const chunks = new Map([
        ['old', { sizeBytes: 1000, lastAccessed: 100 }],
        ['middle', { sizeBytes: 1000, lastAccessed: 200 }],
        ['recent', { sizeBytes: 1000, lastAccessed: 300 }],
      ]);

      const toEvict = selectChunksToEvict(chunks, 2000, 3000);

      expect(toEvict).toEqual(['old']);
    });

    it('should evict multiple chunks if needed', () => {
      const chunks = new Map([
        ['a', { sizeBytes: 500, lastAccessed: 100 }],
        ['b', { sizeBytes: 500, lastAccessed: 200 }],
        ['c', { sizeBytes: 500, lastAccessed: 300 }],
        ['d', { sizeBytes: 500, lastAccessed: 400 }],
      ]);

      const toEvict = selectChunksToEvict(chunks, 1000, 2000);

      expect(toEvict).toHaveLength(2);
      expect(toEvict).toContain('a');
      expect(toEvict).toContain('b');
    });

    it('should return empty array if under target', () => {
      const chunks = new Map([['a', { sizeBytes: 500, lastAccessed: 100 }]]);

      const toEvict = selectChunksToEvict(chunks, 1000, 500);

      expect(toEvict).toEqual([]);
    });

    it('should handle exact target size', () => {
      const chunks = new Map([
        ['a', { sizeBytes: 1000, lastAccessed: 100 }],
        ['b', { sizeBytes: 1000, lastAccessed: 200 }],
      ]);

      const toEvict = selectChunksToEvict(chunks, 2000, 2000);

      expect(toEvict).toEqual([]);
    });
  });

  describe('calculateSliceParams', () => {
    it('should calculate slice bounds with radius', () => {
      const position = [50, 50, 50];
      const shape = [100, 100, 100];
      const radius = [10, 5, 2];

      const { start, end } = calculateSliceParams(position, shape, radius);

      expect(start).toEqual([40, 45, 48]);
      expect(end).toEqual([61, 56, 53]);
    });

    it('should clamp to array boundaries', () => {
      const position = [5, 95, 50];
      const shape = [100, 100, 100];
      const radius = [10, 10, 10];

      const { start, end } = calculateSliceParams(position, shape, radius);

      expect(start).toEqual([0, 85, 40]); // First clamped to 0
      expect(end).toEqual([16, 100, 61]); // Second clamped to 100
    });

    it('should handle zero radius', () => {
      const position = [50.5, 50.5, 50.5];
      const shape = [100, 100, 100];
      const radius = [0, 0, 0];

      const { start, end } = calculateSliceParams(position, shape, radius);

      expect(start).toEqual([50, 50, 50]);
      expect(end).toEqual([52, 52, 52]); // ceil(50.5 + 0 + 1) = 52
    });
  });

  describe('isValidChunkIndex', () => {
    it('should validate valid chunk indices', () => {
      const chunkGrid = [5, 10, 3];

      expect(isValidChunkIndex([0, 0, 0], chunkGrid)).toBe(true);
      expect(isValidChunkIndex([4, 9, 2], chunkGrid)).toBe(true);
      expect(isValidChunkIndex([2, 5, 1], chunkGrid)).toBe(true);
    });

    it('should reject out-of-bounds indices', () => {
      const chunkGrid = [5, 10, 3];

      expect(isValidChunkIndex([5, 0, 0], chunkGrid)).toBe(false);
      expect(isValidChunkIndex([0, 10, 0], chunkGrid)).toBe(false);
      expect(isValidChunkIndex([0, 0, 3], chunkGrid)).toBe(false);
      expect(isValidChunkIndex([-1, 0, 0], chunkGrid)).toBe(false);
    });

    it('should reject mismatched dimensions', () => {
      const chunkGrid = [5, 10];

      expect(isValidChunkIndex([0], chunkGrid)).toBe(false);
      expect(isValidChunkIndex([0, 0, 0], chunkGrid)).toBe(false);
    });
  });

  describe('calculateDataSize', () => {
    it('should calculate size of Float32Array', () => {
      const data = new Float32Array(100);
      expect(calculateDataSize(data)).toBe(400); // 100 * 4 bytes
    });

    it('should calculate size of Uint8Array', () => {
      const data = new Uint8Array(100);
      expect(calculateDataSize(data)).toBe(100); // 100 * 1 byte
    });

    it('should calculate size of ArrayBuffer', () => {
      const data = new ArrayBuffer(512);
      expect(calculateDataSize(data)).toBe(512);
    });
  });

  describe('estimateArrayMemory', () => {
    it('should estimate float32 array memory', () => {
      const shape = [1000, 100];
      const dtype = '<f4';

      const size = estimateArrayMemory(shape, dtype);
      expect(size).toBe(400000); // 100000 * 4 bytes
    });

    it('should estimate uint8 array memory', () => {
      const shape = [1000, 100];
      const dtype = '|u1';

      const size = estimateArrayMemory(shape, dtype);
      expect(size).toBe(100000); // 100000 * 1 byte
    });

    it('should handle float64', () => {
      const shape = [100, 100];
      const dtype = '<f8';

      const size = estimateArrayMemory(shape, dtype);
      expect(size).toBe(80000); // 10000 * 8 bytes
    });

    it('should default to float32 for unknown dtype', () => {
      const shape = [100, 100];
      const dtype = 'unknown';

      const size = estimateArrayMemory(shape, dtype);
      expect(size).toBe(40000); // 10000 * 4 bytes (default)
    });
  });

  describe('shouldEnableLazyLoading', () => {
    it('should enable for large datasets', () => {
      const arrayShape = [1000000, 100]; // ~400MB for float32
      const dtype = '<f4';
      const maxMemoryMB = 500;

      const shouldEnable = shouldEnableLazyLoading(arrayShape, dtype, maxMemoryMB);
      expect(shouldEnable).toBe(true);
    });

    it('should disable for small datasets', () => {
      const arrayShape = [1000, 100]; // ~0.4MB for float32
      const dtype = '<f4';
      const maxMemoryMB = 500;

      const shouldEnable = shouldEnableLazyLoading(arrayShape, dtype, maxMemoryMB);
      expect(shouldEnable).toBe(false);
    });

    it('should use 50% threshold', () => {
      const arrayShape = [65537, 100]; // ~25.0004MB for float32
      const dtype = '<f4';
      const maxMemoryMB = 50;

      const shouldEnable = shouldEnableLazyLoading(arrayShape, dtype, maxMemoryMB);
      expect(shouldEnable).toBe(true); // 25.0004MB > 25MB (50% of 50MB)
    });

    it('should handle uint8 data correctly', () => {
      const arrayShape = [30000000]; // ~30MB for uint8
      const dtype = '|u1';
      const maxMemoryMB = 50;

      const shouldEnable = shouldEnableLazyLoading(arrayShape, dtype, maxMemoryMB);
      expect(shouldEnable).toBe(true);
    });
  });
});
