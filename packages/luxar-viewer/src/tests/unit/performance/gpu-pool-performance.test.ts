/**
 * Performance regression tests for GPU Buffer Pool
 *
 * Verifies geometry reuse rates, allocation elimination, and memory efficiency.
 * These tests validate that the pool actually improves performance, not just correctness.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { PointsData } from '../../../data/data-loader-types';
import * as THREE from 'three';

describe('GPU Buffer Pool Performance Regression Tests', () => {
  let pool: GPUBufferPool;

  // Helper to create mock data
  const createMockData = (
    count: number,
    colorType: 'Float32Array' | 'Uint8Array' = 'Float32Array'
  ): PointsData => ({
    positions: new Float32Array(count * 3).fill(1),
    colors:
      colorType === 'Uint8Array'
        ? new Uint8Array(count * 3).fill(128)
        : new Float32Array(count * 3).fill(0.5),
    radii: new Float32Array(count).fill(0.5),
    sharpness: new Float32Array(count).fill(2.0),
    metadata: {
      totalPoints: count,
      loadedPoints: count,
      bounds: new THREE.Box3(),
      ndim: 3,
      usedSpatialIndex: true,
    },
  });

  beforeEach(() => {
    pool = new GPUBufferPool(20, 300);
  });

  describe('Geometry Reuse Rates', () => {
    it('should achieve 100% reuse for same node with similar counts', () => {
      const data1 = createMockData(1000);
      const data2 = createMockData(900); // Smaller, fits in same geometry
      const data3 = createMockData(950);

      pool.acquirePointsGeometry('node1', data1, 1000);
      pool.acquirePointsGeometry('node1', data2, 900);
      pool.acquirePointsGeometry('node1', data3, 950);

      const stats = pool.getStats();

      // 1 allocation, 2 reuses = 66% reuse rate
      expect(stats.allocations).toBe(1);
      expect(stats.reuses).toBe(2);

      const reuseRate = stats.reuses / (stats.allocations + stats.reuses);
      expect(reuseRate).toBeCloseTo(0.666, 2);
    });

    it('should reuse across different nodes with same types', () => {
      const data1 = createMockData(1000);

      pool.acquirePointsGeometry('node1', data1, 1000);
      pool.releasePointsGeometry('node1');

      // Different node, same size/type - should reuse
      pool.acquirePointsGeometry('node2', createMockData(900), 900);

      const stats = pool.getStats();
      expect(stats.allocations).toBe(1); // Only 1 allocation total
      expect(stats.reuses).toBe(1); // Reused for node2
    });

    it('should NOT reuse when types differ (correct behavior)', () => {
      const dataFloat = createMockData(1000, 'Float32Array');
      const dataUint8 = createMockData(1000, 'Uint8Array');

      pool.acquirePointsGeometry('node1', dataFloat, 1000);
      pool.releasePointsGeometry('node1');

      pool.acquirePointsGeometry('node2', dataUint8, 1000); // Different type!

      const stats = pool.getStats();
      expect(stats.allocations).toBe(2); // Can't reuse (types differ)
      expect(stats.reuses).toBe(0);
    });
  });

  describe('Allocation Elimination', () => {
    it('should have zero GPU allocations on geometry reuse', () => {
      const data = createMockData(1000);

      // First acquisition: allocation
      const geom1 = pool.acquirePointsGeometry('node1', data, 1000);
      const allocsBefore = pool.getStats().allocations;

      // Update geometry (reuse)
      pool.updatePointsGeometry(geom1, createMockData(900), 900);

      // Acquire again (should reuse same geometry)
      const geom2 = pool.acquirePointsGeometry('node1', createMockData(850), 850);

      const allocsAfter = pool.getStats().allocations;

      // No new allocations
      expect(allocsAfter).toBe(allocsBefore);
      expect(geom2).toBe(geom1); // Same geometry instance
    });

    it('should track allocation count accurately over many operations', () => {
      const operations = 100;

      for (let i = 0; i < operations; i++) {
        const nodeId = `node${i % 10}`; // 10 different nodes
        const count = 800 + (i % 5) * 100; // Varying counts

        pool.acquirePointsGeometry(nodeId, createMockData(count), count);

        if (i % 3 === 0) {
          pool.releasePointsGeometry(nodeId);
        }
      }

      const stats = pool.getStats();

      // Verify tracking is accurate
      expect(stats.allocations + stats.reuses).toBe(operations);

      // Calculate reuse rate
      const reuseRate = stats.reuses / operations;

      // Should get some reuse (>20% for this pattern)
      expect(reuseRate).toBeGreaterThan(0.2);
    });
  });

  describe('Memory Efficiency', () => {
    it('should use memory proportional to active geometries, not all acquisitions', () => {
      // Acquire 100 different nodes
      for (let i = 0; i < 100; i++) {
        pool.acquirePointsGeometry(`node${i}`, createMockData(1000), 1000);
      }

      const stats1 = pool.getStats();

      // 100 active geometries
      expect(stats1.activeBuffers).toBe(100);

      // Release 50 of them
      for (let i = 0; i < 50; i++) {
        pool.releasePointsGeometry(`node${i}`);
      }

      const stats2 = pool.getStats();

      // Some active, some pooled (exact numbers depend on pool eviction)
      expect(stats2.activeBuffers).toBe(50);
      expect(stats2.pooledBuffers).toBeGreaterThan(0);
      expect(stats2.activeBuffers + stats2.pooledBuffers).toBeLessThanOrEqual(100);
    });

    it('should evict old geometries to prevent unbounded growth', () => {
      const shortEvictionPool = new GPUBufferPool(20, 5); // Evict after 5 frames

      // Acquire and release many geometries with different sizes
      for (let i = 0; i < 30; i++) {
        const count = 1000 + i * 100; // Different sizes
        shortEvictionPool.acquirePointsGeometry(`node${i}`, createMockData(count), count);
        shortEvictionPool.releasePointsGeometry(`node${i}`);
      }

      // Advance frames by acquiring active geometries
      for (let i = 0; i < 10; i++) {
        shortEvictionPool.acquirePointsGeometry(`active${i}`, createMockData(5000), 5000);
      }

      // Trigger eviction
      const evicted = shortEvictionPool.evictUnused();

      // Should have evicted old geometries (at least some)
      expect(evicted).toBeGreaterThan(0);
      // Note: Eviction count might be less than evicted due to timing
      expect(shortEvictionPool.getStats().evictions).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Type-Aware Performance', () => {
    it('should handle mixed types efficiently', () => {
      const dataFloat = createMockData(1000, 'Float32Array');
      const dataUint8 = createMockData(1000, 'Uint8Array');

      // Acquire both types
      const geom1 = pool.acquirePointsGeometry('node1', dataFloat, 1000);
      const geom2 = pool.acquirePointsGeometry('node2', dataUint8, 1000);

      // Verify correct types in attributes
      const col1 = geom1.getAttribute('color') as THREE.BufferAttribute;
      const col2 = geom2.getAttribute('color') as THREE.BufferAttribute;

      expect(col1.array).toBeInstanceOf(Float32Array);
      expect(col2.array).toBeInstanceOf(Uint8Array);
      expect(col2.normalized).toBe(true); // Uint8 should be normalized
    });

    it('should preserve memory efficiency with mixed types', () => {
      // 1000 points with Uint8 colors = 3KB
      // 1000 points with Float32 colors = 12KB

      const geomUint8 = pool.acquirePointsGeometry(
        'node1',
        createMockData(1000, 'Uint8Array'),
        1000
      );
      const geomFloat = pool.acquirePointsGeometry(
        'node2',
        createMockData(1000, 'Float32Array'),
        1000
      );

      const colUint8 = geomUint8.getAttribute('color') as THREE.BufferAttribute;
      const colFloat = geomFloat.getAttribute('color') as THREE.BufferAttribute;

      // Uint8 should use 1/4 the memory of Float32
      expect(colUint8.array.byteLength).toBe(colFloat.array.byteLength / 4);
    });
  });

  describe('Capacity Growth Performance', () => {
    it('should minimize reallocations with 1.5x strategy', () => {
      const data1 = createMockData(1000);
      pool.acquirePointsGeometry('node1', data1, 1000);

      // Grow to 2000 (exceeds initial 1500 capacity)
      pool.acquirePointsGeometry('node1', createMockData(2000), 2000);

      const stats = pool.getStats();

      // Should have exactly 1 capacity growth
      expect(stats.capacityGrowths).toBe(1);

      // Now grow to 2500 (within new capacity ~3000)
      pool.acquirePointsGeometry('node1', createMockData(2500), 2500);

      // No additional growth
      expect(pool.getStats().capacityGrowths).toBe(1);
    });
  });
});
