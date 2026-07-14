/**
 * Performance regression tests for GPU Buffer Pool
 *
 * Verifies geometry reuse rates, allocation elimination, and memory efficiency.
 * These tests validate that the pool actually improves performance, not just correctness.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { LoadedPointsData } from '../../../data/data-loader-types';
import * as THREE from 'three';

describe('GPU Buffer Pool Performance Regression Tests', () => {
  let pool: GPUBufferPool;

  // Helper to create mock data
  const createMockData = (
    count: number,
    colorType: 'Float32Array' | 'Uint8Array' = 'Float32Array'
  ): LoadedPointsData => ({
    positions: new Float32Array(count * 3).fill(1),
    colors:
      colorType === 'Uint8Array'
        ? new Uint8Array(count * 3).fill(128)
        : new Float32Array(count * 3).fill(0.5),
    radii: new Float32Array(count).fill(0.5),
    sharpness: new Float32Array(count).fill(2.0),
    pointCount: count,
    ndim: 3,
    metadata: {
      totalPoints: count,
      loadedPoints: count,
      bounds: new THREE.Box3(),
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

      // performance.md C4[P2] fix: prior `reuseRate > 0.2` was a very loose
      // threshold for a deterministic counter test (no async / GPU / clock).
      // The acquire pattern is fully deterministic — 100 ops across 10 node
      // IDs (mod 10) with 5 varying counts (mod 5), releasing every 3rd op.
      // Measured outcome: 7 allocations + 93 reuses. The variation in
      // `count` widens the geometry's resize path on first acquire only;
      // subsequent acquires reuse. Tighten the band to catch a halving
      // regression (rate dropping below 0.7).
      const reuseRate = stats.reuses / operations;
      expect(reuseRate).toBeGreaterThanOrEqual(0.85);
      expect(reuseRate).toBeLessThanOrEqual(1.0);
      // Pin exact allocation count: 7 (deterministic given the pattern).
      expect(stats.allocations).toBe(7);
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
        shortEvictionPool.beginFrame();
        shortEvictionPool.acquirePointsGeometry(`active${i}`, createMockData(5000), 5000);
      }

      // Trigger a final sweep. Acquire paths now sweep idle buffers
      // themselves (byte-budget-on-growth fix), so evictions may fire
      // during the loop above — assert the cumulative outcome via stats.
      shortEvictionPool.evictUnused();

      // Should have evicted old geometries (at least some)
      expect(shortEvictionPool.getStats().evictions).toBeGreaterThan(0);
    });
  });

  describe('Type-Aware Performance', () => {
    it('handles Float32 and Uint8 source dtypes uniformly via Float32 interleaved storage', () => {
      // Post-interleaving, pooled storage is uniformly Float32 in one
      // shared `InstancedInterleavedBuffer` (so multiple attributes
      // collapse to one vertex-buffer slot under WebGPU). Uint8 source
      // data is widened (÷255) at upload time to preserve the shader-
      // visible [0, 1] range.
      const dataFloat = createMockData(1000, 'Float32Array');
      const dataUint8 = createMockData(1000, 'Uint8Array');

      const geom1 = pool.acquirePointsGeometry('node1', dataFloat, 1000);
      const geom2 = pool.acquirePointsGeometry('node2', dataUint8, 1000);

      const col1 = geom1.getAttribute('aColor') as THREE.InterleavedBufferAttribute;
      const col2 = geom2.getAttribute('aColor') as THREE.InterleavedBufferAttribute;

      // Both back ends to the same Float32 interleaved layout.
      expect(col1.data.array).toBeInstanceOf(Float32Array);
      expect(col2.data.array).toBeInstanceOf(Float32Array);
    });

    it('memory accounting reflects the widen-to-Float32 trade-off', () => {
      // Post-interleaving cost: Uint8 source widens to Float32 in
      // the pooled buffer (4× the per-instance bytes for that
      // attribute), bought against the WebGPU buffer-count win
      // (one vertex-buffer slot instead of N). The shader sees the
      // same [0, 1] range as before.
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

      const colUint8 = geomUint8.getAttribute('aColor') as THREE.InterleavedBufferAttribute;
      const colFloat = geomFloat.getAttribute('aColor') as THREE.InterleavedBufferAttribute;

      // Both share the same Float32 stride layout (per-buffer storage
      // size depends only on `capacity × stride`, not on source dtype).
      expect(colUint8.data.array.byteLength).toBe(colFloat.data.array.byteLength);
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
