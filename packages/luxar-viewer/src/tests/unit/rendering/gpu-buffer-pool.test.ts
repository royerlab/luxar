/**
 * Unit tests for GPU Buffer Pool
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { PointsData } from '../../../data/data-loader-types';
import * as THREE from 'three';

describe('GPUBufferPool', () => {
  let pool: GPUBufferPool;

  // Helper to create mock PointsData
  const createMockPointsData = (
    count: number,
    colorType: 'Float32Array' | 'Uint8Array' | 'Uint16Array' = 'Float32Array'
  ): PointsData => {
    const colors =
      colorType === 'Uint8Array' ? new Uint8Array(count * 3).fill(128) :
        colorType === 'Uint16Array' ? new Uint16Array(count * 3).fill(32768) :
          new Float32Array(count * 3).fill(0.5);

    return {
      positions: new Float32Array(count * 3),
      colors,
      radii: new Float32Array(count).fill(1.0),
      sharpness: new Float32Array(count).fill(2.0),
      metadata: {
        totalPoints: count,
        loadedPoints: count,
        bounds: new THREE.Box3(),
        ndim: 3,
        usedSpatialIndex: true,
      },
    };
  };

  beforeEach(() => {
    pool = new GPUBufferPool(20, 300); // maxPoolSize=20, evictionFrames=300
  });

  describe('Points Geometry', () => {
    it('should allocate new geometry on first request', () => {
      const data = createMockPointsData(1000);
      const geom = pool.acquirePointsGeometry('node1', data, 1000);
      expect(geom).toBeInstanceOf(THREE.BufferGeometry);
      expect(geom.getAttribute('position')).toBeDefined();
      expect(geom.getAttribute('color')).toBeDefined();
      expect(geom.getAttribute('radius')).toBeDefined();
      expect(geom.getAttribute('sharpness')).toBeDefined();

      const stats = pool.getStats();
      expect(stats.allocations).toBe(1);
      expect(stats.reuses).toBe(0);
    });

    it('should reuse geometry when requesting same node again', () => {
      const data1 = createMockPointsData(1000);
      const data2 = createMockPointsData(900);
      const geom1 = pool.acquirePointsGeometry('node1', data1, 1000);
      const geom2 = pool.acquirePointsGeometry('node1', data2, 900); // Same node, smaller count

      expect(geom2).toBe(geom1); // Same geometry instance
      const stats = pool.getStats();
      expect(stats.allocations).toBe(1);
      expect(stats.reuses).toBe(1); // Second call reused
    });

    it('should grow geometry when count exceeds capacity', () => {
      const data1 = createMockPointsData(1000);
      const data2 = createMockPointsData(2000);
      const geom1 = pool.acquirePointsGeometry('node1', data1, 1000); // Capacity ~1500
      const geom2 = pool.acquirePointsGeometry('node1', data2, 2000); // Needs >1500

      expect(geom2).toBe(geom1); // Same geometry, grown
      const stats = pool.getStats();
      expect(stats.capacityGrowths).toBe(1);
    });

    it('should release geometry back to pool', () => {
      const data = createMockPointsData(1000);
      pool.acquirePointsGeometry('node1', data, 1000);
      pool.releasePointsGeometry('node1');

      const stats = pool.getStats();
      expect(stats.activeBuffers).toBe(0);
      expect(stats.pooledBuffers).toBe(1);
    });

    it('should reuse released geometry for new node', () => {
      const data1 = createMockPointsData(1000);
      const data2 = createMockPointsData(800);
      pool.acquirePointsGeometry('node1', data1, 1000);
      pool.releasePointsGeometry('node1');

      pool.acquirePointsGeometry('node2', data2, 800); // Different node, similar size, same types
      const stats = pool.getStats();
      expect(stats.allocations).toBe(1); // Only one allocation
      expect(stats.reuses).toBe(1); // Reused for node2
    });
  });

  describe('Lines Geometry', () => {
    it('should create InstancedBufferGeometry for lines', () => {
      const geom = pool.acquireLinesGeometry('line1', 500);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);

      // Check instance attributes
      expect(geom.getAttribute('aStartPos')).toBeDefined();
      expect(geom.getAttribute('aEndPos')).toBeDefined();
      expect(geom.getAttribute('aStartColor')).toBeDefined();
      expect(geom.getAttribute('aSegmentLength')).toBeDefined();
    });

    it('should reuse lines geometry', () => {
      const geom1 = pool.acquireLinesGeometry('line1', 500);
      const geom2 = pool.acquireLinesGeometry('line1', 400);

      expect(geom2).toBe(geom1);
      expect(pool.getStats().reuses).toBe(1);
    });

    it('should handle lines release and reuse', () => {
      pool.acquireLinesGeometry('line1', 500);
      pool.releaseLinesGeometry('line1');

      pool.acquireLinesGeometry('line2', 450);
      expect(pool.getStats().allocations).toBe(1);
      expect(pool.getStats().reuses).toBe(1);
    });
  });

  describe('GSplats Geometry', () => {
    it('should create InstancedBufferGeometry for gsplats', () => {
      const geom = pool.acquireGSplatsGeometry('splat1', 300);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);

      // Check instance attributes
      expect(geom.getAttribute('aCenter')).toBeDefined();
      expect(geom.getAttribute('aCholesky01')).toBeDefined();
      expect(geom.getAttribute('aCholesky23')).toBeDefined();
      expect(geom.getAttribute('aCholesky45')).toBeDefined();
      expect(geom.getAttribute('aAmplitude')).toBeDefined();
    });

    it('should reuse gsplats geometry', () => {
      const geom1 = pool.acquireGSplatsGeometry('splat1', 300);
      const geom2 = pool.acquireGSplatsGeometry('splat1', 250);

      expect(geom2).toBe(geom1);
      expect(pool.getStats().reuses).toBe(1);
    });
  });

  describe('LRU Eviction', () => {
    it('should evict geometries unused for >evictionFrames', () => {
      // Create pool with short eviction time for testing
      const testPool = new GPUBufferPool(20, 2); // Evict after 2 frames

      // Acquire and release several different-sized geometries
      // This ensures they go to different buckets and won't be reused
      testPool.acquirePointsGeometry('node1', createMockPointsData(1000), 1000);
      testPool.releasePointsGeometry('node1');

      testPool.acquirePointsGeometry('node2', createMockPointsData(10000), 10000);
      testPool.releasePointsGeometry('node2');

      // Advance frameCount by 3 frames without touching the pooled geometries
      for (let i = 0; i < 3; i++) {
        testPool.acquirePointsGeometry(`active${i}`, createMockPointsData(100000), 100000); // Different size bucket
        // Don't release - keep active
      }

      // Now evict - the first two should be evicted (unused for >2 frames)
      const evicted = testPool.evictUnused();

      expect(evicted).toBeGreaterThanOrEqual(2);
    });

    it('should dispose all geometries on pool disposal', () => {
      pool.acquirePointsGeometry('node1', createMockPointsData(1000), 1000);
      pool.acquireLinesGeometry('line1', 500);
      pool.acquireGSplatsGeometry('splat1', 300);

      pool.dispose();

      const stats = pool.getStats();
      expect(stats.activeBuffers).toBe(0);
      expect(stats.pooledBuffers).toBe(0);
    });
  });

  describe('Size Bucketing', () => {
    it('should use appropriate size buckets', () => {
      // Acquire geometries of different sizes
      pool.acquirePointsGeometry('small', createMockPointsData(500), 500); // → 1K bucket
      pool.acquirePointsGeometry('medium', createMockPointsData(3000), 3000); // → 5K bucket
      pool.acquirePointsGeometry('large', createMockPointsData(20000), 20000); // → 50K bucket

      // Release them
      pool.releasePointsGeometry('small');
      pool.releasePointsGeometry('medium');
      pool.releasePointsGeometry('large');

      // Acquire similar sizes - should reuse from correct buckets
      pool.acquirePointsGeometry('small2', createMockPointsData(600), 600); // Should reuse from 1K bucket
      pool.acquirePointsGeometry('medium2', createMockPointsData(4000), 4000); // Should reuse from 5K bucket

      const stats = pool.getStats();
      expect(stats.reuses).toBeGreaterThanOrEqual(2);
    });
  });

  describe('Statistics Tracking', () => {
    it('should track allocations, reuses, and evictions', () => {
      // Allocate
      pool.acquirePointsGeometry('node1', createMockPointsData(1000), 1000);
      expect(pool.getStats().allocations).toBe(1);

      // Reuse
      pool.acquirePointsGeometry('node1', createMockPointsData(900), 900);
      expect(pool.getStats().reuses).toBe(1);

      // Growth
      pool.acquirePointsGeometry('node1', createMockPointsData(3000), 3000);
      expect(pool.getStats().capacityGrowths).toBe(1);

      // Eviction (tested separately due to frame requirements)
    });

    it('should track active vs pooled buffers', () => {
      pool.acquirePointsGeometry('node1', createMockPointsData(1000), 1000);
      pool.acquireLinesGeometry('line1', 500);

      let stats = pool.getStats();
      expect(stats.activeBuffers).toBe(2);
      expect(stats.pooledBuffers).toBe(0);

      pool.releasePointsGeometry('node1');

      stats = pool.getStats();
      expect(stats.activeBuffers).toBe(1);
      expect(stats.pooledBuffers).toBe(1);
    });
  });

  describe('Multi-Type Support', () => {
    it('should create geometry with Uint8Array colors', () => {
      const data = createMockPointsData(1000, 'Uint8Array');
      const geom = pool.acquirePointsGeometry('node1', data, 1000);

      const colorAttr = geom.getAttribute('color') as THREE.BufferAttribute;
      expect(colorAttr.array).toBeInstanceOf(Uint8Array);
      expect(colorAttr.normalized).toBe(true); // Should be normalized
    });

    it('should create geometry with Uint16Array colors', () => {
      const data = createMockPointsData(1000, 'Uint16Array');
      const geom = pool.acquirePointsGeometry('node1', data, 1000);

      const colorAttr = geom.getAttribute('color') as THREE.BufferAttribute;
      expect(colorAttr.array).toBeInstanceOf(Uint16Array);
      expect(colorAttr.normalized).toBe(true); // Should be normalized
    });

    it('should NOT reuse geometry when types differ', () => {
      const dataFloat = createMockPointsData(1000, 'Float32Array');
      const dataUint8 = createMockPointsData(900, 'Uint8Array');

      pool.acquirePointsGeometry('node1', dataFloat, 1000);
      pool.releasePointsGeometry('node1');

      pool.acquirePointsGeometry('node2', dataUint8, 900); // Different type!

      const stats = pool.getStats();
      // Should create NEW geometry (different types, can't reuse)
      expect(stats.allocations).toBe(2);
      expect(stats.reuses).toBe(0);
    });

    it('should reuse geometry when types match', () => {
      const data1 = createMockPointsData(1000, 'Uint8Array');
      const data2 = createMockPointsData(900, 'Uint8Array'); // Same type

      pool.acquirePointsGeometry('node1', data1, 1000);
      pool.releasePointsGeometry('node1');

      pool.acquirePointsGeometry('node2', data2, 900); // Same type, smaller size

      const stats = pool.getStats();
      // Should REUSE geometry (same types)
      expect(stats.allocations).toBe(1);
      expect(stats.reuses).toBe(1);
    });
  });
});
