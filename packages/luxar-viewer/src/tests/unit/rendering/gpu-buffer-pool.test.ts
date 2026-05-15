/**
 * Unit tests for GPU Buffer Pool
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { LoadedPointsData } from '../../../data/data-loader-types';
import * as THREE from 'three';

describe('GPUBufferPool', () => {
  let pool: GPUBufferPool;

  // Helper to create mock LoadedPointsData
  const createMockLoadedPointsData = (
    count: number,
    colorType: 'Float32Array' | 'Uint8Array' | 'Uint16Array' = 'Float32Array'
  ): LoadedPointsData => {
    const colors =
      colorType === 'Uint8Array'
        ? new Uint8Array(count * 3).fill(128)
        : colorType === 'Uint16Array'
          ? new Uint16Array(count * 3).fill(32768)
          : new Float32Array(count * 3).fill(0.5);

    return {
      positions: new Float32Array(count * 3),
      colors,
      radii: new Float32Array(count).fill(1.0),
      sharpness: new Float32Array(count).fill(2.0),
      pointCount: count,
      ndim: 3,
      metadata: {
        totalPoints: count,
        loadedPoints: count,
        bounds: new THREE.Box3(),
        usedSpatialIndex: true,
      },
    };
  };

  beforeEach(() => {
    pool = new GPUBufferPool(20, 300); // maxPoolSize=20, evictionFrames=300
  });

  describe('Points Geometry', () => {
    it('should allocate new geometry on first request', () => {
      const data = createMockLoadedPointsData(1000);
      const geom = pool.acquirePointsGeometry('node1', data, 1000);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);
      expect((geom as THREE.InstancedBufferGeometry).instanceCount).toBe(1000);
      expect(geom.drawRange.count).toBe(6);
      expect(geom.getAttribute('aCenter')).toBeDefined();
      expect(geom.getAttribute('aColor')).toBeDefined();
      expect(geom.getAttribute('aRadius')).toBeDefined();
      expect(geom.getAttribute('aSharpness')).toBeDefined();

      const stats = pool.getStats();
      expect(stats.allocations).toBe(1);
      expect(stats.reuses).toBe(0);
    });

    it('should reuse geometry when requesting same node again', () => {
      const data1 = createMockLoadedPointsData(1000);
      const data2 = createMockLoadedPointsData(900);
      const geom1 = pool.acquirePointsGeometry('node1', data1, 1000);
      const geom2 = pool.acquirePointsGeometry('node1', data2, 900); // Same node, smaller count

      expect(geom2).toBe(geom1); // Same geometry instance
      const stats = pool.getStats();
      expect(stats.allocations).toBe(1);
      expect(stats.reuses).toBe(1); // Second call reused
    });

    it('should grow geometry when count exceeds capacity', () => {
      const data1 = createMockLoadedPointsData(1000);
      const data2 = createMockLoadedPointsData(2000);
      const geom1 = pool.acquirePointsGeometry('node1', data1, 1000); // Capacity ~1500
      const geom2 = pool.acquirePointsGeometry('node1', data2, 2000); // Needs >1500

      expect(geom2).toBe(geom1); // Same geometry, grown
      const stats = pool.getStats();
      expect(stats.capacityGrowths).toBe(1);
    });

    it('should release geometry back to pool', () => {
      const data = createMockLoadedPointsData(1000);
      pool.acquirePointsGeometry('node1', data, 1000);
      pool.releasePointsGeometry('node1');

      const stats = pool.getStats();
      expect(stats.activeBuffers).toBe(0);
      expect(stats.pooledBuffers).toBe(1);
    });

    it('should reuse released geometry for new node', () => {
      const data1 = createMockLoadedPointsData(1000);
      const data2 = createMockLoadedPointsData(800);
      pool.acquirePointsGeometry('node1', data1, 1000);
      pool.releasePointsGeometry('node1');

      pool.acquirePointsGeometry('node2', data2, 800); // Different node, similar size, same types
      const stats = pool.getStats();
      expect(stats.allocations).toBe(1); // Only one allocation
      expect(stats.reuses).toBe(1); // Reused for node2
    });

    it('positions-only Points get default radius=0.5 and sharpness=2.0', () => {
      const positionsOnly: LoadedPointsData = {
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]),
        pointCount: 3,
        ndim: 3,
        metadata: {
          totalPoints: 3,
          loadedPoints: 3,
          bounds: new THREE.Box3(),
          usedSpatialIndex: true,
        },
      };
      const geom = pool.acquirePointsGeometry('p1', positionsOnly, 3);
      pool.updatePointsGeometry(geom, positionsOnly, 3);

      const radAttr = geom.getAttribute('aRadius');
      const sharpAttr = geom.getAttribute('aSharpness');
      const colAttr = geom.getAttribute('aColor');

      // Active range filled with defaults — not zeros. Use the
      // semantic per-instance accessors so the test works regardless
      // of whether the underlying storage is standalone or interleaved.
      for (let i = 0; i < 3; i++) {
        expect(radAttr.getX(i)).toBeCloseTo(0.5);
        expect(sharpAttr.getX(i)).toBeCloseTo(2.0);
        // White default color (R, G, B = 1.0).
        expect(colAttr.getX(i)).toBeCloseTo(1.0);
        expect(colAttr.getY(i)).toBeCloseTo(1.0);
        expect(colAttr.getZ(i)).toBeCloseTo(1.0);
      }
    });

    it('pool reuse fills defaults when subsequent commit lacks colors/radii', () => {
      // First commit: full data including colors/radii/sharpness
      const full = createMockLoadedPointsData(4);
      const geom = pool.acquirePointsGeometry('p2', full, 4);
      pool.updatePointsGeometry(geom, full, 4);

      // Second commit: same node, but data has no colors/radii/sharpness
      // (e.g. nD slice change reveals points without those optional attrs).
      // Pool reuses the same geometry since types match (Float32 default).
      const sparse: LoadedPointsData = {
        positions: new Float32Array(12), // 4 points × 3
        pointCount: 4,
        ndim: 3,
        metadata: {
          totalPoints: 4,
          loadedPoints: 4,
          bounds: new THREE.Box3(),
          usedSpatialIndex: true,
        },
      };
      pool.updatePointsGeometry(geom, sparse, 4);

      const radAttr = geom.getAttribute('aRadius');
      const colAttr = geom.getAttribute('aColor');
      // Defaults overwrite values left in the reused buffer from `full`.
      // Semantic accessors handle the interleaved storage transparently.
      for (let i = 0; i < 4; i++) {
        expect(radAttr.getX(i)).toBeCloseTo(0.5);
        expect(colAttr.getX(i)).toBeCloseTo(1.0);
        expect(colAttr.getY(i)).toBeCloseTo(1.0);
        expect(colAttr.getZ(i)).toBeCloseTo(1.0);
      }
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
      testPool.acquirePointsGeometry('node1', createMockLoadedPointsData(1000), 1000);
      testPool.releasePointsGeometry('node1');

      testPool.acquirePointsGeometry('node2', createMockLoadedPointsData(10000), 10000);
      testPool.releasePointsGeometry('node2');

      // Advance frameCount by 3 frames without touching the pooled geometries
      for (let i = 0; i < 3; i++) {
        testPool.beginFrame();
        testPool.acquirePointsGeometry(`active${i}`, createMockLoadedPointsData(100000), 100000); // Different size bucket
        // Don't release - keep active
      }

      // Now evict - the first two should be evicted (unused for >2 frames)
      const evicted = testPool.evictUnused();

      expect(evicted).toBeGreaterThanOrEqual(2);
    });

    it('should dispose all geometries on pool disposal', () => {
      pool.acquirePointsGeometry('node1', createMockLoadedPointsData(1000), 1000);
      pool.acquireLinesGeometry('line1', 500);
      pool.acquireGSplatsGeometry('splat1', 300);

      pool.dispose();

      const stats = pool.getStats();
      expect(stats.activeBuffers).toBe(0);
      expect(stats.pooledBuffers).toBe(0);
    });

    it('acquire under pool pressure returns a usable geometry', () => {
      // Stress the pool's release → evict → acquire sequence.
      // Eviction in this codebase is triggered by `releasePointsGeometry`
      // (which calls `evictUnused()` internally), not by acquire — so
      // this test fills the pool, advances frames past evictionFrames,
      // releases to trigger eviction, then asserts the next acquire
      // returns a valid, undisposed geometry. A dispose-during-pool-
      // churn bug would surface as either a thrown error inside
      // acquire or as an already-disposed `attributes.aCenter`.
      const tinyPool = new GPUBufferPool(2, 0); // maxPoolSize=2, evictionFrames=0
      const dataA = createMockLoadedPointsData(1000);
      const dataB = createMockLoadedPointsData(2000);

      tinyPool.acquirePointsGeometry('nodeA', dataA, 1000);
      tinyPool.releasePointsGeometry('nodeA');
      tinyPool.acquirePointsGeometry('nodeB', dataB, 2000);
      tinyPool.releasePointsGeometry('nodeB');

      tinyPool.beginFrame();
      tinyPool.beginFrame();

      const dataC = createMockLoadedPointsData(100000);
      const geomC = tinyPool.acquirePointsGeometry('nodeC', dataC, 100000);

      expect(geomC).toBeDefined();
      expect(geomC.attributes.aCenter).toBeDefined();
      expect(() => tinyPool.releasePointsGeometry('nodeC')).not.toThrow();
    });
  });

  describe('Size Bucketing', () => {
    it('should use appropriate size buckets', () => {
      // Acquire geometries of different sizes
      pool.acquirePointsGeometry('small', createMockLoadedPointsData(500), 500); // → 1K bucket
      pool.acquirePointsGeometry('medium', createMockLoadedPointsData(3000), 3000); // → 5K bucket
      pool.acquirePointsGeometry('large', createMockLoadedPointsData(20000), 20000); // → 50K bucket

      // Release them
      pool.releasePointsGeometry('small');
      pool.releasePointsGeometry('medium');
      pool.releasePointsGeometry('large');

      // Acquire similar sizes - should reuse from correct buckets
      pool.acquirePointsGeometry('small2', createMockLoadedPointsData(600), 600); // Should reuse from 1K bucket
      pool.acquirePointsGeometry('medium2', createMockLoadedPointsData(4000), 4000); // Should reuse from 5K bucket

      const stats = pool.getStats();
      expect(stats.reuses).toBeGreaterThanOrEqual(2);
    });
  });

  describe('Statistics Tracking', () => {
    it('should track allocations, reuses, and evictions', () => {
      // Allocate
      pool.acquirePointsGeometry('node1', createMockLoadedPointsData(1000), 1000);
      expect(pool.getStats().allocations).toBe(1);

      // Reuse
      pool.acquirePointsGeometry('node1', createMockLoadedPointsData(900), 900);
      expect(pool.getStats().reuses).toBe(1);

      // Growth
      pool.acquirePointsGeometry('node1', createMockLoadedPointsData(3000), 3000);
      expect(pool.getStats().capacityGrowths).toBe(1);

      // Eviction (tested separately due to frame requirements)
    });

    it('should track active vs pooled buffers', () => {
      pool.acquirePointsGeometry('node1', createMockLoadedPointsData(1000), 1000);
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
    it('widens Uint8Array color source to Float32 [0,1] in interleaved storage', () => {
      // Post-interleaving, the pool stores ALL per-instance attributes
      // as Float32 in a shared `InstancedInterleavedBuffer` so multiple
      // attributes collapse to one vertex-buffer slot under WebGPU.
      // Uint8 source data (range [0, 255]) is widened with the
      // `normalized: true` divisor (255) at upload time so the shader
      // sees the same [0, 1] range as before. Memory cost: 4× the
      // color buffer; trivially small in absolute terms.
      const data = createMockLoadedPointsData(1000, 'Uint8Array');
      const geom = pool.acquirePointsGeometry('node1', data, 1000);
      pool.updatePointsGeometry(geom, data, 1000);

      const colorAttr = geom.getAttribute('aColor') as THREE.InterleavedBufferAttribute;
      // Underlying storage is the shared Float32 interleaved buffer.
      expect(colorAttr.data.array).toBeInstanceOf(Float32Array);
      // First instance's R/G/B should be in [0, 1].
      expect(colorAttr.getX(0)).toBeGreaterThanOrEqual(0);
      expect(colorAttr.getX(0)).toBeLessThanOrEqual(1);
    });

    it('widens Uint16Array color source to Float32 [0,1] in interleaved storage', () => {
      const data = createMockLoadedPointsData(1000, 'Uint16Array');
      const geom = pool.acquirePointsGeometry('node1', data, 1000);
      pool.updatePointsGeometry(geom, data, 1000);

      const colorAttr = geom.getAttribute('aColor') as THREE.InterleavedBufferAttribute;
      expect(colorAttr.data.array).toBeInstanceOf(Float32Array);
      // Verify in-range — divisor is 65535 for Uint16 normalized.
      expect(colorAttr.getX(0)).toBeGreaterThanOrEqual(0);
      expect(colorAttr.getX(0)).toBeLessThanOrEqual(1);
    });

    it('should NOT reuse geometry when types differ', () => {
      const dataFloat = createMockLoadedPointsData(1000, 'Float32Array');
      const dataUint8 = createMockLoadedPointsData(900, 'Uint8Array');

      pool.acquirePointsGeometry('node1', dataFloat, 1000);
      pool.releasePointsGeometry('node1');

      pool.acquirePointsGeometry('node2', dataUint8, 900); // Different type!

      const stats = pool.getStats();
      // Should create NEW geometry (different types, can't reuse)
      expect(stats.allocations).toBe(2);
      expect(stats.reuses).toBe(0);
    });

    it('should reuse geometry when types match', () => {
      const data1 = createMockLoadedPointsData(1000, 'Uint8Array');
      const data2 = createMockLoadedPointsData(900, 'Uint8Array'); // Same type

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
