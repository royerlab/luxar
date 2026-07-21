/**
 * Unit tests for GPU Buffer Pool
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  GPUBufferPool,
  __setMinInstanceCapacityForTesting,
} from '../../../rendering/gpu-buffer-pool';
import type { LoadedPointsData } from '../../../data/data-loader-types';
import * as THREE from 'three';
import { getSplatTexture } from '../../../rendering/gsplat-geometry';
import { getPointTexture } from '../../../rendering/point-geometry';
import { POINT_FLOATS_PER_POINT } from '../../../rendering/element-texture-layout';

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

  // Read one float of point i's texel block from the geometry's point
  // texture (layout documented in point-geometry.ts: texel 0 =
  // center.xyz + radius, texel 1 = color.rgb + sharpness, texel 2 =
  // scalar + alpha).
  const texel = (geom: THREE.BufferGeometry, i: number, offset: number): number => {
    const arr = getPointTexture(geom)!.image.data as Float32Array;
    return arr[i * POINT_FLOATS_PER_POINT + offset];
  };

  beforeEach(() => {
    pool = new GPUBufferPool(20, 300); // maxPoolSize=20, evictionFrames=300
  });

  describe('Points Geometry', () => {
    it('should allocate new geometry on first request', () => {
      const geom = pool.acquirePointsGeometry('node1', 1000);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);
      // Acquire must NOT bump instanceCount — only a successful texel
      // write does (updateGeometry's post-write prepare). A throwing
      // write would otherwise draw the new count over stale/zero texels.
      expect(geom.instanceCount).toBe(0);
      expect(geom.drawRange.count).toBe(6);
      // Texture-backed storage: per-point data lives in the pooled
      // RGBA32F point texture (3 texels/point); aSortedIndex is the only
      // per-instance attribute. Capacity = ceil(1000 * 1.5) = 1500
      // points -> >= 1500 * 12 floats in the backing store.
      const texture = getPointTexture(geom);
      expect(texture).not.toBeNull();
      expect((texture!.image.data as Float32Array).length).toBeGreaterThanOrEqual(
        1500 * POINT_FLOATS_PER_POINT
      );
      const sortedIndex = geom.getAttribute('aSortedIndex');
      expect(sortedIndex).toBeDefined();
      expect(sortedIndex.array).toBeInstanceOf(Uint32Array);

      const stats = pool.getStats();
      expect(stats.allocations).toBe(1);
      expect(stats.reuses).toBe(0);
    });

    it('should reuse geometry when requesting same node again', () => {
      const geom1 = pool.acquirePointsGeometry('node1', 1000);
      const geom2 = pool.acquirePointsGeometry('node1', 900); // Same node, smaller count

      expect(geom2).toBe(geom1); // Same geometry instance
      const stats = pool.getStats();
      expect(stats.allocations).toBe(1);
      expect(stats.reuses).toBe(1); // Second call reused
    });

    it('should grow-swap to a FRESH geometry when count exceeds capacity', () => {
      const geom1 = pool.acquirePointsGeometry('node1', 1000); // Capacity ~1500
      const geom2 = pool.acquirePointsGeometry('node1', 2000); // Needs >1500

      // Growth is release + reacquire, never an in-place rebuild (that
      // strands the old GPU buffer in the renderer caches — a permanent
      // leak under the WebGPU renderer). The undersized geometry is
      // pooled intact for future best-fit reuse.
      expect(geom2).not.toBe(geom1);
      const stats = pool.getStats();
      expect(stats.capacityGrowths).toBe(1);
      expect(stats.pooledBuffers).toBeGreaterThan(0);
    });

    it('grow path bumps byType.points.allocations (MED-11 regression)', () => {
      // Regression for MED-11: the grow branch allocates fresh storage
      // (real GPU buffer creation) but previously only bumped
      // `stats.capacityGrowths`, leaving `typeStats.points.allocations`
      // stale. The data-loading-monitor uses byType.points.allocations to
      // compute reuse rates; without this fix, a workload of repeated
      // grows reports an inflated reuse rate.
      pool.acquirePointsGeometry('node1', 1000); // fresh allocation
      const allocsAfterFirst = pool.getStats().byType.points.allocations;
      expect(allocsAfterFirst).toBe(1);

      pool.acquirePointsGeometry('node1', 2000); // grow #1
      expect(pool.getStats().byType.points.allocations).toBe(allocsAfterFirst + 1);

      pool.acquirePointsGeometry('node1', 4000); // grow #2
      expect(pool.getStats().byType.points.allocations).toBe(allocsAfterFirst + 2);

      // capacityGrowths should rise in lockstep on the grow branch.
      expect(pool.getStats().capacityGrowths).toBe(2);
    });

    it('didLastAcquireRebuildAttributes is false on in-place reuse, true on grow', () => {
      pool.acquirePointsGeometry('node1', 1000);
      expect(pool.didLastAcquireRebuildAttributes()).toBe(true); // First allocation

      pool.acquirePointsGeometry('node1', 900); // reuse, no grow
      expect(pool.didLastAcquireRebuildAttributes()).toBe(false);

      pool.acquirePointsGeometry('node1', 2000); // grow
      expect(pool.didLastAcquireRebuildAttributes()).toBe(true);
    });

    it('didLastAcquireRebuildAttributes is true when pooled candidate is reclaimed for a new node', () => {
      pool.acquirePointsGeometry('node1', 1000);
      pool.releasePointsGeometry('node1'); // back into the pool

      // Fixed texel layout: any pooled points geometry fits any node.
      pool.acquirePointsGeometry('node2', 1000);
      expect(pool.didLastAcquireRebuildAttributes()).toBe(true);
    });

    it('should release geometry back to pool', () => {
      pool.acquirePointsGeometry('node1', 1000);
      pool.releasePointsGeometry('node1');

      const stats = pool.getStats();
      expect(stats.activeBuffers).toBe(0);
      expect(stats.pooledBuffers).toBe(1);
    });

    it('should reuse released geometry for new node', () => {
      pool.acquirePointsGeometry('node1', 1000);
      pool.releasePointsGeometry('node1');

      pool.acquirePointsGeometry('node2', 800); // Different node, similar size
      const stats = pool.getStats();
      expect(stats.allocations).toBe(1); // Only one allocation
      expect(stats.reuses).toBe(1); // Reused for node2
    });

    it('positions-only Points get default radius=0.5, sharpness=0.5, white color, scalar=0, alpha=1', () => {
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
      const geom = pool.acquirePointsGeometry('p1', 3);
      pool.updatePointsGeometry(geom, positionsOnly, 3);

      // Active range filled with defaults — not zeros. Texel layout:
      // [0..2] center, [3] radius, [4..6] color, [7] sharpness,
      // [8] scalar, [9] alpha.
      for (let i = 0; i < 3; i++) {
        expect(texel(geom, i, 0)).toBeCloseTo(i, 5); // center.x
        expect(texel(geom, i, 3)).toBeCloseTo(0.5, 5); // radius default
        expect(texel(geom, i, 4)).toBeCloseTo(1.0, 5); // white R
        expect(texel(geom, i, 5)).toBeCloseTo(1.0, 5); // white G
        expect(texel(geom, i, 6)).toBeCloseTo(1.0, 5); // white B
        expect(texel(geom, i, 7)).toBeCloseTo(0.5, 5); // [0,1] knob default -> beta=2
        expect(texel(geom, i, 8)).toBe(0.0); // no-scalar identity
        expect(texel(geom, i, 9)).toBe(1.0); // opaque alpha identity
      }
    });

    it('pool reuse fills defaults when subsequent commit lacks colors/radii', () => {
      // First commit: full data including colors/radii/sharpness
      const full = createMockLoadedPointsData(4);
      const geom = pool.acquirePointsGeometry('p2', 4);
      pool.updatePointsGeometry(geom, full, 4);

      // Second commit: same node, but data has no colors/radii/sharpness
      // (e.g. nD slice change reveals points without those optional attrs).
      // The fixed texel layout reuses the same geometry unconditionally.
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

      // Defaults overwrite values left in the reused texture from `full`.
      for (let i = 0; i < 4; i++) {
        expect(texel(geom, i, 3)).toBeCloseTo(0.5, 5); // radius default
        expect(texel(geom, i, 4)).toBeCloseTo(1.0, 5);
        expect(texel(geom, i, 5)).toBeCloseTo(1.0, 5);
        expect(texel(geom, i, 6)).toBeCloseTo(1.0, 5);
      }
    });
  });

  describe('Lines Geometry', () => {
    it('should create InstancedBufferGeometry for lines', () => {
      const geom = pool.acquireLinesGeometry('line1', 500, false);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);

      // Check instance attributes
      expect(geom.getAttribute('aStartPos')).toBeDefined();
      expect(geom.getAttribute('aEndPos')).toBeDefined();
      expect(geom.getAttribute('aStartColor')).toBeDefined();
      expect(geom.getAttribute('aSegmentLength')).toBeDefined();
    });

    it('should reuse lines geometry', () => {
      const geom1 = pool.acquireLinesGeometry('line1', 500, false);
      const geom2 = pool.acquireLinesGeometry('line1', 400, false);

      expect(geom2).toBe(geom1);
      expect(pool.getStats().reuses).toBe(1);
    });

    it('should handle lines release and reuse', () => {
      pool.acquireLinesGeometry('line1', 500, false);
      pool.releaseLinesGeometry('line1');

      pool.acquireLinesGeometry('line2', 450, false);
      expect(pool.getStats().allocations).toBe(1);
      expect(pool.getStats().reuses).toBe(1);
    });
  });

  describe('GSplats Geometry', () => {
    it('should create InstancedBufferGeometry for gsplats', () => {
      const geom = pool.acquireGSplatsGeometry('splat1', 300);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);

      // Texture-backed storage: splat data lives in the pooled RGBA32F
      // texture; aSortedIndex is the only per-instance attribute.
      const texture = getSplatTexture(geom);
      expect(texture).not.toBeNull();
      // Capacity = ceil(300 * 1.5) = 450 splats -> >= 450*16 floats.
      expect((texture!.image.data as Float32Array).length).toBeGreaterThanOrEqual(450 * 16);
      const sortedIndex = geom.getAttribute('aSortedIndex');
      expect(sortedIndex).toBeDefined();
      expect(sortedIndex.array).toBeInstanceOf(Uint32Array);
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
      testPool.acquirePointsGeometry('node1', 1000);
      testPool.releasePointsGeometry('node1');

      testPool.acquirePointsGeometry('node2', 10000);
      testPool.releasePointsGeometry('node2');

      // Advance frameCount by 3 frames without touching the pooled geometries
      for (let i = 0; i < 3; i++) {
        testPool.beginFrame();
        testPool.acquirePointsGeometry(`active${i}`, 100000); // Different size bucket
        // Don't release - keep active
      }

      // The stale pooled geometries must be gone. Note: acquire paths now
      // sweep idle buffers themselves (byte-budget-on-growth fix), so the
      // eviction may already have happened during the loop above — assert
      // the cumulative OUTCOME via stats plus a final explicit sweep,
      // not the return value of one manual call.
      testPool.evictUnused();
      expect(testPool.getStats().evictions).toBeGreaterThanOrEqual(2);
    });

    it('should dispose all geometries on pool disposal', () => {
      pool.acquirePointsGeometry('node1', 1000);
      pool.acquireLinesGeometry('line1', 500, false);
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
      // acquire or as an already-disposed point texture.
      const tinyPool = new GPUBufferPool(2, 0); // maxPoolSize=2, evictionFrames=0

      tinyPool.acquirePointsGeometry('nodeA', 1000);
      tinyPool.releasePointsGeometry('nodeA');
      tinyPool.acquirePointsGeometry('nodeB', 2000);
      tinyPool.releasePointsGeometry('nodeB');

      tinyPool.beginFrame();
      tinyPool.beginFrame();

      const geomC = tinyPool.acquirePointsGeometry('nodeC', 100000);

      expect(geomC).toBeDefined();
      expect(getPointTexture(geomC)).not.toBeNull();
      expect(() => tinyPool.releasePointsGeometry('nodeC')).not.toThrow();
    });
  });

  describe('Size Bucketing', () => {
    it('should use appropriate size buckets', () => {
      // Acquire geometries of different sizes
      pool.acquirePointsGeometry('small', 500); // → 1K bucket
      pool.acquirePointsGeometry('medium', 3000); // → 5K bucket
      pool.acquirePointsGeometry('large', 20000); // → 50K bucket

      // Release them
      pool.releasePointsGeometry('small');
      pool.releasePointsGeometry('medium');
      pool.releasePointsGeometry('large');

      // Acquire similar sizes - should reuse from correct buckets
      pool.acquirePointsGeometry('small2', 600); // Should reuse from 1K bucket
      pool.acquirePointsGeometry('medium2', 4000); // Should reuse from 5K bucket

      const stats = pool.getStats();
      expect(stats.reuses).toBeGreaterThanOrEqual(2);
    });
  });

  describe('Statistics Tracking', () => {
    it('should track allocations, reuses, and evictions', () => {
      // Allocate
      pool.acquirePointsGeometry('node1', 1000);
      expect(pool.getStats().allocations).toBe(1);

      // Reuse
      pool.acquirePointsGeometry('node1', 900);
      expect(pool.getStats().reuses).toBe(1);

      // Growth
      pool.acquirePointsGeometry('node1', 3000);
      expect(pool.getStats().capacityGrowths).toBe(1);

      // Eviction (tested separately due to frame requirements)
    });

    it('should track active vs pooled buffers', () => {
      pool.acquirePointsGeometry('node1', 1000);
      pool.acquireLinesGeometry('line1', 500, false);

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
    it('widens Uint8Array color source to Float32 [0,1] in the point texture', () => {
      // Per-point data lives in the RGBA32F point texture, so ALL source
      // dtypes are widened to Float32 at upload. Uint8 source data
      // (range [0, 255]) is widened with the `normalized: true` divisor
      // (255) so the shader sees the same [0, 1] range as before.
      const data = createMockLoadedPointsData(1000, 'Uint8Array');
      const geom = pool.acquirePointsGeometry('node1', 1000);
      pool.updatePointsGeometry(geom, data, 1000);

      // 128 / 255 lands in texel 1's color slots.
      expect(texel(geom, 0, 4)).toBeCloseTo(128 / 255, 5);
      expect(texel(geom, 0, 5)).toBeCloseTo(128 / 255, 5);
      expect(texel(geom, 0, 6)).toBeCloseTo(128 / 255, 5);
    });

    it('widens Uint16Array color source to Float32 [0,1] in the point texture', () => {
      const data = createMockLoadedPointsData(1000, 'Uint16Array');
      const geom = pool.acquirePointsGeometry('node1', 1000);
      pool.updatePointsGeometry(geom, data, 1000);

      // Divisor is 65535 for Uint16 normalized: 32768 / 65535 ≈ 0.5.
      expect(texel(geom, 0, 4)).toBeCloseTo(32768 / 65535, 5);
    });

    it('REUSES pooled geometry across source dtypes (fixed texel layout)', () => {
      // The interleaved era bucketed pooled geometries by attribute
      // dtype snapshot and re-allocated on a mismatch. The fixed 3-texel
      // layout removes that: dtype normalization happens at upload, so
      // ANY pooled points geometry fits ANY points node.
      const dataUint8 = createMockLoadedPointsData(900, 'Uint8Array');

      pool.acquirePointsGeometry('node1', 1000); // Float32-era tenant
      pool.releasePointsGeometry('node1');

      const geom = pool.acquirePointsGeometry('node2', 900); // reuse, dtype-blind
      const stats = pool.getStats();
      expect(stats.allocations).toBe(1);
      expect(stats.reuses).toBe(1);

      // And the Uint8 upload into the reused texture still normalizes.
      pool.updatePointsGeometry(geom, dataUint8, 900);
      expect(texel(geom, 0, 4)).toBeCloseTo(128 / 255, 5);
    });
  });

  describe('Zero-count safety', () => {
    it('never produces zero-capacity storage when count is 0', () => {
      // Regression guard for the WebGPU "blank scene" issue. When a scene
      // starts at a slice where no instances are visible, the pool used to
      // allocate a buffer sized at `ceil(0 * 1.5) = 0` floats — a
      // zero-length JS array never produces a real GPU buffer on Three's
      // WebGPU backend, and any subsequent grow keeps the empty (or
      // absent) GPU buffer bound. Drawing then fails with "Instance range
      // … requires a larger buffer than the bound buffer size (0)".
      //
      // The fix is `chooseCapacity` + `DEFAULT_MIN_INSTANCE_CAPACITY` in
      // gpu-buffer-pool.ts. The test setup file lowers the floor to zero
      // so other tests can exercise the grow path; restore the production
      // default for this specific assertion.
      __setMinInstanceCapacityForTesting(null);
      try {
        const empty = new GPUBufferPool(20, 300, 5, () => 0);

        const lineGeom = empty.acquireLinesGeometry('zero-lines', 0, false);
        const lineBuf = (lineGeom.getAttribute('aStartPos') as THREE.InterleavedBufferAttribute)
          .data;
        expect((lineBuf.array as Float32Array).length).toBeGreaterThan(0);

        // Texture-backed points: the zero-capacity hazard is the ordering
        // attribute (the texture always has >= 1 row) — mirror gsplats.
        const pointGeom = empty.acquirePointsGeometry('zero-points', 0);
        const pointIdx = pointGeom.getAttribute('aSortedIndex');
        expect((pointIdx.array as Uint32Array).length).toBeGreaterThan(0);
        expect((getPointTexture(pointGeom)!.image.data as Float32Array).length).toBeGreaterThan(0);

        const gsplatGeom = empty.acquireGSplatsGeometry('zero-gsplats', 0);
        // Texture-backed storage: the zero-capacity hazard for gsplats is
        // the ordering attribute (the texture always has >= 1 row).
        const gsplatIdx = gsplatGeom.getAttribute('aSortedIndex');
        expect((gsplatIdx.array as Uint32Array).length).toBeGreaterThan(0);
        expect((getSplatTexture(gsplatGeom)!.image.data as Float32Array).length).toBeGreaterThan(0);
      } finally {
        __setMinInstanceCapacityForTesting(0);
      }
    });
  });
});
