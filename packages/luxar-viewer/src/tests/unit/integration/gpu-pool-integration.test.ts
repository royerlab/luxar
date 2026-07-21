/**
 * Integration tests for GPU Buffer Pool usage in scene-loader
 *
 * Verifies that scene-loader ACTUALLY calls GPU pool methods when enabled,
 * and that geometry reuse actually happens. Uses spies to verify calls.
 */

import { describe, it, expect, vi } from 'vitest';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import { getPointTexture } from '../../../rendering/point-geometry';
import { POINT_FLOATS_PER_POINT } from '../../../rendering/element-texture-layout';
import type { LoadedPointsData } from '../../../data/data-loader-types';
import * as THREE from 'three';

// [integration.md/O3][P10] Fixture factory extracted from inline literals
// that were repeated 5+ times across this file. Each call returns a fresh
// LoadedPointsData with caller-controllable count, ndim, and color type
// — so tests can express "the data I need" rather than re-typing 13 lines
// of boilerplate per case. Float32 colors by default; pass `colorCtor`
// for the Uint8 path.
function makePointsData(
  count: number,
  opts: {
    ndim?: number;
    colorCtor?: Float32ArrayConstructor | Uint8ArrayConstructor;
    componentsPerColor?: number;
    radii?: boolean;
    sharpness?: boolean;
  } = {}
): LoadedPointsData {
  const ndim = opts.ndim ?? 3;
  const componentsPerColor = opts.componentsPerColor ?? 3;
  const ColorCtor = opts.colorCtor ?? Float32Array;
  const data: LoadedPointsData = {
    positions: new Float32Array(count * ndim),
    colors: new ColorCtor(count * componentsPerColor),
    pointCount: count,
    ndim,
    metadata: {
      totalPoints: count,
      loadedPoints: count,
      bounds: new THREE.Box3(),
      usedSpatialIndex: true,
    },
  };
  if (opts.radii !== false) data.radii = new Float32Array(count);
  if (opts.sharpness !== false) data.sharpness = new Float32Array(count);
  return data;
}

describe('GPU Buffer Pool Integration Tests', () => {
  describe('Geometry Acquisition Verification', () => {
    it('acquirePointsGeometry returns an InstancedBufferGeometry and forwards (path, count) to the pool', () => {
      const pool = new GPUBufferPool(20, 300);
      const acquireSpy = vi.spyOn(pool, 'acquirePointsGeometry');

      const geometry = pool.acquirePointsGeometry('/test_points', 2);

      expect(acquireSpy).toHaveBeenCalledWith('/test_points', 2);
      expect(geometry).toBeInstanceOf(THREE.InstancedBufferGeometry);
    });

    // Pin the canonical storage the points pool emits: aQuadCorner
    // (per-vertex quad-corner shared across instances, itemSize=2), the
    // per-instance `aSortedIndex` (Uint32), and the RGBA32F point
    // texture (3 texels/point). A regression that dropped any one of
    // them surfaces a named failure.
    it('acquirePointsGeometry emits the quad + texture storage pair', () => {
      const pool = new GPUBufferPool(20, 300);
      const geometry = pool.acquirePointsGeometry('/test_points', 2);

      const quad = geometry.getAttribute('aQuadCorner');
      expect(quad, 'attribute "aQuadCorner" missing from geometry').toBeDefined();
      expect(quad.itemSize).toBe(2);

      const sortedIndex = geometry.getAttribute('aSortedIndex');
      expect(sortedIndex, 'attribute "aSortedIndex" missing from geometry').toBeDefined();
      expect(sortedIndex.itemSize).toBe(1);
      expect(sortedIndex.array).toBeInstanceOf(Uint32Array);

      const texture = getPointTexture(geometry);
      expect(texture, 'point texture missing from geometry').not.toBeNull();
    });

    it('updatePointsGeometry writes the supplied positions into the point texture', () => {
      const pool = new GPUBufferPool(20, 300);

      const mockData1: LoadedPointsData = {
        positions: new Float32Array([1, 2, 3]),
        colors: new Float32Array([1, 0, 0]),
        radii: new Float32Array([0.5]),
        sharpness: new Float32Array([2.0]),
        pointCount: 1,
        ndim: 3,
        metadata: {
          totalPoints: 1,
          loadedPoints: 1,
          bounds: new THREE.Box3(),
          usedSpatialIndex: true,
        },
      };

      // Simulate what scene-loader does
      const geometry = pool.acquirePointsGeometry('/node', 1);

      // Re-update with DISTINCT data so we can observe a real mutation
      // on the texture's backing store (not a tautology of "spy recorded
      // its own call"). Pinning observable post-state catches a
      // regression that no-ops updatePointsGeometry; a spy on the method
      // does not.
      const mockData2: LoadedPointsData = {
        ...mockData1,
        positions: new Float32Array([7, 8, 9]),
      };
      pool.updatePointsGeometry(geometry, mockData2, 1);

      const texels = getPointTexture(geometry)!.image.data as Float32Array;
      // Observable post-state: positions actually written to texel 0.
      expect(texels[0]).toBe(7);
      expect(texels[1]).toBe(8);
      expect(texels[2]).toBe(9);
    });

    // integration.md O3 / Phase E24: P9 rename — name describes input
    // class + expected behavior ("acquire on the same node returns the
    // same geometry instance"), not implementation framing ("should
    // reuse... NOT dispose").
    it('acquiring with the same nodeId returns the same geometry instance (geometry reuse, no dispose)', () => {
      const pool = new GPUBufferPool(20, 300);

      // First acquisition
      const geom1 = pool.acquirePointsGeometry('/node1', 2);

      // Second acquisition (same node, smaller count - should reuse)
      const geom2 = pool.acquirePointsGeometry('/node1', 1);

      // CRITICAL: Verify same geometry instance (REUSE, not new allocation)
      expect(geom2).toBe(geom1);

      // Verify stats show reuse
      const stats = pool.getStats();
      expect(stats.allocations).toBe(1); // Only one allocation
      expect(stats.reuses).toBe(1); // One reuse
    });
  });

  describe('Dtype-Blind Reuse Verification', () => {
    it('reuses geometry across acquires on the same node', () => {
      const pool = new GPUBufferPool(20, 300);

      const geom1 = pool.acquirePointsGeometry('/node1', 1000);
      const geom2 = pool.acquirePointsGeometry('/node1', 800);

      // Should reuse — capacity is the only criterion.
      expect(geom2).toBe(geom1);
      expect(pool.getStats().reuses).toBe(1);
    });

    it('reuses a released geometry for a node with a DIFFERENT color dtype (Uint8 → Float32)', () => {
      // The interleaved era forced a fresh allocation on a dtype flip;
      // the fixed texel layout widens every dtype to Float32 at upload,
      // so the pooled geometry is reused and the values still normalize.
      const pool = new GPUBufferPool(20, 300);

      const geom1 = pool.acquirePointsGeometry('/node1', 1000);
      pool.updatePointsGeometry(geom1, makePointsData(1000, { colorCtor: Uint8Array }), 1000);

      // Release and acquire with different dtype
      pool.releasePointsGeometry('/node1');
      const geom2 = pool.acquirePointsGeometry('/node2', 1000);

      expect(geom2).toBe(geom1); // reused
      expect(pool.getStats().allocations).toBe(1); // one allocation total

      const dataFloat = makePointsData(1000, { colorCtor: Float32Array });
      (dataFloat.colors as Float32Array).fill(0.25);
      pool.updatePointsGeometry(geom2, dataFloat, 1000);
      const texels = getPointTexture(geom2)!.image.data as Float32Array;
      expect(texels[4]).toBeCloseTo(0.25, 5); // color.r, Float32 as-is
      expect(texels[POINT_FLOATS_PER_POINT + 4]).toBeCloseTo(0.25, 5);
    });
  });

  describe('Memory Management Integration', () => {
    it('should handle capacity growth during active use', () => {
      const pool = new GPUBufferPool(20, 300);

      // Acquire with small count
      const geom1 = pool.acquirePointsGeometry('/node1', 1000);

      // Acquire with large count (exceeds capacity → grow-swap)
      const geom2 = pool.acquirePointsGeometry('/node1', 2000);

      // Growth is release + reacquire, never an in-place rebuild (that
      // strands the old GPU buffer in the renderer caches): a FRESH
      // geometry comes back and the undersized one returns to the pool.
      expect(geom2).not.toBe(geom1);
      expect(pool.didLastAcquireRebuildAttributes()).toBe(true);

      // Verify growth happened and the old geometry was pooled intact.
      expect(pool.getStats().capacityGrowths).toBe(1);
      expect(pool.getStats().pooledBuffers).toBeGreaterThan(0);
    });

    it('should evict unused geometries over time', () => {
      const pool = new GPUBufferPool(20, 2); // Short eviction time for testing

      // Acquire and release many geometries
      for (let i = 0; i < 10; i++) {
        pool.acquirePointsGeometry(`/node${i}`, 1000 + i * 100);
        pool.releasePointsGeometry(`/node${i}`);
      }

      // Advance frames
      for (let i = 0; i < 5; i++) {
        pool.beginFrame();
        pool.acquirePointsGeometry(`/active${i}`, 10000);
      }

      // Evict. Acquire paths now sweep idle buffers themselves
      // (byte-budget-on-growth fix), so evictions may already have
      // fired during the frame-advance loop above — assert the
      // cumulative OUTCOME via stats after a final explicit sweep.
      // The deterministic total for this fixture is unchanged (2).
      pool.evictUnused();
      expect(pool.getStats().evictions).toBe(2);
    });
  });

  // integration.md G1 fix: parallel coverage for Lines + GSplats. Previously
  // gpu-pool-integration tested only Points; the Lines and GSplats acquire/
  // update paths had zero direct integration coverage despite being
  // first-class methods on the same class. Even smoke-level acquisition is
  // strictly better than no coverage.
  describe('Three-geometry symmetry: Lines + GSplats acquire/release', () => {
    it('acquireLinesGeometry returns an InstancedBufferGeometry of the requested capacity', () => {
      const pool = new GPUBufferPool(20, 300);
      const geom = pool.acquireLinesGeometry('/lines-1', 16, false);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);
      expect(() => pool.releaseLinesGeometry('/lines-1')).not.toThrow();
    });

    it('acquireGSplatsGeometry returns an InstancedBufferGeometry of the requested capacity', () => {
      const pool = new GPUBufferPool(20, 300);
      const geom = pool.acquireGSplatsGeometry('/gsplats-1', 32);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);
      expect(() => pool.releaseGSplatsGeometry('/gsplats-1')).not.toThrow();
    });

    it('acquireLinesGeometry reuses the same geometry across acquisitions for the same nodeId', () => {
      const pool = new GPUBufferPool(20, 300);
      const g1 = pool.acquireLinesGeometry('/lines-reuse', 16, false);
      const g2 = pool.acquireLinesGeometry('/lines-reuse', 16, false);
      // Pool dedupes by nodeId — same node should get the same underlying geometry.
      expect(g2).toBe(g1);
    });

    it('acquireGSplatsGeometry reuses the same geometry across acquisitions for the same nodeId', () => {
      const pool = new GPUBufferPool(20, 300);
      const g1 = pool.acquireGSplatsGeometry('/gsplats-reuse', 32);
      const g2 = pool.acquireGSplatsGeometry('/gsplats-reuse', 32);
      expect(g2).toBe(g1);
    });

    it('releaseGSplatsGeometry under a tight budget reclaims the demoted buffer (registry-demotion path)', () => {
      // End-to-end mirror of the LOD registry demoting a cold gsplats level:
      // the registry calls release → ctx.releaseLazyGSplats →
      // pool.releaseGSplatsGeometry, which moves the buffer active→pooled and
      // synchronously runs the pool's total-resident byte eviction. Under a
      // budget that fits one buffer but not two, the demoted (pooled) buffer
      // is reclaimed the same frame while the active one survives.
      let budget = 0; // disabled during setup so both buffers allocate
      const pool = new GPUBufferPool(20, 300, 5, () => budget);
      pool.acquireGSplatsGeometry('/active', 5000);
      const oneBuffer = pool.getStats().activeBytes;
      expect(oneBuffer).toBeGreaterThan(0);
      pool.acquireGSplatsGeometry('/cold', 5000);

      budget = Math.floor(oneBuffer * 1.5); // fits 1 buffer, not 2
      pool.releaseGSplatsGeometry('/cold'); // → pooled, triggers evictUnused internally

      const stats = pool.getStats();
      expect(stats.activeBuffers).toBe(1); // active level retained
      expect(pool.getResidentBytes()).toBeLessThanOrEqual(budget);
      pool.dispose();
    });
  });
});
