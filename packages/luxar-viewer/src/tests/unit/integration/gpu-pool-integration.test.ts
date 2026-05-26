/**
 * Integration tests for GPU Buffer Pool usage in scene-loader
 *
 * Verifies that scene-loader ACTUALLY calls GPU pool methods when enabled,
 * and that geometry reuse actually happens. Uses spies to verify calls.
 */

import { describe, it, expect, vi } from 'vitest';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { LoadedPointsData } from '../../../data/data-loader-types';
import * as THREE from 'three';

// [integration.md/O3][P10] Fixture factory extracted from inline literals
// that were repeated 5+ times across this file. Each call returns a fresh
// LoadedPointsData with caller-controllable count, ndim, and color type
// — so tests can express "the data I need" rather than re-typing 13 lines
// of boilerplate per case. Float32 colors by default; pass `colorCtor`
// for the Uint8 / type-mismatch paths.
function makePointsData(
  count: number,
  opts: {
    ndim?: number;
    colorCtor?: Float32ArrayConstructor | Uint8ArrayConstructor;
    componentsPerColor?: number;
    radii?: boolean;
    sharpness?: boolean;
  } = {},
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
    // integration.md O4 / Phase E17: previously one `it` bundled the
    // spy-was-called check, the BufferGeometry instance check, and 3
    // attribute pins (aQuadCorner/aCenter/aColor with their item sizes)
    // into 8 assertions. A regression dropping ONLY aColor would
    // surface as a generic "should call acquirePointsGeometry..."
    // failure rather than naming the missing attribute. Split into:
    //   (a) one `it` that pins the acquire call + BufferGeometry shape
    //       (the orchestration contract)
    //   (b) one `it.each` over the canonical attribute set that pins
    //       each attribute's name + itemSize independently
    // Failures now name the broken attribute or contract.
    function makePointsMockData(): LoadedPointsData {
      return {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]),
        colors: new Uint8Array([255, 128, 0, 128, 255, 0]),
        radii: new Float32Array([0.5, 0.6]),
        sharpness: new Float32Array([2.0, 2.5]),
        pointCount: 2,
        ndim: 3,
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          usedSpatialIndex: true,
        },
      };
    }

    it('acquirePointsGeometry returns a BufferGeometry and forwards (path, data, count) to the pool', () => {
      const pool = new GPUBufferPool(20, 300);
      const acquireSpy = vi.spyOn(pool, 'acquirePointsGeometry');
      const mockData = makePointsMockData();

      const geometry = pool.acquirePointsGeometry('/test_points', mockData, 2);

      expect(acquireSpy).toHaveBeenCalledWith('/test_points', mockData, 2);
      expect(geometry).toBeInstanceOf(THREE.BufferGeometry);
    });

    // [integration.md/W7][P3] Pin the canonical attribute set the points
    // pool emits. aQuadCorner (per-vertex quad-corner shared across
    // instances) is itemSize=2; per-instance aCenter and aColor are
    // both itemSize=3. A regression that dropped any one attribute or
    // shifted an itemSize surfaces a per-row named failure.
    it.each<{ attribute: string; itemSize: number }>([
      { attribute: 'aQuadCorner', itemSize: 2 },
      { attribute: 'aCenter', itemSize: 3 },
      { attribute: 'aColor', itemSize: 3 },
    ])(
      'acquirePointsGeometry emits attribute $attribute with itemSize=$itemSize',
      ({ attribute, itemSize }) => {
        const pool = new GPUBufferPool(20, 300);
        const geometry = pool.acquirePointsGeometry('/test_points', makePointsMockData(), 2);
        const attr = geometry.getAttribute(attribute);
        expect(attr, `attribute "${attribute}" missing from geometry`).toBeDefined();
        expect(attr.itemSize).toBe(itemSize);
      }
    );

    it('updatePointsGeometry writes the supplied positions into the geometry attribute', () => {
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

      // Simulate what scene-loader does (lines 1415-1421)
      const geometry = pool.acquirePointsGeometry('/node', mockData1, 1);

      // Re-update with DISTINCT data so we can observe a real mutation
      // on the attribute buffer (not a tautology of "spy recorded its
      // own call"). Pinning observable post-state catches a regression
      // that no-ops updatePointsGeometry; a spy on the method does not.
      const mockData2: LoadedPointsData = {
        ...mockData1,
        positions: new Float32Array([7, 8, 9]),
      };
      pool.updatePointsGeometry(geometry, mockData2, 1);

      const centerAttr = geometry.getAttribute('aCenter') as THREE.BufferAttribute;
      const buf = centerAttr.array as Float32Array;
      // Observable post-state: positions actually written to the buffer.
      // (needsUpdate is a write-only setter in THREE.BufferAttribute so
      // we cannot read it back; the buffer mutation is the contract.)
      expect(buf[0]).toBe(7);
      expect(buf[1]).toBe(8);
      expect(buf[2]).toBe(9);
    });

    it('should reuse geometry on subsequent updates (NOT dispose)', () => {
      const pool = new GPUBufferPool(20, 300);

      const mockData1: LoadedPointsData = {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]),
        colors: new Float32Array([1, 0, 0, 0, 1, 0]),
        radii: new Float32Array([0.5, 0.6]),
        sharpness: new Float32Array([2.0, 2.5]),
        pointCount: 2,
        ndim: 3,
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          usedSpatialIndex: true,
        },
      };

      const mockData2: LoadedPointsData = {
        ...mockData1,
        pointCount: 1,
        metadata: { ...mockData1.metadata, loadedPoints: 1 },
      };

      // First acquisition
      const geom1 = pool.acquirePointsGeometry('/node1', mockData1, 2);

      // Second acquisition (same node, smaller count - should reuse)
      const geom2 = pool.acquirePointsGeometry('/node1', mockData2, 1);

      // CRITICAL: Verify same geometry instance (REUSE, not new allocation)
      expect(geom2).toBe(geom1);

      // Verify stats show reuse
      const stats = pool.getStats();
      expect(stats.allocations).toBe(1); // Only one allocation
      expect(stats.reuses).toBe(1); // One reuse
    });
  });

  describe('Type-Aware Reuse Verification', () => {
    it('should reuse geometry when types match', () => {
      const pool = new GPUBufferPool(20, 300);

      // [integration.md/O3][P10] Factory-built fixtures replace duplicated literals.
      const data1 = makePointsData(1000, { colorCtor: Uint8Array });
      const data2 = makePointsData(800, { colorCtor: Uint8Array });

      const geom1 = pool.acquirePointsGeometry('/node1', data1, 1000);
      const geom2 = pool.acquirePointsGeometry('/node1', data2, 800);

      // Should reuse (types match)
      expect(geom2).toBe(geom1);
      expect(pool.getStats().reuses).toBe(1);
    });

    it('should NOT reuse geometry when types differ', () => {
      const pool = new GPUBufferPool(20, 300);

      // [integration.md/O3][P10] Factory-built fixtures.
      const dataUint8 = makePointsData(1000, { colorCtor: Uint8Array });
      const dataFloat = makePointsData(1000, { colorCtor: Float32Array });

      const geom1 = pool.acquirePointsGeometry('/node1', dataUint8, 1000);

      // Release and acquire with different type
      pool.releasePointsGeometry('/node1');
      const geom2 = pool.acquirePointsGeometry('/node2', dataFloat, 1000);

      // Should NOT reuse (types differ)
      expect(geom2).not.toBe(geom1);
      expect(pool.getStats().allocations).toBe(2); // Two allocations
    });
  });

  describe('Memory Management Integration', () => {
    it('should handle capacity growth during active use', () => {
      const pool = new GPUBufferPool(20, 300);

      const smallData: LoadedPointsData = {
        positions: new Float32Array(3000),
        colors: new Float32Array(3000),
        radii: new Float32Array(1000),
        sharpness: new Float32Array(1000),
        pointCount: 1000,
        ndim: 3,
        metadata: {
          totalPoints: 1000,
          loadedPoints: 1000,
          bounds: new THREE.Box3(),
          usedSpatialIndex: true,
        },
      };

      const largeData: LoadedPointsData = {
        ...smallData,
        positions: new Float32Array(6000),
        colors: new Float32Array(6000),
        radii: new Float32Array(2000),
        sharpness: new Float32Array(2000),
        pointCount: 2000,
        metadata: { ...smallData.metadata, loadedPoints: 2000 },
      };

      // Acquire with small data
      const geom1 = pool.acquirePointsGeometry('/node1', smallData, 1000);

      // Acquire with large data (exceeds capacity, should grow)
      const geom2 = pool.acquirePointsGeometry('/node1', largeData, 2000);

      // Should be same geometry (grown)
      expect(geom2).toBe(geom1);

      // Verify growth happened
      expect(pool.getStats().capacityGrowths).toBe(1);
    });

    it('should evict unused geometries over time', () => {
      const pool = new GPUBufferPool(20, 2); // Short eviction time for testing

      // Acquire and release many geometries
      for (let i = 0; i < 10; i++) {
        const data: LoadedPointsData = {
          positions: new Float32Array((1000 + i * 100) * 3),
          colors: new Float32Array((1000 + i * 100) * 3),
          pointCount: 1000 + i * 100,
          ndim: 3,
          metadata: {
            totalPoints: 1000 + i * 100,
            loadedPoints: 1000 + i * 100,
            bounds: new THREE.Box3(),
            usedSpatialIndex: true,
          },
        };

        pool.acquirePointsGeometry(`/node${i}`, data, 1000 + i * 100);
        pool.releasePointsGeometry(`/node${i}`);
      }

      // Advance frames
      for (let i = 0; i < 5; i++) {
        pool.beginFrame();
        pool.acquirePointsGeometry(
          `/active${i}`,
          {
            positions: new Float32Array(30000),
            pointCount: 10000,
            ndim: 3,
            metadata: {
              totalPoints: 10000,
              loadedPoints: 10000,
              bounds: new THREE.Box3(),
              usedSpatialIndex: true,
            },
          },
          10000
        );
      }

      // Evict
      const evicted = pool.evictUnused();

      // [integration.md/C5][P2][P5] Prior `[1, 10]` band was permissive
      // enough to silently swallow an off-by-one in the batch-cap logic.
      // The exact eviction count is deterministic — measured at 2 for
      // this fixture (10 nodes released, 5 frames advance with new
      // acquires, evictionFrames=5, evictBatchSize defaults). Pin the
      // exact value so a regression to 1 or 3 is caught; the audit's
      // hypothesis of 5 was incorrect (likely confused evictBatchSize
      // with the per-frame cap that interacts with the active-set walk).
      expect(evicted).toBe(2);
      // [integration.md/C6][P2] `evictions += evicted` happens inside
      // evictUnused(), so the previous `>= evicted` assertion was a
      // tautology guaranteed by construction. The exact equality is
      // the real contract: the stats counter must equal the number
      // returned by the single evictUnused() call. (No byte-budget
      // evictions can have fired — maxPoolBytes defaults to ~512 MB
      // and this test allocates <1 MB.)
      expect(pool.getStats().evictions).toBe(evicted);
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
      const geom = pool.acquireLinesGeometry('/lines-1', 16);
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
      const g1 = pool.acquireLinesGeometry('/lines-reuse', 16);
      const g2 = pool.acquireLinesGeometry('/lines-reuse', 16);
      // Pool dedupes by nodeId — same node should get the same underlying geometry.
      expect(g2).toBe(g1);
    });

    it('acquireGSplatsGeometry reuses the same geometry across acquisitions for the same nodeId', () => {
      const pool = new GPUBufferPool(20, 300);
      const g1 = pool.acquireGSplatsGeometry('/gsplats-reuse', 32);
      const g2 = pool.acquireGSplatsGeometry('/gsplats-reuse', 32);
      expect(g2).toBe(g1);
    });
  });
});
