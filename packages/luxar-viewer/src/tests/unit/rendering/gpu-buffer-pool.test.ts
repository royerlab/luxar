/**
 * Unit tests for GPU Buffer Pool
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  GPUBufferPool,
  __setMinInstanceCapacityForTesting,
} from '../../../rendering/gpu-buffer-pool';
import type { LoadedPointsData } from '../../../data/data-loader-types';
import type { ProcessedLinesData } from '../../../types/lines';
import * as THREE from 'three';
import { getSplatTexture } from '../../../rendering/gsplat-geometry';
import { getPointTexture } from '../../../rendering/point-geometry';
import { getLineTexture } from '../../../rendering/line-geometry';
import {
  configureElementTextureLayout,
  LINE_FLOATS_PER_SEGMENT,
  POINT_FLOATS_PER_POINT,
  resetElementTextureLayoutForTests,
} from '../../../rendering/element-texture-layout';
import {
  writeSortedIndexOrdering,
  pumpSortedIndexOrderingApply,
  getActiveSortedIndexAttribute,
  hasPendingSortedIndexOrderingApply,
} from '../../../rendering/element-storage';

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

  afterEach(() => {
    resetElementTextureLayoutForTests();
  });

  describe('Points Geometry', () => {
    it('reports only element loss, not clamped allocation headroom', () => {
      configureElementTextureLayout(16); // point cap = 15*16/3 = 80
      __setMinInstanceCapacityForTesting(1);
      const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
      try {
        pool.acquirePointsGeometry('healthy', 60); // 1.5x headroom clamps from 90 to 80
        expect(errors).not.toHaveBeenCalled();

        pool.acquirePointsGeometry('oversized', 100); // real loss: 20 points
        expect(errors).toHaveBeenCalledTimes(1);
        expect(String(errors.mock.calls[0][0])).toContain('last 20 points');
      } finally {
        errors.mockRestore();
        __setMinInstanceCapacityForTesting(0);
      }
    });

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
      const sortedIndex = getActiveSortedIndexAttribute(geom)!;
      expect(sortedIndex).toBeDefined();
      expect(sortedIndex.array).toBeInstanceOf(Uint32Array);
      // Ownership marker consumed by the commit handoff's dispose gate:
      // pool geometries are released back to the free list, never disposed.
      expect(geom.userData.luxarPooled).toBe(true);

      const stats = pool.getStats();
      expect(stats.allocations).toBe(1);
      expect(stats.reuses).toBe(0);
    });

    it('updatePointsGeometry fills identity ordering with ONE collapsed prefix range', () => {
      // The adapter's post-write ordering pass: identity into
      // aSortedIndex[0, count) plus a single collapsed [0, ≥count) update
      // range (fragmented ranges would accumulate on the WebGPU backends,
      // which replay them verbatim).
      const data = createMockLoadedPointsData(4);
      const geom = pool.acquirePointsGeometry('node1', 4);
      pool.updatePointsGeometry(geom, data, 4);

      const attr = getActiveSortedIndexAttribute(geom)! as THREE.InstancedBufferAttribute;
      expect(Array.from((attr.array as Uint32Array).subarray(0, 4))).toEqual([0, 1, 2, 3]);
      expect(attr.updateRanges.length).toBe(1);
      expect(attr.updateRanges[0].start).toBe(0);
      expect(attr.updateRanges[0].count).toBeGreaterThanOrEqual(4);
    });

    it('preserveOrdering keeps the sort permutation while still rewriting texels', () => {
      // The points twin of the gsplats case in splat-texture-storage.test.ts:
      // a same-count recommit with preserveOrdering keeps the depth-sort
      // permutation (a no-worse prior until the re-sort lands) while the
      // texels themselves are refreshed.
      const geom = pool.acquirePointsGeometry('node1', 4);
      pool.updatePointsGeometry(geom, createMockLoadedPointsData(4), 4);
      // The SortWorker landed a depth-sort permutation between commits.
      writeSortedIndexOrdering(geom, new Uint32Array([3, 2, 1, 0]), 4);
      while (pumpSortedIndexOrderingApply(geom).more) {
        /* an ordering streams into the back buffer and swaps in on completion */
      }

      const recommit = createMockLoadedPointsData(4);
      (recommit.positions as Float32Array).fill(7);
      pool.updatePointsGeometry(geom, recommit, 4, { preserveOrdering: true });
      const ordering = getActiveSortedIndexAttribute(geom)!.array as Uint32Array;
      expect(Array.from(ordering.subarray(0, 4))).toEqual([3, 2, 1, 0]);
      const texels = getPointTexture(geom)!.image.data as Float32Array;
      expect(texels[0]).toBe(7); // point 0 center.x — texels WERE rewritten
      expect(geom.instanceCount).toBe(4);

      // Without the flag the identity reset is restored (default behavior).
      // Re-resolve the active attribute: a full identity write also re-homes
      // the geometry on slot 0, so the pre-reset reference is stale by design.
      pool.updatePointsGeometry(geom, createMockLoadedPointsData(4), 4);
      const reset = getActiveSortedIndexAttribute(geom)!.array as Uint32Array;
      expect(Array.from(reset.subarray(0, 4))).toEqual([0, 1, 2, 3]);
    });

    it('append (fromInstance) keeps the PREFIX permutation and gives only the suffix identity', () => {
      // Suffix-only ordering on append — the points twin of the lines
      // case below; resetting the whole attribute would flash the node
      // unsorted on every progressive refinement. GSplats deliberately
      // take that trade instead; see the Phase 4 Stage 2 spec rationale.
      const geom = pool.acquirePointsGeometry('append-points', 6);
      pool.updatePointsGeometry(geom, createMockLoadedPointsData(4), 4);
      writeSortedIndexOrdering(geom, new Uint32Array([3, 2, 1, 0]), 4);
      while (pumpSortedIndexOrderingApply(geom).more) {
        /* an ordering streams into the back buffer and swaps in on completion */
      }

      pool.updatePointsGeometry(geom, createMockLoadedPointsData(6), 6, { fromInstance: 4 });
      const ordering = getActiveSortedIndexAttribute(geom)!.array as Uint32Array;
      expect(Array.from(ordering.subarray(0, 4))).toEqual([3, 2, 1, 0]); // prefix preserved
      expect(Array.from(ordering.subarray(4, 6))).toEqual([4, 5]); // suffix identity
      expect(geom.instanceCount).toBe(6);
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

    it('disposes the superseded geometry after a successful terminal grow', () => {
      const geom1 = pool.acquirePointsGeometry('terminal-points', 1000);
      const disposeSpy = vi.spyOn(geom1, 'dispose');

      const geom2 = pool.acquirePointsGeometry('terminal-points', 2000, {
        canRegrow: false,
      });

      expect(geom2).not.toBe(geom1);
      expect(disposeSpy).toHaveBeenCalledTimes(1);
      expect(pool.getStats().pooledBuffers).toBe(0);
    });

    it('does not accumulate superseded pairs across an unsliced ladder climb', () => {
      pool.acquirePointsGeometry('unsliced-ladder', 1000);
      pool.acquirePointsGeometry('unsliced-ladder', 2000, { canRegrow: false });
      pool.acquirePointsGeometry('unsliced-ladder', 4000, { canRegrow: false });

      expect(pool.getStats().pooledBuffers).toBe(0);
      expect(pool.getStats().evictions).toBe(2);
    });

    it('keeps the successful replacement when superseded disposal throws', () => {
      const oldGeometry = pool.acquirePointsGeometry('throwing-dispose', 1000);
      oldGeometry.addEventListener('dispose', () => {
        throw new Error('dispose listener failed');
      });

      let replacement: THREE.InstancedBufferGeometry | undefined;
      expect(() => {
        replacement = pool.acquirePointsGeometry('throwing-dispose', 2000, {
          canRegrow: false,
        });
      }).not.toThrow();

      expect(pool.activeBuffers.get('throwing-dispose')?.geometry).toBe(replacement);
      expect(pool.getStats().pooledBuffers).toBe(0);
    });

    it('keeps the superseded geometry pooled when the node can still regrow', () => {
      const geom1 = pool.acquirePointsGeometry('growing-points', 1000);
      const disposeSpy = vi.spyOn(geom1, 'dispose');

      pool.acquirePointsGeometry('growing-points', 2000);

      expect(disposeSpy).not.toHaveBeenCalled();
      expect(pool.getStats().pooledBuffers).toBe(1);
    });

    it('disposes superseded Lines and GSplats geometries after terminal grows', () => {
      const line = pool.acquireLinesGeometry('terminal-lines', 500);
      const splat = pool.acquireGSplatsGeometry('terminal-gsplats', 300);
      const lineDispose = vi.spyOn(line, 'dispose');
      const splatDispose = vi.spyOn(splat, 'dispose');

      pool.acquireLinesGeometry('terminal-lines', 2000, { canRegrow: false });
      pool.acquireGSplatsGeometry('terminal-gsplats', 1000, { canRegrow: false });

      expect(lineDispose).toHaveBeenCalledTimes(1);
      expect(splatDispose).toHaveBeenCalledTimes(1);
      expect(pool.getStats().pooledBuffers).toBe(0);
    });

    it('grow-swap cancels a staged ordering apply left on the OLD geometry', () => {
      // The grow path (release + reacquire inside acquire) never goes
      // through releaseDepthSortNode — the coordinator still tracks the
      // NODE, which now points at the fresh geometry. A staged ordering
      // left on the old geometry would therefore never be pumped again,
      // and the strong applies map would pin its ordering array(s) until
      // some future tenant's identity write. The pool-release cancel
      // (splat-texture-storage.test.ts covers the direct-release flavor
      // on the gsplats adapter) must fire on this path too.
      const geom1 = pool.acquirePointsGeometry('node1', 1000);
      pool.updatePointsGeometry(geom1, createMockLoadedPointsData(4), 4);
      writeSortedIndexOrdering(geom1, new Uint32Array([3, 2, 1, 0]), 4);
      expect(hasPendingSortedIndexOrderingApply(geom1)).toBe(true);

      const geom2 = pool.acquirePointsGeometry('node1', 2000); // grow: release + reacquire
      expect(geom2).not.toBe(geom1);
      expect(hasPendingSortedIndexOrderingApply(geom1)).toBe(false);
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
      const geom = pool.acquireLinesGeometry('line1', 500);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);
      // Acquire must NOT bump instanceCount — only a successful texel
      // write does (updateGeometry's post-write prepare).
      expect(geom.instanceCount).toBe(0);
      expect(geom.drawRange.count).toBe(6);
      expect(geom.getAttribute('aQuadCorner')).toBeDefined();

      // Texture-backed storage: per-segment data lives in the pooled
      // RGBA32F line texture (6 texels/segment); aSortedIndex is the only
      // per-instance attribute. Capacity = ceil(500 * 1.5) = 750
      // segments -> >= 750 * 24 floats in the backing store.
      const texture = getLineTexture(geom);
      expect(texture).not.toBeNull();
      expect((texture!.image.data as Float32Array).length).toBeGreaterThanOrEqual(
        750 * LINE_FLOATS_PER_SEGMENT
      );
      const sortedIndex = getActiveSortedIndexAttribute(geom)!;
      expect(sortedIndex).toBeDefined();
      expect(sortedIndex.array).toBeInstanceOf(Uint32Array);
      // Ownership marker consumed by the commit handoff's dispose gate.
      expect(geom.userData.luxarPooled).toBe(true);
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

    it('updateLinesGeometry rejects a torn multi-attribute write atomically', () => {
      // The fused texel writer's fail-loud guard runs BEFORE any store:
      // a short array in the MIDDLE of the source set (endColors here)
      // throws ONE aggregate error with nothing written — no torn mix of
      // new positions + old colors in the line texture (the interleaved
      // era needed a separate pre-flight sweep for the same guarantee).
      const makeLinesData = (count: number): ProcessedLinesData => ({
        startPositions: new Float32Array(count * 3),
        endPositions: new Float32Array(count * 3),
        startColors: new Float32Array(count * 3),
        endColors: new Float32Array(count * 3),
        startWidths: new Float32Array(count),
        endWidths: new Float32Array(count),
        startSharpness: new Float32Array(count),
        endSharpness: new Float32Array(count),
        segmentLengths: new Float32Array(count),
        startJointCode: new Float32Array(count),
        endJointCode: new Float32Array(count),
        segmentCount: count,
      });

      const geom = pool.acquireLinesGeometry('torn-lines', 2);

      // Sentinel commit: a fully valid update.
      const sentinel = makeLinesData(2);
      sentinel.startPositions.set([1, 2, 3, 4, 5, 6]);
      pool.updateLinesGeometry(geom, sentinel, 2);

      const texture = getLineTexture(geom)!;
      const texels = texture.image.data as Float32Array;
      const rangesAfterSentinel = texture.updateRanges.length;
      const versionAfterSentinel = texture.version;
      // texel 0 of each segment starts at the 24-float stride: startPos.xyz.
      expect(texels[0]).toBe(1);
      expect(geom.instanceCount).toBe(2);

      // Torn attempt: valid (new) positions but a SHORT endColors.
      const torn = makeLinesData(2);
      torn.startPositions.set([9, 9, 9, 9, 9, 9]);
      torn.endColors = new Float32Array(3); // needs 2 * 3 = 6
      expect(() => pool.updateLinesGeometry(geom, torn, 2)).toThrow(
        /writeLineTexels: source arrays shorter than count=2 \(.*endColors=3/
      );

      // Atomic: the position texels are UNTOUCHED (sentinel survives)…
      expect(texels[0]).toBe(1);
      expect(texels[1]).toBe(2);
      expect(texels[2]).toBe(3);
      expect(texels[LINE_FLOATS_PER_SEGMENT]).toBe(4);
      // …no NEW upload was registered beyond the sentinel's…
      expect(texture.updateRanges.length).toBe(rangesAfterSentinel);
      expect(texture.version).toBe(versionAfterSentinel);
      // …and instanceCount was not re-prepared (set only AFTER a
      // successful write).
      expect(geom.instanceCount).toBe(2);
    });

    it('append (fromInstance) keeps the PREFIX permutation and gives only the suffix identity', () => {
      // The suffix-only ordering write is load-bearing: resetting the
      // whole aSortedIndex on an append would destroy the live depth-sort
      // permutation and flash the node unsorted on every progressive
      // refinement until the re-sort lands. GSplats deliberately take that
      // trade instead; see the Phase 4 Stage 2 spec rationale.
      const makeLinesData = (count: number): ProcessedLinesData => ({
        startPositions: new Float32Array(count * 3),
        endPositions: new Float32Array(count * 3),
        startColors: new Float32Array(count * 3),
        endColors: new Float32Array(count * 3),
        startWidths: new Float32Array(count),
        endWidths: new Float32Array(count),
        startSharpness: new Float32Array(count),
        endSharpness: new Float32Array(count),
        segmentLengths: new Float32Array(count),
        startJointCode: new Float32Array(count),
        endJointCode: new Float32Array(count),
        segmentCount: count,
      });
      const geom = pool.acquireLinesGeometry('append-lines', 6);
      pool.updateLinesGeometry(geom, makeLinesData(4), 4);
      // The SortWorker landed a permutation between commits.
      writeSortedIndexOrdering(geom, new Uint32Array([3, 2, 1, 0]), 4);
      while (pumpSortedIndexOrderingApply(geom).more) {
        /* an ordering streams into the back buffer and swaps in on completion */
      }

      pool.updateLinesGeometry(geom, makeLinesData(6), 6, { fromInstance: 4 });
      const ordering = getActiveSortedIndexAttribute(geom)!.array as Uint32Array;
      expect(Array.from(ordering.subarray(0, 4))).toEqual([3, 2, 1, 0]); // prefix preserved
      expect(Array.from(ordering.subarray(4, 6))).toEqual([4, 5]); // suffix identity
      expect(geom.instanceCount).toBe(6);
    });
  });

  describe('GSplats Geometry', () => {
    it('should create InstancedBufferGeometry for gsplats', () => {
      const geom = pool.acquireGSplatsGeometry('splat1', 300);
      expect(geom).toBeInstanceOf(THREE.InstancedBufferGeometry);

      // Texture-backed storage: splat data lives in the pooled RGBA32F
      // texture; the aSortedIndex pair is the only per-instance data.
      const texture = getSplatTexture(geom);
      expect(texture).not.toBeNull();
      // Capacity = ceil(300 * 1.5) = 450 splats -> >= 450*16 floats.
      expect((texture!.image.data as Float32Array).length).toBeGreaterThanOrEqual(450 * 16);
      const sortedIndex = getActiveSortedIndexAttribute(geom)!;
      expect(sortedIndex).toBeDefined();
      expect(sortedIndex.array).toBeInstanceOf(Uint32Array);
      // Ownership marker consumed by the commit handoff's dispose gate.
      expect(geom.userData.luxarPooled).toBe(true);
    });

    it('should reuse gsplats geometry', () => {
      const geom1 = pool.acquireGSplatsGeometry('splat1', 300);
      const geom2 = pool.acquireGSplatsGeometry('splat1', 250);

      expect(geom2).toBe(geom1);
      expect(pool.getStats().reuses).toBe(1);
    });
  });

  describe('Grow-path OOM re-claim window', () => {
    // The grow path releases the node's active buffer FIRST, then
    // allocates the replacement — the big typed-array allocation is the
    // realistic OOM throw site, and a grow is exactly when memory is
    // tightest. A throw anywhere after the release used to leave the
    // mesh's still-rendered geometry sitting in the free pool (adoptable
    // by another node → foreign data). The adapters now wrap everything
    // from the release onward and RE-CLAIM the released buffer on a
    // throw. The tests inject the throw by stubbing `evictUnused` (the
    // release path calls it after pushing the buffer into its free
    // bucket, so the re-claim must find and restore it).

    /** Stub evictUnused to throw, assert fn propagates it, restore. */
    const withThrowingEvict = (fn: () => void): void => {
      const spy = vi.spyOn(pool, 'evictUnused').mockImplementation(() => {
        throw new Error('synthetic OOM');
      });
      try {
        expect(fn).toThrow('synthetic OOM');
      } finally {
        spy.mockRestore();
      }
    };

    it('points: a throw during grow re-claims the released buffer', () => {
      const geom1 = pool.acquirePointsGeometry('grow-oom-p', 1000); // capacity 1500

      withThrowingEvict(() => pool.acquirePointsGeometry('grow-oom-p', 2000, { canRegrow: false }));

      // The node's active entry is the ORIGINAL geometry, re-claimed.
      const active = pool.activeBuffers.get('grow-oom-p');
      expect(active).toBeDefined();
      expect(active!.geometry).toBe(geom1);
      expect(active!.inUse).toBe(true);
      // No free-bucket entry aliases it.
      for (const buffers of pool.points.pointBuffers.values()) {
        expect(buffers).not.toContain(active);
      }
      // A follow-up acquire at the original count reuses it in place.
      const reusesBefore = pool.getStats().reuses;
      expect(pool.acquirePointsGeometry('grow-oom-p', 1000)).toBe(geom1);
      expect(pool.getStats().reuses).toBe(reusesBefore + 1);
    });

    it('lines: a throw during grow re-claims the released buffer', () => {
      const geom1 = pool.acquireLinesGeometry('grow-oom-l', 500); // capacity 750

      withThrowingEvict(() => pool.acquireLinesGeometry('grow-oom-l', 2000, { canRegrow: false }));

      const active = pool.activeBuffers.get('grow-oom-l');
      expect(active).toBeDefined();
      expect(active!.geometry).toBe(geom1);
      expect(active!.inUse).toBe(true);
      for (const buffers of pool.lines.lineBuffers.values()) {
        expect(buffers).not.toContain(active);
      }
      const reusesBefore = pool.getStats().reuses;
      expect(pool.acquireLinesGeometry('grow-oom-l', 500)).toBe(geom1);
      expect(pool.getStats().reuses).toBe(reusesBefore + 1);
    });

    it('gsplats: a throw during grow re-claims the released buffer', () => {
      const geom1 = pool.acquireGSplatsGeometry('grow-oom-g', 300); // capacity 450

      withThrowingEvict(() =>
        pool.acquireGSplatsGeometry('grow-oom-g', 1000, { canRegrow: false })
      );

      const active = pool.activeBuffers.get('grow-oom-g');
      expect(active).toBeDefined();
      expect(active!.geometry).toBe(geom1);
      expect(active!.inUse).toBe(true);
      for (const buffers of pool.gsplats.gsplatBuffers.values()) {
        expect(buffers).not.toContain(active);
      }
      const reusesBefore = pool.getStats().reuses;
      expect(pool.acquireGSplatsGeometry('grow-oom-g', 300)).toBe(geom1);
      expect(pool.getStats().reuses).toBe(reusesBefore + 1);
    });

    it('points: a throw AFTER a successful replacement allocation reinstates the original and disposes the replacement', () => {
      // The reclaim's `current !== released` branch: the fresh allocation
      // SUCCEEDS and installs a replacement active entry, then the
      // post-allocation byte sweep throws (evictUnused is called twice on
      // this path — first inside releaseGeometry, then inside the fresh-
      // alloc tail AFTER activeBuffers.set). The reclaim must reinstate
      // the ORIGINAL released buffer as the active entry FIRST and
      // dispose the never-handed-out replacement LAST (via the guarded
      // disposeReplacementAfterReclaim helper).
      const geom1 = pool.acquirePointsGeometry('grow-oom-p2', 1000); // capacity 1500
      const originalDispose = vi.spyOn(geom1, 'dispose');

      let replacement: THREE.InstancedBufferGeometry | undefined;
      const onReplacementDispose = vi.fn();
      let evictCalls = 0;
      const spy = vi.spyOn(pool, 'evictUnused').mockImplementation(() => {
        evictCalls++;
        if (evictCalls === 1) return 0; // releaseGeometry's sweep — succeed (0 evicted)
        // Second call site: fresh-alloc tail, replacement already
        // installed as the node's active entry — capture it BEFORE
        // throwing so its later disposal is observable.
        const entry = pool.activeBuffers.get('grow-oom-p2');
        replacement = entry?.geometry as THREE.InstancedBufferGeometry;
        replacement?.addEventListener('dispose', onReplacementDispose);
        throw new Error('synthetic OOM after alloc');
      });
      try {
        expect(() =>
          pool.acquirePointsGeometry('grow-oom-p2', 2000, { canRegrow: false })
        ).toThrow('synthetic OOM after alloc');
      } finally {
        spy.mockRestore();
      }

      // The throw fired at the SECOND call site, after the replacement
      // entry was installed (this is what distinguishes the branch from
      // the plain release-time-throw tests above).
      expect(evictCalls).toBe(2);
      expect(replacement).toBeDefined();
      expect(replacement).not.toBe(geom1);

      // Original reinstated as the node's active entry.
      const active = pool.activeBuffers.get('grow-oom-p2');
      expect(active).toBeDefined();
      expect(active!.geometry).toBe(geom1);
      expect(active!.inUse).toBe(true);
      expect(originalDispose).not.toHaveBeenCalled();

      // Replacement disposed (it was never handed to the caller) and
      // absent from every free bucket.
      expect(onReplacementDispose).toHaveBeenCalledTimes(1);
      for (const buffers of pool.points.pointBuffers.values()) {
        for (const b of buffers) {
          expect(b.geometry).not.toBe(replacement);
        }
      }

      // A follow-up acquire at the original count reuses the original.
      const reusesBefore = pool.getStats().reuses;
      expect(pool.acquirePointsGeometry('grow-oom-p2', 1000)).toBe(geom1);
      expect(pool.getStats().reuses).toBe(reusesBefore + 1);
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
      pool.acquireLinesGeometry('line1', 500);
      pool.acquireGSplatsGeometry('splat1', 300);

      pool.dispose();

      const stats = pool.getStats();
      expect(stats.activeBuffers).toBe(0);
      expect(stats.pooledBuffers).toBe(0);
    });

    it('a throwing dispose listener aborts the sweep without leaving disposed buffers adoptable', () => {
      // Contract pinned here: evictUnused has NO catch around
      // geometry.dispose() — a throwing dispose listener aborts the
      // pass and the error propagates. But each bucket is committed
      // (evicted buffers removed) BEFORE any dispose runs, so the
      // throw can only leak not-yet-disposed buffers (already
      // unreachable from the pool) — never leave a
      // disposed-but-adoptable zombie that a later acquire (or the
      // grow-reclaim path) could reinstate.
      const testPool = new GPUBufferPool(20, 2); // evict after 2 frames
      const geomA = testPool.acquirePointsGeometry('a', 1000);
      const geomB = testPool.acquirePointsGeometry('b', 900); // same bucket as 'a'
      testPool.releasePointsGeometry('a');
      testPool.releasePointsGeometry('b');
      for (let i = 0; i < 4; i++) testPool.beginFrame(); // both now stale

      geomA.addEventListener('dispose', () => {
        throw new Error('listener boom');
      });
      const disposeB = vi.spyOn(geomB, 'dispose');

      expect(() => testPool.evictUnused()).toThrow('listener boom');

      // The bucket was committed before disposal: neither buffer is
      // adoptable any more…
      expect(testPool.getStats().pooledBuffers).toBe(0);
      const fresh = testPool.acquirePointsGeometry('c', 900);
      expect(fresh).not.toBe(geomA);
      expect(fresh).not.toBe(geomB);
      // …and the abort left geomB leaked-undisposed (the safe
      // direction), not disposed-in-pool.
      expect(disposeB).not.toHaveBeenCalled();
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

    it('keeps stats.allocations and typeStats.points.allocations in sync when the post-allocation sweep throws', () => {
      // Both counters must bump together BEFORE the fresh-allocation
      // eviction sweep: a throwing sweep (dispose listeners can throw)
      // used to land between them and permanently desync the pair.
      const testPool = new GPUBufferPool(20, 300);
      const evictSpy = vi.spyOn(testPool, 'evictUnused');
      evictSpy.mockImplementationOnce(() => 0); // first fresh-alloc sweep: fine
      evictSpy.mockImplementationOnce(() => {
        throw new Error('sweep boom');
      });

      testPool.acquirePointsGeometry('a', 1000);
      expect(() => testPool.acquirePointsGeometry('b', 1000)).toThrow('sweep boom');

      const stats = testPool.getStats();
      expect(stats.allocations).toBe(2);
      expect(stats.byType.points.allocations).toBe(2);
      expect(stats.allocations).toBe(stats.byType.points.allocations);
    });

    it('should track active vs pooled buffers', () => {
      pool.acquirePointsGeometry('node1', 1000);
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

        // Texture-backed lines: the zero-capacity hazard is the ordering
        // attribute (the texture always has >= 1 row) — mirror points.
        const lineGeom = empty.acquireLinesGeometry('zero-lines', 0);
        const lineIdx = lineGeom.getAttribute('aSortedIndex');
        expect((lineIdx.array as Uint32Array).length).toBeGreaterThan(0);
        expect((getLineTexture(lineGeom)!.image.data as Float32Array).length).toBeGreaterThan(0);

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
