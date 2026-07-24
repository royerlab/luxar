/**
 * Points buffer-pool adapter.
 *
 * Owns the points-specific pool state (per-capacity buckets) and
 * implements acquire/release/update for Points geometries. Shared
 * state (activeBuffers, stats, frame counter) is read from the
 * GPUBufferPool reference passed at construction.
 *
 * Per-point data lives in the RGBA32F point texture attached at
 * creation (3 texels/point — see `../point-geometry.ts` for the layout
 * and `../element-texture-layout.ts` for the addressing math); the only
 * per-instance attribute is `aSortedIndex`. The layout is FIXED
 * regardless of which optional fields the dataset carries, so — unlike
 * the interleaved era's dtype/spec bucketing — ANY pooled points
 * geometry fits ANY points node (mirroring the gsplats adapter): reuse
 * keys on capacity alone. Dtype normalization happens at upload time
 * with the same `widenToFloat32` calls (and the same fallback fills)
 * the interleaved path used, so texel values are bit-identical to what
 * the old per-attribute writes produced.
 *
 * The top-level GPUBufferPool owns only shared coordination logic
 * (eviction, frame counter, dispose); this adapter owns Points-specific
 * buffer layout and update behavior.
 */

import * as THREE from 'three';
import { widenToFloat32 } from '../widen-to-float32';
import {
  attachPointStorage,
  getPointTexture,
  pointsNormalizationDivisor,
  writePointTexels,
  type PointTexelSource,
} from '../point-geometry';
import { writeSortedIndexIdentity, writeSortedIndexIdentityRange } from '../element-storage';
import { clampPointCapacity } from '../element-texture-layout';
import type { LoadedPointsData } from '../../data/data-loader-types';
import type { PooledBuffer } from './pool-stats';
import { chooseCapacity } from './capacity';

/**
 * Points render as instanced unit quads, so the indexed draw range is
 * always the 2-triangle base quad (6 indices) while `instanceCount`
 * carries the number of point sprites.
 *
 * Called ONLY from `updateGeometry`, AFTER the texel write succeeds —
 * never at acquire time. Bumping `instanceCount` before the write would
 * let a throwing write draw the new count over stale/zero texels (a
 * grown reuse would render ~N duplicate sprites of point 0 until the
 * next update); the gsplats adapter has the same ordering.
 */
function preparePointsGeometryForDraw(
  geometry: THREE.BufferGeometry,
  pointCount: number
): THREE.InstancedBufferGeometry {
  const instanced = geometry as THREE.InstancedBufferGeometry;
  instanced.instanceCount = pointCount;
  instanced.setDrawRange(0, 6);
  return instanced;
}

function createPointsGeometry(pointCapacity: number): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  const quadCorners = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const indices = new Uint16Array([0, 1, 2, 2, 1, 3]);
  geometry.setAttribute('aQuadCorner', new THREE.BufferAttribute(quadCorners, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.instanceCount = 0;
  geometry.setDrawRange(0, 6);

  // Point data lives in the RGBA32F texture attached here (disposed BY
  // the geometry's dispose event, so every pool dispose site frees it);
  // `aSortedIndex` is the only per-instance attribute.
  attachPointStorage(geometry, pointCapacity);
  // Ownership marker: the commit handoff disposes a replaced geometry
  // ONLY when it is not pool-owned (pool geometries are released back to
  // the free list by acquire, never disposed by the commit layer).
  geometry.userData.luxarPooled = true;
  return geometry;
}

/**
 * Shared-state surface the points adapter reads/writes on the parent
 * GPUBufferPool. Kept narrow so the adapter can be unit-tested with a
 * minimal stub instead of a full pool instance.
 */
export interface PointsAdapterHost {
  activeBuffers: Map<string, PooledBuffer>;
  readonly frameCount: number;
  stats: {
    allocations: number;
    reuses: number;
    evictions: number;
    capacityGrowths: number;
    deferredEvictions: number;
  };
  typeStats: {
    points: { allocations: number; reuses: number; evictions: number };
    lines: { allocations: number; reuses: number; evictions: number };
    gsplats: { allocations: number; reuses: number; evictions: number };
  };
  _lastAcquireRebuilt: boolean;
  getBucket(count: number): number;
  evictUnused(fromAcquire?: boolean): number;
}

export class PointsBufferAdapter {
  /** @internal — pool-bucketed reusable point geometries. */
  readonly pointBuffers = new Map<number, PooledBuffer[]>();

  constructor(private readonly host: PointsAdapterHost) {}

  acquireGeometry(nodeId: string, pointCount: number): THREE.InstancedBufferGeometry {
    const host = this.host;
    host._lastAcquireRebuilt = false;

    // Per-node texture bound: width × maxTextureSize / 3 texels (5.59M
    // points on a 4096-class device). Warns once; the update path
    // clamps its written count to the texture capacity to match.
    pointCount = clampPointCapacity(pointCount);

    const active = host.activeBuffers.get(nodeId);
    if (active && active.type === 'points') {
      if (active.capacity >= pointCount) {
        active.lastUsedFrame = host.frameCount;
        host.stats.reuses++;
        host.typeStats.points.reuses++;
        return active.geometry as THREE.InstancedBufferGeometry;
      }
      // Grow = RELEASE + REACQUIRE — an in-place rebuild strands the
      // old GL/GPU buffer in the renderer caches (hard leak under the
      // WebGPU renderer via the strong Info.memoryMap). Releasing
      // lets the old geometry reach geometry.dispose() through the
      // normal evictor, which frees its buffers (and its texture, via
      // the geometry dispose event) correctly on every backend. The
      // released buffer cannot be re-picked below (capacity <
      // pointCount); the fall-through best-fit/fresh-alloc paths set
      // _lastAcquireRebuilt and the allocation counters.
      host.stats.capacityGrowths++;
      // OOM RE-CLAIM WINDOW. A grow is exactly when memory is tightest,
      // and everything from the release onward can throw: the release's
      // own evict sweep (graceFrame −1, so it may even dispose the buffer
      // we just released), and above all the fresh allocation's big
      // Float32Array in createPointsGeometry — the realistic OOM throw
      // site. Without this catch, the throw propagates out of the commit
      // BEFORE its handoff try/finally, leaving the mesh's still-rendered
      // geometry sitting in the free pool — adoptable by ANOTHER node,
      // which would then overwrite it with foreign data under this
      // node's transform. On a throw we re-claim the released buffer
      // (splice it back out of its free bucket and restore it as this
      // node's active entry) and re-throw, so the pool books stay
      // consistent with what the mesh actually renders. The re-claim can
      // never conflict with best-fit adoption for THIS call: the
      // released buffer's capacity < pointCount, so the scan below can
      // never have picked it. If the release-time sweep already disposed
      // the buffer, re-claim finds nothing and we just re-throw — the
      // mesh keeps rendering its OLD content until the next successful
      // commit (the classic backend lazily re-creates GL resources from
      // the surviving CPU arrays — re-consuming memory under the very
      // OOM being handled, briefly), but no
      // pooled entry aliases it (documented residual).
      const released = active;
      try {
        this.releaseGeometry(nodeId);
        return this.adoptOrAllocate(nodeId, pointCount);
      } catch (error) {
        this.reclaimAfterFailedGrow(nodeId, released);
        throw error;
      }
    }

    return this.adoptOrAllocate(nodeId, pointCount);
  }

  /**
   * Best-fit adoption from the free buckets, else a fresh allocation.
   * Extracted from `acquireGeometry` so the grow path can wrap it (and
   * the preceding release) in the OOM re-claim try/catch above.
   */
  private adoptOrAllocate(nodeId: string, pointCount: number): THREE.InstancedBufferGeometry {
    const host = this.host;

    // BEST-fit, not first-fit — see the gsplats adapter for rationale.
    // The fixed texel layout means any pooled points geometry fits any
    // points node: capacity is the only matching criterion.
    let bestList: PooledBuffer[] | null = null;
    let bestIndex = -1;
    let bestCapacity = Infinity;
    for (const pooled of this.pointBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        if (candidate.capacity >= pointCount && candidate.capacity < bestCapacity) {
          bestList = pooled;
          bestIndex = i;
          bestCapacity = candidate.capacity;
        }
      }
    }
    if (bestList) {
      const candidate = bestList[bestIndex];
      bestList.splice(bestIndex, 1);
      // Adopted geometry may still carry the previous tenant's
      // instanceCount + texels; draw nothing until this node's write
      // sets the real count (a throwing write must not render the
      // previous tenant's content under this node's transform).
      (candidate.geometry as THREE.InstancedBufferGeometry).instanceCount = 0;
      candidate.inUse = true;
      candidate.lastUsedFrame = host.frameCount;
      host.activeBuffers.set(nodeId, candidate);
      host.stats.reuses++;
      host.typeStats.points.reuses++;
      host._lastAcquireRebuilt = true;
      return candidate.geometry as THREE.InstancedBufferGeometry;
    }

    host._lastAcquireRebuilt = true;
    // Growth headroom (1.5×) can itself cross the texture bound; clamp
    // the chosen capacity too (still >= pointCount, which was clamped).
    const capacity = clampPointCapacity(chooseCapacity(pointCount));
    const geometry = createPointsGeometry(capacity);

    const newBuffer: PooledBuffer = {
      geometry,
      capacity,
      type: 'points',
      inUse: true,
      lastUsedFrame: host.frameCount,
    };

    host.activeBuffers.set(nodeId, newBuffer);
    // Both allocation counters bump together, BEFORE the sweep: the
    // sweep runs dispose listeners that may throw, and a bump split
    // across it would permanently desync stats.allocations from
    // typeStats.points.allocations on a throwing sweep.
    host.stats.allocations++;
    host.typeStats.points.allocations++;
    // Fresh allocations count against the byte budget too — sweep idle
    // pooled buffers (see growth-path note above).
    host.evictUnused(true);

    return geometry;
  }

  /**
   * Undo a failed grow (see the OOM re-claim comment in
   * `acquireGeometry`): restore the buffer released at the start of the
   * grow as the node's active entry, so the geometry the mesh still
   * renders is neither adoptable from the free pool nor orphaned.
   *
   * Two sub-cases:
   * - A replacement entry was already installed for the node before the
   *   throw (fresh allocation succeeded, then the post-allocation byte
   *   sweep threw): dispose it — it was never handed to the caller.
   * - The released buffer is found in a free bucket: splice it out and
   *   re-activate it. If the release-time sweep disposed it, it is in no
   *   bucket — nothing to restore (the caller re-throws either way).
   */
  private reclaimAfterFailedGrow(nodeId: string, released: PooledBuffer): void {
    const host = this.host;

    // Reinstate FIRST, dispose the replacement LAST: the throw class this
    // catch defends against plausibly came from a dispose listener, so
    // disposing before re-claiming could itself throw — replacing the
    // original error and leaving `released` free-pooled (the exact
    // aliasing this method exists to prevent).
    const current = host.activeBuffers.get(nodeId);
    if (current && current !== released) {
      host.activeBuffers.delete(nodeId);
    }

    for (const pooled of this.pointBuffers.values()) {
      const index = pooled.indexOf(released);
      if (index !== -1) {
        pooled.splice(index, 1);
        released.inUse = true;
        released.lastUsedFrame = host.frameCount;
        host.activeBuffers.set(nodeId, released);
        this.disposeReplacementAfterReclaim(current);
        return;
      }
    }
    // Not found: already disposed by the release-time sweep — see the
    // "documented residual" note in acquireGeometry.
    this.disposeReplacementAfterReclaim(current);
  }

  /**
   * Dispose a replacement buffer installed before a late throw (never
   * handed to the caller). Guarded: a throwing dispose listener must not
   * mask the original acquire error nor undo the reinstatement above.
   */
  private disposeReplacementAfterReclaim(current: PooledBuffer | undefined): void {
    if (!current) return;
    try {
      current.geometry.dispose();
    } catch {
      // Swallow: the original acquire error is already propagating.
    }
  }

  releaseGeometry(nodeId: string): void {
    const host = this.host;
    const buffer = host.activeBuffers.get(nodeId);
    if (!buffer || buffer.type !== 'points') return;

    host.activeBuffers.delete(nodeId);
    buffer.inUse = false;
    // Stamp the release frame so acquire-triggered byte sweeps later in
    // this same frame grace the buffer (see EvictorCtx.graceFrame) — a
    // released buffer otherwise carries the frame of its last ACQUIRE
    // and the dataset-switch grace never matches. Also makes the
    // just-released buffer the freshest LRU reuse candidate.
    buffer.lastUsedFrame = host.frameCount;

    const bucket = host.getBucket(buffer.capacity);
    if (!this.pointBuffers.has(bucket)) {
      this.pointBuffers.set(bucket, []);
    }
    this.pointBuffers.get(bucket)!.push(buffer);

    host.evictUnused();
  }

  updateGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: LoadedPointsData,
    count: number,
    options?: { preserveOrdering?: boolean; fromInstance?: number }
  ): void {
    const instanced = geometry;
    const texture = getPointTexture(instanced);
    if (!texture) {
      throw new Error(
        'PointsBufferAdapter.updateGeometry: geometry has no point texture — ' +
          'was it acquired from the pool?'
      );
    }
    // Append fast path (Phase 4 Stage 2): the commit layer sets
    // `fromInstance` to the prefix count already on the GPU when this
    // commit only extends it, so the fused writer + ranged upload touch
    // just the `[fromInstance, count)` suffix (see writePointTexels).
    // 0 means a full write.
    const fromInstance = options?.fromInstance ?? 0;

    // Widen each field EXACTLY as the interleaved era did — the same
    // widenToFloat32 calls on the same source slices with the same
    // normalization divisors and the same fallback fills — so the
    // floats written to texels are bit-identical to what the old
    // per-attribute writes uploaded.
    const positionsF32 =
      data.positions instanceof Float32Array
        ? data.positions
        : widenToFloat32(data.positions as ArrayLike<number>);

    // Color layout: 3 (RGB) or 4 (RGBA — alpha = per-point opacity).
    // Strides the staged slice and the writer's per-point reads; the
    // absent-colors fill stays RGB (the writer stamps the 1.0 opaque
    // identity into texel2.y unconditionally either way).
    const colorK: 3 | 4 = data.colors ? (data.colorComponents ?? 3) : 3;
    let colorsF32: Float32Array;
    if (data.colors) {
      colorsF32 = widenToFloat32(
        data.colors.subarray(0, count * colorK) as ArrayLike<number>,
        pointsNormalizationDivisor(data.colors, /*normalized=*/ true)
      );
    } else {
      colorsF32 = new Float32Array(count * 3);
      colorsF32.fill(1.0); // white default
    }

    let radiiF32: Float32Array;
    if (data.radii) {
      radiiF32 = widenToFloat32(
        data.radii.subarray(0, count) as ArrayLike<number>,
        pointsNormalizationDivisor(data.radii, /*normalized=*/ true)
      );
    } else {
      radiiF32 = new Float32Array(count);
      radiiF32.fill(0.5);
    }

    let sharpnessF32: Float32Array;
    if (data.sharpness) {
      sharpnessF32 = widenToFloat32(
        data.sharpness.subarray(0, count) as ArrayLike<number>,
        pointsNormalizationDivisor(data.sharpness, /*normalized=*/ true)
      );
    } else {
      sharpnessF32 = new Float32Array(count);
      sharpnessF32.fill(0.5); // default sharpness knob -> beta=2 (Gaussian)
    }

    // Scalars: absent ⇒ omitted from the source, and the writer stamps
    // the 0.0 identity into texel2.x unconditionally (the fixed-layout
    // analog of the interleaved era's zero-fill of a bound aScalar).
    const scalarsF32 = data.scalars
      ? widenToFloat32(
          data.scalars.subarray(0, count) as ArrayLike<number>,
          pointsNormalizationDivisor(data.scalars, /*normalized=*/ true)
        )
      : undefined;

    const texelSrc: PointTexelSource = {
      positions: positionsF32,
      colors: colorsF32,
      colorComponents: colorK,
      radii: radiiF32,
      sharpness: sharpnessF32,
      scalars: scalarsF32,
    };
    // One fused pass over the staged arrays into the texel layout
    // (replaces the five per-attribute strided writes), then identity
    // ordering. The writer clamps to the texture capacity; mirror that
    // clamp in instanceCount so a bound-clamped node never draws
    // instances whose texels were not written.
    count = writePointTexels(texture, texelSrc, count, { fromPoint: fromInstance });
    if (fromInstance > 0) {
      // Append: keep the prefix's existing ordering and give the appended
      // points identity until a re-sort lands (fromInstance and
      // preserveOrdering are mutually exclusive — append needs
      // count > prev, preserveOrdering needs count === prev).
      writeSortedIndexIdentityRange(instanced, fromInstance, count);
    } else if (!options?.preserveOrdering) {
      // `preserveOrdering` (commit path decides — see
      // commit-points-geometry.ts) keeps a same-count recommit's existing
      // depth-sort permutation as a no-worse prior until the re-sort
      // lands; every other full write resets to identity.
      writeSortedIndexIdentity(instanced, count);
    }

    preparePointsGeometryForDraw(geometry, count);

    // Presence stamps: the fixed texel layout always carries every slot
    // (identity fills when a field is absent), so real source presence
    // rides userData — `hasScalars` drives `supportsScalarColormap`, the
    // rest serve debug/E2E introspection (datasets written before the
    // Python writer stamped `has_colors/has_radii/has_sharpness` into
    // attrs carry no such keys; the flags otherwise live in a
    // metadata dict that never reaches attrs). Refreshed on EVERY update —
    // pool geometries are reused across tenants, and a presence flip must
    // not leak the previous tenant's stamp (the texel writer already
    // restores the identity fills).
    if (!instanced.userData) instanced.userData = {};
    instanced.userData.hasScalars = data.scalars !== undefined;
    instanced.userData.hasColors = !!data.colors;
    instanced.userData.hasRadii = !!data.radii;
    instanced.userData.hasSharpness = !!data.sharpness;
    // RGBA-alpha presence (texel2.y carries a REAL per-point opacity,
    // not the 1.0 identity). syncPointMaterialWithGeometry pushes this
    // into the material's uHasElementAlpha gate on every commit.
    instanced.userData.hasElementAlpha = colorK === 4;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
    delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;

    if (data.metadata.bounds) {
      instanced.boundingBox = data.metadata.bounds.clone();
    } else {
      const box = new THREE.Box3();
      const v = new THREE.Vector3();
      const positions = data.positions;
      for (let i = 0; i < count; i++) {
        v.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        box.expandByPoint(v);
      }
      instanced.boundingBox = box;
    }
    instanced.boundingSphere = new THREE.Sphere();
    instanced.boundingBox.getBoundingSphere(instanced.boundingSphere);
  }

  dispose(): void {
    for (const buffers of this.pointBuffers.values()) {
      for (const buffer of buffers) {
        buffer.geometry.dispose();
      }
    }
    this.pointBuffers.clear();
  }
}
