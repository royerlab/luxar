/**
 * Lines buffer-pool adapter.
 *
 * Owns the lines-specific pool state (per-capacity buckets) and
 * implements acquire/release/update for Line geometries. Shared state
 * (activeBuffers, stats, frame counter) is read from the GPUBufferPool
 * reference passed at construction.
 *
 * Per-segment data lives in the RGBA32F line texture attached at
 * creation (6 texels/segment — see `../line-geometry.ts` for the layout
 * and `../element-texture-layout.ts` for the addressing math); the only
 * per-instance attribute is `aSortedIndex`. The layout is FIXED
 * regardless of which optional fields the dataset carries, so — unlike
 * the interleaved era's colormap-scalar spec bucketing — ANY pooled
 * lines geometry fits ANY lines node (mirroring the points/gsplats
 * adapters): reuse keys on capacity alone.
 *
 * The top-level GPUBufferPool owns only shared coordination logic
 * (eviction, frame counter, dispose); this adapter owns Lines-specific
 * buffer layout and update behavior.
 */

import * as THREE from 'three';
import {
  attachLineStorage,
  computeLineBounds,
  getLineTexture,
  stampLinePresenceFlags,
  writeLineTexels,
} from '../line-geometry';
import {
  cancelSortedIndexOrderingApply,
  writeSortedIndexIdentity,
  writeSortedIndexIdentityRange,
} from '../element-storage';
import { clampLineCapacity } from '../element-texture-layout';
import type { ProcessedLinesData } from '../../types/lines';
import type { PooledBuffer } from './pool-stats';
import { chooseCapacity } from './capacity';

/**
 * Lines render as instanced unit quads, so the indexed draw range is
 * always the 2-triangle base quad (6 indices) while `instanceCount`
 * carries the number of segments.
 *
 * Called ONLY from `updateGeometry`, AFTER the texel write succeeds —
 * never at acquire time. Bumping `instanceCount` before the write would
 * let a throwing write draw the new count over stale/zero texels; the
 * points/gsplats adapters have the same ordering.
 */
function prepareLinesGeometryForDraw(
  geometry: THREE.BufferGeometry,
  segmentCount: number
): THREE.InstancedBufferGeometry {
  const instanced = geometry as THREE.InstancedBufferGeometry;
  instanced.instanceCount = segmentCount;
  instanced.setDrawRange(0, 6);
  return instanced;
}

function createLinesGeometry(segmentCapacity: number): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  const quadCorners = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const indices = new Uint16Array([0, 1, 2, 2, 1, 3]);
  geometry.setAttribute('aQuadCorner', new THREE.BufferAttribute(quadCorners, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.instanceCount = 0;
  geometry.setDrawRange(0, 6);

  // Segment data lives in the RGBA32F texture attached here (disposed BY
  // the geometry's dispose event, so every pool dispose site frees it);
  // The `aSortedIndex`/`aSortedIndexB` ordering pair is the only
  // per-instance data (two distinct buffers from attach, so a new ordering
  // swaps atomically and the vertex layout never changes under a cached
  // WebGPU pipeline).
  attachLineStorage(geometry, segmentCapacity);
  // Ownership marker: the commit handoff disposes a replaced geometry
  // ONLY when it is not pool-owned (pool geometries are released back to
  // the free list by acquire, never disposed by the commit layer).
  geometry.userData.luxarPooled = true;
  return geometry;
}

/**
 * Shared-state surface the lines adapter reads/writes on the parent
 * GPUBufferPool. Kept narrow so the adapter can be unit-tested with a
 * minimal stub instead of a full pool instance.
 */
export interface LinesAdapterHost {
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

export class LinesBufferAdapter {
  /** @internal — pool-bucketed reusable line geometries. */
  readonly lineBuffers = new Map<number, PooledBuffer[]>();

  constructor(private readonly host: LinesAdapterHost) {}

  acquireGeometry(nodeId: string, segmentCount: number): THREE.InstancedBufferGeometry {
    const host = this.host;
    host._lastAcquireRebuilt = false;

    // Per-node texture bound: width × maxTextureSize / 6 texels (2.79M
    // segments on a 4096-class device). Warns once; the update path
    // clamps its written count to the texture capacity to match.
    segmentCount = clampLineCapacity(segmentCount);

    const active = host.activeBuffers.get(nodeId);
    if (active && active.type === 'lines') {
      if (active.capacity >= segmentCount) {
        active.lastUsedFrame = host.frameCount;
        host.stats.reuses++;
        host.typeStats.lines.reuses++;
        return active.geometry as THREE.InstancedBufferGeometry;
      }
      // Grow = RELEASE + REACQUIRE — an in-place rebuild strands the
      // old GL/GPU buffer in the renderer caches (hard leak under the
      // WebGPU renderer via the strong Info.memoryMap). See the points
      // adapter for the full rationale. (The interleaved era's scalar
      // spec-set mismatch branch is gone: the fixed texel layout always
      // carries the scalar slots.)
      host.stats.capacityGrowths++;
      // OOM RE-CLAIM WINDOW — see the points adapter's twin comment.
      // The released buffer can never be picked by the best-fit scan
      // for this call: its capacity < segmentCount.
      const released = active;
      try {
        this.releaseGeometry(nodeId);
        return this.adoptOrAllocate(nodeId, segmentCount);
      } catch (error) {
        this.reclaimAfterFailedGrow(nodeId, released);
        throw error;
      }
    }

    return this.adoptOrAllocate(nodeId, segmentCount);
  }

  /**
   * Best-fit adoption from the free buckets, else a fresh allocation —
   * see the points adapter's twin comment.
   */
  private adoptOrAllocate(nodeId: string, segmentCount: number): THREE.InstancedBufferGeometry {
    const host = this.host;

    // BEST-fit, not first-fit — see the gsplats adapter for rationale.
    // The fixed texel layout means any pooled lines geometry fits any
    // lines node: capacity is the only matching criterion.
    let bestList: PooledBuffer[] | null = null;
    let bestIndex = -1;
    let bestCapacity = Infinity;
    for (const pooled of this.lineBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        if (candidate.capacity >= segmentCount && candidate.capacity < bestCapacity) {
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
      host.typeStats.lines.reuses++;
      host._lastAcquireRebuilt = true;
      return candidate.geometry as THREE.InstancedBufferGeometry;
    }

    host._lastAcquireRebuilt = true;
    // Growth headroom (1.5×) can itself cross the texture bound; clamp
    // the chosen capacity too (still >= segmentCount, which was clamped).
    const capacity = clampLineCapacity(chooseCapacity(segmentCount));
    const geometry = createLinesGeometry(capacity);

    const newBuffer: PooledBuffer = {
      geometry,
      capacity,
      type: 'lines',
      inUse: true,
      lastUsedFrame: host.frameCount,
    };

    host.activeBuffers.set(nodeId, newBuffer);
    // Both allocation counters bump together, BEFORE the sweep — a
    // throwing sweep must not desync them (see the points adapter twin).
    host.stats.allocations++;
    host.typeStats.lines.allocations++;
    // Fresh allocations count against the byte budget too — sweep idle
    // pooled buffers (see growth-path note above).
    host.evictUnused(true);
    return geometry;
  }

  /**
   * Undo a failed grow — see the points adapter's twin comment for the
   * full sub-case breakdown (replacement-entry disposal + free-bucket
   * re-claim; a buffer the release-time sweep disposed stays gone).
   */
  private reclaimAfterFailedGrow(nodeId: string, released: PooledBuffer): void {
    const host = this.host;

    // Reinstate FIRST, dispose the replacement LAST — see the points
    // adapter's twin comment.
    const current = host.activeBuffers.get(nodeId);
    if (current && current !== released) {
      host.activeBuffers.delete(nodeId);
    }

    for (const pooled of this.lineBuffers.values()) {
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
    this.disposeReplacementAfterReclaim(current);
  }

  /** See the points adapter's twin comment. */
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
    if (!buffer || buffer.type !== 'lines') return;

    host.activeBuffers.delete(nodeId);
    buffer.inUse = false;
    // A released buffer is no longer drawn, so an ordering still
    // streaming into it is dead work — and `chunkedApplies` keys its
    // state by GEOMETRY in a strong Map, so leaving it would pin this
    // geometry AND its ordering (4 B/element — ~32 MB for an 8M node)
    // on the free list until the buffer is re-acquired or evicted.
    // Dispose already cancels via the geometry's own listener; release
    // is the other exit from "in use" and needs the same treatment.
    cancelSortedIndexOrderingApply(buffer.geometry as THREE.InstancedBufferGeometry);

    // Stamp the release frame so acquire-triggered byte sweeps later in
    // this same frame grace the buffer (see EvictorCtx.graceFrame) — a
    // released buffer otherwise carries the frame of its last ACQUIRE
    // and the dataset-switch grace never matches. Also makes the
    // just-released buffer the freshest LRU reuse candidate.
    buffer.lastUsedFrame = host.frameCount;

    const bucket = host.getBucket(buffer.capacity);
    if (!this.lineBuffers.has(bucket)) {
      this.lineBuffers.set(bucket, []);
    }
    this.lineBuffers.get(bucket)!.push(buffer);

    host.evictUnused();
  }

  updateGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: ProcessedLinesData,
    count: number,
    options?: { preserveOrdering?: boolean; fromInstance?: number }
  ): void {
    const instanced = geometry;
    const texture = getLineTexture(instanced);
    if (!texture) {
      throw new Error(
        'LinesBufferAdapter.updateGeometry: geometry has no line texture — ' +
          'was it acquired from the pool?'
      );
    }
    // Append fast path (Phase 4 Stage 2): the commit layer sets
    // `fromInstance` to the prefix count already on the GPU when this
    // commit only extends it, so the fused writer + ranged upload touch
    // just the `[fromInstance, count)` suffix (see writeLineTexels).
    // 0 means a full write.
    const fromInstance = options?.fromInstance ?? 0;

    // One fused pass over the staged arrays into the texel layout
    // (replaces the 11–13 per-attribute strided writes; the writer's
    // fail-loud guard runs before ANY store, retiring the interleaved
    // era's separate pre-flight torn-write sweep), then identity
    // ordering. The writer clamps to the texture capacity; mirror that
    // clamp in instanceCount so a bound-clamped node never draws
    // instances whose texels were not written.
    count = writeLineTexels(texture, data, count, { fromSegment: fromInstance });
    if (fromInstance > 0) {
      // Append: keep the prefix's existing ordering and give the appended
      // segments identity until a re-sort lands (fromInstance and
      // preserveOrdering are mutually exclusive — append needs
      // count > prev, preserveOrdering needs count === prev).
      writeSortedIndexIdentityRange(instanced, fromInstance, count);
    } else if (!options?.preserveOrdering) {
      // `preserveOrdering` (commit path decides — see
      // commit-lines-geometry.ts) keeps a same-count recommit's existing
      // depth-sort permutation as a no-worse prior until the re-sort
      // lands; every other full write resets to identity.
      writeSortedIndexIdentity(instanced, count);
    }

    prepareLinesGeometryForDraw(geometry, count);

    // Scalar presence stamp (drives `supportsScalarColormap`) — refreshed
    // on EVERY update; pool geometries are reused across tenants.
    stampLinePresenceFlags(instanced, data);

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
    delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;

    // CRITICAL: Recompute bounding box after position updates, expanded
    // by max width so the rendered footprint is covered by frustum
    // culling (shared helper — see line-geometry.ts).
    computeLineBounds(instanced, data, count);
  }

  dispose(): void {
    for (const buffers of this.lineBuffers.values()) {
      for (const buffer of buffers) {
        buffer.geometry.dispose();
      }
    }
    this.lineBuffers.clear();
  }
}
