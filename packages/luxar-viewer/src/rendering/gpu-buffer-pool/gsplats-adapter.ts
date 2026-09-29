/**
 * GSplats buffer-pool adapter.
 *
 * Owns the gsplats-specific pool state (per-capacity buckets) and
 * implements acquire/release/update for Gaussian-splat geometries.
 * Shared state (activeBuffers, stats, commit counter) is read from the
 * GPUBufferPool reference passed at construction.
 *
 * The top-level GPUBufferPool owns only shared coordination logic
 * (eviction, commit counter, dispose); this adapter owns GSplats-specific
 * buffer layout and update behavior.
 */

import * as THREE from 'three';
import {
  attachSplatStorage,
  computeMaxCholeskyRowNorm,
  getSplatTexture,
  stampGSplatPresenceFlags,
  writeSplatTexels,
} from '../gsplat-geometry';
import {
  cancelSortedIndexOrderingApply,
  holdSortedIndexDrawForAppend,
  holdSortedIndexDrawFromSeed,
  repairSortedIndexForCount,
  sortedIndexDrawHoldTarget,
  writeSortedIndexIdentity,
} from '../element-storage';
import { clampSplatCapacity } from '../element-texture-layout';
import type { GSplatsProjectionBounds } from '../../types/gsplats';
import type { PooledBuffer } from './pool-stats';
import { chooseCapacity } from './capacity';
import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../config/constants';

/**
 * Packed GSplats data ready for GPU upload (from gsplats/projection.ts).
 *
 * Carries arrays only — the splat count M travels as the positional
 * `count` argument of `updateGeometry`, matching the points/lines
 * adapters (whose payloads' own count fields are pipeline metadata the
 * pool never reads).
 */
export interface PackedGSplatsData {
  centers3D: Float32Array; // M * 3
  amplitudes: Float32Array; // M
  /** M * 6, row-major [L00, L10, L11, L20, L21, L22] per splat */
  choleskyFactors: Float32Array;
  colors: Float32Array; // M * 3 (RGB) or M * 4 (RGBA — alpha = per-splat opacity)
  /** Components per color item: 3 (RGB) or 4 (RGBA). Absent means 3. */
  colorComponents?: 3 | 4;
  labelIndices?: Uint32Array;
  /**
   * Precomputed cull metadata from the projection's fused scan (AABB of
   * `centers3D` + max Cholesky row norm). When present, `updateGeometry`
   * skips its two O(N) main-thread scans (mirrors the points adapter's
   * `data.metadata.bounds` fast path); absent ⇒ scan fallback.
   */
  bounds?: GSplatsProjectionBounds;
}

/** Ordering options for {@link GSplatsBufferAdapter.updateGeometry}. */
export interface GSplatsUpdateOptions {
  /** Same-count recommit: keep the existing permutation verbatim. */
  preserveOrdering?: boolean;
  /** Count change on the same buffers: rebuild the permutation from this count. */
  repairFromCount?: number;
  /** Append fast path: the prefix count already on the GPU (0 = full write). */
  fromInstance?: number;
  /**
   * A full write that EXTENDS the previous population into a grown geometry:
   * the previous geometry's drawn permutation of `[0, seedOrdering.length)`,
   * which the draw is held on until the grown population's ordering lands.
   */
  seedOrdering?: Uint32Array;
}

function createGSplatsGeometry(splatCapacity: number): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();

  const quadPositions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  geometry.setAttribute('aQuadCorner', new THREE.Float32BufferAttribute(quadPositions, 2));
  geometry.setIndex([0, 1, 2, 2, 1, 3]);
  // Draw nothing until the first successful texel write sets the real
  // count (mirrors the points adapter): a throwing first write must not
  // let the handoff draw capacity-many unwritten texels.
  geometry.instanceCount = 0;
  geometry.setDrawRange(0, 6);

  // Splat data lives in the RGBA32F texture attached here (disposed BY
  // the geometry's dispose event, so every pool dispose site frees it);
  // The `aSortedIndex`/`aSortedIndexB` ordering pair is the only
  // per-instance data (two distinct buffers from attach, so a new ordering
  // swaps atomically and the vertex layout never changes under a cached
  // WebGPU pipeline).
  attachSplatStorage(geometry, splatCapacity);
  // Ownership marker — see the points adapter's twin comment.
  geometry.userData.luxarPooled = true;
  return geometry;
}

export interface GSplatsAdapterHost {
  activeBuffers: Map<string, PooledBuffer>;
  readonly commitCount: number;
  stats: {
    allocations: number;
    reuses: number;
    evictions: number;
    capacityGrowths: number;
    deferredEvictions: number;
    byteBudgetEvictions: number;
  };
  typeStats: {
    points: { allocations: number; reuses: number; evictions: number };
    lines: { allocations: number; reuses: number; evictions: number };
    gsplats: { allocations: number; reuses: number; evictions: number };
  };
  _lastAcquireRebuilt: boolean;
  getBucket(count: number): number;
  evictUnused(fromAcquire?: boolean): number;
  registerPooledGeometryInvalidation(geometry: THREE.BufferGeometry): void;
}

/**
 * Write the ordering a gsplat commit draws with, and return the instance
 * count to draw NOW (below `count` only while an append's draw is held).
 *
 * - APPEND (`fromSplat > 0`): the texture upload stays suffix-only and the draw
 *   is HELD at the previous population in its previous order until the grown
 *   population's own ordering lands ({@link holdSortedIndexDrawForAppend}).
 *   The old sorted prefix plus a storage-order suffix would render two
 *   independently ordered populations, and a full storage-order reset draws
 *   the whole node unsorted for the 75-225 ms a large sort takes.
 * - `seedOrdering` (a full write into a GROWN geometry that extends the
 *   previous population): the same hold, seeded with the previous geometry's
 *   drawn permutation ({@link holdSortedIndexDrawFromSeed}).
 * - `preserveOrdering` (chosen by commit-gsplats-geometry.ts) keeps a
 *   same-count prior while its re-sort lands; skipping that write also
 *   registers no new update range.
 * - `repairFromCount` on a count CHANGE rebuilds the existing permutation over
 *   the new population rather than discarding it — an nD re-slice changes the
 *   resident count at almost every step, so this is what a timelapse takes.
 * - A vouched prior on a geometry whose draw is still HELD is only valid over
 *   the DRAWN prefix, so it is repaired from there instead.
 * - Anything else resets to identity, which re-homes the geometry on slot 0.
 *   The commit path must therefore call noteDepthSortCommit after this update;
 *   its immediate syncSortedIndexSlot pushes the new slot to visual and pick
 *   materials before either can draw (the per-frame pump re-asserts it).
 */
function writeCommitOrdering(
  geometry: THREE.InstancedBufferGeometry,
  count: number,
  fromSplat: number,
  options: GSplatsUpdateOptions | undefined
): number {
  const drawnBefore = geometry.instanceCount;
  if (fromSplat > 0) return holdSortedIndexDrawForAppend(geometry, drawnBefore, count);
  if (options?.seedOrdering) {
    return holdSortedIndexDrawFromSeed(geometry, options.seedOrdering, count);
  }
  const vouched = options?.preserveOrdering === true || options?.repairFromCount !== undefined;
  if (vouched && sortedIndexDrawHoldTarget(geometry) !== undefined) {
    repairSortedIndexForCount(geometry, drawnBefore, count);
    return count;
  }
  if (options?.preserveOrdering) return count;
  const repairFrom = options?.repairFromCount;
  if (repairFrom !== undefined && repairFrom > 0) {
    // A repair deliberately leaves the slot where it is: its callers only fire
    // when the tenant, geometry and buffers are all unchanged, so the
    // slot/uniform pairing is already established.
    repairSortedIndexForCount(geometry, repairFrom, count);
  } else {
    writeSortedIndexIdentity(geometry, count);
  }
  return count;
}

export class GSplatsBufferAdapter {
  /** @internal — pool-bucketed reusable gsplat geometries. */
  readonly gsplatBuffers = new Map<number, PooledBuffer[]>();

  constructor(private readonly host: GSplatsAdapterHost) {}

  acquireGeometry(nodeId: string, splatCount: number): THREE.InstancedBufferGeometry {
    const host = this.host;
    host._lastAcquireRebuilt = false;

    // Per-node texture bound: width × maxTextureSize / 4 texels (4.19M
    // splats on a 4096-class device). Warns once; the update path
    // clamps its written count to the texture capacity to match.
    splatCount = clampSplatCapacity(splatCount);

    const active = host.activeBuffers.get(nodeId);
    // See the points adapter: `luxarInvalidated` guards a geometry whose
    // out-of-band `dispose()` already fired (defense-in-depth).
    if (active && active.type === 'gsplats' && !active.geometry.userData.luxarInvalidated) {
      if (active.capacity >= splatCount) {
        active.lastUsedCommit = host.commitCount;
        host.stats.reuses++;
        host.typeStats.gsplats.reuses++;
        return active.geometry as THREE.InstancedBufferGeometry;
      }
      // Grow = RELEASE + REACQUIRE — an in-place rebuild strands the
      // old GL/GPU buffer in the renderer caches (hard leak under the
      // WebGPU renderer via the strong Info.memoryMap). See the
      // points adapter for the full rationale. Fall-through best-fit/
      // fresh-alloc sets _lastAcquireRebuilt + allocation counters.
      host.stats.capacityGrowths++;
      // OOM RE-CLAIM WINDOW — see the points adapter's twin comment.
      // The released buffer can never be picked by the best-fit scan
      // for this call (capacity < splatCount).
      const released = active;
      let grown: THREE.InstancedBufferGeometry;
      try {
        this.releaseGeometry(nodeId);
        grown = this.adoptOrAllocate(nodeId, splatCount);
      } catch (error) {
        this.reclaimAfterFailedGrow(nodeId, released);
        throw error;
      }
      // Post-grow reclaim — see the twin comment in `lines-adapter.ts` for why
      // the pair released above escapes BOTH existing sweeps (the release
      // sweep runs before the replacement registers; the acquire sweep
      // grace-skips a buffer stamped with the current commit, and the adopt
      // path sweeps not at all). Outside the try so a throwing dispose
      // listener is not mistaken for a failed grow.
      host.evictUnused(false);
      return grown;
    }

    return this.adoptOrAllocate(nodeId, splatCount);
  }

  /**
   * Best-fit adoption from the free buckets, else a fresh allocation —
   * see the points adapter's twin comment.
   */
  private adoptOrAllocate(nodeId: string, splatCount: number): THREE.InstancedBufferGeometry {
    const host = this.host;

    // BEST-fit, not first-fit: scan every pooled candidate and claim the
    // smallest adequate one. Map iteration order is bucket-insertion
    // order, so first-fit could pin an arbitrarily oversized buffer
    // (e.g. a 52 MB 1M-capacity buffer) to a small node until release.
    let bestList: PooledBuffer[] | null = null;
    let bestIndex = -1;
    let bestCapacity = Infinity;
    for (const pooled of this.gsplatBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        // Skip a free-bucket resident disposed out-of-band — see the
        // points adapter's twin comment.
        if (candidate.geometry.userData.luxarInvalidated) continue;
        if (candidate.capacity >= splatCount && candidate.capacity < bestCapacity) {
          bestList = pooled;
          bestIndex = i;
          bestCapacity = candidate.capacity;
        }
      }
    }
    if (bestList) {
      const candidate = bestList[bestIndex];
      bestList.splice(bestIndex, 1);
      // See the points adapter's twin comment: no previous-tenant
      // content may draw through a throwing write.
      (candidate.geometry as THREE.InstancedBufferGeometry).instanceCount = 0;
      candidate.inUse = true;
      candidate.lastUsedCommit = host.commitCount;
      host.activeBuffers.set(nodeId, candidate);
      host.stats.reuses++;
      host.typeStats.gsplats.reuses++;
      host._lastAcquireRebuilt = true;
      return candidate.geometry as THREE.InstancedBufferGeometry;
    }

    host._lastAcquireRebuilt = true;
    // Growth headroom (1.5×) can itself cross the texture bound; clamp
    // the chosen capacity too (still >= splatCount, which was clamped).
    const capacity = clampSplatCapacity(chooseCapacity(splatCount), false);
    const geometry = createGSplatsGeometry(capacity);
    // Self-invalidation — see the points adapter's twin comment.
    host.registerPooledGeometryInvalidation(geometry);

    const newBuffer: PooledBuffer = {
      geometry,
      capacity,
      type: 'gsplats',
      inUse: true,
      lastUsedCommit: host.commitCount,
    };

    host.activeBuffers.set(nodeId, newBuffer);
    // Both allocation counters bump together, BEFORE the sweep — a
    // throwing sweep must not desync them (see the points adapter twin).
    host.stats.allocations++;
    host.typeStats.gsplats.allocations++;
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

    for (const pooled of this.gsplatBuffers.values()) {
      const index = pooled.indexOf(released);
      if (index !== -1) {
        pooled.splice(index, 1);
        released.inUse = true;
        released.lastUsedCommit = host.commitCount;
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
    if (!buffer || buffer.type !== 'gsplats') return;

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

    // Stamp the release commit so acquire-triggered byte sweeps later in
    // this same commit grace the buffer (see EvictorCtx.graceCommit) — a
    // released buffer otherwise carries the commit of its last ACQUIRE
    // and the dataset-switch grace never matches. Also makes the
    // just-released buffer the freshest LRU reuse candidate.
    buffer.lastUsedCommit = host.commitCount;

    const bucket = host.getBucket(buffer.capacity);
    if (!this.gsplatBuffers.has(bucket)) {
      this.gsplatBuffers.set(bucket, []);
    }
    this.gsplatBuffers.get(bucket)!.push(buffer);

    host.evictUnused();
  }

  updateGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: PackedGSplatsData,
    count: number,
    truncationRadius: number = GSPLAT_DEFAULT_TRUNCATION_RADIUS,
    options?: GSplatsUpdateOptions
  ): void {
    const texture = getSplatTexture(geometry);
    if (!texture) {
      throw new Error(
        'GSplatsBufferAdapter.updateGeometry: geometry has no splat texture — ' +
          'was it acquired from the pool?'
      );
    }
    // Append fast path (Phase 4 Stage 2): the commit layer sets `fromInstance`
    // to the prefix count already on the GPU when this commit only extends it,
    // so the fused writer + ranged upload touch just the `[fromSplat, count)`
    // suffix (see writeSplatTexels). 0 means a full write.
    const fromSplat = options?.fromInstance ?? 0;
    // One fused pass over the staged arrays into the texel layout
    // (replaces the six per-attribute strided writes), then identity
    // ordering. The writer clamps to the texture capacity; mirror that
    // clamp in instanceCount so a bound-clamped node never draws
    // instances whose texels were not written.
    count = writeSplatTexels(
      texture,
      {
        centers: data.centers3D,
        choleskyFactors: data.choleskyFactors,
        amplitudes: data.amplitudes,
        colors: data.colors,
        colorComponents: data.colorComponents,
        labelIndices: data.labelIndices,
      },
      count,
      { fromSplat }
    );
    // Presence stamp — shared chokepoint with the non-pool writer paths;
    // see stampGSplatPresenceFlags (refresh on every update: pool tenants).
    stampGSplatPresenceFlags(geometry, { colorComponents: data.colorComponents });
    const drawCount = writeCommitOrdering(geometry, count, fromSplat, options);
    geometry.instanceCount = drawCount;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
    delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;

    // CRITICAL: Recompute bounding box from updated center positions
    // and expand by max splat extent for correct frustum culling.
    // For each splat, the per-axis extent is truncationRadius × σ_d, where
    // σ_d = ||L[d,:]|| (the row norm of the Cholesky factor). We use the
    // max row norm across all splats and axes as a conservative expansion.
    //
    // Fast path: the projection's fused scan precomputed both (AABB +
    // max row norm) — no O(N) main-thread scans per commit. Computed
    // over the FULL projected set: if `count` was capacity-clamped
    // below it, the box is a conservative superset (safe for frustum
    // culling — same trade the points adapter's metadata.bounds path
    // accepts). Fallback: direct scans over the written count.
    const box = new THREE.Box3();
    let maxRowNorm: number;
    if (data.bounds) {
      const { min, max } = data.bounds;
      box.min.set(min[0], min[1], min[2]);
      box.max.set(max[0], max[1], max[2]);
      maxRowNorm = data.bounds.maxRowNorm;
    } else {
      const v = new THREE.Vector3();
      for (let i = 0; i < count; i++) {
        v.set(data.centers3D[i * 3], data.centers3D[i * 3 + 1], data.centers3D[i * 3 + 2]);
        box.expandByPoint(v);
      }
      maxRowNorm = computeMaxCholeskyRowNorm(data.choleskyFactors, count);
    }
    const expansion = maxRowNorm * truncationRadius;
    box.expandByScalar(expansion);

    geometry.boundingBox = box;
    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);
    geometry.boundingSphere = sphere;
  }

  dispose(): void {
    // Drain the buckets BEFORE disposing — see the points adapter.
    const geometries: THREE.BufferGeometry[] = [];
    for (const buffers of this.gsplatBuffers.values()) {
      for (const buffer of buffers) geometries.push(buffer.geometry);
    }
    this.gsplatBuffers.clear();
    for (const geometry of geometries) geometry.dispose();
  }
}
