/**
 * The pool lifecycle shared by the three texture-backed geometry types
 * (Points, Lines, GSplats): acquire (in-place reuse, grow, best-fit
 * adoption, fresh allocation), the OOM re-claim of a failed grow, release
 * to the free buckets, and dispose.
 *
 * All three store per-element data in an RGBA32F element texture plus the
 * `aSortedIndex`/`aSortedIndexB` ordering pair, with a FIXED layout per
 * type, so any pooled geometry of a type fits any node of that type and
 * reuse keys on capacity alone. The lifecycle is therefore identical; what
 * differs is the per-type {@link TextureBackedLayout} (texture capacity
 * bound, geometry construction) and each subclass's `updateGeometry` (its
 * texel writer, ordering write and bounds).
 *
 * @module rendering/gpu-buffer-pool/texture-backed-adapter
 */

import * as THREE from 'three';
import {
  cancelSortedIndexOrderingApply,
  holdSortedIndexDrawForAppend,
  holdSortedIndexDrawFromSeed,
  releaseSortedIndexDrawHold,
  repairSortedIndexForCount,
  sortedIndexDrawHoldTarget,
  writeSortedIndexIdentity,
} from '../element-storage';
import type { PooledBuffer } from './pool-stats';
import { FreeBucketMap } from './byte-tracked-maps';
import { chooseCapacity } from './capacity';

/** The texture-backed pooled geometry types. */
export type TextureBackedType = PooledBuffer['type'];

/**
 * Shared-state surface an adapter reads/writes on the parent
 * GPUBufferPool. Kept narrow so an adapter can be unit-tested with a
 * minimal stub instead of a full pool instance.
 */
export interface PoolAdapterHost {
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
  typeStats: Record<TextureBackedType, { allocations: number; reuses: number; evictions: number }>;
  _lastAcquireRebuilt: boolean;
  getBucket(count: number): number;
  evictUnused(fromAcquire?: boolean): number;
  registerPooledGeometryInvalidation(geometry: THREE.BufferGeometry): void;
}

/** What one geometry type contributes to the shared lifecycle. */
export interface TextureBackedLayout {
  readonly type: TextureBackedType;
  /**
   * Clamp an element count to the per-node element-texture bound (width ×
   * maxTextureSize / texels-per-element). `warn` is false for the internal
   * re-clamp of the growth headroom, which may cross the bound by itself.
   */
  clampCapacity(count: number, warn?: boolean): number;
  /** Attach this type's element storage to an instanced quad geometry. */
  attachStorage(geometry: THREE.InstancedBufferGeometry, capacity: number): void;
}

/**
 * The unit quad every texture-backed type instances, with this type's
 * element storage attached.
 *
 * Draws nothing (`instanceCount = 0`) until the first successful texel
 * write sets the real count: a throwing first write must not let the commit
 * handoff draw capacity-many unwritten texels. The element texture is
 * disposed BY the geometry's dispose event, so every pool dispose site
 * frees it. The ordering pair is the only per-instance data — two distinct
 * buffers from attach, so a new ordering swaps atomically and the vertex
 * layout never changes under a cached WebGPU pipeline.
 */
function createInstancedQuadGeometry(
  layout: TextureBackedLayout,
  capacity: number
): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  const quadCorners = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const indices = new Uint16Array([0, 1, 2, 2, 1, 3]);
  geometry.setAttribute('aQuadCorner', new THREE.BufferAttribute(quadCorners, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.instanceCount = 0;
  geometry.setDrawRange(0, 6);
  layout.attachStorage(geometry, capacity);
  // Ownership marker: the commit handoff disposes a replaced geometry
  // ONLY when it is not pool-owned (pool geometries are released back to
  // the free list by acquire, never disposed by the commit layer).
  geometry.userData.luxarPooled = true;
  return geometry;
}

/**
 * The shared acquire/release/dispose lifecycle. Subclasses add their own
 * `updateGeometry`; the parent pool reads {@link buffers} (also exposed
 * under each subclass's historical name) for stats and eviction.
 */
export abstract class TextureBackedAdapter {
  /** @internal — pool-bucketed reusable geometries of this type. */
  readonly buffers = new FreeBucketMap();

  protected constructor(
    protected readonly host: PoolAdapterHost,
    private readonly layout: TextureBackedLayout
  ) {}

  acquireGeometry(nodeId: string, count: number): THREE.InstancedBufferGeometry {
    const host = this.host;
    const { type } = this.layout;
    host._lastAcquireRebuilt = false;

    // Per-node texture bound. Warns once; the update path clamps its
    // written count to the texture capacity to match.
    count = this.layout.clampCapacity(count);

    const active = host.activeBuffers.get(nodeId);
    // `luxarInvalidated` guards against a geometry whose out-of-band
    // `dispose()` already fired (the pool's self-invalidation listener
    // normally deletes the entry, so `active` is usually undefined here —
    // this is defense-in-depth against an ordering where it survived).
    if (!active || active.type !== type || active.geometry.userData.luxarInvalidated) {
      return this.adoptOrAllocate(nodeId, count);
    }
    if (active.capacity >= count) {
      active.lastUsedCommit = host.commitCount;
      host.stats.reuses++;
      host.typeStats[type].reuses++;
      return active.geometry as THREE.InstancedBufferGeometry;
    }
    return this.grow(nodeId, active, count);
  }

  /**
   * Grow = RELEASE + REACQUIRE — an in-place rebuild strands the old GL/GPU
   * buffer in the renderer caches (hard leak under the WebGPU renderer via
   * the strong Info.memoryMap). Releasing lets the old geometry reach
   * geometry.dispose() through the normal evictor, which frees its buffers
   * (and its texture, via the geometry dispose event) correctly on every
   * backend. The released buffer cannot be re-picked by the best-fit scan
   * (capacity < count); the fall-through best-fit/fresh-alloc paths set
   * _lastAcquireRebuilt and the allocation counters.
   */
  private grow(
    nodeId: string,
    released: PooledBuffer,
    count: number
  ): THREE.InstancedBufferGeometry {
    this.host.stats.capacityGrowths++;
    // OOM RE-CLAIM WINDOW. A grow is exactly when memory is tightest, and
    // everything from the release onward can throw: the release's own evict
    // sweep (graceCommit −1, so it may even dispose the buffer we just
    // released), and above all the fresh allocation's big texture array —
    // the realistic OOM throw site. Without this catch, the throw propagates
    // out of the commit BEFORE its handoff try/finally, leaving the mesh's
    // still-rendered geometry sitting in the free pool — adoptable by
    // ANOTHER node, which would then overwrite it with foreign data under
    // this node's transform. On a throw we re-claim the released buffer
    // (splice it back out of its free bucket and restore it as this node's
    // active entry) and re-throw, so the pool books stay consistent with
    // what the mesh actually renders. The re-claim can never conflict with
    // best-fit adoption for THIS call: the released buffer's capacity <
    // count. If the release-time sweep already disposed the buffer, re-claim
    // finds nothing and we just re-throw — the mesh keeps rendering its OLD
    // content until the next successful commit (the classic backend lazily
    // re-creates GL resources from the surviving CPU arrays — re-consuming
    // memory under the very OOM being handled, briefly), but no pooled entry
    // aliases it (documented residual).
    let grown: THREE.InstancedBufferGeometry;
    try {
      this.releaseGeometry(nodeId);
      grown = this.adoptOrAllocate(nodeId, count);
    } catch (error) {
      this.reclaimAfterFailedGrow(nodeId, released);
      throw error;
    }
    // POST-GROW RECLAIM. Without this the buffer released above is stranded
    // for the lifetime of the scene, by two independent mechanisms:
    //
    //  - the sweep inside `releaseGeometry` IS unconditional, but it runs
    //    before the larger replacement is registered active, so it measures
    //    `budget - sumActiveBytes()` against PRE-GROWTH accounting, sees
    //    headroom that no longer exists, and evicts nothing;
    //  - the acquire-side sweep in `adoptOrAllocate` does see the new
    //    accounting, but runs with `graceCommit = commitCount` while
    //    `releaseGeometry` has just stamped the released buffer with that
    //    same commit — so the grace skips exactly the buffer we need it to
    //    take. (And on the ADOPT path there is no acquire sweep at all.)
    //
    // So it escapes both. One unconditional pass here, with the replacement
    // already registered, closes it — re-running the EXISTING policy at the
    // right moment rather than adding a disposal path of its own.
    //
    // Deliberately OUTSIDE the try: a throwing dispose listener must not be
    // mistaken for a failed grow and send us into `reclaimAfterFailedGrow`,
    // which would try to reinstate a buffer this node has already replaced.
    //
    // On the grace it overrides: that grace exists for the DATASET SWITCH
    // (release everything, re-acquire in the same commit — without it each
    // allocation's sweep disposes buffers later acquires would have
    // best-fit). A growth is not that shape: the released buffer is by
    // construction SMALLER than what this node now needs, and co-growing
    // siblings are moving up too, so it is a poor adoption candidate. That
    // is an argument, not a measurement — `reuses` across a dataset switch
    // is the check that keeps it honest.
    this.host.evictUnused(false);
    return grown;
  }

  /**
   * Best-fit adoption from the free buckets, else a fresh allocation.
   * Separate from {@link acquireGeometry} so the grow path can wrap it (and
   * the preceding release) in the OOM re-claim try/catch.
   */
  private adoptOrAllocate(nodeId: string, count: number): THREE.InstancedBufferGeometry {
    return this.adoptBestFit(nodeId, count) ?? this.allocate(nodeId, count);
  }

  /**
   * BEST-fit, not first-fit: scan every pooled candidate and claim the
   * smallest adequate one. Map iteration order is bucket-insertion order, so
   * first-fit could pin an arbitrarily oversized buffer (e.g. a 52 MB
   * 1M-capacity buffer) to a small node until release. The fixed texel
   * layout makes capacity the only matching criterion.
   */
  private adoptBestFit(nodeId: string, count: number): THREE.InstancedBufferGeometry | null {
    const host = this.host;
    let bestBucket: number | null = null;
    let bestIndex = -1;
    let bestCapacity = Infinity;
    for (const [bucket, pooled] of this.buffers) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        // Skip a free-bucket resident whose out-of-band dispose already
        // fired (see registerPooledGeometryInvalidation): adopting it
        // would resurrect the #689 use-after-dispose through the free list.
        if (candidate.geometry.userData.luxarInvalidated) continue;
        if (candidate.capacity >= count && candidate.capacity < bestCapacity) {
          bestBucket = bucket;
          bestIndex = i;
          bestCapacity = candidate.capacity;
        }
      }
    }
    if (bestBucket === null) return null;

    const candidate = this.buffers.takeAt(bestBucket, bestIndex);
    // Adopted geometry may still carry the previous tenant's instanceCount +
    // texels; draw nothing until this node's write sets the real count (a
    // throwing write must not render the previous tenant's content under
    // this node's transform).
    (candidate.geometry as THREE.InstancedBufferGeometry).instanceCount = 0;
    candidate.inUse = true;
    candidate.lastUsedCommit = host.commitCount;
    host.activeBuffers.set(nodeId, candidate);
    host.stats.reuses++;
    host.typeStats[this.layout.type].reuses++;
    host._lastAcquireRebuilt = true;
    return candidate.geometry as THREE.InstancedBufferGeometry;
  }

  /** A fresh geometry at the growth-headroom capacity. */
  private allocate(nodeId: string, count: number): THREE.InstancedBufferGeometry {
    const host = this.host;
    const { type } = this.layout;
    host._lastAcquireRebuilt = true;
    // Growth headroom (1.5×) can itself cross the texture bound; clamp the
    // chosen capacity too (still >= count, which was clamped).
    const capacity = this.layout.clampCapacity(chooseCapacity(count), false);
    const geometry = createInstancedQuadGeometry(this.layout, capacity);
    // Self-invalidation: if this pool-owned geometry is disposed out-of-band
    // (scene-graph teardown calls geometry.dispose() directly), drop its
    // activeBuffers entry so a later acquire can't hand it back.
    host.registerPooledGeometryInvalidation(geometry);

    const newBuffer: PooledBuffer = {
      geometry,
      capacity,
      type,
      inUse: true,
      lastUsedCommit: host.commitCount,
    };

    host.activeBuffers.set(nodeId, newBuffer);
    // Both allocation counters bump together, BEFORE the sweep: the sweep
    // runs dispose listeners that may throw, and a bump split across it
    // would permanently desync stats.allocations from
    // typeStats[type].allocations on a throwing sweep.
    host.stats.allocations++;
    host.typeStats[type].allocations++;
    // Fresh allocations count against the byte budget too — sweep idle
    // pooled buffers.
    host.evictUnused(true);
    return geometry;
  }

  /**
   * Undo a failed grow (see the OOM re-claim comment in {@link grow}):
   * restore the buffer released at the start of the grow as the node's
   * active entry, so the geometry the mesh still renders is neither
   * adoptable from the free pool nor orphaned.
   *
   * Two sub-cases:
   * - A replacement entry was already installed for the node before the
   *   throw (fresh allocation succeeded, then the post-allocation byte sweep
   *   threw): dispose it — it was never handed to the caller.
   * - The released buffer is found in a free bucket: splice it out and
   *   re-activate it. If the release-time sweep disposed it, it is in no
   *   bucket — nothing to restore (the caller re-throws either way).
   */
  private reclaimAfterFailedGrow(nodeId: string, released: PooledBuffer): void {
    const host = this.host;

    // Reinstate FIRST, dispose the replacement LAST: the throw class this
    // catch defends against plausibly came from a dispose listener, so
    // disposing before re-claiming could itself throw — replacing the
    // original error and leaving `released` free-pooled (the exact aliasing
    // this method exists to prevent).
    const current = host.activeBuffers.get(nodeId);
    if (current && current !== released) {
      host.activeBuffers.delete(nodeId);
    }

    if (this.buffers.removeFirst((buffer) => buffer === released, false)) {
      // The failed replacement has no commit to end the old draw hold (an
      // append hold; a no-op for a geometry holding nothing).
      releaseSortedIndexDrawHold(released.geometry as THREE.InstancedBufferGeometry);
      released.inUse = true;
      released.lastUsedCommit = host.commitCount;
      host.activeBuffers.set(nodeId, released);
    }
    // Not found: already disposed by the release-time sweep — see the
    // "documented residual" note in grow().
    disposeReplacementAfterReclaim(current);
  }

  releaseGeometry(nodeId: string): void {
    const host = this.host;
    const buffer = host.activeBuffers.get(nodeId);
    if (!buffer || buffer.type !== this.layout.type) return;

    host.activeBuffers.delete(nodeId);
    buffer.inUse = false;
    // A released buffer is no longer drawn, so an ordering still streaming
    // into it is dead work — and `chunkedApplies` keys its state by GEOMETRY
    // in a strong Map, so leaving it would pin this geometry AND its
    // ordering (4 B/element — ~32 MB for an 8M node) on the free list until
    // the buffer is re-acquired or evicted. Dispose already cancels via the
    // geometry's own listener; release is the other exit from "in use" and
    // needs the same treatment.
    //
    // A held append draw is deliberately LEFT held: ending it here
    // would repair the ordering over the whole held population (O(n) plus a
    // full-range upload) for a geometry nobody draws, on every pool grow.
    // Whichever commit writes this geometry next resolves it — a new
    // tenant's full write discards the hold, and this node's own vouched
    // recommit (a failed grow re-claims the buffer) repairs from the drawn
    // prefix (`writeInstancedCommitOrdering`).
    cancelSortedIndexOrderingApply(buffer.geometry as THREE.InstancedBufferGeometry);

    // Stamp the release commit so acquire-triggered byte sweeps later in
    // this same commit grace the buffer (see EvictorCtx.graceCommit) — a
    // released buffer otherwise carries the commit of its last ACQUIRE and
    // the dataset-switch grace never matches. Also makes the just-released
    // buffer the freshest LRU reuse candidate.
    buffer.lastUsedCommit = host.commitCount;

    const bucket = host.getBucket(buffer.capacity);
    this.buffers.pushBuffer(bucket, buffer);

    host.evictUnused();
  }

  dispose(): void {
    // Drain the buckets BEFORE disposing: dispose() fires the pool's
    // self-invalidation listener, which splices free-bucket arrays —
    // clearing first keeps it a no-op here (see GPUBufferPool.dispose).
    const geometries: THREE.BufferGeometry[] = [];
    for (const buffers of this.buffers.values()) {
      for (const buffer of buffers) geometries.push(buffer.geometry);
    }
    this.buffers.clear();
    for (const geometry of geometries) geometry.dispose();
  }
}

/**
 * Dispose a replacement buffer installed before a late throw (never handed
 * to the caller). Guarded: a throwing dispose listener must not mask the
 * original acquire error nor undo the reinstatement before it.
 */
function disposeReplacementAfterReclaim(current: PooledBuffer | undefined): void {
  if (!current) return;
  try {
    current.geometry.dispose();
  } catch {
    // Swallow: the original acquire error is already propagating.
  }
}

/** Ordering options shared by every texture-backed `updateGeometry`. */
export interface InstancedOrderingOptions {
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

/**
 * Write the ordering a commit draws with, and return the instance count to
 * draw NOW (below `count` only while a growth's draw is held). One rule for
 * Points, Lines and GSplats.
 *
 * - APPEND (`fromInstance > 0`): the texel upload stays suffix-only and the
 *   draw is HELD at the previous population in its previous order until the
 *   grown population's own ordering lands ({@link holdSortedIndexDrawForAppend}).
 *   The old sorted prefix plus a storage-order suffix would render two
 *   independently ordered populations, and a full storage-order reset draws
 *   the whole node unsorted for the 75-225 ms a large sort takes.
 * - `seedOrdering` (a full write into a GROWN geometry that extends the
 *   previous population): the same hold, seeded with the previous geometry's
 *   drawn permutation ({@link holdSortedIndexDrawFromSeed}).
 * - `preserveOrdering` (the commit path decides) keeps a same-count prior
 *   while its re-sort lands; skipping that write also registers no new update
 *   range.
 * - `repairFromCount` on a count CHANGE rebuilds the existing permutation over
 *   the new population rather than discarding it — an nD re-slice changes the
 *   resident count at almost every step, so this is what a timelapse takes.
 * - A vouched prior on a geometry whose draw is still HELD is only valid over
 *   the DRAWN prefix, so it is repaired from there instead.
 * - Anything else resets to identity, which re-homes the geometry on slot 0.
 *   The commit path must therefore call `depthSort.noteCommit` after this
 *   update: its immediate syncSortedIndexSlot pushes the new slot to visual
 *   and pick materials before either can draw, and it releases a hold no
 *   ordering will come for (an order-independent mode, depth sort off).
 */
export function writeInstancedCommitOrdering(
  geometry: THREE.InstancedBufferGeometry,
  count: number,
  options: InstancedOrderingOptions | undefined
): number {
  const { fromInstance = 0, seedOrdering } = options ?? {};
  if (fromInstance > 0) {
    return holdSortedIndexDrawForAppend(geometry, geometry.instanceCount, count);
  }
  if (seedOrdering) return holdSortedIndexDrawFromSeed(geometry, seedOrdering, count);
  writeFullCommitOrdering(geometry, count, options ?? {});
  return count;
}

/** The non-growth half of {@link writeInstancedCommitOrdering}: every element is drawn. */
function writeFullCommitOrdering(
  geometry: THREE.InstancedBufferGeometry,
  count: number,
  { preserveOrdering = false, repairFromCount }: InstancedOrderingOptions
): void {
  const vouched = preserveOrdering || repairFromCount !== undefined;
  if (vouched && sortedIndexDrawHoldTarget(geometry) !== undefined) {
    repairSortedIndexForCount(geometry, geometry.instanceCount, count);
    return;
  }
  if (preserveOrdering) return;
  if (repairFromCount !== undefined && repairFromCount > 0) {
    // A repair deliberately leaves the slot where it is: its callers only fire
    // when the tenant, geometry and buffers are all unchanged, so the
    // slot/uniform pairing is already established.
    repairSortedIndexForCount(geometry, repairFromCount, count);
  } else {
    writeSortedIndexIdentity(geometry, count);
  }
}

/**
 * Finish an update for draw: the indexed draw range is always the
 * 2-triangle base quad (6 indices) while `instanceCount` carries the number
 * of elements, and THREE's cached `_maxInstanceCount` is forced to
 * recompute.
 *
 * Called ONLY after the texel write succeeds — never at acquire time.
 * Bumping `instanceCount` before the write would let a throwing write draw
 * the new count over stale/zero texels (a grown reuse would render ~N
 * duplicates of element 0 until the next update).
 */
export function prepareInstancedQuadForDraw(
  geometry: THREE.InstancedBufferGeometry,
  count: number
): void {
  geometry.instanceCount = count;
  geometry.setDrawRange(0, 6);
  // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
  delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;
}
