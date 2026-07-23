/**
 * GPU Buffer Pool - Geometry reuse for Points, Lines, and GSplats
 *
 * Eliminates GPU buffer allocations by reusing BufferGeometry objects
 * across view updates. Uses LRU eviction to prevent unbounded memory growth.
 *
 * Key optimizations:
 * - Reuses geometries when size AND type match (0ms GPU allocation)
 * - In-place data updates (lines: strided attribute writes; points/gsplats:
 *   fused texel writes into the pooled element texture)
 * - Size-based bucketing for efficient matching
 * - LRU eviction after 300 frames of non-use
 * - Multi-type support for all TypedArray formats
 *
 * TYPE SUPPORT (COMPLETE):
 * - Points: FULL multi-type support — Uint8/Uint16/Float16 sources are
 *   widened to Float32 at upload time (normalization divisors preserved)
 *   and written into the fixed 3-texel RGBA32F point texture, so any
 *   pooled points geometry fits any points node (capacity is the only
 *   matching criterion — see points-adapter.ts).
 * - Lines: Float32Array (per ProcessedLinesData interface); geometries
 *   are bucketed by their scalar spec set (`hasScalars`).
 * - GSplats: Float32Array (per PackedGSplatsData interface), stored in
 *   the 4-texel RGBA32F splat texture.
 *
 * Based on Performance Optimization Specification v3.6.0
 */

import * as THREE from 'three';
import { log, Modules } from '../utils/log';
import { estimateGeometryBytes, invalidateCachedByteSize } from './gpu-buffer-pool/geometry-bytes';
import type { LoadedPointsData } from '../data/data-loader-types';
import type { ProcessedLinesData } from '../types/lines';
import type { PooledBuffer, PoolStats } from './gpu-buffer-pool/pool-stats';
import { evictUntilUnderByteBudget } from './gpu-buffer-pool/byte-budget-evictor';
import { PointsBufferAdapter } from './gpu-buffer-pool/points-adapter';
import { LinesBufferAdapter } from './gpu-buffer-pool/lines-adapter';
import { GSplatsBufferAdapter, type PackedGSplatsData } from './gpu-buffer-pool/gsplats-adapter';

// Per-type spec arrays and helpers live in
// ./gpu-buffer-pool/{points,lines,gsplats}-adapter.

// Geometry-agnostic interleaved-buffer helpers live in
// ./gpu-buffer-pool/attribute-codec for reuse by all adapters.

// Re-export so existing consumers that import these from
// `rendering/gpu-buffer-pool` keep working.
export { estimateGeometryBytes, invalidateCachedByteSize };

// Re-export the public surface so the parent rendering/gpu-buffer-pool.ts
// stub (and existing consumers) keep working unchanged.
export type { PackedGSplatsData } from './gpu-buffer-pool/gsplats-adapter';
export type {
  PooledBuffer,
  TypePoolStats,
  PoolStats,
  PooledBufferRef,
} from './gpu-buffer-pool/pool-stats';
export { selectBuffersToEvict } from './gpu-buffer-pool/eviction-policy';

/**
 * Capacity-sizing primitives moved to `gpu-buffer-pool/capacity.ts` so
 * the per-geometry adapters can import them without re-importing this
 * parent barrel (depcruise no-circular). Re-exported here for callers
 * of the parent module (and the wider rationale for the
 * `DEFAULT_MIN_INSTANCE_CAPACITY = 256` floor lives in that file).
 */
export { __setMinInstanceCapacityForTesting, chooseCapacity } from './gpu-buffer-pool/capacity';

/**
 * GPU buffer pool for reusing THREE.BufferGeometry objects.
 *
 * Manages separate pools for Points, Lines, and GSplats geometries,
 * each with different attribute layouts and update patterns.
 */
export class GPUBufferPool {
  /** @internal — points-specific pool state and methods. */
  readonly points: PointsBufferAdapter;
  /** @internal — lines-specific pool state and methods. */
  readonly lines: LinesBufferAdapter;
  /** @internal — gsplats-specific pool state and methods. */
  readonly gsplats: GSplatsBufferAdapter;

  /** @internal — shared with the per-type adapters. */
  activeBuffers = new Map<string, PooledBuffer>(); // nodeId → active geometry
  /** @internal — shared with the per-type adapters; increments via beginFrame(). */
  frameCount = 0;

  private maxPoolSize: number;
  private evictionFrames: number;
  /**
   * Maximum geometries the pool will dispose in a single
   * `evictUnused()` call when not over the hard limit. Without this
   * cap, a single eviction sweep can dispose dozens of buffers
   * synchronously — each `geometry.dispose()` is 5–20 ms on slow
   * GPUs, so a burst stutters visibly. Remaining evictable buffers
   * are deferred to the next frame's eviction sweep. The
   * `mustEvict` (pool-over-limit) path ignores this cap so the pool
   * never grows unbounded.
   */
  private evictBatchSize: number;
  /**
   * Live byte-budget getter (the single VRAM authority shared with LOD
   * retention). The byte-eviction pass targets TOTAL resident bytes
   * (active + pooled) against this value: pooled buffers are disposed
   * (largest-first) only while active + pooled exceeds the budget, and
   * retained for reuse otherwise. Read live (not snapshotted) so the
   * WebGL-context-loss budget backoff applies immediately. Returns `0`
   * to disable byte-budget eviction (count-only behavior).
   */
  private getByteBudget: () => number;

  /** @internal — shared mutable stats; adapters bump fields here. */
  stats = {
    allocations: 0,
    reuses: 0,
    evictions: 0,
    capacityGrowths: 0,
    /** Pooled buffers skipped this `evictUnused` call due to batch cap. */
    deferredEvictions: 0,
  };

  /**
   * Set by acquire methods when the returned geometry is a fresh
   * allocation or a different pool candidate from the previous active
   * one (growth is always release + reacquire — never an in-place
   * rebuild — for all three types). Callers query this via
   * {@link didLastAcquireRebuildAttributes} immediately after acquire
   * and dispatch `invalidateRenderObjectFor(mesh)` when true so Three's
   * WebGPURenderer drops its stale `RenderObject.vertexBuffers` cache.
   *
   * Reset to `false` on every entry into an acquire method, so this
   * field reflects exclusively the *most recent* acquire result. The
   * read-then-act pattern (acquire → didLastAcquireRebuildAttributes →
   * invalidate) is synchronous in all production call paths.
   */
  /** @internal — written by adapters from acquire paths. */
  _lastAcquireRebuilt = false;

  /**
   * One-shot guard: have we already logged the >100MB pooled-buffer
   * warning? Re-checked per `evictUnused` so the noise stays bounded.
   */
  private largePoolWarningEmitted = false;

  // Per-type stats tracking. @internal — shared with adapters.
  typeStats = {
    points: { allocations: 0, reuses: 0, evictions: 0 },
    lines: { allocations: 0, reuses: 0, evictions: 0 },
    gsplats: { allocations: 0, reuses: 0, evictions: 0 },
  };

  constructor(
    maxPoolSize: number = 20,
    evictionFrames: number = 300,
    evictBatchSize: number = 5,
    getByteBudget: () => number = () => 512_000_000
  ) {
    this.maxPoolSize = maxPoolSize;
    this.evictionFrames = evictionFrames;
    this.evictBatchSize = Math.max(1, evictBatchSize);
    this.getByteBudget = getByteBudget;
    this.points = new PointsBufferAdapter(this);
    this.lines = new LinesBufferAdapter(this);
    this.gsplats = new GSplatsBufferAdapter(this);
  }

  /**
   * Advance the frame counter. Call once per frame before any acquire calls.
   * This ensures eviction timing is based on rendered frames, not acquire calls.
   */
  beginFrame(): void {
    this.frameCount++;
  }

  /**
   * Whether the most recent `acquire*Geometry` call returned a
   * geometry whose GPU-visible buffers (lines' interleaved buffer, the
   * points/gsplat element texture + `aSortedIndex`) differ from what was
   * previously in use for that node — the returned geometry is a fresh
   * allocation or a different pool candidate (growth is release +
   * reacquire for all three types).
   *
   * Callers that hold a `THREE.Mesh` pointing at the previously-active
   * geometry should call `invalidateRenderObjectFor(mesh)` (in
   * `data/scene-loader/invalidate-render-object.ts`) when this returns
   * `true`. That forces Three's WebGPURenderer to discard the cached
   * `RenderObject.vertexBuffers` set; without it, WebGPU binds the old
   * GPU buffer next draw and validation fails with "Instance range …
   * requires a larger buffer than the bound buffer size".
   *
   * The flag is overwritten on every acquire call, so consume it
   * immediately after `acquirePointsGeometry` / `acquireLinesGeometry`
   * / `acquireGSplatsGeometry` returns.
   */
  didLastAcquireRebuildAttributes(): boolean {
    return this._lastAcquireRebuilt;
  }

  // =========================================================================
  // Points Geometry Management
  // =========================================================================

  /**
   * Acquire Points geometry from pool (capacity-aware; the fixed texel
   * layout means any pooled points geometry fits any points node).
   * See `PointsBufferAdapter.acquireGeometry` for implementation.
   */
  acquirePointsGeometry(nodeId: string, pointCount: number): THREE.InstancedBufferGeometry {
    return this.points.acquireGeometry(nodeId, pointCount);
  }

  /** Release Points geometry back to pool. */
  releasePointsGeometry(nodeId: string): void {
    this.points.releaseGeometry(nodeId);
  }

  /**
   * Update Points geometry attributes in-place (zero GPU allocations).
   * @param options - `preserveOrdering`: keep the geometry's existing
   *   `aSortedIndex` permutation instead of resetting it to identity
   *   (same-node same-count recommit — the commit path decides; see
   *   commit-points-geometry.ts). `fromInstance`: append fast path
   *   (Phase 4 Stage 2) — write & upload only the `[fromInstance, count)`
   *   suffix, preserving the prefix texels + permutation already on the
   *   GPU.
   */
  updatePointsGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: LoadedPointsData,
    count: number,
    options?: { preserveOrdering?: boolean; fromInstance?: number }
  ): void {
    this.points.updateGeometry(geometry, data, count, options);
  }

  // =========================================================================
  // Lines Geometry Management
  // =========================================================================

  /**
   * Acquire geometry for Lines (instanced per-segment attributes).
   * `hasScalars` declares whether the commit carries colormap scalar
   * columns — the spec set is decided here, at acquire time (a
   * mismatch releases and reacquires; updateLinesGeometry never
   * rebuilds in place).
   */
  acquireLinesGeometry(
    nodeId: string,
    segmentCount: number,
    hasScalars: boolean
  ): THREE.InstancedBufferGeometry {
    return this.lines.acquireGeometry(nodeId, segmentCount, hasScalars);
  }

  /** Release Lines geometry back to pool. */
  releaseLinesGeometry(nodeId: string): void {
    this.lines.releaseGeometry(nodeId);
  }

  /**
   * Update Lines geometry in place.
   * @param options - `fromInstance`: append fast path (Phase 4 Stage 2) —
   *   write & upload only the `[fromInstance, count)` segment suffix,
   *   preserving the prefix already on the GPU (the commit path decides;
   *   see commit-lines-geometry.ts).
   */
  updateLinesGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: ProcessedLinesData,
    count: number,
    options?: { fromInstance?: number }
  ): void {
    this.lines.updateGeometry(geometry, data, count, options);
  }

  // =========================================================================
  // GSplats Geometry Management
  // =========================================================================

  /** Acquire geometry for GSplats (splat texture + `aSortedIndex`). */
  acquireGSplatsGeometry(nodeId: string, splatCount: number): THREE.InstancedBufferGeometry {
    return this.gsplats.acquireGeometry(nodeId, splatCount);
  }

  /** Release GSplats geometry back to pool. */
  releaseGSplatsGeometry(nodeId: string): void {
    this.gsplats.releaseGeometry(nodeId);
  }

  /**
   * Update GSplats geometry in place.
   * @param truncationRadius - Truncation radius in sigmas (default 3.0).
   *   Must match the material's truncationRadius for correct frustum culling.
   * @param options - `preserveOrdering`: keep the geometry's existing
   *   `aSortedIndex` permutation instead of resetting it to identity
   *   (same-node same-count recommit — the commit path decides; see
   *   commit-gsplats-geometry.ts). `fromInstance`: append fast path (Phase 4
   *   Stage 2) — write & upload only the `[fromInstance, count)` suffix,
   *   preserving the prefix texels + permutation already on the GPU.
   */
  updateGSplatsGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: PackedGSplatsData,
    count: number,
    truncationRadius: number = 3.0,
    options?: { preserveOrdering?: boolean; fromInstance?: number }
  ): void {
    this.gsplats.updateGeometry(geometry, data, count, truncationRadius, options);
  }

  // =========================================================================
  // Pool Management
  // =========================================================================

  /**
   * Get size bucket for capacity-based pooling.
   * Buckets: 1K, 5K, 10K, 50K, 100K, 500K, 1M.
   * @internal — called by adapters; public to satisfy PointsAdapterHost.
   */
  getBucket(count: number): number {
    if (count <= 1000) return 1000;
    if (count <= 5000) return 5000;
    if (count <= 10000) return 10000;
    if (count <= 50000) return 50000;
    if (count <= 100000) return 100000;
    if (count <= 500000) return 500000;
    return 1000000;
  }

  /**
   * Evict unused geometries using LRU policy.
   * Returns number of geometries evicted.
   *
   * Public for testing and manual pool management.
   */
  evictUnused(fromAcquire: boolean = false): number {
    let evicted = 0;
    const currentFrame = this.frameCount;

    // Check total pool size
    const totalPooled =
      Array.from(this.points.pointBuffers.values()).reduce((sum, arr) => sum + arr.length, 0) +
      Array.from(this.lines.lineBuffers.values()).reduce((sum, arr) => sum + arr.length, 0) +
      Array.from(this.gsplats.gsplatBuffers.values()).reduce((sum, arr) => sum + arr.length, 0);

    // If pool is over limit, evict aggressively
    const mustEvict = totalPooled > this.maxPoolSize;

    // Per-call eviction batch cap. Without this, a frame in which many
    // buckets simultaneously cross the eviction threshold (common after
    // a long pause + a viewport change) would burst-dispose every
    // qualifying buffer in a single frame. Each `geometry.dispose()`
    // can take 5–20 ms on slow GPUs; a 50-buffer burst stutters
    // visibly. By capping the per-call batch, the remaining evictable
    // buffers are deferred to the next frame's `acquire*` call. The
    // `mustEvict` over-limit path bypasses the cap so we never let the
    // pool drift unbounded above its limit.
    const batchCap = mustEvict ? Number.POSITIVE_INFINITY : this.evictBatchSize;

    // Helper to evict from a pool, returns count evicted. Stops early
    // when the per-call budget is exhausted. Buffers that WERE
    // eviction-eligible but couldn't run this call (batch cap exhausted)
    // are counted as `deferredEvictions` on the global stats so callers
    // can observe the eviction queue stretching across frames.
    const evictFromPool = (pool: Map<number, PooledBuffer[]>, budget: number): number => {
      let poolEvicted = 0;
      if (budget <= 0) return 0;
      for (const [bucket, buffers] of pool.entries()) {
        const kept: PooledBuffer[] = [];
        const toDispose: PooledBuffer[] = [];

        for (const buffer of buffers) {
          const framesSinceUse = currentFrame - buffer.lastUsedFrame;
          const evictable =
            framesSinceUse > this.evictionFrames || (mustEvict && framesSinceUse > 60);

          // Evict if: unused for >evictionFrames OR pool over limit,
          // AND we're under the per-call batch cap.
          if (evictable && poolEvicted < budget) {
            toDispose.push(buffer);
            poolEvicted++;
          } else {
            if (evictable) this.stats.deferredEvictions++;
            kept.push(buffer);
          }
        }

        // Commit the bucket BEFORE disposing: `geometry.dispose()` fires
        // user-registered dispose listeners synchronously, and a throwing
        // listener aborts this pass (no catch here, by contract — see the
        // pool test suite). Dispose-first would leave disposed zombies
        // adoptable in the free bucket (and the grow-reclaim path could
        // reinstate one); commit-first means a throw merely leaks the
        // not-yet-disposed buffers, which are already unreachable from
        // the pool — the safe direction.
        if (kept.length > 0) {
          pool.set(bucket, kept);
        } else {
          pool.delete(bucket);
        }
        for (const buffer of toDispose) {
          buffer.geometry.dispose();
        }

        if (poolEvicted >= budget) break;
      }
      return poolEvicted;
    };

    const pointsEvicted = evictFromPool(this.points.pointBuffers, batchCap);
    const remaining1 = batchCap === Number.POSITIVE_INFINITY ? batchCap : batchCap - pointsEvicted;
    const linesEvicted = evictFromPool(this.lines.lineBuffers, remaining1);
    const remaining2 = batchCap === Number.POSITIVE_INFINITY ? batchCap : remaining1 - linesEvicted;
    const gsplatsEvicted = evictFromPool(this.gsplats.gsplatBuffers, remaining2);

    evicted = pointsEvicted + linesEvicted + gsplatsEvicted;
    this.stats.evictions += evicted;
    this.typeStats.points.evictions += pointsEvicted;
    this.typeStats.lines.evictions += linesEvicted;
    this.typeStats.gsplats.evictions += gsplatsEvicted;

    // Byte-budget pass. Independent of the count-based budget above.
    // Targets TOTAL resident bytes (active + pooled) against the live
    // budget, disposing pooled buffers (largest-first) so the total stays
    // under budget. Active (in-use) buffers are never disposed here; the
    // LOD registry demotes cold active levels to pooled, where this pass
    // then reclaims them. `0` disables the pass (count-only behavior).
    const budget = this.getByteBudget();
    if (budget > 0) {
      const byteEvicted = this._evictUntilUnderByteBudget(budget, fromAcquire);
      evicted += byteEvicted;
      this.stats.evictions += byteEvicted;
    }

    if (evicted > 0) {
      log.info(Modules.GPU_BUFFER_POOL, `Evicted ${evicted} unused geometries (LRU + byte-budget)`);
    }

    return evicted;
  }

  private _evictUntilUnderByteBudget(budget: number, fromAcquire: boolean = false): number {
    if (budget <= 0) return 0; // disabled — never dispose on bytes
    // The pooled-disposal target is the budget MINUS bytes held by active
    // (in-use) buffers, which cannot be disposed. Pooled buffers are then
    // disposed largest-first until active + pooled <= budget. If active
    // alone already exceeds the budget, the target floors at 0 and every
    // pooled buffer is reclaimed (the only safe action — active stays).
    const pooledTarget = Math.max(0, budget - this.sumActiveBytes());
    const sentinel = { emitted: this.largePoolWarningEmitted };
    const evicted = evictUntilUnderByteBudget(
      {
        pointBuffers: this.points.pointBuffers,
        lineBuffers: this.lines.lineBuffers,
        gsplatBuffers: this.gsplats.gsplatBuffers,
        maxPoolBytes: pooledTarget,
        maxPoolSize: this.maxPoolSize,
        typeEvictionCounters: this.typeStats,
        // Same-frame grace applies ONLY to acquire-triggered sweeps —
        // the release path keeps its original semantics (a release
        // followed by evictUnused() may reclaim that very buffer).
        // graceFrame -1 never matches a real frame counter.
        graceFrame: fromAcquire ? this.frameCount : -1,
      },
      sentinel
    );
    this.largePoolWarningEmitted = sentinel.emitted;
    return evicted;
  }

  /** Sum of bytes held by active (in-use) buffers. Uses cached per-geometry estimates. */
  private sumActiveBytes(): number {
    let total = 0;
    for (const buffer of this.activeBuffers.values()) {
      total += estimateGeometryBytes(buffer.geometry);
    }
    return total;
  }

  /** Sum of bytes held by pooled (released, retained-for-reuse) buffers. */
  private sumPooledBytes(): number {
    let total = 0;
    const add = (pool: Map<number, PooledBuffer[]>): void => {
      for (const arr of pool.values()) {
        for (const b of arr) total += estimateGeometryBytes(b.geometry);
      }
    };
    add(this.points.pointBuffers);
    add(this.lines.lineBuffers);
    add(this.gsplats.gsplatBuffers);
    return total;
  }

  /**
   * Total resident VRAM bytes (active + pooled), using real per-geometry
   * capacities. The single source of truth for budget enforcement: the
   * LOD registry queries it to decide when to demote cold levels, and the
   * pool's own byte-eviction pass keeps it under the live budget. Cheaper
   * than {@link getStats} (no per-type breakdown), so it is safe to call
   * once per frame.
   */
  getResidentBytes(): number {
    return this.sumActiveBytes() + this.sumPooledBytes();
  }

  /**
   * Get pool statistics with per-type breakdown.
   */
  getStats(): PoolStats {
    // Calculate per-type pooled buffers
    const pointsPooled = Array.from(this.points.pointBuffers.values()).reduce(
      (sum, arr) => sum + arr.length,
      0
    );
    const linesPooled = Array.from(this.lines.lineBuffers.values()).reduce(
      (sum, arr) => sum + arr.length,
      0
    );
    const gsplatsPooled = Array.from(this.gsplats.gsplatBuffers.values()).reduce(
      (sum, arr) => sum + arr.length,
      0
    );

    // Calculate per-type active buffers and byte totals.
    let pointsActive = 0;
    let linesActive = 0;
    let gsplatsActive = 0;
    let pointsActiveBytes = 0;
    let linesActiveBytes = 0;
    let gsplatsActiveBytes = 0;
    for (const buffer of this.activeBuffers.values()) {
      const bytes = estimateGeometryBytes(buffer.geometry);
      if (buffer.type === 'points') {
        pointsActive++;
        pointsActiveBytes += bytes;
      } else if (buffer.type === 'lines') {
        linesActive++;
        linesActiveBytes += bytes;
      } else if (buffer.type === 'gsplats') {
        gsplatsActive++;
        gsplatsActiveBytes += bytes;
      }
    }

    // per-type pooled bytes + largest pooled buffer.
    let pointsPooledBytes = 0;
    let linesPooledBytes = 0;
    let gsplatsPooledBytes = 0;
    let largestPooledBytes = 0;
    for (const arr of this.points.pointBuffers.values()) {
      for (const b of arr) {
        const bytes = estimateGeometryBytes(b.geometry);
        pointsPooledBytes += bytes;
        if (bytes > largestPooledBytes) largestPooledBytes = bytes;
      }
    }
    for (const arr of this.lines.lineBuffers.values()) {
      for (const b of arr) {
        const bytes = estimateGeometryBytes(b.geometry);
        linesPooledBytes += bytes;
        if (bytes > largestPooledBytes) largestPooledBytes = bytes;
      }
    }
    for (const arr of this.gsplats.gsplatBuffers.values()) {
      for (const b of arr) {
        const bytes = estimateGeometryBytes(b.geometry);
        gsplatsPooledBytes += bytes;
        if (bytes > largestPooledBytes) largestPooledBytes = bytes;
      }
    }

    const activeBytes = pointsActiveBytes + linesActiveBytes + gsplatsActiveBytes;
    const pooledBytes = pointsPooledBytes + linesPooledBytes + gsplatsPooledBytes;

    return {
      ...this.stats,
      activeBuffers: this.activeBuffers.size,
      pooledBuffers: pointsPooled + linesPooled + gsplatsPooled,
      activeBytes,
      pooledBytes,
      totalBytes: activeBytes + pooledBytes,
      largestPooledBytes,
      byType: {
        points: {
          allocations: this.typeStats.points.allocations,
          reuses: this.typeStats.points.reuses,
          evictions: this.typeStats.points.evictions,
          activeBuffers: pointsActive,
          pooledBuffers: pointsPooled,
          activeBytes: pointsActiveBytes,
          pooledBytes: pointsPooledBytes,
        },
        lines: {
          allocations: this.typeStats.lines.allocations,
          reuses: this.typeStats.lines.reuses,
          evictions: this.typeStats.lines.evictions,
          activeBuffers: linesActive,
          pooledBuffers: linesPooled,
          activeBytes: linesActiveBytes,
          pooledBytes: linesPooledBytes,
        },
        gsplats: {
          allocations: this.typeStats.gsplats.allocations,
          reuses: this.typeStats.gsplats.reuses,
          evictions: this.typeStats.gsplats.evictions,
          activeBuffers: gsplatsActive,
          pooledBuffers: gsplatsPooled,
          activeBytes: gsplatsActiveBytes,
          pooledBytes: gsplatsPooledBytes,
        },
      },
    };
  }

  /**
   * Dispose all pooled geometries (for cleanup or context loss).
   */
  dispose(): void {
    // Dispose all active geometries
    for (const buffer of this.activeBuffers.values()) {
      buffer.geometry.dispose();
    }
    this.activeBuffers.clear();

    // Dispose all pooled geometries
    const disposePool = (pool: Map<number, PooledBuffer[]>) => {
      for (const buffers of pool.values()) {
        for (const buffer of buffers) {
          buffer.geometry.dispose();
        }
      }
      pool.clear();
    };

    disposePool(this.points.pointBuffers);
    disposePool(this.lines.lineBuffers);
    disposePool(this.gsplats.gsplatBuffers);

    log.info(Modules.GPU_BUFFER_POOL, 'All pooled geometries disposed');
  }
}
