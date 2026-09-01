/**
 * Byte-budget evictor for GPUBufferPool.
 *
 * Disposes pooled buffers (largest-first across all three type pools)
 * until `pooledBytes <= ctx.maxPoolBytes`. Does NOT touch active buffers
 * (they are in use and cannot be disposed).
 *
 * NOTE: `ctx.maxPoolBytes` is the *pooled-disposal target*, which the
 * caller sets to `liveBudget - activeBytes` so that TOTAL resident bytes
 * (active + pooled) stay under the single VRAM budget. When active bytes
 * already meet/exceed the budget the caller passes 0 here, so every pooled
 * buffer is reclaimed.
 *
 * @module rendering/gpu-buffer-pool/byte-budget-evictor
 */

import { estimateGeometryBytes } from './geometry-bytes';
import type { PooledBuffer } from './pool-stats';
import { selectBuffersToEvict } from './eviction-policy';
import { log, Modules } from '../../utils/log';

/** Pool reference + index inside its bucket array. */
interface BufferRef {
  pool: Map<number, PooledBuffer[]>;
  bucket: number;
  index: number;
  buffer: PooledBuffer;
  bytes: number;
}

/** Inputs the byte-budget pass needs to operate on. */
export interface EvictorCtx {
  readonly pointBuffers: Map<number, PooledBuffer[]>;
  readonly lineBuffers: Map<number, PooledBuffer[]>;
  readonly gsplatBuffers: Map<number, PooledBuffer[]>;
  readonly maxPoolBytes: number;
  readonly maxPoolSize: number;
  /**
   * Same-frame grace for ACQUIRE-triggered sweeps: buffers whose
   * `lastUsedFrame` equals this value are exempt from the byte pass.
   * Without it, a release followed by same-frame acquires would dispose
   * the just-released buffers before those acquires can best-fit them —
   * alloc/dispose churn replacing free reuse. `releaseGeometry` stamps
   * `lastUsedFrame` with the release frame so the grace actually matches
   * (an acquire-time stamp alone would carry a stale frame into the
   * release). Release-triggered sweeps pass -1 (never matches) — byte
   * enforcement on release is unconditional, which is also the backstop
   * bounding the grace: if the frame counter is not advancing (no render
   * loop), acquire sweeps may keep sparing released buffers, but every
   * release re-enforces the budget without grace.
   *
   * THIS GRACE IS VESTIGIAL AS FAR AS ANYONE HAS BEEN ABLE TO MEASURE.
   * Both candidate justifications were checked and neither survives
   * (#2426, #2436):
   *
   *  - DATASET SWITCH, which this comment used to name outright. Cannot
   *    be it: the pool is disposed and reconstructed per dataset
   *    (`lifecycle/dispose.ts` → `gpuBufferPool.dispose()`;
   *    `scene-loader.ts` nulls the field and rebuilds it on the next
   *    load), so no buffer survives a switch to be adopted after one.
   *    The comment asserted a scenario the lifecycle prevents, and it
   *    misdirected a review before anyone checked.
   *  - LOD DEMOTION, the plausible replacement: `releaseLazyGSplats` /
   *    `releaseLazyPoints` / `releaseLazyLines` return a level's buffer
   *    expecting re-promotion to adopt it back. MEASURED under a binding
   *    budget: re-promotion allocates fresh either way (alloc +1,
   *    reuses +0, with and without the post-grow sweep). The demoted
   *    buffer is already gone — `releaseGeometry`'s own sweep runs at
   *    graceFrame -1 and takes it at demotion time — so the grace never
   *    gets the chance to protect it.
   *
   * KEPT ANYWAY, DELIBERATELY. "Protects no case we could construct" is
   * not "protects nothing": some unmeasured same-frame release-then-
   * reacquire may still rely on it, and proving that negative is its own
   * piece of work. Removal is a separate, optional cleanup — do not
   * delete this on the strength of the paragraph above.
   */
  readonly graceFrame: number;
  /** Per-type eviction counters; mutated as buffers dispose. */
  readonly typeEvictionCounters: {
    points: { evictions: number };
    lines: { evictions: number };
    gsplats: { evictions: number };
  };
}

/**
 * Run a byte-budget eviction pass over `ctx`. Returns the count of
 * disposed buffers. `largeWarningState` is a 1-element mutable
 * sentinel so the caller can flip it to `true` after the first warning.
 */
export function evictUntilUnderByteBudget(
  ctx: EvictorCtx,
  largeWarningState: { emitted: boolean }
): number {
  const refs: BufferRef[] = [];
  const collect = (pool: Map<number, PooledBuffer[]>): void => {
    for (const [bucket, arr] of pool.entries()) {
      for (let i = 0; i < arr.length; i++) {
        const buffer = arr[i];
        // Same-frame grace — see EvictorCtx.graceFrame.
        if (buffer.lastUsedFrame === ctx.graceFrame) continue;
        refs.push({
          pool,
          bucket,
          index: i,
          buffer,
          bytes: estimateGeometryBytes(buffer.geometry),
        });
      }
    }
  };
  collect(ctx.pointBuffers);
  collect(ctx.lineBuffers);
  collect(ctx.gsplatBuffers);

  // One-shot warning when a pooled buffer crosses 100 MB. Bounded to
  // emit at most once per pool instance.
  if (!largeWarningState.emitted) {
    const LARGE_POOLED_BYTES_THRESHOLD = 100_000_000;
    const largest = refs.reduce((max, r) => (r.bytes > max ? r.bytes : max), 0);
    if (largest > LARGE_POOLED_BYTES_THRESHOLD) {
      largeWarningState.emitted = true;
      log.warning(
        Modules.GPU_BUFFER_POOL,
        `Pooled buffer of ${(largest / 1024 / 1024).toFixed(1)} MB exceeds the ` +
          '100 MB diagnostic threshold. Eviction of this buffer will pause a frame ' +
          '(geometry.dispose can take 5-20 ms on slow GPUs).'
      );
    }
  }

  const totalBytes = refs.reduce((sum, r) => sum + r.bytes, 0);
  if (totalBytes <= ctx.maxPoolBytes) return 0;

  const targets = selectBuffersToEvict(refs, ctx.maxPoolBytes, totalBytes);

  // Bound the eviction count even though selectBuffersToEvict is
  // finite — defensive guard against malformed ref shapes.
  const maxIterations = Math.max(ctx.maxPoolSize * 3, 16);
  const evictCount = Math.min(targets.length, maxIterations);
  if (targets.length > maxIterations) {
    log.warning(
      Modules.GPU_BUFFER_POOL,
      `Byte-budget eviction capped at ${maxIterations} of ${targets.length} ` +
        'selected buffers. Pool may still be over budget after this pass.'
    );
  }

  // Splice from the lowest-index entries first within each (pool,bucket)
  // so later splices don't invalidate earlier indices.
  const evicted = targets.slice(0, evictCount);
  const grouped = new Map<Map<number, PooledBuffer[]>, Map<number, BufferRef[]>>();
  for (const ref of evicted) {
    let byBucket = grouped.get(ref.pool);
    if (!byBucket) {
      byBucket = new Map();
      grouped.set(ref.pool, byBucket);
    }
    let bucketRefs = byBucket.get(ref.bucket);
    if (!bucketRefs) {
      bucketRefs = [];
      byBucket.set(ref.bucket, bucketRefs);
    }
    bucketRefs.push(ref);
  }
  for (const [pool, byBucket] of grouped) {
    for (const [bucket, bucketRefs] of byBucket) {
      bucketRefs.sort((a, b) => b.index - a.index);
      const arr = pool.get(bucket);
      if (!arr) continue;
      for (const ref of bucketRefs) {
        // Remove from the bucket BEFORE disposing (and count the
        // eviction with the removal): the pool must never hold a
        // disposed-but-adoptable buffer. `geometry.dispose()` fires
        // user-registered dispose listeners synchronously — if one
        // throws, the pass aborts (no catch here, by contract; see the
        // evictor test suite), and dispose-first would leave the zombie
        // in the free bucket where a later adopt/grow-reclaim could
        // reinstate it. Splice-first means a throwing dispose merely
        // leaks this one already-unreachable buffer — the safe
        // direction.
        arr.splice(ref.index, 1);
        ctx.typeEvictionCounters[ref.buffer.type].evictions++;
        ref.buffer.geometry.dispose();
      }
      if (arr.length === 0) pool.delete(bucket);
    }
  }
  return evicted.length;
}
