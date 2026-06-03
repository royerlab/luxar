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
        ref.buffer.geometry.dispose();
        arr.splice(ref.index, 1);
        ctx.typeEvictionCounters[ref.buffer.type].evictions++;
      }
      if (arr.length === 0) pool.delete(bucket);
    }
  }
  return evicted.length;
}
