/**
 * Tests for the per-call eviction-batch cap added to
 * `GPUBufferPool.evictUnused()`.
 *
 * Without the cap, a frame in which many pooled buffers cross the
 * eviction threshold simultaneously disposes every qualifying buffer
 * synchronously — each `geometry.dispose()` is 5–20 ms on slow GPUs,
 * so the burst stutters visibly. The cap defers the excess to the
 * next eviction sweep.
 *
 * The pool's `mustEvict` (over-limit) path bypasses the cap so memory
 * still stays bounded.
 */

import { describe, it, expect } from 'vitest';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';

/**
 * Push N distinct buffers into the pool's free list.
 *
 * Acquires all N geometries first, then releases them. If we
 * released eagerly, each release would put a buffer back in the pool
 * which the next acquire would reuse — net effect of 1 pooled
 * buffer rather than N. By acquiring N up front we guarantee N
 * concurrent active buffers, each landing in the pool on release.
 *
 * Note: every count goes into the same bucket (1000) per the pool's
 * size-bucket policy, so the pool ends up with N entries in bucket
 * 1000 — exactly what the eviction tests need.
 */
function fillPool(pool: GPUBufferPool, count: number): void {
  for (let i = 0; i < count; i++) {
    pool.acquirePointsGeometry(`/n${i}`, 100 + i * 10);
  }
  for (let i = 0; i < count; i++) {
    pool.releasePointsGeometry(`/n${i}`);
  }
}

describe('GPUBufferPool eviction batch cap', () => {
  it('caps disposals per call at evictBatchSize when not over the hard limit', () => {
    // Big maxPoolSize so we never enter the "must evict" path.
    const pool = new GPUBufferPool(
      /* maxPoolSize */ 100,
      /* evictionFrames */ 10,
      /* evictBatchSize */ 3
    );

    // Acquire 8 distinct geometries, then release them so they sit in
    // the pool's free list.
    fillPool(pool, 8);

    // Advance past the eviction threshold so every pooled buffer is
    // eligible.
    for (let frame = 0; frame < 50; frame++) {
      pool.beginFrame();
    }

    // First sweep: caps at evictBatchSize=3.
    const firstEvicted = pool.evictUnused();
    expect(firstEvicted).toBe(3);

    // Second sweep: 3 more.
    const secondEvicted = pool.evictUnused();
    expect(secondEvicted).toBe(3);

    // Third sweep: 2 left → 2 evicted.
    const thirdEvicted = pool.evictUnused();
    expect(thirdEvicted).toBe(2);

    // Fourth sweep: nothing left.
    expect(pool.evictUnused()).toBe(0);
  });

  it('over-limit path bypasses the cap so the pool never grows unbounded', () => {
    // Tiny maxPoolSize forces `mustEvict = true` immediately.
    const pool = new GPUBufferPool(
      /* maxPoolSize */ 2,
      /* evictionFrames */ 10,
      /* evictBatchSize */ 1
    );

    // Acquire 8 geometries; release each so they accumulate in the pool.
    fillPool(pool, 8);

    // Advance past 60 frames (the over-limit-but-recent threshold) so
    // mustEvict + framesSinceUse > 60 fires for every buffer.
    for (let frame = 0; frame < 75; frame++) {
      pool.beginFrame();
    }

    // Single sweep evicts everything qualifying — bypasses the cap of 1.
    const evicted = pool.evictUnused();
    expect(evicted).toBeGreaterThan(1);
  });

  it('clamps evictBatchSize to ≥ 1 even if 0 / negative passed', () => {
    const pool = new GPUBufferPool(
      /* maxPoolSize */ 100,
      /* evictionFrames */ 5,
      /* evictBatchSize */ 0
    );
    fillPool(pool, 3);
    for (let frame = 0; frame < 20; frame++) pool.beginFrame();
    expect(pool.evictUnused()).toBe(1);
  });

  it('default constructor uses evictBatchSize=5', () => {
    // 6 unused entries → first sweep evicts 5, second sweep 1.
    const pool = new GPUBufferPool(/* maxPoolSize */ 100, /* evictionFrames */ 5);
    fillPool(pool, 6);
    for (let frame = 0; frame < 20; frame++) pool.beginFrame();
    expect(pool.evictUnused()).toBe(5);
    expect(pool.evictUnused()).toBe(1);
  });
});
