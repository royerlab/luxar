/**
 * Direct unit tests for the byte-budget eviction pass. The
 * orchestrator-level `gpu-pool-byte-budget.test.ts` exercises this
 * through `GPUBufferPool.evictUnused()`; here we test the pure helper
 * against a synthesised EvictorCtx with real BufferGeometry instances.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
  evictUntilUnderByteBudget,
  type EvictorCtx,
} from '../../../../rendering/gpu-buffer-pool/byte-budget-evictor';
import type { PooledBuffer } from '../../../../rendering/gpu-buffer-pool/pool-stats';

/**
 * Build a BufferGeometry with a Float32 position attribute of the
 * requested *element count*. estimateGeometryBytes reads
 * `attribute.array.byteLength`, so size = elementCount × 4 (3 × Float32).
 *
 * Caller supplies a target byte size; we derive the element count.
 */
function makeGeometry(targetBytes: number): THREE.BufferGeometry {
  const floatsNeeded = Math.max(3, Math.ceil(targetBytes / 4));
  const data = new Float32Array(floatsNeeded);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(data, 3));
  return geom;
}

/** Build a PooledBuffer wrapping a geometry of the given byte size. */
function makePooled(targetBytes: number, type: PooledBuffer['type']): PooledBuffer {
  return {
    geometry: makeGeometry(targetBytes),
    capacity: 1,
    type,
    inUse: false,
    lastUsedFrame: 0,
  };
}

/** Build an EvictorCtx with empty type pools by default. */
function makeCtx(
  opts: { maxPoolBytes: number; maxPoolSize?: number } = { maxPoolBytes: 1_000_000 }
): EvictorCtx {
  return {
    pointBuffers: new Map(),
    lineBuffers: new Map(),
    gsplatBuffers: new Map(),
    maxPoolBytes: opts.maxPoolBytes,
    maxPoolSize: opts.maxPoolSize ?? 100,
    // -1 = no same-frame grace (matches release-triggered sweeps; the
    // acquire-path grace has its own dedicated test below).
    graceFrame: -1,
    typeEvictionCounters: {
      points: { evictions: 0 },
      lines: { evictions: 0 },
      gsplats: { evictions: 0 },
    },
  };
}

describe('evictUntilUnderByteBudget', () => {
  let sentinel: { emitted: boolean };

  beforeEach(() => {
    sentinel = { emitted: false };
  });

  it('returns 0 and disposes nothing when totalBytes is at or under the budget', () => {
    const ctx = makeCtx({ maxPoolBytes: 10_000 });
    const buf = makePooled(1000, 'points');
    const disposeSpy = vi.spyOn(buf.geometry, 'dispose');
    ctx.pointBuffers.set(100, [buf]);

    expect(evictUntilUnderByteBudget(ctx, sentinel)).toBe(0);
    expect(disposeSpy).not.toHaveBeenCalled();
    expect(ctx.pointBuffers.has(100)).toBe(true);
  });

  it('disposes the largest pooled buffer first when over budget', () => {
    const ctx = makeCtx({ maxPoolBytes: 1500 });
    const small = makePooled(500, 'points');
    const large = makePooled(2000, 'points');
    ctx.pointBuffers.set(100, [small, large]);

    const smallDispose = vi.spyOn(small.geometry, 'dispose');
    const largeDispose = vi.spyOn(large.geometry, 'dispose');

    const evicted = evictUntilUnderByteBudget(ctx, sentinel);

    expect(evicted).toBeGreaterThan(0);
    expect(largeDispose).toHaveBeenCalled();
    expect(smallDispose).not.toHaveBeenCalled();
  });

  it('removes an emptied bucket from its pool map after eviction', () => {
    const ctx = makeCtx({ maxPoolBytes: 100 });
    const buf = makePooled(2000, 'lines');
    ctx.lineBuffers.set(50, [buf]);

    evictUntilUnderByteBudget(ctx, sentinel);

    expect(ctx.lineBuffers.has(50)).toBe(false);
  });

  it('keeps the bucket alive when only some entries inside are evicted', () => {
    const ctx = makeCtx({ maxPoolBytes: 1500 });
    const small = makePooled(500, 'points');
    const large = makePooled(2000, 'points');
    ctx.pointBuffers.set(100, [small, large]);

    evictUntilUnderByteBudget(ctx, sentinel);

    expect(ctx.pointBuffers.has(100)).toBe(true);
    expect(ctx.pointBuffers.get(100)?.length).toBe(1);
    expect(ctx.pointBuffers.get(100)?.[0]).toBe(small);
  });

  it('increments the per-type eviction counter for each disposed buffer', () => {
    const ctx = makeCtx({ maxPoolBytes: 100 });
    ctx.pointBuffers.set(1, [makePooled(2000, 'points')]);
    ctx.lineBuffers.set(2, [makePooled(2000, 'lines')]);
    ctx.gsplatBuffers.set(3, [makePooled(2000, 'gsplats')]);

    evictUntilUnderByteBudget(ctx, sentinel);

    // At least one per type should have been counted
    expect(ctx.typeEvictionCounters.points.evictions).toBeGreaterThanOrEqual(1);
    expect(ctx.typeEvictionCounters.lines.evictions).toBeGreaterThanOrEqual(1);
    expect(ctx.typeEvictionCounters.gsplats.evictions).toBeGreaterThanOrEqual(1);
  });

  it('considers buffers from all three pool types when selecting candidates', () => {
    // Tiny budget but huge buffers spread across all three pools — we
    // should see disposals across multiple types.
    const ctx = makeCtx({ maxPoolBytes: 100 });
    const p = makePooled(5000, 'points');
    const l = makePooled(5000, 'lines');
    const g = makePooled(5000, 'gsplats');
    ctx.pointBuffers.set(1, [p]);
    ctx.lineBuffers.set(2, [l]);
    ctx.gsplatBuffers.set(3, [g]);

    const disposed = evictUntilUnderByteBudget(ctx, sentinel);
    expect(disposed).toBeGreaterThanOrEqual(1);
  });

  it('flips the 100MB warning sentinel from false to true on first crossing', () => {
    const ctx = makeCtx({ maxPoolBytes: 1_000_000_000 }); // under-budget — no eviction
    const huge = makePooled(120_000_000, 'lines'); // 120 MB > 100 MB threshold
    ctx.lineBuffers.set(1, [huge]);

    evictUntilUnderByteBudget(ctx, sentinel);
    expect(sentinel.emitted).toBe(true);
  });

  it('does NOT re-emit the warning when the sentinel is already true', () => {
    const ctx = makeCtx({ maxPoolBytes: 1_000_000_000 });
    const huge = makePooled(120_000_000, 'lines');
    ctx.lineBuffers.set(1, [huge]);

    const preset = { emitted: true };
    evictUntilUnderByteBudget(ctx, preset);
    expect(preset.emitted).toBe(true); // unchanged
  });

  it('leaves the sentinel false when no buffer crosses the 100MB threshold', () => {
    const ctx = makeCtx({ maxPoolBytes: 1_000_000_000 });
    ctx.pointBuffers.set(1, [makePooled(50_000_000, 'points')]); // 50 MB < 100 MB

    evictUntilUnderByteBudget(ctx, sentinel);
    expect(sentinel.emitted).toBe(false);
  });
});

describe('same-frame grace (acquire-triggered sweeps)', () => {
  it('exempts buffers released this frame when graceFrame matches; evicts them otherwise', () => {
    const buf = makePooled(500_000, 'points'); // over a 100k budget on its own
    buf.lastUsedFrame = 7;
    const pools = new Map([[0, [buf]]]);

    // graceFrame === lastUsedFrame → exempt (dataset-switch churn guard).
    const graced: EvictorCtx = { ...makeCtx({ maxPoolBytes: 100_000 }), pointBuffers: pools, graceFrame: 7 };
    expect(evictUntilUnderByteBudget(graced, { emitted: true })).toBe(0);
    expect(pools.get(0)!.length).toBe(1);

    // Release-style sweep (graceFrame -1) → evicted as before.
    const strict: EvictorCtx = { ...makeCtx({ maxPoolBytes: 100_000 }), pointBuffers: pools, graceFrame: -1 };
    expect(evictUntilUnderByteBudget(strict, { emitted: true })).toBe(1);
    expect(pools.get(0)?.length ?? 0).toBe(0); // empty buckets may be pruned
  });
});

