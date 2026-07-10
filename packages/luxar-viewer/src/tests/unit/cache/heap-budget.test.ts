import { describe, it, expect } from 'vitest';
import { computeCacheBudgets, readHeapLimitBytes } from '../../../cache/heap-budget';
import { config } from '../../../config';

const MB = 1024 * 1024;
const l0Ceil = config.cache.l0MaxSizeMB * MB;
const l1Ceil = config.cache.l1MaxSizeMB * MB;
const sliceConfig = config.cache.sliceCacheMaxSizeMB * MB;
const target = config.dataLoading.memory.targetHeapUsage;
const SLICE_CAP = 1024 * MB;

describe('readHeapLimitBytes', () => {
  it('returns undefined in the node/jsdom test env (no performance.memory)', () => {
    // The whole point of the fallback: no Chrome heap API here.
    expect(readHeapLimitBytes()).toBeUndefined();
  });
});

describe('computeCacheBudgets', () => {
  it('falls back to fixed config sizes when the heap is unknown', () => {
    const b = computeCacheBudgets(); // no override, no performance.memory → fallback
    expect(b.heapAware).toBe(false);
    expect(b.l0Bytes).toBe(l0Ceil);
    expect(b.l1Bytes).toBe(l1Ceil);
    expect(b.sliceBytes).toBe(sliceConfig);
  });

  it('large heap: L0/L1 stay at their ceilings, S-cache scales up to the cap', () => {
    const heap = 4192 * MB;
    const b = computeCacheBudgets(heap);
    expect(b.heapAware).toBe(true);
    expect(b.l0Bytes).toBe(l0Ceil); // pool is ample → ceilings honored
    expect(b.l1Bytes).toBe(l1Ceil);
    expect(b.sliceBytes).toBe(SLICE_CAP); // residual exceeds cap → clamped
    expect(b.sliceBytes).toBeGreaterThan(sliceConfig); // beats the old fixed 128MB
  });

  it('mid heap: S-cache takes the residual (above the 128MB default, below the cap)', () => {
    const heap = 1024 * MB;
    const b = computeCacheBudgets(heap);
    const pool = heap * target * 0.6;
    expect(b.l0Bytes).toBe(l0Ceil);
    expect(b.l1Bytes).toBe(l1Ceil);
    // residual = pool - 300MB, within (128MB, 1GiB)
    expect(b.sliceBytes).toBe(Math.floor(pool - l0Ceil - l1Ceil));
    expect(b.sliceBytes).toBeGreaterThan(sliceConfig);
    expect(b.sliceBytes).toBeLessThan(SLICE_CAP);
  });

  it('small heap: all tiers scale DOWN and total stays within heap × targetHeapUsage', () => {
    const heap = 512 * MB;
    const b = computeCacheBudgets(heap);
    expect(b.l0Bytes).toBeLessThan(l0Ceil); // chunk caches shrink first
    expect(b.l1Bytes).toBeLessThan(l1Ceil);
    expect(b.sliceBytes).toBeLessThan(sliceConfig); // S-cache below the 128MB default
    // No OOM regime: the three heap tiers together stay under the heap target.
    expect(b.l0Bytes + b.l1Bytes + b.sliceBytes).toBeLessThanOrEqual(Math.ceil(heap * target));
  });

  it('render-proximal priority: chunk caches shrink before the S-cache floor is broken', () => {
    // On a tight heap the S-cache keeps a nonzero heap-relative floor.
    const heap = 512 * MB;
    const b = computeCacheBudgets(heap);
    expect(b.sliceBytes).toBeGreaterThan(0);
  });

  it('huge heap: S-cache is capped at 1 GiB (never an absurd pin)', () => {
    const b = computeCacheBudgets(64 * 1024 * MB);
    expect(b.sliceBytes).toBe(SLICE_CAP);
  });
});
