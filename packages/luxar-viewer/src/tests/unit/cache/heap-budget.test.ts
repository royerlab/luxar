import { describe, it, expect } from 'vitest';
import {
  computeCacheBudgets,
  computeWorkingSetBudgetBytes,
  readHeapLimitBytes,
  inferDeviceClass,
} from '../../../cache/heap-budget';
import { config } from '../../../config';

const MB = 1024 * 1024;
const l0Ceil = config.cache.l0MaxSizeMB * MB;
const l1Ceil = config.cache.l1MaxSizeMB * MB;
const sliceConfig = config.cache.sliceCacheMaxSizeMB * MB;
const target = config.dataLoading.memory.targetHeapUsage;
const SLICE_CAP = 2048 * MB;
/** Expected S-cache bytes for a pool that seats both chunk-cache ceilings. */
const sliceFor = (poolBytes: number) =>
  Math.floor(Math.min(SLICE_CAP, poolBytes - l0Ceil - l1Ceil));

describe('readHeapLimitBytes', () => {
  it('returns undefined in the node/jsdom test env (no performance.memory)', () => {
    // The whole point of the fallback: no Chrome heap API here.
    expect(readHeapLimitBytes()).toBeUndefined();
  });
});

describe('computeWorkingSetBudgetBytes', () => {
  it('uses half of the non-cache share of the configured heap target', () => {
    const heap = 512 * MB;
    expect(computeWorkingSetBudgetBytes(heap)).toBe(Math.floor(heap * target * 0.4 * 0.5));
  });

  it('caps the eager working set on a large measured heap', () => {
    expect(computeWorkingSetBudgetBytes(8 * 1024 * MB)).toBe(512 * MB);
  });

  it('uses an explicit cache-pool override before a measured heap', () => {
    expect(computeWorkingSetBudgetBytes(512 * MB, 768 * MB)).toBe(256 * MB);
  });

  it('derives a WebKit budget from the device-class cache pool when no override exists', () => {
    expect(computeWorkingSetBudgetBytes(undefined, undefined, 384 * MB)).toBe(128 * MB);
  });

  it('uses a measured heap before the device-class cache pool', () => {
    expect(computeWorkingSetBudgetBytes(512 * MB, undefined, 2048 * MB)).toBe(
      Math.floor(512 * MB * target * 0.4 * 0.5)
    );
  });

  it('uses the fixed fallback when no pool or heap signal is available', () => {
    expect(computeWorkingSetBudgetBytes()).toBe(256 * MB);
    expect(computeWorkingSetBudgetBytes(0)).toBe(256 * MB);
    expect(computeWorkingSetBudgetBytes(Number.NaN)).toBe(256 * MB);
  });
});

describe('computeCacheBudgets', () => {
  it('falls back to fixed config sizes when the heap is unknown', () => {
    const b = computeCacheBudgets(); // no override, no performance.memory → fallback
    expect(b.heapAware).toBe(false);
    expect(b.source).toBe('fixed');
    expect(b.l0Bytes).toBe(l0Ceil);
    expect(b.l1Bytes).toBe(l1Ceil);
    expect(b.sliceBytes).toBe(sliceConfig);
  });

  it('explicit pool override (WKWebView/Safari path) splits the pool and reports source=explicit', () => {
    // The native launcher / ?cacheBudgetMB= supplies a pool where the heap is
    // unmeasurable. Precedence over any heap arg. Pool 1536MB → L0/L1 ceilings,
    // S-cache the residual (capped at 1 GiB).
    const b = computeCacheBudgets(undefined, 1536 * MB);
    expect(b.source).toBe('explicit');
    expect(b.heapAware).toBe(true);
    expect(b.l0Bytes).toBe(l0Ceil);
    expect(b.l1Bytes).toBe(l1Ceil);
    expect(b.sliceBytes).toBe(sliceFor(1536 * MB)); // residual 1236MB, under the 2GiB cap
    // Override wins even if a (smaller) heap is also passed.
    const b2 = computeCacheBudgets(256 * MB, 1536 * MB);
    expect(b2.source).toBe('explicit');
    expect(b2.sliceBytes).toBe(sliceFor(1536 * MB));
  });

  it('large heap: L0/L1 stay at their ceilings, S-cache scales up on the residual', () => {
    const heap = 4192 * MB;
    const b = computeCacheBudgets(heap);
    const pool = heap * target * 0.6;
    expect(b.heapAware).toBe(true);
    expect(b.source).toBe('heap');
    expect(b.l0Bytes).toBe(l0Ceil); // pool is ample → ceilings honored
    expect(b.l1Bytes).toBe(l1Ceil);
    expect(b.sliceBytes).toBe(sliceFor(pool)); // residual (~1.7GB), under the 2GiB cap
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

  it('huge heap: S-cache is capped (never an absurd pin)', () => {
    const b = computeCacheBudgets(64 * 1024 * MB);
    expect(b.sliceBytes).toBe(SLICE_CAP);
  });

  it('disabled S-cache: reserves no sliceMin, so chunk caches keep more of a tight pool', () => {
    const heap = 512 * MB; // tight: with S-cache on, L0/L1 shrink to protect sliceMin
    const withSlice = computeCacheBudgets(heap);
    const noSlice = computeCacheBudgets(heap, undefined, undefined, { slice: false });
    expect(noSlice.sliceBytes).toBe(0); // disabled → 0, not just unconstructed
    // The freed sliceMin lets L0/L1 be at least as large as when S-cache is on.
    expect(noSlice.l0Bytes).toBeGreaterThan(withSlice.l0Bytes);
    expect(noSlice.l1Bytes).toBeGreaterThan(withSlice.l1Bytes);
  });

  it('disabled L0: its ceiling flows to the S-cache residual', () => {
    const heap = 4192 * MB;
    const withL0 = computeCacheBudgets(heap);
    const noL0 = computeCacheBudgets(heap, undefined, undefined, { l0: false });
    expect(noL0.l0Bytes).toBe(0);
    expect(noL0.sliceBytes).toBeGreaterThan(withL0.sliceBytes); // gained L0's ~200MB
  });

  it('device-class fallback: used when no override and no heap; source=device-class', () => {
    // cache-setup passes deviceClassPoolBytes() here for WebKit without an
    // override. A 1 GiB laptop pool → ceilings + residual S-cache.
    const b = computeCacheBudgets(undefined, undefined, 1024 * MB);
    expect(b.source).toBe('device-class');
    expect(b.heapAware).toBe(true);
    expect(b.l0Bytes).toBe(l0Ceil);
    expect(b.l1Bytes).toBe(l1Ceil);
    expect(b.sliceBytes).toBe(sliceFor(1024 * MB));
    // A measured heap still wins over the device-class fallback.
    expect(computeCacheBudgets(4192 * MB, undefined, 1024 * MB).source).toBe('heap');
    // An explicit override still wins over both.
    expect(computeCacheBudgets(undefined, 512 * MB, 1024 * MB).source).toBe('explicit');
  });
});

describe('inferDeviceClass (core-count proxy — best-effort)', () => {
  const desktop = {
    userAgent: 'Mozilla/5.0 (Macintosh)',
    maxTouchPoints: 0,
    coarsePointer: false,
    cores: 16,
  };

  it('classifies a mobile UA as mobile', () => {
    expect(inferDeviceClass({ ...desktop, userAgent: 'iPhone', cores: 6 })).toBe('mobile');
    expect(inferDeviceClass({ ...desktop, userAgent: 'Android Mobile', cores: 8 })).toBe('mobile');
  });

  it('classifies a touch + coarse-pointer device as mobile (catches iPadOS masquerading as Mac)', () => {
    expect(
      inferDeviceClass({ userAgent: 'Macintosh', maxTouchPoints: 5, coarsePointer: true, cores: 8 })
    ).toBe('mobile');
  });

  it('classifies a high-core non-touch machine as desktop, a lower-core one as laptop', () => {
    expect(inferDeviceClass({ ...desktop, cores: 16 })).toBe('desktop');
    expect(inferDeviceClass({ ...desktop, cores: 12 })).toBe('desktop'); // threshold is inclusive
    expect(inferDeviceClass({ ...desktop, cores: 8 })).toBe('laptop');
    expect(inferDeviceClass({ ...desktop, cores: 0 })).toBe('laptop'); // unknown cores → laptop
  });
});
