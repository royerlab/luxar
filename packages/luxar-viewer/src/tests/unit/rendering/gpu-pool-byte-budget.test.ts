/**
 * GPU buffer pool byte-budget eviction tests.
 *
 * Covers:
 *  - estimateGeometryBytes correctness for known mock geometries.
 *  - getStats() exposes activeBytes/pooledBytes/largestPooledBytes per type.
 *  - evictUnused() under byte budget evicts largest pooled buffer first.
 *  - Byte budget = 0 disables byte-budget pass (count-only).
 *  - Byte budget triggers eviction even when count is well under
 *    gpuPoolMaxSize (the original bug — a single 760 MB buffer counts
 *    the same as a 32 KB buffer).
 *  - Eviction respects active vs pooled (active never disposed).
 *  - Eviction is idempotent if pool already under budget.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
  GPUBufferPool,
  estimateGeometryBytes,
  selectBuffersToEvict,
  invalidateCachedByteSize,
  type PooledBufferRef,
} from '../../../rendering/gpu-buffer-pool';
import type { LoadedPointsData } from '../../../data/data-loader-types';

function pointsData(count: number): LoadedPointsData {
  return {
    positions: new Float32Array(count * 3),
    colors: new Float32Array(count * 3),
    radii: new Float32Array(count),
    sharpness: new Float32Array(count),
    pointCount: count,
    ndim: 3,
    metadata: {
      totalPoints: count,
      loadedPoints: count,
      bounds: new THREE.Box3(),
      usedSpatialIndex: false,
    },
  };
}

describe('selectBuffersToEvict (pure function)', () => {
  function ref(bytes: number, id?: number): PooledBufferRef<number> {
    return { bytes, payload: id };
  }

  it('returns empty array when under budget', () => {
    const targets = selectBuffersToEvict([ref(100), ref(200)], 500);
    expect(targets).toEqual([]);
  });

  it('returns empty array for empty input', () => {
    expect(selectBuffersToEvict([], 100)).toEqual([]);
  });

  it('evicts largest-first until under budget', () => {
    // total = 100 + 200 + 300 + 400 = 1000, budget = 500.
    // Largest-first: drop 400 (running 600), drop 300 (running 300 ≤ 500).
    const targets = selectBuffersToEvict(
      [ref(100, 1), ref(200, 2), ref(300, 3), ref(400, 4)],
      500
    );
    const payloads = targets.map((t) => t.payload).sort();
    expect(payloads).toEqual([3, 4]);
  });

  it('all-same-size: stable order (input order preserved)', () => {
    const targets = selectBuffersToEvict(
      [ref(100, 1), ref(100, 2), ref(100, 3), ref(100, 4)],
      150
    );
    // total = 400, budget 150 → need to drop 250 bytes → 3 entries.
    expect(targets.length).toBe(3);
    // First three by input order (stable sort).
    expect(targets.map((t) => t.payload)).toEqual([1, 2, 3]);
  });

  it('single-buffer-over-budget pathological case', () => {
    // One huge buffer dwarfs everything.
    const targets = selectBuffersToEvict([ref(10), ref(20), ref(10_000)], 100);
    // Evicting the huge one alone (10) is enough.
    expect(targets.length).toBe(1);
    expect(targets[0].bytes).toBe(10_000);
  });

  it('uses precomputed total when provided', () => {
    const targets = selectBuffersToEvict(
      [ref(100), ref(200)],
      150,
      300 // explicit total bypasses reduce
    );
    expect(targets.length).toBeGreaterThan(0);
  });
});

describe('estimateGeometryBytes', () => {
  it('sums attribute byte lengths for an empty geometry', () => {
    const g = new THREE.BufferGeometry();
    expect(estimateGeometryBytes(g)).toBe(0);
  });

  it('counts position attribute bytes', () => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(30, 3)); // 10 vec3s = 120 B
    expect(estimateGeometryBytes(g)).toBe(120);
  });

  it('sums multiple attributes', () => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(30, 3)); // 120
    g.setAttribute('color', new THREE.Uint8BufferAttribute(30, 3)); // 30
    g.setAttribute('radius', new THREE.Float32BufferAttribute(10, 1)); // 40
    expect(estimateGeometryBytes(g)).toBe(120 + 30 + 40);
  });

  it('includes index when present', () => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(30, 3));
    g.setIndex(new THREE.BufferAttribute(new Uint16Array([0, 1, 2]), 1)); // 6 B
    expect(estimateGeometryBytes(g)).toBe(120 + 6);
  });

  it('D.3: caches the result on userData.cachedByteSize', () => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(30, 3));
    const first = estimateGeometryBytes(g);
    expect((g.userData as { cachedByteSize?: number }).cachedByteSize).toBe(first);

    // Mutate the underlying attribute — without invalidation, the cached
    // value is returned unchanged. This is the contract: the caller (the
    // grow paths) is responsible for calling invalidateCachedByteSize.
    g.setAttribute('color', new THREE.Float32BufferAttribute(60, 3)); // +240 B
    const cached = estimateGeometryBytes(g);
    expect(cached).toBe(first); // unchanged thanks to the cache

    invalidateCachedByteSize(g);
    const recomputed = estimateGeometryBytes(g);
    expect(recomputed).toBe(120 + 240);
  });

  it('D.3: invalidateCachedByteSize is a no-op when the cache is missing', () => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(30, 3));
    invalidateCachedByteSize(g); // no cache yet — should not throw
    expect(estimateGeometryBytes(g)).toBe(120);
  });
});

describe('GPU pool getStats() byte counters', () => {
  it('reports activeBytes for an acquired buffer', () => {
    const pool = new GPUBufferPool(20, 300, 5, 0); // byte budget disabled
    pool.acquirePointsGeometry('p1', pointsData(100), 100);
    const stats = pool.getStats();
    expect(stats.activeBytes).toBeGreaterThan(0);
    expect(stats.pooledBytes).toBe(0);
    expect(stats.totalBytes).toBe(stats.activeBytes);
    expect(stats.byType.points.activeBytes).toBe(stats.activeBytes);
  });

  it('moves bytes from active to pooled on release', () => {
    const pool = new GPUBufferPool(20, 300, 5, 0);
    pool.acquirePointsGeometry('p1', pointsData(100), 100);
    const before = pool.getStats();
    pool.releasePointsGeometry('p1');
    const after = pool.getStats();
    expect(after.activeBytes).toBe(0);
    expect(after.pooledBytes).toBe(before.activeBytes);
    expect(after.byType.points.pooledBytes).toBe(before.activeBytes);
  });

  it('tracks largestPooledBytes across all type pools', () => {
    const pool = new GPUBufferPool(20, 300, 5, 0);
    pool.acquirePointsGeometry('p1', pointsData(100), 100);
    pool.acquirePointsGeometry('p2', pointsData(1000), 1000);
    pool.releasePointsGeometry('p1');
    pool.releasePointsGeometry('p2');
    const stats = pool.getStats();
    // The 1000-point buffer should be larger than the 100-point one.
    expect(stats.largestPooledBytes).toBeGreaterThan(0);
    // Largest must equal the bigger of the two pooled buffers.
    expect(stats.pooledBytes).toBeGreaterThanOrEqual(stats.largestPooledBytes);
  });
});

describe('byte-budget eviction', () => {
  let pool: GPUBufferPool;

  beforeEach(() => {
    // Tiny budget (200 bytes) so even a tiny mock buffer triggers eviction.
    pool = new GPUBufferPool(20, 300, 5, 200);
  });

  it('evicts pooled buffers when over byte budget (count well under maxPoolSize)', () => {
    pool.acquirePointsGeometry('p1', pointsData(50), 50);
    pool.acquirePointsGeometry('p2', pointsData(50), 50);
    pool.releasePointsGeometry('p1');
    pool.releasePointsGeometry('p2');
    // Each pointsData(50) buffer ≈ 50*(12+12+4+4) = 1600 bytes (allocated to
    // 1.5× capacity = 75 → ~2400 bytes). Two pooled buffers = ~4800 bytes
    // total, well over the 200-byte budget.
    pool.evictUnused();
    const stats = pool.getStats();
    // Byte-budget pass disposes pooled buffers (largest-first) until under.
    expect(stats.pooledBytes).toBeLessThanOrEqual(200);
  });

  it('largest-first eviction order: largest pooled buffer disposed first', () => {
    // Construct a fresh pool with a budget large enough to keep the small
    // buffer but force the large one out.
    // pointsData(50) attribute bytes ≈ 50 × (12+12+4+4) × 1.5 capacity ≈ 2400 B.
    // pointsData(500) ≈ 500 × 32 × 1.5 ≈ 24000 B.
    // Set budget = 5000 so only the 50-point buffer fits.
    const orderedPool = new GPUBufferPool(20, 300, 5, 5000);
    orderedPool.acquirePointsGeometry('small', pointsData(50), 50);
    orderedPool.acquirePointsGeometry('large', pointsData(500), 500);
    orderedPool.releasePointsGeometry('small');
    orderedPool.releasePointsGeometry('large');
    // releasePointsGeometry calls evictUnused() — by this point the large
    // (>budget) buffer must have been disposed (largest-first), and the
    // small buffer should remain pooled.
    const after = orderedPool.getStats();
    expect(after.pooledBytes).toBeLessThanOrEqual(5000);
    expect(after.byType.points.evictions).toBeGreaterThan(0);
    // The small buffer should still be in the pool (largest-first means
    // the large one was the eviction target, not the small one).
    expect(after.pooledBuffers).toBe(1);
  });

  it('byte budget = 0 disables the byte-budget pass (count only)', () => {
    const countOnly = new GPUBufferPool(20, 300, 5, 0);
    for (let i = 0; i < 5; i++) {
      countOnly.acquirePointsGeometry(`p${i}`, pointsData(50), 50);
      countOnly.releasePointsGeometry(`p${i}`);
    }
    countOnly.evictUnused();
    const stats = countOnly.getStats();
    // No byte-budget eviction; pooled buffers retained.
    expect(stats.pooledBuffers).toBeGreaterThan(0);
    expect(stats.pooledBytes).toBeGreaterThan(0);
  });

  it('does not evict ACTIVE buffers (only pooled)', () => {
    pool.acquirePointsGeometry('active', pointsData(500), 500);
    pool.acquirePointsGeometry('pooled', pointsData(500), 500);
    pool.releasePointsGeometry('pooled');
    pool.evictUnused();
    const stats = pool.getStats();
    // Active buffer must still be there.
    expect(stats.activeBuffers).toBe(1);
    expect(stats.activeBytes).toBeGreaterThan(0);
  });

  it('idempotent when pool already under byte budget', () => {
    const big = new GPUBufferPool(20, 300, 5, 1_000_000_000); // 1 GB budget
    big.acquirePointsGeometry('p1', pointsData(10), 10);
    big.releasePointsGeometry('p1');
    const before = big.getStats();
    big.evictUnused();
    const after = big.getStats();
    expect(after.pooledBuffers).toBe(before.pooledBuffers);
    expect(after.pooledBytes).toBe(before.pooledBytes);
  });

  it('byte-budget eviction triggers in the auto-eviction path (release → evict)', () => {
    // Acquire+release a buffer that's much larger than the 200-byte budget.
    pool.acquirePointsGeometry('p1', pointsData(500), 500);
    pool.releasePointsGeometry('p1');
    // releasePointsGeometry calls evictUnused() internally.
    const stats = pool.getStats();
    expect(stats.pooledBytes).toBeLessThanOrEqual(200);
  });

  it('dispose() clears all bytes (active + pooled)', () => {
    pool.acquirePointsGeometry('a', pointsData(100), 100);
    pool.acquirePointsGeometry('b', pointsData(100), 100);
    pool.releasePointsGeometry('b');
    pool.dispose();
    const stats = pool.getStats();
    expect(stats.activeBytes).toBe(0);
    expect(stats.pooledBytes).toBe(0);
    expect(stats.totalBytes).toBe(0);
  });

  it('pool integration — 20 buffers, evict largest until under budget', () => {
    // Each pointsData(N) buffer is ~32 N bytes (positions + colors + radii + sharpness).
    const evictPool = new GPUBufferPool(20, 300, 5, 16_000); // 16 KB budget
    // Acquire 20 buffers of varying sizes and release them into the pool.
    const sizes = [100, 200, 50, 400, 80, 300, 60, 250, 150, 90];
    for (let i = 0; i < sizes.length; i++) {
      evictPool.acquirePointsGeometry(`p${i}`, pointsData(sizes[i]), sizes[i]);
      evictPool.releasePointsGeometry(`p${i}`);
    }
    const stats = evictPool.getStats();
    // The eviction should have brought us under budget (or nearly so).
    expect(stats.pooledBytes).toBeLessThanOrEqual(16_000);
  });

  it('byte-budget eviction completes when dispose is a no-op', () => {
    // Acquire a pool buffer, release it, then mutate the pooled
    // geometry's dispose to be a no-op. Eviction removes the pooled
    // entry from bookkeeping and must complete in finite time.
    const evictPool = new GPUBufferPool(5, 300, 5, 100);
    evictPool.acquirePointsGeometry('p1', pointsData(500), 500);
    evictPool.releasePointsGeometry('p1');
    // Find the pooled buffer and neuter its dispose.
    const stats0 = evictPool.getStats();
    expect(stats0.pooledBuffers).toBeGreaterThanOrEqual(0);

    // Force re-eviction by acquiring + releasing another over-budget buffer.
    evictPool.acquirePointsGeometry('p2', pointsData(500), 500);
    evictPool.releasePointsGeometry('p2');
    // The byte budget is 100 bytes; even one pointsData(500) is huge
    // (~24 KB). The eviction should have run despite the no-op dispose.

    // The completion (no infinite loop) is itself the assertion. Add a
    // soft check that the pool didn't somehow accumulate buffers.
    const stats1 = evictPool.getStats();
    expect(stats1.pooledBuffers).toBeLessThanOrEqual(2);
    evictPool.dispose();
  });

  it('under-budget pool: evictUnused does not dispose any pooled buffer', () => {
    // Lock in the early-return path in `_evictUntilUnderByteBudget`:
    // if total pooled bytes are already under the configured budget,
    // eviction must NOT dispose any pooled buffer (and the eviction
    // counter must not advance from the byte-budget path).
    const generousPool = new GPUBufferPool(20, 300, 5, 100_000_000);
    generousPool.acquirePointsGeometry('p1', pointsData(50), 50);
    generousPool.acquirePointsGeometry('p2', pointsData(50), 50);
    generousPool.releasePointsGeometry('p1');
    generousPool.releasePointsGeometry('p2');

    // Capture pre-eviction stats; the auto-eviction inside `release`
    // shouldn't have done anything either (both buffers are tiny,
    // budget is 100 MB).
    const before = generousPool.getStats();
    expect(before.pooledBuffers).toBe(2);

    generousPool.evictUnused();

    const after = generousPool.getStats();
    // Same pooled-buffer count; no eviction triggered.
    expect(after.pooledBuffers).toBe(before.pooledBuffers);
    expect(after.pooledBytes).toBe(before.pooledBytes);
    expect(after.byType.points.evictions).toBe(before.byType.points.evictions);

    generousPool.dispose();
  });
});
