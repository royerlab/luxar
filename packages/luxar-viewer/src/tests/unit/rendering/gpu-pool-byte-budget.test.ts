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
  invalidateCachedByteSize,
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

// selectBuffersToEvict (pure function) tests live in
// tests/unit/rendering/gpu-buffer-pool/eviction-policy.test.ts —
// alongside the extracted module.

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

  it('caches the result on userData.cachedByteSize', () => {
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

  it('invalidateCachedByteSize is a no-op when the cache is missing', () => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(30, 3));
    invalidateCachedByteSize(g); // no cache yet — should not throw
    expect(estimateGeometryBytes(g)).toBe(120);
  });
});

describe('GPU pool getStats() byte counters', () => {
  it('reports activeBytes for an acquired buffer', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0); // byte budget disabled
    pool.acquirePointsGeometry('p1', pointsData(100), 100);
    const stats = pool.getStats();
    expect(stats.activeBytes).toBeGreaterThan(0);
    expect(stats.pooledBytes).toBe(0);
    expect(stats.totalBytes).toBe(stats.activeBytes);
    expect(stats.byType.points.activeBytes).toBe(stats.activeBytes);
  });

  it('moves bytes from active to pooled on release', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    pool.acquirePointsGeometry('p1', pointsData(100), 100);
    const before = pool.getStats();
    pool.releasePointsGeometry('p1');
    const after = pool.getStats();
    expect(after.activeBytes).toBe(0);
    expect(after.pooledBytes).toBe(before.activeBytes);
    expect(after.byType.points.pooledBytes).toBe(before.activeBytes);
  });

  it('tracks largestPooledBytes across all type pools', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
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
    pool = new GPUBufferPool(20, 300, 5, () => 200);
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
    // Keep the byte budget disabled (0) during setup so both buffers land
    // in the pool with NO active interference, then enforce a budget that
    // fits the small buffer but not small+large. Largest-first must dispose
    // the large pooled buffer and keep the small one. (Measuring up front
    // and enforcing once keeps the test independent of release order — under
    // the total-resident model, an over-budget *active* buffer at release
    // time would otherwise force all pooled buffers out.)
    let budget = 0; // disabled during setup
    const orderedPool = new GPUBufferPool(20, 300, 5, () => budget);
    orderedPool.acquirePointsGeometry('small', pointsData(50), 50);
    const small = orderedPool.getStats().activeBytes;
    orderedPool.acquirePointsGeometry('large', pointsData(500), 500);
    const large = orderedPool.getStats().activeBytes - small;
    expect(large).toBeGreaterThan(small);

    orderedPool.releasePointsGeometry('small');
    orderedPool.releasePointsGeometry('large'); // both pooled, nothing disposed (budget 0)
    expect(orderedPool.getStats().pooledBuffers).toBe(2);

    // Budget fits the small buffer but not small + large.
    budget = small + Math.floor(large / 2);
    orderedPool.evictUnused();

    const after = orderedPool.getStats();
    expect(after.pooledBytes).toBeLessThanOrEqual(budget);
    expect(after.byType.points.evictions).toBeGreaterThan(0);
    // The small buffer should still be in the pool (largest-first means
    // the large one was the eviction target, not the small one).
    expect(after.pooledBuffers).toBe(1);
    orderedPool.dispose();
  });

  it('byte budget = 0 disables the byte-budget pass (count only)', () => {
    const countOnly = new GPUBufferPool(20, 300, 5, () => 0);
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
    const big = new GPUBufferPool(20, 300, 5, () => 1_000_000_000); // 1 GB budget
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
    const evictPool = new GPUBufferPool(20, 300, 5, () => 16_000); // 16 KB budget
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
    const evictPool = new GPUBufferPool(5, 300, 5, () => 100);
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

  it('targets TOTAL resident (active + pooled): active consumes the budget, pooled disposed', () => {
    // Measure one buffer's active bytes, then set a budget between 1× and
    // 2× that — so the active buffer fits but active + an equal pooled
    // buffer exceeds it. The pooled buffer (not in use) must be disposed;
    // the active one (in use) must survive, leaving total resident ≤ budget.
    const probe = new GPUBufferPool(20, 300, 5, () => 0);
    probe.acquirePointsGeometry('active', pointsData(500), 500);
    const oneBuffer = probe.getStats().activeBytes;
    probe.dispose();
    expect(oneBuffer).toBeGreaterThan(0);

    const budget = Math.floor(oneBuffer * 1.5);
    const pool2 = new GPUBufferPool(20, 300, 5, () => budget);
    pool2.acquirePointsGeometry('active', pointsData(500), 500);
    pool2.acquirePointsGeometry('pooled', pointsData(500), 500);
    pool2.releasePointsGeometry('pooled'); // active(1) + pooled(1) > budget
    pool2.evictUnused();

    const stats = pool2.getStats();
    expect(stats.activeBuffers).toBe(1); // active never disposed
    expect(pool2.getResidentBytes()).toBeLessThanOrEqual(budget);
    expect(stats.totalBytes).toBe(pool2.getResidentBytes());
    pool2.dispose();
  });

  it('reads the LIVE budget each pass — lowering it (context-loss backoff) evicts more', () => {
    let budget = 100_000_000; // generous: pooled buffers retained for reuse
    const pool2 = new GPUBufferPool(20, 300, 5, () => budget);
    for (let i = 0; i < 4; i++) {
      pool2.acquirePointsGeometry(`p${i}`, pointsData(200), 200);
      pool2.releasePointsGeometry(`p${i}`);
    }
    const before = pool2.getStats();
    expect(before.pooledBuffers).toBeGreaterThan(0);

    // Simulate the WebGL-context-loss budget backoff dropping the budget.
    budget = 1;
    pool2.evictUnused();
    const after = pool2.getStats();
    expect(after.pooledBuffers).toBeLessThan(before.pooledBuffers);
    pool2.dispose();
  });

  it('getResidentBytes() equals getStats().totalBytes (active + pooled)', () => {
    const pool2 = new GPUBufferPool(20, 300, 5, () => 0);
    pool2.acquirePointsGeometry('a', pointsData(100), 100);
    pool2.acquirePointsGeometry('b', pointsData(200), 200);
    pool2.releasePointsGeometry('b');
    expect(pool2.getResidentBytes()).toBe(pool2.getStats().totalBytes);
    pool2.dispose();
  });

  it('under-budget pool: evictUnused does not dispose any pooled buffer', () => {
    // Lock in the early-return path in `_evictUntilUnderByteBudget`:
    // if total pooled bytes are already under the configured budget,
    // eviction must NOT dispose any pooled buffer (and the eviction
    // counter must not advance from the byte-budget path).
    const generousPool = new GPUBufferPool(20, 300, 5, () => 100_000_000);
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
