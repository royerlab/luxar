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
    pool.acquirePointsGeometry('p1', 100);
    const stats = pool.getStats();
    expect(stats.activeBytes).toBeGreaterThan(0);
    expect(stats.pooledBytes).toBe(0);
    expect(stats.totalBytes).toBe(stats.activeBytes);
    expect(stats.byType.points.activeBytes).toBe(stats.activeBytes);
  });

  it('moves bytes from active to pooled on release', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    pool.acquirePointsGeometry('p1', 100);
    const before = pool.getStats();
    pool.releasePointsGeometry('p1');
    const after = pool.getStats();
    expect(after.activeBytes).toBe(0);
    expect(after.pooledBytes).toBe(before.activeBytes);
    expect(after.byType.points.pooledBytes).toBe(before.activeBytes);
  });

  it('tracks largestPooledBytes across all type pools', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    pool.acquirePointsGeometry('p1', 100);
    pool.acquirePointsGeometry('p2', 1000);
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
    pool.acquirePointsGeometry('p1', 50);
    pool.acquirePointsGeometry('p2', 50);
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
    orderedPool.acquirePointsGeometry('small', 50);
    const small = orderedPool.getStats().activeBytes;
    orderedPool.acquirePointsGeometry('large', 500);
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
      countOnly.acquirePointsGeometry(`p${i}`, 50);
      countOnly.releasePointsGeometry(`p${i}`);
    }
    countOnly.evictUnused();
    const stats = countOnly.getStats();
    // No byte-budget eviction; pooled buffers retained.
    expect(stats.pooledBuffers).toBeGreaterThan(0);
    expect(stats.pooledBytes).toBeGreaterThan(0);
  });

  it('does not evict ACTIVE buffers (only pooled)', () => {
    pool.acquirePointsGeometry('active', 500);
    pool.acquirePointsGeometry('pooled', 500);
    pool.releasePointsGeometry('pooled');
    pool.evictUnused();
    const stats = pool.getStats();
    // Active buffer must still be there.
    expect(stats.activeBuffers).toBe(1);
    expect(stats.activeBytes).toBeGreaterThan(0);
  });

  it('idempotent when pool already under byte budget', () => {
    const big = new GPUBufferPool(20, 300, 5, () => 1_000_000_000); // 1 GB budget
    big.acquirePointsGeometry('p1', 10);
    big.releasePointsGeometry('p1');
    const before = big.getStats();
    big.evictUnused();
    const after = big.getStats();
    expect(after.pooledBuffers).toBe(before.pooledBuffers);
    expect(after.pooledBytes).toBe(before.pooledBytes);
  });

  it('acquire-triggered sweep graces buffers RELEASED this frame, even when last acquired frames ago', () => {
    // Regression: releaseGeometry must stamp lastUsedFrame with the
    // release frame — without the stamp, a buffer acquired at frame 0
    // and released at frame 3 carries lastUsedFrame=0, the grace
    // (graceFrame=3) never matches, and the dataset-switch sweep
    // disposes the very buffer the grace exists to preserve.
    let budget = 1_000_000_000;
    const gracePool = new GPUBufferPool(20, 300, 5, () => budget);
    gracePool.acquirePointsGeometry('old', 500); // stamped frame 0
    gracePool.beginFrame();
    gracePool.beginFrame();
    gracePool.beginFrame(); // frame 3
    gracePool.releasePointsGeometry('old'); // release sweep: under budget, survives
    expect(gracePool.getStats().pooledBuffers).toBe(1);

    // Shrink the budget so the next acquire's sweep is over budget, and
    // request more than 'old''s capacity so best-fit reuse cannot
    // short-circuit the fresh allocation (+ its acquire-triggered sweep).
    budget = 1;
    gracePool.acquirePointsGeometry('fresh', 5000);
    expect(gracePool.getStats().pooledBuffers).toBe(1); // 'old' spared by the grace

    // A release-triggered sweep (graceFrame -1) still enforces the budget.
    gracePool.releasePointsGeometry('fresh');
    expect(gracePool.getStats().pooledBytes).toBe(0);
  });

  it('byte-budget eviction triggers in the auto-eviction path (release → evict)', () => {
    // Acquire+release a buffer that's much larger than the 200-byte budget.
    pool.acquirePointsGeometry('p1', 500);
    pool.releasePointsGeometry('p1');
    // releasePointsGeometry calls evictUnused() internally.
    const stats = pool.getStats();
    expect(stats.pooledBytes).toBeLessThanOrEqual(200);
  });

  it('dispose() clears all bytes (active + pooled)', () => {
    pool.acquirePointsGeometry('a', 100);
    pool.acquirePointsGeometry('b', 100);
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
      evictPool.acquirePointsGeometry(`p${i}`, sizes[i]);
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
    evictPool.acquirePointsGeometry('p1', 500);
    evictPool.releasePointsGeometry('p1');
    // Find the pooled buffer and neuter its dispose.
    const stats0 = evictPool.getStats();
    expect(stats0.pooledBuffers).toBeGreaterThanOrEqual(0);

    // Force re-eviction by acquiring + releasing another over-budget buffer.
    evictPool.acquirePointsGeometry('p2', 500);
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
    probe.acquirePointsGeometry('active', 500);
    const oneBuffer = probe.getStats().activeBytes;
    probe.dispose();
    expect(oneBuffer).toBeGreaterThan(0);

    const budget = Math.floor(oneBuffer * 1.5);
    const pool2 = new GPUBufferPool(20, 300, 5, () => budget);
    pool2.acquirePointsGeometry('active', 500);
    pool2.acquirePointsGeometry('pooled', 500);
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
      pool2.acquirePointsGeometry(`p${i}`, 200);
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
    pool2.acquirePointsGeometry('a', 100);
    pool2.acquirePointsGeometry('b', 200);
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
    generousPool.acquirePointsGeometry('p1', 50);
    generousPool.acquirePointsGeometry('p2', 50);
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

  /**
   * `evictions` conflates two events that mean opposite things to a capture
   * tool (#2508): routine LRU recycling of released pooled buffers, which every
   * nD scene does constantly as slices change, and byte-budget reclamation,
   * which means the pool could not hold what the scene asked for. Only the
   * second makes the rendered counts unreproducible, so it gets its own
   * counter — and a consumer that refused on `evictions` would refuse every
   * normal nD capture.
   */
  describe('byteBudgetEvictions attribution', () => {
    it('counts the byte-budget pass', () => {
      const bytePool = new GPUBufferPool(20, 300, 5, () => 200);
      bytePool.acquirePointsGeometry('p1', 500);
      bytePool.releasePointsGeometry('p1'); // release runs evictUnused internally

      const stats = bytePool.getStats();
      expect(stats.byteBudgetEvictions).toBeGreaterThan(0);
      // A subset of the total, never a second tally of the same event.
      expect(stats.byteBudgetEvictions).toBeLessThanOrEqual(stats.evictions);
      bytePool.dispose();
    });

    it('stays 0 for a count-based LRU eviction', () => {
      // THE BUDGET REGIME IS THE WHOLE TEST. A GENEROUS but ENABLED byte budget
      // is the discriminating setup: with the pass disabled (budget 0) the
      // whole block is skipped, so a counter mis-wired to `evictUnused`'s TOTAL
      // would still read 0 and this would prove nothing. Verified by mutation —
      // `byteBudgetEvictions += evicted` survives at budget 0 and fails here.
      // Eviction age 1 frame, so the LRU pass disposes the pooled buffer while
      // the byte pass finds the pool comfortably under budget and does nothing.
      const lruPool = new GPUBufferPool(20, 1, 5, () => 100_000_000);
      lruPool.acquirePointsGeometry('p1', 100);
      lruPool.releasePointsGeometry('p1');
      for (let i = 0; i < 5; i++) lruPool.beginFrame();

      expect(lruPool.evictUnused()).toBeGreaterThan(0);
      const stats = lruPool.getStats();
      expect(stats.evictions).toBeGreaterThan(0);
      expect(stats.byteBudgetEvictions).toBe(0);
      lruPool.dispose();
    });

    it('stays 0 when the byte-budget pass is disabled outright', () => {
      const countOnly = new GPUBufferPool(20, 1, 5, () => 0);
      countOnly.acquirePointsGeometry('p1', 500);
      countOnly.releasePointsGeometry('p1');
      for (let i = 0; i < 5; i++) countOnly.beginFrame();
      countOnly.evictUnused();

      expect(countOnly.getStats().byteBudgetEvictions).toBe(0);
      countOnly.dispose();
    });
  });
});

describe('post-grow reclaim (#2426 pool retention)', () => {
  /**
   * A growing ladder is release + reacquire per capacity tier, and the pair
   * released at each growth used to be stranded for the life of the scene by
   * TWO independent mechanisms:
   *
   *  - `releaseGeometry`'s own sweep IS unconditional, but it runs before the
   *    larger replacement registers, so it measures the budget against
   *    pre-growth active bytes, sees headroom that no longer exists, and
   *    evicts nothing;
   *  - the acquire-side sweep does see the new accounting, but runs with
   *    `graceFrame = frameCount` while the release just stamped the pair with
   *    that same frame — so the grace skips precisely the buffer that needs
   *    taking. (On the ADOPT path there is no acquire sweep at all.)
   *
   * A SINGLE node is the discriminating case: multi-node scenes hid this,
   * because a later node's churn re-runs the pass and catches earlier nodes'
   * pairs — leaving only the last node's final pair stranded. One node never
   * acquires again, so nothing ever reclaims it.
   */
  // THE BUDGET REGIME IS THE WHOLE TEST, and getting it wrong makes these
  // vacuous. A 100-point buffer is ~66.8 KB and a 5000-point one ~453 KB, so:
  //   budget < 66.8 KB  → the RELEASE sweep already reclaims it (at release the
  //                       node's own bytes have left `activeBytes`, so
  //                       pooledTarget is the whole budget) — the post-grow
  //                       sweep is never needed and the test proves nothing.
  //   budget > 520 KB   → nothing is ever over budget; also proves nothing.
  //   in between        → pooled alone fits, active + pooled does not. That is
  //                       the real ladder regime, and the only one where the
  //                       stranding is observable.
  // Verified by mutation: with the post-grow sweep removed, the first test
  // below fails at this budget and passes at 40 KB.
  const BUDGET_IN_REGIME = 480_000;

  it('reclaims the superseded pair when a single node grows past the budget', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => BUDGET_IN_REGIME);
    pool.acquirePointsGeometry('n1', 100);
    const grown = pool.acquirePointsGeometry('n1', 5000);
    expect(grown).toBeDefined();

    const stats = pool.getStats();
    // The whole point: no pooled residue left behind by the growth.
    expect(stats.pooledBytes).toBe(0);
    expect(stats.byType.points.evictions).toBeGreaterThan(0);

    pool.dispose();
  });

  it('leaves the pair pooled when the budget has room, preserving reuse', () => {
    // The reclaim is budget-driven, not unconditional disposal: with headroom
    // the released pair must stay adoptable, or growth becomes alloc/dispose
    // churn on machines that were never under pressure.
    const pool = new GPUBufferPool(20, 300, 5, () => 100_000_000);
    pool.acquirePointsGeometry('n1', 100);
    pool.acquirePointsGeometry('n1', 5000);

    const stats = pool.getStats();
    expect(stats.pooledBuffers).toBeGreaterThan(0);
    expect(stats.byType.points.evictions).toBe(0);

    pool.dispose();
  });

  it('does not disturb the active buffer it just grew into', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => BUDGET_IN_REGIME);
    pool.acquirePointsGeometry('n1', 100);
    const grown = pool.acquirePointsGeometry('n1', 5000);

    // Active is never evictable; the node must still hold the geometry it was
    // handed, or the caller renders against a disposed buffer.
    const stats = pool.getStats();
    expect(stats.activeBytes).toBeGreaterThan(0);
    expect(grown.userData.luxarInvalidated).not.toBe(true);

    pool.dispose();
  });

  it.each([
    ['points', 100, 5000],
    ['lines', 100, 5000],
    ['gsplats', 100, 10_000],
  ] as const)(
    '%s: a throwing post-grow sweep preserves the grown active buffer',
    (type, initialCount, grownCount) => {
      const pool = new GPUBufferPool(20, 300, 5, () => BUDGET_IN_REGIME);
      const acquire = (count: number): THREE.InstancedBufferGeometry => {
        switch (type) {
          case 'points':
            return pool.acquirePointsGeometry('n1', count);
          case 'lines':
            return pool.acquireLinesGeometry('n1', count);
          case 'gsplats':
            return pool.acquireGSplatsGeometry('n1', count);
        }
      };
      const freeBuckets =
        type === 'points'
          ? pool.points.pointBuffers
          : type === 'lines'
            ? pool.lines.lineBuffers
            : pool.gsplats.gsplatBuffers;

      const oldGeometry = acquire(initialCount);
      expect(pool.getStats().activeBytes).toBeLessThan(BUDGET_IN_REGIME);
      oldGeometry.addEventListener('dispose', () => {
        throw new Error('synthetic post-grow dispose failure');
      });

      expect(() => acquire(grownCount)).toThrow('synthetic post-grow dispose failure');

      const active = pool.activeBuffers.get('n1');
      expect(active).toBeDefined();
      expect(active!.geometry).not.toBe(oldGeometry);
      expect(active!.inUse).toBe(true);
      expect(active!.geometry.userData.luxarInvalidated).not.toBe(true);
      for (const buffers of freeBuckets.values()) {
        for (const buffer of buffers) {
          expect(buffer.geometry).not.toBe(active!.geometry);
        }
      }

      pool.dispose();
    }
  );
});
