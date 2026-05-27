/**
 * Comprehensive tests for ChunkPrefetcher
 *
 * Tests cover:
 * - Chunk index parsing (v2/v3 formats)
 * - Adjacent chunk calculation
 * - Queue management and deduplication
 * - Concurrency limiting
 * - Integration with MultiLevelCachingStore
 * - URL parameter handling
 * - Race condition safety
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ChunkPrefetcher } from '../../../cache/chunk-prefetcher';

/**
 * Deterministic polling helper — waits for a condition to become true.
 * Replaces hardcoded setTimeout delays that can flake on slow CI.
 */
async function waitFor(condition: () => boolean, timeout = 2000): Promise<void> {
  const start = Date.now();
  while (!condition() && Date.now() - start < timeout) {
    await new Promise((r) => setTimeout(r, 5));
  }
}

// Mock MultiLevelCachingStore for unit tests. The prefetcher uses
// `getResult` (Result-typed) on the store; tests inspect calls on it.
class MockStore {
  getResult = vi.fn().mockResolvedValue({ ok: true, value: new Uint8Array([1, 2, 3]) });
  setPrefetcher = vi.fn();
}

describe('ChunkPrefetcher - Unit Tests', () => {
  let mockStore: MockStore;
  let prefetcher: ChunkPrefetcher;

  beforeEach(() => {
    vi.clearAllMocks();
    mockStore = new MockStore();
    prefetcher = new ChunkPrefetcher(mockStore as any, { enabled: true });
    // Register bounds for common test paths so prefetcher can compute valid neighbors.
    // Without bounds, getAdjacentChunks returns [] to avoid out-of-bounds 404s.
    prefetcher.registerArrayBounds('points/positions', [10240, 10240, 10240], [1024, 1024, 1024]);
    prefetcher.registerArrayBounds('data/volume', [100, 100, 100, 30], [10, 10, 10, 10]);
    prefetcher.registerArrayBounds('data/values', [10240], [1024]);
    prefetcher.registerArrayBounds('test', [10240, 4], [1024, 4]);
  });

  describe('Chunk Index Parsing', () => {
    it('should parse zarr v2 chunk keys (dot notation)', async () => {
      // Register bounds so prefetcher knows valid chunk range (large enough for all neighbors)
      prefetcher.registerArrayBounds('points/positions', [10240, 10240, 10240], [1024, 1024, 1024]);

      prefetcher.onAccess('points/positions/0.1.2');

      // Wait for all prefetch operations to complete
      await waitFor(() => {
        const s = prefetcher.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      // Verify adjacent chunks were generated (v2 format)
      expect(mockStore.getResult).toHaveBeenCalled();
      const calls = mockStore.getResult.mock.calls.map((call: any[]) => call[0]);

      // Should have ±1 in each dimension (5 neighbors: dim0 can't go negative)
      expect(calls).toContain('points/positions/1.1.2'); // +1 in dim 0
      expect(calls).toContain('points/positions/0.0.2'); // -1 in dim 1
      expect(calls).toContain('points/positions/0.2.2'); // +1 in dim 1
      expect(calls).toContain('points/positions/0.1.1'); // -1 in dim 2
      expect(calls).toContain('points/positions/0.1.3'); // +1 in dim 2

      // Should NOT have negative indices
      expect(calls).not.toContain('points/positions/-1.1.2'); // dim 0 can't go negative
    });

    it('should parse zarr v3 chunk keys (path notation)', async () => {
      // Create prefetcher with debug enabled
      const debugPrefetcher = new ChunkPrefetcher(mockStore as any, {
        enabled: true,
        debug: true,
      });

      // Register bounds so prefetcher knows valid chunk range
      debugPrefetcher.registerArrayBounds(
        'points/positions',
        [10240, 10240, 10240],
        [1024, 1024, 1024]
      );

      debugPrefetcher.onAccess('points/positions/c/0/1/2');

      // Wait for all prefetch operations to complete
      await waitFor(() => {
        const s = debugPrefetcher.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      const calls = mockStore.getResult.mock.calls.map((call: any[]) => call[0]);

      // Should have ±1 in each dimension (v3 format, 5 neighbors total)
      expect(calls.length).toBe(5); // 3 dims * 2 - 1 (can't go negative in dim 0)
      expect(calls).toContain('points/positions/c/1/1/2');
      expect(calls).toContain('points/positions/c/0/0/2');
      expect(calls).toContain('points/positions/c/0/2/2');
      expect(calls).toContain('points/positions/c/0/1/1');
      expect(calls).toContain('points/positions/c/0/1/3');
    });

    it('should skip non-chunk keys (metadata)', () => {
      prefetcher.onAccess('.zattrs');
      prefetcher.onAccess('.zarray');
      prefetcher.onAccess('.zgroup');

      // Should not prefetch anything for metadata files
      expect(mockStore.getResult).not.toHaveBeenCalled();
    });

    // [cache OOS] V3 detection now uses an anchored regex
    // /\/c\/\d+(\/\d+)*$/ instead of bare `includes('/c/')`. The
    // pre-fix `.includes('/c/')` check would misclassify a key whose
    // BASE PATH happens to contain `/c/` (e.g. a path through a
    // directory called "cleanup") as v3 — corrupting the base-path
    // extraction. With the anchor, only true v3 chunk keys (ending in
    // /c/N/M/...) take the v3 branch.
    it('does not misclassify v2 keys whose base path happens to contain /c/', async () => {
      const debugPrefetcher = new ChunkPrefetcher(mockStore as any, {
        enabled: true,
        debug: true,
      });

      // v2-style key where the base path itself has "/c/" embedded
      // (e.g. an array under a directory called "cleanup"). Bounds
      // registered under the actual base path; the v2-style dot-separated
      // chunk index is the trailing token.
      debugPrefetcher.registerArrayBounds(
        'scenes/cleanup/positions',
        [10240, 10240, 10240],
        [1024, 1024, 1024]
      );
      debugPrefetcher.onAccess('scenes/cleanup/positions/1.2.3');

      await waitFor(() => {
        const s = debugPrefetcher.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      const calls = mockStore.getResult.mock.calls.map((call: any[]) => call[0]);

      // The pre-fix code would have stripped `/c/positions/1.2.3` (anchor-
      // less regex `/\/c\/[\d/]+$/`), producing basePath `scenes`, looking
      // up unknown bounds, and short-circuiting to no prefetches. The
      // post-fix v3 anchor doesn't match this key at all, so the v2
      // branch runs and produces the correct ±1 adjacents.
      expect(calls.length).toBeGreaterThan(0);
      expect(calls).toContain('scenes/cleanup/positions/0.2.3');
      expect(calls).toContain('scenes/cleanup/positions/2.2.3');
      expect(calls).toContain('scenes/cleanup/positions/1.1.3');
      expect(calls).toContain('scenes/cleanup/positions/1.3.3');
      expect(calls).toContain('scenes/cleanup/positions/1.2.2');
      expect(calls).toContain('scenes/cleanup/positions/1.2.4');
    });

    it('should handle 1D chunks', () => {
      // Register bounds for 1D array with multiple chunks
      prefetcher.registerArrayBounds('data', [10240], [1024]);

      prefetcher.onAccess('data/0');

      const calls = mockStore.getResult.mock.calls.map((call: any[]) => call[0]);

      // Should have only ±1 in single dimension
      expect(calls).toContain('data/1'); // +1
      expect(calls).not.toContain('data/-1'); // No negative
      expect(calls).toHaveLength(1); // Only 1 neighbor (can't go negative)
    });

    it('should handle 4D chunks', async () => {
      // Register bounds for 4D array with multiple chunks per dimension
      prefetcher.registerArrayBounds(
        'data',
        [10240, 10240, 10240, 10240],
        [1024, 1024, 1024, 1024]
      );

      prefetcher.onAccess('data/1.2.3.4');

      // Wait for all prefetch operations to complete
      await waitFor(() => {
        const s = prefetcher.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      const calls = mockStore.getResult.mock.calls.map((call: any[]) => call[0]);

      // Should have 2 neighbors per dimension * 4 dimensions = 8 total
      expect(calls).toHaveLength(8);
      expect(calls).toContain('data/0.2.3.4'); // -1 in dim 0
      expect(calls).toContain('data/2.2.3.4'); // +1 in dim 0
      expect(calls).toContain('data/1.1.3.4'); // -1 in dim 1
      expect(calls).toContain('data/1.3.3.4'); // +1 in dim 1
      expect(calls).toContain('data/1.2.2.4'); // -1 in dim 2
      expect(calls).toContain('data/1.2.4.4'); // +1 in dim 2
      expect(calls).toContain('data/1.2.3.3'); // -1 in dim 3
      expect(calls).toContain('data/1.2.3.5'); // +1 in dim 3
    });
  });

  describe('Concurrency Limiting', () => {
    it('should limit concurrent prefetches to maxConcurrent', async () => {
      // Use a controlled store whose gets never resolve until we say so,
      // so we can inspect the in-flight count deterministically.
      const resolvers: Array<(value: { ok: true; value: Uint8Array }) => void> = [];
      const controlledStore = {
        getResult: vi.fn().mockImplementation(
          () =>
            new Promise<{ ok: true; value: Uint8Array }>((resolve) => {
              resolvers.push((v) => resolve(v));
            })
        ),
        setPrefetcher: vi.fn(),
      };

      const limitedPrefetcher = new ChunkPrefetcher(controlledStore as any, {
        enabled: true,
        maxConcurrent: 2,
      });

      // Register bounds so neighbors are generated
      limitedPrefetcher.registerArrayBounds('data', [10240, 10240], [1024, 1024]);

      // Trigger prefetch (will queue 4 neighbors for v2 2D chunk at center)
      limitedPrefetcher.onAccess('data/1.1');

      // Wait for the queue to be drained into in-flight slots
      await waitFor(() => controlledStore.getResult.mock.calls.length >= 2);

      const stats = limitedPrefetcher.getStats();

      // Should have at most 2 in flight (the controlled promises are pending)
      expect(stats.inFlight).toBeLessThanOrEqual(2);

      // Resolve all to clean up
      resolvers.forEach((r) => r({ ok: true, value: new Uint8Array([1]) }));
    });

    it('should process queue when slots free up', async () => {
      const limitedPrefetcher = new ChunkPrefetcher(mockStore as any, {
        enabled: true,
        maxConcurrent: 1,
      });

      // Register bounds so prefetcher knows valid chunk range
      limitedPrefetcher.registerArrayBounds('data', [10240, 10240], [1024, 1024]);

      // Trigger prefetch (will queue 4 neighbors)
      limitedPrefetcher.onAccess('data/1.1');

      // Wait for all to complete
      await waitFor(() => {
        const s = limitedPrefetcher.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      // All should eventually be fetched
      expect(mockStore.getResult).toHaveBeenCalledTimes(4);
    });
  });

  describe('Deduplication', () => {
    it('should not queue same chunk twice (mockStore.getResult exactly equals unique-neighbor count)', () => {
      // cache.md W17 fix: previous version asserted `<= 4` which was
      // satisfied trivially because the synchronous mock resolves before
      // we read stats (queued+inFlight collapsed to 0 either way).
      // Stronger contract: the mock store's getResult call count after
      // two onAccess calls to the same key equals the count after one
      // call — proving dedup. (We don't pin "exactly 4" because the
      // exact neighbor count depends on the chunk's position relative
      // to bounds and chunk-shape; the load-bearing invariant is dedup.)
      const callCountFromMock = () => mockStore.getResult.mock.calls.length;
      mockStore.getResult.mockClear();
      prefetcher.onAccess('points/positions/1.1.1');
      const afterFirst = callCountFromMock();
      prefetcher.onAccess('points/positions/1.1.1');
      const afterSecond = callCountFromMock();
      // Dedup: second access to the same key must not trigger any new fetches.
      expect(afterSecond).toBe(afterFirst);
      // Sanity: the first access must have triggered SOME fetches.
      expect(afterFirst).toBeGreaterThan(0);
    });

    it('should not queue chunks already in flight (exact-4 dedup, not over-eviction)', async () => {
      // cache.md W18 fix: same `<= 4` weakness as W17.
      const resolvers: Array<(value: { ok: true; value: Uint8Array }) => void> = [];
      const slowMockStore = {
        getResult: vi.fn().mockImplementation(
          () =>
            new Promise<{ ok: true; value: Uint8Array }>((resolve) => {
              resolvers.push((v) => resolve(v));
            })
        ),
        setPrefetcher: vi.fn(),
      };

      const slowPrefetcher = new ChunkPrefetcher(slowMockStore as any, {
        enabled: true,
        maxConcurrent: 4,
      });

      slowPrefetcher.registerArrayBounds('data', [10240, 10240], [1024, 1024]);
      slowPrefetcher.onAccess('data/1.1');
      await waitFor(() => slowMockStore.getResult.mock.calls.length > 0);
      const callsAfterFirst = slowMockStore.getResult.mock.calls.length;
      slowPrefetcher.onAccess('data/1.1');
      // Give the prefetcher a chance to schedule (or refuse) more work.
      await new Promise((r) => setTimeout(r, 10));
      // Dedup: second access must not trigger any new getResult calls
      // while the first batch is still in flight.
      expect(slowMockStore.getResult.mock.calls.length).toBe(callsAfterFirst);

      resolvers.forEach((r) => r({ ok: true, value: new Uint8Array([1]) }));
    });
  });

  describe('registerArrayBounds normalization [cache.md/G5][P5]', () => {
    it('strips a leading slash so bounds are usable from both call sites', () => {
      // [cache.md/G5][P5] Source comment at chunk-prefetcher.ts:289-296
      // explicitly normalizes the leading slash so zarrita keys (with
      // leading `/`) and loader paths (without) both find the same
      // bounds — but no test directly drove the leading-slash path.
      //
      // The contract we pin: when the producer registers with leading
      // slash, the lookup via the SAME shape (slashed key) still finds
      // bounds and dispatches neighbors. Mutating the source to drop the
      // `if (normalized !== arrayPath) this.maxChunkIndices.set(arrayPath, ...)`
      // line would break the slashed-form lookup, which this test pins.
      const slashedStore = new MockStore();
      const p = new ChunkPrefetcher(slashedStore as any, { enabled: true });
      p.registerArrayBounds('/leading/path', [10240, 10240], [1024, 1024]);

      // Access via the SLASHED key — must prefetch neighbors because
      // the bounds lookup succeeds via the `arrayPath` entry.
      p.onAccess('/leading/path/1.1');
      expect(slashedStore.getResult.mock.calls.length).toBeGreaterThan(0);

      // A fresh prefetcher registered WITHOUT a leading slash must also
      // succeed when accessed via the un-slashed form (the canonical
      // case — control for the slashed test above).
      const unSlashedStore = new MockStore();
      const p2 = new ChunkPrefetcher(unSlashedStore as any, { enabled: true });
      p2.registerArrayBounds('leading/path', [10240, 10240], [1024, 1024]);
      p2.onAccess('leading/path/1.1');
      expect(unSlashedStore.getResult.mock.calls.length).toBeGreaterThan(0);

      // Symmetry — same neighbor count from both registration forms.
      expect(slashedStore.getResult.mock.calls.length).toBe(
        unSlashedStore.getResult.mock.calls.length
      );
    });
  });

  describe('Configuration', () => {
    it('should respect enabled flag', () => {
      const disabledPrefetcher = new ChunkPrefetcher(mockStore as any, { enabled: false });

      disabledPrefetcher.onAccess('data/0.0');

      // Should not prefetch anything when disabled
      expect(mockStore.getResult).not.toHaveBeenCalled();
    });

    it('prefetches adjacent chunks when constructed with enabled=true (parity with disabled case)', () => {
      // Confirms the disabled case in the previous test isn't a no-op for some
      // unrelated reason — same setup but enabled=true must trigger prefetch.
      const enabledPrefetcher = new ChunkPrefetcher(mockStore as any, { enabled: true });
      enabledPrefetcher.registerArrayBounds('data', [2048, 2048], [1024, 1024]);

      enabledPrefetcher.onAccess('data/0.0');

      expect(mockStore.getResult).toHaveBeenCalled();
    });

    it('should use custom maxConcurrent', async () => {
      const customPrefetcher = new ChunkPrefetcher(mockStore as any, {
        enabled: true,
        maxConcurrent: 10,
      });

      // Register bounds so prefetcher generates adjacent chunks
      customPrefetcher.registerArrayBounds('data', [10240, 10240, 10240], [1024, 1024, 1024]);

      customPrefetcher.onAccess('data/1.1.1'); // 3D → 6 neighbors

      // Wait for all to be dispatched (no queuing needed with maxConcurrent=10)
      await waitFor(() => {
        const s = customPrefetcher.getStats();
        return s.queued === 0;
      });

      const stats = customPrefetcher.getStats();

      // With maxConcurrent=10, all 6 should fit
      expect(stats.inFlight).toBeLessThanOrEqual(10);
      expect(stats.queued).toBe(0);
    });
  });

  describe('Error Handling', () => {
    it('should propagate Missing results without throwing', async () => {
      const errorMockStore = {
        getResult: vi.fn().mockResolvedValue({ ok: false, error: { kind: 'Missing' } }),
        setPrefetcher: vi.fn(),
      };

      const errorPrefetcher = new ChunkPrefetcher(errorMockStore as any, {
        enabled: true,
      });

      // Register bounds so prefetcher generates adjacent chunks
      errorPrefetcher.registerArrayBounds('data', [10240, 10240], [1024, 1024]);

      // Should not throw
      expect(() => {
        errorPrefetcher.onAccess('data/1.1');
      }).not.toThrow();

      // Wait for all error responses to settle
      await waitFor(() => {
        const s = errorPrefetcher.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      // Should have attempted prefetch
      expect(errorMockStore.getResult).toHaveBeenCalled();
    });
  });

  describe('Cascade prevention', () => {
    // Pre-fix, MultiLevelCachingStore.getResult() unconditionally
    // called prefetcher.onAccess(key) — including when the request
    // ITSELF originated from prefetcher.processQueue(). Result: a
    // demand for K enqueues K-1/K+1; the prefetcher fetches K+1 via
    // store.getResult(K+1); the store calls onAccess(K+1) and
    // enqueues K+2; etc. The chain terminates only when MAX_SEEN_SIZE
    // (10 000) is reached.
    //
    // Fix: prefetcher.processQueue() now passes
    // { suppressPrefetch: true } so getResult() skips the onAccess
    // callback for prefetch-originated reads.
    it('processQueue passes suppressPrefetch=true to store.getResult', async () => {
      // 1D bounds: 10 chunks → demand at chunk 5 enqueues 4 and 6.
      prefetcher.registerArrayBounds('cascade', [10240], [1024]);

      const cascadeStore = new MockStore();
      const cascadePrefetcher = new ChunkPrefetcher(cascadeStore as any, {
        enabled: true,
        maxConcurrent: 4,
      });
      cascadePrefetcher.registerArrayBounds('cascade', [10240], [1024]);

      cascadePrefetcher.onAccess('cascade/5');

      await waitFor(() => {
        const s = cascadePrefetcher.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      // The 2 prefetched calls (4 and 6) must arrive with the
      // suppressPrefetch flag.
      const prefetchCalls = cascadeStore.getResult.mock.calls.filter(
        (call: unknown[]) => call[0] === 'cascade/4' || call[0] === 'cascade/6'
      );
      expect(prefetchCalls.length).toBe(2);
      for (const [, options] of prefetchCalls) {
        expect((options as { suppressPrefetch?: boolean })?.suppressPrefetch).toBe(true);
      }
    });

    it('does not fan out beyond immediate neighbors (cascade-simulating store)', async () => {
      // Simulate the real store's behavior: getResult internally
      // calls prefetcher.onAccess UNLESS suppressPrefetch is set.
      // Pre-fix this would walk: 5 → {4, 6} → {3, 5, 7} → {2, 4, 6, 8} ...
      // Post-fix: 5 → {4, 6}. Done.
      let cascadePrefetcher: ChunkPrefetcher | undefined;
      const cascadingStore = {
        getResult: vi.fn(
          (
            key: string,
            options?: { suppressPrefetch?: boolean }
          ): Promise<{ ok: boolean; value: Uint8Array }> => {
            if (!options?.suppressPrefetch) {
              cascadePrefetcher?.onAccess(key);
            }
            return Promise.resolve({ ok: true, value: new Uint8Array([1, 2, 3]) });
          }
        ),
        setPrefetcher: vi.fn(),
      };

      cascadePrefetcher = new ChunkPrefetcher(cascadingStore as any, {
        enabled: true,
        maxConcurrent: 4,
      });
      cascadePrefetcher.registerArrayBounds('cascade', [10240], [1024]);

      cascadePrefetcher.onAccess('cascade/5');

      await waitFor(() => {
        const s = cascadePrefetcher!.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      const fetchedKeys = cascadingStore.getResult.mock.calls.map(
        (call: unknown[]) => call[0] as string
      );
      // Only the two immediate neighbors should be fetched. No 3, 7,
      // 2, 8 would mean cascade got blocked at depth 1.
      expect(new Set(fetchedKeys)).toEqual(new Set(['cascade/4', 'cascade/6']));
      expect(fetchedKeys).not.toContain('cascade/3');
      expect(fetchedKeys).not.toContain('cascade/7');
    });
  });

  describe('Statistics', () => {
    // workers.md O3 / cache.md W21 [P2]: audit-id moved to comment per Phase E55.
    it('should return accurate statistics', async () => {
      // [cache.md/W21][P2] Previous version conflated `queued + inFlight === 4`
      // into a single sum. That assertion is satisfied by *any* split of the 4
      // neighbors between the two counters — including a regression that puts
      // all 4 into one bucket. Strengthen by pinning each counter exactly,
      // using a blocking mock so the dispatch state is observable.
      //
      // With maxConcurrent=2 and a 4-neighbor onAccess, the deterministic
      // split is exactly 2 inFlight (the dispatched ones) + 2 queued.
      const resolvers: Array<(value: { ok: true; value: Uint8Array }) => void> = [];
      const slowMockStore = {
        getResult: vi.fn().mockImplementation(
          () =>
            new Promise<{ ok: true; value: Uint8Array }>((resolve) => {
              resolvers.push((v) => resolve(v));
            })
        ),
        setPrefetcher: vi.fn(),
      };
      const statsPrefetcher = new ChunkPrefetcher(slowMockStore as any, {
        enabled: true,
        maxConcurrent: 2,
      });
      statsPrefetcher.registerArrayBounds('data', [10240, 10240], [1024, 1024]);

      statsPrefetcher.onAccess('data/1.1'); // 4 neighbors

      // Wait until processQueue has dispatched as many as it can.
      await waitFor(() => slowMockStore.getResult.mock.calls.length >= 2);

      const stats1 = statsPrefetcher.getStats();
      expect(stats1.enabled).toBe(true);
      // Exact split, not just a sum: 2 dispatched + 2 awaiting a free slot.
      expect(stats1.inFlight).toBe(2);
      expect(stats1.queued).toBe(2);

      // Resolve all four fetches so the queue drains.
      resolvers.forEach((r) => r({ ok: true, value: new Uint8Array([1]) }));
      // Allow chained .finally() ⇒ processQueue dispatches the remaining 2.
      await waitFor(() => slowMockStore.getResult.mock.calls.length >= 4);
      // Resolve the second wave too.
      resolvers.slice(2).forEach((r) => r({ ok: true, value: new Uint8Array([1]) }));

      await waitFor(() => {
        const s = statsPrefetcher.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      const stats2 = statsPrefetcher.getStats();
      expect(stats2.queued).toBe(0);
      expect(stats2.inFlight).toBe(0);
    });
  });

  describe('Dispose lifecycle', () => {
    it('queued chunks are not fetched after dispose()', async () => {
      // Build a store whose getResult never resolves so the queue stays
      // populated until dispose() runs. After dispose, no further
      // getResult calls should be initiated.
      const blocking = new MockStore();
      let resolve: (v: { ok: true; value: Uint8Array }) => void = () => {};
      blocking.getResult = vi.fn(
        () =>
          new Promise((r) => {
            resolve = r;
          })
      ) as unknown as typeof blocking.getResult;

      const localPrefetcher = new ChunkPrefetcher(blocking as any, {
        enabled: true,
        maxConcurrent: 1,
      });
      localPrefetcher.registerArrayBounds('data', [10240, 10240], [1024, 1024]);
      localPrefetcher.onAccess('data/1.1');
      // Yield once so processQueue dispatches the first fetch.
      await new Promise((r) => setTimeout(r, 0));
      const callsBeforeDispose = (blocking.getResult as unknown as { mock: { calls: unknown[] } })
        .mock.calls.length;
      expect(callsBeforeDispose).toBeGreaterThanOrEqual(1);

      localPrefetcher.dispose();
      // Resolve the in-flight promise so its `.finally()` runs after dispose.
      resolve({ ok: true, value: new Uint8Array([1, 2, 3]) });
      // Yield through microtasks to let .finally() complete.
      await new Promise((r) => setTimeout(r, 10));

      const callsAfterDispose = (blocking.getResult as unknown as { mock: { calls: unknown[] } })
        .mock.calls.length;
      // The .finally() must NOT re-enter processQueue and dispatch the
      // remaining queued chunks.
      expect(callsAfterDispose).toBe(callsBeforeDispose);
      expect(localPrefetcher.getStats().enabled).toBe(false);
    });

    it('onAccess after dispose() is a no-op', () => {
      prefetcher.dispose();
      const callsBefore = (mockStore.getResult as unknown as { mock: { calls: unknown[] } }).mock
        .calls.length;
      prefetcher.onAccess('points/positions/1.1.1');
      const callsAfter = (mockStore.getResult as unknown as { mock: { calls: unknown[] } }).mock
        .calls.length;
      expect(callsAfter).toBe(callsBefore);
      expect(prefetcher.getStats().queued).toBe(0);
    });

    it('priority queue: high-priority entries dispatch before remaining normal-priority ones (commit 8.3)', async () => {
      // Block dispatch to inspect ordering. With maxConcurrent=1 the
      // first normal entry occupies the slot; remaining entries sit
      // in their tier queues until the slot frees. When it does,
      // processQueue must drain highQueue before normalQueue.
      const blocking = new MockStore();
      const dispatched: string[] = [];
      const resolvers: Array<() => void> = [];
      blocking.getResult = vi.fn((key: string) => {
        dispatched.push(key);
        return new Promise((resolve) => {
          resolvers.push(() => resolve({ ok: true as const, value: new Uint8Array([1]) }));
        });
      }) as unknown as typeof blocking.getResult;

      const local = new ChunkPrefetcher(blocking as any, {
        enabled: true,
        maxConcurrent: 1,
      });
      // Enqueue normal first so it occupies the only dispatch slot.
      local.enqueueWithPriority(['n1', 'n2', 'n3'], 'normal');
      // Yield so the first normal entry actually dispatches.
      await new Promise((r) => setTimeout(r, 0));
      expect(dispatched).toEqual(['n1']);

      // Now add high-priority entries while n2/n3 still sit in normalQueue.
      local.enqueueWithPriority(['h1', 'h2'], 'high');
      // n1 still in flight; n2/n3 in normalQueue; h1/h2 in highQueue.
      const stats = local.getStats();
      expect(stats.queuedHigh).toBe(2);
      expect(stats.queuedNormal).toBe(2);

      // Release each fetch in order; verify high entries dispatch
      // before the remaining normal ones.
      resolvers[0]();
      await new Promise((r) => setTimeout(r, 0));
      expect(dispatched[1]).toBe('h1');
      resolvers[1]();
      await new Promise((r) => setTimeout(r, 0));
      expect(dispatched[2]).toBe('h2');
      resolvers[2]();
      await new Promise((r) => setTimeout(r, 0));
      // After the highs drain, remaining normals get their turn.
      expect(dispatched.slice(3).sort()).toEqual(['n2'].sort());

      // Drain remaining.
      while (resolvers.length > dispatched.length) {
        // No-op; we just want all the dispatched ones to settle.
        break;
      }
      for (const r of resolvers) r();

      local.dispose();
    });

    it('priority queue: normal-then-high re-enqueue promotes the entry (commit 8.3)', async () => {
      const blocking = new MockStore();
      blocking.getResult = vi.fn(
        () => new Promise(() => {})
      ) as unknown as typeof blocking.getResult;
      const local = new ChunkPrefetcher(blocking as any, {
        enabled: true,
        maxConcurrent: 0, // Block dispatch entirely so we can read tier state.
      });
      local.enqueueWithPriority(['k'], 'normal');
      expect(local.getStats().queuedNormal).toBe(1);
      expect(local.getStats().queuedHigh).toBe(0);
      // Promote.
      local.enqueueWithPriority(['k'], 'high');
      expect(local.getStats().queuedHigh).toBe(1);
      expect(local.getStats().queuedNormal).toBe(0);
      local.dispose();
    });

    // workers.md O3 / cache.md G15 [P5]: audit-id moved to comment per Phase E55.
    it('enqueueWithPriority with an empty iterable does NOT trigger processQueue', () => {
      // [cache.md/G15][P5] Source guards `if (added > 0) processQueue()`
      // at line 158 — but the early-return for the empty-iterable case
      // and the `enabled || isDisposed` guard at line 153 had no test pin.
      // Pin: an empty input must not trigger a fetch.
      const local = new ChunkPrefetcher(mockStore as any, { enabled: true });
      mockStore.getResult.mockClear();
      local.enqueueWithPriority([], 'high');
      local.enqueueWithPriority([], 'normal');
      expect(mockStore.getResult).not.toHaveBeenCalled();
      expect(local.getStats().queuedHigh).toBe(0);
      expect(local.getStats().queuedNormal).toBe(0);
      local.dispose();
    });

    // workers.md O3 / cache.md G15 [P5]: audit-id moved to comment per Phase E55.
    it('enqueueWithPriority after dispose() is a no-op', () => {
      // [cache.md/G15][P5] The `if (!this.enabled || this.isDisposed) return`
      // guard had no test pin. Pin: post-dispose enqueue must not enqueue,
      // not dispatch, and not throw.
      const local = new ChunkPrefetcher(mockStore as any, { enabled: true });
      local.dispose();
      mockStore.getResult.mockClear();
      expect(() => local.enqueueWithPriority(['k'], 'high')).not.toThrow();
      expect(() => local.enqueueWithPriority(['k'], 'normal')).not.toThrow();
      expect(mockStore.getResult).not.toHaveBeenCalled();
      expect(local.getStats().queuedHigh).toBe(0);
      expect(local.getStats().queuedNormal).toBe(0);
    });

    // workers.md O3 / cache.md G4 [P5]: audit-id moved to comment per Phase E55.
    it('evicts half of `seen` when MAX_SEEN_SIZE is exceeded', async () => {
      // [cache.md/G4][P5] Source line 106-112 trims half the `seen` set when
      // `seen.size > MAX_SEEN_SIZE` (10 000). This significant code path had
      // no behavioral test — mutating the eviction count to a no-op would
      // let the set grow unboundedly. Pin the contract via the observable
      // invariant: re-feeding a previously-seen key after eviction triggers
      // a fresh fetch (proves that key was forgotten).
      //
      // Use a 1-D bounds so each onAccess only triggers ±1 neighbor
      // dispatch (cheap), and a non-blocking mock so the queue drains
      // immediately. Register a span large enough to hold MAX_SEEN_SIZE
      // distinct keys.
      const fastStore = new MockStore();
      const local = new ChunkPrefetcher(fastStore as any, {
        enabled: true,
        maxConcurrent: 16,
      });
      // 1-D: 12 000 chunks, each 1024 bytes → keys data/0 .. data/11999.
      local.registerArrayBounds('data', [12000 * 1024], [1024]);

      // Touch 10 001 distinct keys → exceeds MAX_SEEN_SIZE (10 000) on the
      // 10 001-st insertion. Source evicts floor(10 000/2) = 5 000 oldest
      // entries from the head of the iteration order (Set insertion order).
      for (let i = 0; i < 10_001; i++) {
        local.onAccess(`data/${i}`);
      }

      // Drain any in-flight dispatches before counting (mockStore.getResult
      // resolves synchronously enough that this is a no-op in practice).
      await waitFor(() => {
        const s = local.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      const callsBefore = fastStore.getResult.mock.calls.length;

      // Key `data/0` was inserted FIRST → it must have been in the evicted
      // half. Re-accessing it now triggers a *fresh* neighbor fan-out
      // (mutation that disables the trim leaves the key in `seen` and
      // the early-return at line 100 would prevent any new fetches).
      fastStore.getResult.mockClear();
      local.onAccess('data/0');
      await waitFor(() => {
        const s = local.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      // After re-access, at least 1 neighbor fetch must have fired for
      // `data/0` (the +1 neighbor `data/1` — and possibly its parsedCache
      // re-entry). If the trim never ran, the seen-set early-return would
      // produce 0 calls.
      expect(fastStore.getResult.mock.calls.length).toBeGreaterThan(0);

      // Sanity: data/10000 (the most-recent prior access) is still in `seen`,
      // so re-accessing it triggers NO new fetches.
      fastStore.getResult.mockClear();
      local.onAccess('data/10000');
      await waitFor(() => {
        const s = local.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });
      expect(fastStore.getResult.mock.calls.length).toBe(0);

      // Total dispatches across the burst should be finite/bounded — not
      // a runaway cascade.
      expect(callsBefore).toBeGreaterThan(0);

      local.dispose();
    });

    it(
      'in-flight prefetch.finally does not re-enter processQueue after dispose',
      { timeout: 5_000 },
      async () => {
        // Simulates the production race: prefetcher dispatches N fetches,
        // dispose() runs while fetches are in flight, fetches resolve, the
        // .finally() handlers must NOT dispatch any further fetches against
        // the disposed store. Locks in the isDisposed guard at the
        // continuation site, not just the entry-point.
        const slow = new MockStore();
        const resolvers: Array<(v: { ok: true; value: Uint8Array }) => void> = [];
        slow.getResult = vi.fn(
          () =>
            new Promise((r) => {
              resolvers.push(r);
            })
        ) as unknown as typeof slow.getResult;

        const local = new ChunkPrefetcher(slow as any, {
          enabled: true,
          maxConcurrent: 4,
        });
        local.registerArrayBounds('data', [10240, 10240], [1024, 1024]);

        // Trigger four neighbors → all four go in-flight at maxConcurrent=4.
        local.onAccess('data/1.1');
        // Yield so processQueue dispatches.
        await new Promise((r) => setTimeout(r, 0));
        const callsAtDispatch = (slow.getResult as unknown as { mock: { calls: unknown[] } }).mock
          .calls.length;
        expect(callsAtDispatch).toBe(4);

        local.dispose();

        // Resolve all four promises so their .finally() handlers run.
        for (const resolve of resolvers) {
          resolve({ ok: true, value: new Uint8Array([1]) });
        }
        // Yield through several microtask turns to let any (incorrect) recursion fire.
        for (let i = 0; i < 5; i++) {
          await new Promise((r) => setTimeout(r, 0));
        }

        const callsAfterFinally = (slow.getResult as unknown as { mock: { calls: unknown[] } }).mock
          .calls.length;
        // The .finally() must not re-enter processQueue and add new dispatches.
        expect(callsAfterFinally).toBe(callsAtDispatch);
      }
    );
  });
});

describe('ChunkPrefetcher - Integration Tests', () => {
  it('should integrate with MultiLevelCachingStore via setPrefetcher', () => {
    // Create mock store (don't need real initialization)
    const mockIntegrationStore = {
      getResult: vi.fn().mockResolvedValue({ ok: true, value: new Uint8Array([1, 2, 3]) }),
      setPrefetcher: vi.fn(),
      prefetcher: null as any,
    };

    // Implement setPrefetcher to mimic real store (use arrow function to avoid 'this' issues)
    mockIntegrationStore.setPrefetcher = vi.fn((p: any) => {
      mockIntegrationStore.prefetcher = p;
    });

    // Create and attach prefetcher
    const prefetcher = new ChunkPrefetcher(mockIntegrationStore as any, {
      enabled: true,
    });

    mockIntegrationStore.setPrefetcher(prefetcher);

    // Verify connection
    expect(mockIntegrationStore.prefetcher).toBe(prefetcher);

    // Dispose (simulate store.dispose())
    mockIntegrationStore.prefetcher = null;
    expect(mockIntegrationStore.prefetcher).toBeNull();
  });

  it('should be triggered by store when onAccess is called', async () => {
    const mockIntegrationStore = {
      getResult: vi.fn().mockResolvedValue({ ok: true, value: new Uint8Array([1, 2, 3]) }),
    };

    // Create prefetcher with spy-able onAccess
    const prefetcher = new ChunkPrefetcher(mockIntegrationStore as any, {
      enabled: true,
    });

    const onAccessSpy = vi.spyOn(prefetcher, 'onAccess');

    // Register bounds so prefetcher knows valid chunk range
    prefetcher.registerArrayBounds('test', [2048, 4], [1024, 4]);

    // Simulate store calling onAccess after L2/L3 hit
    prefetcher.onAccess('test/0.0');

    // Verify onAccess was called
    expect(onAccessSpy).toHaveBeenCalledWith('test/0.0');

    // Wait for prefetch to attempt store.getResult()
    await waitFor(() => mockIntegrationStore.getResult.mock.calls.length > 0);

    // Verify prefetcher tried to fetch chunks
    expect(mockIntegrationStore.getResult).toHaveBeenCalled();
  });
});
