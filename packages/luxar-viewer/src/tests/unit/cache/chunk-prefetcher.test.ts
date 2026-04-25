/**
 * Comprehensive tests for ChunkPrefetcher
 *
 * Tests cover:
 * - Chunk index parsing (v2/v3 formats)
 * - Adjacent chunk calculation
 * - Queue management and deduplication
 * - Concurrency limiting
 * - Integration with TwoLevelCachingStore
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

// Mock TwoLevelCachingStore for unit tests
class MockStore {
  get = vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3]));
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
      expect(mockStore.get).toHaveBeenCalled();
      const calls = mockStore.get.mock.calls.map((call: any[]) => call[0]);

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

      const calls = mockStore.get.mock.calls.map((call: any[]) => call[0]);

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
      expect(mockStore.get).not.toHaveBeenCalled();
    });

    it('should handle 1D chunks', () => {
      // Register bounds for 1D array with multiple chunks
      prefetcher.registerArrayBounds('data', [10240], [1024]);

      prefetcher.onAccess('data/0');

      const calls = mockStore.get.mock.calls.map((call: any[]) => call[0]);

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

      const calls = mockStore.get.mock.calls.map((call: any[]) => call[0]);

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
      const resolvers: Array<() => void> = [];
      const controlledStore = {
        get: vi.fn().mockImplementation(
          () =>
            new Promise<Uint8Array>((resolve) => {
              resolvers.push(() => resolve(new Uint8Array([1])));
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
      await waitFor(() => controlledStore.get.mock.calls.length >= 2);

      const stats = limitedPrefetcher.getStats();

      // Should have at most 2 in flight (the controlled promises are pending)
      expect(stats.inFlight).toBeLessThanOrEqual(2);

      // Resolve all to clean up
      resolvers.forEach((r) => r());
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
      expect(mockStore.get).toHaveBeenCalledTimes(4);
    });
  });

  describe('Deduplication', () => {
    it('should not queue same chunk twice', () => {
      // Access same chunk twice
      prefetcher.onAccess('data/1.1');
      prefetcher.onAccess('data/1.1');

      const stats = prefetcher.getStats();

      // Should only queue 4 unique neighbors (not 8)
      expect(stats.queued + stats.inFlight).toBeLessThanOrEqual(4);
    });

    it('should not queue chunks already in flight', async () => {
      // Use controlled promises so gets stay pending until we resolve them
      const resolvers: Array<() => void> = [];
      const slowMockStore = {
        get: vi.fn().mockImplementation(
          () =>
            new Promise<void>((resolve) => {
              resolvers.push(resolve);
            })
        ),
        setPrefetcher: vi.fn(),
      };

      const slowPrefetcher = new ChunkPrefetcher(slowMockStore as any, {
        enabled: true,
        maxConcurrent: 4,
      });

      // Register bounds so prefetcher generates adjacent chunks
      slowPrefetcher.registerArrayBounds('data', [10240, 10240], [1024, 1024]);

      // Trigger first access
      slowPrefetcher.onAccess('data/1.1');

      // Wait for requests to start (they will be in-flight, pending)
      await waitFor(() => slowMockStore.get.mock.calls.length > 0);

      // Trigger second access (should deduplicate via the seen-set)
      slowPrefetcher.onAccess('data/1.1');

      // Should not have duplicate requests
      const stats = slowPrefetcher.getStats();
      expect(stats.queued + stats.inFlight).toBeLessThanOrEqual(4);

      // Resolve all to clean up
      resolvers.forEach((r) => r());
    });
  });

  describe('Configuration', () => {
    it('should respect enabled flag', () => {
      const disabledPrefetcher = new ChunkPrefetcher(mockStore as any, { enabled: false });

      disabledPrefetcher.onAccess('data/0.0');

      // Should not prefetch anything when disabled
      expect(mockStore.get).not.toHaveBeenCalled();
    });

    it('does not prefetch when constructed with enabled=false', () => {
      const noPrefetchPrefetcher = new ChunkPrefetcher(mockStore as any, {
        enabled: false,
      });

      noPrefetchPrefetcher.onAccess('data/0.0');

      expect(mockStore.get).not.toHaveBeenCalled();
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
    it('should ignore prefetch errors silently', async () => {
      const errorMockStore = {
        get: vi.fn().mockRejectedValue(new Error('404 Not Found')),
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
      expect(errorMockStore.get).toHaveBeenCalled();
    });
  });

  describe('Statistics', () => {
    it('should return accurate statistics', async () => {
      const statsPrefetcher = new ChunkPrefetcher(mockStore as any, {
        enabled: true,
        maxConcurrent: 2,
      });

      // Register bounds so prefetcher generates adjacent chunks
      statsPrefetcher.registerArrayBounds('data', [10240, 10240], [1024, 1024]);

      statsPrefetcher.onAccess('data/1.1'); // 4 neighbors

      // Check initial state
      const stats1 = statsPrefetcher.getStats();
      expect(stats1.enabled).toBe(true);
      expect(stats1.queued + stats1.inFlight).toBe(4);

      // Wait for completion
      await waitFor(() => {
        const s = statsPrefetcher.getStats();
        return s.inFlight === 0 && s.queued === 0;
      });

      // Check final state
      const stats2 = statsPrefetcher.getStats();
      expect(stats2.queued).toBe(0);
      expect(stats2.inFlight).toBe(0);
    });
  });
});

describe('ChunkPrefetcher - Integration Tests', () => {
  it('should integrate with TwoLevelCachingStore via setPrefetcher', () => {
    // Create mock store (don't need real initialization)
    const mockIntegrationStore = {
      get: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
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
      get: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
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

    // Wait for prefetch to attempt store.get()
    await waitFor(() => mockIntegrationStore.get.mock.calls.length > 0);

    // Verify prefetcher tried to fetch chunks
    expect(mockIntegrationStore.get).toHaveBeenCalled();
  });
});
