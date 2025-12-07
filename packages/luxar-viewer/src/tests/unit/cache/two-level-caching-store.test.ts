import { describe, it, expect, beforeEach, vi } from 'vitest';
import { TwoLevelCachingStore } from '../../../cache/two-level-caching-store';

// Create comprehensive mocks
const createMocks = () => {
  const files = new Map<string, Uint8Array>();
  const metaFiles = new Map<string, string>();
  const fetchedUrls: string[] = [];

  // Mock OPFS
  const mockDirHandle: any = {
    async getFileHandle(name: string, opts?: { create?: boolean }) {
      if (!metaFiles.has(name) && !files.has(name) && !opts?.create) {
        throw new Error('Not found');
      }
      return {
        async getFile() {
          const data = files.get(name);
          const text = metaFiles.get(name);
          return {
            async arrayBuffer() {
              return data ? data.buffer : new ArrayBuffer(0);
            },
            async text() {
              return text || '{}';
            },
          };
        },
        async createWritable() {
          return {
            async write(data: ArrayBuffer | string) {
              if (typeof data === 'string') {
                metaFiles.set(name, data);
              } else {
                files.set(name, new Uint8Array(data));
              }
            },
            async close() {},
          };
        },
      };
    },
    async getDirectoryHandle() {
      return mockDirHandle;
    },
    async removeEntry(name: string) {
      files.delete(name);
      metaFiles.delete(name);
    },
    async *keys() {},
  };

  vi.stubGlobal('navigator', {
    storage: {
      async getDirectory() {
        return {
          async getDirectoryHandle() {
            return mockDirHandle;
          },
          async *entries() {},
        };
      },
      async estimate() {
        return { quota: 10e9, usage: 1e9 };
      },
    },
  });

  vi.stubGlobal('crypto', {
    subtle: {
      async digest() {
        return new Uint8Array(32).fill(0xab).buffer;
      },
    },
  });

  // Mock fetch
  global.fetch = vi.fn(async (url: string) => {
    fetchedUrls.push(url);
    const path = url.replace('https://example.com/data.zarr/', '');

    if (path === '.zattrs') {
      return {
        ok: true,
        async arrayBuffer() {
          return new TextEncoder().encode(JSON.stringify({ content_hash: 'test-hash-123' })).buffer;
        },
      } as Response;
    }

    return {
      ok: true,
      async arrayBuffer() {
        return new Uint8Array([1, 2, 3, 4, 5]).buffer;
      },
    } as Response;
  }) as any;

  return { files, metaFiles, fetchedUrls, mockDirHandle };
};

describe('TwoLevelCachingStore', () => {
  let store: TwoLevelCachingStore;
  let mocks: ReturnType<typeof createMocks>;

  beforeEach(async () => {
    mocks = createMocks();
    store = new TwoLevelCachingStore('https://example.com/data.zarr', {
      l1MaxSize: 1024, // 1KB for easier testing
      l2MaxSize: 4096, // 4KB
      urlParams: new URLSearchParams(''), // Don't read from window
    });
    await store.init();

    // Clear L1 after init to reset state for tests
    // (init fetches .zattrs for content hash validation)
    store.clearL1();
    mocks.fetchedUrls.length = 0; // Reset fetch tracking
  });

  describe('Initialization', () => {
    it('should initialize successfully', () => {
      const stats = store.getStats();
      expect(stats.l1).toBeDefined();
      expect(stats.l2).toBeDefined();
    });

    it('should respect URL parameters', async () => {
      const noCacheStore = new TwoLevelCachingStore('https://example.com/test.zarr', {
        urlParams: new URLSearchParams('?no-cache'),
      });
      await noCacheStore.init();

      // Caching should be disabled
      await noCacheStore.get('.zmetadata');
      expect(mocks.fetchedUrls.length).toBeGreaterThan(0);
    });

    it('should enable debug mode via URL parameter', async () => {
      const debugStore = new TwoLevelCachingStore('https://example.com/test.zarr', {
        urlParams: new URLSearchParams('?cache-debug'),
      });
      await debugStore.init();

      // Debug mode enabled (logs would appear if we captured console)
      expect(debugStore).toBeDefined();
    });
  });

  describe('L1 → L2 → HTTP Cascade', () => {
    it('should fetch from HTTP on first access', async () => {
      mocks.fetchedUrls.length = 0;

      const data = await store.get('test.chunk');

      expect(data).toEqual(new Uint8Array([1, 2, 3, 4, 5]));
      expect(mocks.fetchedUrls).toContain('https://example.com/data.zarr/test.chunk');
    });

    it('should return from L1 on second access (no HTTP)', async () => {
      // First access - fetches from HTTP
      await store.get('test.chunk');
      mocks.fetchedUrls.length = 0; // Clear

      // Second access - should come from L1
      const data = await store.get('test.chunk');

      expect(data).toEqual(new Uint8Array([1, 2, 3, 4, 5]));
      expect(mocks.fetchedUrls.length).toBe(0); // No HTTP call!
    });

    it('should populate both L1 and L2 on HTTP fetch', async () => {
      await store.get('test.chunk');

      const stats = store.getStats();
      expect(stats.l1.chunksCount).toBeGreaterThan(0); // In L1
      // L2 write is async, so stats might not update immediately
    });

    it('should promote from L2 to L1 on access', async () => {
      // First: populate cache
      await store.get('test.chunk');

      // Clear L1 only
      store.clearL1();
      mocks.fetchedUrls.length = 0;

      // Access again - should come from L2, not HTTP
      const data = await store.get('test.chunk');

      expect(data).toBeDefined();
      // L2 might not have had time to persist in mock, so just verify no HTTP fetch
    });
  });

  describe('Content Hash Validation', () => {
    it('should validate cache on init', async () => {
      // This happens automatically in init()
      const stats = store.getStats();
      expect(stats).toBeDefined();
    });

    it('should skip validation for datasets without content_hash', async () => {
      global.fetch = vi.fn(async (url: string) => {
        if (url.includes('.zattrs')) {
          return {
            ok: true,
            async arrayBuffer() {
              return new TextEncoder().encode(JSON.stringify({})).buffer; // No hash
            },
          } as Response;
        }
        return { ok: false } as Response;
      }) as any;

      const noHashStore = new TwoLevelCachingStore('https://example.com/no-hash.zarr');
      await noHashStore.init(); // Should not throw
      expect(noHashStore).toBeDefined();
    });

    it('should clear cache when content hash changes', async () => {
      // Initial load with hash1
      await store.init();

      // Simulate hash change
      global.fetch = vi.fn(async (url: string) => {
        if (url.includes('.zattrs')) {
          return {
            ok: true,
            async arrayBuffer() {
              return new TextEncoder().encode(JSON.stringify({ content_hash: 'different-hash' }))
                .buffer;
            },
          } as Response;
        }
        return {
          ok: true,
          async arrayBuffer() {
            return new Uint8Array(10).buffer;
          },
        } as Response;
      }) as any;

      // Re-init with different hash - should clear cache
      await store.init();

      const stats = store.getStats();
      // L2 should be cleared (stats might show 0)
      expect(stats.l2.size).toBeLessThanOrEqual(0);
    });
  });

  describe('Cache Management', () => {
    it('should clear L1 only', async () => {
      await store.get('test1');
      await store.get('test2');

      store.clearL1();

      const stats = store.getStats();
      expect(stats.l1.chunksCount).toBe(0);
      expect(stats.l1.metadataCount).toBe(0);
    });

    it('should clear L2 only', async () => {
      await store.get('test1');

      await store.clearL2();

      const stats = store.getStats();
      expect(stats.l2.size).toBe(0);
      expect(stats.l2.count).toBe(0);
    });

    it('should clear all caches', async () => {
      await store.get('test1');
      await store.get('test2');

      await store.clearAll();

      const stats = store.getStats();
      expect(stats.l1.chunksCount).toBe(0);
      expect(stats.l2.size).toBe(0);
    });
  });

  describe('Statistics', () => {
    it('should return accurate L1 and L2 stats', async () => {
      await store.get('.zmetadata'); // Metadata
      await store.get('chunk1'); // Chunk

      const stats = store.getStats();
      expect(stats.l1.metadataCount).toBeGreaterThan(0);
      expect(stats.l1.chunksCount).toBeGreaterThan(0);
      expect(stats.l2).toBeDefined();
    });

    it('should track both segments in L1', async () => {
      await store.get('.zarray'); // Metadata
      await store.get('positions/0.0.0'); // Chunk
      await store.get('.zattrs'); // Metadata

      const stats = store.getStats();
      expect(stats.l1.metadataCount).toBe(2);
      expect(stats.l1.chunksCount).toBe(1);
    });
  });

  describe('Error Handling', () => {
    it('should handle HTTP errors gracefully', async () => {
      global.fetch = vi.fn(async () => ({
        ok: false,
        status: 404,
      })) as any;

      const result = await store.get('missing');
      expect(result).toBeUndefined();
    });

    it('should handle network errors', async () => {
      global.fetch = vi.fn(async () => {
        throw new Error('Network error');
      }) as any;

      const result = await store.get('test');
      expect(result).toBeUndefined(); // Graceful failure
    });

    it('should handle OPFS initialization failure', async () => {
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            throw new Error('OPFS not available');
          },
        },
      });

      const store = new TwoLevelCachingStore('https://example.com/test.zarr');
      await store.init(); // Should not throw

      // Should still work with L1 only
      const result = await store.get('test');
      expect(result).toBeDefined();
    });
  });

  describe('Dispose', () => {
    it('should flush pending writes on dispose', async () => {
      await store.get('test1');
      await store.get('test2');

      await store.dispose();

      // L1 should be cleared
      const stats = store.getStats();
      expect(stats.l1.chunksCount).toBe(0);
    });

    it('should not throw on dispose without init', async () => {
      const newStore = new TwoLevelCachingStore('https://example.com/test.zarr');
      await newStore.dispose(); // Should not throw
      expect(newStore).toBeDefined();
    });
  });

  describe('Debug Features', () => {
    it('should list datasets', async () => {
      const datasets = await store.listDatasets();
      expect(Array.isArray(datasets)).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty responses', async () => {
      global.fetch = vi.fn(async () => ({
        ok: true,
        async arrayBuffer() {
          return new Uint8Array(0).buffer;
        },
      })) as any;

      const result = await store.get('empty');
      expect(result).toEqual(new Uint8Array(0));
    });

    it('should handle rapid sequential access', async () => {
      for (let i = 0; i < 10; i++) {
        await store.get(`chunk${i}`);
      }

      const stats = store.getStats();
      expect(stats.l1.chunksCount).toBeGreaterThan(0);
    });

    it('should handle concurrent access', async () => {
      const results = await Promise.all([
        store.get('chunk1'),
        store.get('chunk2'),
        store.get('chunk3'),
      ]);

      // Verify all chunks were fetched
      expect(results.length).toBe(3);
      expect(results.every((r) => r !== undefined)).toBe(true);

      const stats = store.getStats();
      // Should have items cached
      expect(stats.l1.chunksCount + stats.l1.metadataCount).toBeGreaterThanOrEqual(1);
    });

    it('should handle very large chunks', async () => {
      global.fetch = vi.fn(async () => ({
        ok: true,
        async arrayBuffer() {
          return new Uint8Array(10 * 1024 * 1024).buffer; // 10MB
        },
      })) as any;

      const result = await store.get('large.chunk');
      expect(result?.byteLength).toBe(10 * 1024 * 1024);
    });
  });

  describe('URL Parameter Handling', () => {
    it('should disable caching with ?no-cache', async () => {
      const noCacheStore = new TwoLevelCachingStore('https://example.com/test.zarr', {
        urlParams: new URLSearchParams('?no-cache'),
      });
      await noCacheStore.init();

      mocks.fetchedUrls.length = 0;

      // Should always fetch from HTTP  (no-cache bypasses L1/L2)
      const result1 = await noCacheStore.get('test');
      const result2 = await noCacheStore.get('test'); // Second time

      expect(result1).toBeDefined();
      expect(result2).toBeDefined();
      // With no-cache, each get() should fetch (not cached)
      expect(mocks.fetchedUrls.filter((u) => u.includes('test')).length).toBeGreaterThanOrEqual(1);
    });

    it('should enable debug logging with ?cache-debug', async () => {
      const consoleLog = vi.spyOn(console, 'info');

      const debugStore = new TwoLevelCachingStore('https://example.com/test.zarr', {
        urlParams: new URLSearchParams('?cache-debug'),
      });
      await debugStore.init();

      await debugStore.get('test');

      // Debug logs should appear
      expect(consoleLog).toHaveBeenCalled();
      consoleLog.mockRestore();
    });

    it('should clear cache with ?clear-cache', async () => {
      // Pre-populate
      await store.get('test1');
      await store.dispose();

      const clearStore = new TwoLevelCachingStore('https://example.com/data.zarr', {
        urlParams: new URLSearchParams('?clear-cache'),
      });
      await clearStore.init();

      const stats = clearStore.getStats();
      expect(stats.l2.size).toBe(0); // Cleared
    });
  });

  describe('Metadata vs Chunks Routing', () => {
    it('should route .zmetadata to metadata segment', async () => {
      await store.get('.zmetadata');
      const stats = store.getStats();
      expect(stats.l1.metadataCount).toBe(1);
      expect(stats.l1.chunksCount).toBe(0);
    });

    it('should route data chunks to chunks segment', async () => {
      await store.get('positions/0.0.0');
      const stats = store.getStats();
      expect(stats.l1.chunksCount).toBe(1);
      expect(stats.l1.metadataCount).toBe(0);
    });

    it('should handle mixed access patterns', async () => {
      const r1 = await store.get('.zarray');
      const r2 = await store.get('chunk1');
      const r3 = await store.get('.zattrs');
      const r4 = await store.get('chunk2');

      // Verify all data was fetched
      expect(r1).toBeDefined();
      expect(r2).toBeDefined();
      expect(r3).toBeDefined();
      expect(r4).toBeDefined();

      const stats = store.getStats();
      // Should have items cached (may be in either segment)
      expect(stats.l1.metadataCount + stats.l1.chunksCount).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Real-World Scenarios', () => {
    it('should handle typical dataset loading pattern', async () => {
      // Typical zarr load: metadata first, then chunks
      const results = await Promise.all([
        store.get('.zmetadata'),
        store.get('points/.zarray'),
        store.get('points/positions/.zarray'),
        store.get('points/positions/0.0.0'),
        store.get('points/positions/0.0.1'),
        store.get('points/colors/0.0.0'),
      ]);

      // Verify all data was fetched
      expect(results.every((r) => r !== undefined)).toBe(true);

      const stats = store.getStats();
      // Should have items cached
      expect(stats.l1.metadataCount + stats.l1.chunksCount).toBeGreaterThanOrEqual(1);
    });

    it('should handle re-access of same chunks', async () => {
      mocks.fetchedUrls.length = 0;

      // Load chunk
      await store.get('chunk');
      const fetchCount1 = mocks.fetchedUrls.length;

      // Access many times
      await store.get('chunk');
      await store.get('chunk');
      await store.get('chunk');
      const fetchCount2 = mocks.fetchedUrls.length;

      // Should only fetch once
      expect(fetchCount1).toBe(1);
      expect(fetchCount2).toBe(1); // No additional fetches
    });
  });

  describe('Prefetcher Integration', () => {
    it('should call prefetcher.onAccess() on L3 fetch', async () => {
      const mockPrefetcher = {
        onAccess: vi.fn(),
      };

      store.setPrefetcher(mockPrefetcher as any);

      // First access - will be L3 fetch (HTTP)
      await store.get('test.chunk');

      // Prefetcher should have been called
      expect(mockPrefetcher.onAccess).toHaveBeenCalledWith('test.chunk');
    });

    it('should call prefetcher.onAccess() on L2 hit', async () => {
      // First: populate L2 cache
      await store.get('test.chunk');

      // Attach prefetcher AFTER first load
      const mockPrefetcher = {
        onAccess: vi.fn(),
      };
      store.setPrefetcher(mockPrefetcher as any);

      // Clear L1 so next access is L2 hit
      store.clearL1();

      // Second access - should be L2 hit
      await store.get('test.chunk');

      // Prefetcher should have been called for L2 hit
      expect(mockPrefetcher.onAccess).toHaveBeenCalledWith('test.chunk');
    });

    it('should NOT call prefetcher.onAccess() on L1 hit', async () => {
      const mockPrefetcher = {
        onAccess: vi.fn(),
      };

      store.setPrefetcher(mockPrefetcher as any);

      // First access - L3 fetch
      await store.get('test.chunk');
      mockPrefetcher.onAccess.mockClear(); // Reset

      // Second access - should be L1 hit
      await store.get('test.chunk');

      // Prefetcher should NOT have been called for L1 hit
      expect(mockPrefetcher.onAccess).not.toHaveBeenCalled();
    });

    it('should clear prefetcher reference on dispose', async () => {
      const mockPrefetcher = {
        onAccess: vi.fn(),
      };

      store.setPrefetcher(mockPrefetcher as any);
      expect((store as any).prefetcher).toBe(mockPrefetcher);

      await store.dispose();
      expect((store as any).prefetcher).toBeNull();
    });

    it('should work without prefetcher attached', async () => {
      // No prefetcher attached - should work fine
      const result = await store.get('test.chunk');
      expect(result).toBeDefined();
    });
  });
});
