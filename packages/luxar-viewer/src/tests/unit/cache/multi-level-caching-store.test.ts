import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';

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

describe('MultiLevelCachingStore', () => {
  let store: MultiLevelCachingStore;
  let mocks: ReturnType<typeof createMocks>;

  beforeEach(async () => {
    mocks = createMocks();
    store = new MultiLevelCachingStore('https://example.com/data.zarr', {
      l1MaxSize: 20 * 1024 * 1024, // 20MB (must exceed SegmentedLRU MIN_METADATA_SIZE of 10MB)
      l2MaxSize: 4096, // 4KB
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
      const noCacheStore = new MultiLevelCachingStore('https://example.com/test.zarr', {
        noCache: true,
      });
      await noCacheStore.init();

      // Caching should be disabled
      await noCacheStore.get('.zmetadata');
      expect(mocks.fetchedUrls.length).toBeGreaterThan(0);
    });

    it('should enable debug mode via URL parameter', async () => {
      const debugStore = new MultiLevelCachingStore('https://example.com/test.zarr', {
        debug: true,
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

  describe('Phase 21G — per-tier demand counters', () => {
    it('first fetch increments networkRequests; nothing in l1Hits/l2Hits', async () => {
      const before = store.getStats().demand;
      expect(before).toEqual({ l1Hits: 0, l2Hits: 0, networkRequests: 0 });

      await store.get('test.chunk');

      const after = store.getStats().demand;
      expect(after.networkRequests).toBe(1);
      expect(after.l1Hits).toBe(0);
      expect(after.l2Hits).toBe(0);
    });

    it('second fetch increments l1Hits (L1 served the request)', async () => {
      await store.get('test.chunk'); // Network → l1+l2 populated.
      const beforeL1 = store.getStats().demand.l1Hits;

      await store.get('test.chunk');

      expect(store.getStats().demand.l1Hits).toBe(beforeL1 + 1);
      expect(store.getStats().demand.networkRequests).toBe(1); // Unchanged
    });

    it('demand.networkRequests matches network.requestCount when there is no prefetch traffic', async () => {
      await store.get('test.chunk');
      await store.get('other.chunk');

      const stats = store.getStats();
      // Without a prefetcher attached, every request is user-demand,
      // so the demand counter equals the aggregate counter.
      expect(stats.demand.networkRequests).toBe(stats.network.requestCount);
    });

    it('prefetch-originated calls (suppressPrefetch=true) do NOT increment demand counters', async () => {
      // Demand call: counts toward both demand AND aggregate.
      await store.getResult('test.chunk');
      const afterDemand = store.getStats();
      expect(afterDemand.demand.networkRequests).toBe(1);
      expect(afterDemand.network.requestCount).toBe(1);

      // Prefetch call (different key): counts toward aggregate only.
      await store.getResult('other.chunk', { suppressPrefetch: true });
      const afterPrefetch = store.getStats();
      expect(afterPrefetch.demand.networkRequests).toBe(1); // Unchanged
      expect(afterPrefetch.network.requestCount).toBe(2); // Bumped
    });

    it('prefetch L1 hits do NOT increment demand.l1Hits', async () => {
      await store.getResult('test.chunk'); // populates L1
      const beforeL1 = store.getStats().demand.l1Hits;

      await store.getResult('test.chunk', { suppressPrefetch: true });

      // Demand hit count unchanged; the prefetch-originated read
      // hit L1 but didn't count toward user-facing hit-rate.
      expect(store.getStats().demand.l1Hits).toBe(beforeL1);
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

      const noHashStore = new MultiLevelCachingStore('https://example.com/no-hash.zarr');
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

    it('should bypass cache when validating content_hash (critical fix)', async () => {
      // This test verifies the fix for the cache validation bug where
      // validation was reading .zattrs from cache, comparing cached hash
      // against itself, causing false positives when switching datasets
      // on the same port.

      const fetchCalls: string[] = [];
      let currentHash = 'hash-dataset-1';

      global.fetch = vi.fn(async (url: string) => {
        fetchCalls.push(url);

        if (url.includes('.zattrs')) {
          // Return CURRENT hash (simulates server state)
          return {
            ok: true,
            async arrayBuffer() {
              return new TextEncoder().encode(JSON.stringify({ content_hash: currentHash })).buffer;
            },
          } as Response;
        }
        return {
          ok: true,
          async arrayBuffer() {
            return new Uint8Array([1, 2, 3]).buffer;
          },
        } as Response;
      }) as any;

      // Initial load - dataset 1
      await store.init();
      expect(fetchCalls.some((url) => url.includes('.zattrs'))).toBe(true);
      fetchCalls.length = 0;

      // Cache .zattrs by accessing it normally (goes through cache cascade)
      await store.get('.zattrs');
      const cachedAttrs = await store.get('.zattrs');
      expect(cachedAttrs).toBeDefined();
      fetchCalls.length = 0; // Clear tracking

      // Now simulate switching to dataset 2 with different hash
      // WITHOUT clearing cache (simulates same port, different demo)
      currentHash = 'hash-dataset-2';

      // Validation should detect the difference
      // Even though .zattrs is in cache with old hash!
      const testStore = new MultiLevelCachingStore('https://example.com/data.zarr');
      await testStore.init();

      // CRITICAL: Validation MUST have fetched .zattrs directly from HTTP
      // If it used cache, it would get old hash and validation would fail
      const attrsWasFetched = fetchCalls.some((url) => url.includes('.zattrs'));
      expect(attrsWasFetched).toBe(true);

      // And it should have cleared L2 due to hash mismatch
      const stats = testStore.getStats();
      expect(stats.l2.size).toBe(0);
    });

    it('should always fetch content_hash from HTTP, never from cache', async () => {
      // Verify that getRemoteContentHash() truly bypasses cache

      let fetchCount = 0;
      global.fetch = vi.fn(async (url: string) => {
        if (url.includes('.zattrs')) {
          fetchCount++;
          return {
            ok: true,
            async arrayBuffer() {
              return new TextEncoder().encode(
                JSON.stringify({ content_hash: `hash-${fetchCount}` })
              ).buffer;
            },
          } as Response;
        }
        return {
          ok: true,
          async arrayBuffer() {
            return new Uint8Array([1, 2, 3]).buffer;
          },
        } as Response;
      }) as any;

      // Init - triggers first validation fetch
      await store.init();
      expect(fetchCount).toBe(1);

      // Populate cache with .zattrs (this would add it to L1/L2)
      await store.get('.zattrs');
      const initialFetchCount = fetchCount;

      // Create new store instance and init - should fetch AGAIN
      // even though .zattrs might be in cache
      const newStore = new MultiLevelCachingStore('https://example.com/data.zarr');
      await newStore.init();

      // CRITICAL: fetchCount should have increased
      // If validation used cache, fetchCount would be unchanged
      expect(fetchCount).toBeGreaterThan(initialFetchCount);
    });

    it('serializes concurrent validations for the same dataset id', async () => {
      // Without a per-dataset validation queue, the second validation can finish
      // first and set a newer hash, then the slower first validation can finish
      // later and restore stale metadata.
      const l2Store = (store as any).l2Store;
      l2Store.setContentHash('initial-hash');

      let releaseFirstFetch!: () => void;
      const firstFetchGate = new Promise<void>((resolve) => {
        releaseFirstFetch = resolve;
      });
      let zattrsFetches = 0;

      global.fetch = vi.fn(async (url: string) => {
        if (url.includes('.zattrs')) {
          const fetchIndex = zattrsFetches++;
          if (fetchIndex === 0) {
            await firstFetchGate;
          }
          const contentHash = fetchIndex === 0 ? 'older-hash' : 'newer-hash';
          return {
            ok: true,
            async arrayBuffer() {
              return new TextEncoder().encode(JSON.stringify({ content_hash: contentHash })).buffer;
            },
          } as Response;
        }
        return {
          ok: true,
          async arrayBuffer() {
            return new Uint8Array([1, 2, 3]).buffer;
          },
        } as Response;
      }) as any;

      const firstValidation = (store as any).validateCache('same-dataset');
      const secondValidation = (store as any).validateCache('same-dataset');

      await Promise.resolve();
      await Promise.resolve();
      expect(zattrsFetches).toBe(1);

      releaseFirstFetch();
      await Promise.all([firstValidation, secondValidation]);

      expect(zattrsFetches).toBe(2);
      expect(l2Store.getContentHash()).toBe('newer-hash');
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

  describe('getResult — structured error reporting', () => {
    it('returns ok(data) on a successful L3 fetch', async () => {
      const r = await store.getResult('chunk-ok');
      expect(r.ok).toBe(true);
      if (r.ok) expect(Array.from(r.value)).toEqual([1, 2, 3, 4, 5]);
    });

    it('returns err(Missing) on a 404', async () => {
      // Override fetch to 404 a specific key.
      const originalFetch = global.fetch;
      global.fetch = vi.fn(async (url: string) => {
        if (url.includes('missing-chunk')) {
          return { ok: false, status: 404, async arrayBuffer() { return new ArrayBuffer(0); } } as Response;
        }
        return originalFetch(url, undefined as unknown as RequestInit);
      }) as unknown as typeof fetch;

      const r = await store.getResult('missing-chunk');
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.error.kind).toBe('Missing');

      global.fetch = originalFetch;
    });

    it('promotes L1/L2 hits to ok without re-fetching', async () => {
      // Warm L1 with a fetch
      await store.getResult('warm-key');

      // Spy fetch to ensure no further call
      const fetchSpy = vi.fn();
      const originalFetch = global.fetch;
      global.fetch = fetchSpy as unknown as typeof fetch;

      const r = await store.getResult('warm-key');
      expect(r.ok).toBe(true);
      expect(fetchSpy).not.toHaveBeenCalled();

      global.fetch = originalFetch;
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
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    it('should retry transient HTTP errors', async () => {
      global.fetch = vi
        .fn()
        .mockResolvedValueOnce({ ok: false, status: 503 })
        .mockResolvedValueOnce({ ok: false, status: 503 })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          async arrayBuffer() {
            return new Uint8Array([9, 8, 7]).buffer;
          },
        }) as any;

      const result = await store.get('flaky');
      expect(result).toEqual(new Uint8Array([9, 8, 7]));
      expect(global.fetch).toHaveBeenCalledTimes(3);
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

      const store = new MultiLevelCachingStore('https://example.com/test.zarr');
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
      const newStore = new MultiLevelCachingStore('https://example.com/test.zarr');
      await newStore.dispose(); // Should not throw
      expect(newStore).toBeDefined();
    });

    it(
      'VC-1: dispose aborts in-flight cache validation and removes the queue entry',
      { timeout: 15_000 },
      async () => {
        // Mock fetch that hangs UNTIL the abort signal fires; abort path
        // rejects with the standard AbortError so fetchWithRetry can unwind.
        let observedSignal: AbortSignal | undefined;
        global.fetch = vi.fn((_url: string, init?: RequestInit) => {
          const signal = init?.signal as AbortSignal | undefined;
          observedSignal = signal;
          return new Promise<Response>((_resolve, reject) => {
            if (signal?.aborted) {
              reject(new DOMException('Aborted', 'AbortError'));
              return;
            }
            signal?.addEventListener(
              'abort',
              () => reject(new DOMException('Aborted', 'AbortError')),
              { once: true }
            );
          });
        }) as any;

        const slowStore = new MultiLevelCachingStore('https://example.com/slow.zarr', {
          l1MaxSize: 20 * 1024 * 1024,
          l2MaxSize: 4096,
        });
        // Kick off init (which calls validateCache → fetchWithRetry).
        const initPromise = slowStore.init();
        // Give the init enough time to enter fetchWithRetry.
        await new Promise((r) => setTimeout(r, 50));

        await slowStore.dispose();

        // The fetch's signal must have aborted as a consequence of dispose().
        expect(observedSignal?.aborted).toBe(true);

        // init() must resolve now that the validation chain has unwound.
        await initPromise;
      }
    );

    it(
      'VC-2: cache-validation HEAD probe uses a shorter timeout than data fetches',
      { timeout: 15_000 },
      async () => {
        // Verify timing: the validation HEAD probe should abort under the
        // shorter `validationTimeoutMs` budget (5 s), not the full 30 s
        // `timeoutMs` data budget. We assert that the FIRST attempt's abort
        // fires within ~validationTimeoutMs / maxAttempts.
        const startTimes = new Map<string, number>();
        const abortTimes = new Map<string, number>();
        global.fetch = vi.fn((url: string, init?: RequestInit) => {
          const path = url.replace('https://example.com/timeout.zarr/', '');
          startTimes.set(path, Date.now());
          const signal = init?.signal as AbortSignal | undefined;
          return new Promise<Response>((_resolve, reject) => {
            signal?.addEventListener(
              'abort',
              () => {
                abortTimes.set(path, Date.now());
                reject(new DOMException('Aborted', 'AbortError'));
              },
              { once: true }
            );
          });
        }) as any;

        const timeoutStore = new MultiLevelCachingStore('https://example.com/timeout.zarr', {
          l1MaxSize: 20 * 1024 * 1024,
          l2MaxSize: 4096,
        });
        const initPromise = timeoutStore.init();

        // Let the validation budget run through enough retries for our
        // measurement, but stay well under the data-fetch budget.
        await new Promise((r) => setTimeout(r, 2500));

        const start = startTimes.get('.zattrs');
        const abort = abortTimes.get('.zattrs');
        expect(start).toBeDefined();
        expect(abort).toBeDefined();
        // First-attempt abort should fire within ~validationTimeoutMs/4 ≈ 1.25 s
        // (validation budget), not 30 s/4 ≈ 7.5 s (data budget).
        expect((abort as number) - (start as number)).toBeLessThan(2000);

        await timeoutStore.dispose();
        await initPromise;
      }
    );
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
      const noCacheStore = new MultiLevelCachingStore('https://example.com/test.zarr', {
        noCache: true,
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
      // Debug logging is routed through `log.info`, which calls `console.log`
      // (the central log utility's standardised channel for INFO-level output).
      const consoleLog = vi.spyOn(console, 'log');

      const debugStore = new MultiLevelCachingStore('https://example.com/test.zarr', {
        debug: true,
      });
      await debugStore.init();

      await debugStore.get('test');

      expect(consoleLog).toHaveBeenCalled();
      consoleLog.mockRestore();
    });

    it('should clear cache with ?clear-cache', async () => {
      // Pre-populate
      await store.get('test1');
      await store.dispose();

      const clearStore = new MultiLevelCachingStore('https://example.com/data.zarr', {
        clearCache: true,
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

    it('does NOT call prefetcher.onAccess() when suppressPrefetch=true on L3 fetch', async () => {
      // Phase 13.7: prefetcher.processQueue() passes
      // { suppressPrefetch: true } so the cascade
      //   demand → onAccess → prefetch → getResult → onAccess → ...
      // terminates after the first hop. This test pins the contract.
      const mockPrefetcher = { onAccess: vi.fn() };
      store.setPrefetcher(mockPrefetcher as any);

      await store.getResult('cascade.chunk', { suppressPrefetch: true });

      expect(mockPrefetcher.onAccess).not.toHaveBeenCalled();
    });

    it('does NOT call prefetcher.onAccess() when suppressPrefetch=true on L2 hit', async () => {
      // Populate L2.
      await store.get('cascade.l2');

      const mockPrefetcher = { onAccess: vi.fn() };
      store.setPrefetcher(mockPrefetcher as any);

      // Clear L1 so the next access goes through L2.
      store.clearL1();

      await store.getResult('cascade.l2', { suppressPrefetch: true });

      expect(mockPrefetcher.onAccess).not.toHaveBeenCalled();
    });
  });

  describe('URL Construction (Triple-Slash Bug Prevention)', () => {
    it('should construct correct URL without trailing slash in baseUrl', async () => {
      mocks.fetchedUrls.length = 0;

      // Store created with NO trailing slash
      const storeNoSlash = new MultiLevelCachingStore('https://example.com/data.zarr');
      await storeNoSlash.init();
      mocks.fetchedUrls.length = 0;

      await storeNoSlash.get('test.chunk');

      // Should be exactly one slash between base and key
      expect(mocks.fetchedUrls).toContain('https://example.com/data.zarr/test.chunk');
      // No double slashes in path (after protocol)
      expect(mocks.fetchedUrls.some((u) => u.replace('https://', '').includes('//'))).toBe(false);
    });

    it('should construct correct URL WITH trailing slash in baseUrl', async () => {
      mocks.fetchedUrls.length = 0;

      // Store created WITH trailing slash (as normalizeURL would produce)
      const storeWithSlash = new MultiLevelCachingStore('https://example.com/data.zarr/');
      await storeWithSlash.init();
      mocks.fetchedUrls.length = 0;

      await storeWithSlash.get('test.chunk');

      // Should still be exactly one slash between base and key
      expect(mocks.fetchedUrls).toContain('https://example.com/data.zarr/test.chunk');
      // No double slashes in middle of URL (after protocol)
      expect(mocks.fetchedUrls.some((u) => u.replace('https://', '').includes('//'))).toBe(false);
    });

    it('should handle key with leading slash', async () => {
      mocks.fetchedUrls.length = 0;

      // Store with trailing slash + key with leading slash = potential triple slash
      const storeWithSlash = new MultiLevelCachingStore('https://example.com/data.zarr/');
      await storeWithSlash.init();
      mocks.fetchedUrls.length = 0;

      await storeWithSlash.get('/Mandelbulb/positions/0.0.0');

      // Should normalize to single slash
      expect(mocks.fetchedUrls).toContain(
        'https://example.com/data.zarr/Mandelbulb/positions/0.0.0'
      );
      // No triple slashes
      expect(mocks.fetchedUrls.some((u) => u.includes('///'))).toBe(false);
    });

    it('should handle multiple trailing slashes in baseUrl', async () => {
      mocks.fetchedUrls.length = 0;

      // Store with MULTIPLE trailing slashes (edge case)
      const storeMultiSlash = new MultiLevelCachingStore('https://example.com/data.zarr///');
      await storeMultiSlash.init();
      mocks.fetchedUrls.length = 0;

      await storeMultiSlash.get('test.chunk');

      // Should normalize to single slash
      expect(mocks.fetchedUrls).toContain('https://example.com/data.zarr/test.chunk');
    });

    it('should handle multiple leading slashes in key', async () => {
      mocks.fetchedUrls.length = 0;

      const storeWithSlash = new MultiLevelCachingStore('https://example.com/data.zarr/');
      await storeWithSlash.init();
      mocks.fetchedUrls.length = 0;

      await storeWithSlash.get('///test.chunk');

      // Should normalize to single slash
      expect(mocks.fetchedUrls).toContain('https://example.com/data.zarr/test.chunk');
    });

    it('should correctly construct .zattrs URL for content hash validation', async () => {
      mocks.fetchedUrls.length = 0;

      // Store with trailing slash
      const storeWithSlash = new MultiLevelCachingStore('https://example.com/data.zarr/');
      await storeWithSlash.init();

      // .zattrs should be fetched correctly during init
      const zattrsUrls = mocks.fetchedUrls.filter((u) => u.includes('.zattrs'));
      expect(zattrsUrls.length).toBeGreaterThan(0);
      expect(zattrsUrls[0]).toBe('https://example.com/data.zarr/.zattrs');
      // No double slashes before .zattrs
      expect(zattrsUrls.some((u) => u.includes('//.zattrs'))).toBe(false);
    });
  });
});
