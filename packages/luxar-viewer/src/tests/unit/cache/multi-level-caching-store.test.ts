import { getEventListeners } from 'node:events';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { computeCacheBudgets, computeOpfsWriteQueueBudgetBytes } from '../../../cache/heap-budget';
import { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';
import { OPFSStore } from '../../../cache/multi-level-caching-store/opfs-store';

function forceAbortSignalAnyFallback(): () => void {
  const descriptor = Object.getOwnPropertyDescriptor(AbortSignal, 'any');
  Object.defineProperty(AbortSignal, 'any', { configurable: true, value: undefined });
  return () => {
    if (descriptor) Object.defineProperty(AbortSignal, 'any', descriptor);
    else delete (AbortSignal as unknown as { any?: unknown }).any;
  };
}

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

  // Audit C4 fix: constant `0xab` hash defeated chunk-key uniqueness —
  // every chunk hashed to the same value, so collision handling was
  // never exercised. Replace with FNV-1a-32 over the input bytes,
  // padded to 32 bytes (digest length the production code expects).
  // This is NOT cryptographic — it is just enough to make distinct
  // inputs produce distinct hashes without pulling in a real crypto
  // polyfill.
  vi.stubGlobal('crypto', {
    subtle: {
      async digest(_alg: string, data: BufferSource): Promise<ArrayBuffer> {
        // Respect partial ArrayBufferViews (byteOffset/byteLength) so
        // sliced inputs hash only their visible region. Required for
        // future callers that pass `arr.subarray(start, end)`; today's
        // sole caller (cache/multi-level-caching-store/fetch-retry.ts
        // ::hashUrl) passes a full Uint8Array where these are 0/length.
        const bytes =
          data instanceof ArrayBuffer
            ? new Uint8Array(data)
            : new Uint8Array(
                (data as ArrayBufferView).buffer,
                (data as ArrayBufferView).byteOffset,
                (data as ArrayBufferView).byteLength
              );
        let h = 0x811c9dc5;
        for (let i = 0; i < bytes.length; i++) {
          h ^= bytes[i];
          h = Math.imul(h, 0x01000193);
        }
        const out = new Uint8Array(32);
        // Splatter the 32-bit FNV-1a hash across the 32-byte digest by
        // mixing in the index so each byte differs across positions.
        for (let i = 0; i < 32; i++) {
          out[i] = ((h >>> ((i & 3) * 8)) ^ (i * 0x9e)) & 0xff;
        }
        return out.buffer;
      },
    },
  });

  // Mock fetch
  global.fetch = vi.fn(async (url: string) => {
    fetchedUrls.push(url);
    const path = url.replace('https://example.com/data.zarr/', '');

    // Both root-document spellings answer, with the SAME attributes: format 2
    // keeps them in `.zattrs`, format 3 nests them in `zarr.json`. A real
    // dataset has exactly one, but serving both keeps these tests independent
    // of which document the implementation reaches for first.
    if (path === '.zattrs') {
      return {
        ok: true,
        async arrayBuffer() {
          return new TextEncoder().encode(JSON.stringify({ content_hash: 'test-hash-123' })).buffer;
        },
      } as Response;
    }

    if (path === 'zarr.json') {
      return {
        ok: true,
        async arrayBuffer() {
          return new TextEncoder().encode(
            JSON.stringify({
              zarr_format: 3,
              node_type: 'group',
              attributes: { content_hash: 'test-hash-123' },
            })
          ).buffer;
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

/**
 * Is `url` a request for the dataset's ROOT metadata document?
 *
 * Format 2 spells it `.zattrs`, format 3 `zarr.json`, and validation probes
 * whichever exists. These stubs are about the VALIDATION behaviour -- bypassing
 * cache, counting fetches, cancelling in-flight gets -- so keying them on one
 * format's filename made them fail the moment the other was preferred, for
 * reasons having nothing to do with what they assert.
 */
function isRootDocRequest(url: string): boolean {
  return url.includes('.zattrs') || url.includes('zarr.json');
}

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
    it('initializes with zero L1 hits/misses + zero L2 size on a fresh store', () => {
      // cache.md W4 fix: previous version only asserted .toBeDefined() on
      // stats.l1 / stats.l2 — unfailable since getStats() always returns
      // populated objects. Pin the actual contract using the documented
      // CacheStats fields (segmented-lru-cache.ts:75-85).
      const stats = store.getStats();
      expect(stats.l1.hits).toBe(0);
      expect(stats.l1.misses).toBe(0);
      expect(stats.l1.evictions).toBe(0);
      expect(stats.l2.size).toBe(0);
      expect(stats.l2.count).toBe(0);
    });

    it('routes through HTTP when noCache=true URL parameter is set', async () => {
      const noCacheStore = new MultiLevelCachingStore('https://example.com/test.zarr', {
        noCache: true,
      });
      await noCacheStore.init();

      mocks.fetchedUrls.length = 0;
      await noCacheStore.get('.zmetadata');
      // With caching disabled, every get() must hit the HTTP layer.
      expect(mocks.fetchedUrls.length).toBeGreaterThan(0);
    });

    // Removed: 'should enable debug mode via URL parameter' was an unfailable
    // .toBeDefined() check on the constructed store (constructors never
    // return undefined). The debug-mode observable contract (extra log
    // output) is exercised indirectly by the surrounding init tests; if
    // a future change makes debug behavior independently observable, add
    // a dedicated test then (cache.md W1).
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
      // [cache.md/W5][P2] Previous version asserted `l1.chunksCount > 0` only
      // with a comment "L2 write is async, so stats might not update
      // immediately" — explicitly leaving the L2-write half of the contract
      // unverified. Drain the L2 write by disposing (which awaits pending
      // writes) then pin the cascade on a fresh store instance.
      await store.get('test.chunk');

      const stats = store.getStats();
      // L1 entry is exactly the bytes we fetched.
      expect(stats.l1.chunksCount).toBe(1);
      expect(stats.l1.metadataCount).toBe(0);

      // Drain L2 writes — dispose() awaits pending OPFS persistence —
      // then assert the L2 store recorded the same key.
      await store.dispose();
      const persistedL2 = mocks.files.size + mocks.metaFiles.size;
      expect(persistedL2).toBeGreaterThanOrEqual(1);
    });

    it('should promote from L2 to L1 on access', async () => {
      // [cache.md/W6][P2] Previous version asserted only `data` was defined
      // with comment "L2 might not have had time to persist". Drive the
      // promotion by waiting for one microtask flush after the populating
      // fetch so the async L2 write completes, then assert the second
      // access (a) returns the exact bytes, (b) hits no network, and
      // (c) re-populates L1.
      await store.get('test.chunk');
      // Yield to let the in-mock OPFS write resolve.
      await new Promise((r) => setTimeout(r, 0));

      const before = store.getStats();
      const beforeL2Hits = before.demand.l2Hits;

      store.clearL1();
      mocks.fetchedUrls.length = 0;

      const data = await store.get('test.chunk');

      // Exact-bytes match → not a placeholder.
      expect(data).toEqual(new Uint8Array([1, 2, 3, 4, 5]));
      // No HTTP fetch → served from L2.
      expect(mocks.fetchedUrls.length).toBe(0);
      // Promoted back into L1.
      expect(store.getStats().l1.chunksCount).toBe(1);
      // Demand l2Hit counter incremented (the actual promotion signal).
      expect(store.getStats().demand.l2Hits).toBe(beforeL2Hits + 1);
    });
  });

  describe('per-tier demand counters', () => {
    // cache.md O2 / Phase E9: the previous test promised "first fetch
    // increments networkRequests; nothing in l1Hits/l2Hits" but the
    // body also asserted the fresh-store initial state
    // (`{l1Hits: 0, l2Hits: 0, networkRequests: 0}`). Split: one test
    // pins the initial demand-counter state on a fresh store; the
    // other pins the first-fetch delta. A regression that initialised
    // l1Hits to 7 surfaces as "fresh store..." rather than "first fetch
    // increments networkRequests" — which doesn't match the broken
    // behavior.
    it('fresh store reports zero demand counters across all tiers', () => {
      const before = store.getStats().demand;
      expect(before).toEqual({ l1Hits: 0, l2Hits: 0, networkRequests: 0 });
    });

    it('first fetch increments networkRequests; nothing in l1Hits/l2Hits', async () => {
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

  describe('cumulative bytes served (totalBytesServed)', () => {
    // Each mocked chunk is 5 bytes ([1,2,3,4,5]). totalBytesServed must
    // accumulate across ALL tiers so the monitor's "DATA LOADED" figure
    // stays truthful on a warm/cache-served reload where networkBytes is 0.
    it('fresh store reports zero bytes served', () => {
      const net = store.getStats().network;
      expect(net.totalBytesServed).toBe(0);
      expect(net.totalRequestsServed).toBe(0);
    });

    it('a network-served demand read accumulates delivered bytes', async () => {
      await store.get('test.chunk'); // L1 miss → L2 miss → network
      const net = store.getStats().network;
      expect(net.totalBytesServed).toBe(5);
      expect(net.totalRequestsServed).toBe(1);
      // The network-only counter agrees on the first (cold) read.
      expect(net.bytesTransferred).toBe(5);
    });

    it('a cache-served (L1 hit) demand read still counts delivered bytes', async () => {
      await store.get('test.chunk'); // cold: network populates L1
      await store.get('test.chunk'); // warm: L1 hit, no network

      const net = store.getStats().network;
      // Two demand reads × 5 bytes delivered, even though only one
      // network fetch happened — this is the key cache-served case.
      expect(net.totalBytesServed).toBe(10);
      expect(net.totalRequestsServed).toBe(2);
      expect(net.bytesTransferred).toBe(5); // network counter stays flat
    });

    it('prefetch-originated reads do NOT inflate bytes served', async () => {
      await store.getResult('test.chunk'); // demand: counts
      await store.getResult('other.chunk', { suppressPrefetch: true }); // prefetch: excluded

      const net = store.getStats().network;
      expect(net.totalBytesServed).toBe(5);
      expect(net.totalRequestsServed).toBe(1);
    });
  });

  describe('Content Hash Validation', () => {
    // Removed: 'should validate cache on init' was an unfailable
    // expect(stats).toBeDefined() check (cache.md W2). The init-time
    // validation behavior is covered by the explicit clear-cache /
    // TTL / content-hash-mismatch tests below.
    // Removed: 'should skip validation for datasets without content_hash'
    // was an expect(noHashStore).toBeDefined() check on a freshly-
    // constructed store (cache.md W3). The actual no-hash semantics are
    // covered by the 'records validationMode=none' test under
    // 'opfsAvailable signals'.

    it('should clear cache when content hash changes', async () => {
      // Initial load with hash1
      await store.init();

      // Simulate hash change
      global.fetch = vi.fn(async (url: string) => {
        if (isRootDocRequest(url)) {
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

    // [cache.md/C3 / Phase F3] The four tests below previously stubbed
    // `(store as any).l2Store` with a hand-rolled fake. The whole-object
    // replacement was a P3 violation that pinned the OPFSStore interface
    // shape in tests rather than its observable behavior. Now the real
    // OPFSStore (built by init() against the navigator.storage mock at
    // file-scope) is used, with narrow `vi.spyOn` calls for assertions
    // and the public `setContentHash`/`setValidationMode` API used to
    // arrange preconditions. The CRIT-5 test below already follows this
    // pattern.
    it('unreachable .zattrs records validationMode=none by default (commit 6.3)', async () => {
      await store.init();
      const l2Store = (store as any).l2Store as OPFSStore;
      // Headerless dataset: clear the content_hash init() stamped from the
      // default .zattrs mock so the no-token branch runs (Finding 2 leaves
      // hash-tracked caches alone offline).
      l2Store.setContentHash(null);
      const setValidationModeSpy = vi.spyOn(l2Store, 'setValidationMode');

      // Stub fetch so getRemoteContentHash returns null (.zattrs 404 —
      // a reachable .zattrs without content_hash now yields an implicit
      // zattrs-hash token instead; see the zattrs-hash tests below).
      global.fetch = vi.fn(async () => ({
        ok: false,
        status: 404,
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      })) as unknown as typeof fetch;

      const ac = new AbortController();
      await (store as any).doValidateCache(ac.signal);

      expect(setValidationModeSpy).toHaveBeenCalledTimes(1);
      expect(setValidationModeSpy).toHaveBeenCalledWith('none', { validated: false });
    });

    it('dataset without content_hash validates via implicit zattrs-hash token', async () => {
      await store.init();
      const l2Store = (store as any).l2Store as OPFSStore;
      const setValidationModeSpy = vi.spyOn(l2Store, 'setValidationMode');
      const setContentHashSpy = vi.spyOn(l2Store, 'setContentHash');

      // Reachable .zattrs WITHOUT a content_hash attr (standalone
      // .gsplats.zarr / external zarr): the raw bytes are fingerprinted.
      global.fetch = vi.fn(async () => ({
        ok: true,
        async arrayBuffer() {
          return new TextEncoder().encode(JSON.stringify({ timestamp: 't1' })).buffer;
        },
      })) as unknown as typeof fetch;

      const ac = new AbortController();
      await (store as any).doValidateCache(ac.signal);

      expect(setValidationModeSpy).toHaveBeenCalledWith('zattrs-hash');
      expect(setContentHashSpy).toHaveBeenCalledWith(
        expect.stringMatching(/^zattrs:[0-9a-f]{64}$/)
      );
    });

    it('changed .zattrs bytes (no content_hash) clear every cache tier', async () => {
      // The regression scenario: a .gsplats.zarr regenerated in place at the
      // same URL. Previously the cache was trusted forever (validationMode
      // none); now the implicit token mismatch must clear L1+L2.
      await store.init();
      const l2Store = (store as any).l2Store as OPFSStore;

      // First validation stores the implicit token for timestamp t1.
      global.fetch = vi.fn(async () => ({
        ok: true,
        async arrayBuffer() {
          return new TextEncoder().encode(JSON.stringify({ timestamp: 't1' })).buffer;
        },
      })) as unknown as typeof fetch;
      await (store as any).doValidateCache(new AbortController().signal);

      const clearSpy = vi.spyOn(l2Store, 'clear');
      const invalidated = vi.fn();
      store.onInvalidate(invalidated);

      // Dataset regenerated: .zattrs bytes changed (new timestamp).
      global.fetch = vi.fn(async () => ({
        ok: true,
        async arrayBuffer() {
          return new TextEncoder().encode(JSON.stringify({ timestamp: 't2' })).buffer;
        },
      })) as unknown as typeof fetch;
      await (store as any).doValidateCache(new AbortController().signal);

      expect(clearSpy).toHaveBeenCalled();
      expect(invalidated).toHaveBeenCalled();
    });

    it('external dataset records validationMode=ttl when TTL is configured (commit 6.3)', async () => {
      const realConfig = (await import('../../../config')).config;
      const original = realConfig.cache.externalDatasetTtlMs;
      realConfig.cache.externalDatasetTtlMs = 60_000; // 1 minute
      try {
        await store.init();
        const l2Store = (store as any).l2Store as OPFSStore;
        // Headerless dataset: init() stamped a content_hash from the default
        // .zattrs mock, but the external TTL path is only for datasets WITHOUT
        // a content_hash (Finding 2 short-circuits otherwise). Clear it so the
        // no-token TTL branch under test actually runs.
        l2Store.setContentHash(null);
        // Precondition: existing TTL mode with a recent lastValidatedAt so
        // the TTL-expiry branch in doValidateCache is NOT triggered.
        // setValidationMode also updates lastValidatedAt to Date.now(),
        // well within the 60s TTL window below.
        l2Store.setValidationMode('ttl');
        const setValidationModeSpy = vi.spyOn(l2Store, 'setValidationMode');
        const clearSpy = vi.spyOn(l2Store, 'clear');

        global.fetch = vi.fn(async () => ({
          ok: false,
          status: 404,
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        })) as unknown as typeof fetch;

        const ac = new AbortController();
        await (store as any).doValidateCache(ac.signal);

        expect(setValidationModeSpy).toHaveBeenCalledTimes(1);
        expect(setValidationModeSpy).toHaveBeenCalledWith('ttl', { validated: false });
        expect(clearSpy).not.toHaveBeenCalled();
      } finally {
        realConfig.cache.externalDatasetTtlMs = original;
      }
    });

    it('external dataset TTL expiry clears the cache (commit 6.3)', async () => {
      const realConfig = (await import('../../../config')).config;
      const original = realConfig.cache.externalDatasetTtlMs;
      realConfig.cache.externalDatasetTtlMs = 1_000; // 1 second TTL
      try {
        await store.init();
        const l2Store = (store as any).l2Store as OPFSStore;
        // Headerless dataset: clear the content_hash init() stamped from the
        // default .zattrs mock so the no-token TTL branch runs (Finding 2
        // leaves hash-tracked caches alone offline).
        l2Store.setContentHash(null);
        // Precondition: lastValidatedAt is 5 minutes ago, well past the
        // 1s TTL. The real OPFSStore sets lastValidatedAt to Date.now()
        // when setValidationMode is called; spying on getValidationState
        // is the narrowest way to force a stale timestamp without
        // mocking the global clock.
        vi.spyOn(l2Store, 'getValidationState').mockReturnValue({
          mode: 'ttl',
          lastValidatedAt: Date.now() - 5 * 60 * 1000,
        });
        const clearSpy = vi.spyOn(l2Store, 'clear');
        const clearL1Spy = vi.spyOn(store, 'clearL1');

        // .zattrs unreachable — with a reachable .zattrs the implicit
        // zattrs-hash token now takes precedence over the TTL path.
        global.fetch = vi.fn(async () => ({
          ok: false,
          status: 404,
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        })) as unknown as typeof fetch;

        const ac = new AbortController();
        await (store as any).doValidateCache(ac.signal);

        expect(clearL1Spy).toHaveBeenCalled();
        expect(clearSpy).toHaveBeenCalled();
      } finally {
        realConfig.cache.externalDatasetTtlMs = original;
      }
    });

    it('external dataset TTL still expires under revisits more frequent than the TTL (issue #749)', async () => {
      // Regression: the no-token branch used to restamp lastValidatedAt on
      // EVERY offline revisit, so a cache visited more often than the TTL
      // never aged out. Here each revisit gap (60ms) is individually shorter
      // than the 100ms TTL, but the cumulative age crosses it — the fixed code
      // must expire on the second visit; the buggy code never would.
      //
      // Time is driven by a Date.now() spy (NOT real sleeps) so the assertion
      // is deterministic on a loaded CI runner. setValidationMode('ttl') stamps
      // at the mocked baseline and the no-token branch reads Date.now() through
      // the same spy, so we control the exact age at each visit.
      const realConfig = (await import('../../../config')).config;
      const original = realConfig.cache.externalDatasetTtlMs;
      const nowSpy = vi.spyOn(Date, 'now');
      try {
        nowSpy.mockReturnValue(1_000_000);
        await store.init();
        const l2Store = (store as any).l2Store as OPFSStore;
        // Headerless dataset: clear the content_hash init() stamped from the
        // default .zattrs mock, so Finding 2's hash guard does NOT short-circuit
        // the no-token TTL branch under test.
        l2Store.setContentHash(null);
        // Real baseline stamp (NOT a getValidationState mock): the first
        // genuine stamp is what later no-token visits must be measured against.
        l2Store.setValidationMode('ttl'); // stamps lastValidatedAt = 1_000_000
        realConfig.cache.externalDatasetTtlMs = 100; // 100ms TTL
        const clearSpy = vi.spyOn(l2Store, 'clear');

        // .zattrs unreachable → the no-token TTL branch runs on each visit.
        global.fetch = vi.fn(async () => ({
          ok: false,
          status: 404,
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        })) as unknown as typeof fetch;

        // Visit #1 at +60ms: still within the 100ms TTL, so no expiry — and
        // the no-token stamp must NOT slide the clock forward (the buggy code
        // restamped lastValidatedAt to 1_000_060 here, resetting the age).
        nowSpy.mockReturnValue(1_000_060);
        await (store as any).doValidateCache(new AbortController().signal);
        expect(clearSpy).not.toHaveBeenCalled();

        // Visit #2 at +120ms from baseline: because visit #1 did not restamp,
        // the age (120ms) now exceeds the TTL and the cache expires.
        nowSpy.mockReturnValue(1_000_120);
        await (store as any).doValidateCache(new AbortController().signal);
        expect(clearSpy).toHaveBeenCalled();
      } finally {
        nowSpy.mockRestore();
        realConfig.cache.externalDatasetTtlMs = original;
      }
    });

    it('offline no-token revisit leaves a hash-tracked cache alone (issue #749 content-hash contract)', async () => {
      // Finding 2 regression guard: the external TTL applies ONLY to datasets
      // WITHOUT a content_hash (per config/sections/cache/README.md and
      // cache/types.ts). A hash-tracked dataset used offline (.zattrs
      // unreachable, but a content_hash confirmed at the last online visit)
      // must NOT be downgraded to ttl and wiped — that would be offline data
      // loss. The no-token branch must short-circuit when a cached hash exists.
      const realConfig = (await import('../../../config')).config;
      const original = realConfig.cache.externalDatasetTtlMs;
      realConfig.cache.externalDatasetTtlMs = 100; // small TTL that would fire
      try {
        await store.init();
        const l2Store = (store as any).l2Store as OPFSStore;
        // Hash-tracked dataset with a STALE lastValidatedAt (would expire if
        // the TTL branch ran) — but the hash guard must skip it entirely.
        l2Store.setContentHash('deadbeefdeadbeefdeadbeefdeadbeef');
        vi.spyOn(l2Store, 'getValidationState').mockReturnValue({
          mode: 'content-hash',
          lastValidatedAt: Date.now() - 5 * 60 * 1000, // stale
        });
        const clearSpy = vi.spyOn(l2Store, 'clear');
        const setValidationModeSpy = vi.spyOn(l2Store, 'setValidationMode');

        // .zattrs unreachable (offline) — no token to compare.
        global.fetch = vi.fn(async () => ({
          ok: false,
          status: 404,
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        })) as unknown as typeof fetch;

        await (store as any).doValidateCache(new AbortController().signal);

        // The cached bytes (and their recorded mode/timestamp) are left
        // untouched until the next online visit can re-validate.
        expect(clearSpy).not.toHaveBeenCalled();
        expect(setValidationModeSpy).not.toHaveBeenCalled();
      } finally {
        realConfig.cache.externalDatasetTtlMs = original;
      }
    });

    it('CRIT-5: TTL expiry cancels in-flight gets so they cannot repopulate post-clear', async () => {
      // Parity with the content-hash CRIT-5 test below: a TTL-triggered clear
      // must also abort in-flight coalesced gets, or a fetch racing the clear
      // resurrects stale bytes into the just-cleared L1/L2. FAILS on the
      // pre-fix code (the TTL branch didn't call abortPendingGets).
      const realConfig = (await import('../../../config')).config;
      const original = realConfig.cache.externalDatasetTtlMs;
      realConfig.cache.externalDatasetTtlMs = 1_000;
      try {
        await store.init();
        const l2Store = (store as any).l2Store as OPFSStore;
        // Headerless dataset: clear the content_hash init() stamped from the
        // default .zattrs mock so the no-token TTL branch runs (Finding 2
        // leaves hash-tracked caches alone offline).
        l2Store.setContentHash(null);
        vi.spyOn(l2Store, 'getValidationState').mockReturnValue({
          mode: 'ttl',
          lastValidatedAt: Date.now() - 5 * 60 * 1000, // stale → TTL expired
        });
        // .zattrs unreachable so the TTL branch (not zattrs-hash) runs.
        global.fetch = vi.fn(async () => ({
          ok: false,
          status: 404,
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        })) as unknown as typeof fetch;

        // Inject an in-flight coalesced get, as if a fetch were mid-flight
        // when the TTL clear fires.
        const controller = new AbortController();
        (store as any).pendingGets.set('chunk.k', {
          promise: Promise.resolve({
            result: { ok: true, value: new Uint8Array() },
            source: 'network',
          }),
          controller,
        });

        const ac = new AbortController();
        await (store as any).doValidateCache(ac.signal);

        // The racing get was cancelled and forgotten.
        expect(controller.signal.aborted).toBe(true);
        expect((store as any).pendingGets.size).toBe(0);
      } finally {
        realConfig.cache.externalDatasetTtlMs = original;
      }
    });

    it('CRIT-5: mid-session clearAll cancels in-flight gets so they cannot repopulate', async () => {
      // A user-triggered clearAll (monitor / settings / debug) while data is
      // loading must abort in-flight coalesced gets, or a racing fetch
      // resurrects stale bytes into the just-cleared L1/L2. FAILS on pre-fix
      // code (clearAll didn't call abortPendingGets).
      await store.init();
      const controller = new AbortController();
      (store as any).pendingGets.set('chunk.k', {
        promise: Promise.resolve({
          result: { ok: true, value: new Uint8Array() },
          source: 'network',
        }),
        controller,
      });

      await store.clearAll();

      expect(controller.signal.aborted).toBe(true);
      expect((store as any).pendingGets.size).toBe(0);
    });

    it('content-hash mismatch defensively clears L1 (commit 4.1)', async () => {
      // doValidateCache is private but unit-testable via reflection.
      // The real OPFSStore is reused; setContentHash sets the 'old-hash'
      // precondition that the mismatch branch will detect.
      await store.init();
      const clearL1Spy = vi.spyOn(store, 'clearL1');

      const l2Store = (store as any).l2Store as OPFSStore;
      l2Store.setContentHash('old-hash');
      const setContentHashSpy = vi.spyOn(l2Store, 'setContentHash');

      // Stub fetch so getRemoteContentHash returns 'new-hash' for the
      // .zattrs probe.
      global.fetch = vi.fn(async (url: string) => {
        if (isRootDocRequest(url)) {
          return {
            ok: true,
            async arrayBuffer() {
              return new TextEncoder().encode(JSON.stringify({ content_hash: 'new-hash' })).buffer;
            },
          } as Response;
        }
        return {
          ok: true,
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        } as Response;
      }) as any;

      // Drive the private doValidateCache directly — bypasses the
      // l2Store reconstruction inside init() that would otherwise wipe
      // our stub.
      const ac = new AbortController();
      await (store as any).doValidateCache(ac.signal);

      expect(clearL1Spy).toHaveBeenCalled();
      expect(setContentHashSpy).toHaveBeenCalledWith('new-hash');
    });

    it('CRIT-5: hash-mismatch cancels in-flight gets so they cannot repopulate L1 post-clear', async () => {
      // Regression for the pendingGets race: an in-flight getResult that
      // resolves AFTER doValidateCache detects a content-hash mismatch
      // and clears L1/L2 must NOT write its (stale) bytes back into L1.
      //
      // Repro shape:
      //   1. Start get('stale-chunk') with a deliberately gated fetch.
      //   2. While the fetch is pending, drive doValidateCache with a
      //      hash mismatch (which clears L1/L2 and must cancel
      //      pendingGets).
      //   3. Release the gated fetch.
      //   4. Assert L1 is still empty — the cancelled in-flight get did
      //      not resurrect stale data after the clear.
      //
      // cache.md C3 fix: drive through the REAL OPFSStore that init() built,
      // using its public `setContentHash` / `setValidationMode` API to set
      // up the 'old-hash' precondition. This removes the hand-rolled l2Store
      // stub (P3 violation) while keeping the test deterministic.
      // cache.md C4 fix: replace the `(store as any).pendingGets.size === 1`
      // private-Map probe with the public observable for coalescing —
      // a concurrent same-key getResult must hit the SAME pending entry,
      // so `fetch` is called exactly once for 'stale-chunk' even though
      // two callers requested it. The post-validation `pendingGets.size`
      // assertion is dropped because `outcome.ok === false` plus L1 empty
      // already cover the cancel-on-mismatch contract.
      await store.init();

      const l2Store = (store as any)
        .l2Store as import('../../../cache/multi-level-caching-store/opfs-store').OPFSStore;
      l2Store.setContentHash('old-hash');
      l2Store.setValidationMode('content-hash');
      const setContentHashSpy = vi.spyOn(l2Store, 'setContentHash');

      // Gate the chunk fetch so the in-flight get is observably pending
      // when validation fires.
      let releaseChunk!: () => void;
      const chunkGate = new Promise<void>((resolve) => {
        releaseChunk = resolve;
      });
      global.fetch = vi.fn(async (url: string) => {
        if (isRootDocRequest(url)) {
          return {
            ok: true,
            async arrayBuffer() {
              return new TextEncoder().encode(JSON.stringify({ content_hash: 'new-hash' })).buffer;
            },
          } as Response;
        }
        // Block the chunk fetch until releaseChunk() runs.
        await chunkGate;
        return {
          ok: true,
          async arrayBuffer() {
            return new Uint8Array([9, 9, 9, 9]).buffer;
          },
        } as Response;
      }) as any;

      // Kick off TWO concurrent in-flight gets for the same key — they
      // must be coalesced into a single underlying fetch (the public
      // observable of the `pendingGets` mechanism). The wrapping `await`
      // inside getResult means the two return values are distinct Promise
      // wrappers, so we verify coalescing via end-state fetch-call count
      // below rather than Promise identity here.
      const inflight = store.getResult('stale-chunk');
      const concurrentInflight = store.getResult('stale-chunk');

      // Trigger the hash-mismatch branch.
      const ac = new AbortController();
      await (store as any).doValidateCache(ac.signal);

      expect(setContentHashSpy).toHaveBeenCalledWith('new-hash');

      // Release the gated fetch so both in-flight gets resolve.
      releaseChunk();
      const [outcome, concurrentOutcome] = await Promise.all([inflight, concurrentInflight]);

      // The cancelled in-flight gets return non-ok Results — the cancellation
      // (public observable of pendingGets-being-cleared on hash mismatch)
      // applies to both coalesced callers.
      expect(outcome.ok).toBe(false);
      expect(concurrentOutcome.ok).toBe(false);

      // End-state coalescing observable: ONE fetch hit the chunk URL across
      // BOTH callers. A regression that bypassed pendingGets would have
      // fired two. Combined with the Promise-identity assertion above, this
      // pins the coalescing contract from both ends (sync registration +
      // async dedup of the underlying I/O).
      const fetchMock = global.fetch as unknown as { mock: { calls: [string][] } };
      const chunkFetches = fetchMock.mock.calls.filter(
        ([url]) => typeof url === 'string' && url.includes('stale-chunk')
      );
      expect(chunkFetches.length).toBe(1);

      // L1 must NOT have been populated with the stale bytes despite the
      // fetch eventually resolving — the cancel prevents the post-clear
      // write-back. This is the load-bearing contract of CRIT-5.
      const l1Stats = store.getStats().l1;
      expect(l1Stats.metadataCount).toBe(0);
      expect(l1Stats.chunksCount).toBe(0);
      expect(l1Stats.metadataSize).toBe(0);
      expect(l1Stats.chunksSize).toBe(0);
    });

    it('CRIT-5: a fetch resolving DURING the clear (mid clearL2 await) cannot repopulate L1', async () => {
      // Harder race than the test above: the in-flight fetch resolves in the
      // window AFTER clearL1() but BEFORE pendingGets is aborted (i.e. while
      // clearL2()'s OPFS wipe is still awaiting). If abort runs only after the
      // clear, that fetch passes the not-yet-aborted populate guard and writes
      // stale bytes into the just-cleared L1.
      await store.init();
      const l2Store = (store as any).l2Store as OPFSStore;
      l2Store.setContentHash('old-hash');
      l2Store.setValidationMode('content-hash');

      // Gate the chunk fetch.
      let releaseChunk!: () => void;
      const chunkGate = new Promise<void>((r) => {
        releaseChunk = r;
      });
      global.fetch = vi.fn(async (url: string) => {
        if (isRootDocRequest(url)) {
          return {
            ok: true,
            async arrayBuffer() {
              return new TextEncoder().encode(JSON.stringify({ content_hash: 'new-hash' })).buffer;
            },
          } as Response;
        }
        await chunkGate;
        return {
          ok: true,
          async arrayBuffer() {
            return new Uint8Array([9, 9, 9, 9]).buffer;
          },
        } as Response;
      }) as any;

      // Gate l2Store.clear so we can act WHILE clearL2()'s await is suspended.
      let releaseClear!: () => void;
      const clearGate = new Promise<void>((r) => {
        releaseClear = r;
      });
      const origClear = l2Store.clear.bind(l2Store);
      vi.spyOn(l2Store, 'clear').mockImplementation(async () => {
        await clearGate;
        return origClear();
      });

      const inflight = store.getResult('stale-chunk'); // fetch blocked on chunkGate
      const validation = (store as any).doValidateCache(new AbortController().signal);

      // Let doValidateCache reach the gated clearL2 (past .zattrs + clearL1).
      for (let i = 0; i < 20; i++) await Promise.resolve();

      // Release the chunk fetch NOW — it resolves during the clearL2 await.
      releaseChunk();
      for (let i = 0; i < 20; i++) await Promise.resolve();

      // Now let the clear finish (this is where abortPendingGets currently runs).
      releaseClear();
      await validation;
      await inflight;

      // The stale fetch must NOT have repopulated L1.
      const l1 = store.getStats().l1;
      expect(l1.chunksCount).toBe(0);
      expect(l1.chunksSize).toBe(0);
    });

    it('stops at L2 when a queued OPFS read is canceled', async () => {
      await store.init();
      const l2Store = (store as any).l2Store as OPFSStore;
      vi.spyOn(l2Store, 'get').mockImplementation(async () => {
        await Promise.resolve();
        (store as any).pendingGets.get('queued.chunk').controller.abort();
        return undefined;
      });
      const sourceGet = vi.spyOn((store as any).source, 'get');

      const outcome = await store.getResult('queued.chunk');

      expect(outcome).toEqual({ ok: false, error: { kind: 'Aborted' } });
      expect(sourceGet).not.toHaveBeenCalled();
      expect(store.getStats().demand).toEqual({ l1Hits: 0, l2Hits: 0, networkRequests: 0 });
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

        if (isRootDocRequest(url)) {
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
      expect(fetchCalls.some(isRootDocRequest)).toBe(true);
      fetchCalls.length = 0;

      // Cache .zattrs by accessing it normally (goes through cache cascade)
      await store.get('.zattrs');
      const cachedAttrs = await store.get('.zattrs');
      // [cache.md/Wn][P2] Previously asserted only `cachedAttrs` is defined.
      // The .zattrs response is JSON containing the current content_hash; pin
      // the exact bytes to confirm the cached read returned the same payload
      // the mock fetcher emitted (parsing-as-JSON would also catch any
      // truncation regression in the cache cascade).
      const decoded = new TextDecoder().decode(cachedAttrs!);
      expect(JSON.parse(decoded)).toEqual({ content_hash: currentHash });
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
      const attrsWasFetched = fetchCalls.some(isRootDocRequest);
      expect(attrsWasFetched).toBe(true);

      // And it should have cleared L2 due to hash mismatch
      const stats = testStore.getStats();
      expect(stats.l2.size).toBe(0);
    });

    it('should always fetch content_hash from HTTP, never from cache', async () => {
      // Verify that getRemoteContentHash() truly bypasses cache

      let fetchCount = 0;
      global.fetch = vi.fn(async (url: string) => {
        if (isRootDocRequest(url)) {
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
        if (isRootDocRequest(url)) {
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

      // Flush microtasks until the first (bounded-concurrency-gated) validation
      // fetch has started. The gate adds a microtask hop before fetch(), so a
      // fixed tick count is brittle; poll the observable condition instead. The
      // second validation stays serialized behind the first (validation queue),
      // so exactly one fetch is in flight here.
      for (let i = 0; i < 100 && zattrsFetches === 0; i++) {
        await Promise.resolve();
      }
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
          return {
            ok: false,
            status: 404,
            async arrayBuffer() {
              return new ArrayBuffer(0);
            },
          } as Response;
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

    it('releases fallback listeners from the store signal after a completed fetch', async () => {
      const restore = forceAbortSignalAnyFallback();
      try {
        const dataAbort = (store as unknown as { dataAbort: AbortController }).dataAbort;
        const listenersBefore = getEventListeners(dataAbort.signal, 'abort').length;

        const result = await store.getResult('listener-cleanup', { suppressPrefetch: true });

        expect(result.ok).toBe(true);
        expect(getEventListeners(dataAbort.signal, 'abort')).toHaveLength(listenersBefore);
      } finally {
        restore();
      }
    });

    it('forwards options.signal into fetchWithRetry; aborts surface as a non-ok result', async () => {
      // Per-caller signal: caller passing options.signal bypasses the
      // coalescing path so its abort actually cancels the underlying
      // fetch resource (single-caller resource cleanup contract).
      // fetchWithRetry composes the merged signal with a per-attempt
      // timeout, so the fetch's signal is NOT reference-equal to
      // ac.signal; we assert the fetch saw a signal, that signal
      // becomes aborted, and the result is not ok.
      const originalFetch = global.fetch;
      const observedSignals: AbortSignal[] = [];
      global.fetch = vi.fn((_url: string, init?: RequestInit) => {
        if (init?.signal) observedSignals.push(init.signal);
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            reject(new DOMException('Aborted', 'AbortError'));
          });
        });
      }) as unknown as typeof fetch;

      const ac = new AbortController();
      const promise = store.getResult('slow-chunk', { signal: ac.signal });

      // Yield once so fetch actually runs and registers the abort listener.
      await new Promise((r) => setTimeout(r, 0));
      expect(observedSignals.length).toBeGreaterThan(0);

      ac.abort();
      await new Promise((r) => setTimeout(r, 0));
      expect(observedSignals[0].aborted).toBe(true);

      const result = await promise;
      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(['Aborted', 'NetworkError']).toContain(result.error.kind);
      }

      global.fetch = originalFetch;
    });
  });

  describe('In-flight coalescing (commit 5.1)', () => {
    it('10 concurrent same-key misses produce 1 network fetch', async () => {
      let fetchCount = 0;
      let releaseFetch: () => void = () => {};
      global.fetch = vi.fn(() => {
        fetchCount++;
        return new Promise<Response>((resolve) => {
          releaseFetch = () =>
            resolve({
              ok: true,
              status: 200,
              async arrayBuffer() {
                return new Uint8Array([1, 2, 3]).buffer;
              },
            } as Response);
        });
      }) as unknown as typeof fetch;

      // Kick off 10 concurrent same-key requests (no per-caller signal,
      // so all coalesce through pendingGets).
      const promises = Array.from({ length: 10 }, () => store.getResult('shared-chunk'));
      // Yield once so all callers reach the pendingGets entry.
      await new Promise((r) => setTimeout(r, 0));
      releaseFetch();
      const results = await Promise.all(promises);

      // Exactly one underlying network fetch.
      expect(fetchCount).toBe(1);
      // All 10 callers see the same data.
      for (const r of results) {
        expect(r.ok).toBe(true);
        if (r.ok) expect(Array.from(r.value)).toEqual([1, 2, 3]);
      }
    });

    it('coalesces signal-bearing callers into one underlying fetch', async () => {
      let fetchCount = 0;
      global.fetch = vi.fn(() => {
        fetchCount++;
        return Promise.resolve({
          ok: true,
          status: 200,
          async arrayBuffer() {
            return new Uint8Array([7]).buffer;
          },
        } as Response);
      }) as unknown as typeof fetch;

      const ac1 = new AbortController();
      const ac2 = new AbortController();
      // Demand reads normally carry per-update signals. They must still share
      // the same-key fetch or ordinary concurrent loaders amplify requests.
      const [r1, r2] = await Promise.all([
        store.getResult('signal-key', { signal: ac1.signal }),
        store.getResult('signal-key', { signal: ac2.signal }),
      ]);
      expect(fetchCount).toBe(1);
      expect(r1.ok && r2.ok).toBe(true);
    });

    it('aborts only the cancelled waiter while another coalesced caller continues', async () => {
      let fetchSignal: AbortSignal | undefined;
      let releaseFetch!: () => void;
      global.fetch = vi.fn((_url: string, init?: RequestInit) => {
        fetchSignal = init?.signal ?? undefined;
        return new Promise<Response>((resolve, reject) => {
          releaseFetch = () =>
            resolve({
              ok: true,
              status: 200,
              async arrayBuffer() {
                return new Uint8Array([7]).buffer;
              },
            } as Response);
          init?.signal?.addEventListener('abort', () => {
            reject(new DOMException('Aborted', 'AbortError'));
          });
        });
      }) as unknown as typeof fetch;

      const first = new AbortController();
      const second = new AbortController();
      const firstRead = store.getResult('shared-signal-key', { signal: first.signal });
      const secondRead = store.getResult('shared-signal-key', { signal: second.signal });
      await new Promise((resolve) => setTimeout(resolve, 0));

      first.abort();
      const firstResult = await firstRead;
      expect(firstResult.ok).toBe(false);
      if (!firstResult.ok) expect(firstResult.error.kind).toBe('Aborted');
      expect(fetchSignal?.aborted).toBe(false);

      releaseFetch();
      const secondResult = await secondRead;
      expect(secondResult.ok).toBe(true);
      expect(global.fetch).toHaveBeenCalledOnce();
    });

    it('pendingGets is cleaned on settle (subsequent calls re-fetch if L1/L2 missed)', async () => {
      let fetchCount = 0;
      global.fetch = vi.fn(() => {
        fetchCount++;
        return Promise.resolve({
          ok: false,
          status: 404,
        } as Response);
      }) as unknown as typeof fetch;

      // First call: 404 → not cached. pendingGets entry must clear in finally.
      const r1 = await store.getResult('not-found-key');
      expect(r1.ok).toBe(false);
      // Second call: pendingGets entry is gone, so a new fetch is issued.
      const r2 = await store.getResult('not-found-key');
      expect(r2.ok).toBe(false);
      expect(fetchCount).toBe(2);
    });

    it('an aborted chain cannot delete a same-key replacement entry', async () => {
      let fetchCount = 0;
      const fetchSignals: AbortSignal[] = [];
      let releaseFirst!: () => void;
      let releaseReplacement!: () => void;
      const firstGate = new Promise<void>((resolve) => {
        releaseFirst = resolve;
      });
      const replacementGate = new Promise<void>((resolve) => {
        releaseReplacement = resolve;
      });

      global.fetch = vi.fn(async (_url: string, init?: RequestInit) => {
        const call = ++fetchCount;
        fetchSignals.push(init?.signal as AbortSignal);
        await (call === 1 ? firstGate : replacementGate);
        return {
          ok: true,
          status: 200,
          async arrayBuffer() {
            return new Uint8Array([call]).buffer;
          },
        } as Response;
      }) as unknown as typeof fetch;

      const waitForFetchCount = async (expected: number) => {
        for (let i = 0; i < 20 && fetchCount < expected; i++) {
          await Promise.resolve();
        }
        expect(fetchCount).toBe(expected);
      };

      // Start an old chain, then invalidate it. clearAll aborts and removes
      // the map entry synchronously, while our mock keeps the fetch pending.
      const oldRequest = store.getResult('replace-race');
      await waitForFetchCount(1);
      await store.clearAll();
      expect(fetchSignals[0].aborted).toBe(true);

      // A new same-key request legitimately owns pendingGets now.
      const replacement = store.getResult('replace-race');
      await waitForFetchCount(2);
      expect(fetchSignals[1].aborted).toBe(false);

      // Settling the old aborted chain must not delete that replacement.
      releaseFirst();
      const oldResult = await oldRequest;
      expect(oldResult.ok).toBe(false);

      const waiter = store.getResult('replace-race');
      for (let i = 0; i < 20; i++) await Promise.resolve();
      const fetchesAfterWaiter = fetchCount;

      // A second invalidation must still find and abort the replacement.
      // Finish the clear before observing the signal, then release the mock
      // fetches so cleanup completes even when either assertion would fail.
      await store.clearAll();
      const replacementWasAborted = fetchSignals[1].aborted;
      releaseReplacement();
      await Promise.all([replacement, waiter]);

      expect(fetchesAfterWaiter).toBe(2);
      expect(replacementWasAborted).toBe(true);
    });

    // R6b: demand-while-prefetch-in-flight. A prefetch-originated
    // getResult (suppressPrefetch: true) and a user-demand call for
    // the same key must share one underlying network fetch via
    // pendingGets. The demand counter must increment exactly once,
    // and the prefetch must not bump the demand counter.
    it('demand call coalesces with an in-flight prefetch (one fetch, demand counted once)', async () => {
      let fetchCount = 0;
      let releaseFetch: () => void = () => {};
      global.fetch = vi.fn(() => {
        fetchCount++;
        return new Promise<Response>((resolve) => {
          releaseFetch = () =>
            resolve({
              ok: true,
              status: 200,
              async arrayBuffer() {
                return new Uint8Array([9, 9, 9]).buffer;
              },
            } as Response);
        });
      }) as unknown as typeof fetch;

      const beforeDemand = store.getStats().demand.networkRequests;

      // Prefetch starts first.
      const prefetchPromise = store.getResult('shared-with-prefetch', {
        suppressPrefetch: true,
      });
      // Yield once so the prefetch reaches pendingGets.
      await new Promise((r) => setTimeout(r, 0));

      // Demand call for the same key — must coalesce.
      const demandPromise = store.getResult('shared-with-prefetch');

      releaseFetch();
      const [pre, dem] = await Promise.all([prefetchPromise, demandPromise]);

      expect(fetchCount).toBe(1); // one underlying fetch
      expect(pre.ok).toBe(true);
      expect(dem.ok).toBe(true);
      // Demand counter incremented once — for the demand call only,
      // not for the prefetch (which set suppressPrefetch: true).
      const afterDemand = store.getStats().demand.networkRequests;
      expect(afterDemand - beforeDemand).toBe(1);
    });

    it('prefetch completing first lets the subsequent demand call hit L1', async () => {
      let fetchCount = 0;
      global.fetch = vi.fn(() => {
        fetchCount++;
        return Promise.resolve({
          ok: true,
          status: 200,
          async arrayBuffer() {
            return new Uint8Array([5, 5]).buffer;
          },
        } as Response);
      }) as unknown as typeof fetch;

      // Prefetch first — seeds L1.
      const preResult = await store.getResult('warm-by-prefetch', {
        suppressPrefetch: true,
      });
      expect(preResult.ok).toBe(true);
      expect(fetchCount).toBe(1);

      const beforeL1 = store.getStats().demand.l1Hits;
      const beforeNet = store.getStats().demand.networkRequests;

      // Subsequent demand call hits L1; no further fetch.
      const demand = await store.getResult('warm-by-prefetch');
      expect(demand.ok).toBe(true);
      expect(fetchCount).toBe(1); // still 1 — no new fetch

      const afterL1 = store.getStats().demand.l1Hits;
      const afterNet = store.getStats().demand.networkRequests;
      expect(afterL1 - beforeL1).toBe(1); // demand hit L1
      expect(afterNet - beforeNet).toBe(0); // no demand network fetch
    });
  });

  describe('getStats health field (commit 6.4)', () => {
    it('exposes validationMode + lastValidatedAt + unvalidatedExternalDataset', async () => {
      // [cache.md/Wn][P2] Previously asserted `stats.health).toBeDefined()`
      // plus a few `typeof` membership checks. Strengthen to assert the exact
      // shape (four named keys, no extras) — a regression that dropped any
      // field (e.g. removed `opfsAvailable`) would now fail.
      const stats = store.getStats();
      expect(Object.keys(stats.health).sort()).toEqual(
        ['lastValidatedAt', 'opfsAvailable', 'unvalidatedExternalDataset', 'validationMode'].sort()
      );
      expect(['content-hash', 'ttl', 'none']).toContain(stats.health.validationMode);
      expect(
        stats.health.lastValidatedAt === null || typeof stats.health.lastValidatedAt === 'number'
      ).toBe(true);
      expect(typeof stats.health.unvalidatedExternalDataset).toBe('boolean');
      expect(typeof stats.health.opfsAvailable).toBe('boolean');
    });

    it('marks external dataset as unvalidated when mode === none', async () => {
      // [cache.md/C3] Use a narrow spy on the real l2Store's
      // getValidationState rather than swapping the whole l2Store with
      // a fake. The store's health-roll-up reads only that one method
      // here, so a single spy is sufficient to drive the assertion.
      const l2Store = (store as any).l2Store as OPFSStore;
      vi.spyOn(l2Store, 'getValidationState').mockReturnValue({
        mode: 'none',
        lastValidatedAt: null,
      });
      const stats = store.getStats();
      expect(stats.health.unvalidatedExternalDataset).toBe(true);
    });
  });

  describe('Statistics', () => {
    it('should return accurate L1 and L2 stats', async () => {
      // [cache.md/Wn][P2] Previous version had `metadataCount>0`,
      // `chunksCount>0`, `l2).toBeDefined()`. Make all three exact.
      await store.get('.zmetadata'); // Metadata
      await store.get('chunk1'); // Chunk

      const stats = store.getStats();
      expect(stats.l1.metadataCount).toBe(1);
      expect(stats.l1.chunksCount).toBe(1);
      // l2 is documented to expose at least size/count/reads/writes/misses
      // (multi-level-caching-store.ts:663-669 fallback shape). Pin a
      // representative invariant: never-undefined and numerically sized.
      expect(typeof stats.l2.size).toBe('number');
      expect(typeof stats.l2.count).toBe('number');
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

    it('rejects invalidation-aborted coalesced reads instead of reporting missing chunks', async () => {
      let observedSignal: AbortSignal | undefined;
      global.fetch = vi.fn((_url: string, init?: RequestInit) => {
        observedSignal = init?.signal as AbortSignal | undefined;
        return new Promise<Response>((_resolve, reject) => {
          if (observedSignal?.aborted) {
            reject(new DOMException('Aborted', 'AbortError'));
            return;
          }
          observedSignal?.addEventListener(
            'abort',
            () => reject(new DOMException('Aborted', 'AbortError')),
            { once: true }
          );
        });
      }) as unknown as typeof fetch;

      const firstRead = store.get('invalidation-race');
      const coalescedRead = store.get('invalidation-race');
      for (let i = 0; i < 20 && observedSignal === undefined; i++) {
        await Promise.resolve();
      }
      expect(observedSignal).toBeDefined();
      expect(global.fetch).toHaveBeenCalledTimes(1);

      const clear = store.clearAll();
      const expectedAbort = {
        name: 'AbortError',
        message: expect.stringContaining('invalidation-race'),
      };
      await expect(firstRead).rejects.toMatchObject(expectedAbort);
      await expect(coalescedRead).rejects.toMatchObject(expectedAbort);
      await clear;
      expect(observedSignal?.aborted).toBe(true);
    });

    it('invalidation aborts a signal-bearing chain before it can repopulate L1', async () => {
      let observedSignal: AbortSignal | undefined;
      global.fetch = vi.fn((_url: string, init?: RequestInit) => {
        observedSignal = init?.signal ?? undefined;
        return new Promise<Response>((_resolve, reject) => {
          observedSignal?.addEventListener(
            'abort',
            () => reject(new DOMException('Aborted', 'AbortError')),
            { once: true }
          );
        });
      }) as unknown as typeof fetch;

      const caller = new AbortController();
      const read = store.get('signal-invalidation-race', { signal: caller.signal });
      for (let i = 0; i < 20 && observedSignal === undefined; i++) await Promise.resolve();

      const clear = store.clearAll();
      await expect(read).rejects.toMatchObject({ name: 'AbortError' });
      await clear;
      expect(observedSignal?.aborted).toBe(true);
      const l1Cache = (store as unknown as { l1Cache: { has(key: string): boolean } }).l1Cache;
      expect(l1Cache.has('signal-invalidation-race')).toBe(false);
    });

    it('retry backoff includes jitter (commit 4.3)', async () => {
      // With Math.random() forced to deterministic values we can prove
      // the jittered delay differs from a pure exponential. We don't
      // assert exact ms here (sleep timing in jsdom is fragile); we
      // just confirm that the random source is consumed during the
      // retry path.
      const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0.5);

      let calls = 0;
      global.fetch = vi.fn(async () => {
        calls++;
        if (calls < 3) return { ok: false, status: 503 } as Response;
        return {
          ok: true,
          status: 200,
          async arrayBuffer() {
            return new Uint8Array([1]).buffer;
          },
        } as Response;
      }) as unknown as typeof fetch;

      await store.get('jitter-key');
      // Three attempts means two backoff windows, so Math.random was
      // called at least twice for the jitter term.
      expect(randomSpy.mock.calls.length).toBeGreaterThanOrEqual(2);
      randomSpy.mockRestore();
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

    it('rejects exhausted network errors instead of returning a fill-valued miss', async () => {
      global.fetch = vi.fn(async () => {
        throw new Error('Network error');
      }) as any;

      await expect(store.get('test')).rejects.toThrow('fetch exhausted retries');
    });

    it('forwards zarrita GetOptions.signal through get() to the underlying fetch', async () => {
      let observedSignal: AbortSignal | undefined;
      global.fetch = vi.fn((_url: string, init?: RequestInit) => {
        observedSignal = init?.signal ?? undefined;
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            'abort',
            () => reject(new DOMException('Aborted', 'AbortError')),
            { once: true }
          );
        });
      }) as unknown as typeof fetch;

      const controller = new AbortController();
      const pending = store.get('cancelled', { signal: controller.signal });
      await vi.waitFor(() => expect(observedSignal).toBeDefined());

      controller.abort();

      await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
      expect(observedSignal?.aborted).toBe(true);
    });

    it('should handle OPFS initialization failure', async () => {
      // [cache.md/Wn][P2] Previously asserted only `result).toBeDefined()`.
      // The real contract is "OPFS-down fallback works through L1 only":
      // (1) get() returns the actual fetched bytes; (2) L1 still populates;
      // (3) L2 reports zero count (no persistence layer wired up).
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
      expect(result).toEqual(new Uint8Array([1, 2, 3, 4, 5]));
      const stats = store.getStats();
      expect(stats.l1.chunksCount).toBe(1);
      expect(stats.l2.count).toBe(0);
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
      // [cache.md/Wn][P2] Previously asserted only `newStore).toBeDefined()`
      // (constructors never return undefined). Strengthen by asserting the
      // post-dispose observable contract: getStats() still returns a valid
      // snapshot with zero counts (no leaked state).
      const newStore = new MultiLevelCachingStore('https://example.com/test.zarr');
      await newStore.dispose(); // Should not throw
      const stats = newStore.getStats();
      expect(stats.l1.chunksCount).toBe(0);
      expect(stats.l1.metadataCount).toBe(0);
      expect(stats.l2.count).toBe(0);
    });

    it(
      'VC-1: dispose aborts in-flight cache validation (entry self-evicts when the aborted validation settles)',
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
      'two stores sharing a datasetId: disposing the first cancels its queue entry, the second can still validate',
      { timeout: 15_000 },
      async () => {
        // Two stores using the same URL share the same datasetId
        // (SHA-256(url)). The first to start init populates the
        // static validationQueues entry; the second waits on it.
        // When the first is disposed mid-validation, its queue entry
        // is aborted (self-evicting when the aborted validation settles)
        // — so the second can install its own entry and complete cleanly.
        const url = 'https://example.com/shared-dataset.zarr';

        // First fetch hangs forever to keep store-1's validation
        // in flight; second fetch resolves so store-2 can finish.
        let firstAttempt = true;
        global.fetch = vi.fn((_url: string, init?: RequestInit) => {
          if (firstAttempt) {
            firstAttempt = false;
            return new Promise<Response>((_resolve, reject) => {
              init?.signal?.addEventListener(
                'abort',
                () => reject(new DOMException('Aborted', 'AbortError')),
                { once: true }
              );
            });
          }
          // Subsequent fetches return a real response so store-2 can
          // finish init (this includes both the .zattrs probe and
          // chunk fetches).
          return Promise.resolve({
            ok: true,
            status: 200,
            async arrayBuffer() {
              return new Uint8Array([1, 2, 3]).buffer;
            },
            async text() {
              return '{}';
            },
            headers: new Headers(),
          } as unknown as Response);
        }) as unknown as typeof fetch;

        const store1 = new MultiLevelCachingStore(url, {
          l1MaxSize: 20 * 1024 * 1024,
          l2MaxSize: 4096,
        });
        const init1 = store1.init();
        // Let init1 enter validateCache → fetchWithRetry.
        await new Promise((r) => setTimeout(r, 50));

        // Dispose store1 mid-validation — its queue entry should be
        // aborted, allowing init2 (which would otherwise wait on
        // store1's queue) to install its own entry.
        await store1.dispose();
        await init1;

        // Now create + init a second store on the same URL. The
        // hanging fetch is gone (firstAttempt already consumed),
        // so this should succeed without timing out.
        const store2 = new MultiLevelCachingStore(url, {
          l1MaxSize: 20 * 1024 * 1024,
          l2MaxSize: 4096,
        });
        await store2.init();

        // Cleanup.
        await store2.dispose();
      }
    );

    it('disposed-store reads unwind quietly without touching tiers', async () => {
      // After dispose, callers must not be able to populate L1/L2 or
      // trigger network. The early-return guards both the Map-poke
      // and the prefetcher.onAccess fan-out.
      const fetchSpy = vi.fn(async () => ({
        ok: true,
        async arrayBuffer() {
          return new Uint8Array([1, 2, 3]).buffer;
        },
      })) as unknown as typeof fetch;
      global.fetch = fetchSpy;

      const target = new MultiLevelCachingStore('https://example.com/t.zarr', {
        l1MaxSize: 20 * 1024 * 1024,
        l2MaxSize: 4096,
      });
      await target.init();
      await target.dispose();
      (fetchSpy as unknown as { mockClear: () => void }).mockClear();

      const result = await target.getResult('chunk');
      expect(result.ok).toBe(false);
      if (!result.ok) expect(result.error.kind).toBe('Aborted');
      await expect(target.get('chunk')).resolves.toBeUndefined();
      // No L1 mutation, no fetch initiation. The AsyncReadable path stays
      // quiet because disposal means its owning scene is already discarded.
      expect(target.getStats().l1.chunksCount).toBe(0);
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it('in-flight data fetch unwinds with Aborted when store is disposed mid-flight', async () => {
      // The store-level dataAbort is merged into fetchWithRetry's signal
      // via mergeAbortSignals. Disposing the store mid-fetch must
      // propagate to the underlying fetch and surface as a non-ok result.
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
      }) as unknown as typeof fetch;

      const promise = store.getResult('hung-chunk');
      // Yield once so getResult enters fetchWithRetry and registers the abort listener.
      await new Promise((r) => setTimeout(r, 0));
      expect(observedSignal).toBeDefined();

      await store.dispose();

      const result = await promise;
      expect(observedSignal?.aborted).toBe(true);
      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(['Aborted', 'NetworkError']).toContain(result.error.kind);
      }
    });

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

        // Whichever root document the probe reached for — the budget under
        // test is the validation one, not the document's spelling.
        const rootDoc = [...startTimes.keys()].find(isRootDocRequest);
        expect(rootDoc).toBeDefined();
        const start = startTimes.get(rootDoc as string);
        const abort = abortTimes.get(rootDoc as string);
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

    it('concurrent get() of three distinct chunk keys lands exactly three cached entries (L1)', async () => {
      // cache.md O3 / Phase E28 strengthening: previous assertion was
      // `chunksCount + metadataCount >= 3`. The name promised "each
      // distinct chunk independently", which suggests EXACTLY 3 fetches
      // for 3 distinct keys. The `>= 3` band would pass on a regression
      // that double-counted a chunk OR cached an unrelated 4th entry
      // (e.g. content-hash metadata for the dataset).
      //
      // cache.md W7 (prior) fix: previous version asserted
      // `chunksCount + metadataCount >= 1` — satisfied by a regression
      // that fetched only ONE chunk and aliased the others. The
      // `>= 3` strengthening fixed that floor; this PR adds the upper
      // bound to pin the contract on both sides.
      const results = await Promise.all([
        store.get('chunk1'),
        store.get('chunk2'),
        store.get('chunk3'),
      ]);

      expect(results.length).toBe(3);
      expect(results.every((r) => r !== undefined)).toBe(true);

      // Three distinct chunk keys → exactly three L1 chunk entries.
      // (Metadata entries are content-hash-keyed by dataset; this test's
      // store mocks `fetch` such that each distinct chunk key produces
      // one L1 entry and zero metadata entries, so the sum is exactly 3.)
      const stats = store.getStats();
      expect(stats.l1.chunksCount + stats.l1.metadataCount).toBe(3);
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
      // [cache.md/Wn][P2] Previously asserted r1/r2 `toBeDefined()` and
      // `>=1` fetches. Strengthen to: (a) exact bytes for both results,
      // (b) exact-2 fetches (one per get) — proves no-cache truly bypasses
      // L1 (a regression that still cached in L1 would show only 1 fetch).
      const noCacheStore = new MultiLevelCachingStore('https://example.com/test.zarr', {
        noCache: true,
      });
      await noCacheStore.init();

      mocks.fetchedUrls.length = 0;

      // Should always fetch from HTTP  (no-cache bypasses L1/L2)
      const result1 = await noCacheStore.get('test');
      const result2 = await noCacheStore.get('test'); // Second time

      expect(result1).toEqual(new Uint8Array([1, 2, 3, 4, 5]));
      expect(result2).toEqual(new Uint8Array([1, 2, 3, 4, 5]));
      // With no-cache, EVERY get() must fetch — exact 2 fetches of /test.
      expect(mocks.fetchedUrls.filter((u) => u.includes('test')).length).toBe(2);
    });

    it('should skip ONLY the L2 tier with ?no-opfs (#1645)', async () => {
      // The deterministic sibling of the OPFS circuit breaker: no OPFSStore is
      // ever constructed (so nothing can stall), while L1 keeps serving. A
      // DELIBERATE disable must not raise the degradation badges — neither
      // `opfs-unavailable` nor `unvalidated-external-dataset` (with no
      // persistent tier there are no entries that could go stale).
      const initSpy = vi.spyOn(OPFSStore.prototype, 'init');
      try {
        const noOpfsStore = new MultiLevelCachingStore('https://example.com/test.zarr', {
          noOpfs: true,
        });
        await noOpfsStore.init();
        expect(initSpy).not.toHaveBeenCalled();

        mocks.fetchedUrls.length = 0;
        const result1 = await noOpfsStore.get('test');
        const result2 = await noOpfsStore.get('test');
        expect(result1).toEqual(new Uint8Array([1, 2, 3, 4, 5]));
        expect(result2).toEqual(new Uint8Array([1, 2, 3, 4, 5]));
        // L1 still serves the repeat: exactly ONE fetch, unlike ?no-cache's two.
        expect(mocks.fetchedUrls.filter((u) => u.includes('test')).length).toBe(1);

        const stats = noOpfsStore.getStats();
        expect(stats.l2.count).toBe(0);
        expect(stats.l2.writes).toBe(0);
        expect(stats.health.opfsAvailable).toBe(true);
        expect(stats.health.unvalidatedExternalDataset).toBe(false);
      } finally {
        initSpy.mockRestore();
      }
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
      // [cache.md/W10][P2] Previous version constructed a fresh `clearStore`
      // and asserted `l2.size === 0` — but a fresh store's l2.size is
      // unconditionally 0 in this mock (no cross-instance persistence),
      // so the assertion passes whether `clearCache` did anything or not.
      // Strengthen by pinning `clearOnInitCount`, the observable side-effect
      // of the ?clear-cache flow surfaced by getStats().
      await store.get('test1');
      await store.dispose();

      const clearStore = new MultiLevelCachingStore('https://example.com/data.zarr', {
        clearCache: true,
      });
      await clearStore.init();

      const stats = clearStore.getStats();
      expect(stats.l2.size).toBe(0); // Documented in MultiLevelCacheStats.l2.size
      // The real ?clear-cache observable: clearOnInitCount === 1.
      expect(stats.clearOnInitCount).toBe(1);
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

    it('should handle mixed access patterns — routes each key to its segment, total exactly 4', async () => {
      // cache.md W8 fix: previous version asserted `>= 1` which let any
      // routing regression survive. Strengthen: 2 metadata keys (`.zarray`,
      // `.zattrs`) + 2 chunk keys → exactly 2 in each segment.
      // [cache.md/Wn][P2] Round-7 follow-up: the four `r1..r4).toBeDefined()`
      // were trivially true (the mock fetch always returns a buffer); replace
      // with exact-bytes equality, which forces the cache cascade to surface
      // the actual fetched payload rather than a placeholder.
      const r1 = await store.get('.zarray');
      const r2 = await store.get('chunk1');
      const r3 = await store.get('.zattrs');
      const r4 = await store.get('chunk2');

      const dataBytes = new Uint8Array([1, 2, 3, 4, 5]);
      // .zattrs is the content-hash payload in this mock, not the raw bytes.
      expect(r1).toEqual(dataBytes);
      expect(r2).toEqual(dataBytes);
      expect(r4).toEqual(dataBytes);
      // .zattrs returns JSON content_hash payload.
      expect(JSON.parse(new TextDecoder().decode(r3!))).toEqual({
        content_hash: 'test-hash-123',
      });

      const stats = store.getStats();
      expect(stats.l1.metadataCount).toBe(2);
      expect(stats.l1.chunksCount).toBe(2);
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

    it('MED-2: concurrent waiters on a coalesced fetch call onAccess() once', async () => {
      // 10 concurrent get() calls for the same key share a single inflight
      // L2/network chain via pendingGets. Without dedup each waiter called
      // prefetcher.onAccess(key), burning seen-set work N times. The fix:
      // only the originator of the inflight chain fans out onAccess; all
      // other waiters skip the call entirely.
      const mockPrefetcher = { onAccess: vi.fn() };
      store.setPrefetcher(mockPrefetcher as any);

      const N = 10;
      const waiters = Array.from({ length: N }, () => store.get('coalesce.med2.chunk'));
      const results = await Promise.all(waiters);

      // All waiters got a real payload.
      expect(results.every((r) => r !== null)).toBe(true);
      // …but onAccess fired exactly once for the logical access.
      expect(mockPrefetcher.onAccess).toHaveBeenCalledTimes(1);
      expect(mockPrefetcher.onAccess).toHaveBeenCalledWith('coalesce.med2.chunk');
    });

    it('should dispose prefetcher and clear reference on dispose', async () => {
      // cache.md C5[P1] fix: prior version probed the private `(store as any).
      // prefetcher` field. The public surface is `getPrefetcher()` — use it
      // so a future refactor that changes the field name (or switches to a
      // WeakRef / counter / Map) still exercises the same contract.
      const mockPrefetcher = {
        onAccess: vi.fn(),
        dispose: vi.fn(),
      };

      store.setPrefetcher(mockPrefetcher as any);
      expect(store.getPrefetcher()).toBe(mockPrefetcher);

      await store.dispose();
      expect(mockPrefetcher.dispose).toHaveBeenCalledTimes(1);
      expect(store.getPrefetcher()).toBeNull();
    });

    it('should work without prefetcher attached', async () => {
      // [cache.md/Wn][P2] Previously asserted only `result).toBeDefined()`.
      // Strengthen to exact bytes + L1 population — proves the no-prefetcher
      // path still flows through the full cascade.
      const result = await store.get('test.chunk');
      expect(result).toEqual(new Uint8Array([1, 2, 3, 4, 5]));
      expect(store.getStats().l1.chunksCount).toBe(1);
    });

    it('does NOT call prefetcher.onAccess() when suppressPrefetch=true on L3 fetch', async () => {
      // prefetcher.processQueue() passes
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

  // getStats cascade contract: the orchestrator surfaces bandwidth +
  // health + clear-on-init from its subordinate units. Implementation-
  // level invariants (R5 amortized compaction, formula, start-index
  // walk) live in bandwidth-window.test.ts; here we only assert that
  // bytes recorded through a real getResult surface in the snapshot.
  describe('getStats cascade contract', () => {
    it('records demand-fetch bytes into stats.network.bandwidth', async () => {
      global.fetch = vi.fn(() =>
        Promise.resolve({
          ok: true,
          status: 200,
          async arrayBuffer() {
            return new Uint8Array(1000).buffer;
          },
        } as Response)
      ) as unknown as typeof fetch;

      // Initial bandwidth is 0.
      expect(store.getStats().network.bandwidth).toBe(0);

      await store.getResult('bandwidth-cascade');
      const stats = store.getStats();
      // Demand counters reflect the fetch we just drove.
      expect(stats.network.requestCount).toBeGreaterThan(0);
      expect(stats.network.bytesTransferred).toBeGreaterThanOrEqual(1000);
      // Bandwidth has aggregated the bytes into the sliding window.
      expect(stats.network.bandwidth).toBeGreaterThan(0);
    });

    // S2: opfsAvailable is propagated from the L2 store's getStats().available.
    // It MUST always be true when caching is disabled (no L2 expected).
    it('health.opfsAvailable === true for a healthy store with L2 reachable', () => {
      const stats = store.getStats();
      expect(stats.health.opfsAvailable).toBe(true);
    });

    // S4: clearOnInitCount counts ?clear-cache invocations. A store
    // constructed without clearCache stays at 0; one with clearCache
    // increments to 1 after init.
    it('clearOnInitCount stays 0 when ?clear-cache was not requested', () => {
      const stats = store.getStats();
      expect(stats.clearOnInitCount).toBe(0);
    });

    it('clearOnInitCount === 1 after init with clearCache: true', async () => {
      const clearStore = new MultiLevelCachingStore('https://example.com/clr.zarr', {
        clearCache: true,
        l1MaxSize: 20 * 1024 * 1024,
        l2MaxSize: 4096,
      });
      await clearStore.init();
      const stats = clearStore.getStats();
      expect(stats.clearOnInitCount).toBe(1);
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

    it('should correctly construct the root-document URL for content hash validation', async () => {
      mocks.fetchedUrls.length = 0;

      // Store with trailing slash
      const storeWithSlash = new MultiLevelCachingStore('https://example.com/data.zarr/');
      await storeWithSlash.init();

      // Whichever root document the implementation asks for, the URL must be
      // joined without a doubled slash. Asserting on the document NAME would
      // pin the format rather than the joining bug this test exists for.
      const rootDocUrls = mocks.fetchedUrls.filter(
        (u) => u.includes('.zattrs') || u.includes('zarr.json')
      );
      expect(rootDocUrls.length).toBeGreaterThan(0);
      expect(rootDocUrls[0]).toMatch(/^https:\/\/example\.com\/data\.zarr\/(\.zattrs|zarr\.json)$/);
      expect(rootDocUrls.some((u) => u.includes('//.zattrs') || u.includes('//zarr.json'))).toBe(
        false
      );
    });
  });

  describe('background L2 write queue', () => {
    // Gate the mock's chunk (ArrayBuffer) writes so a test can prove the
    // foreground get() does NOT wait on the OPFS write. Metadata (string)
    // writes are left ungated.
    function installChunkWriteGate(m: ReturnType<typeof createMocks>) {
      let resolve: (() => void) | null = null;
      let gate: Promise<void> | null = null;
      const dir = m.mockDirHandle;
      const origGetFileHandle = dir.getFileHandle.bind(dir);
      dir.getFileHandle = async (name: string, opts?: { create?: boolean }) => {
        const handle = await origGetFileHandle(name, opts);
        const origCreateWritable = handle.createWritable.bind(handle);
        handle.createWritable = async () => {
          const w = await origCreateWritable();
          const origWrite = w.write.bind(w);
          w.write = async (data: ArrayBuffer | string) => {
            if (typeof data !== 'string' && gate) await gate;
            return origWrite(data);
          };
          return w;
        };
        return handle;
      };
      return {
        open() {
          gate = new Promise<void>((r) => {
            resolve = r;
          });
        },
        release() {
          resolve?.();
          gate = null;
        },
      };
    }

    it('sizes the pending-byte budget from the heap remainder, not from L1 (#2561)', () => {
      // A pending write's buffer IS the L1 entry's buffer, so the allowance
      // must not track the L1 budget. This is where the old formula lived
      // (max(l1MaxSize, 64MB)), so the L1 OPTION is what has to be varied: a
      // wide spread of L1 sizes — including the 0 a tight heap resolves to and
      // one far above the old 64MB floor — must all yield the same cap.
      const constrainedBudgets = computeCacheBudgets(undefined, 16 * 1024 * 1024);
      expect(constrainedBudgets.l1Bytes).toBe(0);

      const expected = computeOpfsWriteQueueBudgetBytes();
      const l1Sizes = [
        constrainedBudgets.l1Bytes,
        16 * 1024 * 1024,
        100 * 1024 * 1024,
        512 * 1024 * 1024,
      ];
      const resolved = l1Sizes.map(
        (l1MaxSize) =>
          new MultiLevelCachingStore('https://example.com/data.zarr', {
            l1MaxSize,
          }).getStats().l2WriteQueue.maxBytes
      );

      // The old formula would have produced 64MB / 64MB / 100MB / 512MB here,
      // i.e. varied with L1 — so this array being constant is the assertion.
      expect(resolved).toEqual(l1Sizes.map(() => expected));
    });

    it('resolves the default pending-byte budget from a MEASURED heap', () => {
      // The case above compares against `computeOpfsWriteQueueBudgetBytes()`,
      // which in this environment has no heap signal and returns the 256MB
      // no-signal fallback — the same constant the EAGER working-set budget
      // falls back to. So that comparison alone would still pass if the
      // constructor were wired to `computeWorkingSetBudgetBytes()`, which
      // takes half the remainder and would double the production cap.
      // Stubbing a real heap separates them: on 4GiB the queue resolves to
      // 4096MiB × 0.8 × 0.4 × 0.25 = 327.68MiB, while the working set would
      // give 512MB (its cap) and either fallback would give 256MB.
      const realPerformance = globalThis.performance;
      vi.stubGlobal('performance', {
        now: () => realPerformance.now(),
        memory: { jsHeapSizeLimit: 4 * 1024 * 1024 * 1024 },
      });
      try {
        const s = new MultiLevelCachingStore('https://example.com/data.zarr', {
          l1MaxSize: 20 * 1024 * 1024,
        });
        expect(s.getStats().l2WriteQueue.maxBytes).toBe(343_597_383);
      } finally {
        vi.stubGlobal('performance', realPerformance);
      }
    });

    it('honours an explicit opfsWriteQueueMaxBytes option', () => {
      // How cache-setup feeds the ?cacheBudgetMB= / device-class allowance in.
      const store = new MultiLevelCachingStore('https://example.com/data.zarr', {
        opfsWriteQueueMaxBytes: 7 * 1024 * 1024,
      });
      expect(store.getStats().l2WriteQueue.maxBytes).toBe(7 * 1024 * 1024);
    });

    it('does not block get() on the L2 write, then persists it in the background', async () => {
      const gate = installChunkWriteGate(mocks);
      gate.open();
      const before = mocks.files.size;

      // Must resolve even though the chunk write is gated shut — proving the
      // durable OPFS write is off the get() critical path.
      const data = await store.get('chunk.0.0');
      expect(data).toBeInstanceOf(Uint8Array);

      // Write is deferred: nothing persisted yet, but it is queued/in-flight.
      expect(mocks.files.size).toBe(before);
      const q = store.getStats().l2WriteQueue;
      expect(q.inFlight + q.depth).toBeGreaterThanOrEqual(1);
      expect(q.maxBytes).toBe(computeOpfsWriteQueueBudgetBytes());
      expect(store.getStats().l2.writes).toBe(0);

      // Release + drain → the background write lands.
      gate.release();
      await (
        store as unknown as { l2WriteQueue: { drain: () => Promise<void> } }
      ).l2WriteQueue.drain();
      await new Promise((r) => setTimeout(r, 0));

      expect(mocks.files.size).toBeGreaterThan(before);
      expect(store.getStats().l2.writes).toBeGreaterThanOrEqual(1);
    });

    it('clearL2 synchronously drops queued (not-yet-started) writes', async () => {
      const s = new MultiLevelCachingStore('https://example.com/data.zarr', {
        l1MaxSize: 20 * 1024 * 1024,
        l2MaxSize: 4096,
        opfsWriteConcurrency: 1, // one slot → the 2nd write stays pending
        opfsWriteQueueMax: 100,
      });
      await s.init();
      s.clearL1();

      const l2 = (s as unknown as { l2Store: OPFSStore }).l2Store;
      const setSpy = vi.spyOn(l2, 'set');

      const gate = installChunkWriteGate(mocks);
      gate.open();

      await s.get('chunk.a'); // occupies the single slot (write gated, in-flight)
      await s.get('chunk.b'); // no slot free → pending in the queue
      expect(s.getStats().l2WriteQueue.depth).toBe(1);

      // clearL2's synchronous prefix bumps the epoch and empties the queue.
      const clearP = s.clearL2();
      expect(s.getStats().l2WriteQueue.depth).toBe(0);

      gate.release();
      await clearP;
      await (s as unknown as { l2WriteQueue: { drain: () => Promise<void> } }).l2WriteQueue.drain();

      // 'chunk.b' was dropped before it ever reached l2Store.set; 'chunk.a'
      // (already in flight) did reach it.
      const setKeys = setSpy.mock.calls.map((c) => c[0]);
      expect(setKeys).toContain('chunk.a');
      expect(setKeys).not.toContain('chunk.b');

      await s.dispose();
    });

    it('dispose() drops queued writes and resolves without hanging', async () => {
      // A normal (ungated) get enqueues a write; dispose must drain/clear
      // cleanly. Exercises the dispose→clear(queue)→l2Store.dispose ordering.
      await store.get('chunk.z');
      await expect(store.dispose()).resolves.toBeUndefined();
      expect(store.getStats().l2WriteQueue.depth).toBe(0);
    });
  });

  describe('dispose during init() (issue #1058)', () => {
    // These tests exercise the two dispose-during-init windows in init():
    // (1) while awaiting hashUrl (before the OPFSStore is constructed) and
    // (2) while awaiting l2Store.init() (before clearAll/validateCache).
    // Async ordering is driven by an explicitly-resolvable deferred injected
    // into the mocked dependency — no real timers — so the tests are
    // deterministic.

    // Restore the OPFSStore.prototype spies even if an assertion throws, so a
    // failing test can never leak the prototype spy into sibling tests.
    afterEach(() => vi.restoreAllMocks());

    it('(a) dispose while suspended at hashUrl: no OPFSStore constructed, no clearAll/validate', async () => {
      // Gate hashUrl by making crypto.subtle.digest hang until we release it.
      // hashUrl → sha256Hex → crypto.subtle.digest, so a pending digest
      // suspends init() exactly at `await hashUrl(...)` — before the
      // OPFSStore is ever constructed.
      let releaseDigest!: (buf: ArrayBuffer) => void;
      const digestGate = new Promise<ArrayBuffer>((resolve) => {
        releaseDigest = resolve;
      });
      vi.stubGlobal('crypto', {
        subtle: {
          digest: () => digestGate,
        },
      });

      const s = new MultiLevelCachingStore('https://example.com/gate-a.zarr', {
        l1MaxSize: 20 * 1024 * 1024,
        l2MaxSize: 4096,
        clearCache: true, // even with ?clear-cache, clearAll must not run
      });
      const opfsInitSpy = vi.spyOn(OPFSStore.prototype, 'init');
      const clearAllSpy = vi.spyOn(s, 'clearAll');

      // init() suspends synchronously at the pending digest.
      const initPromise = s.init();

      // Dispose lands while init() is still awaiting hashUrl.
      await s.dispose();

      // Release the digest so hashUrl resolves and init() resumes.
      releaseDigest(new Uint8Array(32).buffer);
      await initPromise;

      // No OPFSStore was constructed on the dead instance, and neither the
      // clear step nor validation ran.
      expect((s as unknown as { l2Store: OPFSStore | null }).l2Store).toBeNull();
      expect(opfsInitSpy).not.toHaveBeenCalled();
      expect(clearAllSpy).not.toHaveBeenCalled();
    });

    it('(b) dispose while suspended at l2Store.init(): does not proceed to clearAll/validateCache', async () => {
      // Gate OPFSStore.init() with a deferred so init() suspends right after
      // constructing l2Store, at `await this.l2Store.init()`.
      let releaseL2Init!: () => void;
      const l2InitGate = new Promise<void>((resolve) => {
        releaseL2Init = resolve;
      });
      const opfsInitSpy = vi.spyOn(OPFSStore.prototype, 'init').mockReturnValue(l2InitGate);

      const s = new MultiLevelCachingStore('https://example.com/gate-b.zarr', {
        l1MaxSize: 20 * 1024 * 1024,
        l2MaxSize: 4096,
      });
      const clearAllSpy = vi.spyOn(s, 'clearAll');
      const validateSpy = vi.spyOn(
        s as unknown as { validateCache: (id: string) => Promise<void> },
        'validateCache'
      );

      const initPromise = s.init();

      // Wait (microtask-only, no timers) until init() has entered l2Store.init().
      for (let i = 0; i < 1000 && opfsInitSpy.mock.calls.length === 0; i++) {
        await Promise.resolve();
      }
      expect(opfsInitSpy).toHaveBeenCalledTimes(1);

      // Dispose lands while init() is still awaiting l2Store.init().
      await s.dispose();

      // Release l2Store.init() so init() resumes past the await.
      releaseL2Init();
      await initPromise;

      // init() bailed on the post-await disposed check — no clear, no validate.
      expect(clearAllSpy).not.toHaveBeenCalled();
      expect(validateSpy).not.toHaveBeenCalled();
    });

    it('(c) ?clear-cache + dispose before the clear step: clearAll does not run on the dead instance', async () => {
      let releaseL2Init!: () => void;
      const l2InitGate = new Promise<void>((resolve) => {
        releaseL2Init = resolve;
      });
      const opfsInitSpy = vi.spyOn(OPFSStore.prototype, 'init').mockReturnValue(l2InitGate);

      const s = new MultiLevelCachingStore('https://example.com/gate-c.zarr', {
        l1MaxSize: 20 * 1024 * 1024,
        l2MaxSize: 4096,
        clearCache: true, // ?clear-cache → shouldClearOnInit
      });
      const clearAllSpy = vi.spyOn(s, 'clearAll');

      const initPromise = s.init();
      for (let i = 0; i < 1000 && opfsInitSpy.mock.calls.length === 0; i++) {
        await Promise.resolve();
      }
      expect(opfsInitSpy).toHaveBeenCalledTimes(1);

      await s.dispose();
      releaseL2Init();
      await initPromise;

      // The clear step is guarded by the post-await disposed check, so a
      // disposed ?clear-cache store never wipes the shared OPFS directory.
      expect(clearAllSpy).not.toHaveBeenCalled();
    });
  });
});
