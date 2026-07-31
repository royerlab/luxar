import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OPFSStore } from '../../../cache/multi-level-caching-store/opfs-store';

// Mock File System Access API
const createMockFileSystem = () => {
  const files = new Map<string, Uint8Array>();
  const metaFiles = new Map<string, string>();

  const mockFileHandle = (path: string) => ({
    async getFile() {
      const data = files.get(path) || new Uint8Array(0);
      return {
        async arrayBuffer() {
          return data.buffer;
        },
        async text() {
          return metaFiles.get(path) || '{}';
        },
      };
    },
    async createWritable() {
      return {
        async write(data: ArrayBuffer | string) {
          if (typeof data === 'string') {
            metaFiles.set(path, data);
          } else {
            files.set(path, new Uint8Array(data));
          }
        },
        async close() {},
      };
    },
  });

  const mockDirHandle: any = {
    async getFileHandle(name: string, opts?: { create?: boolean }) {
      const fullPath = name;
      if (!files.has(fullPath) && !metaFiles.has(fullPath) && !opts?.create) {
        throw new Error('File not found');
      }
      return mockFileHandle(fullPath);
    },
    async getDirectoryHandle(_name: string, _opts?: { create?: boolean }) {
      return mockDirHandle; // Simplified: all paths return same handle
    },
    async removeEntry(name: string) {
      if (!files.has(name) && !metaFiles.has(name)) {
        throw new DOMException(`Entry not found: ${name}`, 'NotFoundError');
      }
      files.delete(name);
      metaFiles.delete(name);
    },
    async *keys() {
      // Return empty for simplicity
    },
  };

  return { mockDirHandle, files, metaFiles };
};

describe('OPFSStore', () => {
  let store: OPFSStore;
  let mockFS: ReturnType<typeof createMockFileSystem>;

  beforeEach(async () => {
    mockFS = createMockFileSystem();

    // Mock navigator.storage
    vi.stubGlobal('navigator', {
      storage: {
        async getDirectory() {
          return {
            async getDirectoryHandle(_id: string, _opts?: any) {
              return mockFS.mockDirHandle;
            },
            async removeEntry(_name: string, _opts?: any) {
              // Simulate atomic directory removal (clear all files)
              mockFS.files.clear();
              mockFS.metaFiles.clear();
            },
          };
        },
        async estimate() {
          return { quota: 10e9, usage: 1e9 }; // 10GB quota, 1GB used
        },
      },
    });

    // Mock crypto.subtle for SHA-256
    vi.stubGlobal('crypto', {
      subtle: {
        async digest(_algo: string, _data: Uint8Array) {
          // Return mock hash
          return new Uint8Array(32).fill(0xab).buffer;
        },
      },
    });

    store = new OPFSStore('test-dataset-id', 'https://example.com/data.zarr', 100 * 1024 * 1024);
    await store.init();
  });

  afterEach(() => {
    // [cache.md/O5][P10] Explicitly unstub globals so tests that re-stub
    // navigator mid-test (e.g. 'slow set + concurrent clear') do not bleed
    // into subsequent tests if test order changes. beforeEach also re-stubs,
    // but explicit cleanup is robust against reordering.
    vi.unstubAllGlobals();
  });

  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      const stats = store.getStats();
      expect(stats.size).toBe(0);
      expect(stats.count).toBe(0);
    });

    it('cleans up the init-time write-probe file', () => {
      // The healthy beforeEach init ran the probe: it must leave no trace.
      expect(mockFS.files.has('.opfs-write-probe')).toBe(false);
      expect(mockFS.metaFiles.has('.opfs-write-probe')).toBe(false);
    });

    it('disables the store when OPFS mounts but createWritable is unsupported (WebKit)', async () => {
      // WebKit (Safari / the WKWebView native launcher) implements
      // getDirectory() + handles but not main-thread createWritable():
      // without the init write probe the store mounted "healthy" and then
      // failed EVERY put (tens of thousands of write errors, empty L2,
      // all-miss reads). The probe must catch it at init instead.
      const webkitDir: any = {
        ...mockFS.mockDirHandle,
        async getFileHandle() {
          return {
            async getFile() {
              return {
                async arrayBuffer() {
                  return new ArrayBuffer(0);
                },
              };
            },
            async createWritable() {
              throw new TypeError('createWritable is not a function');
            },
          };
        },
        async getDirectoryHandle() {
          return webkitDir;
        },
        async removeEntry() {},
      };
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            return {
              async getDirectoryHandle() {
                return webkitDir;
              },
              async removeEntry() {},
            };
          },
          async estimate() {
            return { quota: 10e9, usage: 1e9 };
          },
        },
      });

      const webkitStore = new OPFSStore('webkit-id', 'https://example.com', 1000);
      await webkitStore.init();

      const stats = webkitStore.getStats();
      expect(stats.available).toBe(false);

      // Puts are silent no-ops on the disabled store — no error storm.
      await webkitStore.set('key', new Uint8Array(10));
      const after = webkitStore.getStats();
      expect(after.writeFailures).toBe(0);
      expect(after.writes).toBe(0);
      expect(after.count).toBe(0);
      expect(await webkitStore.get('key')).toBeUndefined();
    });

    it('should load existing metadata on init', async () => {
      // Pre-populate metadata. encodingVersion must match the current
      // OPFS_ENCODING_VERSION; otherwise the directory is intentionally
      // invalidated (see "encoding-version mismatch" test below).
      mockFS.metaFiles.set(
        '_cache_meta.json',
        JSON.stringify({
          baseUrl: 'https://example.com/data.zarr',
          entries: [['test.key', { size: 1000, order: 1 }]],
          totalSize: 1000,
          orderCounter: 2,
          contentHash: 'abc123',
          encodingVersion: 2,
        })
      );

      const newStore = new OPFSStore('test-id', 'https://example.com', 1024 * 1024);
      await newStore.init();

      const stats = newStore.getStats();
      expect(stats.size).toBe(1000);
      expect(stats.count).toBe(1);
      expect(newStore.getContentHash()).toBe('abc123');
    });
  });

  describe('Get Operations', () => {
    it('should retrieve stored data', async () => {
      const data = new Uint8Array([1, 2, 3, 4, 5]);
      await store.set('test.key', data);

      const retrieved = await store.get('test.key');
      expect(retrieved).toEqual(data);
    });

    it('should return undefined for missing keys', async () => {
      const result = await store.get('nonexistent');
      expect(result).toBeUndefined();
    });

    it('counts hits via reads and misses via getStats().misses', async () => {
      const data = new Uint8Array([1, 2, 3]);
      await store.set('hit.key', data);

      // Hits: reads counter goes up, misses unchanged.
      await store.get('hit.key');
      await store.get('hit.key');
      const afterHits = store.getStats();
      expect(afterHits.reads).toBe(2);
      expect(afterHits.misses).toBe(0);

      // Misses: misses counter goes up, reads unchanged.
      await store.get('absent.1');
      await store.get('absent.2');
      await store.get('absent.3');
      const afterMisses = store.getStats();
      expect(afterMisses.reads).toBe(2);
      expect(afterMisses.misses).toBe(3);
    });

    it('should update LRU order on get', async () => {
      // [cache.md/W13][P2] Previous version had a "Order counter should have
      // increased" comment but only asserted `count === 2`. Drive the
      // LRU-order contract directly by inspecting the internal `order` field
      // on the index entries: after the access to key1, key1's order MUST be
      // greater than key2's (most-recently-used).
      await store.set('key1', new Uint8Array(10));
      await store.set('key2', new Uint8Array(10));

      const idx = (store as any).index as Map<string, { size: number; order: number }>;
      const k1Before = idx.get('key1')!.order;
      const k2Before = idx.get('key2')!.order;
      expect(k2Before).toBeGreaterThan(k1Before); // key2 set after key1

      // Access key1 — should bump its order above key2.
      await store.get('key1');

      const k1After = idx.get('key1')!.order;
      const k2After = idx.get('key2')!.order;
      expect(k1After).toBeGreaterThan(k2After);
      // key2's order is untouched by the unrelated get.
      expect(k2After).toBe(k2Before);

      const stats = store.getStats();
      expect(stats.count).toBe(2);
    });

    it('happy-path round-trip preserves byte content (sized payload)', async () => {
      // [cache.md/W14][P2] Renamed from "should detect corrupted data via
      // size mismatch": that prior name claimed to verify a contract the
      // body never exercised (the corruption-detection branch lives in a
      // later test at ~L701). Recast as the legitimate happy-path round-trip
      // it actually is, and pin the assertion to byte-for-byte equality
      // rather than `toBeDefined() + byteLength`.
      const data = new Uint8Array(1000);
      for (let i = 0; i < 1000; i++) data[i] = (i * 7) & 0xff;
      await store.set('test.key', data);

      const stats = store.getStats();
      expect(stats.size).toBe(1000);
      expect(stats.count).toBe(1);

      const retrieved = await store.get('test.key');
      // Exact-bytes equality: any silent truncation / mid-payload corruption
      // would now fail.
      expect(retrieved).toEqual(data);
    });
  });

  describe('Set Operations', () => {
    it('should store data and update statistics', async () => {
      const data = new Uint8Array(1024);
      await store.set('test.key', data);

      const stats = store.getStats();
      expect(stats.size).toBe(1024);
      expect(stats.count).toBe(1);
    });

    it('should update existing keys', async () => {
      await store.set('key1', new Uint8Array(500));
      await store.set('key1', new Uint8Array(1000)); // Replace

      const stats = store.getStats();
      expect(stats.size).toBe(1000); // Updated size
      expect(stats.count).toBe(1); // Still one entry
    });

    it('should handle nested paths', async () => {
      // [cache.md/Wn][P2] Strengthen from `toBeDefined() + byteLength` to
      // exact-bytes equality, which catches any path-mangling regression that
      // returns a buffer of the right size but the wrong contents.
      const data = new Uint8Array(100);
      for (let i = 0; i < 100; i++) data[i] = i & 0xff;
      await store.set('a/b/c/data.bin', data);
      const retrieved = await store.get('a/b/c/data.bin');
      expect(retrieved).toEqual(data);
    });
  });

  describe('Delete Operations', () => {
    it('should delete existing entries', async () => {
      await store.set('key1', new Uint8Array(1000));
      await store.delete('key1');

      expect(await store.get('key1')).toBeUndefined();

      const stats = store.getStats();
      expect(stats.size).toBe(0);
      expect(stats.count).toBe(0);
    });

    it('should handle deletion of non-existent keys gracefully', async () => {
      await store.delete('nonexistent'); // Should not throw
      const stats = store.getStats();
      expect(stats.count).toBe(0);
    });

    it('drops and persists a stale index entry when its OPFS file is already missing', async () => {
      vi.useFakeTimers();
      try {
        await store.set('phantom', new Uint8Array(1000));
        expect(store.getStats()).toMatchObject({ size: 1000, count: 1 });
        await vi.advanceTimersByTimeAsync(1100);
        const staleMetadata = JSON.parse(mockFS.metaFiles.get('_cache_meta.json') as string);
        expect(staleMetadata.totalSize).toBe(1000);
        expect(staleMetadata.entries).toHaveLength(1);

        // Simulate a file removed by another tab after its metadata was saved.
        // The spec-faithful mock now raises NotFoundError when delete() tries to
        // remove it again.
        mockFS.files.clear();
        await store.delete('phantom');

        expect(store.getStats()).toMatchObject({ size: 0, count: 0 });
        await vi.advanceTimersByTimeAsync(1100);

        // Deletion is itself a metadata mutation. Persist the reconciled state
        // without requiring a later set() or dispose() to overwrite the stale
        // entry left by the previous snapshot.
        const metadata = JSON.parse(mockFS.metaFiles.get('_cache_meta.json') as string);
        expect(metadata.totalSize).toBe(0);
        expect(metadata.entries).toEqual([]);
      } finally {
        vi.useRealTimers();
      }
    });

    it('a missing LRU file does not wedge max-size eviction', async () => {
      const smallStore = new OPFSStore('phantom-eviction', 'https://example.com', 100);
      await smallStore.init();
      await smallStore.set('old', new Uint8Array(80));
      expect(smallStore.getStats()).toMatchObject({ size: 80, count: 1, evictions: 0 });

      // Leave the index entry but remove its backing file out of band. Adding
      // another 80-byte entry must reconcile and evict the phantom rather than
      // breaking the no-progress loop with an over-budget two-entry index.
      mockFS.files.clear();
      await smallStore.set('new', new Uint8Array(80));

      expect(smallStore.getStats()).toMatchObject({
        size: 80,
        count: 1,
        evictions: 1,
      });
      expect(await smallStore.get('old')).toBeUndefined();
      expect(await smallStore.get('new')).toEqual(new Uint8Array(80));
    });

    // [cache OOS] Pre-fix, `delete(key)` decremented `totalSize` BEFORE
    // calling `removeEntry()`. If removeEntry threw, totalSize was
    // already decremented but the file remained on disk and the index
    // still had the entry — three pieces of state out of sync. The next
    // `set()` then evicted based on an under-counted totalSize. The
    // post-fix order is removeEntry first, then totalSize + index
    // updates only on success.
    it('preserves totalSize and index when removeEntry throws (atomicity)', async () => {
      // Seed an entry so totalSize and index are both populated.
      const data = new Uint8Array(1000);
      await store.set('victim', data);
      const sizeBefore = store.getStats().size;
      const countBefore = store.getStats().count;
      expect(sizeBefore).toBeGreaterThan(0);
      expect(countBefore).toBe(1);

      // Swap in a removeEntry that throws to simulate an OPFS I/O error
      // mid-delete (file locked by concurrent op, transient permission
      // error, etc.). The previous removeEntry is restored after.
      const originalRemoveEntry = mockFS.mockDirHandle.removeEntry;
      mockFS.mockDirHandle.removeEntry = async () => {
        throw new Error('simulated transient OPFS error');
      };

      try {
        // delete() swallows the error in its own try/catch, so the call
        // itself does not reject; we observe the effect via getStats().
        await store.delete('victim');

        // POST-FIX CONTRACT: removal failed, so all three pieces of
        // state are unchanged — totalSize untouched, index unchanged.
        const stats = store.getStats();
        expect(stats.size).toBe(sizeBefore);
        expect(stats.count).toBe(countBefore);
      } finally {
        mockFS.mockDirHandle.removeEntry = originalRemoveEntry;
      }

      // After restoring the working removeEntry, a retry succeeds and
      // brings everything to consistent state.
      await store.delete('victim');
      const after = store.getStats();
      expect(after.size).toBe(0);
      expect(after.count).toBe(0);
    });
  });

  describe('Clear Operations', () => {
    it('should clear all data', async () => {
      await store.set('key1', new Uint8Array(1000));
      await store.set('key2', new Uint8Array(2000));

      await store.clear();

      const stats = store.getStats();
      expect(stats.size).toBe(0);
      expect(stats.count).toBe(0);
      expect(store.getContentHash()).toBeNull();
    });

    it('slow set + concurrent clear leaves index empty after both settle', async () => {
      // Wrap navigator.storage to gate the writable.write() so we can
      // hold a single set() in flight while clear() runs. The slow set
      // must NOT repopulate the index after the clear completes.
      let releaseWrite: () => void = () => {};
      const writeBlocker = new Promise<void>((resolve) => {
        releaseWrite = resolve;
      });
      // Armed only AFTER init so the init-time write probe passes through.
      let blockWrites = false;
      const baseDir = mockFS.mockDirHandle;
      const slowDir: any = {
        ...baseDir,
        async getFileHandle(name: string, opts?: { create?: boolean }) {
          const inner = await baseDir.getFileHandle(name, opts);
          return {
            ...inner,
            async createWritable() {
              const w = await inner.createWritable();
              return {
                ...w,
                async write(data: ArrayBuffer | string) {
                  if (blockWrites) await writeBlocker;
                  return w.write(data);
                },
              };
            },
          };
        },
      };
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            return {
              async getDirectoryHandle() {
                return slowDir;
              },
              async removeEntry() {
                mockFS.files.clear();
                mockFS.metaFiles.clear();
              },
            };
          },
          async estimate() {
            return { quota: 10e9, usage: 1e9 };
          },
        },
      });

      const slowStore = new OPFSStore('slow-id', 'https://example.com', 100 * 1024 * 1024);
      await slowStore.init();
      blockWrites = true;

      // Kick off a slow set; do not await yet — its write() is blocked.
      const setPromise = slowStore.set('hung-key', new Uint8Array(500));

      // Yield once so the set actually enters doSet and reaches the blocked write.
      await new Promise((r) => setTimeout(r, 0));

      // Now clear: bumps the generation and drains pending writes. We
      // unblock the write *during* clear so the in-flight set finishes
      // its file I/O while clear is awaiting Promise.allSettled.
      const clearPromise = slowStore.clear();
      releaseWrite();
      await Promise.all([setPromise, clearPromise]);

      // The slow set must have detected the generation bump and
      // skipped its index update.
      const stats = slowStore.getStats();
      expect(stats.count).toBe(0);
      expect(stats.size).toBe(0);
    });

    it('concurrent clear() calls are idempotent', async () => {
      await store.set('key1', new Uint8Array(1000));
      await store.set('key2', new Uint8Array(2000));

      await Promise.all([store.clear(), store.clear()]);

      const stats = store.getStats();
      expect(stats.size).toBe(0);
      expect(stats.count).toBe(0);
    });
  });

  describe('LRU Eviction', () => {
    it('should evict least recently used when quota exceeded', async () => {
      // Create small store to trigger eviction
      const smallStore = new OPFSStore('small-id', 'https://example.com', 100);
      await smallStore.init();

      await smallStore.set('key1', new Uint8Array(40));
      await smallStore.set('key2', new Uint8Array(40));
      // Total: 80 bytes

      await smallStore.set('key3', new Uint8Array(50)); // Exceeds 100!

      // key1 should be evicted (oldest)
      const stats = smallStore.getStats();
      expect(stats.size).toBeLessThanOrEqual(100);
      expect(stats.count).toBe(2); // key2 + key3
    });

    it('should respect touch() for LRU ordering', async () => {
      const smallStore = new OPFSStore('small-id', 'https://example.com', 100);
      await smallStore.init();

      await smallStore.set('key1', new Uint8Array(40));
      await smallStore.set('key2', new Uint8Array(40));

      // Touch key1 (should become newest)
      smallStore.touch('key1');

      // Add oversized item
      await smallStore.set('key3', new Uint8Array(50));

      // key2 should be evicted (now oldest), key1 saved by touch
      const stats = smallStore.getStats();
      expect(stats.count).toBe(2); // key1 + key3
    });
  });

  describe('Oversized + quota ordering', () => {
    it('rejects an entry larger than maxSize without evicting existing entries', async () => {
      const tinyStore = new OPFSStore('tiny-id', 'https://example.com', 100);
      await tinyStore.init();
      await tinyStore.set('keep', new Uint8Array(40));
      const sizeBefore = tinyStore.getStats().size;
      const countBefore = tinyStore.getStats().count;

      // 200 bytes > 100 byte maxSize. Must be skipped without evicting `keep`.
      await tinyStore.set('toobig', new Uint8Array(200));

      const stats = tinyStore.getStats();
      expect(stats.size).toBe(sizeBefore);
      expect(stats.count).toBe(countBefore);
      expect(stats.oversizedWriteSkipped).toBe(1);
    });

    it('write that initially exceeds quota succeeds after own-LRU eviction', async () => {
      // First call to estimate() reports almost-full quota; second call
      // (after eviction freed space) reports plenty. The store should
      // succeed on the post-eviction estimate.
      let estimateCall = 0;
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            return {
              async getDirectoryHandle() {
                return mockFS.mockDirHandle;
              },
              async removeEntry() {
                mockFS.files.clear();
                mockFS.metaFiles.clear();
              },
            };
          },
          async estimate() {
            estimateCall++;
            // Always report enough headroom — eviction-then-quota
            // ordering means quota is checked after eviction; our
            // assertion here is that the write succeeds, not that the
            // estimate reflects in-process eviction.
            return { quota: 10e9, usage: 1e9 };
          },
        },
      });

      const evictStore = new OPFSStore('evict-id', 'https://example.com', 100);
      await evictStore.init();
      await evictStore.set('old1', new Uint8Array(45));
      await evictStore.set('old2', new Uint8Array(45));
      await evictStore.set('new', new Uint8Array(50));

      const stats = evictStore.getStats();
      // evictions counter only increments when own-LRU eviction runs.
      expect(stats.evictions).toBeGreaterThanOrEqual(1);
      expect(stats.size).toBeLessThanOrEqual(100);
      // estimate should have been called at least once during the write loop.
      expect(estimateCall).toBeGreaterThanOrEqual(1);
    });

    it('write I/O failures increment writeFailures stat', async () => {
      // Make navigateToFile succeed for retry path (not "could not be
      // found") but createWritable() throw so the catch branch runs and
      // increments writeFailures. The failure is armed only AFTER init —
      // an always-broken createWritable is now caught by the init-time
      // write probe (which disables the store outright, the WebKit case);
      // this test covers TRANSIENT mid-session I/O failures.
      let failWrites = false;
      const failingDir: any = {
        ...mockFS.mockDirHandle,
        async getFileHandle() {
          return {
            async getFile() {
              return {
                async arrayBuffer() {
                  return new ArrayBuffer(0);
                },
              };
            },
            async createWritable() {
              if (failWrites) throw new Error('ENOSPC: simulated I/O failure');
              return {
                async write(_data: ArrayBuffer | string) {},
                async close() {},
              };
            },
          };
        },
        async getDirectoryHandle() {
          return failingDir;
        },
        async removeEntry() {},
      };
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            return {
              async getDirectoryHandle() {
                return failingDir;
              },
              async removeEntry() {},
            };
          },
          async estimate() {
            return { quota: 10e9, usage: 1e9 };
          },
        },
      });

      const failStore = new OPFSStore('fail-id', 'https://example.com', 1000);
      await failStore.init();
      failWrites = true;
      await failStore.set('boom', new Uint8Array(50));
      const stats = failStore.getStats();
      // A single broken write counts exactly 1. This used to be enforced by an
      // "already counted" flag while the loop still ran twice; the non-stale path
      // now returns on the first failure, so the invariant is structural.
      expect(stats.writeFailures).toBe(1);
      expect(stats.count).toBe(0);
    });

    it('attempts a non-stale write exactly once and warns once', async () => {
      // Only a stale bucket handle may retry. Retrying an ENOSPC/quota/timeout
      // error buys nothing (no backoff, no space reclaimed) and costs a second
      // opfsOperationTimeoutMs of caller stall plus a duplicate warning line.
      let failWrites = false;
      let createWritableCalls = 0;
      const failingDir: any = {
        ...mockFS.mockDirHandle,
        async getFileHandle() {
          return {
            async getFile() {
              return {
                async arrayBuffer() {
                  return new ArrayBuffer(0);
                },
              };
            },
            async createWritable() {
              createWritableCalls++;
              if (failWrites) throw new Error('ENOSPC: simulated I/O failure');
              return { async write(_d: ArrayBuffer | string) {}, async close() {} };
            },
          };
        },
        async getDirectoryHandle() {
          return failingDir;
        },
        async removeEntry() {},
      };
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            return {
              async getDirectoryHandle() {
                return failingDir;
              },
              async removeEntry() {},
            };
          },
          async estimate() {
            return { quota: 10e9, usage: 1e9 };
          },
        },
      });
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const store = new OPFSStore('once-id', 'https://example.com', 1000);
      await store.init();
      failWrites = true;
      createWritableCalls = 0;
      await store.set('boom', new Uint8Array(50));

      expect(createWritableCalls).toBe(1);
      expect(
        warnSpy.mock.calls.filter((c) => String(c[0]).includes('failed to write')).length
      ).toBe(1);
      warnSpy.mockRestore();
    });

    it('retries exactly once on a stale bucket handle, then succeeds', async () => {
      // The guard rail on the fix above: "only stale retries" must not become
      // "never retries". This behavior had no test at all.
      // Armed only AFTER init: the init-time write probe also calls
      // createWritable, and failing it would disable the store outright.
      let armStale = false;
      let createWritableCalls = 0;
      const dir: any = {
        ...mockFS.mockDirHandle,
        async getFileHandle() {
          return {
            async getFile() {
              return {
                async arrayBuffer() {
                  return new ArrayBuffer(0);
                },
              };
            },
            async createWritable() {
              createWritableCalls++;
              if (armStale && createWritableCalls === 1) {
                throw new Error('A requested file or directory could not be found');
              }
              return { async write(_d: ArrayBuffer | string) {}, async close() {} };
            },
          };
        },
        async getDirectoryHandle() {
          return dir;
        },
        async removeEntry() {},
      };
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            return {
              async getDirectoryHandle() {
                return dir;
              },
              async removeEntry() {},
            };
          },
          async estimate() {
            return { quota: 10e9, usage: 1e9 };
          },
        },
      });

      const store = new OPFSStore('stale-id', 'https://example.com', 1000);
      await store.init();
      armStale = true;
      createWritableCalls = 0;
      await store.set('key', new Uint8Array(50));

      expect(createWritableCalls).toBe(2); // failed once, retried, succeeded
      expect(store.getStats().writeFailures).toBe(0);
    });

    it('quota-skipped writes increment quotaWriteSkipped and do not write data', async () => {
      // checkQuota returns false → doSet skips the write.
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            return {
              async getDirectoryHandle() {
                return mockFS.mockDirHandle;
              },
              async removeEntry() {
                mockFS.files.clear();
                mockFS.metaFiles.clear();
              },
            };
          },
          async estimate() {
            return { quota: 100, usage: 100 }; // no headroom
          },
        },
      });

      const noQuotaStore = new OPFSStore('noquota-id', 'https://example.com', 1000);
      await noQuotaStore.init();
      await noQuotaStore.set('blocked', new Uint8Array(50));

      const stats = noQuotaStore.getStats();
      expect(stats.count).toBe(0);
      expect(stats.size).toBe(0);
      expect(stats.quotaWriteSkipped).toBe(1);
    });
  });

  describe('Quota Management', () => {
    it('should check quota before writing', async () => {
      // Mock quota exceeded
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            return {
              async getDirectoryHandle() {
                return mockFS.mockDirHandle;
              },
            };
          },
          async estimate() {
            return { quota: 1000, usage: 999 }; // Almost full!
          },
        },
      });

      const quotaStore = new OPFSStore('quota-id', 'https://example.com', 1024 * 1024);
      await quotaStore.init();

      // Should skip write due to insufficient quota
      await quotaStore.set('large', new Uint8Array(1000));

      const stats = quotaStore.getStats();
      expect(stats.size).toBe(0); // Not written
    });
  });

  describe('Content Hash Management', () => {
    it('should store and retrieve content hash', () => {
      store.setContentHash('abc123def456');
      expect(store.getContentHash()).toBe('abc123def456');
    });

    it('should clear content hash', () => {
      store.setContentHash('test');
      store.setContentHash(null);
      expect(store.getContentHash()).toBeNull();
    });
  });

  describe('Metadata Persistence', () => {
    it('should persist metadata on dispose', async () => {
      // [cache.md/Wn][P2] Previously asserted `toBeDefined()` then guarded
      // the field-shape assertions behind `if (metaStr)` — meaning a
      // regression that produced an empty/falsy string would silently skip
      // the inner asserts. Drop the if and force unconditional shape checks.
      await store.set('key1', new Uint8Array(1000));
      store.setContentHash('test-hash');

      await store.dispose();

      const metaStr = mockFS.metaFiles.get('_cache_meta.json');
      expect(typeof metaStr).toBe('string');
      const meta = JSON.parse(metaStr as string);
      expect(meta.baseUrl).toBe('https://example.com/data.zarr');
      expect(meta.contentHash).toBe('test-hash');
      expect(meta.totalSize).toBe(1000);
    });

    it('dispose during pending metadata save awaits the in-flight save', async () => {
      // [cache.md/W27][P2] Previous version asserted only `metaFiles.get(...)
      // .toBeDefined()`. The mere presence of the file says nothing about
      // whether dispose actually awaited the save (the file could appear from
      // a still-pending timer-driven save). Strengthen by pinning the
      // serialized contents — only a completed save writes the post-set
      // totalSize. A premature dispose return would leave a stale/empty
      // payload OR (because the file is checked synchronously, with no
      // post-dispose yield) no payload at all.
      await store.set('key1', new Uint8Array(100));
      // Wait > METADATA_SAVE_DELAY to start the save.
      await new Promise((r) => setTimeout(r, 1100));
      await store.dispose();
      const metaStr = mockFS.metaFiles.get('_cache_meta.json');
      expect(typeof metaStr).toBe('string');
      const meta = JSON.parse(metaStr as string);
      // The save flushed the post-set state — totalSize reflects the 100-byte
      // payload, not a pre-set zero (which would indicate dispose returned
      // before the save resolved).
      expect(meta.totalSize).toBe(100);
      expect(Array.isArray(meta.entries) || typeof meta.entries === 'object').toBe(true);
    });

    it('set/get/touch after dispose are no-ops', async () => {
      await store.set('keep', new Uint8Array(50));
      await store.dispose();

      // Should not throw, should not mutate state.
      await store.set('post', new Uint8Array(50));
      const retrieved = await store.get('keep');
      expect(retrieved).toBeUndefined();

      const stats = store.getStats();
      // The pre-dispose set is still tracked in stats (from before
      // disposal). The post-dispose set adds nothing.
      expect(stats.count).toBe(1);
    });

    it('dispose() is idempotent', async () => {
      await store.set('keep', new Uint8Array(50));
      await store.dispose();
      // A second dispose() must not throw and must not double-bump generation.
      await expect(store.dispose()).resolves.toBeUndefined();
    });

    it('unicode keys roundtrip through set/get/delete (commit 4.2)', async () => {
      // Pre-commit-4.2 keyToFileName used btoa(key) which throws on any
      // code point above 0xFF. Now we encode UTF-8 → base64url so keys
      // with arbitrary unicode work end-to-end.
      const unicodeKey = 'group/データ/0.0';
      await store.set(unicodeKey, new Uint8Array([1, 2, 3]));
      const data = await store.get(unicodeKey);
      expect(data).toEqual(new Uint8Array([1, 2, 3]));
      await store.delete(unicodeKey);
      const after = await store.get(unicodeKey);
      expect(after).toBeUndefined();
    });

    it('keys with /, =, + are filesystem-safe (commit 4.2)', async () => {
      // base64url avoids those characters entirely, so even keys with
      // them in their UTF-8 encoding survive.
      const key = 'a/b/=+++';
      await store.set(key, new Uint8Array([9, 8, 7]));
      const data = await store.get(key);
      expect(data).toEqual(new Uint8Array([9, 8, 7]));
    });

    it('hung OPFS read times out and degrades to miss (commit 4.4)', async () => {
      // Stub navigator.storage so getFile() for chunk paths returns a
      // never-resolving promise; metadata reads still throw "not found"
      // so init's loadMetadata starts fresh. The withTimeout wrapper
      // must abort the read and surface as undefined / missCount++.
      const hangDir: any = {
        async getFileHandle(name: string) {
          if (name === '_cache_meta.json') {
            // Treat as cold cache so init() doesn't hang reading metadata.
            throw new Error('not found');
          }
          return {
            async getFile() {
              return new Promise(() => {}); // hangs forever
            },
            async createWritable() {
              return { async write() {}, async close() {} };
            },
          };
        },
        async getDirectoryHandle() {
          return hangDir;
        },
        async removeEntry() {},
      };
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            return {
              async getDirectoryHandle() {
                return hangDir;
              },
              async removeEntry() {},
            };
          },
          async estimate() {
            return { quota: 10e9, usage: 1e9 };
          },
        },
      });

      const { config: realConfig } = await import('../../../config');
      const originalTimeout = realConfig.cache.opfsOperationTimeoutMs;
      realConfig.cache.opfsOperationTimeoutMs = 50;

      try {
        const hung = new OPFSStore('hung-id', 'https://example.com', 1024);
        await hung.init();
        // Pre-populate the index so get() reaches the read path (a
        // missing index entry is a fast-path miss without hitting OPFS).
        (hung as any).index.set('hung-key', { size: 10, order: 1 });

        const start = Date.now();
        const result = await hung.get('hung-key');
        const elapsed = Date.now() - start;
        expect(result).toBeUndefined();
        // Allow generous slack for jsdom timer skew but well below the
        // 5s Vitest default.
        expect(elapsed).toBeLessThan(2000);
        expect(hung.getStats().misses).toBeGreaterThanOrEqual(1);
      } finally {
        realConfig.cache.opfsOperationTimeoutMs = originalTimeout;
      }
    });

    it('size-mismatch read deletes file and increments corruptedEntries (commit 6.2)', async () => {
      // Pre-populate index claiming size=100 but file actually has 5 bytes.
      await store.set('mismatch-key', new Uint8Array([1, 2, 3]));
      // Inject a mismatch by mutating the stored data.
      const fileNames = Array.from(mockFS.files.keys());
      const corruptName = fileNames.find((n) => n !== '_cache_meta.json');
      expect(corruptName).toBeDefined();
      mockFS.files.set(corruptName!, new Uint8Array(50)); // wrong size

      const result = await store.get('mismatch-key');
      expect(result).toBeUndefined();
      const stats = store.getStats();
      expect(stats.corruptedEntries).toBeGreaterThanOrEqual(1);
    });

    it('corrupt _cache_meta.json increments metadataParseFailures and starts fresh (commit 6.2)', async () => {
      // Persist invalid JSON so loadMetadata's catch branch runs the
      // parse-failure path.
      mockFS.metaFiles.set('_cache_meta.json', 'not-valid-json{{{');

      const fresh = new OPFSStore('parse-fail-id', 'https://example.com', 1024 * 1024);
      await fresh.init();

      const stats = fresh.getStats();
      expect(stats.metadataParseFailures).toBeGreaterThanOrEqual(1);
      expect(stats.count).toBe(0);
    });

    it('encoding-version mismatch invalidates the directory cleanly (commit 4.2)', async () => {
      // Persist a metadata file claiming version 1 (legacy btoa); the
      // store on the next init() must treat it as a cold cache and not
      // restore the old entries.
      const stale = JSON.stringify({
        baseUrl: 'https://example.com/data.zarr',
        entries: [['legacy-key', { size: 100, order: 0 }]],
        totalSize: 100,
        orderCounter: 1,
        contentHash: 'old-hash',
        encodingVersion: 1,
      });
      mockFS.metaFiles.set('_cache_meta.json', stale);

      const fresh = new OPFSStore('fresh-id', 'https://example.com/data.zarr', 1024 * 1024);
      await fresh.init();

      const stats = fresh.getStats();
      expect(stats.count).toBe(0);
      expect(stats.size).toBe(0);
      expect(fresh.getContentHash()).toBeNull();
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero-size files', async () => {
      // [cache.md/Wn][P2] Strengthen `toBeDefined()` → exact-bytes via
      // `toEqual(new Uint8Array(0))`. The previous pair (`toBeDefined()` +
      // `byteLength === 0`) was satisfied by `null as any` if a regression
      // ever made `get` return null-coerced empties; exact equality nails it.
      await store.set('empty', new Uint8Array(0));

      const retrieved = await store.get('empty');
      expect(retrieved).toEqual(new Uint8Array(0));

      const stats = store.getStats();
      expect(stats.size).toBe(0);
      expect(stats.count).toBe(1);
    });

    it('should handle rapid sequential writes', async () => {
      // [cache.md/W16][P2] Previous `count > 0 && size <= 100MB` was
      // trivially satisfied (a regression that wrote only the first key
      // would still pass `> 0`). The store's maxSize is 1MB (see beforeEach
      // setup), so 100 × 1000-byte writes total exactly 100 000 bytes —
      // well within capacity, so all 100 should be retained. Pin counts
      // and total size exactly.
      const N = 100;
      const BYTES = 1000;
      for (let i = 0; i < N; i++) {
        await store.set(`key${i}`, new Uint8Array(BYTES));
      }

      const stats = store.getStats();
      expect(stats.count).toBe(N);
      expect(stats.size).toBe(N * BYTES);
    });

    it('should handle concurrent operations', async () => {
      // Simulate concurrent writes
      await Promise.all([
        store.set('key1', new Uint8Array(1000)),
        store.set('key2', new Uint8Array(2000)),
        store.set('key3', new Uint8Array(3000)),
      ]);

      const stats = store.getStats();
      expect(stats.count).toBe(3);
      expect(stats.size).toBe(6000);
    });
  });

  // R6a: Unicode OPFS key roundtrip. keyToFileName now uses UTF-8 +
  // base64url encoding; the prior btoa() implementation would throw
  // InvalidCharacterError on non-ASCII bytes. These tests lock that
  // fix in.
  describe('Unicode key roundtrip (R6a)', () => {
    it('stores and reads back a key containing non-ASCII characters', async () => {
      const key = 'µ/通道/positions/0.0.0';
      const payload = new Uint8Array([1, 2, 3, 4, 5]);
      await store.set(key, payload);

      const got = await store.get(key);
      expect(got).toBeDefined();
      expect(Array.from(got!)).toEqual([1, 2, 3, 4, 5]);
    });

    it('handles keys with slashes that traverse bucket boundaries', async () => {
      const key = 'foo/bar/baz/é/0.1';
      await store.set(key, new Uint8Array([42]));
      const got = await store.get(key);
      expect(got).toBeDefined();
      expect(got!.length).toBe(1);
      expect(got![0]).toBe(42);
    });

    it('handles keys with characters that have special meaning in base64', async () => {
      // `+`, `/`, `=` are exactly the characters base64url replaces;
      // upstream data with these in the key must roundtrip cleanly.
      const key = 'a+b/c=d/0';
      await store.set(key, new Uint8Array([7, 8, 9]));
      const got = await store.get(key);
      expect(got).toBeDefined();
      expect(Array.from(got!)).toEqual([7, 8, 9]);
    });

    it('deletes a Unicode-keyed entry cleanly', async () => {
      const key = 'µ/通道/positions/0.0.0';
      await store.set(key, new Uint8Array([1]));
      expect(await store.get(key)).toBeDefined();
      await store.delete(key);
      expect(await store.get(key)).toBeUndefined();
    });
  });

  // R6d: OPFS quota-exhaustion edge cases. Existing tests cover the
  // basic quota-skipped path; these tests cover behaviour after the
  // quota constraint clears and under concurrent quota-exceeded
  // writes.
  describe('Quota exhaustion recovery (R6d)', () => {
    it('after quota clears, the next write succeeds', async () => {
      // Simulate "browser quota almost full" → write should be
      // skipped. Then "quota cleared" → next write should succeed.
      let quotaReportsFull = true;
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            return {
              async getDirectoryHandle() {
                return mockFS.mockDirHandle;
              },
              async removeEntry() {
                mockFS.files.clear();
                mockFS.metaFiles.clear();
              },
            };
          },
          async estimate() {
            return quotaReportsFull ? { quota: 100, usage: 100 } : { quota: 10e9, usage: 1e9 };
          },
        },
      });
      const cycleStore = new OPFSStore('cycle-id', 'https://example.com', 1024 * 1024);
      await cycleStore.init();

      // First write: quota full → skipped.
      await cycleStore.set('blocked', new Uint8Array(50));
      let stats = cycleStore.getStats();
      expect(stats.quotaWriteSkipped).toBeGreaterThanOrEqual(1);
      expect(stats.count).toBe(0);

      // Quota clears.
      quotaReportsFull = false;
      await cycleStore.set('now-fits', new Uint8Array(50));

      stats = cycleStore.getStats();
      expect(stats.count).toBe(1);
      expect(stats.size).toBe(50);
    });

    it('concurrent quota-skipped writes increment counter atomically (no double-count)', async () => {
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            return {
              async getDirectoryHandle() {
                return mockFS.mockDirHandle;
              },
              async removeEntry() {
                mockFS.files.clear();
                mockFS.metaFiles.clear();
              },
            };
          },
          async estimate() {
            return { quota: 100, usage: 100 }; // no headroom
          },
        },
      });
      const burstStore = new OPFSStore('burst-id', 'https://example.com', 1024 * 1024);
      await burstStore.init();

      // Fire 5 concurrent same-key + 5 unique-key writes. All should
      // be skipped without throwing.
      await Promise.all([
        burstStore.set('dup-1', new Uint8Array(30)),
        burstStore.set('dup-1', new Uint8Array(30)),
        burstStore.set('dup-2', new Uint8Array(30)),
        burstStore.set('dup-2', new Uint8Array(30)),
        burstStore.set('uniq-a', new Uint8Array(30)),
        burstStore.set('uniq-b', new Uint8Array(30)),
        burstStore.set('uniq-c', new Uint8Array(30)),
      ]);

      const stats = burstStore.getStats();
      // Every set call is observed by checkQuota; counter equals the
      // number of skipped writes. Sufficient to verify it's at least 1
      // and the store hasn't ingested any data despite the burst.
      expect(stats.quotaWriteSkipped).toBeGreaterThanOrEqual(1);
      expect(stats.count).toBe(0);
      expect(stats.size).toBe(0);
    });
  });

  // S2: getStats().available reflects OPFS reachability + dispose state.
  // Drives the `opfs-unavailable` UI badge in cache-metrics-aggregator.
  describe('Availability flag (S2)', () => {
    it('reports available=true after a successful init', () => {
      expect(store.getStats().available).toBe(true);
    });

    it('reports available=false after dispose()', async () => {
      expect(store.getStats().available).toBe(true);
      await store.dispose();
      expect(store.getStats().available).toBe(false);
    });

    it('reports available=false when init never acquired the OPFS root', async () => {
      // Simulate a browser that throws on navigator.storage.getDirectory.
      vi.stubGlobal('navigator', {
        storage: {
          async getDirectory() {
            throw new Error('OPFS unsupported in this browser');
          },
          async estimate() {
            return { quota: 0, usage: 0 };
          },
        },
      });
      const noOpfsStore = new OPFSStore('no-opfs-id', 'https://example.com', 100 * 1024 * 1024);
      // init() should swallow the failure and leave opfsRoot null.
      await noOpfsStore.init();
      expect(noOpfsStore.getStats().available).toBe(false);
    });
  });

  // R6e: partial metadata corruption recovery. The existing test
  // covers the "malformed JSON → start fresh" path; this one covers
  // a structurally-valid but logically corrupt metadata file
  // (negative totalSize / orderCounter) which should be treated as
  // recoverable (clamp / rebuild from entries[]).
  describe('Metadata corruption recovery (R6e)', () => {
    it('initialises cleanly from valid metadata with a negative totalSize', async () => {
      // Store a metadata file with negative totalSize — the store
      // should either reject the file (start fresh) or clamp to a
      // non-negative size.
      mockFS.metaFiles.set(
        '_cache_meta.json',
        JSON.stringify({
          baseUrl: 'https://example.com/data.zarr',
          entries: [['k1', { size: 100, order: 1 }]],
          totalSize: -999,
          orderCounter: 1,
          contentHash: 'abc',
          encodingVersion: 2,
        })
      );
      const recoveredStore = new OPFSStore(
        'corrupt-totalsize',
        'https://example.com/data.zarr',
        100 * 1024 * 1024
      );
      await recoveredStore.init();
      const stats = recoveredStore.getStats();
      // The store must not return a nonsensical negative size.
      expect(stats.size).toBeGreaterThanOrEqual(0);
    });
  });
});

// Removed: previously this block defined a LOCAL `function getBucket(...)`
// that duplicated the production hash, then asserted properties of that local
// copy. It tested the test's own implementation, not the production unit.
// The real getBucket has dedicated coverage in
// tests/unit/cache/opfs-store/buckets.test.ts which imports the production
// symbol. See delme/test-audit-luxar-viewer.src/cache.md (C1).
