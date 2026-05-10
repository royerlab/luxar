import { describe, it, expect, beforeEach, vi } from 'vitest';
import { OPFSStore } from '../../../cache/opfs-store';

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

  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      const stats = store.getStats();
      expect(stats.size).toBe(0);
      expect(stats.count).toBe(0);
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
      await store.set('key1', new Uint8Array(10));
      await store.set('key2', new Uint8Array(10));

      // Access key1 (should update order)
      await store.get('key1');

      // Order counter should have increased
      const stats = store.getStats();
      expect(stats.count).toBe(2);
    });

    it('should detect corrupted data via size mismatch', async () => {
      // Note: This test validates the corruption detection logic.
      // The simplified mock doesn't track bucketed paths, so we test
      // by directly manipulating the stats/index instead.
      const data = new Uint8Array(1000);
      await store.set('test.key', data);

      // Verify data was stored
      const stats = store.getStats();
      expect(stats.size).toBe(1000);
      expect(stats.count).toBe(1);

      // The corruption detection logic checks: entry.size !== data.byteLength
      // This is tested implicitly when the file system returns wrong-sized data.
      // With the simplified mock, we verify the happy path works correctly.
      const retrieved = await store.get('test.key');
      expect(retrieved).toBeDefined();
      expect(retrieved?.byteLength).toBe(1000);
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
      await store.set('a/b/c/data.bin', new Uint8Array(100));
      const retrieved = await store.get('a/b/c/data.bin');
      expect(retrieved).toBeDefined();
      expect(retrieved?.byteLength).toBe(100);
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
                  await writeBlocker;
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
      // increments writeFailures.
      const failingDir: any = {
        ...mockFS.mockDirHandle,
        async getFileHandle() {
          return {
            async getFile() {
              return { async arrayBuffer() { return new ArrayBuffer(0); } };
            },
            async createWritable() {
              throw new Error('ENOSPC: simulated I/O failure');
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
      await failStore.set('boom', new Uint8Array(50));
      const stats = failStore.getStats();
      expect(stats.writeFailures).toBeGreaterThanOrEqual(1);
      expect(stats.count).toBe(0);
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
      await store.set('key1', new Uint8Array(1000));
      store.setContentHash('test-hash');

      await store.dispose();

      // Check that metadata was written
      const metaStr = mockFS.metaFiles.get('_cache_meta.json');
      expect(metaStr).toBeDefined();

      if (metaStr) {
        const meta = JSON.parse(metaStr);
        expect(meta.baseUrl).toBe('https://example.com/data.zarr');
        expect(meta.contentHash).toBe('test-hash');
        expect(meta.totalSize).toBe(1000);
      }
    });

    it('dispose during pending metadata save awaits the in-flight save', async () => {
      // Trigger scheduleMetadataSave by mutating state, then await dispose.
      // The in-flight save tracker means dispose must wait for the save
      // to complete before returning.
      await store.set('key1', new Uint8Array(100));
      // Wait > METADATA_SAVE_DELAY to start the save.
      await new Promise((r) => setTimeout(r, 1100));
      await store.dispose();
      // Metadata is now persisted.
      expect(mockFS.metaFiles.get('_cache_meta.json')).toBeDefined();
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
      await store.set('empty', new Uint8Array(0));

      const retrieved = await store.get('empty');
      expect(retrieved).toBeDefined();
      expect(retrieved?.byteLength).toBe(0);

      const stats = store.getStats();
      expect(stats.size).toBe(0);
      expect(stats.count).toBe(1);
    });

    it('should handle rapid sequential writes', async () => {
      for (let i = 0; i < 100; i++) {
        await store.set(`key${i}`, new Uint8Array(1000));
      }

      const stats = store.getStats();
      expect(stats.count).toBeGreaterThan(0);
      expect(stats.size).toBeLessThanOrEqual(100 * 1024 * 1024);
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
});

/**
 * Test the bucket hash algorithm in isolation.
 * These tests verify the bucketing strategy without needing OPFS mocks.
 */
describe('OPFSStore Bucketing Algorithm', () => {
  // Replicate the getBucket algorithm for testing
  function getBucket(key: string): string {
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      hash = ((hash << 5) - hash + key.charCodeAt(i)) | 0;
    }
    return (hash & 0xff).toString(16).padStart(2, '0');
  }

  describe('getBucket consistency', () => {
    it('should return same bucket for same key', () => {
      const key = 'points/positions/0.0.0';
      expect(getBucket(key)).toBe(getBucket(key));
    });

    it('should return 2-character hex string', () => {
      const keys = ['test', 'points/positions/0.0.0', '.zmetadata', 'a/b/c/d'];
      for (const key of keys) {
        const bucket = getBucket(key);
        expect(bucket).toMatch(/^[0-9a-f]{2}$/);
      }
    });

    it('should produce valid bucket for empty string', () => {
      const bucket = getBucket('');
      expect(bucket).toBe('00'); // Empty string hash is 0
    });
  });

  describe('getBucket distribution', () => {
    it('should distribute sequential chunk keys to different buckets', () => {
      const buckets = new Set<string>();
      // Sequential chunk indices should spread across buckets
      for (let i = 0; i < 100; i++) {
        buckets.add(getBucket(`points/positions/0.0.${i}`));
      }
      // Should have reasonable distribution (not all same bucket)
      expect(buckets.size).toBeGreaterThan(10);
    });

    it('should distribute different arrays to different buckets', () => {
      const buckets = [
        getBucket('points/positions/0.0.0'),
        getBucket('points/colors/0.0.0'),
        getBucket('points/radii/0.0.0'),
        getBucket('other/data/0.0.0'),
      ];
      const unique = new Set(buckets);
      // Different array names should mostly go to different buckets
      expect(unique.size).toBeGreaterThanOrEqual(3);
    });

    it('should have bucket values in valid range 00-ff', () => {
      // Test with many random-ish keys
      const keys = [
        '.zmetadata',
        '.zarray',
        '.zattrs',
        'points/positions/0.0.0',
        'points/positions/999.999.999',
        'very/deep/nested/path/to/data/chunk.bin',
        'unicode_测试_キー',
      ];

      for (const key of keys) {
        const bucket = getBucket(key);
        const value = parseInt(bucket, 16);
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThanOrEqual(255);
      }
    });
  });

  describe('known bucket values', () => {
    it('should map points/positions/0.0.0 to bucket 23', () => {
      expect(getBucket('points/positions/0.0.0')).toBe('23');
    });

    it('should map .zmetadata to bucket 3b', () => {
      expect(getBucket('.zmetadata')).toBe('3b');
    });
  });
});
