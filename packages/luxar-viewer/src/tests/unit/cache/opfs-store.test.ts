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
      // Pre-populate metadata
      mockFS.metaFiles.set(
        '_cache_meta.json',
        JSON.stringify({
          baseUrl: 'https://example.com/data.zarr',
          entries: [['test.key', { size: 1000, order: 1 }]],
          totalSize: 1000,
          orderCounter: 2,
          contentHash: 'abc123',
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
      const data = new Uint8Array(1000);
      await store.set('test.key', data);

      // Corrupt the file (change size)
      mockFS.files.set('test.key', new Uint8Array(500)); // Wrong size!

      const result = await store.get('test.key');
      expect(result).toBeUndefined(); // Should detect corruption

      // Should remove from index
      const stats = store.getStats();
      expect(stats.count).toBe(0); // Removed corrupted entry
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
