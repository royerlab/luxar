/**
 * TODO #4 regression — "cache eviction does not appear to trigger when expected."
 *
 * Root cause (verified, now fixed): eviction was gated only on the
 * configured `maxSize` (default l2MaxSizeMB: 2048 → 2 GB), but a separate
 * quota gate (`navigator.storage.estimate()`) rejected writes long before
 * `totalSize` ever approached 2 GB. When the browser grants less OPFS than
 * `maxSize` (common on Firefox, private mode, and smaller disks), the quota
 * gate fired first and writes were silently skipped — the maxSize-based
 * eviction loop never ran, so the LRU never recycled space. The cache
 * "froze" holding the OLDEST admitted chunks and rejecting newer ones.
 *
 * Fix: `doSet` now evicts LRU entries on quota pressure (not just on
 * maxSize pressure) and re-checks, since deleting files genuinely frees
 * the browser quota. Both tests share identical quota dynamics (quota =
 * 1000 bytes, usage tracks bytes actually on disk):
 *   - QUOTA-DRIVEN case: maxSize = 2 GB → eviction now fires under quota
 *     pressure; newest retained, oldest evicted (was: frozen, evictions 0).
 *   - MAXSIZE-DRIVEN case: maxSize = 900 → the original maxSize eviction
 *     path still works; newest retained, oldest evicted.
 */

import { describe, it, expect, afterEach, vi } from 'vitest';
import { OPFSStore } from '../../../cache/multi-level-caching-store/opfs-store';

const META = '_cache_meta.json';

/**
 * Minimal File System Access mock whose `estimate()` reports `usage`
 * computed from the bytes actually written to disk, so eviction (which
 * deletes files) genuinely frees quota — mirroring real OPFS.
 */
function mockOPFS(quota: number, opts?: { failDelete?: boolean }) {
  const files = new Map<string, Uint8Array>();
  const metaFiles = new Map<string, string>();

  const fileHandle = (path: string) => ({
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
          if (typeof data === 'string') metaFiles.set(path, data);
          else files.set(path, new Uint8Array(data));
        },
        async close() {},
      };
    },
  });

  const dirHandle: any = {
    async getFileHandle(name: string, opts?: { create?: boolean }) {
      if (!files.has(name) && !metaFiles.has(name) && !opts?.create) {
        throw new Error('File not found');
      }
      return fileHandle(name);
    },
    async getDirectoryHandle() {
      return dirHandle; // all buckets collapse to one handle for the test
    },
    async removeEntry(name: string) {
      // Simulate a persistently failing per-file delete (transient OPFS
      // I/O error) so the eviction loops' no-progress guards are exercised.
      if (opts?.failDelete) throw new Error('removeEntry failed');
      files.delete(name);
      metaFiles.delete(name);
    },
    async *keys() {},
  };

  const usageBytes = (): number => {
    let total = 0;
    for (const [name, bytes] of files) {
      if (name === META) continue;
      total += bytes.byteLength;
    }
    return total;
  };

  vi.stubGlobal('navigator', {
    storage: {
      async getDirectory() {
        return {
          async getDirectoryHandle() {
            return dirHandle;
          },
          async removeEntry() {
            files.clear();
            metaFiles.clear();
          },
        };
      },
      async estimate() {
        return { quota, usage: usageBytes() };
      },
    },
  });

  return { files, metaFiles };
}

describe('TODO #4 — L2 eviction vs. browser quota', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('RECYCLES under quota pressure even when maxSize ≫ quota: eviction fires, newest retained', async () => {
    mockOPFS(1000);

    // maxSize dwarfs the granted quota — mirrors the default
    // l2MaxSizeMB: 2048 on a browser that grants ~1 KB here. Before the
    // fix this froze (evictions 0, newest rejected); now quota-pressure
    // eviction keeps admitting new chunks.
    const store = new OPFSStore('quota-driven', 'https://example.com', 2 * 1024 * 1024 * 1024);
    await store.init();

    for (let i = 0; i < 20; i++) {
      await store.set(`chunk-${i}`, new Uint8Array(100));
    }

    const stats = store.getStats();

    // Eviction now runs in response to quota pressure (not just maxSize).
    expect(stats.evictions).toBeGreaterThan(0);
    expect(stats.count).toBeGreaterThan(0);

    // Newest retained, oldest evicted — the cache no longer freezes.
    expect(await store.get('chunk-19')).toBeDefined();
    expect(await store.get('chunk-0')).toBeUndefined();
  });

  it('RECYCLES with maxSize aligned to quota: evictions fire, newest retained, oldest evicted', async () => {
    mockOPFS(1000);

    // Same quota dynamics, but maxSize set to the real ceiling. Eviction
    // (which runs BEFORE the quota check) frees space, so newer chunks
    // keep getting admitted — healthy LRU recycling.
    const store = new OPFSStore('aligned', 'https://example.com', 900);
    await store.init();

    for (let i = 0; i < 20; i++) {
      await store.set(`chunk-${i}`, new Uint8Array(100));
    }

    const stats = store.getStats();

    expect(stats.evictions).toBeGreaterThan(0);
    expect(stats.count).toBeGreaterThan(0);

    // Newest retained, oldest evicted — the opposite of the freeze case.
    expect(await store.get('chunk-19')).toBeDefined();
    expect(await store.get('chunk-0')).toBeUndefined();
  });

  it('TERMINATES when delete() always fails during maxSize eviction (no infinite loop)', async () => {
    // Generous quota isolates the maxSize eviction path; failDelete makes
    // every per-file delete throw. Before the no-progress guard, the
    // maxSize loop re-selected the same undeletable head key forever and
    // over-counted evictions. The guard must break the loop instead.
    const { files } = mockOPFS(10e9, { failDelete: true });
    const store = new OPFSStore('faildelete', 'https://example.com', 100);
    await store.init();

    // Seed an entry occupying most of the 100-byte cache.
    await store.set('old', new Uint8Array(80));
    expect(files.size).toBeGreaterThan(0);

    // 80 + 80 > 100 → eviction required, but delete() can never free space.
    // If the loop spins, this test exceeds its timeout and fails loudly.
    await store.set('new', new Uint8Array(80));

    // The undeletable entry must NOT be counted as evicted (no over-count).
    expect(store.getStats().evictions).toBe(0);
  }, 5000);
});
