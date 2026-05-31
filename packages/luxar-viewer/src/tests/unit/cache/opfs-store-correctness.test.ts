/**
 * OPFSStore correctness regressions for two issues found while reviewing
 * the cache stack:
 *
 *  - #1 same-key write serialization: `set()` previously only awaited the
 *    single in-flight write at entry. When ≥2 callers awaited the same
 *    promise they resumed together and ran `doSet()` concurrently —
 *    overlapping `createWritable()`+write+close on the SAME OPFS file,
 *    which the FS Access API does not guarantee is safe. The fix chains
 *    same-key writes so their file I/O never overlaps.
 *
 *  - #3 orphan reads: `get()` read the file before consulting the index,
 *    so a file present on disk but absent from the index (e.g. a
 *    generation-skipped write whose best-effort delete failed) was served
 *    — potentially resurrecting bytes a content-hash invalidation meant to
 *    drop. The fix makes the index the source of truth: an unindexed key
 *    is a miss.
 */

import { describe, it, expect, afterEach, vi } from 'vitest';
import { OPFSStore } from '../../../cache/multi-level-caching-store/opfs-store';

/**
 * File System Access mock that tracks how many writers are concurrently
 * inside the createWritable→close window for each file path, so a test can
 * assert that same-key writes never overlap. `write()` yields a microtask
 * to widen the window and make any overlap observable.
 */
function mockOPFS() {
  const files = new Map<string, Uint8Array>();
  const metaFiles = new Map<string, string>();
  const activePerPath = new Map<string, number>();
  let maxConcurrentSameFile = 0;
  // Total writers in flight across ALL (non-metadata) files at once. Lets a
  // test prove that distinct-key writes genuinely overlap in time (i.e.
  // per-key serialization does NOT serialize unrelated keys).
  let activeTotal = 0;
  let maxConcurrentAnyFile = 0;

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
      const active = (activePerPath.get(path) ?? 0) + 1;
      activePerPath.set(path, active);
      if (path !== '_cache_meta.json') {
        maxConcurrentSameFile = Math.max(maxConcurrentSameFile, active);
        activeTotal += 1;
        maxConcurrentAnyFile = Math.max(maxConcurrentAnyFile, activeTotal);
      }
      return {
        async write(data: ArrayBuffer | string) {
          await Promise.resolve(); // yield so overlapping writers are observable
          if (typeof data === 'string') metaFiles.set(path, data);
          else files.set(path, new Uint8Array(data));
        },
        async close() {
          activePerPath.set(path, (activePerPath.get(path) ?? 1) - 1);
          if (path !== '_cache_meta.json') activeTotal -= 1;
        },
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
      return dirHandle;
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
            return dirHandle;
          },
          async removeEntry() {
            files.clear();
            metaFiles.clear();
          },
        };
      },
      async estimate() {
        return { quota: 10e9, usage: 0 };
      },
    },
  });

  return {
    files,
    metaFiles,
    getMaxConcurrentSameFile: () => maxConcurrentSameFile,
    getMaxConcurrentAnyFile: () => maxConcurrentAnyFile,
  };
}

describe('OPFSStore correctness (#1 write serialization, #3 orphan reads)', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('#1 serializes concurrent same-key writes — no overlapping OPFS file writes', async () => {
    const { getMaxConcurrentSameFile } = mockOPFS();
    const store = new OPFSStore('serialize', 'https://example.com', 10 * 1024 * 1024);
    await store.init();

    // Fire many concurrent writes to the SAME key. Before the fix, the
    // waiters resumed together and ran doSet() — and thus createWritable()
    // — concurrently on one file. The chaining fix must keep them serial.
    await Promise.all(
      Array.from({ length: 6 }, (_, i) => store.set('hot-key', new Uint8Array(100 + i)))
    );

    // Never more than one writer inside a single file's write window.
    expect(getMaxConcurrentSameFile()).toBe(1);

    // End state is a single, coherent entry. Chaining preserves arrival
    // order, so the last write (i=5 → 100+5 bytes) wins and `size` tracks
    // exactly that one entry — no double-counting drift.
    const stats = store.getStats();
    expect(stats.count).toBe(1);
    expect(stats.size).toBe(105);
    expect(await store.get('hot-key')).toBeDefined();
  });

  it('#1 distinct keys still write concurrently (serialization is per-key, not global)', async () => {
    const { getMaxConcurrentSameFile, getMaxConcurrentAnyFile } = mockOPFS();
    const store = new OPFSStore('parallel', 'https://example.com', 10 * 1024 * 1024);
    await store.init();

    await Promise.all(
      Array.from({ length: 6 }, (_, i) => store.set(`key-${i}`, new Uint8Array(100)))
    );

    // Per-key serialization must not serialize *different* keys. Prove the
    // distinct-key writes genuinely overlapped in time: more than one
    // writer was in flight across files simultaneously...
    expect(getMaxConcurrentAnyFile()).toBeGreaterThan(1);
    // ...while still never overlapping two writers on the SAME file...
    expect(getMaxConcurrentSameFile()).toBe(1);
    // ...and all six landed.
    expect(store.getStats().count).toBe(6);
  });

  it('#3 does not serve an orphaned file with no index entry', async () => {
    const { files } = mockOPFS();
    const store = new OPFSStore('orphan', 'https://example.com', 10 * 1024 * 1024);
    await store.init();

    // Write a real entry (file on disk + index entry), then simulate an
    // orphan by dropping ONLY the index entry — the file stays on disk,
    // exactly the state a generation-skipped write with a failed
    // best-effort delete would leave behind.
    await store.set('ghost', new Uint8Array(64));
    const filesBefore = files.size;
    expect(filesBefore).toBeGreaterThan(0);
    (store as unknown as { index: Map<string, unknown> }).index.delete('ghost');

    // The orphan must NOT be served — the index is the source of truth.
    expect(await store.get('ghost')).toBeUndefined();
    // And it counts as a miss.
    expect(store.getStats().misses).toBeGreaterThan(0);
  });
});
