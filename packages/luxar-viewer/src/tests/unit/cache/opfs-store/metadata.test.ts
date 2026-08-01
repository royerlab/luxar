import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  OPFSMetadataManager,
  type MetadataSnapshot,
} from '../../../../cache/multi-level-caching-store/opfs-store/metadata';
import { OPFS_ENCODING_VERSION } from '../../../../cache/types';

/**
 * Minimal mock OPFS root.
 *
 *   rootFiles: Map<name, contents>      // sibling files at root, e.g. _cache_meta.json
 *   buckets:   Map<bucket, Map<name, contents>>  // hex subdirectories with files
 *
 * The root handle's `keys()` yields both the root-file names and the
 * bucket names interleaved, matching OPFS's "entries at this level"
 * semantics. cleanupOrphans filters by hex pattern to find buckets.
 */
function mockRoot() {
  const rootFiles = new Map<string, string>();
  const buckets = new Map<string, Map<string, string>>();

  function makeFileHandle(map: Map<string, string>, name: string) {
    return {
      async getFile() {
        return {
          async text() {
            return map.get(name) ?? '';
          },
        };
      },
      async createWritable() {
        return {
          async write(data: string) {
            map.set(name, data);
          },
          async close() {},
        };
      },
    };
  }

  function makeBucketHandle(bucket: string): FileSystemDirectoryHandle {
    return {
      async getFileHandle(name: string, opts?: { create?: boolean }) {
        const files = buckets.get(bucket)!;
        if (!files.has(name) && !opts?.create) throw new Error('not found');
        if (opts?.create && !files.has(name)) files.set(name, '');
        return makeFileHandle(files, name) as unknown as FileSystemFileHandle;
      },
      async removeEntry(name: string) {
        buckets.get(bucket)?.delete(name);
      },
      async *keys() {
        for (const k of buckets.get(bucket)!.keys()) yield k;
      },
    } as unknown as FileSystemDirectoryHandle;
  }

  const root = {
    async getFileHandle(name: string, opts?: { create?: boolean }) {
      if (!rootFiles.has(name) && !opts?.create) throw new Error('not found');
      if (opts?.create && !rootFiles.has(name)) rootFiles.set(name, '');
      return makeFileHandle(rootFiles, name) as unknown as FileSystemFileHandle;
    },
    async getDirectoryHandle(name: string, opts?: { create?: boolean }) {
      if (!buckets.has(name) && !opts?.create) throw new Error('not found');
      if (!buckets.has(name)) buckets.set(name, new Map());
      return makeBucketHandle(name);
    },
    async removeEntry(name: string) {
      rootFiles.delete(name);
      buckets.delete(name);
    },
    async *keys() {
      for (const k of rootFiles.keys()) yield k;
      for (const k of buckets.keys()) yield k;
    },
  } as unknown as FileSystemDirectoryHandle;

  return { root, rootFiles, buckets };
}

function seedMetadata(
  rootFiles: Map<string, string>,
  data:
    | Partial<{
        baseUrl: string;
        entries: Array<[string, { size: number; order: number }]>;
        totalSize: number;
        orderCounter: number;
        contentHash: string | null;
        encodingVersion: number;
        validationMode: string;
        lastValidatedAt: number;
      }>
    | string
) {
  const payload = typeof data === 'string' ? data : JSON.stringify(data);
  rootFiles.set('_cache_meta.json', payload);
}

describe('OPFSMetadataManager', () => {
  let mgr: OPFSMetadataManager;
  beforeEach(() => {
    mgr = new OPFSMetadataManager();
  });

  describe('load()', () => {
    it('returns null on cold start (file missing)', async () => {
      const { root } = mockRoot();
      expect(await mgr.load(root)).toBeNull();
    });

    it('returns a populated LoadOutcome on a clean parse', async () => {
      const { root, rootFiles } = mockRoot();
      seedMetadata(rootFiles, {
        baseUrl: 'https://example.com/d.zarr',
        entries: [
          ['k1', { size: 100, order: 1 }],
          ['k2', { size: 200, order: 2 }],
        ],
        totalSize: 300,
        orderCounter: 3,
        contentHash: 'abcdef',
        encodingVersion: OPFS_ENCODING_VERSION,
        validationMode: 'content-hash',
        lastValidatedAt: 1000,
      });

      const outcome = await mgr.load(root);
      expect(outcome).not.toBeNull();
      expect(outcome!.index.get('k1')).toEqual({ size: 100, order: 1 });
      expect(outcome!.totalSize).toBe(300);
      expect(outcome!.orderCounter).toBe(3);
      expect(outcome!.contentHash).toBe('abcdef');
      expect(outcome!.validationMode).toBe('content-hash');
      expect(outcome!.lastValidatedAt).toBe(1000);
      expect(outcome!.needsOrphanCleanup).toBe(false);
    });

    it('starts fresh on encoding-version mismatch (no orphan cleanup signal)', async () => {
      const { root, rootFiles } = mockRoot();
      seedMetadata(rootFiles, {
        entries: [['k', { size: 1, order: 0 }]],
        totalSize: 1,
        orderCounter: 1,
        encodingVersion: OPFS_ENCODING_VERSION + 1,
      });

      const outcome = await mgr.load(root);
      expect(outcome).not.toBeNull();
      expect(outcome!.index.size).toBe(0);
      expect(outcome!.totalSize).toBe(0);
      expect(outcome!.needsOrphanCleanup).toBe(false);
      expect(mgr.parseFailures).toBe(0);
    });

    it('returns fresh + needsOrphanCleanup=true on JSON parse failure; bumps parseFailures', async () => {
      const { root, rootFiles } = mockRoot();
      seedMetadata(rootFiles, '{ this is not json');

      const outcome = await mgr.load(root);
      expect(outcome).not.toBeNull();
      expect(outcome!.index.size).toBe(0);
      expect(outcome!.needsOrphanCleanup).toBe(true);
      expect(mgr.parseFailures).toBe(1);
    });

    it('clamps NaN/Infinity/negative totalSize to entries-derived sum or 0', async () => {
      const { root, rootFiles } = mockRoot();
      seedMetadata(rootFiles, {
        entries: [['k', { size: 50, order: 0 }]],
        totalSize: Number.NaN,
        orderCounter: 1,
        encodingVersion: OPFS_ENCODING_VERSION,
      });

      const outcome = await mgr.load(root);
      // NaN persisted → persistedTotal becomes 0; abs(0 - 50) > 1 ⇒ recompute = 50.
      expect(outcome!.totalSize).toBe(50);
    });

    it('recomputes totalSize when persisted disagrees with entries by > 1 byte', async () => {
      const { root, rootFiles } = mockRoot();
      seedMetadata(rootFiles, {
        entries: [
          ['k1', { size: 100, order: 0 }],
          ['k2', { size: 200, order: 1 }],
        ],
        totalSize: 9999, // lies
        orderCounter: 2,
        encodingVersion: OPFS_ENCODING_VERSION,
      });

      const outcome = await mgr.load(root);
      expect(outcome!.totalSize).toBe(300);
    });

    it('keeps persisted totalSize when it matches entries within 1 byte', async () => {
      const { root, rootFiles } = mockRoot();
      seedMetadata(rootFiles, {
        entries: [['k1', { size: 100, order: 0 }]],
        totalSize: 100,
        orderCounter: 1,
        encodingVersion: OPFS_ENCODING_VERSION,
      });

      const outcome = await mgr.load(root);
      expect(outcome!.totalSize).toBe(100);
    });
  });

  describe('scheduleSave / hasPendingSave / cancelPendingSave / awaitInFlight', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });
    afterEach(() => {
      vi.useRealTimers();
    });

    function snap(overrides: Partial<MetadataSnapshot> = {}): MetadataSnapshot {
      return {
        baseUrl: 'https://example.com/d.zarr',
        entries: [['k', { size: 1, order: 0 }]],
        totalSize: 1,
        orderCounter: 1,
        contentHash: null,
        validationMode: 'none',
        ...overrides,
      };
    }

    it('debounces: rapid schedule calls produce one save with the latest snapshot', async () => {
      const { root, rootFiles } = mockRoot();
      const onError = vi.fn();
      const snapshots = [snap({ totalSize: 1 }), snap({ totalSize: 2 }), snap({ totalSize: 3 })];
      let i = 0;
      const getSnapshot = vi.fn(() => snapshots[i++] ?? snap({ totalSize: 99 }));

      // Three rapid schedules — last one wins.
      mgr.scheduleSave({ root, getSnapshot, delayMs: 1000, onError });
      mgr.scheduleSave({ root, getSnapshot, delayMs: 1000, onError });
      mgr.scheduleSave({ root, getSnapshot, delayMs: 1000, onError });
      expect(getSnapshot).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(1000);
      await mgr.awaitInFlight();

      // Only ONE save fired -> getSnapshot called once.
      expect(getSnapshot).toHaveBeenCalledTimes(1);
      // The first call to getSnapshot returns snapshots[0] = totalSize:1.
      const written = JSON.parse(rootFiles.get('_cache_meta.json')!);
      expect(written.totalSize).toBe(1);
      expect(onError).not.toHaveBeenCalled();
    });

    it('hasPendingSave is true between schedule and fire, false afterwards', async () => {
      const { root } = mockRoot();
      const onError = vi.fn();
      expect(mgr.hasPendingSave()).toBe(false);

      mgr.scheduleSave({ root, getSnapshot: snap, delayMs: 500, onError });
      expect(mgr.hasPendingSave()).toBe(true);

      await vi.advanceTimersByTimeAsync(500);
      expect(mgr.hasPendingSave()).toBe(false);
      await mgr.awaitInFlight();
    });

    it('cancelPendingSave prevents the save from firing', async () => {
      const { root, rootFiles } = mockRoot();
      const onError = vi.fn();
      const getSnapshot = vi.fn(() => snap());

      mgr.scheduleSave({ root, getSnapshot, delayMs: 1000, onError });
      mgr.cancelPendingSave();
      expect(mgr.hasPendingSave()).toBe(false);
      await vi.advanceTimersByTimeAsync(2000);

      expect(getSnapshot).not.toHaveBeenCalled();
      expect(rootFiles.has('_cache_meta.json')).toBe(false);
    });

    it('onError is invoked on write failure (does not throw)', async () => {
      // Build a root whose writable.write() throws.
      const failingRoot = {
        async getFileHandle() {
          return {
            async createWritable() {
              return {
                async write() {
                  throw new Error('quota exceeded');
                },
                async close() {},
              };
            },
          };
        },
      } as unknown as FileSystemDirectoryHandle;
      const onError = vi.fn();

      mgr.scheduleSave({ root: failingRoot, getSnapshot: snap, delayMs: 100, onError });
      await vi.advanceTimersByTimeAsync(100);
      await mgr.awaitInFlight();

      expect(onError).toHaveBeenCalledTimes(1);
      expect((onError.mock.calls[0][0] as Error).message).toBe('quota exceeded');
    });

    it('awaitInFlight resolves immediately when nothing is in flight', async () => {
      await expect(mgr.awaitInFlight()).resolves.toBeUndefined();
    });

    it('awaitInFlight awaits a save started by the timer', async () => {
      const { root, rootFiles } = mockRoot();
      const onError = vi.fn();
      mgr.scheduleSave({ root, getSnapshot: snap, delayMs: 100, onError });
      // Fire the timer; the inFlight promise should now exist.
      await vi.advanceTimersByTimeAsync(100);
      // awaitInFlight should resolve once the write completes.
      await mgr.awaitInFlight();
      expect(rootFiles.has('_cache_meta.json')).toBe(true);
    });

    it('direct save() swallows errors silently (used by dispose)', async () => {
      const failingRoot = {
        async getFileHandle() {
          throw new Error('opfs gone');
        },
      } as unknown as FileSystemDirectoryHandle;

      await expect(
        mgr.save(failingRoot, {
          baseUrl: 'x',
          entries: [],
          totalSize: 0,
          orderCounter: 0,
          contentHash: null,
          validationMode: 'none',
        })
      ).resolves.toBeUndefined();
    });
  });

  describe('cleanupOrphans', () => {
    it('removes files in hex buckets that are not in the expected set', async () => {
      const { root, buckets } = mockRoot();
      // Two hex buckets each with a few files.
      buckets.set(
        'aa',
        new Map([
          ['keep1', ''],
          ['drop1', ''],
          ['drop2', ''],
        ])
      );
      buckets.set('bb', new Map([['keep2', '']]));

      await mgr.cleanupOrphans(root, new Set(['keep1', 'keep2']));

      expect(mgr.orphansRemoved).toBe(2);
      expect(buckets.get('aa')!.has('drop1')).toBe(false);
      expect(buckets.get('aa')!.has('drop2')).toBe(false);
      expect(buckets.get('aa')!.has('keep1')).toBe(true);
      expect(buckets.get('bb')!.has('keep2')).toBe(true);
    });

    it('ignores non-hex bucket names (e.g. _cache_meta.json sibling files)', async () => {
      const { root, rootFiles, buckets } = mockRoot();
      // Root-level _cache_meta.json — must not be touched.
      rootFiles.set('_cache_meta.json', '{}');
      // Plus one hex bucket.
      buckets.set('00', new Map([['orphan', '']]));

      await mgr.cleanupOrphans(root, new Set());

      expect(mgr.orphansRemoved).toBe(1);
      expect(rootFiles.has('_cache_meta.json')).toBe(true);
      expect(buckets.get('00')!.has('orphan')).toBe(false);
    });

    it('halts before the next delete once shouldStop turns true (dispose mid-crawl)', async () => {
      const { root, buckets } = mockRoot();
      buckets.set(
        'aa',
        new Map([
          ['drop1', ''],
          ['drop2', ''],
        ])
      );
      buckets.set('bb', new Map([['drop3', '']]));

      // Flip to "stopped" after the first successful delete — the crawl must
      // bail before removing anything else, leaving the later orphans intact.
      // (The mock iterates in insertion order, so drop1 goes first.)
      await mgr.cleanupOrphans(root, new Set(), () => mgr.orphansRemoved >= 1);

      expect(mgr.orphansRemoved).toBe(1);
      expect(buckets.get('aa')!.has('drop1')).toBe(false);
      expect(buckets.get('aa')!.has('drop2')).toBe(true);
      expect(buckets.get('bb')!.has('drop3')).toBe(true);
    });
  });
});
