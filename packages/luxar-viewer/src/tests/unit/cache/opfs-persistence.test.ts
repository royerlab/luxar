/**
 * L2 (OPFS) persistence under continuous load.
 *
 * Measured in headless Chrome during real playback:
 *  - the index file (`_cache_meta.json`) was NEVER written while chunks kept
 *    arriving — its save was a pure trailing debounce, re-armed by every
 *    write (65k attempts, 0 saves in 53 s) — so a killed or navigated-away
 *    session left an index of ~40 entries for ~5,000 files on disk;
 *  - those unindexed files were never reclaimed (orphan cleanup only ran on
 *    a CORRUPT index), so disk use grew across interrupted sessions;
 *  - every write paid a `navigator.storage.estimate()` (2-5 ms), which made
 *    the background write queue saturate and drop 35-50% of fetched chunks.
 *
 * Each block below pins one of those, against the real `OPFSStore` over the
 * in-memory OPFS mock.
 */

import { describe, it, expect, afterEach, vi } from 'vitest';
import { OPFSStore } from '../../../cache/multi-level-caching-store/opfs-store';
import { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';
import { OpfsWriteQueue } from '../../../cache/multi-level-caching-store/opfs-write-queue';
import {
  getBucket,
  keyToFileName,
} from '../../../cache/multi-level-caching-store/opfs-store/buckets';
import { OPFS_ENCODING_VERSION } from '../../../cache/types';
import { createFakeOpfsRoot } from '../../mocks/opfs.mock';
import { perfCounters } from '../../../profiling/perf-counters';

const META = '_cache_meta.json';

/**
 * A dataset directory with REAL hex bucket subdirectories (the default fake is
 * flat, so its bucket dirs are not enumerable). Orphan reconciliation walks
 * `root.keys()` → `bucket.keys()`, which is what this models.
 */
function bucketedDatasetDir() {
  // Chrome's real NotFoundError text. (A message naming the file would contain
  // "json" for `_cache_meta.json`, which the metadata loader's parse-failure
  // sniff used to misread as a corrupt index.)
  const notFound = () =>
    new DOMException(
      'A requested file or directory could not be found at the time an operation was processed.',
      'NotFoundError'
    );
  const rootFiles = new Map<string, string>();
  const buckets = new Map<string, Map<string, Uint8Array>>();

  const bucketFileHandle = (files: Map<string, Uint8Array>, name: string) => ({
    kind: 'file' as const,
    name,
    async getFile() {
      const data = files.get(name) ?? new Uint8Array(0);
      return {
        size: data.byteLength,
        async arrayBuffer() {
          return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
        },
        async text() {
          return '';
        },
      };
    },
    async createWritable() {
      return {
        async write(data: ArrayBuffer | ArrayBufferView) {
          const bytes =
            data instanceof ArrayBuffer
              ? new Uint8Array(data.slice(0))
              : new Uint8Array(
                  data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
                );
          files.set(name, bytes);
        },
        async close() {},
        async abort() {},
      };
    },
  });

  const bucketHandle = (bucket: string) => ({
    kind: 'directory' as const,
    name: bucket,
    async getFileHandle(name: string, opts?: { create?: boolean }) {
      const files = buckets.get(bucket)!;
      if (!files.has(name) && !opts?.create) {
        throw notFound();
      }
      return bucketFileHandle(files, name);
    },
    async removeEntry(name: string) {
      const files = buckets.get(bucket);
      if (!files?.has(name)) throw notFound();
      files.delete(name);
    },
    async *keys() {
      for (const k of [...(buckets.get(bucket)?.keys() ?? [])]) yield k;
    },
  });

  const dir = {
    kind: 'directory' as const,
    name: 'dataset',
    async getFileHandle(name: string, opts?: { create?: boolean }) {
      if (!rootFiles.has(name) && !opts?.create) {
        throw notFound();
      }
      return {
        kind: 'file' as const,
        name,
        async getFile() {
          return {
            size: rootFiles.get(name)?.length ?? 0,
            async text() {
              return rootFiles.get(name) ?? '';
            },
            async arrayBuffer() {
              return new ArrayBuffer(0);
            },
          };
        },
        async createWritable() {
          return {
            async write(data: unknown) {
              rootFiles.set(name, typeof data === 'string' ? data : '');
            },
            async close() {},
            async abort() {},
          };
        },
      };
    },
    async getDirectoryHandle(name: string, opts?: { create?: boolean }) {
      if (!buckets.has(name)) {
        if (!opts?.create) throw notFound();
        buckets.set(name, new Map());
      }
      return bucketHandle(name);
    },
    async removeEntry(name: string) {
      if (rootFiles.delete(name)) return;
      if (buckets.delete(name)) return;
      throw notFound();
    },
    async *keys() {
      for (const k of [...rootFiles.keys()]) yield k;
      for (const k of [...buckets.keys()]) yield k;
    },
    async *entries() {},
  };

  /** Put a chunk file on disk WITHOUT indexing it (a write the index never saw). */
  function plant(key: string, bytes: Uint8Array): void {
    const bucket = getBucket(key);
    if (!buckets.has(bucket)) buckets.set(bucket, new Map());
    buckets.get(bucket)!.set(keyToFileName(key), bytes);
  }
  /** Put an arbitrary (non-luxar) file name into a bucket. */
  function plantRaw(bucket: string, name: string, bytes: Uint8Array): void {
    if (!buckets.has(bucket)) buckets.set(bucket, new Map());
    buckets.get(bucket)!.set(name, bytes);
  }
  function fileCount(): number {
    let n = 0;
    for (const files of buckets.values()) n += files.size;
    return n;
  }
  function wipe(): void {
    rootFiles.clear();
    buckets.clear();
  }
  return { dir, rootFiles, buckets, plant, plantRaw, fileCount, wipe };
}

function installBucketed() {
  const disk = bucketedDatasetDir();
  createFakeOpfsRoot({ datasetDir: disk.dir, onRemoveDataset: () => disk.wipe() }).install();
  return disk;
}

/** Let the store's background orphan reconcile finish (method is new — optional call). */
async function settleReconcile(store: OPFSStore): Promise<void> {
  await (
    store as unknown as { awaitOrphanReconcile?: () => Promise<void> }
  ).awaitOrphanReconcile?.();
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('OPFSStore index persistence under a continuous write stream', () => {
  it('writes the index at least every 2 s while set() keeps arriving every 30 ms', async () => {
    vi.useFakeTimers();
    const fake = createFakeOpfsRoot().install();
    const store = new OPFSStore('continuous', 'https://example.com/d.zarr', 1e9);
    await store.init();
    perfCounters.reset();

    for (let i = 0; i < 5000 / 30; i++) {
      await store.set(`chunk-${i}`, new Uint8Array(64));
      await vi.advanceTimersByTimeAsync(30);
    }

    expect(perfCounters.get('opfs.indexSaves')).toBeGreaterThanOrEqual(2);
    // And what is on disk is a recent index, not the empty one from init.
    const meta = JSON.parse(fake.metaFiles.get(META) ?? '{"entries":[]}');
    expect(meta.entries.length).toBeGreaterThan(100);
    await store.dispose();
  });

});

describe('OPFSStore orphan reconciliation on open', () => {
  async function sessionWithIndex(disk: ReturnType<typeof installBucketed>, keys: string[]) {
    const s = new OPFSStore('orph', 'https://example.com/d.zarr', 1e9);
    await s.init();
    await settleReconcile(s);
    s.setContentHash('hash-1');
    for (const k of keys) await s.set(k, new Uint8Array(10).fill(7));
    await s.dispose();
    expect(disk.rootFiles.has(META)).toBe(true);
  }

  it('re-indexes chunk files the last index never recorded (interrupted session)', async () => {
    const disk = installBucketed();
    await sessionWithIndex(disk, ['a/0', 'a/1', 'a/2']);
    // A killed session wrote these files but never saved the index.
    for (let i = 0; i < 40; i++) disk.plant(`b/${i}`, new Uint8Array(20).fill(i));

    const store = new OPFSStore('orph', 'https://example.com/d.zarr', 1e9);
    await store.init();
    await settleReconcile(store);

    // Every file on disk is now indexed and served as an L2 hit.
    expect(store.getStats().count).toBe(43);
    expect(store.getStats().orphansReindexed).toBe(40);
    expect(store.getStats().size).toBe(3 * 10 + 40 * 20);
    expect(await store.get('b/17')).toEqual(new Uint8Array(20).fill(17));
    expect(await store.get('a/1')).toEqual(new Uint8Array(10).fill(7));
    // Recovered entries are the LRU head: the index's own entries stay newer.
    await store.dispose();
    const meta = JSON.parse(disk.rootFiles.get(META)!);
    expect(meta.entries.length).toBe(43);
  });

  it('deletes unrecoverable files: foreign names and names in the wrong bucket', async () => {
    const disk = installBucketed();
    await sessionWithIndex(disk, ['a/0']);
    disk.plantRaw('00', '!!not-base64!!', new Uint8Array(5));
    // A valid base64url name placed in a bucket its key does not hash to.
    const key = 'misplaced/key';
    const wrongBucket = getBucket(key) === '00' ? '01' : '00';
    disk.plantRaw(wrongBucket, keyToFileName(key), new Uint8Array(5));

    const store = new OPFSStore('orph', 'https://example.com/d.zarr', 1e9);
    await store.init();
    await settleReconcile(store);

    expect(disk.fileCount()).toBe(1); // only a/0 remains
    expect(store.getStats().count).toBe(1);
    expect(store.getStats().orphanedFilesRemoved).toBe(2);
    await store.dispose();
  });

  it('deletes orphans that would push the index past maxSize', async () => {
    const disk = installBucketed();
    await sessionWithIndex(disk, ['a/0']); // 10 bytes indexed
    for (let i = 0; i < 20; i++) disk.plant(`b/${i}`, new Uint8Array(10));

    // maxSize 100: room for the indexed 10 B plus 9 recovered files.
    const store = new OPFSStore('orph', 'https://example.com/d.zarr', 100);
    await store.init();
    await settleReconcile(store);

    const stats = store.getStats();
    expect(stats.size).toBeLessThanOrEqual(100);
    expect(stats.count).toBe(10);
    // Disk matches the index: nothing left unaccounted for.
    expect(disk.fileCount()).toBe(stats.count);
    await store.dispose();
  });

  it('deletes every orphan when there is no trusted index (no content hash on disk)', async () => {
    const disk = installBucketed();
    for (let i = 0; i < 12; i++) disk.plant(`b/${i}`, new Uint8Array(10));
    expect(disk.fileCount()).toBe(12);

    // No _cache_meta.json at all: the files' provenance is unknown.
    const store = new OPFSStore('orph', 'https://example.com/d.zarr', 1e9);
    await store.init();
    expect(disk.fileCount(), 'files still on disk right after init').toBe(12);
    await settleReconcile(store);

    expect(disk.fileCount()).toBe(0);
    expect(store.getStats().count).toBe(0);
    expect(store.getStats().orphanedFilesRemoved).toBe(12);
    await store.dispose();
  });

  it('bounds the work per open and finishes over later sessions', async () => {
    const disk = installBucketed();
    disk.rootFiles.set(
      META,
      JSON.stringify({
        baseUrl: 'https://example.com/d.zarr',
        entries: [],
        totalSize: 0,
        orderCounter: 0,
        contentHash: 'hash-1',
        encodingVersion: OPFS_ENCODING_VERSION,
        validationMode: 'content-hash',
      })
    );
    const limit = (OPFSStore as unknown as { ORPHAN_RECONCILE_MAX_ACTIONS?: number })
      .ORPHAN_RECONCILE_MAX_ACTIONS;
    expect(limit, 'OPFSStore.ORPHAN_RECONCILE_MAX_ACTIONS').toBeGreaterThan(0);
    const total = limit! + 50;
    for (let i = 0; i < total; i++) disk.plant(`b/${i}`, new Uint8Array(4));

    const first = new OPFSStore('orph', 'https://example.com/d.zarr', 1e9);
    await first.init();
    await settleReconcile(first);
    expect(first.getStats().count).toBe(limit);
    await first.dispose();

    const second = new OPFSStore('orph', 'https://example.com/d.zarr', 1e9);
    await second.init();
    await settleReconcile(second);
    expect(second.getStats().count).toBe(total);
    await second.dispose();
  });

  it('a clear() while the reconcile runs leaves an empty store and disk', async () => {
    const disk = installBucketed();
    await sessionWithIndex(disk, ['a/0']);
    for (let i = 0; i < 200; i++) disk.plant(`b/${i}`, new Uint8Array(4));

    const store = new OPFSStore('orph', 'https://example.com/d.zarr', 1e9);
    await store.init();
    await store.clear();
    await settleReconcile(store);

    expect(store.getStats().count).toBe(0);
    expect(disk.fileCount()).toBe(0);
    await store.dispose();
  });
});

/**
 * Two tabs share one dataset directory (same URL ⇒ same datasetId). Tab B opened
 * the REPUBLISHED dataset: its validation cleared the directory and wrote the new
 * identity. Tab A, still at the old hash, keeps streaming chunks into that same
 * directory. A never rewrites the identity (it already "wrote" its own), so none
 * of A's later files are listed by B's index — and the next session, trusting the
 * new identity, must not serve them as the new dataset.
 */
describe('OPFSStore two tabs on one directory across a republish', () => {
  const URL = 'https://example.com/d.zarr';

  it.fails('never serves a chunk an old-hash tab wrote after the directory moved to the new hash', async () => {
    installBucketed();
    const a = new OPFSStore('tabs', URL, 1e9);
    await a.init();
    await settleReconcile(a);
    a.setContentHash('hash-old');
    await a.set('k/0', new Uint8Array(8).fill(1));

    const b = new OPFSStore('tabs', URL, 1e9);
    await b.init();
    await settleReconcile(b);
    await b.clear(); // B's validation saw the new hash
    b.setContentHash('hash-new');
    await b.set('k/1', new Uint8Array(8).fill(2));

    // A keeps streaming the OLD dataset's chunks into the shared directory.
    await a.set('k/2', new Uint8Array(8).fill(9));
    await a.dispose();
    await b.dispose(); // B's index (hash-new, k/1 only) is the last one saved

    const next = new OPFSStore('tabs', URL, 1e9);
    await next.init();
    next.setContentHash('hash-new'); // a matching validation
    expect(await next.get('k/2'), 'old-hash bytes served as the new dataset').toBeUndefined();
    expect(await next.get('k/1')).toEqual(new Uint8Array(8).fill(2));
    await settleReconcile(next);
    expect(next.getStats().count).toBe(1);
    await next.dispose();
  });
});

/**
 * A reload can land at ANY moment, including before the index save that would
 * have recorded the last chunk writes: a save started during unload never
 * completes across a navigation (measured on Chromium: a zero-byte index), and
 * the save itself is debounced. The chunk FILES are durable the moment their
 * write closes, so what must not depend on the index is whether the next
 * session can find them — and only once the dataset identity that makes
 * indexed entries trustworthy has been checked.
 */
describe('OPFSStore reload at any moment: chunk files outlive a lost index save', () => {
  const URL = 'https://example.com/d.zarr';
  const HASH = 'hash-reload';
  const N = 24;
  const bytesOf = (i: number) => new Uint8Array(16).fill(i + 1);
  const keyOf = (i: number) => `points/c/${i}`;

  type Disk = ReturnType<typeof installBucketed>;

  /** Every `_cache_meta.json` write from now on vanishes: the save the reload killed. */
  function dropIndexWrites(disk: Disk): void {
    const base = disk.dir.getFileHandle;
    disk.dir.getFileHandle = async (name: string, opts?: { create?: boolean }) => {
      const handle = await base(name, opts);
      if (name !== META) return handle;
      return {
        ...handle,
        async createWritable() {
          return { async write() {}, async close() {}, async abort() {} };
        },
      };
    };
  }

  /**
   * Hold the background orphan crawl at its first directory listing, so the
   * reads below are the FIRST-FRAME reads a reload makes before it finishes.
   */
  function holdReconcile(disk: Disk): () => void {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => (release = resolve));
    const base = disk.dir.keys;
    disk.dir.keys = async function* () {
      await gate;
      yield* base();
    };
    return () => {
      disk.dir.keys = base;
      release();
    };
  }

  /**
   * A session that validated `hash`, wrote N chunks, and was unloaded before any
   * index save after the first `savedFirst` writes could land. Abandoned, not
   * disposed: an unload runs no teardown.
   */
  async function interruptedSession(disk: Disk, hash: string, savedFirst = 0): Promise<void> {
    const intactGetFileHandle = disk.dir.getFileHandle;
    const s = new OPFSStore('reload', URL, 1e9);
    await s.init();
    await settleReconcile(s);
    s.setContentHash(hash);
    const meta = (s as unknown as { metadata: { awaitInFlight(): Promise<void> } }).metadata;
    if (savedFirst === 0) dropIndexWrites(disk);
    for (let i = 0; i < N; i++) {
      if (savedFirst > 0 && i === savedFirst) {
        // The last index save that completed: it lists the first `savedFirst`.
        s.flushMetadata();
        await meta.awaitInFlight();
        dropIndexWrites(disk);
      }
      await s.set(keyOf(i), bytesOf(i));
    }
    (s as unknown as { metadata: { cancelPendingSave(): void } }).metadata.cancelPendingSave();
    disk.dir.getFileHandle = intactGetFileHandle; // the next session's saves land
    expect(disk.fileCount(), 'chunk files the interrupted session left on disk').toBe(N);
  }

  /** Reopen, validate the same hash, read every chunk before the crawl runs. */
  async function expectAllServedOnFirstRead(disk: Disk): Promise<void> {
    const release = holdReconcile(disk);
    const store = new OPFSStore('reload', URL, 1e9);
    await store.init();
    store.setContentHash(HASH); // what a matching validation does before reads
    const served: number[] = [];
    for (let i = 0; i < N; i++) {
      const data = await store.get(keyOf(i));
      if (data && data.every((b) => b === i + 1) && data.byteLength === 16) served.push(i);
    }
    expect(served.length, 'chunks served from L2 on the first read').toBe(N);
    expect(store.getStats().misses).toBe(0);
    expect(disk.fileCount(), 'files on disk after the first reads').toBe(N);

    release();
    await settleReconcile(store);
    expect(disk.fileCount(), 'files on disk after the reconcile').toBe(N);
    const stats = store.getStats();
    expect(stats.count).toBe(N);
    expect(stats.size).toBe(N * 16);
    await store.dispose();
    // The final save now lists every recovered chunk.
    expect(JSON.parse(disk.rootFiles.get(META) ?? '{"entries":[]}').entries.length).toBe(N);
  }

  it('no index save ever landed (unload before the first save)', async () => {
    const disk = installBucketed();
    await interruptedSession(disk, HASH);
    expect(disk.rootFiles.has(META)).toBe(false);
    await expectAllServedOnFirstRead(disk);
  });

  it('a zero-byte index (a save cut off by the navigation)', async () => {
    const disk = installBucketed();
    await interruptedSession(disk, HASH);
    disk.rootFiles.set(META, '');
    await expectAllServedOnFirstRead(disk);
  });

  it('a stale index missing the last writes (reload inside the debounce)', async () => {
    const disk = installBucketed();
    await interruptedSession(disk, HASH, N - 6);
    expect(JSON.parse(disk.rootFiles.get(META)!).entries.length).toBe(N - 6);
    await expectAllServedOnFirstRead(disk);
  });

  it('a different content hash still invalidates the unindexed files', async () => {
    const disk = installBucketed();
    await interruptedSession(disk, 'hash-old');
    const store = new OPFSStore('reload', URL, 1e9);
    await store.init();
    // Validation compares the remote hash against the cached one; the cached
    // one must survive the lost index save, or a changed dataset cannot be seen.
    expect(store.getContentHash()).toBe('hash-old');
    await store.clear(); // what a mismatching validation does
    store.setContentHash(HASH);
    for (let i = 0; i < N; i++) expect(await store.get(keyOf(i))).toBeUndefined();
    await settleReconcile(store);
    expect(disk.fileCount()).toBe(0);
    await store.dispose();
  });

  it('a zero-byte chunk file (a chunk write cut off before close) is not served', async () => {
    const disk = installBucketed();
    await interruptedSession(disk, HASH);
    const torn = 'points/c/torn';
    disk.plant(torn, new Uint8Array(0));
    const store = new OPFSStore('reload', URL, 1e9);
    await store.init();
    store.setContentHash(HASH);
    expect(await store.get(torn)).toBeUndefined();
    await settleReconcile(store);
    expect(store.getStats().count).toBe(N);
    await store.dispose();
  });
});

describe('MultiLevelCachingStore reload at any moment: no network for chunks already on disk', () => {
  const BASE = 'https://example.com/reload.zarr';

  function stubSource(hash: { value: string }) {
    const chunkFetches: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        const path = String(url).replace(`${BASE}/`, '');
        const json = (o: unknown) => new TextEncoder().encode(JSON.stringify(o)).buffer;
        if (path === '.zattrs') {
          return {
            ok: true,
            async arrayBuffer() {
              return json({ content_hash: hash.value });
            },
          };
        }
        if (path === 'zarr.json') {
          return {
            ok: true,
            async arrayBuffer() {
              return json({
                zarr_format: 3,
                node_type: 'group',
                attributes: { content_hash: hash.value },
              });
            },
          };
        }
        chunkFetches.push(path);
        const bytes = new TextEncoder().encode(`${hash.value}:${path}`);
        return {
          ok: true,
          async arrayBuffer() {
            return bytes.buffer;
          },
        };
      })
    );
    return chunkFetches;
  }

  async function session(): Promise<MultiLevelCachingStore> {
    const s = new MultiLevelCachingStore(BASE, { l1MaxSize: 20 * 1024 * 1024, l2MaxSize: 1e9 });
    await s.init();
    return s;
  }

  const keys = Array.from({ length: 16 }, (_, i) => `points/c/${i}`);

  async function interrupted(disk: ReturnType<typeof installBucketed>, hash: { value: string }) {
    stubSource(hash);
    const a = await session();
    // The index save never lands — the reload killed it.
    const base = disk.dir.getFileHandle;
    disk.dir.getFileHandle = async (name: string, opts?: { create?: boolean }) => {
      const handle = await base(name, opts);
      if (name !== META) return handle;
      return {
        ...handle,
        async createWritable() {
          return { async write() {}, async close() {}, async abort() {} };
        },
      };
    };
    for (const k of keys) await a.get(k);
    await (a as unknown as { l2WriteQueue: { drain(): Promise<void> } }).l2WriteQueue.drain();
    const l2 = (a as unknown as { l2Store: { metadata: { cancelPendingSave(): void } } }).l2Store;
    l2.metadata.cancelPendingSave();
    disk.dir.getFileHandle = base;
    expect(disk.fileCount()).toBeGreaterThanOrEqual(keys.length);
  }

  it('serves every chunk written before the reload from L2', async () => {
    const disk = installBucketed();
    const hash = { value: 'hash-a' };
    await interrupted(disk, hash);

    const chunkFetches = stubSource(hash);
    const b = await session();
    for (const k of keys) {
      expect(new TextDecoder().decode(await b.get(k))).toBe(`hash-a:${k}`);
    }
    expect(chunkFetches, 'chunks re-fetched after the reload').toEqual([]);
    expect(b.getStats().demand.l2Hits).toBe(keys.length);
    await b.dispose();
  });

  it('a changed content hash re-fetches every chunk (no stale bytes)', async () => {
    const disk = installBucketed();
    const hash = { value: 'hash-a' };
    await interrupted(disk, hash);

    hash.value = 'hash-b';
    const chunkFetches = stubSource(hash);
    const b = await session();
    for (const k of keys) {
      expect(new TextDecoder().decode(await b.get(k))).toBe(`hash-b:${k}`);
    }
    expect(chunkFetches.length).toBe(keys.length);
    await b.dispose();
  });
});

describe('OPFSStore quota estimate caching', () => {
  it('does not call navigator.storage.estimate() on every write', async () => {
    const estimate = vi.fn(async () => ({ quota: 10e9, usage: 1e9 }));
    createFakeOpfsRoot({ estimate }).install();
    const store = new OPFSStore('quota-cache', 'https://example.com/d.zarr', 1e9);
    await store.init();
    estimate.mockClear();

    for (let i = 0; i < 200; i++) await store.set(`k${i}`, new Uint8Array(1024));

    expect(store.getStats().writes).toBe(200);
    expect(estimate.mock.calls.length).toBeLessThanOrEqual(2);
    await store.dispose();
  });

  it('re-estimates once the cached answer is 30 s old', async () => {
    vi.useFakeTimers();
    const estimate = vi.fn(async () => ({ quota: 10e9, usage: 1e9 }));
    createFakeOpfsRoot({ estimate }).install();
    const store = new OPFSStore('quota-age', 'https://example.com/d.zarr', 1e9);
    await store.init();
    await store.set('a', new Uint8Array(8));
    const afterFirst = estimate.mock.calls.length;
    await store.set('b', new Uint8Array(8));
    expect(estimate.mock.calls.length).toBe(afterFirst);

    await vi.advanceTimersByTimeAsync(30_001);
    await store.set('c', new Uint8Array(8));
    expect(estimate.mock.calls.length).toBe(afterFirst + 1);
    await store.dispose();
  });

  it('never admits a write the cached estimate cannot cover: it re-estimates first', async () => {
    // The cache must not turn a full disk into silently-failing writes: the
    // cached headroom is debited per byte written, and once it runs out the
    // store asks the browser again rather than trusting a stale "plenty".
    let usage = 0;
    const estimate = vi.fn(async () => ({ quota: 1000, usage }));
    createFakeOpfsRoot({ estimate }).install();
    const store = new OPFSStore('quota-debit', 'https://example.com/d.zarr', 1e9);
    await store.init();
    usage = 0;
    for (let i = 0; i < 5; i++) {
      await store.set(`k${i}`, new Uint8Array(100));
      usage += 100;
    }
    // Something else on the origin fills the quota behind our back.
    usage = 1000;
    const calls = estimate.mock.calls.length;
    // 500 B of cached headroom remain on paper; a 460 B write (506 B with the
    // x1.1 margin) does not fit in it, so the store must re-estimate, then refuse
    // or evict rather than write past the real quota.
    await store.set('big', new Uint8Array(460));
    expect(estimate.mock.calls.length).toBeGreaterThan(calls);
    await store.dispose();
  });
});

describe('OPFSStore write path buffer handling', () => {
  it('writes the caller view directly instead of copying its buffer', async () => {
    const fake = createFakeOpfsRoot().install();
    const written: unknown[] = [];
    const baseGetFileHandle = fake.datasetDir.getFileHandle;
    fake.datasetDir.getFileHandle = async (...args: any[]) => {
      const handle = await baseGetFileHandle(...args);
      const baseCreate = handle.createWritable.bind(handle);
      return {
        ...handle,
        async createWritable() {
          const w = await baseCreate();
          return {
            ...w,
            async write(data: unknown) {
              written.push(data);
              return w.write(data);
            },
          };
        },
      };
    };
    const store = new OPFSStore('nocopy', 'https://example.com/d.zarr', 1e9);
    await store.init();
    written.length = 0;

    // A sub-view on a larger buffer: only the view's bytes may land on disk.
    const backing = new Uint8Array(64).map((_, i) => i);
    const view = backing.subarray(8, 24);
    await store.set('view', view);

    expect(written[0]).toBe(view);
    expect(await store.get('view')).toEqual(backing.slice(8, 24));
    await store.dispose();
  });
});

describe('Background write queue + OPFSStore under playback-rate arrivals', () => {
  it('drops far fewer chunks once writes stop paying a per-write quota estimate', async () => {
    vi.useFakeTimers();
    // Realistic costs: estimate() ~5 ms, a chunk write ~1 ms. One chunk
    // arrives per ms (a playback stream); the queue drains 4-wide.
    const estimate = vi.fn(
      () =>
        new Promise<{ quota: number; usage: number }>((resolve) =>
          setTimeout(() => resolve({ quota: 10e9, usage: 1e9 }), 5)
        )
    );
    const fake = createFakeOpfsRoot({ estimate }).install();
    const store = new OPFSStore('rate', 'https://example.com/d.zarr', 10e9);
    await store.init();
    // Slow close() only AFTER init: init's write probe is not timer-driven here.
    const baseGetFileHandle = fake.datasetDir.getFileHandle;
    fake.datasetDir.getFileHandle = async (...args: any[]) => {
      const handle = await baseGetFileHandle(...args);
      const baseCreate = handle.createWritable.bind(handle);
      return {
        ...handle,
        async createWritable() {
          const w = await baseCreate();
          return {
            ...w,
            async close() {
              await new Promise((r) => setTimeout(r, 1));
              return w.close();
            },
          };
        },
      };
    };

    const chunk = 128 * 1024;
    const queue = new OpfsWriteQueue({
      concurrency: 4,
      maxDepth: 16_384,
      maxBytes: 4 * 1024 * 1024,
    });
    const n = 1000;
    for (let i = 0; i < n; i++) {
      const data = new Uint8Array(chunk);
      queue.enqueue(`c${i}`, () => store.set(`c${i}`, data), chunk);
      await vi.advanceTimersByTimeAsync(1);
    }
    await vi.advanceTimersByTimeAsync(5000);
    await queue.drain();

    const dropped = queue.stats().dropped;
    // Accounting is exact: every arrival is either written or counted dropped.
    expect(store.getStats().writes + dropped).toBe(n);
    // Before: ~1/3 of arrivals dropped (6 ms per write / 4 wide > 1 ms arrivals).
    expect(dropped).toBeLessThan(n * 0.02);
    // The slow close() is timer-driven: keep the clock moving through dispose.
    const disposed = store.dispose();
    await vi.advanceTimersByTimeAsync(100);
    await disposed;
  });
});
