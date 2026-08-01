/**
 * OPFSStore correctness regressions found while reviewing the cache stack:
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
 *
 *  - #1073 delete/write clobber: `delete(key)` wraps `removeEntry()` in a
 *    timeout. When the timeout wins the caller returns while the
 *    non-cancellable removeEntry keeps running; a same-key replacement write
 *    could complete first and then be deleted by that straggling removeEntry,
 *    leaving the index describing bytes no longer on disk. The fix serializes
 *    the REAL removeEntry ahead of any later same-key write (a delete-barrier
 *    captured synchronously at set() entry), reconciles off the real
 *    settlement, drains in-flight deletes at dispose(), and degrades safely
 *    (dropping the write) if a delete never settles.
 */

import { describe, it, expect, afterEach, vi } from 'vitest';
import { OPFSStore } from '../../../cache/multi-level-caching-store/opfs-store';

/**
 * File System Access mock that tracks how many writers are concurrently
 * inside the createWritable→close window for each file path, so a test can
 * assert that same-key writes never overlap. `write()` yields a microtask
 * to widen the window and make any overlap observable.
 */
function mockOPFS(opts?: { onRemoveEntry?: (name: string) => Promise<void> | void }) {
  const files = new Map<string, Uint8Array>();
  const metaFiles = new Map<string, string>();
  const activePerPath = new Map<string, number>();
  let maxConcurrentSameFile = 0;
  // Total writers in flight across ALL (non-metadata) files at once. Lets a
  // test prove that distinct-key writes genuinely overlap in time (i.e.
  // per-key serialization does NOT serialize unrelated keys).
  let activeTotal = 0;
  let maxConcurrentAnyFile = 0;
  // Set true if a delete's removeEntry ever runs while a writer is inside the
  // createWritable→close window on the SAME file (the mutual-exclusion the
  // #1073 chaining must guarantee in both directions).
  let removeOverlappedWrite = false;

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
      // Optional hook lets a test GATE removeEntry (return a promise released
      // manually) to reproduce a removeEntry that outruns delete()'s timeout.
      if (opts?.onRemoveEntry) await opts.onRemoveEntry(name);
      if ((activePerPath.get(name) ?? 0) > 0) removeOverlappedWrite = true;
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
    getRemoveOverlappedWrite: () => removeOverlappedWrite,
  };
}

describe('OPFSStore correctness (#1 write serialization, #3 orphan reads, #1073 delete/write clobber)', () => {
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

  it('#1073 a timed-out delete never clobbers a later same-key replacement write', async () => {
    // Gate removeEntry so it does not resolve until we release it — modelling a
    // removeEntry that keeps running after delete()'s caller gives up on its
    // timeout. Before the fix, the replacement write completed first and the
    // straggling removeEntry then deleted it, leaving the index describing bytes
    // no longer on disk (get() → undefined).
    let gated = false; // enable only after init() so the write-probe cleanup runs
    let releaseRemove!: () => void;
    const removeGate = new Promise<void>((resolve) => {
      releaseRemove = resolve;
    });
    const { getRemoveOverlappedWrite } = mockOPFS({
      onRemoveEntry: async () => {
        if (gated) await removeGate;
      },
    });

    const { config } = await import('../../../config');
    const originalTimeout = config.cache.opfsOperationTimeoutMs;
    const store = new OPFSStore('race', 'https://example.com', 10 * 1024 * 1024);

    try {
      config.cache.opfsOperationTimeoutMs = 30; // short deadline so delete() returns fast
      await store.init();
      gated = true;

      // (1) Original entry lands: file on disk + index entry.
      await store.set('race', new Uint8Array([1]));
      expect(store.getStats()).toMatchObject({ size: 1, count: 1 });

      // (2) delete('race') — removeEntry is gated, so the caller returns on the
      // 30ms timeout with index/size preserved (reconcile happens off the REAL
      // settlement, not the early timeout return).
      await store.delete('race');
      expect(store.getStats()).toMatchObject({ size: 1, count: 1 });

      // Widen the deadline BEFORE the replacement write so a slow CI cannot make
      // the delete-barrier itself time out (the barrier reads config lazily).
      config.cache.opfsOperationTimeoutMs = 10_000;

      // (3) Replacement write. It must NOT start while the older delete's
      // removeEntry is still in flight, so it waits on the delete-barrier.
      const setPromise = store.set('race', new Uint8Array([2, 3]));
      let setDone = false;
      void setPromise.then(() => {
        setDone = true;
      });

      // (4) The replacement is genuinely HELD by the barrier: after a real delay
      // it still has not completed. Pre-fix it would have finished here and the
      // released removeEntry below would then clobber it.
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(setDone).toBe(false);

      // (5) Release the gated removeEntry: the real delete settles (reconciling
      // index/size), THEN the replacement write runs on the now-clean slot.
      releaseRemove();
      await setPromise;

      // (6) The replacement survives and index/size match disk.
      expect(await store.get('race')).toEqual(new Uint8Array([2, 3]));
      const stats = store.getStats();
      expect(stats.count).toBe(1);
      expect(stats.size).toBe(2);
      // removeEntry serialized ahead of the write — they never overlapped.
      expect(getRemoveOverlappedWrite()).toBe(false);
    } finally {
      config.cache.opfsOperationTimeoutMs = originalTimeout;
      releaseRemove(); // ensure no dangling gated promise
      await store.dispose();
    }
  });

  it('#1073 the delete-barrier is per-key — a gated delete never blocks another key', async () => {
    // Serialization is per key: a delete of 'A' hung on the gate must not delay
    // an unrelated write to 'B'.
    let gated = false; // enable only after init() so the write-probe cleanup runs
    let releaseRemove!: () => void;
    const removeGate = new Promise<void>((resolve) => {
      releaseRemove = resolve;
    });
    mockOPFS({
      onRemoveEntry: async () => {
        if (gated) await removeGate;
      },
    });

    const { config } = await import('../../../config');
    const originalTimeout = config.cache.opfsOperationTimeoutMs;
    const store = new OPFSStore('race-perkey', 'https://example.com', 10 * 1024 * 1024);

    try {
      config.cache.opfsOperationTimeoutMs = 30;
      await store.init();
      gated = true;

      await store.set('A', new Uint8Array([1]));
      await store.set('B', new Uint8Array([1]));
      // Start a delete of 'A' that hangs on the gate (not awaited to completion).
      const delA = store.delete('A');

      // A write to 'B' shares no chain with 'A' and must complete promptly.
      await store.set('B', new Uint8Array([9]));
      expect(await store.get('B')).toEqual(new Uint8Array([9]));

      releaseRemove();
      await delA;
      // 'A' is gone once its (now released) removeEntry settles; only 'B' left.
      expect(await store.get('A')).toBeUndefined();
      expect(store.getStats()).toMatchObject({ count: 1, size: 1 });
    } finally {
      config.cache.opfsOperationTimeoutMs = originalTimeout;
      releaseRemove();
      await store.dispose();
    }
  });

  it('#1073 a delete arriving AFTER a queued same-key write does not self-stall/drop that write', async () => {
    // The delete-barrier is snapshotted SYNCHRONOUSLY at set() entry. A delete
    // that arrives AFTER a write joined the chain is sequenced BEHIND that write
    // (real = thatWrite.then(doDelete)); it must NOT be in the write's barrier,
    // or the write would wait on a `real` chained behind itself — burning the
    // full timeout and dropping a valid write. Reproduce: set#1 in-flight, set#2
    // queued, THEN delete arrives.
    const { getRemoveOverlappedWrite } = mockOPFS();

    const { config } = await import('../../../config');
    const originalTimeout = config.cache.opfsOperationTimeoutMs;
    const store = new OPFSStore('selfwait', 'https://example.com', 10 * 1024 * 1024);

    try {
      // Large deadline: a self-stall would burn all of it, which the timing
      // assertion below would catch.
      config.cache.opfsOperationTimeoutMs = 2000;
      await store.init();

      // Commit K first so the later delete has an index entry to act on.
      await store.set('K', new Uint8Array([0]));

      const set1 = store.set('K', new Uint8Array([1]));
      const set2 = store.set('K', new Uint8Array([2, 2])); // queued behind set1
      const del = store.delete('K'); // chains behind set2 — must not enter set2's barrier

      const start = Date.now();
      await Promise.all([set1, set2]);
      const elapsed = Date.now() - start;

      // set#2 completed promptly (no self-wait) and was NOT dropped.
      expect(elapsed).toBeLessThan(500);
      expect(store.getStats().clobberBarrierWriteSkipped).toBe(0);

      // The delete runs AFTER both writes and removes the key.
      await del;
      expect(await store.get('K')).toBeUndefined();
      expect(store.getStats()).toMatchObject({ count: 0, size: 0 });
      expect(getRemoveOverlappedWrite()).toBe(false);
    } finally {
      config.cache.opfsOperationTimeoutMs = originalTimeout;
      await store.dispose();
    }
  });

  it('#1073 degrade-safe — a never-settling delete drops the same-key write, and a later write recovers', async () => {
    let gated = false;
    let releaseRemove!: () => void;
    const removeGate = new Promise<void>((resolve) => {
      releaseRemove = resolve;
    });
    mockOPFS({
      onRemoveEntry: async () => {
        if (gated) await removeGate;
      },
    });

    const { config } = await import('../../../config');
    const originalTimeout = config.cache.opfsOperationTimeoutMs;
    const store = new OPFSStore('degrade', 'https://example.com', 10 * 1024 * 1024);

    try {
      config.cache.opfsOperationTimeoutMs = 30;
      await store.init();
      gated = true;

      await store.set('race', new Uint8Array([1]));
      // Delete whose removeEntry never resolves within the deadline (not released).
      await store.delete('race');
      // Capture the still-in-flight real delete so we can await it after release.
      const inflight = (
        OPFSStore as unknown as {
          pendingDeletesByDataset: Map<string, Map<string, Set<Promise<void>>>>;
        }
      ).pendingDeletesByDataset
        .get('degrade')!
        .get('race')!;

      // Same-key write: its barrier waits on the hung delete, times out, and the
      // write is DROPPED — resolving (never rejecting) and incrementing the
      // counter rather than risking a clobber.
      await expect(store.set('race', new Uint8Array([7, 7]))).resolves.toBeUndefined();
      expect(store.getStats().clobberBarrierWriteSkipped).toBe(1);
      // The dropped value is not served; state stays consistent (the old, still
      // on-disk entry is what the index describes — degrade-safe, no clobber).
      expect(await store.get('race')).toEqual(new Uint8Array([1]));

      // Once the delete finally settles it reconciles: the key becomes a miss.
      releaseRemove();
      await Promise.allSettled([...inflight]);
      expect(await store.get('race')).toBeUndefined();
      expect(store.getStats()).toMatchObject({ count: 0, size: 0 });

      // A later write (no delete in flight now) recovers the key normally.
      await store.set('race', new Uint8Array([9]));
      expect(await store.get('race')).toEqual(new Uint8Array([9]));
      expect(store.getStats()).toMatchObject({ count: 1, size: 1 });
    } finally {
      config.cache.opfsOperationTimeoutMs = originalTimeout;
      releaseRemove();
      await store.dispose();
    }
  });

  it('#1073 dispose() drains an in-flight delete so the final snapshot matches disk', async () => {
    // doDelete chains behind pendingWrites and can be outstanding when dispose()
    // runs. dispose() must drain it (bounded) so the deferred removeEntry +
    // reconcile land BEFORE metadataSnapshot() — otherwise the persisted
    // snapshot lists a key that is being deleted (a cross-instance #1073).
    let gated = false;
    let releaseRemove!: () => void;
    const removeGate = new Promise<void>((resolve) => {
      releaseRemove = resolve;
    });
    const { metaFiles } = mockOPFS({
      onRemoveEntry: async () => {
        if (gated) await removeGate;
      },
    });

    const { config } = await import('../../../config');
    const originalTimeout = config.cache.opfsOperationTimeoutMs;
    const store = new OPFSStore('dispose-drain', 'https://example.com', 10 * 1024 * 1024);

    try {
      config.cache.opfsOperationTimeoutMs = 30;
      await store.init();
      gated = true;

      await store.set('doomed', new Uint8Array([1]));
      // Delete times out at the caller but its real removeEntry stays in flight.
      await store.delete('doomed');

      // Begin teardown; release the gate so the drained delete can settle.
      const disposePromise = store.dispose();
      releaseRemove();
      await disposePromise;

      // The persisted snapshot must NOT list the deleted key.
      const meta = JSON.parse(metaFiles.get('_cache_meta.json') ?? '{}');
      const persistedKeys: string[] = (meta.entries ?? []).map((e: [string, unknown]) => e[0]);
      expect(persistedKeys).not.toContain('doomed');
    } finally {
      config.cache.opfsOperationTimeoutMs = originalTimeout;
      releaseRemove();
    }
  });

  it('#1073 delete() during an in-flight first write of the key serializes behind it (no silent no-op)', async () => {
    // A delete arriving while the key's write is still in flight — index not
    // yet updated — must chain behind that write and remove it. An early
    // return keyed on the index alone would silently skip the delete and let
    // the write survive an operation that arrived after it.
    const { getRemoveOverlappedWrite } = mockOPFS();
    const store = new OPFSStore('inflight-del', 'https://example.com', 10 * 1024 * 1024);

    try {
      await store.init();

      // First write of the key: set() installs the pending write and suspends
      // inside doSet() before the index is updated.
      const setPromise = store.set('fresh', new Uint8Array([1, 2, 3]));
      // The key is not indexed yet, but the write is pending — the delete must
      // still take effect (sequenced after the write).
      const deletePromise = store.delete('fresh');
      await Promise.all([setPromise, deletePromise]);

      expect(await store.get('fresh')).toBeUndefined();
      expect(store.getStats()).toMatchObject({ count: 0, size: 0 });
      expect(getRemoveOverlappedWrite()).toBe(false);
    } finally {
      await store.dispose();
    }
  });

  it('#1073 a predecessor delete that outlives dispose() still gates a successor store same-key write', async () => {
    // dispose()'s delete-drain is bounded, so a removeEntry can outlive the
    // instance. The pending-delete registry is class-level and keyed by
    // datasetId precisely so a successor same-URL store (same datasetId, same
    // shared OPFS directory) barriers on the predecessor's straggling delete
    // instead of writing a replacement that the old removeEntry then clobbers.
    let gated = false;
    let releaseRemove!: () => void;
    const removeGate = new Promise<void>((resolve) => {
      releaseRemove = resolve;
    });
    const { getRemoveOverlappedWrite } = mockOPFS({
      onRemoveEntry: async (name) => {
        // Gate only data files: the successor's init() write-probe cleanup
        // must stay un-gated or init itself would hang on the gate.
        if (gated && name !== '.opfs-write-probe') await removeGate;
      },
    });

    const { config } = await import('../../../config');
    const originalTimeout = config.cache.opfsOperationTimeoutMs;
    const predecessor = new OPFSStore('xinstance', 'https://example.com', 10 * 1024 * 1024);
    let successor: OPFSStore | null = null;

    try {
      config.cache.opfsOperationTimeoutMs = 30;
      await predecessor.init();
      gated = true;

      await predecessor.set('k', new Uint8Array([1]));
      // Caller returns on the 30ms timeout; the real removeEntry stays in flight.
      await predecessor.delete('k');
      // The bounded dispose drain times out too — the removeEntry OUTLIVES the
      // instance.
      await predecessor.dispose();

      config.cache.opfsOperationTimeoutMs = 10_000;
      successor = new OPFSStore('xinstance', 'https://example.com', 10 * 1024 * 1024);
      await successor.init();

      // The successor's replacement write must be HELD by the predecessor's
      // still-outstanding delete, not land and then be clobbered by it.
      const setPromise = successor.set('k', new Uint8Array([2, 2]));
      let setDone = false;
      void setPromise.then(() => {
        setDone = true;
      });
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(setDone).toBe(false);

      // Release the straggler: it settles first, THEN the successor writes.
      releaseRemove();
      await setPromise;

      expect(await successor.get('k')).toEqual(new Uint8Array([2, 2]));
      expect(getRemoveOverlappedWrite()).toBe(false);
    } finally {
      config.cache.opfsOperationTimeoutMs = originalTimeout;
      releaseRemove();
      await predecessor.dispose();
      if (successor) await successor.dispose();
    }
  });
});
