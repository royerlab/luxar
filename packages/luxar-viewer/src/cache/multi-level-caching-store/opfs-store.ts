import type { CacheValidationMode } from '../types';
import { log, Modules } from '../../utils/log';
import { config } from '../../config';
import { OPFSBucketCache, getBucket, keyToFileName } from './opfs-store/buckets';
import { OPFSMetadataManager, type MetadataSnapshot } from './opfs-store/metadata';
import { withTimeout } from './opfs-store/opfs-timeout';

type IterableFileSystemDirectoryHandle = FileSystemDirectoryHandle & {
  keys(): AsyncIterableIterator<string>;
  entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
};

interface PersistedMetadataFile {
  baseUrl?: string;
  contentHash?: string;
  totalSize?: number;
  entries?: unknown[];
}

export interface CachedDatasetSummary {
  url: string;
  hash: string;
  size: number;
  count: number;
}

/**
 * OPFS persistence layer (L2 cache) with LRU eviction and shallow bucketing.
 *
 * Uses 256 bucket directories to distribute files and avoid filesystem limits.
 * Files are stored as: `{bucket}/{base64-encoded-key}`
 *
 * Structure:
 * ```
 * zarr-cache-{hash}/
 * ├── 00/           (~250 files per bucket)
 * │   ├── UG9pbnRz...
 * │   └── ...
 * ├── 01/
 * ├── ...
 * ├── ff/
 * └── _cache_meta.json
 * ```
 *
 * Benefits:
 * - Max ~250-500 files per directory instead of 65,000+
 * - Only 256 bucket handles to cache (trivial memory)
 * - clear() iterates 256 directories, not 65,000 files
 * - Preserves fast flat access (2 async calls vs 1)
 */
export class OPFSStore {
  private opfsRoot: FileSystemDirectoryHandle | null = null;
  private index = new Map<string, { size: number; order: number }>();
  private orderCounter = 0;
  private totalSize = 0;
  private maxSize: number;
  private datasetId: string;
  private baseUrl: string;
  private contentHash: string | null = null;
  // External-dataset validation mode + last-validated timestamp.
  // Persisted to OPFSMetadata so the next session can apply a TTL
  // window across page loads.
  private validationMode: CacheValidationMode = 'none';
  private lastValidatedAt: number | null = null;

  // Read/write tracking for monitoring. `readCount` is the L2 hit
  // counter — only incremented when get() returns a value. `missCount`
  // counts get() calls that returned undefined (file not present /
  // size mismatch / I/O error). Together they let consumers compute
  // an L2 hit rate without separate plumbing.
  private readCount = 0;
  private writeCount = 0;
  private missCount = 0;

  // Health counters surfaced via getStats().
  // `oversizedWriteSkipped`: doSet() rejected an entry larger than
  //   maxSize so it could not have been written without violating the
  //   cache size invariant.
  // `quotaWriteSkipped`: navigator.storage.estimate() reported
  //   insufficient quota even after own-LRU eviction.
  // `evictions`: number of own-LRU entries evicted to make room for
  //   incoming writes.
  // `writeFailures`: doSet() catch branch — file I/O threw after
  //   retries.
  // `corruptedEntries`: get() detected a size mismatch between index
  //   and on-disk data and removed the bad file.
  // Counters: `parseFailures` (loadMetadata could not JSON-parse) and
  // `orphansRemoved` (cleanupOrphans reclaim count) live on the
  // metadata manager and are read by getStats().
  private oversizedWriteSkipped = 0;
  private quotaWriteSkipped = 0;
  private evictions = 0;
  private writeFailures = 0;
  private corruptedEntries = 0;

  // Bucket handle cache (256 possible buckets: 00-ff)
  private buckets = new OPFSBucketCache();

  // Owns _cache_meta.json: load/save + debounced-save timer + orphan reclaim.
  private metadata = new OPFSMetadataManager();
  private static readonly METADATA_SAVE_DELAY = 1000; // 1 second debounce

  // Serialize concurrent writes to the same key to prevent race conditions
  private pendingWrites = new Map<string, Promise<void>>();

  // Generation token: every clear() bumps this. doSet() captures the
  // generation when it begins and discards its index/metadata mutation
  // if the generation has advanced — preventing a slow write that
  // started before clear() from repopulating the post-clear index.
  // Plain bookkeeping for stale-write detection; not a public API.
  private generation = 0;

  // Lifecycle: set by dispose(). Synchronous early-return on
  // get/set/touch so a disposed store cannot mutate state. Distinct
  // from the no-OPFS path (`!this.opfsRoot`) — disposed means the
  // owner explicitly tore the store down, OPFS-unavailable means the
  // browser never gave us a directory.
  private disposed = false;

  constructor(datasetId: string, baseUrl: string, maxSize: number) {
    this.datasetId = datasetId;
    this.baseUrl = baseUrl;
    this.maxSize = maxSize;
  }

  /**
   * Enumerate every `zarr-cache-*` dataset present in OPFS. Reads each
   * dataset's `_cache_meta.json` directly — no OPFSStore instance is
   * created. Used by the debug-cache helpers (`window.__luxarDebug`)
   * and the cache E2E suite to inspect persisted datasets without
   * mounting them.
   */
  static async listAll(): Promise<CachedDatasetSummary[]> {
    const datasets: CachedDatasetSummary[] = [];
    try {
      const opfsRoot = await navigator.storage.getDirectory();
      const iterableRoot = opfsRoot as IterableFileSystemDirectoryHandle;
      for await (const [name, handle] of iterableRoot.entries()) {
        if (!name.startsWith('zarr-cache-') || handle.kind !== 'directory') continue;
        try {
          const directoryHandle = handle as FileSystemDirectoryHandle;
          const metaHandle = await directoryHandle.getFileHandle('_cache_meta.json');
          const file = await metaHandle.getFile();
          const meta = JSON.parse(await file.text()) as PersistedMetadataFile;
          datasets.push({
            url: meta.baseUrl || 'unknown',
            hash: meta.contentHash?.slice(0, 16) || 'none',
            size: meta.totalSize || 0,
            count: meta.entries?.length || 0,
          });
        } catch {
          // Skip corrupted/invalid cache directories.
        }
      }
    } catch {
      // OPFS not available.
    }
    return datasets;
  }

  /**
   * Initialize OPFS directory and load metadata.
   */
  async init(): Promise<void> {
    try {
      const root = await navigator.storage.getDirectory();
      this.opfsRoot = await root.getDirectoryHandle(this.datasetId, { create: true });
      await this.applyMetadataOnLoad();
    } catch (error) {
      log.warning(Modules.CACHE, 'OPFSStore failed to initialize', error);
      this.opfsRoot = null;
    }
  }

  private async applyMetadataOnLoad(): Promise<void> {
    if (!this.opfsRoot) return;
    const outcome = await this.metadata.load(this.opfsRoot);
    if (!outcome) {
      // Cold start (file missing). Defaults already match the constructor.
      return;
    }
    this.index = outcome.index;
    this.totalSize = outcome.totalSize;
    this.orderCounter = outcome.orderCounter;
    this.contentHash = outcome.contentHash;
    this.validationMode = outcome.validationMode;
    this.lastValidatedAt = outcome.lastValidatedAt;
    if (outcome.needsOrphanCleanup) {
      const expected = new Set<string>();
      for (const key of this.index.keys()) {
        expected.add(keyToFileName(key));
      }
      await this.metadata.cleanupOrphans(this.opfsRoot, expected).catch(() => {
        // Best-effort; ignore reclaim failures.
      });
    }
  }

  private metadataSnapshot(): MetadataSnapshot {
    return {
      baseUrl: this.baseUrl,
      entries: Array.from(this.index.entries()),
      totalSize: this.totalSize,
      orderCounter: this.orderCounter,
      contentHash: this.contentHash,
      validationMode: this.validationMode,
      lastValidatedAt: this.lastValidatedAt ?? undefined,
    };
  }

  /**
   * Get a file from OPFS and update LRU order.
   */
  async get(key: string): Promise<Uint8Array | undefined> {
    if (this.disposed || !this.opfsRoot) {
      this.missCount++;
      return undefined;
    }

    // The index is the source of truth for what's cached. A key absent
    // from it is either genuinely uncached or an orphaned file — e.g. a
    // generation-skipped write whose best-effort delete failed, leaving
    // bytes on disk that a content-hash invalidation meant to drop.
    // Serving such a file could resurrect stale data, so treat an
    // unindexed key as a miss. Bonus: skips an OPFS read for uncached
    // keys (the common cold-miss path).
    if (!this.index.has(key)) {
      this.missCount++;
      return undefined;
    }

    const timeoutMs = config.cache.opfsOperationTimeoutMs;
    try {
      const data = await withTimeout(
        (async () => {
          const fileHandle = await this.buckets.navigateToFile(this.opfsRoot!, key, false);
          const file = await fileHandle.getFile();
          return new Uint8Array(await file.arrayBuffer());
        })(),
        timeoutMs,
        `get(${key})`
      );

      // Verify size matches metadata
      const entry = this.index.get(key);
      if (entry && entry.size !== data.byteLength) {
        log.warning(Modules.CACHE, `OPFSStore size mismatch for ${key}, removing corrupted entry`);
        this.corruptedEntries++;
        await this.delete(key);
        this.missCount++;
        return undefined;
      }

      // Update LRU order
      this.touch(key);
      this.readCount++;

      return data;
    } catch (error) {
      // Timeouts and any other I/O failures degrade to a cache miss;
      // production code never observes a throw here (zarrita's
      // AsyncReadable.get must not throw).
      const msg = error instanceof Error ? error.message : String(error);
      if (msg.startsWith('OPFS timeout')) {
        log.warning(Modules.CACHE, msg);
      }
      this.missCount++;
      return undefined;
    }
  }

  /**
   * Write a file to OPFS with LRU eviction.
   */
  async set(key: string, data: Uint8Array): Promise<void> {
    if (this.disposed || !this.opfsRoot) return;

    // Chain this write onto any in-flight write for the same key so the
    // doSet() bodies run strictly in arrival order. Previously we only
    // awaited the single pending write at entry; if ≥2 callers awaited the
    // same promise they would resume together and run doSet() — and thus
    // createWritable()+write()+close() on the SAME OPFS file — concurrently.
    // The File System Access API does not guarantee overlapping writables
    // to one file are safe (they can corrupt the file or throw). Chaining
    // makes a given key's file I/O strictly sequential. (The in-memory
    // index update cannot itself race: it is a synchronous block with no
    // await, so it is atomic per call.) `prev.then(run, run)` runs our
    // write whether the previous one resolved or rejected (doSet swallows
    // its own errors, but stay defensive). The tail-check in `finally`
    // avoids a finishing earlier write deleting a newer writer's entry.
    const prev = this.pendingWrites.get(key);
    const run = (): Promise<void> => this.doSet(key, data);
    const writePromise = prev ? prev.then(run, run) : run();
    this.pendingWrites.set(key, writePromise);
    try {
      await writePromise;
    } finally {
      if (this.pendingWrites.get(key) === writePromise) {
        this.pendingWrites.delete(key);
      }
    }
  }

  /**
   * Internal write implementation (called by set() after serialization).
   */
  private async doSet(key: string, data: Uint8Array): Promise<void> {
    if (!this.opfsRoot) return;

    // Capture the generation at entry. If clear() (or dispose()) bumps
    // the generation while the write is pending, the post-write index
    // mutation must be skipped — otherwise a slow set() that began
    // before clear() will repopulate the just-cleared cache.
    const startGeneration = this.generation;

    const size = data.byteLength;

    // Reject oversized entries up front. A single item larger than
    // maxSize would otherwise evict every existing entry and still
    // leave totalSize > maxSize after insertion, breaking the cache
    // size invariant. Mirror of LRUCache.set()'s oversized guard.
    if (size > this.maxSize) {
      this.oversizedWriteSkipped++;
      return;
    }

    // LRU eviction until we have space — O(1) per eviction via Map
    // insertion order. Run BEFORE the quota check so the browser sees
    // the freed space when we ask navigator.storage.estimate().
    // Each iteration must make progress (index shrinks) — bail if delete()
    // fails to remove the head entry (transient I/O error) so we never
    // spin forever re-selecting the same undeletable key, and so evictions
    // counts only entries actually removed.
    while (this.totalSize + size > this.maxSize && this.index.size > 0) {
      const before = this.index.size;
      const lruKey = this.index.keys().next().value;
      if (lruKey === undefined) break;
      // Note: delete() already decrements totalSize, don't double-decrement
      await this.delete(lruKey);
      if (this.index.size === before) break; // no progress — avoid spinning on a failing delete
      this.evictions++;
    }

    // Quota-pressure eviction. The own-LRU loop above only fires when
    // `totalSize` approaches `maxSize`, but the browser-granted OPFS quota
    // is frequently smaller than `maxSize` (default 2 GB) — on Firefox,
    // private mode, and small disks. Without this loop, a tight quota
    // would skip the write while the maxSize-based trigger never fires, so
    // the cache freezes holding stale entries and silently drops new ones
    // (TODO #4). checkQuota() reflects bytes actually on disk and delete()
    // frees them, so evicting LRU entries genuinely recovers quota. Evict
    // and re-check until the write fits or there is nothing left to evict.
    // Each iteration must make progress (index shrinks) — bail if delete()
    // fails to remove the head entry (transient I/O error) so we never spin.
    let hasQuota = await this.checkQuota(size);
    while (!hasQuota && this.index.size > 0) {
      const before = this.index.size;
      const lruKey = this.index.keys().next().value;
      if (lruKey === undefined) break;
      await this.delete(lruKey);
      if (this.index.size === before) break; // no progress — avoid spinning on a failing delete
      this.evictions++;
      hasQuota = await this.checkQuota(size);
    }

    if (!hasQuota) {
      this.quotaWriteSkipped++;
      log.warning(
        Modules.CACHE,
        'OPFSStore insufficient storage quota even after eviction, skipping write'
      );
      return;
    }

    // Write to OPFS (with one retry on stale bucket handle). The
    // entire navigate→createWritable→write→close chain is wrapped in
    // withTimeout so a hung handle cannot stall the cache.
    const timeoutMs = config.cache.opfsOperationTimeoutMs;
    // HIGH-2 fix: a single broken write must count as 1 in writeFailures,
    // not once per retry attempt. Track whether we've already counted a
    // failure for this call so the second attempt doesn't double-count.
    let writeFailedThisCall = false;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        await withTimeout(
          (async () => {
            const fileHandle = await this.buckets.navigateToFile(this.opfsRoot!, key, true);
            const writable = await fileHandle.createWritable();
            // Slice the view's portion, NOT data.buffer directly. If the Uint8Array is
            // a view on a larger ArrayBuffer (e.g., from a sub-slice), data.buffer would
            // write the entire underlying buffer, corrupting the stored data. slice()
            // copies only the relevant bytes. Cast is safe: network data is never SharedArrayBuffer.
            const bytes = data.buffer.slice(
              data.byteOffset,
              data.byteOffset + data.byteLength
            ) as ArrayBuffer;
            await writable.write(bytes);
            await writable.close();
          })(),
          timeoutMs,
          `set(${key})`
        );

        // Stale-write check: if clear()/dispose() ran while we were
        // writing, the post-write index update must be skipped. Best-
        // effort delete the file we just wrote so the directory matches
        // the (now-empty) index.
        if (this.generation !== startGeneration) {
          try {
            const bucket = getBucket(key);
            const bucketHandle = await this.buckets.getHandle(this.opfsRoot!, bucket, false);
            if (bucketHandle) {
              await bucketHandle.removeEntry(keyToFileName(key));
            }
          } catch {
            // Best-effort; orphaned file is harmless and will be reclaimed
            // by the next clear() / orphan cleanup pass.
          }
          return;
        }

        // Update index — delete+re-insert to move to end (MRU position)
        const existingEntry = this.index.get(key);
        if (existingEntry) {
          this.totalSize -= existingEntry.size;
          this.totalSize = Math.max(0, this.totalSize);
          this.index.delete(key);
        }
        this.index.set(key, { size, order: this.orderCounter++ });
        this.totalSize += size;
        this.writeCount++;

        // Compact order counters to prevent overflow after long sessions
        if (this.orderCounter > 1e12) {
          this.compactOrderCounter();
        }

        // Debounced metadata save
        this.scheduleMetadataSave();
        return; // Success — exit retry loop
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        if (attempt === 0 && errorMsg.includes('could not be found')) {
          // Stale bucket handle from a concurrent clear() — invalidate and retry
          const bucket = getBucket(key);
          this.buckets.invalidate(bucket);
          continue;
        }
        if (!writeFailedThisCall) {
          this.writeFailures++;
          writeFailedThisCall = true;
        }
        log.warning(Modules.CACHE, `OPFSStore failed to write ${key}: ${errorMsg}`);
      }
    }
  }

  /**
   * Delete a file from OPFS.
   */
  async delete(key: string): Promise<void> {
    if (!this.opfsRoot) return;

    const entry = this.index.get(key);
    if (!entry) return;

    try {
      // [cache OOS] Order matters: file removal must succeed BEFORE
      // we update `totalSize` and the index. The previous order
      // (totalSize decrement → removeEntry → index.delete) left the
      // accumulator desynchronised when removeEntry threw — totalSize
      // had already been decremented but the file was still on disk
      // and the index still had the entry. The next `set()` then made
      // eviction decisions based on the wrong size. With the new
      // ordering, an exception in removeEntry leaves all three pieces
      // of state — totalSize, index, disk — consistent as if delete()
      // had never been called, so callers can retry safely.
      const bucket = getBucket(key);
      const bucketHandle = await this.buckets.getHandle(this.opfsRoot, bucket, false);
      if (bucketHandle) {
        const fileName = keyToFileName(key);
        await bucketHandle.removeEntry(fileName);
      }
      this.totalSize = Math.max(0, this.totalSize - entry.size);
      this.index.delete(key);
    } catch {
      // File doesn't exist (or transient I/O error), ignore. State is
      // unchanged so callers can retry.
    }
  }

  /**
   * Clear all OPFS data for this dataset.
   *
   * Uses atomic delete-and-recreate instead of iterating entries, which avoids
   * race conditions when a previous page context still holds open file handles
   * (e.g., quick-succession page refreshes with fire-and-forget L2 writes).
   */
  async clear(): Promise<void> {
    // Bump generation FIRST so any in-flight doSet() that completes
    // after this point sees the mismatch and skips its index update.
    this.generation++;

    // Drain pending same-key writes. Each set() pushes a promise into
    // pendingWrites; awaiting them lets in-flight writes finish their
    // file I/O — they will detect the generation mismatch and skip
    // mutating the index.
    if (this.pendingWrites.size > 0) {
      await Promise.allSettled([...this.pendingWrites.values()]);
    }

    // Clear all in-memory state first — ensures no stale handles are used
    // even if the filesystem operations below fail
    this.index = new Map();
    this.buckets.clear();
    this.totalSize = 0;
    this.orderCounter = 0;
    this.contentHash = null;
    this.validationMode = 'none';
    this.lastValidatedAt = null;
    this.readCount = 0;
    this.writeCount = 0;
    this.missCount = 0;
    this.oversizedWriteSkipped = 0;
    this.quotaWriteSkipped = 0;
    this.evictions = 0;
    this.writeFailures = 0;
    this.corruptedEntries = 0;
    this.metadata.parseFailures = 0;
    this.metadata.orphansRemoved = 0;

    if (this.opfsRoot) {
      try {
        // Atomic approach: remove entire dataset directory and recreate it.
        // This is more robust than iterating entries, which can fail if a
        // previous page context still holds open file handles on bucket dirs.
        const root = await navigator.storage.getDirectory();
        await root.removeEntry(this.datasetId, { recursive: true });
        this.opfsRoot = await root.getDirectoryHandle(this.datasetId, { create: true });
      } catch (error) {
        log.warning(
          Modules.CACHE,
          'OPFSStore failed to clear atomically, retrying entry-by-entry',
          error
        );
        // Fallback: try to remove entries individually (best-effort)
        try {
          if (this.opfsRoot) {
            const iterableRoot = this.opfsRoot as IterableFileSystemDirectoryHandle;
            for await (const name of iterableRoot.keys()) {
              try {
                await this.opfsRoot.removeEntry(name, { recursive: true });
              } catch {
                // Skip locked/in-use entries — they'll be orphaned but harmless
              }
            }
          }
        } catch {
          // Iterator itself failed — directory may be inaccessible
        }
        // Re-obtain a fresh root handle regardless
        try {
          const root = await navigator.storage.getDirectory();
          this.opfsRoot = await root.getDirectoryHandle(this.datasetId, { create: true });
        } catch {
          this.opfsRoot = null;
        }
      }
    }
  }

  /**
   * Check if enough storage quota is available.
   */
  async checkQuota(requiredBytes: number): Promise<boolean> {
    try {
      const estimate = await navigator.storage.estimate();
      const available = (estimate.quota || 0) - (estimate.usage || 0);
      return available > requiredBytes * 1.1; // 10% safety margin
    } catch {
      return true; // Assume OK if API unavailable
    }
  }

  /**
   * Get cache statistics.
   */
  getStats(): {
    size: number;
    count: number;
    reads: number;
    writes: number;
    misses: number;
    oversizedWriteSkipped: number;
    quotaWriteSkipped: number;
    evictions: number;
    writeFailures: number;
    corruptedEntries: number;
    metadataParseFailures: number;
    orphanedFilesRemoved: number;
    /**
     * S2: `true` when OPFS was reachable on init and the store is
     * still alive. `false` when init couldn't acquire a directory
     * handle (browser without OPFS support, private mode in some
     * configs) or after dispose(). Drives the `opfs-unavailable`
     * status badge.
     */
    available: boolean;
  } {
    return {
      size: this.totalSize,
      count: this.index.size,
      reads: this.readCount,
      writes: this.writeCount,
      misses: this.missCount,
      oversizedWriteSkipped: this.oversizedWriteSkipped,
      quotaWriteSkipped: this.quotaWriteSkipped,
      evictions: this.evictions,
      writeFailures: this.writeFailures,
      corruptedEntries: this.corruptedEntries,
      metadataParseFailures: this.metadata.parseFailures,
      orphanedFilesRemoved: this.metadata.orphansRemoved,
      available: this.opfsRoot !== null && !this.disposed,
    };
  }

  /**
   * Update LRU order for a key.
   */
  touch(key: string): void {
    if (this.disposed) return;
    const entry = this.index.get(key);
    if (entry) {
      // Delete+re-insert to move to end (MRU position) — O(1) with Map
      this.index.delete(key);
      this.index.set(key, { size: entry.size, order: this.orderCounter++ });

      // Compact order counters to prevent overflow after long sessions
      if (this.orderCounter > 1e12) {
        this.compactOrderCounter();
      }

      // Intentionally do NOT schedule a metadata save here. touch() only
      // reorders the in-memory LRU; persisting that on every read forced a
      // full-index JSON serialize per read-burst. Read-driven order is
      // best-effort — it rides along with the next structural save (set /
      // delete / contentHash / validationMode) and is flushed
      // unconditionally on dispose(). Losing it on a hard crash only yields
      // a slightly stale eviction order next session, which is harmless.
    }
  }

  /**
   * Set content hash for cache invalidation.
   */
  setContentHash(hash: string | null): void {
    this.contentHash = hash;
    this.scheduleMetadataSave();
  }

  /**
   * Get cached content hash.
   */
  getContentHash(): string | null {
    return this.contentHash;
  }

  /**
   * Record the validation mode used for this dataset. Persisted to
   * `_cache_meta.json` so a follow-up session can re-evaluate (e.g.
   * a TTL window).
   */
  setValidationMode(mode: CacheValidationMode): void {
    this.validationMode = mode;
    this.lastValidatedAt = Date.now();
    this.scheduleMetadataSave();
  }

  /**
   * Read the current validation mode and last-validated timestamp.
   * Returned together so callers can apply a TTL check atomically.
   */
  getValidationState(): { mode: CacheValidationMode; lastValidatedAt: number | null } {
    return { mode: this.validationMode, lastValidatedAt: this.lastValidatedAt };
  }

  /**
   * Renumber all order entries to prevent orderCounter overflow.
   * Called when orderCounter exceeds a safe threshold (1e12).
   */
  private compactOrderCounter(): void {
    const entries = [...this.index.entries()].sort((a, b) => a[1].order - b[1].order);
    this.index.clear();
    entries.forEach(([key, entry], i) => {
      entry.order = i;
      this.index.set(key, entry);
    });
    this.orderCounter = entries.length;
  }

  /**
   * Tear down the store. Drains pending writes, awaits any in-flight
   * metadata save, then flushes a final snapshot UNCONDITIONALLY before
   * marking the store disposed so subsequent set/get/touch are no-ops.
   *
   * Order matters:
   * 1. Bump generation FIRST so any in-flight doSet that resolves
   *    afterwards detects the mismatch and skips its index update.
   * 2. Cancel the debounced timer (we flush directly below).
   * 3. Drain pendingWrites so file I/O for in-flight set() calls finishes
   *    and the index reflects settled state before we snapshot it.
   * 4. Await any metadata save already mid-write so the final save wins
   *    on disk (last writer), then write the latest snapshot. The flush is
   *    unconditional (not gated on hasPendingSave) because read-driven LRU
   *    order no longer schedules its own save (see touch()); dispose is
   *    where a read-only session's order gets persisted.
   * 5. Set disposed = true.
   */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.generation++;

    this.metadata.cancelPendingSave();

    if (this.pendingWrites.size > 0) {
      await Promise.allSettled([...this.pendingWrites.values()]);
    }

    await this.metadata.awaitInFlight();
    if (this.opfsRoot) {
      await this.metadata.save(this.opfsRoot, this.metadataSnapshot());
    }

    this.disposed = true;
  }

  // ========== Private Methods ==========

  private scheduleMetadataSave(): void {
    if (!this.opfsRoot) return;
    this.metadata.scheduleSave({
      root: this.opfsRoot,
      getSnapshot: () => this.metadataSnapshot(),
      delayMs: OPFSStore.METADATA_SAVE_DELAY,
      onError: (error) => {
        this.writeFailures++;
        log.warning(Modules.CACHE, 'OPFSStore metadata save failed', error);
      },
    });
  }
}
