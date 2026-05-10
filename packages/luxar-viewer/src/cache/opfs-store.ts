import type { OPFSMetadata } from './types';
import { log, Modules } from '../utils/log';

type IterableFileSystemDirectoryHandle = FileSystemDirectoryHandle & {
  keys(): AsyncIterableIterator<string>;
};

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

  // Read/write tracking for monitoring. `readCount` is the L2 hit
  // counter — only incremented when get() returns a value. `missCount`
  // counts get() calls that returned undefined (file not present /
  // size mismatch / I/O error). Together they let consumers compute
  // an L2 hit rate without separate plumbing.
  private readCount = 0;
  private writeCount = 0;
  private missCount = 0;

  // Bucket handle cache (256 possible buckets: 00-ff)
  private bucketHandles = new Map<string, FileSystemDirectoryHandle>();

  // Debounced metadata save
  private metadataSaveTimeout: ReturnType<typeof setTimeout> | null = null;
  private static readonly METADATA_SAVE_DELAY = 1000; // 1 second debounce

  // Serialize concurrent writes to the same key to prevent race conditions
  private pendingWrites = new Map<string, Promise<void>>();

  // Generation token: every clear() bumps this. doSet() captures the
  // generation when it begins and discards its index/metadata mutation
  // if the generation has advanced — preventing a slow write that
  // started before clear() from repopulating the post-clear index.
  // Plain bookkeeping for stale-write detection; not a public API.
  private generation = 0;

  constructor(datasetId: string, baseUrl: string, maxSize: number) {
    this.datasetId = datasetId;
    this.baseUrl = baseUrl;
    this.maxSize = maxSize;
  }

  /**
   * Initialize OPFS directory and load metadata.
   */
  async init(): Promise<void> {
    try {
      const root = await navigator.storage.getDirectory();
      this.opfsRoot = await root.getDirectoryHandle(this.datasetId, { create: true });
      await this.loadMetadata();
    } catch (error) {
      log.warning(Modules.CACHE, 'OPFSStore failed to initialize', error);
      this.opfsRoot = null;
    }
  }

  /**
   * Get a file from OPFS and update LRU order.
   */
  async get(key: string): Promise<Uint8Array | undefined> {
    if (!this.opfsRoot) {
      this.missCount++;
      return undefined;
    }

    try {
      const fileHandle = await this.navigateToFile(key, false);
      const file = await fileHandle.getFile();
      const data = new Uint8Array(await file.arrayBuffer());

      // Verify size matches metadata
      const entry = this.index.get(key);
      if (entry && entry.size !== data.byteLength) {
        log.warning(Modules.CACHE, `OPFSStore size mismatch for ${key}, removing corrupted entry`);
        await this.delete(key);
        this.missCount++;
        return undefined;
      }

      // Update LRU order
      this.touch(key);
      this.readCount++;

      return data;
    } catch {
      this.missCount++;
      return undefined;
    }
  }

  /**
   * Write a file to OPFS with LRU eviction.
   */
  async set(key: string, data: Uint8Array): Promise<void> {
    if (!this.opfsRoot) return;

    // Await any pending write for this key to prevent race conditions
    const pending = this.pendingWrites.get(key);
    if (pending) {
      await pending;
    }

    const writePromise = this.doSet(key, data);
    this.pendingWrites.set(key, writePromise);
    try {
      await writePromise;
    } finally {
      this.pendingWrites.delete(key);
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

    // Check quota before writing
    if (!(await this.checkQuota(size))) {
      log.warning(Modules.CACHE, 'OPFSStore insufficient storage quota, skipping write');
      return;
    }

    // LRU eviction until we have space — O(1) per eviction via Map insertion order
    while (this.totalSize + size > this.maxSize && this.index.size > 0) {
      const lruKey = this.index.keys().next().value;
      if (lruKey !== undefined) {
        // Note: delete() already decrements totalSize, don't double-decrement
        await this.delete(lruKey);
      } else {
        break;
      }
    }

    // Write to OPFS (with one retry on stale bucket handle)
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const fileHandle = await this.navigateToFile(key, true);
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

        // Stale-write check: if clear()/dispose() ran while we were
        // writing, the post-write index update must be skipped. Best-
        // effort delete the file we just wrote so the directory matches
        // the (now-empty) index.
        if (this.generation !== startGeneration) {
          try {
            const bucket = this.getBucket(key);
            const bucketHandle = await this.getBucketHandle(bucket, false);
            if (bucketHandle) {
              await bucketHandle.removeEntry(this.keyToFileName(key));
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
          const bucket = this.getBucket(key);
          this.invalidateBucketHandle(bucket);
          continue;
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

    try {
      // Update totalSize before deleting
      const entry = this.index.get(key);
      if (entry) {
        this.totalSize -= entry.size;
        this.totalSize = Math.max(0, this.totalSize);
      }

      // Get bucket and delete file from it
      const bucket = this.getBucket(key);
      const bucketHandle = await this.getBucketHandle(bucket, false);
      if (bucketHandle) {
        const fileName = this.keyToFileName(key);
        await bucketHandle.removeEntry(fileName);
      }
      this.index.delete(key);
    } catch {
      // File doesn't exist, ignore
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
    this.bucketHandles = new Map();
    this.totalSize = 0;
    this.orderCounter = 0;
    this.contentHash = null;
    this.readCount = 0;
    this.writeCount = 0;
    this.missCount = 0;

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
  } {
    return {
      size: this.totalSize,
      count: this.index.size,
      reads: this.readCount,
      writes: this.writeCount,
      misses: this.missCount,
    };
  }

  /**
   * Update LRU order for a key.
   */
  touch(key: string): void {
    const entry = this.index.get(key);
    if (entry) {
      // Delete+re-insert to move to end (MRU position) — O(1) with Map
      this.index.delete(key);
      this.index.set(key, { size: entry.size, order: this.orderCounter++ });

      // Compact order counters to prevent overflow after long sessions
      if (this.orderCounter > 1e12) {
        this.compactOrderCounter();
      }

      this.scheduleMetadataSave();
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
   * Flush pending metadata writes and clear in-memory state.
   */
  async dispose(): Promise<void> {
    if (this.metadataSaveTimeout) {
      clearTimeout(this.metadataSaveTimeout);
      this.metadataSaveTimeout = null;
      await this.saveMetadata();
    }
  }

  // ========== Private Methods ==========

  /**
   * Compute bucket index (0-255) from cache key using simple hash.
   * Distributes ~65,000 files into ~256 buckets = ~250 files each.
   *
   * @returns Two-character hex string (00-ff)
   */
  private getBucket(key: string): string {
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      hash = ((hash << 5) - hash + key.charCodeAt(i)) | 0;
    }
    return (hash & 0xff).toString(16).padStart(2, '0');
  }

  /**
   * Get bucket directory handle, with caching.
   * Only 256 possible buckets, so caching is memory-efficient.
   *
   * If a cached handle turns out to be stale (e.g., directory was deleted during
   * a clear() operation), callers should use invalidateBucketHandle() and retry.
   */
  private async getBucketHandle(
    bucket: string,
    create: boolean
  ): Promise<FileSystemDirectoryHandle | null> {
    if (!this.opfsRoot) return null;

    // Check cache first
    const cached = this.bucketHandles.get(bucket);
    if (cached) return cached;

    try {
      const handle = await this.opfsRoot.getDirectoryHandle(bucket, { create });
      this.bucketHandles.set(bucket, handle);
      return handle;
    } catch {
      return null;
    }
  }

  /**
   * Invalidate a cached bucket handle (e.g., after a stale handle error).
   */
  private invalidateBucketHandle(bucket: string): void {
    this.bucketHandles.delete(bucket);
  }

  /**
   * Convert cache key to OPFS-safe filename using base64 encoding.
   * Example: "points/positions/0.0.0" → "cG9pbnRzL3Bvc2l0aW9ucy8wLjAuMA"
   */
  private keyToFileName(key: string): string {
    const base64 = btoa(key);
    // Replace base64 special chars with filesystem-safe alternatives
    return base64.replace(/\//g, '_').replace(/=/g, '-').replace(/\+/g, '.');
  }

  /**
   * Navigate to file within its bucket directory.
   * Structure: {root}/{bucket}/{base64-filename}
   */
  private async navigateToFile(key: string, create: boolean): Promise<FileSystemFileHandle> {
    const bucket = this.getBucket(key);
    const bucketHandle = await this.getBucketHandle(bucket, create);
    if (!bucketHandle) {
      throw new Error(`Cannot access bucket ${bucket}`);
    }
    const fileName = this.keyToFileName(key);
    return bucketHandle.getFileHandle(fileName, { create });
  }

  private scheduleMetadataSave(): void {
    if (this.metadataSaveTimeout) {
      clearTimeout(this.metadataSaveTimeout);
    }
    this.metadataSaveTimeout = setTimeout(() => {
      this.saveMetadata()
        .catch((error) => {
          log.warning(Modules.CACHE, 'OPFSStore metadata save failed', error);
        })
        .finally(() => {
          this.metadataSaveTimeout = null;
        });
    }, OPFSStore.METADATA_SAVE_DELAY);
  }

  private async loadMetadata(): Promise<void> {
    if (!this.opfsRoot) return;

    try {
      const metaHandle = await this.opfsRoot.getFileHandle('_cache_meta.json');
      const file = await metaHandle.getFile();
      const meta: OPFSMetadata = JSON.parse(await file.text());

      // Reconstruct Map sorted by ascending order so Map insertion order = LRU order
      const entries = (meta.entries || []).slice();
      entries.sort((a, b) => a[1].order - b[1].order);
      this.index = new Map(entries);
      this.totalSize = meta.totalSize || 0;
      this.orderCounter = meta.orderCounter || 0;
      this.contentHash = meta.contentHash || null;
    } catch {
      // No metadata yet, start fresh
      this.index = new Map();
      this.totalSize = 0;
      this.orderCounter = 0;
      this.contentHash = null;
    }
  }

  private async saveMetadata(): Promise<void> {
    if (!this.opfsRoot) return;

    try {
      const metaHandle = await this.opfsRoot.getFileHandle('_cache_meta.json', {
        create: true,
      });
      const writable = await metaHandle.createWritable();
      const metadata: OPFSMetadata = {
        baseUrl: this.baseUrl,
        entries: Array.from(this.index.entries()),
        totalSize: this.totalSize,
        orderCounter: this.orderCounter,
        contentHash: this.contentHash,
      };
      await writable.write(JSON.stringify(metadata));
      await writable.close();
    } catch (error) {
      // Ignore metadata save failures
      log.warning(Modules.CACHE, 'OPFSStore failed to save metadata', error);
    }
  }
}
