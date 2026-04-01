import type { OPFSMetadata } from './types';
import { log, Modules } from '../utils/log';

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

  // Read/write tracking for monitoring
  private readCount = 0;
  private writeCount = 0;

  // Bucket handle cache (256 possible buckets: 00-ff)
  private bucketHandles = new Map<string, FileSystemDirectoryHandle>();

  // Debounced metadata save
  private metadataSaveTimeout: ReturnType<typeof setTimeout> | null = null;
  private static readonly METADATA_SAVE_DELAY = 1000; // 1 second debounce

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
    if (!this.opfsRoot) return undefined;

    try {
      const fileHandle = await this.navigateToFile(key, false);
      const file = await fileHandle.getFile();
      const data = new Uint8Array(await file.arrayBuffer());

      // Verify size matches metadata
      const entry = this.index.get(key);
      if (entry && entry.size !== data.byteLength) {
        log.warning(Modules.CACHE, `OPFSStore size mismatch for ${key}, removing corrupted entry`);
        await this.delete(key);
        return undefined;
      }

      // Update LRU order
      this.touch(key);
      this.readCount++;

      return data;
    } catch {
      return undefined;
    }
  }

  /**
   * Write a file to OPFS with LRU eviction.
   */
  async set(key: string, data: Uint8Array): Promise<void> {
    if (!this.opfsRoot) return;

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

    // Write to OPFS
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

      // Update index — delete+re-insert to move to end (MRU position)
      const existingEntry = this.index.get(key);
      if (existingEntry) {
        this.totalSize -= existingEntry.size;
        this.index.delete(key);
      }
      this.index.set(key, { size, order: this.orderCounter++ });
      this.totalSize += size;
      this.writeCount++;

      // Debounced metadata save
      this.scheduleMetadataSave();
    } catch (error) {
      // Log error with message (error objects don't serialize well in console)
      const errorMsg = error instanceof Error ? error.message : String(error);
      log.warning(Modules.CACHE, `OPFSStore failed to write ${key}: ${errorMsg}`);
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
   */
  async clear(): Promise<void> {
    if (this.opfsRoot) {
      try {
        // Remove all entries (bucket directories + metadata file)
        for await (const name of (this.opfsRoot as any).keys()) {
          await this.opfsRoot.removeEntry(name, { recursive: true });
        }
      } catch (error) {
        log.warning(Modules.CACHE, 'OPFSStore failed to clear', error);
      }
    }
    // Clear all in-memory state
    this.index = new Map();
    this.bucketHandles = new Map();
    this.totalSize = 0;
    this.orderCounter = 0;
    this.contentHash = null;
    this.readCount = 0;
    this.writeCount = 0;
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
  getStats(): { size: number; count: number; reads: number; writes: number } {
    return {
      size: this.totalSize,
      count: this.index.size,
      reads: this.readCount,
      writes: this.writeCount,
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
   * Flush pending metadata writes and clear in-memory state.
   */
  async dispose(): Promise<void> {
    if (this.metadataSaveTimeout) {
      clearTimeout(this.metadataSaveTimeout);
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
      this.saveMetadata().catch(() => {});
      this.metadataSaveTimeout = null;
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
