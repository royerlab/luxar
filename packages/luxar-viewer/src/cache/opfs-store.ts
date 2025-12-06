import type { OPFSMetadata } from './types';

/**
 * OPFS persistence layer (L2 cache) with LRU eviction.
 * Handles all File System Access API interactions for zarr chunk storage.
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
      console.warn('[OPFSStore] Failed to initialize:', error);
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
        console.warn(`[OPFSStore] Size mismatch for ${key}, removing corrupted entry`);
        await this.delete(key);
        return undefined;
      }

      // Update LRU order
      this.touch(key);

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
      console.warn('[OPFSStore] Insufficient storage quota, skipping write');
      return;
    }

    // LRU eviction until we have space
    while (this.totalSize + size > this.maxSize && this.index.size > 0) {
      // Find LRU entry (lowest order number)
      let lruKey: string | null = null;
      let lruOrder = Infinity;
      for (const [k, v] of this.index) {
        if (v.order < lruOrder) {
          lruOrder = v.order;
          lruKey = k;
        }
      }

      if (lruKey) {
        const entry = this.index.get(lruKey)!;
        await this.delete(lruKey);
        this.totalSize -= entry.size;
      }
    }

    // Write to OPFS
    try {
      const fileHandle = await this.navigateToFile(key, true);
      const writable = await fileHandle.createWritable();
      // Type assertion: Uint8Arrays from network are always ArrayBuffer, not SharedArrayBuffer
      await writable.write(data.buffer as ArrayBuffer);
      await writable.close();

      // Update index
      const existingEntry = this.index.get(key);
      if (existingEntry) {
        this.totalSize -= existingEntry.size;
      }
      this.index.set(key, { size, order: this.orderCounter++ });
      this.totalSize += size;

      // Debounced metadata save
      this.scheduleMetadataSave();
    } catch (error) {
      console.warn(`[OPFSStore] Failed to write ${key}:`, error);
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

      const parts = key.split('/');
      let dir = this.opfsRoot;
      for (let i = 0; i < parts.length - 1; i++) {
        dir = await dir.getDirectoryHandle(parts[i]);
      }
      await dir.removeEntry(parts[parts.length - 1]);
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
        for await (const name of (this.opfsRoot as any).keys()) {
          await this.opfsRoot.removeEntry(name, { recursive: true });
        }
      } catch (error) {
        console.warn('[OPFSStore] Failed to clear:', error);
      }
    }
    this.index = new Map();
    this.totalSize = 0;
    this.orderCounter = 0;
    this.contentHash = null;
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
  getStats(): { size: number; count: number } {
    return {
      size: this.totalSize,
      count: this.index.size,
    };
  }

  /**
   * Update LRU order for a key.
   */
  touch(key: string): void {
    const entry = this.index.get(key);
    if (entry) {
      entry.order = this.orderCounter++;
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

  private async navigateToFile(key: string, create: boolean): Promise<FileSystemFileHandle> {
    const parts = key.split('/');
    let dir = this.opfsRoot!;
    for (let i = 0; i < parts.length - 1; i++) {
      dir = await dir.getDirectoryHandle(parts[i], { create });
    }
    return dir.getFileHandle(parts[parts.length - 1], { create });
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

      // Reconstruct Map from stored array
      this.index = new Map(meta.entries || []);
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
      console.warn('[OPFSStore] Failed to save metadata:', error);
    }
  }
}
