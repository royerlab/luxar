/**
 * Image Label Loader — Lazy per-element image fetching from zarr.
 *
 * Image labels are stored per node as two zarr arrays:
 *   - image_label_offsets (uint64, N+1): byte offset of each image in image_label_bytes
 *   - image_label_bytes (uint8, no compression): concatenated encoded image blobs
 *
 * Image i = image_label_bytes[offsets[i] : offsets[i+1]], as raw JPEG/WebP/PNG bytes.
 * Empty entries (offsets[i] === offsets[i+1]) return null.
 *
 * Image data can be hundreds of MB, so image bytes are fetched per-element on
 * demand using zarr slice access. Unlike LabelLoader, this loader still
 * bulk-loads its offsets because they are small relative to the image payload.
 *
 * Decoded images are cached as blob URLs in an LRU cache with automatic
 * URL.revokeObjectURL on eviction.
 */

import * as zarr from '../../zarr';
import { readArray, slice } from '../../zarr';
import { LRUCache } from '../../../cache/lru-cache';
import { log, Modules } from '../../../utils/log';
import { detectMimeType } from '../../../utils/image-mime';

/** Cached image entry with blob URL and size for LRU tracking. */
interface CachedImage {
  blobUrl: string;
  size: number;
}

/** Default LRU cache size: 50 MB of decoded image blob URLs. */
const DEFAULT_MAX_CACHE_BYTES = 50 * 1024 * 1024;

export class ImageLabelLoader {
  /** Cached offsets per node path (loaded eagerly once per node). */
  private offsetsCache = new Map<string, BigUint64Array>();

  /** In-flight offsets loading promises for request coalescing. */
  private offsetsInflight = new Map<string, Promise<BigUint64Array>>();

  /** Cached zarr array handles per node (opened once). */
  private bytesArrayCache = new Map<string, zarr.Array<zarr.DataType>>();

  /** Per-element image cache with LRU eviction (revokes blob URLs on eviction). */
  private imageCache: LRUCache<CachedImage>;

  /** In-flight image requests for deduplication. */
  private imageInflight = new Map<string, Promise<string | null>>();

  constructor(
    _store: zarr.Readable,
    private rootLoc: zarr.Location<zarr.Readable>,
    maxCacheBytes: number = DEFAULT_MAX_CACHE_BYTES
  ) {
    this.imageCache = new LRUCache<CachedImage>(
      maxCacheBytes,
      (entry) => entry.size,
      (_key, entry) => {
        // Revoke blob URL when evicted from cache
        URL.revokeObjectURL(entry.blobUrl);
      }
    );
  }

  /**
   * Get a blob URL for a single element's image label.
   * Triggers lazy load of offsets on first access per node.
   * Returns null if: no image exists, or index is out of range.
   */
  async getImageUrl(nodePath: string, elementIndex: number): Promise<string | null> {
    const cacheKey = `${nodePath}:${elementIndex}`;

    // Check LRU cache
    const cached = this.imageCache.get(cacheKey);
    if (cached) return cached.blobUrl;

    // Check in-flight (request coalescing)
    const existing = this.imageInflight.get(cacheKey);
    if (existing) return existing;

    const promise = this.doFetchImage(nodePath, elementIndex, cacheKey);
    this.imageInflight.set(cacheKey, promise);

    try {
      return await promise;
    } finally {
      this.imageInflight.delete(cacheKey);
    }
  }

  /**
   * Check if a node has image labels based on its cached .zattrs metadata.
   */
  hasImageLabels(nodeAttrs: Record<string, unknown>): boolean {
    return nodeAttrs?.has_image_labels === true;
  }

  /** Clean up all caches and revoke all blob URLs. */
  dispose(): void {
    this.imageCache.clear(); // onEvict revokes URLs
    this.offsetsCache.clear();
    this.offsetsInflight.clear();
    this.bytesArrayCache.clear();
    this.imageInflight.clear();
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  /**
   * Load all offsets for a node from zarr.
   * Offsets are small (~N*8 bytes) so bulk-loading is fine.
   */
  private async loadOffsets(nodePath: string): Promise<BigUint64Array> {
    const cached = this.offsetsCache.get(nodePath);
    if (cached) return cached;

    const existing = this.offsetsInflight.get(nodePath);
    if (existing) return existing;

    const promise = this.doLoadOffsets(nodePath);
    this.offsetsInflight.set(nodePath, promise);

    try {
      const result = await promise;
      this.offsetsCache.set(nodePath, result);
      return result;
    } finally {
      this.offsetsInflight.delete(nodePath);
    }
  }

  private async doLoadOffsets(nodePath: string): Promise<BigUint64Array> {
    const cleanPath = nodePath.startsWith('/') ? nodePath.slice(1) : nodePath;
    const offsetsLoc = this.rootLoc.resolve(`${cleanPath}/image_label_offsets`);
    const offsetsArr = await zarr.open(offsetsLoc, { kind: 'array' });
    const offsetsData = await readArray(offsetsArr);
    return offsetsData.data as BigUint64Array;
  }

  /**
   * Open (or return cached) the image_label_bytes zarr array for a node.
   */
  private async openBytesArray(nodePath: string): Promise<zarr.Array<zarr.DataType>> {
    const cached = this.bytesArrayCache.get(nodePath);
    if (cached) return cached;

    const cleanPath = nodePath.startsWith('/') ? nodePath.slice(1) : nodePath;
    const bytesLoc = this.rootLoc.resolve(`${cleanPath}/image_label_bytes`);
    const arr = await zarr.open(bytesLoc, { kind: 'array' });
    this.bytesArrayCache.set(nodePath, arr);
    return arr;
  }

  /**
   * Fetch a single element's image bytes, create blob URL, cache it.
   */
  private async doFetchImage(
    nodePath: string,
    elementIndex: number,
    cacheKey: string
  ): Promise<string | null> {
    try {
      const offsets = await this.loadOffsets(nodePath);

      if (elementIndex < 0 || elementIndex >= offsets.length - 1) return null;

      const start = Number(offsets[elementIndex]);
      const end = Number(offsets[elementIndex + 1]);

      // Empty entry — no image for this element
      if (start === end) return null;

      const bytesArr = await this.openBytesArray(nodePath);

      // Fetch only the byte range for this image (zarr fetches overlapping chunks)
      const imageData = await readArray(bytesArr, [slice(start, end)]);
      const imageBytes = imageData.data as Uint8Array;

      // Detect MIME type from magic bytes
      const mimeType = detectMimeType(imageBytes);

      // Create blob URL (use new Uint8Array copy to get a plain ArrayBuffer for Blob)
      const copy = new Uint8Array(imageBytes);
      const blob = new Blob([copy], { type: mimeType });
      const blobUrl = URL.createObjectURL(blob);

      // Cache with size tracking
      this.imageCache.set(cacheKey, { blobUrl, size: imageBytes.byteLength });

      return blobUrl;
    } catch (error) {
      log.warning(
        Modules.SCENE_LOADER,
        `Failed to load image label for ${nodePath}[${elementIndex}]: ${
          error instanceof Error ? error.message : error
        }`
      );
      return null;
    }
  }
}
