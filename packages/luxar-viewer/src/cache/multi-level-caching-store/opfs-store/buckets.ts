/**
 * OPFS-bucket directory layout.
 *
 * To avoid putting tens of thousands of files into a single OPFS
 * directory (slow on every browser), keys are hashed into one of 256
 * hex-named buckets ("00".."ff"). Each bucket holds ~250 files on
 * average for a 65K-key dataset.
 *
 * The bucket-handle cache lets us avoid re-fetching the same directory
 * handle on every read/write: 256 entries max, single Map, dropped on
 * clear().
 */

/**
 * Compute the bucket index (00..ff) for a cache key via a simple
 * djb2-like rolling hash, masked to 8 bits.
 */
export function getBucket(key: string): string {
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = ((hash << 5) - hash + key.charCodeAt(i)) | 0;
  }
  return (hash & 0xff).toString(16).padStart(2, '0');
}

/**
 * Convert a cache key to a filesystem-safe filename via UTF-8 -> base64url.
 *
 * Zarr keys can include non-ASCII group/array names, so the key is
 * encoded to UTF-8 bytes before base64url conversion. The output
 * filename is safe across all browser OPFS implementations.
 *
 * Bumping OPFS_ENCODING_VERSION invalidates any directory persisted
 * with a different output; loadMetadata treats it as a cold cache.
 * (The move of every dataset directory under the `luxar/` namespace dir —
 * see `opfs-root.ts` — needed no bump: pre-namespace directories at the
 * OPFS root are simply never read again.)
 */
export function keyToFileName(key: string): string {
  const bytes = new TextEncoder().encode(key);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  const base64 = btoa(binary);
  return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

/**
 * Cached map of bucket name -> directory handle, with the
 * stale-handle recovery primitive (`invalidate`) used after a
 * concurrent clear() leaves dangling handles in memory.
 */
export class OPFSBucketCache {
  private handles = new Map<string, FileSystemDirectoryHandle>();

  /**
   * Look up or create the directory handle for `bucket`. Returns
   * `null` if the handle cannot be obtained (e.g., create=false and
   * the directory does not exist).
   */
  async getHandle(
    root: FileSystemDirectoryHandle,
    bucket: string,
    create: boolean
  ): Promise<FileSystemDirectoryHandle | null> {
    const cached = this.handles.get(bucket);
    if (cached) return cached;
    try {
      const handle = await root.getDirectoryHandle(bucket, { create });
      this.handles.set(bucket, handle);
      return handle;
    } catch {
      return null;
    }
  }

  /**
   * Drop the cached handle for `bucket`. Used after a stale-handle
   * failure (e.g., "could not be found" from OPFS) so the next access
   * re-fetches a fresh handle.
   */
  invalidate(bucket: string): void {
    this.handles.delete(bucket);
  }

  /**
   * Drop every cached handle. Used by `OPFSStore.clear()` after the
   * directory tree has been wiped.
   */
  clear(): void {
    this.handles = new Map();
  }

  /**
   * Resolve a cache key to its bucket file handle:
   *   `<root>/<bucket(key)>/<keyToFileName(key)>`.
   */
  async navigateToFile(
    root: FileSystemDirectoryHandle,
    key: string,
    create: boolean
  ): Promise<FileSystemFileHandle> {
    const bucket = getBucket(key);
    const handle = await this.getHandle(root, bucket, create);
    if (!handle) {
      throw new Error(`Cannot access bucket ${bucket}`);
    }
    const fileName = keyToFileName(key);
    return handle.getFileHandle(fileName, { create });
  }
}
