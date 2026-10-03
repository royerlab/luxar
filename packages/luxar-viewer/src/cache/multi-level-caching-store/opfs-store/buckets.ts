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
 * The tag of the content hash a chunk file is written under: 16 hex chars (two
 * 32-bit FNV-1a passes), or '' for none. Part of every chunk file's name (see
 * {@link keyToFileName}), so a file written under one hash is never read, nor
 * recovered as an unindexed file, as another hash's chunk — even when a tab
 * still at the old hash keeps writing into a directory that the republished
 * dataset's tab has already cleared.
 */
export function hashTag(hash: string | null): string {
  if (hash === null) return '';
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let i = 0; i < hash.length; i++) {
    const c = hash.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193);
    h2 = Math.imul(h2 ^ c, 0x811c9dc5) ^ (h2 >>> 13);
  }
  const hex = (n: number): string => (n >>> 0).toString(16).padStart(8, '0');
  return hex(h1) + hex(h2);
}

const HASH_TAG = /^[0-9a-f]{16}$/;

/**
 * Convert a cache key, written under the hash whose {@link hashTag} is `tag`,
 * to a filesystem-safe filename: `{tag}.{base64url(UTF-8 key)}`, or the bare
 * base64url when `tag` is '' (`.` never occurs in base64url).
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
export function keyToFileName(key: string, tag: string): string {
  const bytes = new TextEncoder().encode(key);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  const base64 = btoa(binary);
  const name = base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
  return tag === '' ? name : `${tag}.${name}`;
}

/**
 * Inverse of {@link keyToFileName}: recover the cache key and hash tag a file
 * name encodes, or `null` when the name is not one this encoding can produce (a
 * foreign or truncated file). Round-trips through `keyToFileName` so only a
 * canonical encoding is accepted. Used by the orphan reconcile to re-index chunk
 * files the persisted index never recorded.
 */
export function fileNameToKey(fileName: string): { key: string; tag: string } | null {
  const dot = fileName.indexOf('.');
  const tag = dot < 0 ? '' : fileName.slice(0, dot);
  const encoded = fileName.slice(dot + 1);
  if (tag !== '' && !HASH_TAG.test(tag)) return null;
  if (encoded.length === 0 || !/^[A-Za-z0-9_-]+$/.test(encoded)) return null;
  try {
    const base64 = encoded.replace(/-/g, '+').replace(/_/g, '/');
    const binary = atob(base64 + '='.repeat((4 - (base64.length % 4)) % 4));
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const key = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    return keyToFileName(key, tag) === fileName ? { key, tag } : null;
  } catch {
    return null;
  }
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
   * Resolve a cache key written under hash tag `tag` to its bucket file handle:
   *   `<root>/<bucket(key)>/<keyToFileName(key, tag)>`.
   */
  async navigateToFile(
    root: FileSystemDirectoryHandle,
    key: string,
    tag: string,
    create: boolean
  ): Promise<FileSystemFileHandle> {
    const bucket = getBucket(key);
    const handle = await this.getHandle(root, bucket, create);
    if (!handle) {
      throw new Error(`Cannot access bucket ${bucket}`);
    }
    const fileName = keyToFileName(key, tag);
    return handle.getFileHandle(fileName, { create });
  }
}
