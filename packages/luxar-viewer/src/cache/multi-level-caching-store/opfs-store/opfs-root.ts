/**
 * The viewer's OPFS namespace directory.
 *
 * Every OPFS directory the viewer persists lives under ONE top-level folder,
 * `luxar/`, rather than at the origin's OPFS root. An embedder's own OPFS
 * usage (or another library on the same origin) never collides with the
 * `zarr-cache-<hash>` dataset directories, and a host can wipe the viewer's
 * footprint with a single `removeEntry('luxar', { recursive: true })`.
 *
 * The dataset-directory prefix (`zarr-cache-`) and `OPFS_ENCODING_VERSION`
 * are unchanged: moving under `luxar/` is itself the invalidation — a
 * pre-namespace `zarr-cache-*` directory at the OPFS root is simply never
 * read again.
 *
 * @module cache/multi-level-caching-store/opfs-store/opfs-root
 */

/** Name of the top-level OPFS directory that holds every viewer-owned entry. */
export const OPFS_NAMESPACE_DIR = 'luxar';

/**
 * Resolve the `luxar/` namespace directory under the origin's OPFS root.
 *
 * With `create: true` the directory is created on demand (the store's mount
 * path). With `create: false` a missing namespace directory — a cold origin
 * that has never run the viewer — surfaces as the platform `NotFoundError`
 * `DOMException`, which read-only callers such as `OPFSStore.listAll` treat as
 * "nothing cached".
 *
 * Rejects when OPFS itself is unavailable (`navigator.storage.getDirectory`
 * absent or rejecting), exactly like the raw root call it wraps.
 */
export async function getLuxarOpfsRoot(options: {
  create: boolean;
}): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(OPFS_NAMESPACE_DIR, { create: options.create });
}
