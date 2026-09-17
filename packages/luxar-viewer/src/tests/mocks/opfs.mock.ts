/**
 * OPFS (Origin Private File System) Mock
 *
 * Mocks the browser's Origin Private File System API used by the caching
 * layer. Two flavours:
 *
 * - {@link installOPFSMock} — the GLOBAL default (installed by `setup.ts`):
 *   `navigator.storage.getDirectory()` rejects, so code under test exercises
 *   its OPFS-unavailable fallback.
 * - {@link createFakeOpfsRoot} — an in-memory OPFS root for the cache tests
 *   that need a WORKING L2. It models the directory levels production
 *   actually walks: the origin root → the viewer's `luxar/` namespace dir
 *   (`opfs-store/opfs-root.ts`) → a `zarr-cache-<hash>` dataset dir.
 */

import { vi } from 'vitest';
import { OPFS_NAMESPACE_DIR } from '../../cache/multi-level-caching-store/opfs-store/opfs-root';

/**
 * Install OPFS mock
 *
 * Configures navigator.storage to reject with appropriate error
 * This allows cache tests to verify fallback behavior
 */
export function installOPFSMock(): void {
  (globalThis as any).navigator.storage = {
    getDirectory: vi.fn().mockRejectedValue(new Error('OPFS not available in test environment')),
    estimate: vi.fn().mockResolvedValue({ quota: 0, usage: 0 }),
  };
}

/** The flat in-memory dataset directory {@link createFakeOpfsRoot} builds by default. */
export interface FakeOpfsDatasetDir {
  kind: 'directory';
  name: string;
  /** `(name, { create? })` — loosely typed so tests can wrap it with `(...args)`. */
  getFileHandle: (...args: any[]) => Promise<any>;
  /** `(bucket, { create? })` — every bucket resolves to the dataset dir itself. */
  getDirectoryHandle: (...args: any[]) => Promise<any>;
  /** `(name, { recursive? })` — `NotFoundError` for an absent file. */
  removeEntry: (...args: any[]) => Promise<void>;
  keys: () => AsyncGenerator<string>;
  entries: () => AsyncGenerator<[string, any]>;
  /** Wipe every file (what removing the dataset directory does). */
  clear: () => void;
}

/** What {@link createFakeOpfsRoot} returns. */
export interface FakeOpfsRoot {
  /** The origin's OPFS root — accepts ONLY the `luxar/` namespace dir. */
  root: FileSystemDirectoryHandle & { getDirectoryHandle: ReturnType<typeof vi.fn> };
  /** The `luxar/` namespace directory (dataset dirs are children of this). */
  luxarDir: FileSystemDirectoryHandle & {
    getDirectoryHandle: ReturnType<typeof vi.fn>;
    removeEntry: ReturnType<typeof vi.fn>;
  };
  /** The dataset directory every `zarr-cache-*` id resolves to (see notes). */
  datasetDir: FakeOpfsDatasetDir;
  /** Chunk bytes by file name (bucket dirs collapse, names are unique per key). */
  files: Map<string, Uint8Array>;
  /** String files (`_cache_meta.json`, the write probe) by file name. */
  metaFiles: Map<string, string>;
  /** Every dataset id requested from the namespace dir, in call order. */
  datasetIds: string[];
  /** `navigator.storage` replacement (`getDirectory` resolves to `root`). */
  storage: { getDirectory: () => Promise<FileSystemDirectoryHandle>; estimate: () => Promise<any> };
  /**
   * `vi.stubGlobal('navigator', { storage })`. Overrides swap the dataset dir
   * (a test's instrumented handle) and/or the quota estimate while keeping
   * the same root → `luxar/` → dataset chain.
   */
  install(overrides?: { datasetDir?: unknown; estimate?: () => Promise<any> }): FakeOpfsRoot;
}

/** Options for {@link createFakeOpfsRoot}. */
export interface FakeOpfsRootOptions {
  /**
   * A test-built dataset-level handle to hand out instead of the default flat
   * one. Every dataset id resolves to it.
   */
  datasetDir?: unknown;
  /**
   * What removing a dataset directory does (`luxarDir.removeEntry(id, …)`).
   * Defaults to wiping the built-in `files` / `metaFiles`; a test supplying
   * its own `datasetDir` passes the wipe of ITS maps here.
   */
  onRemoveDataset?: (datasetId: string) => void;
  /** `navigator.storage.estimate()` replacement (default: 10 GB quota, 1 GB used). */
  estimate?: () => Promise<any>;
}

/**
 * Build an in-memory OPFS root for cache tests.
 *
 * Models exactly the levels production walks, and nothing looser:
 *
 * - `root.getDirectoryHandle(name)` accepts only {@link OPFS_NAMESPACE_DIR}.
 *   Anything else throws — a store that reaches for a dataset directory at
 *   the origin root (the pre-namespace layout) fails every test that mounts
 *   it, not just the one asserting the layout. `create: false` before the
 *   namespace dir exists throws the platform `NotFoundError`, so
 *   `OPFSStore.listAll` on a cold origin is testable.
 * - `luxarDir.getDirectoryHandle(id, …)` records `id` in `datasetIds` and
 *   returns {@link FakeOpfsRoot.datasetDir}; `luxarDir.removeEntry(id, …)`
 *   wipes it; `luxarDir.entries()` yields one `[id, datasetDir]` per
 *   recorded id (what `listAll` iterates).
 * - The dataset dir is FLAT: every bucket `getDirectoryHandle` returns the
 *   dataset dir itself and files are keyed by name alone. Bucket names carry
 *   no information the tests need (a key's base64url file name is unique on
 *   its own), and the flat shape is what lets tests monkeypatch ONE handle's
 *   `getFileHandle` / `removeEntry` to inject faults at any depth.
 *
 * Both `datasetDir` and every handle it returns are plain objects with
 * own-property methods, so tests may spread or reassign them freely.
 */
export function createFakeOpfsRoot(options: FakeOpfsRootOptions = {}): FakeOpfsRoot {
  const files = new Map<string, Uint8Array>();
  const metaFiles = new Map<string, string>();
  const datasetIds: string[] = [];

  const fileHandle = (path: string) => ({
    kind: 'file' as const,
    name: path,
    async getFile() {
      const data = files.get(path) ?? new Uint8Array(0);
      return {
        size: data.byteLength,
        async arrayBuffer() {
          return data.buffer;
        },
        async text() {
          return metaFiles.get(path) ?? '{}';
        },
      };
    },
    async createWritable() {
      return {
        async write(data: ArrayBuffer | ArrayBufferView | string) {
          if (typeof data === 'string') {
            metaFiles.set(path, data);
          } else if (data instanceof ArrayBuffer) {
            files.set(path, new Uint8Array(data));
          } else {
            files.set(
              path,
              new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))
            );
          }
        },
        async close() {},
        async abort() {},
      };
    },
  });

  const defaultDatasetDir: FakeOpfsDatasetDir = {
    kind: 'directory',
    name: 'dataset',
    async getFileHandle(name: string, opts?: { create?: boolean }) {
      if (!files.has(name) && !metaFiles.has(name) && !opts?.create) {
        throw new DOMException(`Entry not found: ${name}`, 'NotFoundError');
      }
      return fileHandle(name);
    },
    async getDirectoryHandle() {
      return defaultDatasetDir; // all buckets collapse to the dataset dir
    },
    async removeEntry(name: string) {
      if (!files.has(name) && !metaFiles.has(name)) {
        throw new DOMException(`Entry not found: ${name}`, 'NotFoundError');
      }
      files.delete(name);
      metaFiles.delete(name);
    },
    async *keys() {
      // Flat fake: bucket directories are not enumerable as children.
    },
    async *entries() {},
    clear() {
      files.clear();
      metaFiles.clear();
    },
  };

  const current: { datasetDir: unknown; estimate: () => Promise<any> } = {
    datasetDir: options.datasetDir ?? defaultDatasetDir,
    estimate: options.estimate ?? (async () => ({ quota: 10e9, usage: 1e9 })),
  };
  const onRemoveDataset =
    options.onRemoveDataset ??
    (() => {
      files.clear();
      metaFiles.clear();
    });

  const luxarDir = {
    kind: 'directory' as const,
    name: OPFS_NAMESPACE_DIR,
    getDirectoryHandle: vi.fn(async (id: string, _opts?: { create?: boolean }) => {
      if (!datasetIds.includes(id)) datasetIds.push(id);
      return current.datasetDir;
    }),
    removeEntry: vi.fn(async (id: string, _opts?: { recursive?: boolean }) => {
      onRemoveDataset(id);
    }),
    async *entries(): AsyncGenerator<[string, unknown]> {
      // `listAll` skips a child whose `kind` is not 'directory'; a custom
      // datasetDir that wants to be listed must carry `kind: 'directory'`.
      for (const id of datasetIds) yield [id, current.datasetDir];
    },
    async *keys() {
      yield* datasetIds;
    },
  };

  let namespaceCreated = false;
  const root = {
    kind: 'directory' as const,
    name: '',
    getDirectoryHandle: vi.fn(async (name: string, opts?: { create?: boolean }) => {
      if (name !== OPFS_NAMESPACE_DIR) {
        throw new TypeError(
          `fake OPFS root: viewer entries must live under "${OPFS_NAMESPACE_DIR}/", got "${name}"`
        );
      }
      if (!namespaceCreated && !opts?.create) {
        throw new DOMException(`Entry not found: ${name}`, 'NotFoundError');
      }
      namespaceCreated = true;
      return luxarDir;
    }),
    async *entries(): AsyncGenerator<[string, unknown]> {
      if (namespaceCreated) yield [OPFS_NAMESPACE_DIR, luxarDir];
    },
    async *keys() {
      if (namespaceCreated) yield OPFS_NAMESPACE_DIR;
    },
  };

  const storage = {
    getDirectory: async () => root as unknown as FileSystemDirectoryHandle,
    estimate: () => current.estimate(),
  };

  const fake: FakeOpfsRoot = {
    root: root as unknown as FakeOpfsRoot['root'],
    luxarDir: luxarDir as unknown as FakeOpfsRoot['luxarDir'],
    datasetDir: defaultDatasetDir,
    files,
    metaFiles,
    datasetIds,
    storage,
    install(overrides = {}) {
      if (overrides.datasetDir !== undefined) current.datasetDir = overrides.datasetDir;
      if (overrides.estimate) current.estimate = overrides.estimate;
      vi.stubGlobal('navigator', { storage });
      return fake;
    },
  };
  return fake;
}
