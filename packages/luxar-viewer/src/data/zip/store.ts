/**
 * Luxar's zipped-store reader: a thin, lifecycle-safe wrapper over
 * `@zarrita/storage`'s `ZipFileStore`.
 *
 * The wrapper exists for one reason — WHEN the central directory is read.
 * Stock `ZipFileStore` reads it in its CONSTRUCTOR and memoizes the resulting
 * promise unconditionally:
 *
 * ```js
 * constructor(reader, opts) { this.info = unzip(reader).then(...) }
 * ```
 *
 * Two consequences, both bad once the store sits behind a cache with a
 * dataset-switch lifecycle:
 *
 * 1. **A failure is permanent.** One transient blip while reading the ~141 kB
 *    preamble rejects `info` forever; every later `get` throws the same stale
 *    error, and the only recovery is a page reload.
 * 2. **Construction does I/O.** Building a store — which the cache layer may do
 *    while merely deciding policy — fires network requests before anyone has
 *    asked for a key.
 *
 * So: build lazily on first read, memoize **only on success**, and let a failed
 * attempt be retried by the next caller. Reads are driven by a per-store
 * `AbortController` that only `dispose()` trips, never a per-caller signal —
 * one cancelled chunk must not poison the directory every other chunk needs.
 *
 * @module data/zip/store
 */

import ZipFileStore from '@zarrita/storage/zip';
import type { AbsolutePath, AsyncReadable } from '@zarrita/storage';

import { normalizeZipEntries } from './entries';
import { LuxarHttpRangeReader } from './range-reader';

/**
 * The options `ZipFileStore` is constructed with, in one place so the store and
 * the facade cannot drift (the entry re-key is what makes a nested `zip -r`
 * archive readable at all — see {@link normalizeZipEntries}).
 */
export function createZipStoreOptions(
  url: string
): NonNullable<ConstructorParameters<typeof ZipFileStore>[1]> {
  return {
    transformEntries: (entries) => normalizeZipEntries(entries, url),
  };
}

/** Reads a zipped zarr store over HTTP range requests. */
export class LuxarZipStore implements AsyncReadable {
  #store: ZipFileStore | undefined;
  #opening: Promise<ZipFileStore> | undefined;
  #disposed = false;
  readonly #reader: LuxarHttpRangeReader;

  constructor(private readonly url: string) {
    this.#reader = new LuxarHttpRangeReader(url);
  }

  /**
   * Fresh identity of the archive, delegated to the reader so the `HEAD` it
   * costs also seeds the length `ZipFileStore` would otherwise ask for
   * separately.
   */
  probeIdentity(signal?: AbortSignal): Promise<string | null> {
    return this.#reader.probeIdentity(signal);
  }

  /**
   * Resolve the underlying store, reading the central directory on the first
   * call. Concurrent callers share one attempt; a failed attempt is discarded
   * so the next caller retries rather than inheriting the error.
   */
  async #open(): Promise<ZipFileStore> {
    if (this.#store) return this.#store;
    if (this.#disposed) throw new Error(`Zipped store already disposed: ${this.url}`);

    this.#opening ??= (async () => {
      const store = new ZipFileStore(this.#reader, createZipStoreOptions(this.url));
      // Retain the ranges the DIRECTORY read touches, and only those: unzipit
      // reads a fixed 65,557-byte tail and then re-reads the central directory
      // that mostly sits inside it. Member payloads are cached a layer up, by
      // chunk key, so retaining them here would only double the memory.
      this.#reader.retainReads(true);
      try {
        // Force the directory read HERE so a failure surfaces as this promise
        // rejecting — and so the memoize-on-success below is meaningful.
        await store.has('/zarr.json' as AbsolutePath);
      } finally {
        this.#reader.retainReads(false);
      }
      return store;
    })()
      .then((store) => {
        // Only a SUCCESSFUL open is remembered.
        if (!this.#disposed) this.#store = store;
        return store;
      })
      .catch((error: unknown) => {
        this.#opening = undefined;
        throw error;
      });

    return this.#opening;
  }

  async get(key: string): Promise<Uint8Array | undefined> {
    return (await this.#open()).get(key as AbsolutePath);
  }

  async has(key: string): Promise<boolean> {
    return (await this.#open()).has(key as AbsolutePath);
  }

  /**
   * Drop the archive. The directory is not reusable afterwards; a disposed
   * store refuses to re-open rather than quietly starting a fresh download for
   * a dataset the caller has already navigated away from.
   */
  dispose(): void {
    this.#disposed = true;
    this.#store = undefined;
    this.#opening = undefined;
  }
}
