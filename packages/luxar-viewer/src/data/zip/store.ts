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
import type { AbsolutePath, AsyncReadable, GetOptions } from '@zarrita/storage';

import { ArchiveFaultError } from '../../cache/chunk-source';
import type { ChunkSourceGetOptions } from '../../cache/chunk-source';
import { normalizeZipEntries } from './entries';
import { LuxarHttpRangeReader } from './range-reader';

/**
 * Bytes of slack added to a member's one-GET window beyond what the CENTRAL
 * directory says its local header needs.
 *
 * The local header repeats the name but may carry a DIFFERENT extra field:
 * Info-ZIP's extended timestamp is 9 bytes centrally and 13 locally, ZIP64
 * adds a 20-byte local block. 64 covers the common writers; a larger local
 * extra field costs one extra payload GET, never a wrong read.
 */
export const LOCAL_HEADER_SLACK_BYTES = 64;

/** Size of a zip local file header before its name and extra field. */
const LOCAL_HEADER_FIXED_BYTES = 30;

/** The central-directory fields a one-GET member window is sized from. */
interface RawZipEntry {
  relativeOffsetOfLocalHeader: number;
  fileNameLength: number;
  extraFieldLength: number;
  compressedSize: number;
  generalPurposeBitFlag: number;
}

/**
 * `unzipit`'s parsed central-directory record for an entry, if its private
 * `_rawEntry` still has the shape we rely on (`@zarrita/storage`'s own
 * `getRange` depends on the same field). Anything else returns `undefined` and
 * the read falls back to `unzipit`'s own two-request path.
 */
function rawZipEntry(entry: unknown): RawZipEntry | undefined {
  const raw = (entry as { _rawEntry?: Record<string, unknown> } | null)?._rawEntry;
  if (!raw || typeof raw !== 'object') return undefined;
  const fields = [
    'relativeOffsetOfLocalHeader',
    'fileNameLength',
    'extraFieldLength',
    'compressedSize',
    'generalPurposeBitFlag',
  ] as const;
  for (const field of fields) {
    if (typeof raw[field] !== 'number') return undefined;
  }
  return raw as unknown as RawZipEntry;
}

/** One ranged window covering a member's local header AND its payload. */
function memberWindow(entry: unknown): { offset: number; size: number } | undefined {
  const raw = rawZipEntry(entry);
  // Encrypted members are refused by `unzipit` before it reads a byte.
  if (!raw || raw.generalPurposeBitFlag & 0x1) return undefined;
  return {
    offset: raw.relativeOffsetOfLocalHeader,
    size:
      LOCAL_HEADER_FIXED_BYTES +
      raw.fileNameLength +
      raw.extraFieldLength +
      LOCAL_HEADER_SLACK_BYTES +
      raw.compressedSize,
  };
}

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
  /** The normalized entry table, captured as `ZipFileStore` builds it. */
  #entries: Record<string, unknown> | undefined;
  #disposed = false;
  readonly #reader: LuxarHttpRangeReader;
  readonly #abort = new AbortController();

  constructor(private readonly url: string) {
    this.#reader = new LuxarHttpRangeReader(url, this.#abort.signal);
  }

  /**
   * Fresh identity of the archive, delegated to the reader so the `HEAD` it
   * costs also seeds the length `ZipFileStore` would otherwise ask for
   * separately.
   */
  probeIdentity(signal?: AbortSignal, timeoutMs?: number): Promise<string | null> {
    // BOTH arguments, deliberately. Declaring only `signal` still satisfies the
    // port structurally — an optional two-arg method accepts a one-arg
    // implementation — so dropping the budget here was invisible to the type
    // checker AND to tests that drove the reader or a full-arity fake. The
    // budget is what keeps an unresponsive host from hanging the first paint,
    // since this probe runs inside `init()` → `validateCache`.
    return this.#reader.probeIdentity(signal, timeoutMs);
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
      // ONE suffix GET for length + tail (a no-op when the identity probe
      // already made it), so the directory read below touches the network only
      // for a central directory larger than the tail window.
      await this.#reader.primeTail();
      const options = createZipStoreOptions(this.url);
      const normalize = options.transformEntries;
      const store = new ZipFileStore(this.#reader, {
        ...options,
        transformEntries: (entries) => {
          const normalized = normalize ? normalize(entries) : entries;
          this.#entries = normalized;
          return normalized;
        },
      });
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
        if (error instanceof ArchiveFaultError) throw error;
        const cause = error instanceof Error ? error : new Error(String(error));
        throw new ArchiveFaultError(cause.message, this.url, { cause });
      });

    return this.#opening;
  }

  /**
   * Read one member with ONE ranged GET (header + payload, sized from the
   * central directory) instead of `unzipit`'s two serial ones.
   *
   * @param _signal - Advisory only (see `ArchiveByteReader.get`); zarrita, which
   *   also reads this store directly, passes its `GetOptions` here instead.
   * @param options - `priority` classes the member's window fetch in the gate.
   */
  async get(
    key: string,
    _signal?: AbortSignal | GetOptions,
    options?: ChunkSourceGetOptions
  ): Promise<Uint8Array | undefined> {
    const store = await this.#open();
    // `@zarrita/storage`'s `stripPrefix` is literally `key.slice(1)`.
    const window = memberWindow(this.#entries?.[key.slice(1)]);
    if (!window) return store.get(key as AbsolutePath);
    return this.#reader.readMember(
      window.offset,
      window.size,
      () => store.get(key as AbsolutePath),
      options?.priority
    );
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
    // Actually cancel what is in flight. Relabelling an outcome while the bytes
    // keep arriving is not cancellation.
    this.#abort.abort();
    this.#store = undefined;
    this.#opening = undefined;
    this.#entries = undefined;
  }
}
