/**
 * The zipped-store byte source: chunks read out of one archive.
 *
 * This is what lets `MultiLevelCachingStore` cache a `.zarr.zip` at all. The
 * store used to build a URL per chunk, which an archive member does not have;
 * with the {@link ChunkSource} seam it just asks this for bytes, and L1/L2 keep
 * keying on whole chunks exactly as they do for a directory store.
 *
 * Caching matters more here than for a directory store, not less. Every
 * member read is a `Range` request against the SAME URL, so a repeat read cannot
 * be served from the browser's HTTP cache the way a repeat GET of a per-chunk
 * URL can — the reader deliberately sends `cache: 'no-store'`, because Chrome
 * serialises same-URL range GETs through its HTTP-cache writer lock otherwise.
 * The chunk cache is what makes those repeats free.
 *
 * Each member costs ONE ranged GET: `LuxarZipStore` sizes a window from the
 * central directory (local header + name + extra field + slack + compressed
 * payload) and answers both of `unzipit`'s serial reads — the 30-byte local
 * header, then the payload — from it. (A local extra field larger than the slack
 * costs one more GET for that member.) Opening the archive is ONE suffix GET too,
 * shared with the identity probe when that runs first.
 *
 * @module cache/chunk-source/zip-chunk-source
 */

import { ArchiveFaultError } from '../chunk-source';
import type {
  ArchiveByteReader,
  ChunkFetchOutcome,
  ChunkSource,
  ChunkSourceGetOptions,
} from '../chunk-source';
import type { RemoteValidationToken } from '../multi-level-caching-store/validation-queue';

/** Reads chunks out of a zipped zarr store. */
export class ZipChunkSource implements ChunkSource {
  /**
   * @param archiveUrl - Used for identity and the HEAD identity probe.
   * @param reader - The archive itself, injected (see {@link ArchiveByteReader}).
   */
  constructor(
    private readonly archiveUrl: string,
    private readonly reader: ArchiveByteReader
  ) {}

  /**
   * The archive URL VERBATIM — never folded onto its unzipped twin.
   *
   * `scene.luxar.zarr.zip` and `scene.luxar.zarr/` cache the same decoded chunk
   * bytes but have different key namespaces (an archive has no
   * `overlays/x/y.png`, and a `has` miss means something different in each). A
   * shared OPFS bucket would let one serve the other's members with nothing
   * failing loudly.
   */
  get identity(): string {
    return this.archiveUrl;
  }

  get describe(): string {
    return this.archiveUrl;
  }

  async get(
    key: string,
    signal?: AbortSignal,
    options?: ChunkSourceGetOptions
  ): Promise<ChunkFetchOutcome> {
    if (signal?.aborted) return { kind: 'aborted' };
    try {
      const data = await this.reader.get(key, signal, options);
      if (signal?.aborted) return { kind: 'aborted' };
      // A key absent from the central directory is a plain miss — and, unlike
      // the directory store's 404, it costs no request at all.
      if (!data) return { kind: 'missing' };
      // `bytesOverWire` is the DECOMPRESSED length here, which over-reports a
      // DEFLATE archive: `unzipit` inflates internally and never tells us the
      // compressed size it actually moved. Honest alternative would be
      // threading the figure out of the reader; noted rather than faked.
      return { kind: 'ok', data, bytesOverWire: data.byteLength };
    } catch (error) {
      if (signal?.aborted) return { kind: 'aborted' };
      const cause = error instanceof Error ? error : new Error(String(error));
      // An archive-level fault (no Range support, archive missing, access
      // denied) is not a per-key miss: report it as fatal so the store rethrows
      // and the crafted remedy reaches the user instead of an empty scene.
      if (cause instanceof ArchiveFaultError) return { kind: 'fatal', cause };
      return { kind: 'error', cause };
    }
  }

  /**
   * Identity of the archive, delegated to the container.
   *
   * A directory store re-reads its root `zarr.json` to answer "is this still
   * the scene I cached?". An archive cannot: its root attrs live inside the
   * file, and the reader's central directory is a snapshot — a replaced archive
   * would keep answering from the old offsets. One probe on the archive covers
   * the WHOLE store instead, which is both cheaper and a stronger guarantee
   * than a single document's hash.
   *
   * The reader owns the request so the one suffix GET it makes also seeds the
   * archive length and retains the tail that opening reads next. Requires the
   * headers to be readable cross-origin; `luxar serve` exposes them.
   */
  async probeIdentityToken(options: {
    signal?: AbortSignal;
    timeoutMsOverride?: number;
  }): Promise<RemoteValidationToken | null> {
    // The budget matters: this probe runs inside `init()` → `validateCache`,
    // in front of the first paint, and a hanging server would otherwise
    // block the load with no fail-fast at all.
    const hash = await this.reader.probeIdentity?.(options.signal, options.timeoutMsOverride);
    return hash ? { hash, mode: 'archive-etag' } : null;
  }

  dispose(): void {
    this.reader.dispose();
  }
}
