/**
 * The byte source behind `MultiLevelCachingStore`.
 *
 * The caching store used to hold a `baseUrl: string` and build every chunk URL
 * itself (`buildUrl(baseUrl, key)`), which quietly hard-wired one assumption
 * into the whole cache stack: that a chunk is addressable as a URL. That is
 * true for a directory store and false for a zipped one, where a chunk is a
 * member INSIDE a single archive and can only be reached by asking the archive
 * for it. Same for a future reference/kerchunk store.
 *
 * So the store now takes a `ChunkSource`: "give me the bytes for this key, and
 * tell me whether the thing you are reading still has the identity I cached".
 * Everything above it — L1, L2/OPFS, the segmented LRU, the prefetcher — keeps
 * keying on whole objects and is untouched.
 *
 * Two deliberate shapes:
 *
 * - **The source returns materialized bytes, not a `Response`.** The old L3
 *   block held a `FetchResponseScope` and had to `dispose()` it in a `finally`
 *   to cancel bodies it never read. A zip member has no `Response` at all, so
 *   that could not survive as a shared contract. Ownership of the response
 *   lifetime moves INTO the HTTP source, where it belongs.
 * - **`bytesOverWire` is separate from `data.byteLength`.** They are equal for
 *   plain HTTP, but a compressed archive member transfers fewer bytes than it
 *   yields. Collapsing the two would make the bandwidth meter over-report a
 *   DEFLATE archive by its compression ratio.
 *
 * @module cache/chunk-source
 */

import type { RemoteValidationToken } from './multi-level-caching-store/validation-queue';

/**
 * Result of asking a source for one key.
 *
 * Maps 1:1 onto the `Result<Uint8Array, CacheError>` the store already returns,
 * so the L3 branch stays a plain switch and no new error taxonomy appears.
 */
export type ChunkFetchOutcome =
  /** Bytes in hand. `bytesOverWire` is what the network actually moved. */
  | { kind: 'ok'; data: Uint8Array; bytesOverWire: number }
  /** The source answered, and this key is not there (an HTTP 404, say). */
  | { kind: 'missing' }
  /** A caller signal, a store disposal, or an invalidation cancelled the read. */
  | { kind: 'aborted' }
  /** Transient failure after the source exhausted its own retries. */
  | { kind: 'error'; cause: Error };

/** Where `MultiLevelCachingStore` gets its bytes. */
export interface ChunkSource {
  /**
   * Stable identity of the thing being read, used to derive the OPFS bucket.
   *
   * NEVER normalized: `scene.luxar.zarr.zip` and `scene.luxar.zarr/` are
   * different sources with different key namespaces, and collapsing them into
   * one bucket would let an archive read a directory's cached members (and
   * vice versa) with no failure to notice.
   */
  readonly identity: string;

  /** Human-readable label for logs and the OPFS index metadata. */
  readonly describe: string;

  /** Fetch one key. Must not throw: failures come back as an outcome. */
  get(key: string, signal?: AbortSignal): Promise<ChunkFetchOutcome>;

  /**
   * Re-read the source's identity token, bypassing every cache tier, so the
   * store can tell "same dataset" from "something else now serves this
   * address". `null` means the source could not answer — which is not evidence
   * of a change (see the TTL handling in `validateCache`).
   */
  probeIdentityToken(options: {
    signal?: AbortSignal;
    timeoutMsOverride?: number;
  }): Promise<RemoteValidationToken | null>;

  /** Release anything the source holds open. Called from the store's dispose. */
  dispose(): void;
}
