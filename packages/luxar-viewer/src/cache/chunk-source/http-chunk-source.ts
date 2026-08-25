/**
 * The directory-store byte source: one HTTP request per chunk.
 *
 * This is the behaviour {@link MultiLevelCachingStore} had inline before the
 * {@link ChunkSource} seam existed, moved verbatim rather than rewritten —
 * same `buildUrl`, same `fetchWithRetry`, same retry budget, same
 * body-cancellation on a response we decline to read. The store's ~100
 * existing tests drive it through `global.fetch` and none of them changed.
 *
 * The one thing that genuinely moved is ownership of the `Response` lifetime.
 * The store used to hold a `FetchResponseScope` and `dispose()` it in a
 * `finally`; now that lives here, because a non-HTTP source has no response to
 * dispose and the seam must not mention one.
 *
 * @module cache/chunk-source/http-chunk-source
 */

import type { ChunkFetchOutcome, ChunkSource } from '../chunk-source';
import { buildUrl, fetchWithRetry } from '../multi-level-caching-store/fetch-retry';
import {
  getRemoteContentHash,
  type RemoteValidationToken,
} from '../multi-level-caching-store/validation-queue';

/** Reads chunks from a directory-backed zarr store over HTTP. */
export class HttpChunkSource implements ChunkSource {
  constructor(private readonly baseUrl: string) {}

  /**
   * The dataset URL verbatim.
   *
   * Deliberately unchanged from what `hashUrl(baseUrl)` hashed before this
   * refactor, so existing OPFS buckets keep resolving — normalizing it here
   * would silently rotate every user's on-disk cache for no gain.
   */
  get identity(): string {
    return this.baseUrl;
  }

  get describe(): string {
    return this.baseUrl;
  }

  async get(key: string, signal?: AbortSignal): Promise<ChunkFetchOutcome> {
    let scope;
    try {
      scope = await fetchWithRetry(buildUrl(this.baseUrl, key), { signal });
    } catch (error) {
      return { kind: 'error', cause: error instanceof Error ? error : new Error(String(error)) };
    }

    if (!scope) {
      // `fetchWithRetry` returns undefined when it gave up: either the caller
      // aborted, or the retry budget ran out. The store distinguishes those by
      // inspecting its own signals, so report the retry case and let it
      // re-classify an abort it knows about.
      return signal?.aborted
        ? { kind: 'aborted' }
        : { kind: 'error', cause: new Error(`fetch exhausted retries for ${key}`) };
    }

    try {
      if (!scope.response.ok) return { kind: 'missing' };
      const data = new Uint8Array(await scope.response.arrayBuffer());
      // Over HTTP the decoded body IS what the meter counted before this
      // refactor; keeping them equal preserves the existing byte totals.
      return { kind: 'ok', data, bytesOverWire: data.byteLength };
    } finally {
      scope.dispose();
    }
  }

  probeIdentityToken(options: {
    signal?: AbortSignal;
    timeoutMsOverride?: number;
  }): Promise<RemoteValidationToken | null> {
    return getRemoteContentHash(this.baseUrl, options);
  }

  dispose(): void {
    // Nothing held open: each `get` disposes its own response scope.
  }
}
