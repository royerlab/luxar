/**
 * The directory-store byte source: one HTTP request per chunk.
 *
 * This is the behaviour `MultiLevelCachingStore` had inline before the
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
    let exhaustedCause: unknown;
    try {
      const outcome = await fetchWithRetry(
        buildUrl(this.baseUrl, key),
        { signal, onExhausted: (error) => (exhaustedCause = error) },
        async (attempt) => {
          if (!attempt.response.ok) return { kind: 'missing' } as const;
          if (signal?.aborted) return { kind: 'aborted' } as const;
          const data = await attempt.readBody();
          return { kind: 'ok', data, bytesOverWire: data.byteLength } as const;
        }
      );
      return (
        outcome ??
        (signal?.aborted
          ? { kind: 'aborted' }
          : {
              kind: 'error',
              cause: new Error(
                `fetch exhausted retries for ${key}${
                  exhaustedCause instanceof Error ? `: ${exhaustedCause.message}` : ''
                }`,
                { cause: exhaustedCause }
              ),
            })
      );
    } catch (error) {
      return { kind: 'error', cause: error instanceof Error ? error : new Error(String(error)) };
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
