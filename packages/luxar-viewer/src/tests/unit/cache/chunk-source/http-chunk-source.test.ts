/**
 * Unit tests for the directory-store byte source.
 *
 * `MultiLevelCachingStore`'s own suite already drives this path end to end
 * through `global.fetch` (and did not change when the seam was extracted).
 * What it does NOT do is pin the source's contract directly: the mapping from
 * an HTTP answer to a {@link ChunkFetchOutcome}, and the response-body
 * cancellation that moved out of the store and into here.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { HttpChunkSource } from '../../../../cache/chunk-source/http-chunk-source';

const BASE = 'https://example.com/data.zarr';

function bodyResponse(bytes: Uint8Array, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: '',
    headers: new Headers(),
    body: null,
    arrayBuffer: async () => bytes.buffer.slice(0) as ArrayBuffer,
  } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('HttpChunkSource — identity', () => {
  it('keeps the dataset URL verbatim so existing OPFS buckets still resolve', () => {
    const source = new HttpChunkSource(BASE);
    expect(source.identity).toBe(BASE);
    expect(source.describe).toBe(BASE);
  });

  it('does NOT fold a zipped URL onto its directory twin', () => {
    // Different key namespaces; sharing a bucket would let one serve the
    // other's members with nothing failing loudly.
    expect(new HttpChunkSource(`${BASE}.zip`).identity).not.toBe(
      new HttpChunkSource(BASE).identity
    );
  });
});

describe('HttpChunkSource — outcomes', () => {
  it('returns bytes for a 200, with bytesOverWire equal to the body length', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => bodyResponse(new Uint8Array([1, 2, 3, 4])))
    );

    const outcome = await new HttpChunkSource(BASE).get('c/0/0');

    expect(outcome.kind).toBe('ok');
    if (outcome.kind !== 'ok') return;
    expect(Array.from(outcome.data)).toEqual([1, 2, 3, 4]);
    // Equal over plain HTTP — the invariant that keeps the byte counters
    // reading exactly as they did before the seam existed.
    expect(outcome.bytesOverWire).toBe(outcome.data.byteLength);
  });

  it('builds the chunk URL from the base and the key', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      bodyResponse(new Uint8Array(1))
    );
    vi.stubGlobal('fetch', fetchMock);

    await new HttpChunkSource(BASE).get('c/0/0');

    expect(fetchMock.mock.calls[0]?.[0]).toBe(`${BASE}/c/0/0`);
  });

  it('maps a 404 to `missing`, not to an error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => bodyResponse(new Uint8Array(0), 404))
    );

    expect((await new HttpChunkSource(BASE).get('c/9/9')).kind).toBe('missing');
  });

  it.each([400, 401, 403, 410, 416])(
    'maps HTTP %s to an error instead of pretending the key is missing',
    async (status) => {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => bodyResponse(new Uint8Array(0), status))
      );

      const outcome = await new HttpChunkSource(BASE).get('c/9/9');

      expect(outcome.kind).toBe('error');
      if (outcome.kind !== 'error') return;
      expect(outcome.cause.message).toContain(String(status));
      expect(outcome.cause.message).toContain('c/9/9');
    }
  );

  it('retries a 5xx and succeeds when the server recovers', async () => {
    let attempt = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        attempt += 1;
        return attempt === 1
          ? bodyResponse(new Uint8Array(0), 503)
          : bodyResponse(new Uint8Array([7]));
      })
    );

    const outcome = await new HttpChunkSource(BASE).get('c/0/0');

    expect(outcome.kind).toBe('ok');
    expect(attempt).toBeGreaterThan(1);
  });

  it('reports `aborted` when the caller signal is already aborted', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => bodyResponse(new Uint8Array(1)))
    );

    const controller = new AbortController();
    controller.abort();

    expect((await new HttpChunkSource(BASE).get('c/0/0', controller.signal)).kind).toBe('aborted');
  });

  it('surfaces exhausted retries as `error` rather than throwing', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      })
    );

    const outcome = await new HttpChunkSource(BASE).get('c/0/0');
    // `fetchWithRetry` swallows the throw and returns undefined after its
    // budget, so this lands on the retry-exhaustion branch — NOT the `catch`.
    // Either way the contract holds: a source reports, it never throws.
    expect(outcome.kind).toBe('error');
    if (outcome.kind !== 'error') return;
    expect(outcome.cause.message).toContain('exhausted retries');
  });

  it('maps an ABORT DURING THE BODY READ to `aborted`, not a throw', async () => {
    // The branch the fix was written for, and the one the existing
    // already-aborted case never reaches: the signal fires after the headers
    // arrive, so `arrayBuffer()` is the thing that rejects. A throw here escapes
    // as NetworkError → undefined → zarrita fills the chunk. Silently wrong
    // geometry, which is why this branch has to be pinned.
    const controller = new AbortController();
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        const response = bodyResponse(new Uint8Array(4));
        Object.defineProperty(response, 'arrayBuffer', {
          value: async () => {
            controller.abort();
            throw new DOMException('aborted', 'AbortError');
          },
        });
        return response;
      })
    );

    const outcome = await new HttpChunkSource(BASE).get('c/0/0', controller.signal);

    expect(outcome.kind).toBe('aborted');
  });

  it('maps a non-abort body failure to `error`, carrying the cause', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        const response = bodyResponse(new Uint8Array(4));
        Object.defineProperty(response, 'arrayBuffer', {
          value: async () => {
            throw new TypeError('network died mid-body');
          },
        });
        return response;
      })
    );

    const outcome = await new HttpChunkSource(BASE).get('c/0/0');

    expect(outcome.kind).toBe('error');
    if (outcome.kind !== 'error') return;
    expect(outcome.cause.message).toContain('mid-body');
  });

  it('cancels a body it declines to read', async () => {
    const cancel = vi.fn(async () => undefined);
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        const response = bodyResponse(new Uint8Array(0), 404);
        Object.defineProperty(response, 'body', {
          value: { cancel, locked: false } as unknown as ReadableStream,
        });
        return response;
      })
    );

    await new HttpChunkSource(BASE).get('c/0/0');

    // Ownership of this moved out of MultiLevelCachingStore's `finally` and
    // into the source; without it the server keeps streaming an ignored body.
    expect(cancel).toHaveBeenCalled();
  });
});

describe('HttpChunkSource — identity probe', () => {
  it('delegates to the root-document probe and reports its token', async () => {
    // The one method the suite did not touch. It is what `validateCache` calls,
    // so a silent failure here means a replaced dataset is never detected.
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => bodyResponse(new TextEncoder().encode('{"content_hash":"abc123"}')))
    );

    const token = await new HttpChunkSource(BASE).probeIdentityToken({});

    expect(token?.hash).toContain('abc123');
    expect(token?.mode).toBe('content-hash');
  });

  it('reports null when no root document answers', async () => {
    // "Cannot tell" — never a change verdict, or every offline load would wipe
    // a perfectly good cache.
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => bodyResponse(new Uint8Array(0), 404))
    );

    expect(await new HttpChunkSource(BASE).probeIdentityToken({})).toBeNull();
  });
});
