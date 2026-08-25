/**
 * Tests for the zipped-store byte source.
 *
 * Two things matter here and are not covered by the store's own tests: the
 * outcome mapping the caching store switches on, and the identity probe — which
 * is the ONLY way a zipped store can answer "is this still the scene I cached?",
 * because its root attrs live inside the archive and the reader's directory is
 * a snapshot.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import type { ArchiveByteReader } from '../../../../cache/chunk-source';
import { ZipChunkSource } from '../../../../cache/chunk-source/zip-chunk-source';

const ARCHIVE_URL = 'https://example.com/scene.luxar.zarr.zip';

/** The archive is injected, so reads need no network at all. */
function fakeReader(
  get: (key: string) => Promise<Uint8Array | undefined>
): ArchiveByteReader & { disposed: boolean } {
  return {
    get,
    disposed: false,
    dispose() {
      this.disposed = true;
    },
  };
}

const emptyReader = () => fakeReader(async () => undefined);

function headResponse(headers: Record<string, string>, ok = true): Response {
  return {
    ok,
    status: ok ? 200 : 404,
    statusText: '',
    headers: new Headers(headers),
    arrayBuffer: async () => new ArrayBuffer(0),
  } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ZipChunkSource — identity', () => {
  it('keeps the archive URL verbatim, distinct from its directory twin', () => {
    // The OPFS bucket is derived from this. Folding the two together would let
    // an archive serve a directory's cached members, and vice versa, with
    // nothing failing loudly — they hold the same bytes under different keys.
    const source = new ZipChunkSource(ARCHIVE_URL, emptyReader());
    expect(source.identity).toBe(ARCHIVE_URL);
    expect(source.identity).not.toBe('https://example.com/scene.luxar.zarr/');
  });
});

describe('ZipChunkSource — identity probe', () => {
  it('prefers the ETag', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => headResponse({ etag: '"abc123"', 'last-modified': 'Mon, 01 Jan 2035' }))
    );

    const token = await new ZipChunkSource(ARCHIVE_URL, emptyReader()).probeIdentityToken({});

    expect(token).toEqual({ hash: 'etag:"abc123"', mode: 'archive-etag' });
  });

  it('falls back to modification time AND size when there is no ETag', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        headResponse({ 'last-modified': 'Mon, 01 Jan 2035 00:00:00 GMT', 'content-length': '4096' })
      )
    );

    const token = await new ZipChunkSource(ARCHIVE_URL, emptyReader()).probeIdentityToken({});

    expect(token?.mode).toBe('archive-etag');
    expect(token?.hash).toContain('4096');
  });

  it('uses HEAD and bypasses the browser cache', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      headResponse({ etag: '"x"' })
    );
    vi.stubGlobal('fetch', fetchMock);

    await new ZipChunkSource(ARCHIVE_URL, emptyReader()).probeIdentityToken({});

    const init = fetchMock.mock.calls[0]?.[1];
    expect(init?.method).toBe('HEAD');
    // A cached probe answer would defeat the point of probing.
    expect(init?.cache).toBe('no-store');
  });

  it('returns null — never a change verdict — when it cannot tell', async () => {
    // Offline, CORS-blocked, or a server exposing neither header. None of these
    // say the scene changed, and reporting one would raise a false banner.
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      })
    );
    expect(await new ZipChunkSource(ARCHIVE_URL, emptyReader()).probeIdentityToken({})).toBeNull();

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => headResponse({}))
    );
    expect(await new ZipChunkSource(ARCHIVE_URL, emptyReader()).probeIdentityToken({})).toBeNull();

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => headResponse({}, false))
    );
    expect(await new ZipChunkSource(ARCHIVE_URL, emptyReader()).probeIdentityToken({})).toBeNull();
  });
});

describe('ZipChunkSource — outcomes', () => {
  it('returns bytes for a present key', async () => {
    const reader = fakeReader(async () => new Uint8Array([1, 2, 3]));
    const outcome = await new ZipChunkSource(ARCHIVE_URL, reader).get('/points/c/0');

    expect(outcome.kind).toBe('ok');
    if (outcome.kind !== 'ok') return;
    expect(Array.from(outcome.data)).toEqual([1, 2, 3]);
  });

  it('maps an absent member to `missing`', async () => {
    const outcome = await new ZipChunkSource(ARCHIVE_URL, emptyReader()).get('/nope/c/0');
    expect(outcome.kind).toBe('missing');
  });

  it('reports `aborted` without touching the archive', async () => {
    const reader = fakeReader(async () => {
      throw new Error('should not be read');
    });
    const controller = new AbortController();
    controller.abort();

    const outcome = await new ZipChunkSource(ARCHIVE_URL, reader).get(
      '/zarr.json',
      controller.signal
    );

    expect(outcome.kind).toBe('aborted');
  });

  it('disposes the injected archive', () => {
    const reader = emptyReader();
    new ZipChunkSource(ARCHIVE_URL, reader).dispose();
    expect(reader.disposed).toBe(true);
  });

  it('surfaces a broken archive as `error`, not as a throw', async () => {
    // A source that throws would escape the caching store's Result contract.
    const reader = fakeReader(async () => {
      throw new TypeError('Failed to fetch');
    });

    const outcome = await new ZipChunkSource(ARCHIVE_URL, reader).get('/zarr.json');
    expect(outcome.kind).toBe('error');
  });
});
