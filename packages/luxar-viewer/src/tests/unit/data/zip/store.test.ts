/**
 * Tests for `LuxarZipStore` — the wrapper that exists to control WHEN the
 * central directory is read.
 *
 * Stock `ZipFileStore` reads it in its constructor and memoizes the promise
 * unconditionally, so building a store does I/O and one transient failure is
 * permanent. Both are pinned here against a real archive served through a
 * stubbed, range-honouring `fetch`.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { zipSync, strToU8 } from 'fflate';
import type { AbsolutePath } from '@zarrita/storage';
import { LuxarZipStore } from '../../../../data/zip/store';

const URL_ = 'https://example.com/scene.luxar.zarr.zip';

const ARCHIVE = zipSync({
  'zarr.json': strToU8('{"zarr_format":3,"node_type":"group"}'),
  'points/c/0/0': strToU8('CHUNK'),
});

/**
 * A `fetch` that speaks just enough HTTP for the range reader: HEAD for the
 * length, and 206 with `Content-Range` for a ranged GET.
 */
function serveArchive(onRequest?: (init: RequestInit) => void) {
  return vi.fn(async (_url: string, init: RequestInit = {}) => {
    onRequest?.(init);
    if (init.method === 'HEAD') {
      return {
        ok: true,
        status: 200,
        statusText: '',
        headers: new Headers({ 'content-length': String(ARCHIVE.length) }),
        arrayBuffer: async () => new ArrayBuffer(0),
      } as unknown as Response;
    }
    const header = (init.headers as Record<string, string> | undefined)?.Range ?? '';
    const [, start, end] = /bytes=(\d+)-(\d+)/.exec(header) ?? [];
    const from = Number(start);
    const to = Number(end);
    const slice = ARCHIVE.slice(from, to + 1);
    return {
      ok: true,
      status: 206,
      statusText: '',
      headers: new Headers({ 'content-range': `bytes ${from}-${to}/${ARCHIVE.length}` }),
      arrayBuffer: async () =>
        slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength),
    } as unknown as Response;
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('LuxarZipStore', () => {
  it('reads no bytes until the first get — construction is not I/O', async () => {
    const fetchMock = serveArchive();
    vi.stubGlobal('fetch', fetchMock);

    const store = new LuxarZipStore(URL_);
    expect(fetchMock).not.toHaveBeenCalled();

    await store.get('/zarr.json' as AbsolutePath);
    expect(fetchMock).toHaveBeenCalled();
  });

  it('resolves keys out of the archive', async () => {
    vi.stubGlobal('fetch', serveArchive());
    const store = new LuxarZipStore(URL_);

    const root = await store.get('/zarr.json' as AbsolutePath);
    expect(new TextDecoder().decode(root)).toContain('"node_type":"group"');
    expect(new TextDecoder().decode(await store.get('/points/c/0/0' as AbsolutePath))).toBe(
      'CHUNK'
    );
  });

  it('reads the central directory ONCE across many gets', async () => {
    const fetchMock = serveArchive();
    vi.stubGlobal('fetch', fetchMock);
    const store = new LuxarZipStore(URL_);

    await store.get('/zarr.json' as AbsolutePath);
    const afterFirst = fetchMock.mock.calls.length;
    await store.get('/points/c/0/0' as AbsolutePath);
    const afterSecond = fetchMock.mock.calls.length;

    // The second read costs only its own member reads — no HEAD, no directory.
    expect(afterSecond - afterFirst).toBeLessThan(afterFirst);
  });

  it('RETRIES after a failed open instead of failing forever', async () => {
    // The whole reason this wrapper exists: stock ZipFileStore memoizes a
    // rejected `info` promise, so one blip kills the store for the session.
    // The reader tolerates a failed HEAD by falling back to a probe GET, so a
    // single-request failure is not enough to fail an open — take the network
    // down entirely for the first attempt.
    let offline = true;
    const working = serveArchive();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init: RequestInit = {}) => {
        if (offline) throw new TypeError('Failed to fetch');
        return working(url, init);
      })
    );

    const store = new LuxarZipStore(URL_);
    await expect(store.get('/zarr.json' as AbsolutePath)).rejects.toThrow();
    offline = false;

    // Same store, next caller: succeeds.
    const root = await store.get('/zarr.json' as AbsolutePath);
    expect(new TextDecoder().decode(root)).toContain('"node_type":"group"');
  });

  it('returns undefined for a missing key without any extra request', async () => {
    const fetchMock = serveArchive();
    vi.stubGlobal('fetch', fetchMock);
    const store = new LuxarZipStore(URL_);

    await store.get('/zarr.json' as AbsolutePath);
    const before = fetchMock.mock.calls.length;

    expect(await store.get('/nope/c/0' as AbsolutePath)).toBeUndefined();
    // A miss is answered from the in-memory directory — unlike a directory
    // store, which pays a 404 for it.
    expect(fetchMock.mock.calls.length).toBe(before);
  });

  it('refuses to re-open after dispose', async () => {
    vi.stubGlobal('fetch', serveArchive());
    const store = new LuxarZipStore(URL_);
    await store.get('/zarr.json' as AbsolutePath);

    store.dispose();
    await expect(store.get('/zarr.json' as AbsolutePath)).rejects.toThrow(/disposed/);
  });
});
