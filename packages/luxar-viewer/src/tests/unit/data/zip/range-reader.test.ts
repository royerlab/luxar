/**
 * Unit tests for the zipped-store HTTP range reader.
 *
 * The load-bearing case is a server that IGNORES `Range` and answers `200` with
 * the whole file. Upstream's reader accepts that silently and hands the full
 * body back as if it were the requested window, so the zip parser reads garbage
 * at every offset and the archive looks corrupt. These tests pin the loud
 * failure instead — which matters concretely, because Python's `http.server`
 * (the Playwright fixture server) has no Range support at all.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  LuxarHttpRangeReader,
  RangeUnsupportedError,
  parseContentRangeTotal,
} from '../../../../data/zip/range-reader';

const URL_ = 'https://example.com/scene.luxar.zarr.zip';

function response(
  body: Uint8Array,
  init: { status?: number; headers?: Record<string, string> } = {}
): Response {
  const headers = new Headers(init.headers ?? {});
  return {
    ok: (init.status ?? 200) < 400,
    status: init.status ?? 200,
    statusText: 'x',
    headers,
    arrayBuffer: async () => body.buffer.slice(0) as ArrayBuffer,
  } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('parseContentRangeTotal', () => {
  it('reads the total off a well-formed header', () => {
    expect(parseContentRangeTotal('bytes 0-9/1234')).toBe(1234);
  });

  it('returns null for a missing header or an unknown total', () => {
    expect(parseContentRangeTotal(null)).toBeNull();
    expect(parseContentRangeTotal('bytes 0-9/*')).toBeNull();
  });
});

describe('LuxarHttpRangeReader.getLength', () => {
  it('uses HEAD Content-Length when available, and caches it', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      response(new Uint8Array(0), { headers: { 'content-length': '4096' } })
    );
    vi.stubGlobal('fetch', fetchMock);

    const reader = new LuxarHttpRangeReader(URL_);
    expect(await reader.getLength()).toBe(4096);
    expect(await reader.getLength()).toBe(4096);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({ method: 'HEAD' });
  });

  it('falls back to a one-byte ranged GET when HEAD is rejected', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      if (init.method === 'HEAD') return response(new Uint8Array(0), { status: 405 });
      return response(new Uint8Array([1]), {
        status: 206,
        headers: { 'content-range': 'bytes 0-0/9001' },
      });
    });
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    expect(await new LuxarHttpRangeReader(URL_).getLength()).toBe(9001);
  });

  it('reports a range-less server from the probe rather than guessing a length', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      if (init.method === 'HEAD') return response(new Uint8Array(0), { status: 405 });
      return response(new Uint8Array(500), { status: 200 }); // ignored the Range
    });
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    await expect(new LuxarHttpRangeReader(URL_).getLength()).rejects.toBeInstanceOf(
      RangeUnsupportedError
    );
  });

  it.each([
    [404, /archive was not found/i],
    [410, /archive was not found/i],
    [401, /access to the archive was denied/i],
    [403, /access to the archive was denied/i],
  ])('reports HTTP %i without blaming Range support', async (status, message) => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      if (init.method === 'HEAD') return response(new Uint8Array(0), { status: 405 });
      return response(new Uint8Array(0), { status });
    });
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    const error = await new LuxarHttpRangeReader(URL_).getLength().catch((caught) => caught);
    expect(error).not.toBeInstanceOf(RangeUnsupportedError);
    expect(String(error)).toMatch(message);
    expect(String(error)).not.toMatch(/luxar serve/);
  });

  it('names Content-Range exposure when a probe cannot read the header', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      if (init.method === 'HEAD') return response(new Uint8Array(0), { status: 405 });
      return response(new Uint8Array([1]), { status: 206 });
    });
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    await expect(new LuxarHttpRangeReader(URL_).getLength()).rejects.toThrow(
      /Content-Range.*Access-Control-Expose-Headers/i
    );
  });
});

describe('LuxarHttpRangeReader.read', () => {
  it('sends the right Range header and returns the partial body', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      response(new Uint8Array([7, 8, 9]), {
        status: 206,
        headers: { 'content-range': 'bytes 10-12/100' },
      })
    );
    vi.stubGlobal('fetch', fetchMock);

    const bytes = await new LuxarHttpRangeReader(URL_).read(10, 3);

    expect(Array.from(bytes)).toEqual([7, 8, 9]);
    const headers = fetchMock.mock.calls[0]?.[1]?.headers as Record<string, string>;
    expect(headers.Range).toBe('bytes=10-12');
  });

  it('short-circuits a zero-length read without touching the network', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    expect((await new LuxarHttpRangeReader(URL_).read(0, 0)).length).toBe(0);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('THROWS on a 200 answer to a ranged read instead of accepting the whole file', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => response(new Uint8Array(1000), { status: 200 }))
    );

    const reader = new LuxarHttpRangeReader(URL_);
    await expect(reader.read(10, 3)).rejects.toBeInstanceOf(RangeUnsupportedError);
    // The message must point at the fix, not just the symptom.
    await expect(reader.read(10, 3)).rejects.toThrow(/luxar serve/);
  });

  it('reports a missing archive without Range advice', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => response(new Uint8Array(0), { status: 404 }))
    );

    const error = await new LuxarHttpRangeReader(URL_).read(10, 3).catch((caught) => caught);
    expect(error).not.toBeInstanceOf(RangeUnsupportedError);
    expect(String(error)).toMatch(/archive was not found.*404/i);
    expect(String(error)).not.toMatch(/luxar serve/);
  });

  it('detects an archive that changed size mid-read', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      if (init.method === 'HEAD') {
        return response(new Uint8Array(0), { headers: { 'content-length': '100' } });
      }
      return response(new Uint8Array([1]), {
        status: 206,
        headers: { 'content-range': 'bytes 0-0/250' }, // grew underneath us
      });
    });
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    const reader = new LuxarHttpRangeReader(URL_);
    expect(await reader.getLength()).toBe(100);
    await expect(reader.read(0, 1)).rejects.toThrow(/changed size during the read/);
  });
});
