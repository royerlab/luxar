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

  it('THROWS when a partial response body is shorter than the requested window', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        response(new Uint8Array([7, 8]), {
          status: 206,
          headers: { 'content-range': 'bytes 10-12/100' },
        })
      )
    );

    const read = new LuxarHttpRangeReader(URL_).read(10, 3);
    await expect(read).rejects.toBeInstanceOf(RangeUnsupportedError);
    await expect(read).rejects.toThrow(/requested window was 3 bytes.*returned 2/i);
  });

  it('THROWS when Content-Range starts at a different offset', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        response(new Uint8Array([7, 8, 9]), {
          status: 206,
          headers: { 'content-range': 'bytes 0-2/100' },
        })
      )
    );

    const read = new LuxarHttpRangeReader(URL_).read(10, 3);
    await expect(read).rejects.toBeInstanceOf(RangeUnsupportedError);
    await expect(read).rejects.toThrow(/requested window was bytes 10-12.*reported bytes 0-2/i);
  });

  it('THROWS when Content-Range ends at a different offset', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        response(new Uint8Array([7, 8, 9]), {
          status: 206,
          headers: { 'content-range': 'bytes 10-99/100' },
        })
      )
    );

    const read = new LuxarHttpRangeReader(URL_).read(10, 3);
    await expect(read).rejects.toBeInstanceOf(RangeUnsupportedError);
    await expect(read).rejects.toThrow(/requested window was bytes 10-12.*reported bytes 10-99/i);
  });

  it('accepts a correct-sized partial body when Content-Range is unparseable', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        response(new Uint8Array([7, 8, 9]), {
          status: 206,
          headers: { 'content-range': 'bytes=10-12/100' },
        })
      )
    );

    const bytes = await new LuxarHttpRangeReader(URL_).read(10, 3);
    expect(Array.from(bytes)).toEqual([7, 8, 9]);
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

describe('LuxarHttpRangeReader.probeIdentity', () => {
  it('prefers the ETag', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        response(new Uint8Array(0), {
          headers: { etag: '"abc123"', 'content-length': '4096' },
        })
      )
    );

    expect(await new LuxarHttpRangeReader(URL_).probeIdentity()).toBe('etag:"abc123"');
  });

  it('falls back to modification time AND size when there is no ETag', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        response(new Uint8Array(0), {
          headers: { 'last-modified': 'Mon, 01 Jan 2035 00:00:00 GMT', 'content-length': '4096' },
        })
      )
    );

    expect(await new LuxarHttpRangeReader(URL_).probeIdentity()).toBe(
      'mtime:Mon, 01 Jan 2035 00:00:00 GMT:4096'
    );
  });

  it('falls back to a ranged GET when HEAD is refused, and still seeds the length', async () => {
    // A HEAD-hostile host would otherwise yield no token at all → validation
    // mode `none` → the archive is never re-checked and a replaced one keeps
    // serving stale chunks.
    const fetchMock = vi.fn(async (_url: string, init: RequestInit = {}) => {
      if (init.method === 'HEAD') return response(new Uint8Array(0), { status: 405 });
      return response(new Uint8Array([0]), {
        status: 206,
        headers: { etag: '"from-get"', 'content-range': 'bytes 0-0/7777' },
      });
    });
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    const reader = new LuxarHttpRangeReader(URL_);
    expect(await reader.probeIdentity()).toBe('etag:"from-get"');
    // Content-Range carries the TOTAL; Content-Length on a 206 is the range.
    expect(await reader.getLength()).toBe(7777);
  });

  it('returns null rather than a false verdict when it cannot tell', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      })
    );
    expect(await new LuxarHttpRangeReader(URL_).probeIdentity()).toBeNull();

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => response(new Uint8Array(0), { headers: {} }))
    );
    expect(await new LuxarHttpRangeReader(URL_).probeIdentity()).toBeNull();
  });

  it('SEEDS the length, so getLength costs no second request', async () => {
    // The whole point of the probe living here: identity and length are the
    // same HEAD, and paying twice adds a serialised round trip to every load.
    const fetchMock = vi.fn(async () =>
      response(new Uint8Array(0), { headers: { etag: '"x"', 'content-length': '9001' } })
    );
    vi.stubGlobal('fetch', fetchMock);

    const reader = new LuxarHttpRangeReader(URL_);
    await reader.probeIdentity();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    expect(await reader.getLength()).toBe(9001);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe('LuxarHttpRangeReader — retained ranges', () => {
  it('serves a read contained in an already-retained range from memory', async () => {
    // This is the guaranteed duplication: unzipit reads a fixed 65,557-byte
    // tail, then re-reads the central directory sitting inside it.
    const body = new Uint8Array(Array.from({ length: 100 }, (_, i) => i));
    const fetchMock = vi.fn(async (_url: string, init: RequestInit = {}) => {
      const header = (init.headers as Record<string, string> | undefined)?.Range ?? '';
      const [, start, end] = /bytes=(\d+)-(\d+)/.exec(header) ?? [];
      const from = Number(start);
      const to = Number(end);
      return response(body.slice(from, to + 1), {
        status: 206,
        headers: { 'content-range': `bytes ${from}-${to}/100` },
      });
    });
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    const reader = new LuxarHttpRangeReader(URL_);
    reader.retainReads(true);
    await reader.read(0, 100);
    const afterFirst = fetchMock.mock.calls.length;

    const inner = await reader.read(10, 5);

    expect(Array.from(inner)).toEqual([10, 11, 12, 13, 14]);
    expect(fetchMock.mock.calls.length).toBe(afterFirst);
  });

  it('STITCHES a read that overruns a retained range, fetching only the gap', async () => {
    // The case this class exists for, and the one plain containment misses:
    // unzipit retains a 65,557-byte tail and then asks for a central directory
    // LARGER than it, starting before it. Only the missing prefix should move.
    const body = new Uint8Array(Array.from({ length: 100 }, (_, i) => i));
    const asked: string[] = [];
    const fetchMock = vi.fn(async (_url: string, init: RequestInit = {}) => {
      const header = (init.headers as Record<string, string> | undefined)?.Range ?? '';
      asked.push(header);
      const [, start, end] = /bytes=(\d+)-(\d+)/.exec(header) ?? [];
      const from = Number(start);
      const to = Number(end);
      return response(body.slice(from, to + 1), {
        status: 206,
        headers: { 'content-range': `bytes ${from}-${to}/100` },
      });
    });
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    const reader = new LuxarHttpRangeReader(URL_);
    reader.retainReads(true);
    await reader.read(60, 40); // the "tail": [60, 100)
    asked.length = 0;

    // Overlaps the tail but starts before it — exactly the directory's shape.
    const stitched = await reader.read(50, 30); // [50, 80)

    expect(Array.from(stitched)).toEqual(Array.from({ length: 30 }, (_, i) => 50 + i));
    // One request, and only for the 10 missing bytes.
    expect(asked).toEqual(['bytes=50-59']);
  });

  it('rejects a short stitched prefix instead of zero-filling the gap', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        response(new Uint8Array(40), {
          status: 206,
          headers: { 'content-range': 'bytes 60-99/100' },
        })
      )
      .mockResolvedValueOnce(
        response(new Uint8Array(1), {
          status: 206,
          headers: { 'content-range': 'bytes 50-59/100' },
        })
      );
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    const reader = new LuxarHttpRangeReader(URL_);
    reader.retainReads(true);
    await reader.read(60, 40);

    const read = reader.read(50, 30);
    await expect(read).rejects.toBeInstanceOf(RangeUnsupportedError);
    await expect(read).rejects.toThrow(/requested window was 10 bytes.*returned 1/i);
  });

  it('stops retaining once the cap is reached, rather than growing without bound', async () => {
    // The only thing between a pathological central directory and the heap.
    const big = LuxarHttpRangeReader.MAX_RETAINED_BYTES + 1;
    const fetchMock = vi.fn(async (_url: string, init: RequestInit = {}) => {
      const header = (init.headers as Record<string, string> | undefined)?.Range ?? '';
      const [, start, end] = /bytes=(\d+)-(\d+)/.exec(header) ?? [];
      const from = Number(start);
      const to = Number(end);
      return response(new Uint8Array(to - from + 1), {
        status: 206,
        headers: { 'content-range': `bytes ${from}-${to}/${big + 10}` },
      });
    });
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    const reader = new LuxarHttpRangeReader(URL_);
    reader.retainReads(true);
    await reader.read(0, big); // over the cap — must not be retained
    const afterFirst = fetchMock.mock.calls.length;

    await reader.read(10, 5); // fully inside it, but nothing was kept
    expect(fetchMock.mock.calls.length).toBeGreaterThan(afterFirst);
  });

  it('does NOT retain once the directory phase is over', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit = {}) => {
      const header = (init.headers as Record<string, string> | undefined)?.Range ?? '';
      const [, start, end] = /bytes=(\d+)-(\d+)/.exec(header) ?? [];
      return response(new Uint8Array(Number(end) - Number(start) + 1), {
        status: 206,
        headers: { 'content-range': `bytes ${start}-${end}/100` },
      });
    });
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    // Member payloads are cached a layer up by chunk key; retaining them here
    // would double the memory for the same bytes.
    const reader = new LuxarHttpRangeReader(URL_);
    await reader.read(0, 50);
    const afterFirst = fetchMock.mock.calls.length;
    await reader.read(10, 5);

    expect(fetchMock.mock.calls.length).toBeGreaterThan(afterFirst);
  });
});
