/**
 * Request economy of a hosted `.zarr.zip`: how many round trips an open and a
 * member read cost.
 *
 * Every request is a Range GET against ONE URL, so on a high-latency host the
 * count of SERIAL requests is the whole story. Measured at 100 ms RTT / 25 Mbps
 * over HTTP/2 (CT zip): two GETs per member (unzipit reads the 30-byte local
 * header, then the payload) plus a three-round-trip open cost first frame 5.9 s
 * vs 6.3 s for the same scene unzipped once both were collapsed.
 *
 * The server here is a strict, range-honouring fake over a REAL archive, and it
 * records every request so the counts are exact.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { zipSync, strToU8 } from 'fflate';
import { LuxarZipStore } from '../../../../data/zip/store';
import { perfCounters } from '../../../../profiling/perf-counters';

const URL_ = 'https://example.com/scene.luxar.zarr.zip';
const TAIL = 65_557;

interface Served {
  method: string;
  range: string | null;
  cache: RequestCache | undefined;
}

/** Deterministic, incompressible-looking bytes. */
function noise(length: number, seed = 1): Uint8Array {
  const out = new Uint8Array(length);
  let x = seed;
  for (let i = 0; i < length; i++) {
    x = (Math.imul(x, 1103515245) + 12345) >>> 0;
    out[i] = x >>> 24;
  }
  return out;
}

/** A strict range server: HEAD, `bytes=a-b`, `bytes=a-`, and suffix `bytes=-n`. */
function serve(
  archive: Uint8Array,
  headers: Record<string, string> = { etag: '"v1"' },
  exposeContentRange = true
) {
  const seen: Served[] = [];
  const fetchMock = vi.fn(async (_url: string, init: RequestInit = {}) => {
    const range = (init.headers as Record<string, string> | undefined)?.Range ?? null;
    seen.push({ method: init.method ?? 'GET', range, cache: init.cache });
    const size = archive.length;
    if (init.method === 'HEAD') {
      return new Response(null, {
        status: 200,
        headers: { 'content-length': String(size), ...headers },
      });
    }
    const m = /^bytes=(\d*)-(\d*)$/.exec(range ?? '');
    if (!m) {
      return new Response(archive.slice(), {
        status: 200,
        headers: { 'content-length': String(size), ...headers },
      });
    }
    let start: number;
    let end: number;
    if (m[1] === '') {
      start = Math.max(0, size - Number(m[2]));
      end = size - 1;
    } else {
      start = Number(m[1]);
      end = m[2] === '' ? size - 1 : Math.min(size - 1, Number(m[2]));
    }
    return new Response(archive.slice(start, end + 1), {
      status: 206,
      headers: {
        ...(exposeContentRange ? { 'content-range': `bytes ${start}-${end}/${size}` } : {}),
        ...headers,
      },
    });
  });
  return { fetchMock, seen };
}

/** An archive larger than the tail window: the big member sits OUTSIDE it. */
function bigArchive(level: 0 | 6 = 0) {
  const big = noise(200_000);
  const archive = zipSync(
    {
      'points/c/0/0': [big, { level }],
      'zarr.json': strToU8('{"zarr_format":3,"node_type":"group"}'),
    },
    { level }
  );
  expect(archive.length).toBeGreaterThan(TAIL * 2);
  return { archive, big };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('LuxarZipStore — opening', () => {
  it('opens with exactly ONE request (a suffix GET), not HEAD + length + tail', async () => {
    const { archive } = bigArchive();
    const { fetchMock, seen } = serve(archive);
    vi.stubGlobal('fetch', fetchMock);

    const store = new LuxarZipStore(URL_);
    expect(await store.has('/zarr.json')).toBe(true);

    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ method: 'GET', range: `bytes=-${TAIL}` });
  });

  it('costs ONE request for identity probe + open together', async () => {
    const { archive } = bigArchive();
    const { fetchMock, seen } = serve(archive);
    vi.stubGlobal('fetch', fetchMock);

    const store = new LuxarZipStore(URL_);
    expect(await store.probeIdentity(undefined, 5_000)).toBe('etag:"v1"');
    expect(await store.has('/zarr.json')).toBe(true);

    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ range: `bytes=-${TAIL}`, cache: 'no-store' });
  });

  it('costs ONE request for identity probe + open without identity headers', async () => {
    const { archive } = bigArchive();
    const { fetchMock, seen } = serve(archive, {});
    vi.stubGlobal('fetch', fetchMock);

    const store = new LuxarZipStore(URL_);
    expect(await store.probeIdentity(undefined, 5_000)).toBeNull();
    expect(await store.has('/zarr.json')).toBe(true);

    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ method: 'GET', range: `bytes=-${TAIL}` });
  });

  it('keeps the mtime+size identity when a host sends no ETag', async () => {
    const { archive } = bigArchive();
    const { fetchMock } = serve(archive, { 'last-modified': 'Mon, 01 Jan 2035 00:00:00 GMT' });
    vi.stubGlobal('fetch', fetchMock);

    const store = new LuxarZipStore(URL_);
    // The TOTAL (from Content-Range), never the 65,557-byte range length.
    expect(await store.probeIdentity()).toBe(
      `mtime:Mon, 01 Jan 2035 00:00:00 GMT:${archive.length}`
    );
  });

  it('opens an archive smaller than the tail window in one request, members included', async () => {
    const archive = zipSync({
      'zarr.json': strToU8('{"zarr_format":3,"node_type":"group"}'),
      'points/c/0/0': strToU8('CHUNK'),
    });
    const { fetchMock, seen } = serve(archive);
    vi.stubGlobal('fetch', fetchMock);

    const store = new LuxarZipStore(URL_);
    expect(new TextDecoder().decode(await store.get('/points/c/0/0'))).toBe('CHUNK');
    expect(seen).toHaveLength(1);
  });
});

describe('LuxarZipStore — opening on imperfect hosts', () => {
  it('does not use HEAD for a tokenless probe when Content-Range is hidden', async () => {
    const { archive } = bigArchive();
    const { fetchMock, seen } = serve(archive, {}, false);
    vi.stubGlobal('fetch', fetchMock);

    expect(await new LuxarZipStore(URL_).probeIdentity(undefined, 5_000)).toBeNull();
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ method: 'GET', range: `bytes=-${TAIL}` });
  });

  it('opens and reads a member when Content-Range is hidden', async () => {
    const { archive, big } = bigArchive();
    const { fetchMock, seen } = serve(archive, undefined, false);
    vi.stubGlobal('fetch', fetchMock);

    const store = new LuxarZipStore(URL_);
    expect(await store.has('/zarr.json')).toBe(true);
    expect(await store.get('/points/c/0/0')).toEqual(big);
    expect(seen.some((request) => request.method === 'HEAD')).toBe(true);
  });

  it('uses HEAD for mtime identity when Content-Range is hidden', async () => {
    const { archive } = bigArchive();
    const { fetchMock, seen } = serve(
      archive,
      { 'last-modified': 'Mon, 01 Jan 2035 00:00:00 GMT' },
      false
    );
    vi.stubGlobal('fetch', fetchMock);

    const store = new LuxarZipStore(URL_);
    expect(await store.probeIdentity(undefined, 5_000)).toBe(
      `mtime:Mon, 01 Jan 2035 00:00:00 GMT:${archive.length}`
    );
    expect(seen.map((request) => request.method)).toEqual(['GET', 'HEAD']);
  });

  it('falls back to explicit ranges when a host refuses SUFFIX ranges (416)', async () => {
    const { archive, big } = bigArchive();
    const strict = serve(archive);
    const fetchMock = vi.fn(async (url: string, init: RequestInit = {}) => {
      const range = (init.headers as Record<string, string> | undefined)?.Range ?? '';
      if (range.startsWith('bytes=-')) {
        return new Response(null, { status: 416, headers: { 'content-range': '*/0' } });
      }
      return strict.fetchMock(url, init);
    });
    vi.stubGlobal('fetch', fetchMock);

    const store = new LuxarZipStore(URL_);
    expect(await store.get('/points/c/0/0')).toEqual(big);
  });

  it('still reports a host that ignores Range with the actionable Range error', async () => {
    const { archive } = bigArchive();
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async (_url: string, init: RequestInit = {}) =>
          new Response(init.method === 'HEAD' ? null : archive.slice(), {
            status: 200,
            headers: { 'content-length': String(archive.length) },
          })
      )
    );

    const store = new LuxarZipStore(URL_);
    await expect(store.get('/zarr.json')).rejects.toThrow(/honours HTTP Range requests/);
  });

  it('counts the probe-fetched tail in fetch.bytes', async () => {
    const { archive } = bigArchive();
    vi.stubGlobal('fetch', serve(archive).fetchMock);
    perfCounters.reset();

    await new LuxarZipStore(URL_).probeIdentity();

    expect(perfCounters.get('fetch.bytes')).toBe(TAIL);
  });
});

describe('LuxarZipStore — member reads', () => {
  it.each([
    ['STORED', 0],
    ['DEFLATE', 6],
  ] as const)(
    'reads a %s member outside the tail with exactly ONE ranged GET',
    async (_, level) => {
      const { archive, big } = bigArchive(level);
      const { fetchMock, seen } = serve(archive);
      vi.stubGlobal('fetch', fetchMock);

      const store = new LuxarZipStore(URL_);
      await store.has('/zarr.json');
      const before = seen.length;

      const bytes = await store.get('/points/c/0/0');

      expect(bytes).toEqual(big);
      expect(seen.length - before).toBe(1);
      expect(seen.at(-1)?.cache).toBe('no-store');
    }
  );

  it('still reads correctly (one extra GET) when the LOCAL extra field outgrows the slack', async () => {
    // Writers may put a larger extra field in the local header than in the
    // central directory (Info-ZIP timestamps, ZIP64 sizes). The one-GET window
    // is sized from the central directory plus slack; when the local header is
    // bigger than that, the payload must be re-read rather than mis-sliced.
    const payload = noise(100_000, 7);
    const archive = storedArchiveWithLocalExtra('points/c/0/0', payload, 4_000);
    const { fetchMock, seen } = serve(archive);
    vi.stubGlobal('fetch', fetchMock);

    const store = new LuxarZipStore(URL_);
    await store.has('/zarr.json');
    const before = seen.length;

    expect(await store.get('/points/c/0/0')).toEqual(payload);
    expect(seen.length - before).toBe(2);
  });
});

/**
 * A two-member STORED archive whose first member's LOCAL header carries
 * `localExtra` bytes of extra field while its central record carries none.
 */
function storedArchiveWithLocalExtra(
  name: string,
  payload: Uint8Array,
  localExtra: number
): Uint8Array {
  const enc = new TextEncoder();
  const members = [
    { name: enc.encode(name), data: payload, extra: localExtra },
    { name: enc.encode('zarr.json'), data: enc.encode('{"zarr_format":3}'), extra: 0 },
  ];
  const localSize = members.reduce((t, m) => t + 30 + m.name.length + m.extra + m.data.length, 0);
  const centralSize = members.reduce((t, m) => t + 46 + m.name.length, 0);
  const bytes = new Uint8Array(localSize + centralSize + 22);
  const view = new DataView(bytes.buffer);
  const offsets: number[] = [];
  let at = 0;
  for (const m of members) {
    offsets.push(at);
    view.setUint32(at, 0x04034b50, true);
    view.setUint16(at + 4, 20, true);
    view.setUint32(at + 18, m.data.length, true);
    view.setUint32(at + 22, m.data.length, true);
    view.setUint16(at + 26, m.name.length, true);
    view.setUint16(at + 28, m.extra, true);
    bytes.set(m.name, at + 30);
    // An "unknown" extra block: id 0xcafe, payload of zeros.
    if (m.extra > 0) {
      view.setUint16(at + 30 + m.name.length, 0xcafe, true);
      view.setUint16(at + 32 + m.name.length, m.extra - 4, true);
    }
    bytes.set(m.data, at + 30 + m.name.length + m.extra);
    at += 30 + m.name.length + m.extra + m.data.length;
  }
  const centralOffset = at;
  members.forEach((m, i) => {
    view.setUint32(at, 0x02014b50, true);
    view.setUint16(at + 4, 20, true);
    view.setUint16(at + 6, 20, true);
    view.setUint32(at + 20, m.data.length, true);
    view.setUint32(at + 24, m.data.length, true);
    view.setUint16(at + 28, m.name.length, true);
    view.setUint32(at + 42, offsets[i], true);
    bytes.set(m.name, at + 46);
    at += 46 + m.name.length;
  });
  view.setUint32(at, 0x06054b50, true);
  view.setUint16(at + 8, members.length, true);
  view.setUint16(at + 10, members.length, true);
  view.setUint32(at + 12, centralSize, true);
  view.setUint32(at + 16, centralOffset, true);
  return bytes;
}
