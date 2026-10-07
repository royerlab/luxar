/**
 * A zip member read honours its caller's abort signal (A11).
 *
 * A superseded view on a `.zarr.zip` used to keep its member range GETs queued
 * in the fetch gate (and in flight) at full priority: `LuxarZipStore.get`
 * dropped the signal and `readMember` fetched its window with none, so only a
 * store `dispose()` could cancel it.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { zipSync, strToU8 } from 'fflate';
import { LuxarZipStore } from '../../../../data/zip/store';
import { ZipChunkSource } from '../../../../cache/chunk-source/zip-chunk-source';
import {
  FetchPriorityCell,
  resetFetchTransport,
  withFetchGate,
} from '../../../../utils/fetch-concurrency';
import { config } from '../../../../config';

/** The gate's widths, as configured (`config.dataLoading.network.fetchGate`). */
const { maxChunkFetches: MAX_CONCURRENT_CHUNK_FETCHES } = config.dataLoading.network.fetchGate;

const URL_ = 'https://example.com/abort.luxar.zarr.zip';

function noise(length: number): Uint8Array {
  const out = new Uint8Array(length);
  let x = 3;
  for (let i = 0; i < length; i++) {
    x = (Math.imul(x, 1103515245) + 12345) >>> 0;
    out[i] = x >>> 24;
  }
  return out;
}

/** A strict range server over a real archive; records every Range it serves. */
function serve(archive: Uint8Array) {
  const ranges: string[] = [];
  const fetchMock = vi.fn(async (_url: string, init: RequestInit = {}) => {
    const range = (init.headers as Record<string, string> | undefined)?.Range ?? '';
    ranges.push(range);
    const size = archive.length;
    const m = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (!m) return new Response(archive.slice(), { status: 200 });
    const start = m[1] === '' ? Math.max(0, size - Number(m[2])) : Number(m[1]);
    const end = m[1] === '' ? size - 1 : Math.min(size - 1, Number(m[2]));
    return new Response(archive.slice(start, end + 1), {
      status: 206,
      headers: { 'content-range': `bytes ${start}-${end}/${size}`, etag: '"v1"' },
    });
  });
  return { fetchMock, ranges };
}

function archive(): Uint8Array {
  return zipSync(
    {
      'points/c/0/0': [noise(200_000), { level: 0 }],
      'zarr.json': strToU8('{"zarr_format":3,"node_type":"group"}'),
    },
    { level: 0 }
  );
}

/** Occupy every data-lane slot until released. */
function fillDataLane(): () => Promise<void> {
  const releases: Array<() => void> = [];
  const done = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES }, () =>
    withFetchGate(() => new Promise<void>((resolve) => releases.push(resolve)), 'data')
  );
  return async () => {
    for (const release of releases) release();
    await Promise.all(done);
  };
}

async function flush(): Promise<void> {
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

afterEach(() => {
  resetFetchTransport();
  vi.unstubAllGlobals();
});

describe('zip member reads and the caller signal', () => {
  it('an aborted member read leaves the gate queue and never fetches its window', async () => {
    const { fetchMock, ranges } = serve(archive());
    vi.stubGlobal('fetch', fetchMock);
    const store = new LuxarZipStore(URL_);
    const source = new ZipChunkSource(URL_, store);
    expect(await store.has('/zarr.json')).toBe(true);
    const opened = ranges.length;

    const release = fillDataLane();
    await flush();
    const controller = new AbortController();
    let settled = false;
    const read = source
      .get('/points/c/0/0', controller.signal, { priority: new FetchPriorityCell('speculative') })
      .finally(() => (settled = true));
    await flush();
    controller.abort();
    await flush();
    const settledBeforeRelease = settled;
    await release();
    await read;
    await flush();

    expect(settledBeforeRelease).toBe(true);
    expect(await read).toEqual({ kind: 'aborted' });
    expect(ranges.length).toBe(opened);
  });

  it('a store dispose still cancels a member read that carries a caller signal', async () => {
    const { fetchMock, ranges } = serve(archive());
    vi.stubGlobal('fetch', fetchMock);
    const store = new LuxarZipStore(URL_);
    expect(await store.has('/zarr.json')).toBe(true);
    const opened = ranges.length;

    const release = fillDataLane();
    await flush();
    const caller = new AbortController();
    const read = store.get('/points/c/0/0', caller.signal);
    const outcome = read.then(
      () => 'resolved',
      (error: unknown) => (error as Error).name
    );
    await flush();
    store.dispose();
    await flush();
    await release();

    expect(await outcome).toBe('AbortError');
    await flush();
    expect(ranges.length).toBe(opened);
  });
});
