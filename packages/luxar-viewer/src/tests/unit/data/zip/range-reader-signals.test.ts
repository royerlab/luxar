/**
 * The range reader's probe signals and length fallback.
 *
 * - The identity probe merged the caller's signal with its timeout by adding
 *   `abort` listeners that were never removed (one leaked closure per probe on
 *   a long-lived caller signal), and aborted the merge WITHOUT the caller's
 *   reason.
 * - The `getLength` fallback (`HEAD`, then `bytes=0-0`) used bare `fetch`: no
 *   fetch gate, no retry budget, no `cache: 'no-store'`, no timeout.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { LuxarHttpRangeReader } from '../../../../data/zip/range-reader';
import { resetFetchTransport, withFetchGate } from '../../../../utils/fetch-concurrency';
import { config } from '../../../../config';

/** The gate's widths, as configured (`config.dataLoading.network.fetchGate`). */
const { maxMetadataFetches: MAX_CONCURRENT_METADATA_FETCHES } =
  config.dataLoading.network.fetchGate;

const URL_ = 'https://example.com/signals.luxar.zarr.zip';

afterEach(() => {
  resetFetchTransport();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

async function flush(): Promise<void> {
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

/** Count `abort` listeners added to and removed from `signal`. */
function trackAbortListeners(signal: AbortSignal): () => number {
  let live = 0;
  const add = signal.addEventListener.bind(signal);
  const remove = signal.removeEventListener.bind(signal);
  vi.spyOn(signal, 'addEventListener').mockImplementation((type, listener, options) => {
    if (type === 'abort') live += 1;
    add(type, listener, options);
  });
  vi.spyOn(signal, 'removeEventListener').mockImplementation((type, listener, options) => {
    if (type === 'abort') live -= 1;
    remove(type, listener, options);
  });
  return () => live;
}

describe('LuxarHttpRangeReader.probeIdentity — caller signal', () => {
  it('leaves no abort listener behind on the caller signal', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(new Uint8Array(4), {
            status: 206,
            headers: { etag: '"v1"', 'content-range': 'bytes 0-3/4' },
          })
      )
    );
    const caller = new AbortController();
    const live = trackAbortListeners(caller.signal);

    const reader = new LuxarHttpRangeReader(URL_);
    expect(await reader.probeIdentity(caller.signal, 5_000)).toBe('etag:"v1"');
    expect(live()).toBe(0);
  });

  it("forwards the caller's abort reason to the probe request", async () => {
    const reasons: unknown[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(
        (_url: string, init: RequestInit = {}) =>
          new Promise<Response>((_, reject) => {
            const signal = init.signal!;
            const fail = (): void => {
              reasons.push(signal.reason);
              reject(signal.reason);
            };
            if (signal.aborted) fail();
            else signal.addEventListener('abort', fail);
          })
      )
    );
    const caller = new AbortController();
    const reason = new Error('view superseded');
    const reader = new LuxarHttpRangeReader(URL_);
    const probe = reader.probeIdentity(caller.signal, 60_000);
    await flush();
    caller.abort(reason);

    expect(await probe).toBeNull();
    expect(reasons[0]).toBe(reason);
  });
});

describe('LuxarHttpRangeReader.getLength — gated like every other request', () => {
  it('waits for a metadata-lane slot and bypasses the HTTP cache', async () => {
    const inits: RequestInit[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init: RequestInit = {}) => {
        inits.push(init);
        return new Response(null, { status: 200, headers: { 'content-length': '4096' } });
      })
    );
    const releases: Array<() => void> = [];
    const busy = Array.from({ length: MAX_CONCURRENT_METADATA_FETCHES }, () =>
      withFetchGate(() => new Promise<void>((resolve) => releases.push(resolve)), 'metadata')
    );
    await flush();

    const reader = new LuxarHttpRangeReader(URL_);
    const length = reader.getLength();
    await flush();
    const requestsWhileFull = inits.length;
    for (const release of releases) release();
    await Promise.all(busy);

    expect(await length).toBe(4096);
    expect(requestsWhileFull).toBe(0);
    expect(inits[0]).toMatchObject({ method: 'HEAD', cache: 'no-store' });
  });
});
