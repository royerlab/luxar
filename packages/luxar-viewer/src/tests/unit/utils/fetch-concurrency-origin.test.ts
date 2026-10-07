/**
 * The HTTP/1.1 cap is per ORIGIN, and the no-cache store wrapper names its
 * origin and signal to the gate.
 *
 * One plain `http:` URL (a local kiosk backdrop, `luxar serve` on the LAN) used
 * to narrow EVERY origin's lanes to the six-socket HTTP/1.1 budget for the rest
 * of the session; and `boundedConcurrencyStore` gated without an origin (so an
 * h2 host never got its wide lane) and without the caller's signal (so a
 * superseded no-cache read kept its queue place).
 */

import { afterEach, describe, it, expect } from 'vitest';
import {
  boundedConcurrencyStore,
  getFetchLaneLimit,
  noteFetchUrl,
  noteOriginProtocol,
  resetFetchTransport,
  resetOriginProtocols,
  withFetchGate,
} from '../../../utils/fetch-concurrency';
import { config } from '../../../config';

/** The gate's widths, as configured (`config.dataLoading.network.fetchGate`). */
const {
  http1MaxChunkFetches: HTTP1_MAX_CONCURRENT_CHUNK_FETCHES,
  http1MaxMetadataFetches: HTTP1_MAX_CONCURRENT_METADATA_FETCHES,
  maxChunkFetches: MAX_CONCURRENT_CHUNK_FETCHES,
  maxMetadataFetches: MAX_CONCURRENT_METADATA_FETCHES,
} = config.dataLoading.network.fetchGate;

const LAN = 'http://10.0.0.55:8001';
const CDN = 'https://cdn.example.org';

/** A gated call whose body stays open until released. */
function holder(origin: string | undefined, lane: 'data' | 'metadata' = 'data') {
  let release!: () => void;
  let started = false;
  const released = new Promise<void>((resolve) => (release = resolve));
  const done = withFetchGate(
    () => {
      started = true;
      return released;
    },
    lane,
    'demand',
    origin
  );
  return {
    done,
    release: () => release(),
    get started() {
      return started;
    },
  };
}

async function flush(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

async function drain(calls: Array<ReturnType<typeof holder>>): Promise<void> {
  for (const h of calls) h.release();
  await Promise.all(calls.map((h) => h.done));
}

afterEach(() => {
  resetOriginProtocols();
  resetFetchTransport();
});

describe('HTTP/1.1 lane cap is per origin', () => {
  it('an http: URL narrows only its own origin', () => {
    noteFetchUrl(`${LAN}/data/Backdrop/part_3/zarr.json`);
    expect(getFetchLaneLimit('data', LAN)).toBe(HTTP1_MAX_CONCURRENT_CHUNK_FETCHES);
    expect(getFetchLaneLimit('metadata', LAN)).toBe(HTTP1_MAX_CONCURRENT_METADATA_FETCHES);
    expect(getFetchLaneLimit('data', CDN)).toBe(MAX_CONCURRENT_CHUNK_FETCHES);
    expect(getFetchLaneLimit('metadata', CDN)).toBe(MAX_CONCURRENT_METADATA_FETCHES);
  });

  it('a TLS origin keeps its full lane while the http: origin is held to its sockets', async () => {
    noteFetchUrl(`${LAN}/scene.luxar.zarr/zarr.json`);
    const tls = Array.from({ length: 10 }, () => holder(CDN));
    const lan = Array.from({ length: HTTP1_MAX_CONCURRENT_CHUNK_FETCHES + 3 }, () => holder(LAN));
    await flush();
    const tlsStarted = tls.filter((h) => h.started).length;
    const lanStarted = lan.filter((h) => h.started).length;
    await drain([...tls, ...lan]);

    expect(tlsStarted).toBe(10);
    expect(lanStarted).toBe(HTTP1_MAX_CONCURRENT_CHUNK_FETCHES);
  });
});

describe('boundedConcurrencyStore names its origin and signal', () => {
  it('gives a multiplexed no-cache store its wide lane', async () => {
    const h2 = 'https://h2.example';
    noteOriginProtocol(h2, 'h2');
    let active = 0;
    let peak = 0;
    const releasers: Array<() => void> = [];
    const fake = {
      url: new URL(`${h2}/scene.luxar.zarr`),
      get: (_key: string) => {
        active += 1;
        peak = Math.max(peak, active);
        return new Promise<Uint8Array>((resolve) =>
          releasers.push(() => {
            active -= 1;
            resolve(new Uint8Array(1));
          })
        );
      },
    };
    const store = boundedConcurrencyStore(fake);
    const calls = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES + 8 }, (_, i) =>
      store.get(`points/c/${i}`)
    );
    await flush();
    const peakBeforeRelease = peak;
    while (releasers.length) {
      releasers.shift()!();
      await flush();
    }
    await Promise.all(calls);
    expect(peakBeforeRelease).toBe(MAX_CONCURRENT_CHUNK_FETCHES + 8);
  });

  it('frees the queue place of a no-cache read whose caller aborts', async () => {
    const busy = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES }, () => holder(undefined));
    await flush();
    let ran = false;
    const fake = {
      url: 'https://example.org/scene.luxar.zarr',
      get: async (_key: string, _opts?: { signal?: AbortSignal }) => {
        ran = true;
        return new Uint8Array(1);
      },
    };
    const store = boundedConcurrencyStore(fake);
    const controller = new AbortController();
    let failure: unknown;
    const read = store
      .get('points/c/0', { signal: controller.signal })
      .catch((error: unknown) => (failure = error));
    controller.abort();
    await flush();
    const nameBeforeRelease = (failure as { name?: string } | undefined)?.name;

    await drain(busy);
    await read;
    await flush();
    expect(nameBeforeRelease).toBe('AbortError');
    expect(ran).toBe(false);
  });
});
