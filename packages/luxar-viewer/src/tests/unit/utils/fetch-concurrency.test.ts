import { afterEach, describe, it, expect, vi } from 'vitest';
import { perfCounters } from '../../../profiling/perf-counters';
import {
  boundedConcurrencyStore,
  fetchLaneForKey,
  getFetchLaneLimit,
  getActiveFetchCount,
  noteFetchUrl,
  originOfUrl,
  resetFetchTransport,
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

/**
 * The bounded-concurrency gate is what prevents net::ERR_INSUFFICIENT_RESOURCES
 * when a large LOD level's selection fans out thousands of chunk fetches. Pins
 * the cap + the pass-through surface; fails on the pre-fix unbounded path.
 */
describe('fetch-concurrency gate', () => {
  afterEach(() => resetFetchTransport());

  it('classifies nested zarr documents into the metadata lane only', () => {
    for (const key of ['zarr.json', 'group/.zattrs', 'a/b/.zarray', '.zgroup', '.zmetadata']) {
      expect(fetchLaneForKey(key)).toBe('metadata');
    }
    expect(fetchLaneForKey('group/colors/c/3/0')).toBe('data');
    expect(fetchLaneForKey('group/zarr.json/c/0')).toBe('data');
  });

  it('only keys requests to origins that URL parsing can identify', () => {
    expect(originOfUrl('https://cdn.example.org/scene.zarr/zarr.json')).toBe(
      'https://cdn.example.org'
    );
    expect(originOfUrl('data:text/plain,chunk')).toBeUndefined();
    expect(originOfUrl('http://[')).toBeUndefined();
  });

  it('leaves lane limits alone for an invalid URL', () => {
    noteFetchUrl('http://[');
    expect(getFetchLaneLimit('data')).toBe(MAX_CONCURRENT_CHUNK_FETCHES);
    expect(getFetchLaneLimit('metadata')).toBe(MAX_CONCURRENT_METADATA_FETCHES);
  });

  it('rejects an already aborted request before taking a slot or calling it', async () => {
    const controller = new AbortController();
    const reason = new Error('superseded');
    controller.abort(reason);
    const fetch = vi.fn(async () => 1);

    await expect(withFetchGate(fetch, 'data', 'demand', undefined, controller.signal)).rejects.toBe(
      reason
    );
    expect(fetch).not.toHaveBeenCalled();
    expect(getActiveFetchCount('data')).toBe(0);
  });

  it('never exceeds MAX_CONCURRENT_CHUNK_FETCHES concurrent calls', async () => {
    let active = 0;
    let peak = 0;
    const releasers: Array<() => void> = [];
    const fire = () =>
      withFetchGate(() => {
        active += 1;
        peak = Math.max(peak, active);
        return new Promise<void>((resolve) =>
          releasers.push(() => {
            active -= 1;
            resolve();
          })
        );
      });

    const N = MAX_CONCURRENT_CHUNK_FETCHES * 4;
    const calls = Array.from({ length: N }, fire);
    await Promise.resolve();
    await Promise.resolve();
    expect(active).toBe(MAX_CONCURRENT_CHUNK_FETCHES);
    expect(peak).toBeLessThanOrEqual(MAX_CONCURRENT_CHUNK_FETCHES);

    while (releasers.length) {
      releasers.shift()!();
      await Promise.resolve();
      await Promise.resolve();
      expect(active).toBeLessThanOrEqual(MAX_CONCURRENT_CHUNK_FETCHES);
    }
    await Promise.all(calls);
    expect(peak).toBe(MAX_CONCURRENT_CHUNK_FETCHES);
  });

  it('boundedConcurrencyStore throttles get and passes other members through', async () => {
    let active = 0;
    let peak = 0;
    const releasers: Array<() => void> = [];
    const fake = {
      get: () => {
        active += 1;
        peak = Math.max(peak, active);
        return new Promise<Uint8Array>((resolve) =>
          releasers.push(() => {
            active -= 1;
            resolve(new Uint8Array(1));
          })
        );
      },
      contents: () => ['a'],
    };
    const store = boundedConcurrencyStore(fake);
    expect(store.contents()).toEqual(['a']);
    const calls = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES * 2 }, () => store.get());
    await Promise.resolve();
    await Promise.resolve();
    expect(peak).toBeLessThanOrEqual(MAX_CONCURRENT_CHUNK_FETCHES);
    while (releasers.length) {
      releasers.shift()!();
      await Promise.resolve();
      await Promise.resolve();
    }
    await Promise.all(calls);
  });

  it('lets metadata start while every data-body slot is occupied', async () => {
    const releaseData: Array<() => void> = [];
    const data = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES }, () =>
      withFetchGate(() => new Promise<void>((resolve) => releaseData.push(resolve)))
    );
    await Promise.resolve();

    let metadataStarted = false;
    const metadata = withFetchGate(async () => {
      metadataStarted = true;
    }, 'metadata');
    await Promise.resolve();

    expect(metadataStarted).toBe(true);
    await metadata;
    releaseData.forEach((release) => release());
    await Promise.all(data);
  });

  it('bounds metadata independently from data', async () => {
    let active = 0;
    let peak = 0;
    const release: Array<() => void> = [];
    const requests = Array.from({ length: MAX_CONCURRENT_METADATA_FETCHES * 2 }, () =>
      withFetchGate(() => {
        active += 1;
        peak = Math.max(peak, active);
        return new Promise<void>((resolve) =>
          release.push(() => {
            active -= 1;
            resolve();
          })
        );
      }, 'metadata')
    );
    await Promise.resolve();
    await Promise.resolve();

    expect(active).toBe(MAX_CONCURRENT_METADATA_FETCHES);
    while (release.length) {
      release.shift()!();
      await Promise.resolve();
      await Promise.resolve();
    }
    await Promise.all(requests);
    expect(peak).toBe(MAX_CONCURRENT_METADATA_FETCHES);
  });

  it("fits an http: origin's lanes into its six HTTP/1.1 sockets once seen", () => {
    const lan = 'http://10.0.0.55:8001';
    const cdn = 'https://cdn.example.org';
    expect(HTTP1_MAX_CONCURRENT_CHUNK_FETCHES + HTTP1_MAX_CONCURRENT_METADATA_FETCHES).toBe(6);
    noteFetchUrl(`${cdn}/scene.luxar.zarr/zarr.json`);
    expect(getFetchLaneLimit('data', cdn)).toBe(MAX_CONCURRENT_CHUNK_FETCHES);
    noteFetchUrl(`${lan}/data/Backdrop/part_3/zarr.json`);
    expect(getFetchLaneLimit('data', lan)).toBe(HTTP1_MAX_CONCURRENT_CHUNK_FETCHES);
    expect(getFetchLaneLimit('metadata', lan)).toBe(HTTP1_MAX_CONCURRENT_METADATA_FETCHES);
    // The origin never widens again within the session; others never narrowed.
    noteFetchUrl(`${cdn}/other/zarr.json`);
    expect(getFetchLaneLimit('data', lan)).toBe(HTTP1_MAX_CONCURRENT_CHUNK_FETCHES);
    expect(getFetchLaneLimit('data', cdn)).toBe(MAX_CONCURRENT_CHUNK_FETCHES);
  });

  it('drains in-flight leases down to a cap that shrank under them', async () => {
    let active = 0;
    let peak = 0;
    const release: Array<() => void> = [];
    const lan = 'http://localhost:8005';
    const fire = () =>
      withFetchGate(
        () => {
          active += 1;
          peak = Math.max(peak, active);
          return new Promise<void>((resolve) =>
            release.push(() => {
              active -= 1;
              resolve();
            })
          );
        },
        'data',
        'demand',
        lan
      );
    const calls = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES * 2 }, fire);
    await Promise.resolve();
    await Promise.resolve();
    expect(active).toBe(MAX_CONCURRENT_CHUNK_FETCHES);

    noteFetchUrl(`${lan}/scene.luxar.zarr/zarr.json`);
    peak = 0;
    while (release.length) {
      release.shift()!();
      await Promise.resolve();
      await Promise.resolve();
    }
    await Promise.all(calls);
    // Leases started after the shrink never pushed concurrency past the new cap.
    expect(peak).toBeGreaterThan(0);
    expect(peak).toBeLessThanOrEqual(HTTP1_MAX_CONCURRENT_CHUNK_FETCHES);
    expect(active).toBe(0);
  });
});

describe('fetch-concurrency perf counters', () => {
  it('tallies metadata-lane requests, highWater and queued wait only', async () => {
    perfCounters.reset();
    let nowMs = 0;
    const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => nowMs);
    try {
      const releasers: Array<() => void> = [];
      const fire = () =>
        withFetchGate(() => new Promise<void>((resolve) => releasers.push(resolve)), 'metadata');
      const N = MAX_CONCURRENT_METADATA_FETCHES + 2;
      const calls = Array.from({ length: N }, fire);
      await Promise.resolve();
      expect(perfCounters.get('fetch.metadata.requests')).toBe(N);
      expect(perfCounters.get('fetch.metadata.highWater')).toBe(MAX_CONCURRENT_METADATA_FETCHES);
      expect(perfCounters.get('fetch.metadata.queueWaitMs')).toBe(0);

      // The two queued calls start 7 ms after they were enqueued.
      nowMs = 7;
      while (releasers.length) {
        releasers.shift()!();
        await Promise.resolve();
        await Promise.resolve();
      }
      await Promise.all(calls);
      expect(perfCounters.get('fetch.metadata.queueWaitMs')).toBe(14);
      expect(perfCounters.get('fetch.metadata.highWater')).toBe(MAX_CONCURRENT_METADATA_FETCHES);
      expect(perfCounters.get('fetch.data.requests')).toBe(0);
    } finally {
      nowSpy.mockRestore();
    }
  });
});
