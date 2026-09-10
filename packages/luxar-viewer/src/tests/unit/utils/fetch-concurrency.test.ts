import { describe, it, expect } from 'vitest';
import {
  boundedConcurrencyStore,
  fetchLaneForKey,
  withFetchGate,
  MAX_CONCURRENT_CHUNK_FETCHES,
  MAX_CONCURRENT_METADATA_FETCHES,
} from '../../../utils/fetch-concurrency';

/**
 * The bounded-concurrency gate is what prevents net::ERR_INSUFFICIENT_RESOURCES
 * when a large LOD level's selection fans out thousands of chunk fetches. Pins
 * the cap + the pass-through surface; fails on the pre-fix unbounded path.
 */
describe('fetch-concurrency gate', () => {
  it('classifies nested zarr documents into the metadata lane only', () => {
    for (const key of ['zarr.json', 'group/.zattrs', 'a/b/.zarray', '.zgroup', '.zmetadata']) {
      expect(fetchLaneForKey(key)).toBe('metadata');
    }
    expect(fetchLaneForKey('group/colors/c/3/0')).toBe('data');
    expect(fetchLaneForKey('group/zarr.json/c/0')).toBe('data');
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
});
