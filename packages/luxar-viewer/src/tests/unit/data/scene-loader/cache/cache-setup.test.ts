/**
 * Telemetry-state resolution tests for cache-setup.ts.
 *
 * Mocks the heavy cache constructors so this test exercises ONLY the
 * branching logic that maps URL flags + app config to a
 * CacheTelemetryState. End-to-end behaviour of caches is covered
 * elsewhere (multi-level-caching-store, chunk-prefetcher).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

import { MultiLevelCachingStore } from '../../../../../cache/multi-level-caching-store';

// Mock ONLY the I/O-heavy tiers (OPFS/network). NOTE: cache-setup.ts imports
// the CONCRETE modules, not the `cache` barrel — the previous barrel mock was
// silently ineffective and these tests ran against the real store. The
// lightweight in-memory caches (SliceCache, DecompressedChunkCache) stay real
// so the wiring tests below exercise actual clear/invalidate behavior.
const invalidateListeners: Array<() => void> = [];
vi.mock('../../../../../cache/multi-level-caching-store', () => {
  return {
    MultiLevelCachingStore: vi.fn().mockImplementation(() => ({
      init: vi.fn().mockResolvedValue(undefined),
      setPrefetcher: vi.fn(),
      onInvalidate: vi.fn((cb: () => void) => {
        invalidateListeners.push(cb);
      }),
    })),
  };
});
vi.mock('../../../../../cache/chunk-prefetcher', () => ({
  ChunkPrefetcher: vi.fn().mockImplementation(() => ({})),
}));

vi.mock('../../../../../data/zarr', () => ({
  createStoreForUrl: vi.fn().mockImplementation(() => ({})),
}));

import { setupCaches } from '../../../../../data/scene-loader/cache/cache-setup';
import { deviceClassPoolBytes } from '../../../../../cache/heap-budget';
import { config as appConfig } from '../../../../../config';

describe('setupCaches — cache telemetry state resolution', () => {
  // Snapshot/restore the relevant config flags between tests so each
  // case sees a known starting point.
  let originalEnabled: boolean;
  let originalL0Enabled: boolean;

  beforeEach(() => {
    originalEnabled = appConfig.cache.enabled;
    originalL0Enabled = appConfig.cache.l0Enabled;
  });

  afterEach(() => {
    appConfig.cache.enabled = originalEnabled;
    appConfig.cache.l0Enabled = originalL0Enabled;
  });

  it('a .zarr.zip is cached like its directory twin', async () => {
    // Phase 2 inverts what Phase 1 pinned here. MultiLevelCachingStore now takes
    // a ChunkSource instead of building chunk URLs from a base, so an archive
    // member is reachable and a zipped store gets the full tier stack. Caching
    // earns MORE on an archive: ~2 requests per member, and repeat reads cannot
    // fall back to the browser HTTP cache the way a per-chunk URL can.
    appConfig.cache.enabled = true;
    appConfig.cache.l0Enabled = true;

    const zipped = await setupCaches('http://example.com/scene.luxar.zarr.zip', {});
    expect(zipped.cachingStore).not.toBeNull();
    expect(zipped.l0Cache).not.toBeNull();

    const directory = await setupCaches('http://example.com/scene.luxar.zarr/', {});
    expect(directory.cachingStore).not.toBeNull();
  });

  it('?no-cache resolves to disabled-no-cache regardless of app config', async () => {
    appConfig.cache.enabled = true;
    appConfig.cache.l0Enabled = true;

    const result = await setupCaches('http://example.com/scene.zarr/', {
      noCache: true,
    });

    expect(result.telemetryState).toEqual({ kind: 'disabled-no-cache' });
    expect(result.cachingStore).toBeNull();
    expect(result.l0Cache).toBeNull();
  });

  it('app config disables cache + l0 + slice → disabled-config', async () => {
    appConfig.cache.enabled = false;
    appConfig.cache.l0Enabled = false;
    const prevSlice = appConfig.cache.sliceCacheEnabled;
    appConfig.cache.sliceCacheEnabled = false;
    try {
      const result = await setupCaches('http://example.com/scene.zarr/', {});
      expect(result.telemetryState).toEqual({ kind: 'disabled-config' });
      expect(result.cachingStore).toBeNull();
      expect(result.l0Cache).toBeNull();
    } finally {
      appConfig.cache.sliceCacheEnabled = prevSlice;
    }
  });

  it('S-cache-only configuration (cache + l0 off, slice on) → enabled', async () => {
    appConfig.cache.enabled = false;
    appConfig.cache.l0Enabled = false;
    const prevSlice = appConfig.cache.sliceCacheEnabled;
    appConfig.cache.sliceCacheEnabled = true;
    try {
      const result = await setupCaches('http://example.com/scene.zarr/', {});
      // An active S-cache still serves slice revisits and reports live
      // stats — the monitor must not claim caching is disabled.
      expect(result.telemetryState).toEqual({ kind: 'enabled' });
      expect(result.sliceCache).not.toBeNull();
    } finally {
      appConfig.cache.sliceCacheEnabled = prevSlice;
    }
  });

  it('L1/L2 enabled in config → enabled', async () => {
    appConfig.cache.enabled = true;
    appConfig.cache.l0Enabled = true;

    const result = await setupCaches('http://example.com/scene.zarr/', {});

    expect(result.telemetryState).toEqual({ kind: 'enabled' });
    expect(result.cachingStore).not.toBeNull();
    expect(result.l0Cache).not.toBeNull();
  });

  it('hands the caching store a zip SOURCE for an archive and a bare URL otherwise', async () => {
    // The two must never be interchangeable. A zipped store and its unzipped
    // twin cache the same decoded bytes but have different key namespaces, and
    // the OPFS bucket is derived from the source's identity — so passing a bare
    // URL for an archive would both fail to read it and risk sharing a bucket.
    appConfig.cache.enabled = true;
    appConfig.cache.l0Enabled = true;

    await setupCaches('http://example.com/scene.luxar.zarr.zip', {});
    const zippedArg = vi.mocked(MultiLevelCachingStore).mock.calls.at(-1)?.[0];

    await setupCaches('http://example.com/scene.zarr/', {});
    const directoryArg = vi.mocked(MultiLevelCachingStore).mock.calls.at(-1)?.[0];

    expect(typeof zippedArg).toBe('object');
    expect(zippedArg).toHaveProperty('identity', 'http://example.com/scene.luxar.zarr.zip');
    expect(directoryArg).toBe('http://example.com/scene.zarr/');
  });

  it('feeds the ?cacheBudgetMB pool through to the L2 write-queue byte cap', async () => {
    // The write queue's retained-byte allowance resolves from the same memory
    // model as the tiers, so the explicit pool (`?cacheBudgetMB=` / the native
    // launcher — the WKWebView path, where the heap is unmeasurable) must
    // reach it. Dropping either argument at the construction site leaves the
    // rest of this suite green while that override silently stops applying.
    appConfig.cache.enabled = true;
    appConfig.cache.l0Enabled = true;

    const capFor = async (cacheBudgetMB?: number): Promise<number | undefined> => {
      await setupCaches('http://example.com/scene.zarr/', { cacheBudgetMB });
      return vi.mocked(MultiLevelCachingStore).mock.calls.at(-1)?.[1]?.opfsWriteQueueMaxBytes;
    };

    // A pool becomes a remainder (× 0.4/0.6) of which the queue takes a
    // quarter: 2048MiB → 341.33MiB, 768MiB → 128MiB. TWO distinct overrides,
    // because a single one cannot discriminate: `inferDeviceClass` can resolve
    // this environment to the 2048MB desktop pool, in which case dropping the
    // override argument entirely would still yield the 2048MB number.
    expect(await capFor(2048)).toBe(357_913_941);
    expect(await capFor(768)).toBe(134_217_728);

    // With no override and no measurable heap it falls to the device-class
    // pool, or to the fixed 256MB where there are no device signals at all.
    const fallbackPool = deviceClassPoolBytes();
    expect(await capFor(undefined)).toBe(
      fallbackPool === undefined ? 256 * 1024 * 1024 : Math.floor(fallbackPool * (0.4 / 0.6) * 0.25)
    );
  });

  it('L0-only configuration (l1/l2 off, l0 on) → enabled', async () => {
    appConfig.cache.enabled = false;
    appConfig.cache.l0Enabled = true;

    const result = await setupCaches('http://example.com/scene.zarr/', {});

    // L0-only counts as enabled — UI keys off the top-level state and
    // shows L0 stats independently.
    expect(result.telemetryState).toEqual({ kind: 'enabled' });
    expect(result.cachingStore).toBeNull();
    expect(result.l0Cache).not.toBeNull();
  });
});

// Regression tests (deep-double-check): the SliceCache wiring — creation
// gating across sliceCacheEnabled / ?no-cache / ?no-slice-cache, and the
// content-hash invalidation registration — previously had NO test at all
// (mutating the gate condition or deleting the onInvalidate registration
// passed the whole suite).
describe('setupCaches — SliceCache gating + invalidation wiring', () => {
  let originalEnabled: boolean;
  let originalL0Enabled: boolean;
  let originalSliceEnabled: boolean;

  beforeEach(() => {
    originalEnabled = appConfig.cache.enabled;
    originalL0Enabled = appConfig.cache.l0Enabled;
    originalSliceEnabled = appConfig.cache.sliceCacheEnabled;
    invalidateListeners.length = 0;
  });

  afterEach(() => {
    appConfig.cache.enabled = originalEnabled;
    appConfig.cache.l0Enabled = originalL0Enabled;
    appConfig.cache.sliceCacheEnabled = originalSliceEnabled;
  });

  it('creates the SliceCache when enabled and no disabling flag is set', async () => {
    appConfig.cache.enabled = true;
    appConfig.cache.sliceCacheEnabled = true;
    const result = await setupCaches('http://example.com/scene.zarr/', {});
    expect(result.sliceCache).not.toBeNull();
  });

  it('?no-slice-cache disables ONLY the SliceCache (L0/L1/L2 stay on)', async () => {
    appConfig.cache.enabled = true;
    appConfig.cache.l0Enabled = true;
    appConfig.cache.sliceCacheEnabled = true;
    const result = await setupCaches('http://example.com/scene.zarr/', {
      noSliceCache: true,
    });
    expect(result.sliceCache).toBeNull();
    expect(result.l0Cache).not.toBeNull();
    expect(result.cachingStore).not.toBeNull();
  });

  it('?no-opfs keeps every other tier: caching store, L0 and SliceCache all construct', async () => {
    appConfig.cache.enabled = true;
    appConfig.cache.l0Enabled = true;
    appConfig.cache.sliceCacheEnabled = true;
    const result = await setupCaches('http://example.com/scene.zarr/', {
      noOpfs: true,
    });
    // The L2 skip happens INSIDE MultiLevelCachingStore.init() (l2Store
    // stays null; covered by the store's own unit tests). From the
    // outside: the store must still construct WITH the flag, and every
    // in-memory tier stays on.
    expect(result.cachingStore).not.toBeNull();
    const { MultiLevelCachingStore } =
      await import('../../../../../cache/multi-level-caching-store');
    const ctorOptions = vi.mocked(MultiLevelCachingStore).mock.calls.at(-1)?.[1];
    expect(ctorOptions).toMatchObject({ noOpfs: true });
    expect(result.l0Cache).not.toBeNull();
    expect(result.sliceCache).not.toBeNull();
  });

  it('?no-cache disables the SliceCache along with every other tier', async () => {
    appConfig.cache.enabled = true;
    appConfig.cache.sliceCacheEnabled = true;
    const result = await setupCaches('http://example.com/scene.zarr/', {
      noCache: true,
    });
    expect(result.sliceCache).toBeNull();
  });

  it('config sliceCacheEnabled=false disables the SliceCache', async () => {
    appConfig.cache.enabled = true;
    appConfig.cache.sliceCacheEnabled = false;
    const result = await setupCaches('http://example.com/scene.zarr/', {});
    expect(result.sliceCache).toBeNull();
  });

  it('registers a content-hash invalidation listener that empties the SliceCache', async () => {
    appConfig.cache.enabled = true;
    appConfig.cache.sliceCacheEnabled = true;
    const result = await setupCaches('http://example.com/scene.zarr/', {});
    const sc = result.sliceCache!;
    sc.set('node|sig', { payload: [1, 2, 3], bytes: 24 });
    expect(sc.getStats().count).toBe(1);

    // Fire every registered invalidation callback (content-hash bump).
    for (const cb of invalidateListeners) cb();

    expect(sc.getStats().count).toBe(0);
  });

  it('does NOT register a SliceCache invalidation listener when L1/L2 are off (documented gap)', async () => {
    appConfig.cache.enabled = false;
    appConfig.cache.l0Enabled = false;
    appConfig.cache.sliceCacheEnabled = true;
    const result = await setupCaches('http://example.com/scene.zarr/', {});
    expect(result.sliceCache).not.toBeNull();
    expect(invalidateListeners.length).toBe(0);
  });
});
