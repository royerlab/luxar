/**
 * Telemetry-state resolution tests for cache-setup.ts.
 *
 * Mocks the heavy cache constructors so this test exercises ONLY the
 * branching logic that maps URL flags + app config to a
 * CacheTelemetryState. End-to-end behaviour of caches is covered
 * elsewhere (multi-level-caching-store, chunk-prefetcher).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

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

vi.mock('zarrita', () => ({
  registry: {},
  FetchStore: vi.fn().mockImplementation(() => ({})),
  // Stubbed for vitest strict-mock compatibility; cache-setup.ts
  // doesn't open zarr groups itself, but it imports through paths
  // that may transitively touch the zarr namespace.
  withMaybeConsolidatedMetadata: undefined,
}));

import { setupCaches } from '../../../../../data/scene-loader/cache/cache-setup';
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

  it('app config disables both cache + l0 → disabled-config', async () => {
    appConfig.cache.enabled = false;
    appConfig.cache.l0Enabled = false;

    const result = await setupCaches('http://example.com/scene.zarr/', {});

    expect(result.telemetryState).toEqual({ kind: 'disabled-config' });
    expect(result.cachingStore).toBeNull();
    expect(result.l0Cache).toBeNull();
  });

  it('L1/L2 enabled in config → enabled', async () => {
    appConfig.cache.enabled = true;
    appConfig.cache.l0Enabled = true;

    const result = await setupCaches('http://example.com/scene.zarr/', {});

    expect(result.telemetryState).toEqual({ kind: 'enabled' });
    expect(result.cachingStore).not.toBeNull();
    expect(result.l0Cache).not.toBeNull();
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
