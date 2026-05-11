/**
 * Telemetry-state resolution tests for cache-setup.ts.
 *
 * Mocks the heavy cache constructors so this test exercises ONLY the
 * branching logic that maps URL flags + app config to a
 * CacheTelemetryState. End-to-end behaviour of caches is covered
 * elsewhere (multi-level-caching-store, chunk-prefetcher).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock the cache module so setupCaches doesn't try to construct real
// MultiLevelCachingStore / DecompressedChunkCache / ChunkPrefetcher
// instances. We don't care about their behaviour here — only the
// telemetryState branch the function chooses.
vi.mock('../../../../cache', () => {
  return {
    MultiLevelCachingStore: vi.fn().mockImplementation(() => ({
      init: vi.fn().mockResolvedValue(undefined),
      setPrefetcher: vi.fn(),
      onInvalidate: vi.fn(),
    })),
    ChunkPrefetcher: vi.fn().mockImplementation(() => ({})),
    DecompressedChunkCache: vi.fn().mockImplementation(() => ({
      clear: vi.fn(),
    })),
  };
});

vi.mock('zarrita', () => ({
  registry: {},
  FetchStore: vi.fn().mockImplementation(() => ({})),
}));

import { setupCaches } from '../../../../data/scene-loader/cache-setup';
import { config as appConfig } from '../../../../config';

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

