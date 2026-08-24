import { describe, expect, it, vi } from 'vitest';

import { MonitorProviderRegistry } from '../../../../ui/data-loading-monitor/providers';

describe('MonitorProviderRegistry', () => {
  it('starts with empty scene-scoped provider state', () => {
    const providers = new MonitorProviderRegistry(vi.fn());

    expect(providers.cacheStatsProvider).toBeNull();
    expect(providers.l0CacheProvider).toBeNull();
    expect(providers.sliceCacheProvider).toBeNull();
    expect(providers.cacheTelemetryState).toBeUndefined();
    expect(providers.gpuBufferPoolProvider).toBeNull();
    expect(providers.profiler).toBeNull();
    expect(providers.lodProgressProvider).toBeNull();
    expect(providers.failedLoadsProvider).toBeNull();
    expect(providers.lodStates).toEqual(new Map());
    expect(providers.drawOrderProvider).toBeNull();
    expect(providers.drawOrderStates).toEqual(new Map());
    expect(providers.accumulatorProviders).toEqual({
      points: null,
      lines: null,
      gsplats: null,
    });
  });

  it('owns provider registration and preserves structure-dirty semantics', () => {
    const markStructureDirty = vi.fn();
    const providers = new MonitorProviderRegistry(markStructureDirty);
    const cacheStatsProvider = { getStats: vi.fn() };
    const l0CacheProvider = { getStats: vi.fn(), clear: vi.fn() };
    const sliceCacheProvider = { getStats: vi.fn(), clear: vi.fn() };
    const gpuBufferPoolProvider = { getStats: vi.fn() };
    const profiler = { getTimings: vi.fn() };
    const lodProgressProvider = { getLODStates: vi.fn() };
    const failedLoadsProvider = { getFailedPaths: vi.fn(), retryAll: vi.fn() };
    const drawOrderProvider = { getDrawOrderStates: vi.fn() };
    const accumulatorProvider = { getStats: vi.fn() };

    providers.setCacheStatsProvider(cacheStatsProvider as never);
    providers.setL0CacheProvider(l0CacheProvider as never);
    providers.setSliceCacheProvider(sliceCacheProvider as never);
    providers.setCacheTelemetryState({ kind: 'disabled-no-cache' });
    providers.setGPUBufferPoolProvider(gpuBufferPoolProvider as never);
    providers.setProfiler(profiler as never);
    providers.setLODProgressProvider(lodProgressProvider as never);
    providers.setFailedLoadsProvider(failedLoadsProvider as never);
    providers.setDrawOrderProvider(drawOrderProvider as never);
    providers.setAccumulatorProvider('points', accumulatorProvider as never);

    expect(providers.cacheStatsProvider).toBe(cacheStatsProvider);
    expect(providers.l0CacheProvider).toBe(l0CacheProvider);
    expect(providers.sliceCacheProvider).toBe(sliceCacheProvider);
    expect(providers.cacheTelemetryState).toEqual({ kind: 'disabled-no-cache' });
    expect(providers.gpuBufferPoolProvider).toBe(gpuBufferPoolProvider);
    expect(providers.profiler).toBe(profiler);
    expect(providers.lodProgressProvider).toBe(lodProgressProvider);
    expect(providers.failedLoadsProvider).toBe(failedLoadsProvider);
    expect(providers.drawOrderProvider).toBe(drawOrderProvider);
    expect(providers.accumulatorProviders.points).toBe(accumulatorProvider);
    expect(markStructureDirty).toHaveBeenCalledTimes(5);
  });

  it('clears stale snapshots and restores fresh empty containers on reset', () => {
    const markStructureDirty = vi.fn();
    const providers = new MonitorProviderRegistry(markStructureDirty);
    const previousLodStates = providers.lodStates;
    const previousDrawOrderStates = providers.drawOrderStates;
    const previousAccumulatorProviders = providers.accumulatorProviders;

    providers.setLODProgressProvider({ getLODStates: vi.fn() } as never);
    providers.lodStates.set('/node', {} as never);
    providers.setDrawOrderProvider({ getDrawOrderStates: vi.fn() } as never);
    providers.drawOrderStates.set('/node', {} as never);
    providers.setAccumulatorProvider('lines', { getStats: vi.fn() } as never);
    providers.setCacheTelemetryState({ kind: 'enabled' });
    markStructureDirty.mockClear();

    providers.resetSceneProviders();

    expect(providers.lodStates).toEqual(new Map());
    expect(providers.lodStates).not.toBe(previousLodStates);
    expect(providers.drawOrderStates).toEqual(new Map());
    expect(providers.drawOrderStates).not.toBe(previousDrawOrderStates);
    expect(providers.accumulatorProviders).toEqual({
      points: null,
      lines: null,
      gsplats: null,
    });
    expect(providers.accumulatorProviders).not.toBe(previousAccumulatorProviders);
    expect(providers.cacheTelemetryState).toBeUndefined();
    expect(providers.lodProgressProvider).toBeNull();
    expect(providers.drawOrderProvider).toBeNull();
    expect(markStructureDirty).toHaveBeenCalledOnce();
  });

  it('drops provider snapshots when live providers are disconnected', () => {
    const providers = new MonitorProviderRegistry(vi.fn());
    providers.lodStates.set('/node', {} as never);
    providers.drawOrderStates.set('/node', {} as never);

    providers.setLODProgressProvider(null);
    providers.setDrawOrderProvider(null);

    expect(providers.lodStates).toEqual(new Map());
    expect(providers.drawOrderStates).toEqual(new Map());
  });
});
