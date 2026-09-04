import { describe, expect, it, vi } from 'vitest';

import { POOLED_GEOMETRY_TYPES } from '../../../../types/data-monitor-types';
import { MonitorProviderRegistry } from '../../../../ui/data-loading-monitor/providers';

const emptyAccumulatorProviders = () =>
  Object.fromEntries(POOLED_GEOMETRY_TYPES.map((type) => [type, null]));

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
    expect(providers.accumulatorProviders).toEqual(emptyAccumulatorProviders());
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
    expect(markStructureDirty).toHaveBeenCalledOnce();
    markStructureDirty.mockClear();

    providers.setGPUBufferPoolProvider(gpuBufferPoolProvider as never);
    expect(markStructureDirty).toHaveBeenCalledOnce();
    markStructureDirty.mockClear();

    providers.setProfiler(profiler as never);
    expect(markStructureDirty).toHaveBeenCalledOnce();
    markStructureDirty.mockClear();

    providers.setLODProgressProvider(lodProgressProvider as never);
    expect(markStructureDirty).toHaveBeenCalledOnce();
    markStructureDirty.mockClear();

    providers.setFailedLoadsProvider(failedLoadsProvider as never);
    providers.setDrawOrderProvider(drawOrderProvider as never);
    expect(markStructureDirty).toHaveBeenCalledOnce();
    markStructureDirty.mockClear();

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
    expect(markStructureDirty).not.toHaveBeenCalled();
  });

  it('refreshes and clears the live snapshots it owns', () => {
    const providers = new MonitorProviderRegistry(vi.fn());
    const lodStates = new Map([['/lod', { kind: 'lod' as const, levelCount: 3 }]]);
    const drawOrderStates = new Map([
      ['/lod/l0', { bucket: 'opaque' as const, depthWrite: true, renderOrder: 0 }],
    ]);

    const densityStates = new Map([
      ['/lod/l0', { keep: 0.25, elementsPerPixel: 9, blendable: true, onScreen: true }],
    ]);

    providers.setLODProgressProvider({ getLODStates: () => lodStates });
    providers.setDrawOrderProvider({ getDrawOrderStates: () => drawOrderStates });
    providers.setDensityProvider({ getDensityStates: () => densityStates });
    providers.refreshLiveSnapshots();

    expect(providers.lodStates).toBe(lodStates);
    expect(providers.drawOrderStates).toBe(drawOrderStates);
    expect(providers.densityStates).toBe(densityStates);

    // App-scoped: a scene reset leaves the density provider wired.
    providers.resetSceneProviders();
    expect(providers.densityProvider).not.toBeNull();
    providers.setDensityProvider(null);
    expect(providers.densityStates).toEqual(new Map());

    providers.clearDrawOrderStates();
    expect(providers.drawOrderStates).toEqual(new Map());
    expect(providers.drawOrderStates).not.toBe(drawOrderStates);
  });

  it('clears stale snapshots and restores fresh empty containers on reset', () => {
    const markStructureDirty = vi.fn();
    const providers = new MonitorProviderRegistry(markStructureDirty);
    const previousLodStates = providers.lodStates;
    const previousDrawOrderStates = providers.drawOrderStates;
    const previousAccumulatorProviders = providers.accumulatorProviders;

    providers.setLODProgressProvider({
      getLODStates: () => new Map([['/node', { kind: 'lod', levelCount: 1 }]]),
    });
    providers.setDrawOrderProvider({
      getDrawOrderStates: () =>
        new Map([['/node', { bucket: 'opaque', depthWrite: true, renderOrder: 0 }]]),
    });
    providers.refreshLiveSnapshots();
    providers.setAccumulatorProvider('lines', { getStats: vi.fn() } as never);
    providers.setCacheStatsProvider({ getStats: vi.fn() } as never);
    providers.setL0CacheProvider({ getStats: vi.fn(), clear: vi.fn() } as never);
    providers.setSliceCacheProvider({ getStats: vi.fn(), clear: vi.fn() } as never);
    providers.setGPUBufferPoolProvider({ getStats: vi.fn() } as never);
    providers.setProfiler({ getTimings: vi.fn() } as never);
    providers.setFailedLoadsProvider({ getFailedPaths: vi.fn(), retryAll: vi.fn() } as never);
    providers.setCacheTelemetryState({ kind: 'enabled' });
    markStructureDirty.mockClear();

    providers.resetSceneProviders();

    expect(providers.lodStates).toEqual(new Map());
    expect(providers.lodStates).not.toBe(previousLodStates);
    expect(providers.drawOrderStates).toEqual(new Map());
    expect(providers.drawOrderStates).not.toBe(previousDrawOrderStates);
    expect(providers.accumulatorProviders).toEqual(emptyAccumulatorProviders());
    expect(providers.accumulatorProviders).not.toBe(previousAccumulatorProviders);
    expect(providers.cacheTelemetryState).toBeUndefined();
    expect(providers.cacheStatsProvider).toBeNull();
    expect(providers.l0CacheProvider).toBeNull();
    expect(providers.sliceCacheProvider).toBeNull();
    expect(providers.gpuBufferPoolProvider).toBeNull();
    expect(providers.profiler).toBeNull();
    expect(providers.lodProgressProvider).toBeNull();
    expect(providers.failedLoadsProvider).toBeNull();
    expect(providers.drawOrderProvider).toBeNull();
    expect(markStructureDirty).toHaveBeenCalledOnce();
  });

  it('drops provider snapshots when live providers are disconnected', () => {
    const markStructureDirty = vi.fn();
    const providers = new MonitorProviderRegistry(markStructureDirty);
    providers.lodStates.set('/node', {} as never);
    providers.drawOrderStates.set('/node', {} as never);

    providers.setLODProgressProvider(null);
    providers.setDrawOrderProvider(null);

    expect(providers.lodStates).toEqual(new Map());
    expect(providers.drawOrderStates).toEqual(new Map());
    expect(markStructureDirty).not.toHaveBeenCalled();
  });
});
