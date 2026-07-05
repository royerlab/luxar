/**
 * Unit tests for `wireMonitorAfterLoad`.
 *
 * The helper is a fan-out: post-scene-load it pushes references to
 * caches, GPU pool, accumulators, profiler, and scene-graph snapshot
 * onto a monitor port. These tests assert the call sequence and the
 * null-guards (the no-monitor / no-cachingStore / no-l0Cache /
 * no-gpuBufferPool / no-profiler branches must each be silent
 * skips).
 *
 * `convertToSceneGraphNode` runs against a real SceneNode — it has
 * its own focused tests (`scene-graph-converter.test.ts`); here we
 * only assert that the conversion result reaches `setSceneGraph`.
 */

import { describe, it, expect, vi } from 'vitest';
import { wireMonitorAfterLoad } from '../../../../../data/scene-loader/monitor/monitor-wiring';
import type { SceneLoaderMonitorPort } from '../../../../../data/scene-loader-monitor-port';
import type { WireMonitorAfterLoadParams } from '../../../../../data/scene-loader/monitor/monitor-wiring';
import type { SceneNode } from '../../../../../data/data-loader-types';
import type { MultiLevelCachingStore } from '../../../../../cache/multi-level-caching-store';
import type { DecompressedChunkCache } from '../../../../../cache/decompressed-chunk-cache';
import type { GPUBufferPool } from '../../../../../rendering/gpu-buffer-pool';
import type { UpdateProfiler } from '../../../../../profiling/update-profiler';

function makeMonitor(): SceneLoaderMonitorPort & {
  __callOrder: string[];
} {
  const callOrder: string[] = [];
  const monitor = {
    connectLoader: vi.fn(),
    disconnectAllLoaders: vi.fn(),
    setCacheStatsProvider: vi.fn(() => callOrder.push('setCacheStatsProvider')),
    setL0CacheProvider: vi.fn(() => callOrder.push('setL0CacheProvider')),
    setGPUBufferPoolProvider: vi.fn(() => callOrder.push('setGPUBufferPoolProvider')),
    setAccumulatorProvider: vi.fn((type: string) =>
      callOrder.push(`setAccumulatorProvider:${type}`)
    ),
    setProfiler: vi.fn(() => callOrder.push('setProfiler')),
    setCacheTelemetryState: vi.fn(() => callOrder.push('setCacheTelemetryState')),
    setLODProgressProvider: vi.fn(() => callOrder.push('setLODProgressProvider')),
    setFailedLoadsProvider: vi.fn(() => callOrder.push('setFailedLoadsProvider')),
    setSceneGraph: vi.fn(() => callOrder.push('setSceneGraph')),
    forceUpdate: vi.fn(() => callOrder.push('forceUpdate')),
    updateVisiblePoints: vi.fn(),
    updateVisibleSegments: vi.fn(),
    updateVisibleSplats: vi.fn(),
    updateVisibleCountsByPath: vi.fn(),
    show: vi.fn(),
    hide: vi.fn(),
    toggle: vi.fn(),
    __callOrder: callOrder,
  };
  return monitor as unknown as SceneLoaderMonitorPort & { __callOrder: string[] };
}

function makeSceneGraph(): SceneNode {
  return {
    path: '/',
    type: 'group',
    attrs: {},
    children: [],
  } as unknown as SceneNode;
}

function makeBaseParams(monitor: SceneLoaderMonitorPort | null): WireMonitorAfterLoadParams {
  return {
    monitor,
    cachingStore: null,
    l0Cache: null,
    cacheTelemetryState: { kind: 'enabled' },
    gpuBufferPool: null,
    profiler: null,
    loaders: new Map(),
    linesLoaders: new Map(),
    gsplatLoaders: new Map(),
    lodGroupRegistry: null,
    sceneGraph: makeSceneGraph(),
    updateVisibleCounts: vi.fn(),
    failedLoads: {
      getFailedPaths: () => [],
      retryAll: vi.fn().mockResolvedValue({ succeeded: [], failed: [] }),
    },
  };
}

describe('wireMonitorAfterLoad — failed-loads provider', () => {
  it('injects the failedLoads provider into the monitor', () => {
    const monitor = makeMonitor();
    const params = makeBaseParams(monitor);
    wireMonitorAfterLoad(params);
    expect(
      (monitor as unknown as { setFailedLoadsProvider: ReturnType<typeof vi.fn> })
        .setFailedLoadsProvider
    ).toHaveBeenCalledWith(params.failedLoads);
  });
});

describe('wireMonitorAfterLoad — null-guard', () => {
  it('is a silent no-op when monitor is null', () => {
    const updateVisibleCounts = vi.fn();
    expect(() =>
      wireMonitorAfterLoad({
        ...makeBaseParams(null),
        updateVisibleCounts,
      })
    ).not.toThrow();
    expect(updateVisibleCounts).not.toHaveBeenCalled();
  });

  it('is a silent no-op when monitor is undefined', () => {
    expect(() =>
      wireMonitorAfterLoad({
        ...makeBaseParams(null),
        monitor: undefined,
      })
    ).not.toThrow();
  });
});

describe('wireMonitorAfterLoad — call ordering', () => {
  it('pushes setCacheTelemetryState before any provider setter', () => {
    const monitor = makeMonitor();
    const cachingStore = {} as MultiLevelCachingStore;

    wireMonitorAfterLoad({
      ...makeBaseParams(monitor),
      cachingStore,
    });

    const order = monitor.__callOrder;
    const telemetryIdx = order.indexOf('setCacheTelemetryState');
    const providerIdx = order.indexOf('setCacheStatsProvider');
    expect(telemetryIdx).toBeGreaterThanOrEqual(0);
    expect(providerIdx).toBeGreaterThan(telemetryIdx);
  });

  it('forceUpdate is the last call', () => {
    const monitor = makeMonitor();
    wireMonitorAfterLoad(makeBaseParams(monitor));
    const order = monitor.__callOrder;
    expect(order[order.length - 1]).toBe('forceUpdate');
  });
});

describe('wireMonitorAfterLoad — cache providers', () => {
  it('setCacheStatsProvider called when cachingStore is present', () => {
    const monitor = makeMonitor();
    const cachingStore = {} as MultiLevelCachingStore;
    wireMonitorAfterLoad({
      ...makeBaseParams(monitor),
      cachingStore,
    });
    expect(monitor.setCacheStatsProvider).toHaveBeenCalledWith(cachingStore);
  });

  it('setCacheStatsProvider skipped when cachingStore is null', () => {
    const monitor = makeMonitor();
    wireMonitorAfterLoad(makeBaseParams(monitor));
    expect(monitor.setCacheStatsProvider).not.toHaveBeenCalled();
  });

  it('setL0CacheProvider wraps l0Cache.getStats and l0Cache.clear', () => {
    const monitor = makeMonitor();
    const getStats = vi.fn(() => ({ size: 7 }));
    const clear = vi.fn();
    const l0Cache = { getStats, clear } as unknown as DecompressedChunkCache;

    wireMonitorAfterLoad({
      ...makeBaseParams(monitor),
      l0Cache,
    });

    const provider = (monitor.setL0CacheProvider as ReturnType<typeof vi.fn>).mock.calls[0][0] as {
      getStats: () => unknown;
      clear: () => void;
    };
    provider.getStats();
    provider.clear();
    expect(getStats).toHaveBeenCalledTimes(1);
    expect(clear).toHaveBeenCalledTimes(1);
  });

  it('setL0CacheProvider skipped when l0Cache is null', () => {
    const monitor = makeMonitor();
    wireMonitorAfterLoad(makeBaseParams(monitor));
    expect(monitor.setL0CacheProvider).not.toHaveBeenCalled();
  });

  it('always pushes the resolved cacheTelemetryState', () => {
    const monitor = makeMonitor();
    const cacheTelemetryState = { kind: 'disabled-no-cache' as const };
    wireMonitorAfterLoad({
      ...makeBaseParams(monitor),
      cacheTelemetryState,
    });
    expect(monitor.setCacheTelemetryState).toHaveBeenCalledWith(cacheTelemetryState);
  });
});

describe('wireMonitorAfterLoad — GPU pool + accumulators', () => {
  it('setGPUBufferPoolProvider called when gpuBufferPool is present', () => {
    const monitor = makeMonitor();
    const gpuBufferPool = {} as GPUBufferPool;
    wireMonitorAfterLoad({
      ...makeBaseParams(monitor),
      gpuBufferPool,
    });
    expect(monitor.setGPUBufferPoolProvider).toHaveBeenCalledWith(gpuBufferPool);
  });

  it('setGPUBufferPoolProvider skipped when gpuBufferPool is null', () => {
    const monitor = makeMonitor();
    wireMonitorAfterLoad(makeBaseParams(monitor));
    expect(monitor.setGPUBufferPoolProvider).not.toHaveBeenCalled();
  });

  it('registers a setAccumulatorProvider for each of points/lines/gsplats', () => {
    const monitor = makeMonitor();
    wireMonitorAfterLoad(makeBaseParams(monitor));
    const calls = (monitor.setAccumulatorProvider as ReturnType<typeof vi.fn>).mock.calls.map(
      (c) => c[0]
    );
    expect(calls).toEqual(['points', 'lines', 'gsplats']);
  });
});

describe('wireMonitorAfterLoad — profiler', () => {
  it('setProfiler called when profiler is present', () => {
    const monitor = makeMonitor();
    const profiler = {} as UpdateProfiler;
    wireMonitorAfterLoad({
      ...makeBaseParams(monitor),
      profiler,
    });
    expect(monitor.setProfiler).toHaveBeenCalledWith(profiler);
  });

  it('setProfiler skipped when profiler is null', () => {
    const monitor = makeMonitor();
    wireMonitorAfterLoad(makeBaseParams(monitor));
    expect(monitor.setProfiler).not.toHaveBeenCalled();
  });
});

describe('wireMonitorAfterLoad — scene graph + visible counts', () => {
  it('setSceneGraph receives the converted SceneGraphNode', () => {
    const monitor = makeMonitor();
    const sceneGraph: SceneNode = {
      path: '/',
      type: 'group',
      attrs: {},
      children: [],
    } as unknown as SceneNode;

    wireMonitorAfterLoad({
      ...makeBaseParams(monitor),
      sceneGraph,
    });

    const arg = (monitor.setSceneGraph as ReturnType<typeof vi.fn>).mock.calls[0][0];
    // Root SceneNode with path '/' converts to display name 'Scene'.
    expect(arg).toMatchObject({ name: 'Scene' });
  });

  it('updateVisibleCounts is invoked exactly once', () => {
    const monitor = makeMonitor();
    const updateVisibleCounts = vi.fn();
    wireMonitorAfterLoad({
      ...makeBaseParams(monitor),
      updateVisibleCounts,
    });
    expect(updateVisibleCounts).toHaveBeenCalledTimes(1);
  });
});
