import type { FailedLoadsProviderPort } from '../../data/scene-loader-monitor-port';
import type { UpdateProfiler } from '../../profiling/update-profiler';
import {
  POOLED_GEOMETRY_TYPES,
  type AccumulatorProvider,
  type CacheMetrics,
  type CacheStatsProvider,
  type CacheTelemetryState,
  type DrawOrderProvider,
  type LODProgressProvider,
  type LODProgressState,
  type NodeDrawOrder,
  type PooledGeometryType,
} from '../../types/data-monitor-types';
import { log, Modules } from '../../utils/log';
import type { MemoryMetrics } from './templates';

type CacheProvider<T> = { getStats: () => T; clear: () => void };
type GPUBufferPoolProvider = { getStats: () => MemoryMetrics['gpuPool'] };

function emptyAccumulatorSlots(): Record<PooledGeometryType, AccumulatorProvider | null> {
  return Object.fromEntries(POOLED_GEOMETRY_TYPES.map((type) => [type, null])) as Record<
    PooledGeometryType,
    AccumulatorProvider | null
  >;
}

export class MonitorProviderRegistry {
  cacheStatsProvider: CacheStatsProvider | null = null;
  l0CacheProvider: CacheProvider<CacheMetrics['l0']> | null = null;
  sliceCacheProvider: CacheProvider<CacheMetrics['slice']> | null = null;
  cacheTelemetryState: CacheTelemetryState | undefined;
  gpuBufferPoolProvider: GPUBufferPoolProvider | null = null;
  profiler: UpdateProfiler | null = null;
  lodProgressProvider: LODProgressProvider | null = null;
  failedLoadsProvider: FailedLoadsProviderPort | null = null;
  lodStates = new Map<string, LODProgressState>();
  drawOrderProvider: DrawOrderProvider | null = null;
  drawOrderStates = new Map<string, NodeDrawOrder>();
  accumulatorProviders = emptyAccumulatorSlots();

  constructor(private readonly markStructureDirty: () => void) {}

  setFailedLoadsProvider(provider: FailedLoadsProviderPort | null): void {
    this.failedLoadsProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'Failed-loads provider connected');
    }
  }

  setCacheStatsProvider(provider: CacheStatsProvider | null): void {
    this.cacheStatsProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'Cache stats provider connected');
    }
  }

  setCacheTelemetryState(state: CacheTelemetryState): void {
    this.cacheTelemetryState = state;
    log.info(Modules.DATA_MONITOR, `Cache telemetry state: ${state.kind}`);
    this.markStructureDirty();
  }

  setL0CacheProvider(provider: CacheProvider<CacheMetrics['l0']> | null): void {
    this.l0CacheProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'L0 cache provider connected');
    }
  }

  setSliceCacheProvider(provider: CacheProvider<CacheMetrics['slice']> | null): void {
    this.sliceCacheProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'SliceCache provider connected');
    }
  }

  setGPUBufferPoolProvider(provider: GPUBufferPoolProvider | null): void {
    this.gpuBufferPoolProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'GPU buffer pool provider connected');
      this.markStructureDirty();
    }
  }

  setProfiler(profiler: UpdateProfiler | null): void {
    this.profiler = profiler;
    if (profiler) {
      log.info(Modules.DATA_MONITOR, 'Update profiler connected');
      this.markStructureDirty();
    }
  }

  setLODProgressProvider(provider: LODProgressProvider | null): void {
    this.lodProgressProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'LOD progress provider connected');
      this.markStructureDirty();
    } else {
      this.lodStates = new Map();
    }
  }

  setDrawOrderProvider(provider: DrawOrderProvider | null): void {
    this.drawOrderProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'Draw-order provider connected');
      this.markStructureDirty();
    } else {
      this.drawOrderStates = new Map();
    }
  }

  setAccumulatorProvider(type: PooledGeometryType, provider: AccumulatorProvider | null): void {
    this.accumulatorProviders[type] = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, `${type} accumulator provider connected`);
    }
  }

  resetSceneProviders(): void {
    this.cacheStatsProvider = null;
    this.l0CacheProvider = null;
    this.sliceCacheProvider = null;
    this.gpuBufferPoolProvider = null;
    this.accumulatorProviders = emptyAccumulatorSlots();
    this.profiler = null;
    this.lodProgressProvider = null;
    this.failedLoadsProvider = null;
    this.lodStates = new Map();
    this.drawOrderProvider = null;
    this.drawOrderStates = new Map();
    // Reset to undefined (not 'not-wired') so the next scene's
    // setCacheTelemetryState call lands cleanly. If the next setup
    // doesn't call the setter, the aggregator falls back to
    // provider-presence inference.
    this.cacheTelemetryState = undefined;
    this.markStructureDirty();
    log.info(Modules.DATA_MONITOR, 'Scene providers reset');
  }
}
