import type { FailedLoadsProviderPort } from '../../data/scene-loader-monitor-port';
import type { UpdateProfiler } from '../../profiling/update-profiler';
import {
  POOLED_GEOMETRY_TYPES,
  type AccumulatorProvider,
  type CacheMetrics,
  type CacheStatsProvider,
  type CacheTelemetryState,
  type DensityProvider,
  type DrawOrderProvider,
  type LODProgressProvider,
  type LODProgressState,
  type MemoryMetrics,
  type NodeDensityState,
  type NodeDrawOrder,
  type PooledGeometryType,
} from '../../types/data-monitor-types';
import { log, Modules } from '../../utils/log';

/** Cache telemetry source with the matching cache-clear operation. */
export type CacheProvider<T> = { getStats: () => T; clear: () => void };

/** Live GPU buffer-pool telemetry source. */
export type GPUBufferPoolProvider = { getStats: () => MemoryMetrics['gpuPool'] };

/** Empty accumulator slots — one per {@link POOLED_GEOMETRY_TYPES} entry. */
function emptyAccumulatorSlots(): Record<PooledGeometryType, AccumulatorProvider | null> {
  return Object.fromEntries(POOLED_GEOMETRY_TYPES.map((type) => [type, null])) as Record<
    PooledGeometryType,
    AccumulatorProvider | null
  >;
}

/**
 * Owns the monitor's scene-scoped providers and their live snapshots.
 *
 * Provider changes call `markStructureDirty` when the orchestrator must
 * rebuild its painted structure; the flag itself remains orchestrator state.
 */
export class MonitorProviderRegistry {
  cacheStatsProvider: CacheStatsProvider | null = null;
  l0CacheProvider: CacheProvider<CacheMetrics['l0']> | null = null;
  sliceCacheProvider: CacheProvider<CacheMetrics['slice']> | null = null;
  // Pre-wiring this defaults to undefined so the aggregator falls back to
  // provider-presence inference; once the setter runs, the explicit state wins.
  cacheTelemetryState: CacheTelemetryState | undefined;
  gpuBufferPoolProvider: GPUBufferPoolProvider | null = null;
  profiler: UpdateProfiler | null = null;
  // Polled each tick; the snapshot drives kind badges, LOD chips, the refining
  // indicator, and the scene-graph header summary.
  lodProgressProvider: LODProgressProvider | null = null;
  /** Failed-load records + retry-all from the SceneLoader; feeds the overview banner. */
  failedLoadsProvider: FailedLoadsProviderPort | null = null;
  lodStates = new Map<string, LODProgressState>();
  // Live per-mesh draw order (blending bucket / depthWrite / renderOrder).
  // Polled every tick because renderOrder is camera-dependent; the snapshot
  // drives the scene-graph tree's draw-order chip.
  drawOrderProvider: DrawOrderProvider | null = null;
  drawOrderStates = new Map<string, NodeDrawOrder>();
  // Live per-node density-guard state (keep fraction / elements per pixel).
  // APP-scoped, unlike the slots above: the guard outlives any one scene, so
  // `resetSceneProviders` leaves it wired. Drives the tree's `drawn 1/K` chip.
  densityProvider: DensityProvider | null = null;
  densityStates = new Map<string, NodeDensityState>();
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
    // Cache tab structure may change between disabled/enabled states.
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
      // Provider availability changes the memory tab structure
      // (from "Not initialized" to full table)
      this.markStructureDirty();
    }
  }

  setProfiler(profiler: UpdateProfiler | null): void {
    this.profiler = profiler;
    if (profiler) {
      log.info(Modules.DATA_MONITOR, 'Update profiler connected');
      // Profiler availability changes the performance tab structure
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

  setDensityProvider(provider: DensityProvider | null): void {
    this.densityProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'Density-guard provider connected');
    } else {
      this.densityStates = new Map();
    }
  }

  setAccumulatorProvider(type: PooledGeometryType, provider: AccumulatorProvider | null): void {
    this.accumulatorProviders[type] = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, `${type} accumulator provider connected`);
    }
  }

  refreshLiveSnapshots(): void {
    if (this.lodProgressProvider) {
      this.lodStates = this.lodProgressProvider.getLODStates();
    }
    if (this.drawOrderProvider) {
      this.drawOrderStates = this.drawOrderProvider.getDrawOrderStates();
    }
    if (this.densityProvider) {
      this.densityStates = this.densityProvider.getDensityStates();
    }
  }

  clearDrawOrderStates(): void {
    this.drawOrderStates = new Map();
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
