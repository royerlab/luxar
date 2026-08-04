/**
 * Port (interface) used by `SceneLoader` to push state into a UI
 * monitor without importing the concrete `DataLoadingMonitor`
 * implementation.
 *
 * The dependency cruiser forbids `data/` reaching into `ui/` directly;
 * this interface lives at the `data/` layer and captures *only* the
 * methods scene-loader actually calls. `core/app.ts` resolves a real
 * `DataLoadingMonitor` instance (which structurally satisfies this
 * port) and injects it through `SceneLoaderManager.setMonitorFactory`.
 *
 * If `SceneLoader` ever needs an additional monitor method, add it
 * here first — that keeps the contract explicit and prevents further
 * coupling growth.
 *
 * Closes the last layer-cruiser known exception
 * (`data/scene-loader.ts → ui/data-monitor-manager.ts`).
 */

import type {
  LoaderMonitor,
  CacheStatsProvider,
  CacheMetrics,
  CacheTelemetryState,
  SceneGraphNode,
  GPUPoolStats,
  AccumulatorStats,
  LODProgressProvider,
  DrawOrderProvider,
  PooledGeometryType,
} from '../types/data-monitor-types';
import type { UpdateProfiler } from '../profiling/update-profiler';
import type { GeometryTypeName } from '../types/format-contract';

/** Provider injected via `setL0CacheProvider`. */
export interface L0CacheProviderPort {
  getStats: () => CacheMetrics['l0'];
  clear: () => void;
}

/** Provider injected via `setSliceCacheProvider`. */
export interface SliceCacheProviderPort {
  getStats: () => CacheMetrics['slice'];
  clear: () => void;
}

/**
 * Provider injected via `setGPUBufferPoolProvider`. `GPUPoolStats`
 * lives in `types/data-monitor-types` so the data/ layer can reference
 * the precise type instead of `unknown`. The shape is cross-layer by
 * nature: the data/ layer pushes the provider, the UI/ layer renders
 * the stats.
 */
export interface GPUBufferPoolProviderPort {
  getStats: () => GPUPoolStats;
}

/**
 * Per-geometry accumulator provider. `AccumulatorStats` lives in
 * `types/data-monitor-types` for the same reason as `GPUPoolStats`:
 * precise types instead of `unknown`.
 */
export interface AccumulatorProviderPort {
  getStats: () => AccumulatorStats | null;
}

/**
 * Provider injected via `setFailedLoadsProvider`. Surfaces the loader's
 * failed-load records in the monitor UI and powers its Retry action —
 * the visible half of the failed-load recovery story (the `window
 * 'online'` listener in `core/app/lifecycle/online-retry.ts` is the
 * automatic half). `retryAll` maps to `SceneLoader.retryAllFailedLoaders`
 * (serialized against the update lock by the loader itself).
 */
export interface FailedLoadsProviderPort {
  getFailedPaths: () => string[];
  /**
   * `deferred: true` ⇒ the batch was refused because a main update held the
   * serialization lock — nothing was retried (see
   * `SceneLoader.retryAllFailedLoaders`). The UI must not report it as a
   * failed re-attempt.
   */
  retryAll: () => Promise<{ succeeded: string[]; failed: string[]; deferred?: boolean }>;
  /**
   * Human-readable failure reason for a single path, or undefined when the
   * path has no recorded failure. Derived from the loader's `FailedLoaderInfo`
   * (`error.message` / classified `kind`). OPTIONAL so structural implementers
   * that only surface the count/paths (e.g. `DataLoadingMonitor`) keep
   * compiling unchanged; the layers panel uses it for its per-row error
   * tooltip and falls back to a generic message when absent.
   */
  getFailedReason?: (path: string) => string | undefined;
}

/**
 * The subset of `DataLoadingMonitor`'s public surface that
 * `SceneLoader` consumes. Implemented structurally by
 * `DataLoadingMonitor`.
 */
export interface SceneLoaderMonitorPort {
  // Loader lifecycle
  connectLoader(path: string, loader: LoaderMonitor): void;
  disconnectAllLoaders(): void;

  // Provider injection
  setCacheStatsProvider(provider: CacheStatsProvider | null): void;
  setL0CacheProvider(provider: L0CacheProviderPort | null): void;
  setSliceCacheProvider(provider: SliceCacheProviderPort | null): void;
  setGPUBufferPoolProvider(provider: GPUBufferPoolProviderPort | null): void;
  setAccumulatorProvider(type: PooledGeometryType, provider: AccumulatorProviderPort | null): void;
  setProfiler(profiler: UpdateProfiler | null): void;
  /**
   * Inject the live LOD / progressive-refinement / cache-residency state
   * provider. Polled each tick to drive the scene-graph tree's kind
   * badges, "LOD x/N" chips, refining indicator, and header summary.
   */
  setLODProgressProvider(provider: LODProgressProvider | null): void;
  /**
   * Inject the live per-mesh draw-order provider (blending bucket,
   * depthWrite, renderOrder keyed by scene-graph path). Polled each tick to
   * drive the scene-graph tree's draw-order chip — pure observability.
   */
  setDrawOrderProvider(provider: DrawOrderProvider | null): void;
  /**
   * Inject the failed-loads provider (count/paths + retry-all). The UI
   * shows a warning banner with a Retry action while failures exist.
   */
  setFailedLoadsProvider(provider: FailedLoadsProviderPort | null): void;
  /**
   * Push the cache telemetry state resolved by `cache-setup.ts` so the
   * UI shows the right disabled-reason. Without this, the aggregator
   * falls back to "not-wired" whenever a `CacheStatsProvider` isn't
   * registered, which conflates `?no-cache` with mid-scene transitions.
   */
  setCacheTelemetryState(state: CacheTelemetryState): void;

  // Scene metadata + per-frame counters
  setSceneGraph(root: SceneGraphNode): void;
  forceUpdate(): void;
  updateVisibleCount(type: GeometryTypeName, count: number): void;
  /**
   * Per-node visible counts after nD slicing, keyed by scene-graph path
   * (mesh `name`). Drives the "(N visible after slicing)" suffix in the
   * scene-graph tree's badge tooltips. Only rendered meshes contribute
   * (hidden subtrees — inactive LOD levels, toggled-off layers — are
   * pruned by the caller's walk).
   */
  updateVisibleCountsByPath(counts: ReadonlyMap<string, number>): void;

  // UI visibility — driven by SceneLoader.showMonitor/hideMonitor/toggleMonitor
  show(): void;
  hide(): void;
  toggle(): void;
}

/**
 * Factory injected at construction time — given a monitor id, returns
 * an implementation (or null when no UI is wired up, e.g. in tests).
 * Called once per `SceneLoader` instance during construction.
 */
export type SceneLoaderMonitorFactory = (id: string) => SceneLoaderMonitorPort | null;
