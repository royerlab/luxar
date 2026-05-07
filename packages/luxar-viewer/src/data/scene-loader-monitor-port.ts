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
 * (`data/scene-loader.ts → ui/monitors/data-monitor-manager.ts`).
 */

import type {
  LoaderMonitor,
  CacheStatsProvider,
  CacheMetrics,
  SceneGraphNode,
} from '../types/data-monitor-types';
import type { UpdateProfiler } from '../profiling/update-profiler';

/** Provider injected via `setL0CacheProvider`. */
export interface L0CacheProviderPort {
  getStats: () => CacheMetrics['l0'];
  clear: () => void;
}

/**
 * Provider injected via `setGPUBufferPoolProvider`. The exact stats
 * shape lives at the ui/ layer (`MemoryMetrics['gpuPool']`); the data/
 * layer just hands the provider through, so `unknown` is the right
 * type here. The structural-typing on the consumer side (the real
 * `DataLoadingMonitor`) tightens it back up.
 */
export interface GPUBufferPoolProviderPort {
  getStats: () => unknown;
}

/**
 * Per-geometry accumulator provider. Same `unknown` reasoning as
 * GPUBufferPoolProviderPort — the data/ layer pushes provider; only
 * the UI layer knows the stats shape.
 */
export interface AccumulatorProviderPort {
  getStats: () => unknown;
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
  setGPUBufferPoolProvider(provider: GPUBufferPoolProviderPort | null): void;
  setAccumulatorProvider(
    type: 'points' | 'lines' | 'gsplats',
    provider: AccumulatorProviderPort | null
  ): void;
  setProfiler(profiler: UpdateProfiler | null): void;

  // Scene metadata + per-frame counters
  setSceneGraph(root: SceneGraphNode): void;
  forceUpdate(): void;
  updateVisibleSegments(count: number): void;
  updateVisibleSplats(count: number): void;

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
