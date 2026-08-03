/**
 * Data Loading Monitor - Main implementation
 *
 * A modern, event-driven monitoring system for the new spatial index-aware
 * data loading architecture. Provides real-time insights into loading
 * performance, cache efficiency, and spatial query patterns.
 */

import type {
  MonitorEvent,
  LoaderMonitor,
  LoaderMetrics,
  QueryInfo,
  MonitorConfig,
  GlobalStats,
  MonitorUIState,
  Recommendation,
  LoaderType,
  CacheMetrics,
  CacheStatsProvider,
  CacheTelemetryState,
  SceneGraphNode,
  SceneGraphState,
  GeometryCounters,
  LODProgressProvider,
  LODProgressState,
} from '../types/data-monitor-types';

// Performance timeline removed - now using hierarchical timing panel
import { aggregateCacheMetrics } from './data-loading-monitor/metrics/cache';
import { calculateRates } from './data-loading-monitor/metrics/rates';
import { updateCacheTab } from './data-loading-monitor/tabs/cache';
import { patchField, updateColorClass } from './data-loading-monitor/tabs/dom-helpers';
import { LoadingAdvisor } from './data-loading-monitor/advisor';
import { EventQueue } from './data-loading-monitor/event-queue';
import { PollingLoop } from './data-loading-monitor/polling-loop';
import { log, Modules } from '../utils/log';
import { config } from '../config';
import { notifier } from '../utils/cross-layer/notifier';
import type { FailedLoadsProviderPort } from '../data/scene-loader-monitor-port';

// Only extract timings and limits from config (these are data values, not styles)
const MonitorTimings = config.dataLoading.monitor.timings;
const MonitorLimits = config.dataLoading.monitor.limits;

// All styling now handled by CSS classes in data-loading-monitor.css

// Helper functions
const VALID_TABS = ['overview', 'cache', 'memory', 'performance', 'insights'] as const;
type ValidTab = (typeof VALID_TABS)[number];
function isValidTab(tab: string): tab is ValidTab {
  return VALID_TABS.includes(tab as ValidTab);
}

import {
  renderLoaderItem,
  renderOverviewContent,
  renderFailedLoadsBanner,
  renderCacheContent,
  renderMemoryContent,
  renderInsightsContent,
  renderSceneGraphTree,
  summariseLodStates,
  lodChipContent,
  nodeStatsContent,
  countAdditiveNodes,
  levelRoleTitleSuffix,
  activeLevelRole,
  formatNumber as templateFormatNumber,
  formatBytes as templateFormatBytes,
  getColorClass,
  getCacheMemoryColorClass,
  calculateReuseRate,
  getReuseRateColorClass,
  CACHE_SECTION_KEYS,
  MONITOR_ICONS,
  countColorClass,
  type MemoryMetrics,
} from './data-loading-monitor/templates';

import {
  renderHierarchicalTimingPanel,
  attachTimingPanelHandlers,
  updateTimingPanelValues,
} from './data-loading-monitor/timing-panel';

import type { UpdateProfiler } from '../profiling/update-profiler';
import { POOLED_GEOMETRY_TYPES } from '../types/data-monitor-types';
import type { PooledGeometryType, AccumulatorProvider } from '../types/data-monitor-types';
import { GEOMETRY_TYPES, type GeometryTypeName } from '../types/format-contract';

/** Empty accumulator slots — one per {@link POOLED_GEOMETRY_TYPES} entry. */
function emptyAccumulatorSlots(): Record<PooledGeometryType, AccumulatorProvider | null> {
  return Object.fromEntries(POOLED_GEOMETRY_TYPES.map((t) => [t, null])) as Record<
    PooledGeometryType,
    AccumulatorProvider | null
  >;
}

/**
 * A fresh all-zero per-type counter record, one slot per geometry type.
 *
 * Written as a plain loop, not `Object.fromEntries(GEOMETRY_TYPES.map(...))`:
 * `calculateSceneGraphStats` calls this twice per scene-graph node, and the
 * `fromEntries` form allocates an intermediate array of `[key, 0]` pairs on every
 * call. Measured on a 5000-node tree that shape cost ~6-10 ms per
 * `setSceneGraph` against ~0.4-1.3 ms for the loop — a 5-15x difference for no
 * behavioural gain.
 */
function zeroCounters(): GeometryCounters {
  const counters = {} as GeometryCounters;
  for (const t of GEOMETRY_TYPES) counters[t] = 0;
  return counters;
}

/** The empty scene-graph state (no scene loaded / scene torn down). */
function emptySceneGraphState(): SceneGraphState {
  return {
    root: null,
    totalNodes: 0,
    nodesByType: zeroCounters(),
    totalByType: zeroCounters(),
    visibleByType: zeroCounters(),
  };
}

/**
 * This node's own element count for `type`, or 0 when it is not that type.
 *
 * The per-type count fields are named after each type's ELEMENT (points have
 * points, lines have segments, gsplats have splats), so a table cannot key them
 * by type name. The `never` tail makes adding a geometry type a compile error
 * here — a plain `return 0` would leave the new type's elements out of every
 * dataset total with nothing to explain why.
 *
 * The tail still returns 0 rather than the unhandled value: breaking at compile
 * time is the point, but at runtime a count must stay a number (returning the
 * type string would poison every total it is summed into).
 *
 * `|| 0` and not `?? 0`: these counts are read straight off zarr `.zattrs` with
 * a bare cast (`scene-graph-converter.ts`), so a hand-edited or third-party store
 * can put `NaN` / `''` / `false` there. Coercing every falsy value keeps one bad
 * attr from poisoning every total it is summed into — `?? 0` would let `NaN`
 * through and turn the whole HUD into `NaN`.
 */
function elementCountOf(node: SceneGraphNode, type: GeometryTypeName): number {
  if (node.type !== type) return 0;
  switch (type) {
    case 'points':
      return node.pointCount || 0;
    case 'lines':
      return node.segmentCount || 0;
    case 'gsplats':
      return node.splatCount || 0;
    default:
      void (type satisfies never);
      return 0;
  }
}

/**
 * Main Data Loading Monitor class
 */
export class DataLoadingMonitor {
  private container: HTMLElement;
  private panel: HTMLElement | null = null;
  private loaders = new Map<string, LoaderMonitor>();
  private events: MonitorEvent[] = [];
  private metrics = new Map<string, LoaderMetrics>();
  private queries = new Map<string, QueryInfo>();

  // UI Components
  private advisor: LoadingAdvisor;

  // Configuration
  private config: MonitorConfig;

  // UI State
  private uiState: MonitorUIState = {
    isVisible: false,
    isExpanded: false,
    activeTab: 'overview',
    timeRange: MonitorTimings.defaultTimeRange,
  };

  // Polling-based update system (decoupled from event emission)
  private pollingLoop: PollingLoop;
  private eventQueue: EventQueue<MonitorEvent>;

  // Performance optimization
  private lastEventCleanup = 0;
  private eventCleanupInterval = MonitorTimings.eventCleanupInterval;
  private maxEventAge = MonitorTimings.maxEventAge;

  // Cached calculations
  private cachedRates = {
    queriesPerSec: 0,
    loadsPerSec: 0,
    hitsPerSec: 0,
    missesPerSec: 0,
    bandwidth: 0,
    lastCalculated: 0,
  };
  private ratesCacheTimeout = MonitorTimings.ratesCacheTimeout;

  // Event listener for external updates (non-blocking - just queues events)
  private eventListener = (event: MonitorEvent) => {
    this.eventQueue.push(event);
  };

  // UI event handler bound to this instance
  private uiEventHandler = this.handleUIEvent.bind(this);

  // Cache stats provider for L1/L2 cache metrics
  private cacheStatsProvider: CacheStatsProvider | null = null;

  // L0 decompressed chunk cache provider
  private l0CacheProvider: { getStats: () => CacheMetrics['l0']; clear: () => void } | null = null;
  private sliceCacheProvider: { getStats: () => CacheMetrics['slice']; clear: () => void } | null =
    null;

  // Explicit cache telemetry state (set by SceneLoader.cache-setup).
  // Pre-wiring this defaults to undefined so the aggregator falls back
  // to provider-presence inference; once setCacheTelemetryState() is
  // called, the explicit state wins.
  private cacheTelemetryState: CacheTelemetryState | undefined;

  // GPU buffer pool reference for dynamic stats retrieval
  private gpuBufferPoolProvider: { getStats: () => MemoryMetrics['gpuPool'] } | null = null;

  // Update profiler reference for hierarchical timing display
  private profiler: UpdateProfiler | null = null;

  // Live LOD / progressive-refinement / cache-residency state provider.
  // Polled each tick; the snapshot drives the scene-graph tree's kind
  // badges, "LOD x/N" chips, refining indicator, and header summary.
  private lodProgressProvider: LODProgressProvider | null = null;
  /** Failed-load records + retry-all, from the SceneLoader (overview banner). */
  private failedLoadsProvider: FailedLoadsProviderPort | null = null;
  /** In-flight guard so the banner's Retry button can't stack batches. */
  private retryFailedLoadsInFlight = false;
  /** Last-rendered failed-loads state; a change marks the overview structure dirty. */
  private lastFailedLoadsSignature = '';
  private lodStates: Map<string, LODProgressState> = new Map();
  /** Per-path visible counts pushed by the SceneLoader's visible-counts walk. */
  private visibleCountsByPath: ReadonlyMap<string, number> = new Map();

  // Accumulator providers for dynamic stats retrieval
  private accumulatorProviders: Record<PooledGeometryType, AccumulatorProvider | null> =
    emptyAccumulatorSlots();

  // DOM element references for efficient updates (avoids full innerHTML replacement)
  private contentContainer: HTMLElement | null = null;

  // Scene graph state
  private sceneGraphState: SceneGraphState = emptySceneGraphState();

  // Track expanded nodes in scene graph tree (by path)
  private expandedNodes = new Set<string>(['/']);

  // Cache-tab sections currently collapsed to their compact one-line
  // summary. All sections start collapsed — 4 stacked full sections
  // (S/L0/L1/L2) overflow the panel; the compact rows carry the same
  // values, so nothing is lost until the user expands for the card view.
  private collapsedCacheSections = new Set<string>(CACHE_SECTION_KEYS);

  // One-shot flag: the next detailed-structure rebuild plays the entrance
  // animation. Set on tab switch + expand; NOT on live structureDirty
  // rebuilds (scene-tree toggles, failed-loads banner), which would replay
  // the staggered reveal as visible flicker.
  private animateNextBuild = false;

  // Memoized path→node index over the current scene-graph tree, rebuilt
  // only when the root reference changes (a wholesale `setSceneGraph`).
  // Makes per-frame LOD-chip patching O(1) per chip instead of a full DFS
  // per chip (previously O(chips × nodes)).
  private sceneGraphNodeIndex = new Map<string, SceneGraphNode>();
  private sceneGraphNodeIndexRoot: SceneGraphNode | null = null;

  // Flag to force a full DOM rebuild on next update (set by structural changes like
  // tree node toggle, scene graph mutation). Cleared after rebuild.
  private structureDirty = false;

  constructor(container: HTMLElement, config?: Partial<MonitorConfig>) {
    this.container = container;
    this.config = {
      position: 'top-right',
      theme: 'dark',
      defaultView: 'compact',
      updateInterval: MonitorTimings.defaultUpdateInterval,
      maxEvents: MonitorLimits.maxEvents,
      showSpatialGrid: true,
      showTimeline: true,
      showRecommendations: true,
      autoExpand: false,
      enableProfiling: true,
      sampleRate: 1,
      ...config,
    };

    // Initialize components
    this.advisor = new LoadingAdvisor();

    // Initialize polling system (decoupled from event emission)
    this.eventQueue = new EventQueue<MonitorEvent>(this.config.maxEvents);
    this.pollingLoop = new PollingLoop({
      interval: this.config.updateInterval,
      onTick: () => this.onPollingTick(),
      onStart: () => log.info(Modules.DATA_MONITOR, 'Polling started'),
      onStop: () => log.info(Modules.DATA_MONITOR, 'Polling stopped'),
    });

    // Create UI
    this.createUI();
  }

  /**
   * Connect a loader for monitoring
   */
  public connectLoader(path: string, loader: LoaderMonitor): void {
    log.info(Modules.DATA_MONITOR, `Connecting loader for ${path}`);

    this.loaders.set(path, loader);
    loader.addEventListener(this.eventListener);

    // Get initial metrics
    const metrics = loader.getMetrics();
    this.metrics.set(path, metrics);

    // Analyze metrics for recommendations
    this.advisor.analyzeMetrics(metrics);
  }

  /**
   * Disconnect a loader
   */
  public disconnectLoader(path: string): void {
    const loader = this.loaders.get(path);
    if (loader) {
      loader.removeEventListener(this.eventListener);
      this.loaders.delete(path);
      this.metrics.delete(path);
    }
  }

  /**
   * Disconnect all loaders and reset monitor state
   * This should be called when loading a new scene
   */
  public disconnectAllLoaders(): void {
    // Disconnect all loaders
    for (const loader of this.loaders.values()) {
      loader.removeEventListener(this.eventListener);
    }
    this.loaders.clear();
    this.metrics.clear();

    // Reset other state but keep recent events for debugging
    this.queries.clear();

    // Reset components
    this.advisor.clear();

    // Drop closures bound to the previous scene's loader. Without this,
    // a `?no-cache` reload (which never re-installs L1/L2 providers)
    // leaves us calling `() => this.l0Cache!.getStats()` against a
    // disposed loader.
    this.resetSceneProviders();

    // Also drop the previous scene's graph display state. If the next
    // scene fails before `setSceneGraph()` runs, the monitor would
    // otherwise show a stale tree alongside cleared loaders.
    this.resetSceneGraphState();

    // Update UI to reflect cleared state if visible
    if (this.uiState.isVisible) {
      this.updateUI();
    }

    log.info(Modules.DATA_MONITOR, 'All loaders disconnected and monitor state reset');
  }

  /**
   * Set the failed-loads provider for the Overview tab's retry banner
   * (count/paths + retry-all; see `FailedLoadsProviderPort`).
   */
  public setFailedLoadsProvider(provider: FailedLoadsProviderPort | null): void {
    this.failedLoadsProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'Failed-loads provider connected');
    }
  }

  /**
   * Set the cache stats provider for L1/L2 cache monitoring.
   * This enables the monitor to display actual cache statistics.
   */
  public setCacheStatsProvider(provider: CacheStatsProvider | null): void {
    this.cacheStatsProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'Cache stats provider connected');
    }
  }

  /**
   * Push the cache telemetry state resolved by `cache-setup.ts`.
   * Called once per scene load so the UI shows the right
   * disabled-reason (`?no-cache` URL flag vs app-config disable vs
   * pre-wiring transition). Without this, the aggregator falls back
   * to provider-presence inference and misrepresents `?no-cache` runs
   * as `not-wired`.
   */
  public setCacheTelemetryState(state: CacheTelemetryState): void {
    this.cacheTelemetryState = state;
    log.info(Modules.DATA_MONITOR, `Cache telemetry state: ${state.kind}`);
    // Cache tab structure may change between disabled/enabled states.
    this.structureDirty = true;
  }

  /**
   * Set the L0 decompressed chunk cache provider for L0 cache monitoring.
   * This enables the monitor to display L0 cache statistics in the Cache tab.
   */
  public setL0CacheProvider(
    provider: { getStats: () => CacheMetrics['l0']; clear: () => void } | null
  ): void {
    this.l0CacheProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'L0 cache provider connected');
    }
  }

  /**
   * Register the SliceCache ("S-cache") stats/clear provider so the Cache tab
   * shows its usage and hit rate. Mirrors {@link setL0CacheProvider}.
   */
  public setSliceCacheProvider(
    provider: { getStats: () => CacheMetrics['slice']; clear: () => void } | null
  ): void {
    this.sliceCacheProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'SliceCache provider connected');
    }
  }

  /**
   * Set the GPU buffer pool provider for Memory tab stats.
   * The provider should have a getStats() method that returns PoolStats.
   */
  public setGPUBufferPoolProvider(
    provider: { getStats: () => MemoryMetrics['gpuPool'] } | null
  ): void {
    this.gpuBufferPoolProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'GPU buffer pool provider connected');
      // Provider availability changes the memory tab structure
      // (from "Not initialized" to full table)
      this.structureDirty = true;
    }
  }

  /**
   * Set the update profiler for Performance tab timing display.
   * The profiler tracks hierarchical timing of scene updates.
   */
  public setProfiler(profiler: UpdateProfiler | null): void {
    this.profiler = profiler;
    if (profiler) {
      log.info(Modules.DATA_MONITOR, 'Update profiler connected');
      // Profiler availability changes the performance tab structure
      this.structureDirty = true;
    }
  }

  /**
   * Set the LOD-progress provider for live LOD / refinement / residency
   * state in the scene-graph tree. Polled each tick. Passing a provider
   * marks the structure dirty so the tree re-renders with chip slots.
   */
  public setLODProgressProvider(provider: LODProgressProvider | null): void {
    this.lodProgressProvider = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, 'LOD progress provider connected');
      this.structureDirty = true;
    } else {
      this.lodStates = new Map();
    }
  }

  /**
   * Set an accumulator provider for Memory tab stats.
   * @param type - Which accumulator, one of {@link POOLED_GEOMETRY_TYPES}
   * @param provider - The accumulator with a getStats() method
   */
  public setAccumulatorProvider(
    type: PooledGeometryType,
    provider: AccumulatorProvider | null
  ): void {
    this.accumulatorProviders[type] = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, `${type} accumulator provider connected`);
    }
  }

  /**
   * Null every closure-bound provider tied to the active scene's loader
   * (cache stats, L0 cache, GPU buffer pool, accumulators, profiler).
   *
   * Required when a scene reload disposes the previous SceneLoader: each
   * provider was a closure that captured the now-disposed loader's
   * fields (`this.l0Cache`, etc.). Calling them after dispose throws —
   * e.g. the L0 closure `() => this.l0Cache!.getStats()` NPEs once the
   * previous loader nulled its `l0Cache`. Under `?no-cache` the next
   * scene never installs replacements either, so without this reset the
   * stale closures live until tab close.
   */
  public resetSceneProviders(): void {
    this.cacheStatsProvider = null;
    this.l0CacheProvider = null;
    this.sliceCacheProvider = null;
    this.gpuBufferPoolProvider = null;
    this.accumulatorProviders = emptyAccumulatorSlots();
    this.profiler = null;
    this.lodProgressProvider = null;
    this.failedLoadsProvider = null;
    this.lodStates = new Map();
    // Reset to undefined (not 'not-wired') so the next scene's
    // setCacheTelemetryState call lands cleanly. If the next setup
    // doesn't call the setter, the aggregator falls back to
    // provider-presence inference.
    this.cacheTelemetryState = undefined;
    this.structureDirty = true;
    log.info(Modules.DATA_MONITOR, 'Scene providers reset');
  }

  /**
   * Reset scene graph display state.
   *
   * Called from `disconnectAllLoaders()` (scene reload) and `dispose()`
   * (full teardown). Without this, a reload that fails before
   * `setSceneGraph()` runs leaves the monitor showing the previous
   * scene's tree, while loaders/providers have already been cleared —
   * a "ghost tree" mismatch in the Scene Graph panel.
   *
   * Also resets `expandedNodes` so a new scene starts from a freshly
   * collapsed tree (the root '/' marker preserves the previous default).
   */
  private resetSceneGraphState(): void {
    this.sceneGraphState = emptySceneGraphState();
    this.expandedNodes = new Set<string>(['/']);
    this.visibleCountsByPath = new Map();
    this.structureDirty = true;
  }

  /**
   * Clear L0 decompressed chunk cache.
   */
  public clearL0Cache(): void {
    if (this.l0CacheProvider) {
      this.l0CacheProvider.clear();
      log.info(Modules.DATA_MONITOR, 'L0 cache cleared');
      this.updateUI();
    }
  }

  /**
   * Clear the SliceCache ("S-cache").
   */
  public clearSliceCache(): void {
    if (this.sliceCacheProvider) {
      this.sliceCacheProvider.clear();
      log.info(Modules.DATA_MONITOR, 'SliceCache cleared');
      this.updateUI();
    }
  }

  /**
   * Clear L1 memory cache.
   */
  public clearL1Cache(): void {
    if (this.cacheStatsProvider) {
      this.cacheStatsProvider.clearL1();
      log.info(Modules.DATA_MONITOR, 'L1 cache cleared');
      this.updateUI();
    }
  }

  /**
   * Clear L2 OPFS cache. Asks for confirmation first since the L2
   * persistent cache requires a network re-fetch to repopulate. The
   * confirm dialog can be skipped by passing `{ skipConfirm: true }`
   * — used by tests and by the debug API where the caller has already
   * confirmed intent.
   */
  public async clearL2Cache(opts?: { skipConfirm?: boolean }): Promise<void> {
    if (!this.cacheStatsProvider) return;
    if (!opts?.skipConfirm && !this.confirmDestructiveCacheAction('Clear L2 (persistent) cache?')) {
      return;
    }
    const sizeBefore = this.cacheStatsProvider.getStats().l2.size;
    await this.cacheStatsProvider.clearL2();
    log.info(Modules.DATA_MONITOR, 'L2 cache cleared');
    if (sizeBefore > 0) {
      const mb = (sizeBefore / 1024 / 1024).toFixed(1);
      notifier.toast(`L2 cache cleared (${mb} MB freed)`);
    } else {
      notifier.toast('L2 cache cleared');
    }
    this.updateUI();
  }

  /**
   * Clear all caches (L0 + L1 + L2). Confirmation dialog can be
   * skipped via `{ skipConfirm: true }`.
   */
  public async clearAllCaches(opts?: { skipConfirm?: boolean }): Promise<void> {
    if (
      !opts?.skipConfirm &&
      !this.confirmDestructiveCacheAction('Clear ALL caches (L0 + L1 + L2)?')
    ) {
      return;
    }
    // Clear L0 + SliceCache first (synchronous)
    if (this.l0CacheProvider) {
      this.l0CacheProvider.clear();
    }
    if (this.sliceCacheProvider) {
      this.sliceCacheProvider.clear();
    }
    // Clear L1 + L2 (L2 is async)
    if (this.cacheStatsProvider) {
      await this.cacheStatsProvider.clearAll();
    }
    log.info(Modules.DATA_MONITOR, 'All caches cleared (S-cache + L0 + L1 + L2)');
    notifier.toast('All caches cleared');
    this.updateUI();
  }

  /**
   * Show a confirmation dialog for destructive cache actions. Falls
   * back to `true` if `window.confirm` is unavailable (jsdom test env).
   */
  private confirmDestructiveCacheAction(message: string): boolean {
    if (typeof window === 'undefined' || typeof window.confirm !== 'function') {
      return true;
    }
    return window.confirm(message);
  }

  /**
   * Set the scene graph for display in the monitor.
   * Called by SceneLoader after loading scene.
   */
  public setSceneGraph(root: SceneGraphNode): void {
    // Calculate stats from scene graph
    const stats = this.calculateSceneGraphStats(root);
    this.sceneGraphState = {
      root,
      ...stats,
    };
    log.info(
      Modules.DATA_MONITOR,
      `Scene graph updated: ${stats.totalNodes} nodes, ${stats.totalByType.points} points, ${stats.totalByType.lines} segments`
    );
    // Scene graph structure changed — need full rebuild on next update
    this.structureDirty = true;
    if (this.uiState.isVisible) {
      this.updateUI();
    }
  }

  /**
   * Get current scene graph state.
   */
  public getSceneGraph(): SceneGraphState {
    return this.sceneGraphState;
  }

  /**
   * Toggle expansion state of a node in the scene graph tree.
   */
  public toggleNodeExpansion(path: string): void {
    if (this.expandedNodes.has(path)) {
      this.expandedNodes.delete(path);
    } else {
      this.expandedNodes.add(path);
    }
    this.structureDirty = true;
    this.updateUI();
  }

  /**
   * Check if a node is expanded in the tree view.
   */
  public isNodeExpanded(path: string): boolean {
    return this.expandedNodes.has(path);
  }

  /**
   * Calculate statistics from scene graph tree.
   */
  private calculateSceneGraphStats(node: SceneGraphNode): Omit<SceneGraphState, 'root'> {
    let totalNodes = 1;
    const nodesByType = zeroCounters();
    const totalByType = zeroCounters();
    for (const t of GEOMETRY_TYPES) {
      if (node.type === t) nodesByType[t] = 1;
      totalByType[t] = elementCountOf(node, t);
    }

    const childStats = node.children.map((child) => this.calculateSceneGraphStats(child));

    // Structural counters (node + per-type node counts) always reflect the
    // real tree — a kind=lod group genuinely contains K child nodes.
    for (const cs of childStats) {
      totalNodes += cs.totalNodes;
      for (const t of GEOMETRY_TYPES) nodesByType[t] += cs.nodesByType[t];
    }

    // Geometry TOTALS: a substitutive kind=lod group's children are
    // mutually-exclusive representations of the SAME data at different
    // resolutions — summing them would inflate the dataset total ~K×. Use
    // the finest level (the last child; the Python writer guarantees
    // coarsest→finest order, see load-lod-group-node.ts) so the total
    // reflects true full-detail size. Partition parts are disjoint and
    // additive children are skipped from the scene graph, so both keep the
    // straight sum.
    const contributing =
      node.kind === 'lod' && childStats.length > 0
        ? [childStats[childStats.length - 1]]
        : childStats;
    for (const cs of contributing) {
      for (const t of GEOMETRY_TYPES) totalByType[t] += cs.totalByType[t];
    }

    // Initialize visible counts to totals (will be updated by scene loader)
    return { totalNodes, nodesByType, totalByType, visibleByType: { ...totalByType } };
  }

  /**
   * Update the count of currently visible elements of one geometry type.
   *
   * Called by SceneLoader (via `updateVisibleCountsInMonitor`) after the
   * per-type commits, so the HUD reports post-nD-clipping and
   * post-progressive-refinement counts rather than raw loaded ones. One
   * kind-keyed setter rather than one method per type: the three bodies were
   * identical apart from the field each wrote.
   */
  public updateVisibleCount(type: GeometryTypeName, count: number): void {
    this.sceneGraphState.visibleByType[type] = count;
  }

  /**
   * Per-node visible counts after nD slicing, keyed by scene-graph path.
   * Pushed by the SceneLoader's visible-counts walk (only rendered meshes
   * contribute). Merged into the tree nodes so badge tooltips can show
   * "(N visible after slicing)" per layer; the merge happens in
   * `updateSceneGraphBadges` on the next poll tick. Nodes whose path is
   * absent from the latest map (the walk prunes non-visible subtrees)
   * have their count cleared so tooltips never show a stale number.
   */
  public updateVisibleCountsByPath(counts: ReadonlyMap<string, number>): void {
    this.visibleCountsByPath = counts;
  }

  /**
   * Polling tick handler - called periodically by the polling loop.
   * This is the main update entry point that:
   * 1. Drains and processes queued events (batch processing)
   * 2. Pulls fresh stats from providers
   * 3. Updates the UI
   */
  private onPollingTick(): void {
    // Skip if monitor is not visible
    if (!this.uiState.isVisible) return;

    // 1. Drain and process all queued events (batch processing)
    const queuedEvents = this.eventQueue.drain();
    for (const event of queuedEvents) {
      this.processEvent(event);
    }

    // 2. Clean old events periodically
    const now = Date.now();
    if (now - this.lastEventCleanup > this.eventCleanupInterval) {
      this.cleanOldEvents();
      this.lastEventCleanup = now;
    }

    // 3. Refresh the live LOD / refinement / residency snapshot so the
    // scene-graph tree's chips and header summary reflect this frame.
    if (this.lodProgressProvider) {
      this.lodStates = this.lodProgressProvider.getLODStates();
    }

    // 4. Update UI (this also pulls fresh stats from providers)
    this.updateUI();
  }

  /**
   * Process a single event (called from polling tick, not from loaders).
   * This is now decoupled from event emission - events are batched and
   * processed during the polling tick.
   */
  private processEvent(event: MonitorEvent): void {
    // Store event in history
    this.events.push(event);
    if (this.events.length > this.config.maxEvents) {
      this.events.shift();
    }

    // Update metrics
    this.updateMetricsFromEvent(event);

    // Update query tracking
    if (event.type === 'query') {
      this.trackQuery(event);
    }

    // Check for issues
    if (this.config.showRecommendations) {
      this.advisor.analyzeEvent(event);

      // Auto-expand on warnings if configured
      if (this.config.autoExpand && this.advisor.hasWarnings()) {
        this.expand();
      }
    }

    // Invalidate cached rates
    this.cachedRates.lastCalculated = 0;
  }

  /**
   * Update metrics from event
   */
  private updateMetricsFromEvent(event: MonitorEvent): void {
    const path = event.data.path || 'unknown';
    const metrics = this.metrics.get(path) || this.createEmptyMetrics(path, event.loader);

    switch (event.type) {
      case 'query':
        metrics.queries++;
        if (event.data.latency) {
          // Update average query time
          const totalTime = metrics.avgQueryTime * (metrics.queries - 1) + event.data.latency;
          metrics.avgQueryTime = totalTime / metrics.queries;
        }
        break;

      case 'load':
        metrics.loads++;
        metrics.elementsLoaded += event.data.elements || 0;
        metrics.bytesLoaded += event.data.memory || 0;
        if (event.data.latency) {
          const totalTime = metrics.avgLoadTime * (metrics.loads - 1) + event.data.latency;
          metrics.avgLoadTime = totalTime / metrics.loads;
        }
        break;

      case 'evict':
        metrics.evictions++;
        break;

      case 'error':
        metrics.errors++;
        break;
    }

    this.metrics.set(path, metrics);
  }

  /**
   * Track active queries
   */
  private trackQuery(event: MonitorEvent): void {
    const queryId = `${event.data.path}-${event.timestamp}`;

    this.queries.set(queryId, {
      id: queryId,
      loader: event.loader,
      path: event.data.path || '',
      startTime: event.timestamp,
      status: 'loading',
      cells: event.data.cells,
      elements: event.data.elements,
      ranges: event.data.ranges,
    });

    // Clean up old queries more efficiently
    if (this.queries.size % MonitorTimings.queryCleanupCheckInterval === 0) {
      const cutoff = Date.now() - MonitorTimings.maxQueryAge;
      const toDelete: string[] = [];
      for (const [id, query] of this.queries) {
        if (query.startTime < cutoff) {
          toDelete.push(id);
        }
      }
      toDelete.forEach((id) => this.queries.delete(id));
    }
  }

  /**
   * Create empty metrics object
   */
  private createEmptyMetrics(path: string, type: LoaderType): LoaderMetrics {
    return {
      type,
      path,
      queries: 0,
      loads: 0,
      evictions: 0,
      errors: 0,
      elementsLoaded: 0,
      bytesLoaded: 0,
      visibleElements: 0,
      avgQueryTime: 0,
      avgLoadTime: 0,
      memoryUsed: 0,
      memoryLimit: MonitorLimits.defaultMemoryLimit, // Will be overridden by actual loader limits
    };
  }

  /**
   * Force an immediate UI update.
   * Used when scene changes or loaders are connected.
   * Triggers an immediate polling tick to process any queued events
   * and refresh the UI.
   */
  public forceUpdate(): void {
    if (this.uiState.isVisible) {
      // Process any queued events and update UI immediately
      this.onPollingTick();
    }
  }

  /**
   * Update UI
   */
  private updateUI(): void {
    if (!this.uiState.isVisible || !this.panel) return;

    // Update based on current view
    if (this.uiState.isExpanded) {
      this.updateDetailedView();
    } else {
      this.updateCompactView();
    }
  }

  /**
   * Handle UI events through event delegation
   */
  private handleUIEvent(event: Event): void {
    // Resolve to the closest actionable ancestor so clicks landing on
    // child elements (e.g. the text spans inside a collapsible cache
    // section header) still trigger the header's action. The nearest
    // `data-action` wins, so buttons nested inside an actionable header
    // (like Clear) keep their own action.
    const target = (event.target as HTMLElement | null)?.closest?.(
      '[data-action]'
    ) as HTMLElement | null;
    if (!target) return;

    const action = target.dataset.action;
    if (!action) return;

    // Handle different actions
    switch (action) {
      case 'expand':
        this.expand();
        break;
      case 'minimize':
        this.minimize();
        break;
      case 'hide':
        this.hide();
        break;
      case 'setTab': {
        const tabId = target.dataset.tabId;
        if (tabId) {
          this.setActiveTab(tabId);
        }
        break;
      }
      case 'setTimeRange': {
        const select = target as HTMLSelectElement;
        this.setTimeRange(select.value);
        break;
      }
      case 'clearL0':
        this.clearL0Cache();
        break;
      case 'clearSlice':
        this.clearSliceCache();
        break;
      case 'clearL1':
        this.clearL1Cache();
        break;
      case 'clearL2':
        this.clearL2Cache();
        break;
      case 'clearAll':
        this.clearAllCaches();
        break;
      case 'retryFailedLoads':
        this.retryFailedLoads();
        break;
      case 'toggleNode': {
        const nodePath = target.dataset.nodePath;
        if (nodePath) {
          this.toggleNodeExpansion(nodePath);
        }
        break;
      }
      case 'toggleCacheSection': {
        const sectionKey = target.dataset.sectionKey;
        if (sectionKey) {
          this.toggleCacheSection(sectionKey);
        }
        break;
      }
    }
  }

  /**
   * Toggle a Cache-tab section between its full metric-card view and
   * the compact one-line header summary. Both views are always in the
   * DOM (visibility is CSS-driven), so the toggle is a pure class flip
   * — no re-render, and the per-tick patcher keeps updating both.
   */
  private toggleCacheSection(sectionKey: string): void {
    if (this.collapsedCacheSections.has(sectionKey)) {
      this.collapsedCacheSections.delete(sectionKey);
    } else {
      this.collapsedCacheSections.add(sectionKey);
    }
    const collapsed = this.collapsedCacheSections.has(sectionKey);
    const section = this.contentContainer?.querySelector(
      `.luxar-cache-section[data-section="${sectionKey}"]`
    );
    section?.classList.toggle('luxar-cache-section--collapsed', collapsed);
    // Transient morph marker: the expand/collapse reveal animation is keyed
    // on THIS class (not on the collapsed state), so it plays only on an
    // interactive toggle — never when a structural rebuild re-creates the
    // sections in their current state.
    if (section) {
      section.classList.add('luxar-cache-section--morph');
      setTimeout(() => section.classList.remove('luxar-cache-section--morph'), 250);
    }
    const header = section?.querySelector('.luxar-cache-section__header');
    header?.setAttribute('title', `Click to ${collapsed ? 'expand' : 'collapse'} this section`);
    header?.setAttribute('aria-expanded', String(!collapsed));
  }

  /**
   * Create UI elements
   */
  private createUI(): void {
    // Styles now in src/styles/components/data-loading-monitor.css

    // Create main panel
    this.panel = document.createElement('div');
    this.updatePanelClasses();

    // Add event delegation listeners. Keydown makes the div-based
    // actionable elements (e.g. the collapsible cache-section headers,
    // which carry role="button" + tabindex) keyboard-operable.
    this.panel.addEventListener('click', this.uiEventHandler);
    this.panel.addEventListener('change', this.uiEventHandler);
    this.panel.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      const target = (e.target as HTMLElement | null)?.closest?.('[data-action]');
      // Native buttons/selects already handle Enter/Space themselves.
      if (!target || target instanceof HTMLButtonElement || target instanceof HTMLSelectElement) {
        return;
      }
      e.preventDefault(); // stop Space from scrolling the panel
      this.uiEventHandler(e);
    });

    // Add to container
    this.container.appendChild(this.panel);

    // Initially hidden
    this.hide();
  }

  /**
   * Update panel CSS classes based on current state
   */
  private updatePanelClasses(): void {
    if (!this.panel) return;

    // Base class (+ glass-surface marker so it themes with the glass themes)
    const classes = ['luxar-data-monitor', 'luxar-glass-surface'];

    // Position class
    classes.push(`luxar-data-monitor--${this.config.position}`);

    // Size class
    if (this.uiState.isExpanded) {
      classes.push('luxar-data-monitor--expanded');
    } else {
      classes.push('luxar-data-monitor--compact');
    }

    this.panel.className = classes.join(' ');
  }

  /**
   * Build the compact-view element summary. Shows one count per geometry
   * type actually present in the scene (points / lines / gsplats), using the
   * same presence test as the overview tab (`updateOverviewTabValues`) so a
   * lines- or gsplats-only dataset no longer mislabels its elements as
   * "points". Falls back to a points entry when nothing has loaded yet.
   */
  private buildCompactGeomSummary(stats: GlobalStats): string {
    const hasPoints = stats.datasetSize > 0 || stats.visiblePoints > 0;
    const hasLines = stats.datasetSegments > 0 || stats.visibleSegments > 0;
    const hasGSplats = stats.datasetSplats > 0 || stats.visibleSplats > 0;

    const entries: string[] = [];
    if (hasPoints) {
      entries.push(
        `<span data-geom="points" title="Points currently on screen (inside the active nD slice). Expand the monitor for totals and per-layer detail">${templateFormatNumber(stats.visiblePoints)} pts</span>`
      );
    }
    if (hasLines) {
      entries.push(
        `<span data-geom="lines" title="Line segments currently on screen (inside the active nD slice). Expand the monitor for totals and per-layer detail">${templateFormatNumber(stats.visibleSegments)} lines</span>`
      );
    }
    if (hasGSplats) {
      entries.push(
        `<span data-geom="splats" title="Gaussian splats currently on screen (inside the active nD slice). Expand the monitor for totals and per-layer detail">${templateFormatNumber(stats.visibleSplats)} splats</span>`
      );
    }
    // Nothing loaded yet → show a points placeholder so the row isn't empty.
    if (entries.length === 0) {
      entries.push(
        `<span data-geom="points" title="Points currently on screen (inside the active nD slice). Expand the monitor for totals and per-layer detail">${templateFormatNumber(stats.visiblePoints)} pts</span>`
      );
    }
    return entries.join('');
  }

  /**
   * Try to update just the compact view values without replacing innerHTML.
   * This preserves event handlers and prevents interaction loss during polling updates.
   * @returns true if incremental update succeeded, false if full rebuild is needed
   */
  private updateCompactViewValues(): boolean {
    const geomsEl = this.panel?.querySelector(
      '.luxar-monitor-compact .luxar-monitor-compact__geoms'
    );
    const memoryEl = this.panel?.querySelector(
      '.luxar-monitor-compact .luxar-monitor-compact__memory'
    );
    const qpsEl = this.panel?.querySelector('.luxar-monitor-compact .luxar-monitor-compact__qps');

    // If structure doesn't exist yet, need full rebuild
    if (!geomsEl || !memoryEl || !qpsEl) return false;

    // Update metrics from all loaders
    for (const [path, loader] of this.loaders) {
      const metrics = loader.getMetrics();
      this.metrics.set(path, metrics);
    }

    const stats = this.getGlobalStats();
    // Rewrite only the geometry summary's inner HTML — it holds no event
    // handlers, so this is safe and leaves the expand button intact.
    geomsEl.innerHTML = this.buildCompactGeomSummary(stats);
    memoryEl.textContent = templateFormatBytes(stats.totalMemory);
    qpsEl.textContent = `${stats.queriesPerSecond.toFixed(1)}/s`;
    return true;
  }

  /**
   * Update compact view.
   * Tries incremental value update first; only does full innerHTML rebuild
   * when the structure doesn't exist yet (first render or after expand/minimize).
   */
  private updateCompactView(): void {
    if (!this.panel) return;

    // Try incremental update first (preserves DOM, no interaction issues)
    if (this.updateCompactViewValues()) {
      return;
    }

    // Update metrics from all loaders first
    for (const [path, loader] of this.loaders) {
      const metrics = loader.getMetrics();
      this.metrics.set(path, metrics);
    }

    const stats = this.getGlobalStats();
    const hasSpatialIndex = stats.activeSpatialLoaders > 0;
    const recommendations = this.advisor.getRecommendations();
    const hasWarnings = recommendations.some((r) => r.severity === 'warning');
    const hasErrors = recommendations.some((r) => r.severity === 'error');

    this.panel.innerHTML = `
      <div class="luxar-glass-refraction" aria-hidden="true"></div>
      <div class="luxar-monitor-compact">
        <span class="luxar-monitor-compact__type" title="${hasSpatialIndex ? 'Loading mode: spatial-index streaming — only the data inside the current view/slice is queried and loaded on demand (scales to arbitrarily large datasets)' : 'Loading mode: direct loading — the dataset is loaded whole, without an on-demand spatial index'}">
          ${hasSpatialIndex ? MONITOR_ICONS.stream : MONITOR_ICONS.box}
        </span>

        <span class="luxar-monitor-compact__geoms">
          ${this.buildCompactGeomSummary(stats)}
        </span>

        <span class="luxar-monitor-compact__memory" title="CPU memory currently held by loaded geometry data, across all layers (points + lines + gsplats)">
          ${templateFormatBytes(stats.totalMemory)}
        </span>

        <span class="luxar-monitor-compact__qps" title="Spatial queries per second — how often the viewer is asking the index for data as you navigate. 0/s when idle is normal">
          ${stats.queriesPerSecond.toFixed(1)}/s
        </span>

        ${hasErrors ? `<span class="luxar-monitor-compact__alert luxar-color--error" title="Errors detected — expand the monitor and open the Insights tab for details and suggested fixes">${MONITOR_ICONS.dot}</span>` : ''}
        ${hasWarnings && !hasErrors ? `<span class="luxar-monitor-compact__alert luxar-color--warning" title="Warnings — expand the monitor and open the Insights tab for details and suggested fixes">${MONITOR_ICONS.dot}</span>` : ''}

        <button class="luxar-data-monitor__expand-btn" data-action="expand" title="Expand into the full Data Loading Monitor: per-tab views of loading, cache, memory, performance, and insights">
          ${MONITOR_ICONS.expand}
        </button>
      </div>
    `;
  }

  /**
   * Build the detailed view structure once (called on expand or tab change)
   */
  private buildDetailedViewStructure(): void {
    if (!this.panel) return;

    // One-shot entrance-animation marker: set by setActiveTab (and the
    // initial expand), consumed here. Live structureDirty rebuilds render
    // WITHOUT the class so the reveal never replays mid-session.
    const animClass = this.animateNextBuild ? ' luxar-data-monitor__content--animate' : '';
    this.animateNextBuild = false;

    this.panel.innerHTML = `
      <div class="luxar-glass-refraction" aria-hidden="true"></div>
      <div class="luxar-monitor-detailed">
        <!-- Header -->
        <div class="luxar-data-monitor__header">
          <div class="luxar-data-monitor__title">Data Loading Monitor</div>
          <div class="luxar-data-monitor__controls">
            <button class="luxar-data-monitor__btn luxar-data-monitor__close-btn" data-action="minimize" title="Minimize">—</button>
            <button class="luxar-data-monitor__btn luxar-data-monitor__close-btn" data-action="hide" title="Close">×</button>
          </div>
        </div>

        <!-- Tabs -->
        <div class="luxar-data-monitor__tabs">
          ${this.renderTabs()}
        </div>

        <!-- Content (updated frequently via targeted patching) -->
        <div class="luxar-data-monitor__content${animClass}">
          ${this.renderTabContent()}
        </div>
      </div>
    `;

    // Cache reference to content container for efficient updates
    this.contentContainer = this.panel.querySelector('.luxar-data-monitor__content');

    // Attach tab-specific handlers after full rebuild
    this.attachTabHandlers();
  }

  /**
   * Attach handlers needed by the current tab after a full content rebuild.
   */
  private attachTabHandlers(): void {
    if (this.uiState.activeTab === 'performance' && this.contentContainer) {
      // Timing panel expand/collapse is a structural change — the incremental
      // updater can't add/remove child rows, so we need a full rebuild.
      attachTimingPanelHandlers(this.contentContainer, () => {
        this.structureDirty = true;
        this.updateUI();
      });
    }
  }

  /**
   * Update detailed view using targeted DOM patching.
   *
   * On each polling tick, tries incremental value updates first (updating only
   * textContent/style of specific elements via data-field attributes). This
   * preserves the DOM tree, event handlers, focus, and scroll position.
   *
   * Falls back to full innerHTML rebuild only when:
   * - Structure doesn't exist yet (first render)
   * - structureDirty flag is set (tree toggle, scene graph mutation)
   * - Incremental update reports structure mismatch
   */
  private updateDetailedView(): void {
    if (!this.panel) return;

    // Failed-loads banner liveness: the overview HTML is rebuilt only when
    // structureDirty (values are otherwise patched in place), and the banner
    // is part of that HTML — so any change in the failed set or the
    // retry-in-flight flag must mark the structure dirty, or the banner
    // appears/disappears/disables only on the next unrelated rebuild.
    const failedLoadsSignature =
      (this.failedLoadsProvider?.getFailedPaths() ?? []).join('|') +
      (this.retryFailedLoadsInFlight ? '#retrying' : '');
    if (failedLoadsSignature !== this.lastFailedLoadsSignature) {
      this.lastFailedLoadsSignature = failedLoadsSignature;
      this.structureDirty = true;
    }

    // Update metrics from all loaders
    for (const [path, loader] of this.loaders) {
      const metrics = loader.getMetrics();
      this.metrics.set(path, metrics);
    }

    // Analyze memory metrics for recommendations
    if (this.config.showRecommendations) {
      const memoryMetrics = this.getMemoryMetrics();
      this.advisor.analyzeMemoryMetrics(memoryMetrics);
    }

    // If structure doesn't exist yet, build it (first render or after tab/view change)
    if (!this.contentContainer) {
      this.buildDetailedViewStructure();
      this.structureDirty = false; // Full build satisfies any pending structural change
      return;
    }

    // Try incremental value update for the active tab
    let updated = false;

    if (!this.structureDirty) {
      switch (this.uiState.activeTab) {
        case 'overview':
          updated = this.updateOverviewTabValues();
          break;
        case 'cache':
          updated = this.updateCacheTabValues();
          break;
        case 'memory':
          updated = this.updateMemoryTabValues();
          break;
        case 'performance':
          if (this.profiler) {
            const timingData = this.profiler.getTimings();
            const refinementData = this.profiler.getRefinementTimings();
            const depthSortData = this.profiler.getDepthSortTimings();
            if (timingData.count > 0 || refinementData.count > 0 || depthSortData.count > 0) {
              updated = updateTimingPanelValues(
                this.contentContainer,
                timingData,
                refinementData,
                depthSortData
              );
            }
          }
          break;
        case 'insights':
          // Always rebuild — structural content (variable-length recommendations list)
          break;
      }
    }

    // Clear structureDirty after checking it (whether we used it or not)
    this.structureDirty = false;

    if (!updated) {
      // Full content rebuild (structure changed or first render for this tab).
      // The entrance-reveal marker is strictly one-shot (tab switch/expand):
      // this PARTIAL rebuild path replaces the container's children while the
      // container itself — and any lingering marker class — survives, so the
      // reveal would replay on every such rebuild (10x/second on tabs that
      // always rebuild, e.g. Insights). Strip it before re-rendering.
      this.contentContainer.classList.remove('luxar-data-monitor__content--animate');
      this.contentContainer.innerHTML = this.renderTabContent();
      this.attachTabHandlers();
    }
  }

  // ─── Per-tab incremental value update functions ───────────────────────

  /**
   * Helper: update a single element's textContent by data-field
   * attribute. Delegates to the shared DOM helper module so per-tab
   * updaters use the same patch contract.
   */
  private patchField(field: string, text: string): boolean {
    return patchField(this.contentContainer, field, text);
  }

  /**
   * Incrementally update overview tab values without rebuilding DOM.
   * Updates primary metric cards, secondary metrics, and scene graph badges.
   */
  private updateOverviewTabValues(): boolean {
    if (!this.contentContainer) return false;

    const stats = this.getGlobalStats();
    const cacheMetrics = this.getCacheMetrics();

    // Update primary metric card values
    const hasPoints = stats.datasetSize > 0 || stats.visiblePoints > 0;
    const hasLines = stats.datasetSegments > 0 || stats.visibleSegments > 0;
    const hasGSplats = stats.datasetSplats > 0 || stats.visibleSplats > 0;

    // Check at least one primary metric exists in DOM (structure validation)
    const anyPrimaryField =
      this.contentContainer.querySelector('[data-field="visible-points"]') ||
      this.contentContainer.querySelector('[data-field="visible-lines"]') ||
      this.contentContainer.querySelector('[data-field="visible-splats"]');
    if (!anyPrimaryField) return false; // Structure not built yet

    // Single-type layouts use " total" suffix in subtitle (matches template rendering)
    const dataTypeCount = [hasPoints, hasLines, hasGSplats].filter(Boolean).length;
    const suffix = dataTypeCount === 1 ? ' total' : '';

    // Count cards: value text plus the state color (neutral with data,
    // dimmed at zero — matches `countColorClass` in the initial render,
    // so a card doesn't stay dimmed after points scroll into view).
    const patchCount = (field: string, visible: number, dataset: number) => {
      const pct = dataset > 0 ? ((visible / dataset) * 100).toFixed(1) : '0';
      this.patchField(field, templateFormatNumber(visible));
      this.patchField(`${field}-sub`, `${pct}% of ${templateFormatNumber(dataset)}${suffix}`);
      const el = this.contentContainer?.querySelector(`[data-field="${field}"]`);
      if (el) this.updateColorClass(el as HTMLElement, countColorClass(visible));
    };
    if (hasPoints) patchCount('visible-points', stats.visiblePoints, stats.datasetSize);
    if (hasLines) patchCount('visible-lines', stats.visibleSegments, stats.datasetSegments);
    if (hasGSplats) patchCount('visible-splats', stats.visibleSplats, stats.datasetSplats);

    // Update secondary metrics
    this.patchField('memory-used', templateFormatBytes(cacheMetrics.totalCacheMemory));
    this.patchField('query-speed', `${stats.avgQueryTime.toFixed(0)}ms`);
    this.patchField('query-rate', `${stats.queriesPerSecond.toFixed(1)}/sec`);
    // "DATA LOADED" card: cumulative bytes delivered across all tiers
    // (L1 + L2 + network), so it stays informative on a warm/cache-served
    // reload where `bytesTransferred` is legitimately 0. The subtitle
    // breaks out how much of that came over the network plus live bandwidth.
    const net = cacheMetrics.network;
    const dataLoaded = net ? (net.totalBytesServed ?? net.bytesTransferred) : 0;
    this.patchField('network-bytes', net ? templateFormatBytes(dataLoaded) : '0B');
    this.patchField(
      'network-detail',
      net
        ? `${templateFormatBytes(net.bytesTransferred)} net · ${templateFormatBytes(net.bandwidth)}/s`
        : '0B net'
    );

    // Update secondary metrics memory progress bar
    const memoryPercent =
      cacheMetrics.memoryLimit > 0
        ? (cacheMetrics.totalCacheMemory / cacheMetrics.memoryLimit) * 100
        : 0;
    const overviewBarFill = this.contentContainer.querySelector(
      '.luxar-secondary-metrics .luxar-progress-bar__fill'
    ) as HTMLElement | null;
    if (overviewBarFill) {
      overviewBarFill.style.width = `${Math.min(100, memoryPercent)}%`;
      this.updateColorClass(overviewBarFill, getCacheMemoryColorClass(memoryPercent));
    }

    // Update scene graph badges (by data-node-path)
    this.updateSceneGraphBadges();

    return true;
  }

  /**
   * Update scene graph tree badge values without rebuilding the tree DOM.
   */
  private updateSceneGraphBadges(): void {
    if (!this.contentContainer || !this.sceneGraphState.root) return;

    // Merge the latest per-path visible counts into the tree nodes so the
    // badge tooltips (via nodeStatsContent) reflect post-slicing visibility.
    this.syncVisibleCountsIntoTree();

    const badges = this.contentContainer.querySelectorAll(
      '.luxar-scene-graph__badge[data-node-path]'
    );
    badges.forEach((badge) => {
      const path = (badge as HTMLElement).dataset.nodePath;
      if (!path) return;
      const node = this.getSceneGraphNodeByPath(path);
      if (!node) return;

      // Same helper as the initial render so text + tooltip stay in sync.
      const stats = nodeStatsContent(node);
      if (stats) {
        badge.textContent = stats.text;
        (badge as HTMLElement).title = stats.title;
      }
    });

    // Patch live LOD chips (active level / loaded-of-total / refining) in
    // place — these change every frame without altering tree structure.
    const lodChips = this.contentContainer.querySelectorAll(
      '.luxar-scene-graph__lod[data-lod-path]'
    );
    lodChips.forEach((chip) => {
      const path = (chip as HTMLElement).dataset.lodPath;
      if (!path) return;
      const node = this.getSceneGraphNodeByPath(path);
      if (!node) return;
      const content = lodChipContent(node, this.lodStates.get(path));
      if (content) {
        chip.textContent = content.text;
        (chip as HTMLElement).title = content.title;
      } else {
        // Node no longer has LOD content (e.g. state vanished on reload):
        // clear rather than leaving a stale value on screen.
        chip.textContent = '';
        (chip as HTMLElement).title = '';
      }
    });

    // Re-mark active/inactive substitutive-level rows: the LOD selector can
    // switch levels between structural rebuilds, so classes + tooltips are
    // patched each tick from the parent group's live state.
    const levelRows = this.contentContainer.querySelectorAll(
      '.luxar-scene-graph__node-row[data-level-of]'
    );
    levelRows.forEach((row) => {
      const el = row as HTMLElement;
      const parentPath = el.dataset.levelOf;
      const indexRaw = el.dataset.levelIndex;
      if (!parentPath || indexRaw === undefined) return;
      // Same helper as the initial render so both derive the role identically.
      const role = activeLevelRole(this.lodStates.get(parentPath), Number(indexRaw));
      el.classList.toggle('luxar-scene-graph__node-row--active-level', role === 'active');
      el.classList.toggle('luxar-scene-graph__node-row--inactive-level', role === 'inactive');
      const baseTitle = el.dataset.baseTitle;
      if (baseTitle !== undefined) {
        el.title = `${baseTitle}${levelRoleTitleSuffix(role)}`;
      }
    });

    // Refresh the header LOD/partition summary line.
    const summaryEl = this.contentContainer.querySelector(
      '[data-field="lod-summary"]'
    ) as HTMLElement | null;
    if (summaryEl) {
      summaryEl.textContent = summariseLodStates(
        this.lodStates,
        countAdditiveNodes(this.sceneGraphState.root)
      );
    }
  }

  /**
   * Sync the latest per-path visible counts (from the SceneLoader's
   * visible-counts walk) onto every geometry node in the tree. The walk
   * prunes non-visible subtrees, so a path ABSENT from the latest map
   * (hidden layer, switched-away substitutive level) has its count reset
   * to `undefined` — the tooltip then omits the "(N visible after
   * slicing)" suffix (unknown) instead of showing a stale number.
   */
  private syncVisibleCountsIntoTree(): void {
    const index = this.ensureSceneGraphNodeIndex();
    if (!index) return;
    for (const node of index.values()) {
      const visible = this.visibleCountsByPath.get(node.path);
      if (node.type === 'points') node.visiblePointCount = visible;
      else if (node.type === 'lines') node.visibleSegmentCount = visible;
      else if (node.type === 'gsplats') node.visibleSplatCount = visible;
    }
  }

  /**
   * Build (or reuse) the memoized path→node index for the current tree.
   * Rebuilt only when the tree root reference changes (a wholesale
   * `setSceneGraph`), so repeated per-frame lookups and full-tree sweeps
   * are O(1)/O(N) rather than a fresh DFS each.
   */
  private ensureSceneGraphNodeIndex(): Map<string, SceneGraphNode> | null {
    const root = this.sceneGraphState.root;
    if (!root) return null;
    if (this.sceneGraphNodeIndexRoot !== root) {
      this.sceneGraphNodeIndex.clear();
      const stack: SceneGraphNode[] = [root];
      while (stack.length > 0) {
        const node = stack.pop()!;
        this.sceneGraphNodeIndex.set(node.path, node);
        for (const child of node.children) stack.push(child);
      }
      this.sceneGraphNodeIndexRoot = root;
    }
    return this.sceneGraphNodeIndex;
  }

  /** Resolve a scene-graph node by path via the memoized path→node index. */
  private getSceneGraphNodeByPath(path: string): SceneGraphNode | null {
    return this.ensureSceneGraphNodeIndex()?.get(path) ?? null;
  }

  /**
   * Incrementally update cache tab values without rebuilding DOM.
   * Delegates to `tabs/cache-tab.ts:updateCacheTab`.
   */
  private updateCacheTabValues(): boolean {
    return updateCacheTab(this.contentContainer, this.getCacheMetrics());
  }

  /**
   * Incrementally update memory tab values without rebuilding DOM.
   */
  private updateMemoryTabValues(): boolean {
    if (!this.contentContainer) return false;

    const metrics = this.getMemoryMetrics();

    // Structure validation
    if (!this.contentContainer.querySelector('[data-field="memory-total"]')) return false;

    // GPU pool table
    if (metrics.gpuPool) {
      for (const type of POOLED_GEOMETRY_TYPES) {
        const typeStats = metrics.gpuPool.byType[type];
        const reuseRate = calculateReuseRate(typeStats.allocations, typeStats.reuses);
        const hasData =
          typeStats.allocations > 0 || typeStats.reuses > 0 || typeStats.activeBuffers > 0;

        const reuseEl = this.contentContainer.querySelector(`[data-field="gpu-${type}-reuse"]`);
        if (reuseEl) {
          reuseEl.textContent = hasData ? `${reuseRate.toFixed(0)}%` : '—';
          this.updateColorClass(
            reuseEl as HTMLElement,
            hasData ? getReuseRateColorClass(reuseRate) : getColorClass('dimmed')
          );
        }
        this.patchField(`gpu-${type}-active`, hasData ? `${typeStats.activeBuffers}` : '—');
        this.patchField(`gpu-${type}-pooled`, hasData ? `${typeStats.pooledBuffers}` : '—');
        this.patchField(`gpu-${type}-allocs`, hasData ? `${typeStats.allocations}` : '—');
      }
      this.patchField(
        'gpu-summary',
        `Total: ${metrics.gpuPool.allocations} allocs · ${metrics.gpuPool.reuses} reuses · ${metrics.gpuPool.evictions} evicted`
      );
    }

    // Accumulator table
    for (const type of POOLED_GEOMETRY_TYPES) {
      const stats = metrics.accumulators[type];
      const hasData = stats !== null && stats.capacity > 0;

      this.patchField(
        `acc-${type}-capacity`,
        hasData ? templateFormatNumber(stats!.capacity) : '—'
      );
      this.patchField(`acc-${type}-memory`, hasData ? `${stats!.memoryMB.toFixed(1)}MB` : '—');

      const growsEl = this.contentContainer.querySelector(`[data-field="acc-${type}-grows"]`);
      if (growsEl) {
        growsEl.textContent = hasData ? `${stats!.growthEvents}` : '—';
        // Update warning class for high growth events (clear when <= 5)
        if (hasData && stats!.growthEvents > 5) {
          this.updateColorClass(growsEl as HTMLElement, getColorClass('warning'));
        } else {
          this.updateColorClass(growsEl as HTMLElement, '');
        }
      }
    }

    // Accumulator summary
    const totalAccMemory =
      (metrics.accumulators.points?.memoryMB ?? 0) +
      (metrics.accumulators.lines?.memoryMB ?? 0) +
      (metrics.accumulators.gsplats?.memoryMB ?? 0);
    const totalAccAllocs =
      (metrics.accumulators.points?.allocations ?? 0) +
      (metrics.accumulators.lines?.allocations ?? 0) +
      (metrics.accumulators.gsplats?.allocations ?? 0);
    this.patchField(
      'acc-summary',
      `Total: ${totalAccMemory.toFixed(1)}MB · ${totalAccAllocs} allocations`
    );

    // Overall total
    const totalAllocations = metrics.gpuPool ? metrics.gpuPool.allocations : 0;
    const totalReuses = metrics.gpuPool ? metrics.gpuPool.reuses : 0;
    const overallReuseRate = calculateReuseRate(totalAllocations, totalReuses);
    this.patchField(
      'memory-total',
      `${totalAllocations} allocs · ${overallReuseRate.toFixed(0)}% reuse · ${totalAccMemory.toFixed(1)}MB`
    );

    return true;
  }

  /**
   * Update CSS color classes on an element, replacing any existing
   * `luxar-color--*` class. Delegates to the shared DOM helper.
   */
  private updateColorClass(el: HTMLElement, newColorClass: string): void {
    updateColorClass(el, newColorClass);
  }

  /**
   * Render tabs
   */
  private renderTabs(): string {
    const tabs = [
      {
        id: 'overview',
        label: 'Overview',
        icon: MONITOR_ICONS.overview,
        tooltip:
          'The big picture: how much of the dataset is on screen, memory and query speed, ' +
          'how much data has been downloaded vs served from cache, and the scene graph tree',
      },
      {
        id: 'cache',
        label: 'Cache',
        icon: MONITOR_ICONS.cache,
        tooltip:
          'The three cache tiers that avoid re-downloading data — L0 (decoded, memory), ' +
          'L1 (raw, memory), L2 (disk, survives reloads) — with sizes, hit rates, ' +
          'validation health, and Clear buttons',
      },
      {
        id: 'memory',
        label: 'Memory',
        icon: MONITOR_ICONS.memory,
        tooltip:
          'Where geometry memory goes: GPU buffer pooling (how often buffers are reused ' +
          'instead of reallocated) and the CPU-side accumulators that grow as data streams in',
      },
      {
        id: 'performance',
        label: 'Performance',
        icon: MONITOR_ICONS.performance,
        tooltip:
          'A timing breakdown of each view update — query, load, project, GPU upload — ' +
          'per step and per geometry type, with rows exceeding the 60fps frame budget highlighted',
      },
      {
        id: 'insights',
        label: 'Insights',
        icon: MONITOR_ICONS.insights,
        tooltip:
          'Automatic diagnosis: detected problems and tuning recommendations for loading ' +
          'and caching, ranked by severity',
      },
    ];

    return tabs
      .map(
        (tab) => `
      <button
        class="luxar-data-monitor__tab ${this.uiState.activeTab === tab.id ? 'luxar-data-monitor__tab--active' : ''}"
        data-action="setTab" data-tab-id="${tab.id}" title="${tab.tooltip}"
      >
        ${tab.icon}<span>${tab.label}</span>
      </button>
    `
      )
      .join('');
  }

  /**
   * Render tab content
   */
  private renderTabContent(): string {
    switch (this.uiState.activeTab) {
      case 'overview':
        return this.renderOverviewTab();
      case 'cache':
        return this.renderCacheTab();
      case 'memory':
        return this.renderMemoryTab();
      case 'performance':
        return this.renderPerformanceTab();
      case 'insights':
        return this.renderInsightsTab();
      default:
        return '';
    }
  }

  /**
   * Retry every failed loader via the injected provider (the monitor-side
   * trigger for `SceneLoader.retryAllFailedLoaders`; the loader serializes
   * the batch against its update lock). Guards against double-clicks while
   * a batch is in flight and refreshes the banner on completion.
   */
  private retryFailedLoads(): void {
    const provider = this.failedLoadsProvider;
    if (!provider || this.retryFailedLoadsInFlight) return;
    if (provider.getFailedPaths().length === 0) return;

    this.retryFailedLoadsInFlight = true;
    this.structureDirty = true; // rebuild the overview so the button disables now
    this.updateUI();
    void provider
      .retryAll()
      .then(({ succeeded, failed, deferred }) => {
        if (deferred) {
          // Nothing was retried — a main update holds the serialization
          // lock. Saying "still failing" here would falsely report a failed
          // re-attempt (the pre-fix behavior).
          notifier.toast(
            'Retry deferred — a data update is in progress; try again in a moment.',
            4000
          );
          return;
        }
        if (failed.length === 0) {
          notifier.toast(
            `Recovered ${succeeded.length} failed load${succeeded.length === 1 ? '' : 's'}.`,
            4000
          );
        } else {
          notifier.toast(
            `Retried failed loads: ${succeeded.length} recovered, ${failed.length} still failing.`,
            5000
          );
        }
      })
      .catch((error) => {
        log.warning(
          Modules.DATA_MONITOR,
          `Retry-all failed: ${error instanceof Error ? error.message : String(error)}`
        );
      })
      .finally(() => {
        this.retryFailedLoadsInFlight = false;
        this.structureDirty = true; // re-enable the button / drop the banner
        this.updateUI();
      });
  }

  /**
   * Render overview tab with cleaner visual hierarchy
   */
  private renderOverviewTab(): string {
    const stats = this.getGlobalStats();
    const cacheMetrics = this.getCacheMetrics();

    // Failed-load warning banner (with a Retry action) ahead of the metrics —
    // failures otherwise surface only as transient toasts.
    const failedPaths = this.failedLoadsProvider?.getFailedPaths() ?? [];
    const banner = renderFailedLoadsBanner(failedPaths, this.retryFailedLoadsInFlight);

    // Use the template function for the main content
    const content = banner + renderOverviewContent(stats, cacheMetrics);

    // Replace the loader list placeholder with scene graph tree (or compact loader list if no scene graph)
    if (this.sceneGraphState.root) {
      return content.replace(
        '<div id="loader-list-content"></div>',
        renderSceneGraphTree(this.sceneGraphState, this.expandedNodes, this.lodStates)
      );
    } else {
      return content.replace(
        '<div id="loader-list-content"></div>',
        this.renderCompactLoaderList()
      );
    }
  }

  /**
   * Render comprehensive cache analytics tab
   */
  private renderCacheTab(): string {
    const stats = this.getGlobalStats();
    const cacheMetrics = this.getCacheMetrics();

    // Use the template function
    return renderCacheContent(stats, cacheMetrics, this.collapsedCacheSections);
  }

  /**
   * Render memory tab with GPU buffer pool and accumulator stats
   */
  private renderMemoryTab(): string {
    // Get fresh stats from providers
    const metrics = this.getMemoryMetrics();
    return renderMemoryContent(metrics);
  }

  /**
   * Get memory metrics from providers
   */
  private getMemoryMetrics(): MemoryMetrics {
    return {
      gpuPool: this.gpuBufferPoolProvider?.getStats() ?? null,
      accumulators: Object.fromEntries(
        POOLED_GEOMETRY_TYPES.map((t) => [t, this.accumulatorProviders[t]?.getStats() ?? null])
      ) as MemoryMetrics['accumulators'],
    };
  }

  /**
   * Render performance tab with hierarchical timing panel
   */
  private renderPerformanceTab(): string {
    // Get timing data from profiler
    const timingData = this.profiler?.getTimings();

    if (!timingData) {
      return `
        <div class="luxar-performance-content">
          <div class="luxar-timing-panel luxar-timing-panel--empty">
            <div class="luxar-timing-panel__empty-msg">
              Profiler not connected. Timing data will appear here once the scene is loaded.
            </div>
          </div>
        </div>
      `;
    }

    return `
      <div class="luxar-performance-content">
        ${renderHierarchicalTimingPanel(
          timingData,
          this.profiler?.getRefinementTimings(),
          this.profiler?.getDepthSortTimings()
        )}
      </div>
    `;
  }

  /**
   * Render insights tab
   */
  private renderInsightsTab(): string {
    const recommendations = this.advisor.getRecommendations();
    return renderInsightsContent(recommendations);
  }

  /**
   * Get global statistics
   */
  public getGlobalStats(): GlobalStats {
    let totalElementsLoaded = 0;
    let totalMemory = 0;
    let totalQueries = 0;
    let totalLoads = 0;
    let totalQueryTime = 0;
    let activeSpatial = 0;

    // Per-loader metrics drive genuine per-loader throughput only
    // (cumulative loaded, memory, query stats). Dataset totals and visible
    // counts are sourced from the scene graph below — symmetric across all
    // three geometry types. Progressive multi-LOD nodes connect as a single
    // loader (their adapter re-paths inner events to the node path), so each
    // node contributes exactly one entry here — no per-LOD double-counting.
    const isSpatialType = (t: string | undefined): boolean =>
      t === 'point-spatial-index' || t === 'lines-spatial-index' || t === 'gsplats-spatial-index';

    for (const metrics of this.metrics.values()) {
      totalElementsLoaded += metrics.elementsLoaded;
      totalMemory += metrics.memoryUsed;
      totalQueries += metrics.queries;
      totalLoads += metrics.loads;
      totalQueryTime += metrics.avgQueryTime * metrics.queries;

      if (isSpatialType(metrics.type)) {
        activeSpatial++;
      }
    }

    // Substitutive kind=lod groups connect one loader per leaf level (eager
    // AND lazy levels are cheap-attached + connected up front, each reporting
    // a `*-spatial-index` metric), but only one level renders at a time.
    // Collapse each group's loaders to a single logical layer so the headline
    // counts don't read K× too high. The excess is derived from the loaders
    // *actually present under each group path* — not from the LOD level count
    // — so a level that is itself a multi-leaf subtree (>1 loader per level)
    // is collapsed correctly rather than under-subtracted. Child loaders are
    // registered at scene-graph paths nested under the group path. Excess is 0
    // unless the provider reports kind=lod groups, so plain scenes are
    // unaffected.
    //
    // Two excesses are tracked from matching populations: `lodLoaderExcess`
    // counts *all* loaders under each group (subtracted from `totalLoaders`,
    // which counts all loaders), while `lodSpatialExcess` counts only the
    // spatial-index–typed loaders (subtracted from `activeSpatial`, which is
    // built from spatial-typed metrics only). Drawing each from its own
    // population keeps a future non-spatial loader nested under a LOD group
    // from over-subtracting `activeSpatial`.
    let lodLoaderExcess = 0;
    let lodSpatialExcess = 0;
    for (const [path, s] of this.lodStates) {
      if (s.kind !== 'lod') continue;
      let present = 0;
      let presentSpatial = 0;
      for (const lp of this.loaders.keys()) {
        if (lp === path || lp.startsWith(`${path}/`)) {
          present++;
          if (isSpatialType(this.metrics.get(lp)?.type)) presentSpatial++;
        }
      }
      if (present > 1) lodLoaderExcess += present - 1;
      if (presentSpatial > 1) lodSpatialExcess += presentSpatial - 1;
    }
    const totalLoaders = Math.max(0, this.loaders.size - lodLoaderExcess);
    activeSpatial = Math.max(0, activeSpatial - lodSpatialExcess);

    // Use cached QPS calculation instead of filtering events again
    this.calculateRates();
    const qps = this.cachedRates.queriesPerSec;

    // Dataset totals + visible counts come from the scene graph, identically
    // for points / lines / gsplats. Visible counts are refreshed each update
    // cycle by `updateVisibleCountsInMonitor` after nD clipping / LOD refine.
    // The display layer keeps per-type NAMED fields (each rendered with its own
    // label, unit noun and DOM id), so this is where the kind-keyed aggregation
    // model is projected onto them.
    const { totalByType, visibleByType } = this.sceneGraphState;
    const datasetSize = totalByType.points;
    const visiblePoints = visibleByType.points;

    const datasetSegments = totalByType.lines;
    const visibleSegments = visibleByType.lines;

    const datasetSplats = totalByType.gsplats;
    const visibleSplats = visibleByType.gsplats;

    return {
      totalLoaders,
      activeSpatialLoaders: activeSpatial,
      totalElementsLoaded,
      totalMemory,
      datasetSize, // Total points in all datasets (from zarr metadata)
      visiblePoints, // Currently visible/rendered points
      datasetSegments, // Total segments in all line datasets
      visibleSegments, // Currently visible segments (for lines, typically equals total)
      datasetSplats, // Total splats in all gsplats datasets
      visibleSplats, // Currently visible splats
      totalQueries,
      totalLoads,
      avgQueryTime: totalQueries > 0 ? totalQueryTime / totalQueries : 0,
      queriesPerSecond: qps,
      recommendations: this.advisor.getRecommendations(),
    };
  }

  /**
   * Get cache metrics aggregated across all loaders. Delegates the
   * multi-source roll-up to
   * `data-loading-monitor/metrics/cache.ts:aggregateCacheMetrics`.
   * The aggregator refreshes `this.metrics` snapshots and reads
   * `this.cachedRates` (already updated by `calculateRates()` here).
   */
  private getCacheMetrics(): CacheMetrics {
    this.calculateRates();
    return aggregateCacheMetrics({
      l0Provider: this.l0CacheProvider,
      sliceProvider: this.sliceCacheProvider,
      cacheStatsProvider: this.cacheStatsProvider,
      loaders: this.loaders,
      metricsCache: this.metrics,
      rates: this.cachedRates,
      telemetryState: this.cacheTelemetryState,
    });
  }

  /**
   * Clean old events based on age
   */
  private cleanOldEvents(): void {
    const cutoff = Date.now() - this.maxEventAge;
    const originalLength = this.events.length;
    this.events = this.events.filter((e) => e.timestamp > cutoff);

    // Log cleanup info only if profiling is enabled
    if (this.events.length < originalLength && this.config.enableProfiling) {
      log.info(Modules.DATA_MONITOR, `Cleaned ${originalLength - this.events.length} old events`);
    }
  }

  /**
   * Calculate all rates with caching. Delegates to
   * `data-loading-monitor/metrics/rates.ts:calculateRates`.
   */
  private calculateRates(): void {
    calculateRates({
      now: Date.now(),
      events: this.events,
      rateWindowMs: MonitorLimits.rateCalculationWindow,
      bandwidthWindowMs: MonitorLimits.bandwidthCalculationWindow,
      cacheTimeoutMs: this.ratesCacheTimeout,
      rates: this.cachedRates,
    });
  }

  private renderCompactLoaderList(): string {
    const loaderEntries = Array.from(this.metrics.entries());

    if (loaderEntries.length === 0) {
      return '<div class="luxar-loader-empty">No active loaders</div>';
    }

    return loaderEntries.map(([path, metrics]) => renderLoaderItem(path, metrics)).join('');
  }

  // Public API

  public show(): void {
    if (this.panel) {
      // Clear the inline display set by hide() rather than hard-coding
      // 'block': the size class governs layout (--expanded → flex column,
      // --compact → default block). Forcing 'block' here overrode
      // `.luxar-data-monitor--expanded { display: flex }`, which broke the
      // flex-column scroll contract and let the scene-graph tree spill past
      // the panel's max-height instead of scrolling inside __content.
      this.panel.style.display = '';
      this.uiState.isVisible = true;
      // Always update immediately when showing to get fresh metrics
      this.updateUI();
      this.startUpdating();
    }
  }

  public hide(): void {
    if (this.panel) {
      this.panel.style.display = 'none';
      this.uiState.isVisible = false;
      this.stopUpdating();
    }
  }

  public toggle(): void {
    if (this.uiState.isVisible) {
      this.hide();
    } else {
      this.show();
    }
  }

  /**
   * Cycle through monitor states: hidden → mini → expanded → hidden
   * This implements the three-state cycling as per the comprehensive plan
   */
  public cycleState(): void {
    if (!this.uiState.isVisible) {
      // Hidden → Mini (show in compact view)
      this.uiState.isExpanded = false;
      this.show();
      // Update panel styles for compact mode
      if (this.panel) {
        this.updatePanelClasses();
      }
      log.info(Modules.DATA_MONITOR, 'Monitor state: Mini view');
    } else if (!this.uiState.isExpanded) {
      // Mini → Expanded
      this.expand();
      log.info(Modules.DATA_MONITOR, 'Monitor state: Expanded view');
    } else {
      // Expanded → Hidden
      this.hide();
      log.info(Modules.DATA_MONITOR, 'Monitor state: Hidden');
    }
  }

  public expand(): void {
    this.uiState.isExpanded = true;
    // Update panel styles to apply fixed width
    if (this.panel) {
      this.updatePanelClasses();
    }
    // Entrance reveal on opening the full panel (like a tab switch).
    this.animateNextBuild = true;
    // Force rebuild of structure when expanding
    this.contentContainer = null;
    this.updateUI();
  }

  public minimize(): void {
    this.uiState.isExpanded = false;
    // Update panel styles to remove fixed width
    if (this.panel) {
      this.updatePanelClasses();
    }
    this.updateUI();
  }

  public collapse(): void {
    this.minimize();
  }

  public isVisible(): boolean {
    return this.uiState.isVisible;
  }

  public isExpanded(): boolean {
    return this.uiState.isExpanded;
  }

  public getRecentEvents(): MonitorEvent[] {
    return [...this.events];
  }

  public getLoaderMetrics(path: string): LoaderMetrics | undefined {
    return this.metrics.get(path);
  }

  public getRecommendations(): Recommendation[] {
    return this.advisor.getRecommendations();
  }

  public setActiveTab(tab: string): void {
    // Use type guard for proper validation
    if (isValidTab(tab)) {
      this.uiState.activeTab = tab;
      // Entrance animation is reserved for tab switches — structureDirty
      // rebuilds during live updates (tree toggles, banner changes) must
      // not replay the staggered reveal (it reads as flicker).
      this.animateNextBuild = true;
      // Rebuild structure when tab changes (tabs need to show active state)
      this.contentContainer = null; // Force rebuild
      this.updateUI();
    } else {
      log.warning(Modules.DATA_MONITOR, `Invalid tab: ${tab}`);
    }
  }

  public setTimeRange(range: string): void {
    this.uiState.timeRange = parseInt(range);
    // Timeline removed - time range is tracked but no longer used
    this.updateUI();
  }

  /**
   * Start the polling loop for periodic updates.
   * Called when the monitor becomes visible.
   */
  private startUpdating(): void {
    // Update immediately, then start polling
    this.updateUI();
    this.pollingLoop.start();
  }

  /**
   * Stop the polling loop.
   * Called when the monitor is hidden.
   */
  private stopUpdating(): void {
    this.pollingLoop.stop();
  }

  // All styling now handled by CSS classes in data-loading-monitor.css

  public dispose(): void {
    const errors: Error[] = [];

    // Step 1: Stop polling loop and clear event queue (safe operation)
    try {
      this.stopUpdating();
      this.eventQueue.clear();
    } catch (error) {
      errors.push(new Error(`Failed to stop polling loop: ${error}`));
    }

    // Step 2: Disconnect all loaders (critical for memory cleanup)
    for (const [path, loader] of this.loaders.entries()) {
      try {
        // Check if loader still has the removeEventListener method
        if (loader && typeof loader.removeEventListener === 'function') {
          loader.removeEventListener(this.eventListener);
        }
      } catch (error) {
        // Log but continue - we want to attempt cleanup of other loaders
        errors.push(new Error(`Failed to disconnect loader '${path}': ${error}`));
      }
    }

    // Clear the loaders map regardless of individual failures
    try {
      this.loaders.clear();
    } catch (error) {
      errors.push(new Error(`Failed to clear loaders map: ${error}`));
    }

    // Step 2b: Drop scene-bound provider closures.
    // disconnectAllLoaders() does this on reload, but a direct dispose()
    // bypasses that path. The closures reference the disposed
    // SceneLoader's `cachingStore` / `l0Cache` / etc.; any stale debug
    // or external poll into the disposed monitor would otherwise NPE.
    try {
      this.resetSceneProviders();
    } catch (error) {
      errors.push(new Error(`Failed to reset scene providers: ${error}`));
    }

    // Step 2c: Clear scene graph display state. A stale monitor
    // reference reading `getSceneGraph()` after dispose would otherwise
    // return the previous scene's tree.
    try {
      this.resetSceneGraphState();
    } catch (error) {
      errors.push(new Error(`Failed to reset scene graph state: ${error}`));
    }

    // Step 3: Remove UI panel (important for DOM cleanup)
    if (this.panel) {
      // Remove event listeners - non-critical if they fail
      try {
        this.panel.removeEventListener('click', this.uiEventHandler);
      } catch (error) {
        // Non-critical - log but continue
        errors.push(new Error(`Failed to remove click listener: ${error}`));
      }

      try {
        this.panel.removeEventListener('change', this.uiEventHandler);
      } catch (error) {
        // Non-critical - log but continue
        errors.push(new Error(`Failed to remove change listener: ${error}`));
      }

      // Remove panel from DOM - critical for cleanup
      try {
        // Check if panel is still in the DOM before removing
        if (this.panel.parentNode) {
          this.panel.remove();
        }
      } catch (error) {
        // Try alternative removal method
        try {
          this.panel.parentNode?.removeChild(this.panel);
        } catch (fallbackError) {
          errors.push(
            new Error(`Failed to remove panel from DOM: ${error}, fallback: ${fallbackError}`)
          );
        }
      }

      // Clear reference regardless of removal success
      this.panel = null;
    }

    // Step 4: Clean up components
    try {
      if (this.advisor && typeof this.advisor.dispose === 'function') {
        this.advisor.dispose();
      }
    } catch (error) {
      errors.push(new Error(`Failed to dispose advisor: ${error}`));
    }

    // Step 5: Clear remaining state to prevent memory leaks
    try {
      this.events = [];
      this.metrics.clear();
      this.queries.clear();
      // Reset UI state so isVisible() / isExpanded can't report stale
      // truthy values after dispose.
      this.uiState.isVisible = false;
      this.uiState.isExpanded = false;
      this.contentContainer = null;
      this.expandedNodes.clear();
    } catch (error) {
      errors.push(new Error(`Failed to clear internal state: ${error}`));
    }

    // Step 6: Report any errors that occurred during disposal
    if (errors.length > 0) {
      // Log all errors for debugging
      log.warning(Modules.DATA_LOADING_MONITOR, 'Disposal completed with errors:');
      errors.forEach((error, index) => {
        log.warning(Modules.DATA_LOADING_MONITOR, `  ${index + 1}. ${error.message}`);
      });

      // Optionally throw a composite error if critical failures occurred
      const criticalErrors = errors.filter(
        (e) =>
          e.message.includes('Failed to clear loaders') ||
          e.message.includes('Failed to clear internal state')
      );

      if (criticalErrors.length > 0) {
        throw new Error(
          `DataLoadingMonitor disposal failed with ${criticalErrors.length} critical error(s): ` +
            criticalErrors.map((e) => e.message).join('; ')
        );
      }
    }
  }
}
