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
  LODProgressProvider,
  DrawOrderProvider,
  DensityProvider,
  MemoryMetrics,
} from '../types/data-monitor-types';

import { aggregateCacheMetrics } from './data-loading-monitor/metrics/cache';
import { aggregateGlobalStats } from './data-loading-monitor/metrics/global-stats';
import { aggregateMemoryMetrics } from './data-loading-monitor/metrics/memory';
import { calculateRates } from './data-loading-monitor/metrics/rates';
import { createCacheActions } from './data-loading-monitor/cache-actions';
import { updateCacheTab } from './data-loading-monitor/tabs/cache';
import { updateMemoryTab } from './data-loading-monitor/tabs/memory';
import { updateOverviewTab } from './data-loading-monitor/tabs/overview';
import { LoadingAdvisor } from './data-loading-monitor/advisor';
import { EventQueue } from './data-loading-monitor/event-queue';
import { PollingLoop } from './data-loading-monitor/polling-loop';
import { MonitorProviderRegistry } from './data-loading-monitor/providers';
import { SceneGraphModel } from './data-loading-monitor/scene-graph-model';
import { updateSceneGraphBadges } from './data-loading-monitor/tabs/scene-graph-badges';
import { compactTooltip, presentHeadlineCounts } from './data-loading-monitor/headline-counts';
import { escapeHtml } from '../utils/escape-html';
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
} from './data-loading-monitor/templates/overview';
import { renderCacheContent, CACHE_SECTION_KEYS } from './data-loading-monitor/templates/cache';
import { renderMemoryContent } from './data-loading-monitor/templates/memory';
import { renderInsightsContent } from './data-loading-monitor/templates/insights';
import { renderSceneGraphTree } from './data-loading-monitor/templates/scene-graph';
import {
  formatNumber as templateFormatNumber,
  formatBytes as templateFormatBytes,
} from './data-loading-monitor/templates/format';
import { MONITOR_ICONS } from './data-loading-monitor/templates/primitives';

import {
  renderHierarchicalTimingPanel,
  attachTimingPanelHandlers,
  updateTimingPanelValues,
} from './data-loading-monitor/timing-panel';

import type { UpdateProfiler } from '../profiling/update-profiler';
// A session whose SortWorker never came up draws every order-dependent
// layer in storage order; without this the only trace is one console error.
import { isDepthSortAvailable } from '../rendering/depth-sort-coordinator';
import type { PooledGeometryType, AccumulatorProvider } from '../types/data-monitor-types';
import type { GeometryTypeName } from '../types/format-contract';

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

  private readonly providers = new MonitorProviderRegistry(() => {
    this.structureDirty = true;
  });
  private readonly cacheActions = createCacheActions(this.providers, () => this.updateUI());
  private readonly sceneGraphModel = new SceneGraphModel(() => {
    this.structureDirty = true;
  });

  /** In-flight guard so the banner's Retry button can't stack batches. */
  private retryFailedLoadsInFlight = false;
  /** Last-rendered failed-loads state; a change marks the overview structure dirty. */
  private lastFailedLoadsSignature = '';
  // DOM element references for efficient updates (avoids full innerHTML replacement)
  private contentContainer: HTMLElement | null = null;

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

  // Flag to force a full DOM rebuild on next update (set by structural changes like
  // tree node toggle, scene graph mutation). Cleared after rebuild.
  private structureDirty = false;

  constructor(container: HTMLElement, config?: Partial<MonitorConfig>) {
    this.container = container;
    this.config = {
      position: 'top-right',
      updateInterval: MonitorTimings.defaultUpdateInterval,
      maxEvents: MonitorLimits.maxEvents,
      showRecommendations: true,
      autoExpand: false,
      enableProfiling: true,
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
    this.providers.setFailedLoadsProvider(provider);
  }

  /**
   * Set the cache stats provider for L1/L2 cache monitoring.
   * This enables the monitor to display actual cache statistics.
   */
  public setCacheStatsProvider(provider: CacheStatsProvider | null): void {
    this.providers.setCacheStatsProvider(provider);
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
    this.providers.setCacheTelemetryState(state);
  }

  /**
   * Set the L0 decompressed chunk cache provider for L0 cache monitoring.
   * This enables the monitor to display L0 cache statistics in the Cache tab.
   */
  public setL0CacheProvider(
    provider: { getStats: () => CacheMetrics['l0']; clear: () => void } | null
  ): void {
    this.providers.setL0CacheProvider(provider);
  }

  /**
   * Register the SliceCache ("S-cache") stats/clear provider so the Cache tab
   * shows its usage and hit rate. Mirrors {@link setL0CacheProvider}.
   */
  public setSliceCacheProvider(
    provider: { getStats: () => CacheMetrics['slice']; clear: () => void } | null
  ): void {
    this.providers.setSliceCacheProvider(provider);
  }

  /**
   * Set the GPU buffer pool provider for Memory tab stats.
   * The provider should have a getStats() method that returns PoolStats.
   */
  public setGPUBufferPoolProvider(
    provider: { getStats: () => MemoryMetrics['gpuPool'] } | null
  ): void {
    this.providers.setGPUBufferPoolProvider(provider);
  }

  /**
   * Set the update profiler for Performance tab timing display.
   * The profiler tracks hierarchical timing of scene updates.
   */
  public setProfiler(profiler: UpdateProfiler | null): void {
    this.providers.setProfiler(profiler);
  }

  /**
   * Set the LOD-progress provider for live LOD / refinement / residency
   * state in the scene-graph tree. Polled each tick. Passing a provider
   * marks the structure dirty so the tree re-renders with chip slots.
   */
  public setLODProgressProvider(provider: LODProgressProvider | null): void {
    this.providers.setLODProgressProvider(provider);
  }

  /**
   * Set the draw-order provider for live per-mesh compositing state (blending
   * bucket / depthWrite / renderOrder) in the scene-graph tree. Polled each
   * tick. Passing a provider marks the structure dirty so the tree re-renders
   * with the draw-order chip slot.
   */
  public setDrawOrderProvider(provider: DrawOrderProvider | null): void {
    this.providers.setDrawOrderProvider(provider);
  }

  /**
   * Live density-guard state per node (the tree's lattice-glyph `1/K` density chip). App-scoped:
   * wired once by the init pipeline and kept across scene switches.
   */
  public setDensityProvider(provider: DensityProvider | null): void {
    this.providers.setDensityProvider(provider);
  }

  /**
   * Set an accumulator provider for Memory tab stats.
   * @param type - Which accumulator, one of `POOLED_GEOMETRY_TYPES`
   * @param provider - The accumulator with a getStats() method
   */
  public setAccumulatorProvider(
    type: PooledGeometryType,
    provider: AccumulatorProvider | null
  ): void {
    this.providers.setAccumulatorProvider(type, provider);
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
    this.providers.resetSceneProviders();
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
    this.sceneGraphModel.resetSceneGraphState();
    this.providers.clearDrawOrderStates();
  }

  /**
   * Clear L0 decompressed chunk cache.
   */
  public clearL0Cache(): void {
    this.cacheActions.clearL0Cache();
  }

  /**
   * Clear the SliceCache ("S-cache").
   */
  public clearSliceCache(): void {
    this.cacheActions.clearSliceCache();
  }

  /**
   * Clear L1 memory cache.
   */
  public clearL1Cache(): void {
    this.cacheActions.clearL1Cache();
  }

  /**
   * Clear L2 OPFS cache. Asks for confirmation first since the L2
   * persistent cache requires a network re-fetch to repopulate. The
   * confirm dialog can be skipped by passing `{ skipConfirm: true }`
   * — used by tests and by the debug API where the caller has already
   * confirmed intent.
   */
  public async clearL2Cache(opts?: { skipConfirm?: boolean }): Promise<void> {
    await this.cacheActions.clearL2Cache(opts);
  }

  /**
   * Clear all caches (L0 + L1 + L2). Confirmation dialog can be
   * skipped via `{ skipConfirm: true }`.
   */
  public async clearAllCaches(opts?: { skipConfirm?: boolean }): Promise<void> {
    await this.cacheActions.clearAllCaches(opts);
  }

  /**
   * Set the scene graph for display in the monitor.
   * Called by SceneLoader after loading scene.
   */
  public setSceneGraph(root: SceneGraphNode): void {
    this.sceneGraphModel.setSceneGraph(root);
    const stats = this.sceneGraphModel.getSceneGraph();
    log.info(
      Modules.DATA_MONITOR,
      `Scene graph updated: ${stats.totalNodes} nodes, ${stats.totalByType.points} points, ${stats.totalByType.lines} segments`
    );
    if (this.uiState.isVisible) {
      this.updateUI();
    }
  }

  /**
   * Get current scene graph state.
   */
  public getSceneGraph(): SceneGraphState {
    return this.sceneGraphModel.getSceneGraph();
  }

  /**
   * Toggle expansion state of a node in the scene graph tree.
   */
  public toggleNodeExpansion(path: string): void {
    this.sceneGraphModel.toggleNodeExpansion(path);
    this.updateUI();
  }

  /**
   * Check if a node is expanded in the tree view.
   */
  public isNodeExpanded(path: string): boolean {
    return this.sceneGraphModel.isNodeExpanded(path);
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
    this.sceneGraphModel.updateVisibleCount(type, count);
  }

  public updateDroppedElementCount(count: number): void {
    this.sceneGraphModel.updateDroppedElementCount(count);
  }

  /**
   * Per-node visible counts after nD slicing, keyed by scene-graph path.
   * Pushed by the SceneLoader's visible-counts walk (only rendered meshes
   * contribute). Merged into the tree nodes so badge tooltips can show
   * "(N visible after slicing)" per layer; the merge happens in
   * `SceneGraphModel.syncVisibleCountsIntoTree()`, run on the poll tick
   * immediately before the incremental badge patch. Nodes whose path is
   * absent from the latest map (the walk prunes non-visible subtrees) have
   * their count cleared so tooltips never show a stale number.
   */
  public updateVisibleCountsByPath(counts: ReadonlyMap<string, number>): void {
    this.sceneGraphModel.updateVisibleCountsByPath(counts);
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

    // 3. Refresh the live LOD / refinement / residency and draw-order snapshots.
    this.providers.refreshLiveSnapshots();

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
      errors: 0,
      elementsLoaded: 0,
      bytesLoaded: 0,
      visibleElements: 0,
      avgQueryTime: 0,
      avgLoadTime: 0,
      memoryUsed: 0,
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
        this.clearL2Cache().catch((error: unknown) => {
          this.reportCacheClearFailure('L2 cache', error);
        });
        break;
      case 'clearAll':
        this.clearAllCaches().catch((error: unknown) => {
          this.reportCacheClearFailure('all caches', error);
        });
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

  private reportCacheClearFailure(label: string, error: unknown): void {
    log.warning(Modules.DATA_MONITOR, `Failed to clear ${label}`, error);
    notifier.toast(`Failed to clear ${label}`);
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
    // luxar-panel-pop: same-string className rewrites do NOT restart the CSS
    // animation, so live data updates never re-trigger the pop — only
    // display:none → block (show/expand) does.
    const classes = ['luxar-data-monitor', 'luxar-glass-surface', 'luxar-panel-pop'];

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
   * type actually present in the scene, from the same shared headline table
   * the Overview tab's hero cards use (`headline-counts.ts`) — so the badge
   * can't disagree with the expanded panel about which types exist, and a
   * lines-, gsplats- or mesh-only dataset no longer mislabels its elements
   * as "points". Falls back to a points entry when nothing has loaded yet.
   */
  private buildCompactGeomSummary(stats: GlobalStats): string {
    const entry = (geom: string, count: number, noun: string, unit: string): string =>
      `<span data-geom="${geom}" title="${escapeHtml(compactTooltip(noun))}">${templateFormatNumber(count)} ${unit}</span>`;

    // `data-geom` keys off the geometry TYPE, not the unit noun, so the
    // attribute stays stable if a unit label is ever reworded.
    const entries = presentHeadlineCounts(stats).map((c) =>
      entry(c.type, c.visible, c.noun, c.unit)
    );
    // Nothing loaded yet → show a points placeholder so the row isn't empty.
    if (entries.length === 0) {
      return entry('points', stats.visiblePoints, 'points', 'pts');
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

        <span class="luxar-monitor-compact__memory" title="CPU memory currently held by loaded geometry data, across all layers (points + lines + gsplats + mesh)">
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
        <div class="luxar-data-monitor__header luxar-panel-header">
          <div class="luxar-data-monitor__title">Data Loading Monitor</div>
          <div class="luxar-data-monitor__controls">
            <button class="luxar-panel-close" data-action="minimize" title="Minimize" aria-label="Minimize the data monitor"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 12h12"/></svg></button>
            <button class="luxar-panel-close" data-action="hide" title="Close" aria-label="Close the data monitor"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12"/><path d="M18 6L6 18"/></svg></button>
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
      (this.providers.failedLoadsProvider?.getFailedPaths() ?? []).join('|') +
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
          if (this.providers.profiler) {
            const timingData = this.providers.profiler.getTimings();
            const refinementData = this.providers.profiler.getRefinementTimings();
            const depthSortData = this.providers.profiler.getDepthSortTimings();
            if (timingData.count > 0 || refinementData.count > 0 || depthSortData.count > 0) {
              updated = updateTimingPanelValues(
                this.contentContainer,
                timingData,
                refinementData,
                depthSortData,
                !isDepthSortAvailable()
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
   * Incrementally update overview tab values without rebuilding DOM.
   * Updates primary metric cards, secondary metrics, and scene graph badges.
   */
  private updateOverviewTabValues(): boolean {
    return updateOverviewTab(
      this.contentContainer,
      this.getGlobalStats(),
      this.getCacheMetrics(),
      () => {
        // Badge tooltips read visible counts directly from the aliased tree nodes.
        this.sceneGraphModel.syncVisibleCountsIntoTree();
        updateSceneGraphBadges(
          this.contentContainer!,
          this.sceneGraphModel,
          this.providers.lodStates,
          this.providers.drawOrderStates,
          this.providers.densityStates
        );
      }
    );
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
    return updateMemoryTab(this.contentContainer, this.getMemoryMetrics());
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
    const provider = this.providers.failedLoadsProvider;
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
    const failedPaths = this.providers.failedLoadsProvider?.getFailedPaths() ?? [];
    const banner = renderFailedLoadsBanner(failedPaths, this.retryFailedLoadsInFlight);

    // Use the template function for the main content
    const content = banner + renderOverviewContent(stats, cacheMetrics);

    // Replace the loader list placeholder with scene graph tree (or compact loader list if no scene graph)
    const sceneGraphState = this.sceneGraphModel.getSceneGraph();
    if (sceneGraphState.root) {
      return content.replace(
        '<div id="loader-list-content"></div>',
        renderSceneGraphTree(
          sceneGraphState,
          this.sceneGraphModel.expandedNodes,
          this.providers.lodStates,
          this.providers.drawOrderStates,
          this.providers.densityStates
        )
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
    return aggregateMemoryMetrics(this.providers);
  }

  /**
   * Render performance tab with hierarchical timing panel
   */
  private renderPerformanceTab(): string {
    // Get timing data from profiler
    const timingData = this.providers.profiler?.getTimings();

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
          this.providers.profiler?.getRefinementTimings(),
          this.providers.profiler?.getDepthSortTimings(),
          !isDepthSortAvailable()
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
    this.calculateRates();
    return aggregateGlobalStats({
      metrics: this.metrics,
      loaders: this.loaders,
      lodStates: this.providers.lodStates,
      rates: this.cachedRates,
      sceneGraph: this.sceneGraphModel.getSceneGraph(),
      recommendations: this.advisor.getRecommendations(),
    });
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
      l0Provider: this.providers.l0CacheProvider,
      sliceProvider: this.providers.sliceCacheProvider,
      cacheStatsProvider: this.providers.cacheStatsProvider,
      loaders: this.loaders,
      metricsCache: this.metrics,
      rates: this.cachedRates,
      telemetryState: this.providers.cacheTelemetryState,
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
      this.sceneGraphModel.clearExpandedNodes();
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
