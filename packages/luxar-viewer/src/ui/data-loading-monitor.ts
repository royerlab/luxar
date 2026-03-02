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
  SceneGraphNode,
  SceneGraphState,
} from './data-monitor-types';

// Performance timeline removed - now using hierarchical timing panel
import { LoadingAdvisor } from './components/loading-advisor';
import { EventQueue } from './components/event-queue';
import { PollingLoop } from './components/polling-loop';
import { log, Modules } from '../utils/log';
import { config } from '../config';

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
  renderCacheContent,
  renderMemoryContent,
  renderInsightsContent,
  renderSceneGraphTree,
  formatNumber as templateFormatNumber,
  formatBytes as templateFormatBytes,
  getColorClass,
  getCacheMemoryColorClass,
  calculateReuseRate,
  getReuseRateColorClass,
  type MemoryMetrics,
} from './data-monitor-templates';

import {
  renderHierarchicalTimingPanel,
  attachTimingPanelHandlers,
  updateTimingPanelValues,
} from './components/hierarchical-timing-panel';

import type { UpdateProfiler } from '../profiling/update-profiler';

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

  // GPU buffer pool reference for dynamic stats retrieval
  private gpuBufferPoolProvider: { getStats: () => MemoryMetrics['gpuPool'] } | null = null;

  // Update profiler reference for hierarchical timing display
  private profiler: UpdateProfiler | null = null;

  // Accumulator providers for dynamic stats retrieval
  private accumulatorProviders: {
    points: { getStats: () => NonNullable<MemoryMetrics['accumulators']['points']> } | null;
    lines: { getStats: () => NonNullable<MemoryMetrics['accumulators']['lines']> } | null;
    gsplats: { getStats: () => NonNullable<MemoryMetrics['accumulators']['gsplats']> } | null;
  } = {
    points: null,
    lines: null,
    gsplats: null,
  };

  // DOM element references for efficient updates (avoids full innerHTML replacement)
  private contentContainer: HTMLElement | null = null;

  // Scene graph state
  private sceneGraphState: SceneGraphState = {
    root: null,
    totalNodes: 0,
    pointsNodes: 0,
    linesNodes: 0,
    gsplatsNodes: 0,
    totalPoints: 0,
    totalSegments: 0,
    visibleSegments: 0,
    totalSplats: 0,
    visibleSplats: 0,
  };

  // Track expanded nodes in scene graph tree (by path)
  private expandedNodes = new Set<string>(['/']);

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

    // Update UI to reflect cleared state if visible
    if (this.uiState.isVisible) {
      this.updateUI();
    }

    log.info(Modules.DATA_MONITOR, 'All loaders disconnected and monitor state reset');
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
   * Set an accumulator provider for Memory tab stats.
   * @param type - The type of accumulator ('points', 'lines', or 'gsplats')
   * @param provider - The accumulator with a getStats() method
   */
  public setAccumulatorProvider(
    type: 'points' | 'lines' | 'gsplats',
    provider: { getStats: () => NonNullable<MemoryMetrics['accumulators']['points']> } | null
  ): void {
    this.accumulatorProviders[type] = provider;
    if (provider) {
      log.info(Modules.DATA_MONITOR, `${type} accumulator provider connected`);
    }
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
   * Clear L2 OPFS cache.
   */
  public async clearL2Cache(): Promise<void> {
    if (this.cacheStatsProvider) {
      await this.cacheStatsProvider.clearL2();
      log.info(Modules.DATA_MONITOR, 'L2 cache cleared');
      this.updateUI();
    }
  }

  /**
   * Clear all caches (L0 + L1 + L2).
   */
  public async clearAllCaches(): Promise<void> {
    // Clear L0 first (synchronous)
    if (this.l0CacheProvider) {
      this.l0CacheProvider.clear();
    }
    // Clear L1 + L2 (L2 is async)
    if (this.cacheStatsProvider) {
      await this.cacheStatsProvider.clearAll();
    }
    log.info(Modules.DATA_MONITOR, 'All caches cleared (L0 + L1 + L2)');
    this.updateUI();
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
      `Scene graph updated: ${stats.totalNodes} nodes, ${stats.totalPoints} points, ${stats.totalSegments} segments`
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
    let pointsNodes = node.type === 'points' ? 1 : 0;
    let linesNodes = node.type === 'lines' ? 1 : 0;
    let gsplatsNodes = node.type === 'gsplats' ? 1 : 0;
    let totalPoints = node.pointCount || 0;
    let totalSegments = node.segmentCount || 0;
    let totalSplats = node.splatCount || 0;

    for (const child of node.children) {
      const childStats = this.calculateSceneGraphStats(child);
      totalNodes += childStats.totalNodes;
      pointsNodes += childStats.pointsNodes;
      linesNodes += childStats.linesNodes;
      gsplatsNodes += childStats.gsplatsNodes;
      totalPoints += childStats.totalPoints;
      totalSegments += childStats.totalSegments;
      totalSplats += childStats.totalSplats;
    }

    // Initialize visible counts to totals (will be updated by scene loader)
    return {
      totalNodes,
      pointsNodes,
      linesNodes,
      gsplatsNodes,
      totalPoints,
      totalSegments,
      visibleSegments: totalSegments,
      totalSplats,
      visibleSplats: totalSplats,
    };
  }

  /**
   * Update the count of currently visible line segments.
   * Called by SceneLoader after processing lines with nD clipping.
   */
  public updateVisibleSegments(count: number): void {
    this.sceneGraphState.visibleSegments = count;
  }

  /**
   * Update the count of currently visible gsplats.
   * Called by SceneLoader after processing gsplats with nD clipping.
   */
  public updateVisibleSplats(count: number): void {
    this.sceneGraphState.visibleSplats = count;
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

    // 3. Update UI (this also pulls fresh stats from providers)
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
        metrics.pointsLoaded += event.data.points || 0;
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
      points: event.data.points,
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
      pointsLoaded: 0,
      bytesLoaded: 0,
      datasetSize: 0,
      visiblePoints: 0,
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
    const target = event.target as HTMLElement;
    if (!target) return;

    // Check for data-action attribute
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
      case 'clearL1':
        this.clearL1Cache();
        break;
      case 'clearL2':
        this.clearL2Cache();
        break;
      case 'clearAll':
        this.clearAllCaches();
        break;
      case 'toggleNode': {
        const nodePath = target.dataset.nodePath;
        if (nodePath) {
          this.toggleNodeExpansion(nodePath);
        }
        break;
      }
    }
  }

  /**
   * Create UI elements
   */
  private createUI(): void {
    // Styles now in src/styles/components/data-loading-monitor.css

    // Create main panel
    this.panel = document.createElement('div');
    this.updatePanelClasses();

    // Add event delegation listeners
    this.panel.addEventListener('click', this.uiEventHandler);
    this.panel.addEventListener('change', this.uiEventHandler);

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

    // Base class
    const classes = ['luxar-data-monitor'];

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
   * Try to update just the compact view values without replacing innerHTML.
   * This preserves event handlers and prevents interaction loss during polling updates.
   * @returns true if incremental update succeeded, false if full rebuild is needed
   */
  private updateCompactViewValues(): boolean {
    const pointsEl = this.panel?.querySelector(
      '.luxar-monitor-compact .luxar-monitor-compact__points'
    );
    const memoryEl = this.panel?.querySelector(
      '.luxar-monitor-compact .luxar-monitor-compact__memory'
    );
    const qpsEl = this.panel?.querySelector('.luxar-monitor-compact .luxar-monitor-compact__qps');

    // If structure doesn't exist yet, need full rebuild
    if (!pointsEl || !memoryEl || !qpsEl) return false;

    // Update metrics from all loaders
    for (const [path, loader] of this.loaders) {
      const metrics = loader.getMetrics();
      this.metrics.set(path, metrics);
    }

    const stats = this.getGlobalStats();
    pointsEl.textContent = templateFormatNumber(stats.visiblePoints);
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
        <span class="luxar-monitor-compact__type" title="Loading mode">
          ${hasSpatialIndex ? '🔍' : '📦'}
        </span>

        <span class="luxar-monitor-compact__points" title="Visible points">
          ${templateFormatNumber(stats.visiblePoints)}
        </span>

        <span class="luxar-monitor-compact__memory" title="Memory usage">
          ${templateFormatBytes(stats.totalMemory)}
        </span>

        <span class="luxar-monitor-compact__qps" title="Queries per second">
          ${stats.queriesPerSecond.toFixed(1)}/s
        </span>

        ${hasErrors ? '<span class="luxar-monitor-compact__alert" title="Errors detected">🔴</span>' : ''}
        ${hasWarnings ? '<span class="luxar-monitor-compact__alert" title="Warnings">🟡</span>' : ''}

        <button class="luxar-data-monitor__expand-btn" data-action="expand" title="Show details">
          ⊞
        </button>
      </div>
    `;
  }

  /**
   * Build the detailed view structure once (called on expand or tab change)
   */
  private buildDetailedViewStructure(): void {
    if (!this.panel) return;

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
        <div class="luxar-data-monitor__content">
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
            if (timingData.count > 0) {
              updated = updateTimingPanelValues(this.contentContainer, timingData);
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
      // Full content rebuild (structure changed or first render for this tab)
      this.contentContainer.innerHTML = this.renderTabContent();
      this.attachTabHandlers();
    }
  }

  // ─── Per-tab incremental value update functions ───────────────────────

  /**
   * Helper: update a single element's textContent by data-field attribute.
   * Returns false if the element was not found.
   */
  private patchField(field: string, text: string): boolean {
    const el = this.contentContainer?.querySelector(`[data-field="${field}"]`);
    if (!el) return false;
    el.textContent = text;
    return true;
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

    if (hasPoints) {
      const pct =
        stats.datasetSize > 0 ? ((stats.visiblePoints / stats.datasetSize) * 100).toFixed(1) : '0';
      this.patchField('visible-points', templateFormatNumber(stats.visiblePoints));
      this.patchField(
        'visible-points-sub',
        `${pct}% of ${templateFormatNumber(stats.datasetSize)}${suffix}`
      );
    }
    if (hasLines) {
      const pct =
        stats.datasetSegments > 0
          ? ((stats.visibleSegments / stats.datasetSegments) * 100).toFixed(1)
          : '0';
      this.patchField('visible-lines', templateFormatNumber(stats.visibleSegments));
      this.patchField(
        'visible-lines-sub',
        `${pct}% of ${templateFormatNumber(stats.datasetSegments)}${suffix}`
      );
    }
    if (hasGSplats) {
      const pct =
        stats.datasetSplats > 0
          ? ((stats.visibleSplats / stats.datasetSplats) * 100).toFixed(1)
          : '0';
      this.patchField('visible-splats', templateFormatNumber(stats.visibleSplats));
      this.patchField(
        'visible-splats-sub',
        `${pct}% of ${templateFormatNumber(stats.datasetSplats)}${suffix}`
      );
    }

    // Update secondary metrics
    this.patchField('memory-used', templateFormatBytes(cacheMetrics.totalCacheMemory));
    this.patchField('query-speed', `${stats.avgQueryTime.toFixed(0)}ms`);
    this.patchField('query-rate', `${stats.queriesPerSecond.toFixed(1)}/sec`);
    this.patchField(
      'network-bytes',
      cacheMetrics.network ? templateFormatBytes(cacheMetrics.network.bytesTransferred) : '0B'
    );
    this.patchField(
      'network-detail',
      cacheMetrics.network
        ? `${cacheMetrics.network.requestCount} req · ${templateFormatBytes(cacheMetrics.network.bandwidth)}/s`
        : '0 req'
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

    const badges = this.contentContainer.querySelectorAll(
      '.luxar-scene-graph__badge[data-node-path]'
    );
    badges.forEach((badge) => {
      const path = (badge as HTMLElement).dataset.nodePath;
      if (!path) return;
      const node = this.findSceneGraphNode(this.sceneGraphState.root!, path);
      if (!node) return;

      let text = '';
      if (node.type === 'points' && node.pointCount !== undefined) {
        text = templateFormatNumber(node.pointCount);
      } else if (node.type === 'lines' && node.segmentCount !== undefined) {
        text = templateFormatNumber(node.segmentCount);
      } else if (node.type === 'gsplats' && node.splatCount !== undefined) {
        text = templateFormatNumber(node.splatCount);
      } else if (node.type === 'group' && node.children.length > 0) {
        text = `${node.children.length}`;
      }
      if (text) {
        badge.textContent = text;
      }
    });
  }

  /**
   * Find a scene graph node by path (depth-first search).
   */
  private findSceneGraphNode(root: SceneGraphNode, path: string): SceneGraphNode | null {
    if (root.path === path) return root;
    for (const child of root.children) {
      const found = this.findSceneGraphNode(child, path);
      if (found) return found;
    }
    return null;
  }

  /**
   * Incrementally update cache tab values without rebuilding DOM.
   */
  private updateCacheTabValues(): boolean {
    if (!this.contentContainer) return false;

    const cacheMetrics = this.getCacheMetrics();

    // Structure validation: check for cache total (always present in L1/L2 view)
    if (!this.contentContainer.querySelector('[data-field="cache-total"]')) return false;

    // L0 stats
    if (cacheMetrics.l0) {
      const l0Total = cacheMetrics.l0.hits + cacheMetrics.l0.misses;
      const l0HitRate = l0Total > 0 ? (cacheMetrics.l0.hits / l0Total) * 100 : 0;

      this.patchField('l0-size', templateFormatBytes(cacheMetrics.l0.size));
      this.patchField('l0-size-sub', `${cacheMetrics.l0.count} chunks`);
      this.patchField('l0-hitrate', `${l0HitRate.toFixed(1)}%`);
      this.patchField(
        'l0-hitrate-sub',
        `${templateFormatNumber(cacheMetrics.l0.hits)} hits · ${templateFormatNumber(cacheMetrics.l0.misses)} miss`
      );
      this.patchField('l0-evictions', templateFormatNumber(cacheMetrics.l0.evictions));

      // Update hit rate color class
      const l0HitrateEl = this.contentContainer.querySelector('[data-field="l0-hitrate"]');
      if (l0HitrateEl) {
        const colorClass =
          l0HitRate > 80
            ? getColorClass('success')
            : l0HitRate > 50
              ? getColorClass('warning')
              : getColorClass('error');
        this.updateColorClass(l0HitrateEl as HTMLElement, colorClass);
      }

      // Update eviction color class (warning when >0, dimmed when 0)
      const l0EvictEl = this.contentContainer.querySelector('[data-field="l0-evictions"]');
      if (l0EvictEl) {
        this.updateColorClass(
          l0EvictEl as HTMLElement,
          cacheMetrics.l0.evictions > 0 ? getColorClass('warning') : getColorClass('dimmed')
        );
      }
    }

    // L1 stats
    if (cacheMetrics.l1) {
      const l1Total = cacheMetrics.l1.hits + cacheMetrics.l1.misses;
      const l1HitRate = l1Total > 0 ? (cacheMetrics.l1.hits / l1Total) * 100 : 0;

      this.patchField('l1-size', templateFormatBytes(cacheMetrics.l1.size));
      this.patchField('l1-size-sub', `${cacheMetrics.l1.count} entries`);
      this.patchField('l1-hitrate', `${l1HitRate.toFixed(1)}%`);
      this.patchField(
        'l1-hitrate-sub',
        `${templateFormatNumber(cacheMetrics.l1.hits)} hits · ${templateFormatNumber(cacheMetrics.l1.misses)} miss`
      );
      this.patchField('l1-evictions', templateFormatNumber(cacheMetrics.l1.evictions));

      // Update hit rate color class
      const l1HitrateEl = this.contentContainer.querySelector('[data-field="l1-hitrate"]');
      if (l1HitrateEl) {
        const colorClass =
          l1HitRate > 80
            ? getColorClass('success')
            : l1HitRate > 50
              ? getColorClass('warning')
              : getColorClass('error');
        this.updateColorClass(l1HitrateEl as HTMLElement, colorClass);
      }

      // Update eviction color class
      const l1EvictEl = this.contentContainer.querySelector('[data-field="l1-evictions"]');
      if (l1EvictEl) {
        this.updateColorClass(
          l1EvictEl as HTMLElement,
          cacheMetrics.l1.evictions > 0 ? getColorClass('warning') : getColorClass('dimmed')
        );
      }
    }

    // L2 stats
    if (cacheMetrics.l2) {
      this.patchField('l2-size', templateFormatBytes(cacheMetrics.l2.size));
      this.patchField('l2-size-sub', `${cacheMetrics.l2.count} entries`);
      this.patchField('l2-io', `${templateFormatNumber(cacheMetrics.l2.reads)} reads`);
      this.patchField('l2-io-sub', `${templateFormatNumber(cacheMetrics.l2.writes)} writes`);
    }

    // Total
    this.patchField('cache-total', templateFormatBytes(cacheMetrics.totalCacheMemory));

    // Update progress bar
    const barFill = this.contentContainer.querySelector(
      '.luxar-cache-total .luxar-progress-bar__fill'
    ) as HTMLElement | null;
    if (barFill) {
      barFill.style.width = `${Math.min(100, cacheMetrics.memoryPercent)}%`;
      this.updateColorClass(barFill, getCacheMemoryColorClass(cacheMetrics.memoryPercent));
    }
    const barLabel = this.contentContainer.querySelector(
      '.luxar-cache-total .luxar-progress-bar__label'
    );
    if (barLabel) {
      barLabel.textContent = `${cacheMetrics.memoryPercent.toFixed(0)}% of ${templateFormatBytes(cacheMetrics.memoryLimit)} limit`;
    }

    return true;
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
      const types = ['points', 'lines', 'gsplats'] as const;
      for (const type of types) {
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
    const accTypes = ['points', 'lines', 'gsplats'] as const;
    for (const type of accTypes) {
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
   * Update CSS color classes on an element, replacing any existing luxar-color--* class.
   * Pass empty string to clear all color classes without adding a new one.
   */
  private updateColorClass(el: HTMLElement, newColorClass: string): void {
    // Remove existing color classes
    const classes = el.className.split(' ').filter((c) => c && !c.startsWith('luxar-color--'));
    if (newColorClass) {
      classes.push(newColorClass);
    }
    el.className = classes.join(' ');
  }

  /**
   * Render tabs
   */
  private renderTabs(): string {
    const tabs = [
      { id: 'overview', label: 'Overview', icon: '📊' },
      { id: 'cache', label: 'Cache', icon: '💾' },
      { id: 'memory', label: 'Memory', icon: '🧠' },
      { id: 'performance', label: 'Performance', icon: '⚡' },
      { id: 'insights', label: 'Insights', icon: '💡' },
    ];

    return tabs
      .map(
        (tab) => `
      <button
        class="luxar-data-monitor__tab ${this.uiState.activeTab === tab.id ? 'luxar-data-monitor__tab--active' : ''}"
        data-action="setTab" data-tab-id="${tab.id}"
      >
        ${tab.icon}&nbsp;${tab.label}
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
   * Render overview tab with cleaner visual hierarchy
   */
  private renderOverviewTab(): string {
    const stats = this.getGlobalStats();
    const cacheMetrics = this.getCacheMetrics();

    // Use the template function for the main content
    const content = renderOverviewContent(stats, cacheMetrics);

    // Replace the loader list placeholder with scene graph tree (or compact loader list if no scene graph)
    if (this.sceneGraphState.root) {
      return content.replace(
        '<div id="loader-list-content"></div>',
        renderSceneGraphTree(this.sceneGraphState, this.expandedNodes)
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
    return renderCacheContent(stats, cacheMetrics);
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
      accumulators: {
        points: this.accumulatorProviders.points?.getStats() ?? null,
        lines: this.accumulatorProviders.lines?.getStats() ?? null,
        gsplats: this.accumulatorProviders.gsplats?.getStats() ?? null,
      },
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
        ${renderHierarchicalTimingPanel(timingData)}
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
    let totalPoints = 0;
    let totalMemory = 0;
    let totalQueries = 0;
    let totalLoads = 0;
    let totalQueryTime = 0;
    let activeSpatial = 0;
    let datasetSize = 0; // Total points in all datasets (from zarr metadata)
    let visiblePoints = 0; // Currently visible/rendered points

    for (const metrics of this.metrics.values()) {
      totalPoints += metrics.pointsLoaded;
      totalMemory += metrics.memoryUsed;
      totalQueries += metrics.queries;
      totalLoads += metrics.loads;
      totalQueryTime += metrics.avgQueryTime * metrics.queries;
      datasetSize += metrics.datasetSize || 0;
      visiblePoints += metrics.visiblePoints || 0;

      if (metrics.type === 'point-spatial-index') activeSpatial++;
    }

    // Use cached QPS calculation instead of filtering events again
    this.calculateRates();
    const qps = this.cachedRates.queriesPerSec;

    // Get line segment stats from scene graph
    const datasetSegments = this.sceneGraphState.totalSegments;
    // Use tracked visible segments (updated by scene loader after nD clipping)
    const visibleSegments = this.sceneGraphState.visibleSegments;

    // Get gsplat stats from scene graph
    const datasetSplats = this.sceneGraphState.totalSplats;
    const visibleSplats = this.sceneGraphState.visibleSplats;

    return {
      totalLoaders: this.loaders.size,
      activeSpatialLoaders: activeSpatial,
      activeFallbackLoaders: 0, // No more fallback loaders
      totalPoints,
      totalMemory,
      datasetSize, // Total points in all datasets (from zarr metadata)
      visiblePoints, // Currently visible/rendered points
      datasetSegments, // Total segments in all line datasets
      visibleSegments, // Currently visible segments (for lines, typically equals total)
      datasetSplats, // Total splats in all gsplats datasets
      visibleSplats, // Currently visible splats
      totalQueries,
      totalLoads,
      totalCacheHits: 0, // L0 cache removed
      totalPointsLoaded: totalPoints, // Alias for compatibility
      totalMemoryUsed: totalMemory, // Alias for compatibility
      globalCacheHitRate: 0, // L0 cache removed
      avgQueryTime: totalQueries > 0 ? totalQueryTime / totalQueries : 0,
      queriesPerSecond: qps,
      recommendations: this.advisor.getRecommendations(),
    };
  }

  /**
   * Get cache metrics aggregated across all loaders.
   * When a CacheStatsProvider is connected, uses actual L1/L2 cache stats.
   */
  private getCacheMetrics(): CacheMetrics {
    let totalCacheMemory = 0;
    let memoryLimit = 0;
    let totalEntries = 0;
    let evictions = 0;

    // Get L0/L1/L2/network breakdown from cache stats providers if available
    let l0Stats: CacheMetrics['l0'] | undefined;
    let l1Stats: CacheMetrics['l1'] | undefined;
    let l2Stats: CacheMetrics['l2'] | undefined;
    let networkStats: CacheMetrics['network'] | undefined;
    let cacheEnabled = true;

    // Get L0 stats from L0 cache provider
    if (this.l0CacheProvider) {
      l0Stats = this.l0CacheProvider.getStats();
    }

    if (this.cacheStatsProvider) {
      const stats = this.cacheStatsProvider.getStats();
      cacheEnabled = this.cacheStatsProvider.isEnabled();

      // L1 stats
      l1Stats = {
        size: stats.l1.metadataSize + stats.l1.chunksSize,
        count: stats.l1.metadataCount + stats.l1.chunksCount,
        hits: stats.l1.hits,
        misses: stats.l1.misses,
        evictions: stats.l1.evictions,
      };

      // L2 stats
      l2Stats = {
        size: stats.l2.size,
        count: stats.l2.count,
        reads: stats.l2.reads,
        writes: stats.l2.writes,
      };

      // Network stats
      networkStats = {
        bytesTransferred: stats.network.bytesTransferred,
        requestCount: stats.network.requestCount,
        bandwidth: stats.network.bandwidth,
      };

      // Update totals from cache stats (L0 + L1 + L2)
      totalCacheMemory = (l0Stats?.size ?? 0) + l1Stats.size + l2Stats.size;
      totalEntries = (l0Stats?.count ?? 0) + l1Stats.count + l2Stats.count;
    } else if (l0Stats) {
      // Only L0 available
      totalCacheMemory = l0Stats.size;
      totalEntries = l0Stats.count;
    }

    // Also aggregate from loaders for memory limit and evictions
    for (const [path, loader] of this.loaders) {
      // Get fresh metrics from the loader
      const metrics = loader.getMetrics();
      this.metrics.set(path, metrics);

      memoryLimit += metrics.memoryLimit;
      evictions += metrics.evictions;

      // If no cache stats provider, fall back to loader metrics
      if (!this.cacheStatsProvider) {
        totalCacheMemory += metrics.memoryUsed;
        if (metrics.spatialIndex && metrics.spatialIndex.rangesInCache !== undefined) {
          totalEntries += metrics.spatialIndex.rangesInCache;
        }
      }
    }

    const memoryPercent = memoryLimit > 0 ? (totalCacheMemory / memoryLimit) * 100 : 0;

    // Calculate rates once and reuse
    this.calculateRates();

    // Calculate hit rate from L1 stats if available
    const totalL1Accesses = l1Stats ? l1Stats.hits + l1Stats.misses : 0;
    const recentHitRate = totalL1Accesses > 0 ? l1Stats!.hits / totalL1Accesses : 0;

    return {
      totalCacheMemory,
      memoryLimit,
      memoryPercent,
      totalEntries,
      totalAccesses: totalL1Accesses,
      recentHitRate,
      evictionsPerMin: evictions,
      avgEntrySize: totalEntries > 0 ? totalCacheMemory / totalEntries : 0,
      reuseRatio: 0,
      hitsPerSecond: l1Stats ? l1Stats.hits / 60 : 0, // Simplified rate
      missesPerSecond: l1Stats ? l1Stats.misses / 60 : 0, // Simplified rate
      avgAccessTime: 0,
      queriesPerSec: this.cachedRates.queriesPerSec,
      loadsPerSec: this.cachedRates.loadsPerSec,
      bandwidth: this.cachedRates.bandwidth,
      // L0/L1/L2/Network breakdown
      l0: l0Stats,
      l1: l1Stats,
      l2: l2Stats,
      network: networkStats,
      enabled: cacheEnabled,
    };
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
   * Calculate all rates with caching
   */
  private calculateRates(): void {
    const now = Date.now();

    // Use cached values if recent enough
    if (now - this.cachedRates.lastCalculated < this.ratesCacheTimeout) {
      return;
    }

    // Single pass through events to calculate all rates
    let queries5s = 0;
    let loads5s = 0;
    let hits5s = 0;
    let misses5s = 0;
    let bandwidth1s = 0;

    const cutoff5s = now - MonitorLimits.rateCalculationWindow;
    const cutoff1s = now - MonitorLimits.bandwidthCalculationWindow;

    // Iterate backwards for early exit optimization
    for (let i = this.events.length - 1; i >= 0; i--) {
      const event = this.events[i];

      // Early exit if event is too old
      if (event.timestamp < cutoff5s) {
        break;
      }

      // Count events in 5s window
      switch (event.type) {
        case 'query':
          queries5s++;
          break;
        case 'load':
          loads5s++;
          // Also count bandwidth for 1s window
          if (event.timestamp > cutoff1s) {
            bandwidth1s += event.data.memory || 0;
          }
          break;
        case 'cache-hit':
          hits5s++;
          break;
        case 'cache-miss':
          misses5s++;
          break;
      }
    }

    // Update cached values (convert window to seconds)
    const windowSeconds = MonitorLimits.rateCalculationWindow / 1000;
    this.cachedRates.queriesPerSec = queries5s / windowSeconds;
    this.cachedRates.loadsPerSec = loads5s / windowSeconds;
    this.cachedRates.hitsPerSec = hits5s / windowSeconds;
    this.cachedRates.missesPerSec = misses5s / windowSeconds;
    this.cachedRates.bandwidth = bandwidth1s;
    this.cachedRates.lastCalculated = now;
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
      this.panel.style.display = 'block';
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
