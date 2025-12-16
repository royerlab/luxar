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
  MonitorEventType,
  LoaderType,
  CacheMetrics,
  CacheStatsProvider,
  SceneGraphNode,
  SceneGraphState,
} from './data-monitor-types';

import { PerformanceTimeline } from './components/performance-timeline';
import { LoadingAdvisor } from './components/loading-advisor';
import { log, Modules } from '../utils/log';
import { config } from '../config';

// Extract commonly used config values
const MonitorColors = config.ui.styles.colors;
const MonitorTypography = config.ui.styles.typography;
const MonitorSpacing = config.ui.styles.spacing;
const MonitorTimings = config.dataLoading.monitor.timings;
const MonitorLimits = config.dataLoading.monitor.limits;

// MonitorStyles removed - all styling now in CSS files

// Helper functions
const VALID_TABS = ['overview', 'cache', 'performance', 'insights'] as const;
type ValidTab = (typeof VALID_TABS)[number];
function isValidTab(tab: string): tab is ValidTab {
  return VALID_TABS.includes(tab as ValidTab);
}

import {
  renderLoaderItem,
  renderOverviewContent,
  renderCacheContent,
  renderInsightsContent,
  renderSceneGraphTree,
} from './data-monitor-templates';

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
  private timeline: PerformanceTimeline;
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

  // Update tracking
  private updateTimer: number | null = null;
  private lastUpdateTime = 0;

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

  // Event listener for external updates
  private eventListener = this.handleLoaderEvent.bind(this);

  // UI event handler bound to this instance
  private uiEventHandler = this.handleUIEvent.bind(this);

  // Cache stats provider for L1/L2 cache metrics
  private cacheStatsProvider: CacheStatsProvider | null = null;

  // DOM element references for efficient updates (avoids full innerHTML replacement)
  private contentContainer: HTMLElement | null = null;

  // Scene graph state
  private sceneGraphState: SceneGraphState = {
    root: null,
    totalNodes: 0,
    pointsNodes: 0,
    linesNodes: 0,
    totalPoints: 0,
    totalSegments: 0,
  };

  // Track expanded nodes in scene graph tree (by path)
  private expandedNodes = new Set<string>(['/']);

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
    this.timeline = new PerformanceTimeline();
    this.advisor = new LoadingAdvisor();

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

    // Schedule an update if visible to show the new loader
    if (this.uiState.isVisible) {
      this.scheduleUpdate();
    }
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
    this.timeline.clear();
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
   * Clear all caches (L1 + L2).
   */
  public async clearAllCaches(): Promise<void> {
    if (this.cacheStatsProvider) {
      await this.cacheStatsProvider.clearAll();
      log.info(Modules.DATA_MONITOR, 'All caches cleared');
      this.updateUI();
    }
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
    let totalPoints = node.pointCount || 0;
    let totalSegments = node.segmentCount || 0;

    for (const child of node.children) {
      const childStats = this.calculateSceneGraphStats(child);
      totalNodes += childStats.totalNodes;
      pointsNodes += childStats.pointsNodes;
      linesNodes += childStats.linesNodes;
      totalPoints += childStats.totalPoints;
      totalSegments += childStats.totalSegments;
    }

    return { totalNodes, pointsNodes, linesNodes, totalPoints, totalSegments };
  }

  /**
   * Handle events from loaders
   */
  private handleLoaderEvent(event: MonitorEvent): void {
    // Store event
    this.events.push(event);
    if (this.events.length > this.config.maxEvents) {
      this.events.shift();
    }

    // Clean old events periodically by time
    const now = Date.now();
    if (now - this.lastEventCleanup > this.eventCleanupInterval) {
      this.cleanOldEvents();
      this.lastEventCleanup = now;
    }

    // Update metrics
    this.updateMetricsFromEvent(event);

    // Update query tracking
    if (event.type === 'query') {
      this.trackQuery(event);
    }

    // Update components (timeline will batch its own rendering)
    this.timeline.addEvent(event);

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

    // Schedule UI update
    this.scheduleUpdate();
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
   * Schedule UI update
   */
  private scheduleUpdate(): void {
    // Skip if monitor is not visible
    if (!this.uiState.isVisible) return;

    if (this.updateTimer !== null) return;

    const now = Date.now();
    const timeSinceLastUpdate = now - this.lastUpdateTime;

    if (timeSinceLastUpdate >= this.config.updateInterval) {
      this.updateUI();
    } else {
      this.updateTimer = window.setTimeout(() => {
        this.updateTimer = null;
        this.updateUI();
      }, this.config.updateInterval - timeSinceLastUpdate);
    }
  }

  /**
   * Force an immediate UI update
   * Used when scene changes or loaders are connected
   */
  public forceUpdate(): void {
    if (this.uiState.isVisible) {
      // Cancel any pending update
      if (this.updateTimer !== null) {
        clearTimeout(this.updateTimer);
        this.updateTimer = null;
      }
      this.updateUI();
    }
  }

  /**
   * Update UI
   */
  private updateUI(): void {
    this.lastUpdateTime = Date.now();

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
   * Update compact view
   */
  private updateCompactView(): void {
    if (!this.panel) return;

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
      <div class="luxar-monitor-compact">
        <!-- Metrics bar -->
        <div class="luxar-secondary-metrics__item">
          <span class="loader-type" title="Loading mode">
            ${hasSpatialIndex ? '🔍' : '📦'}
          </span>

          <span class="points" title="Visible points">
            ${this.formatNumber(stats.visiblePoints)}
          </span>

          <span class="memory" title="Memory usage">
            ${this.formatBytes(stats.totalMemory)}
          </span>

          <span class="qps" title="Queries per second">
            ${stats.queriesPerSecond.toFixed(1)}/s
          </span>

          ${hasErrors ? '<span class="alert" title="Errors detected">🔴</span>' : ''}
          ${hasWarnings ? '<span class="alert" title="Warnings">🟡</span>' : ''}

          <button class="expand-btn" data-action="expand" title="Show details">
            ⊞
          </button>
        </div>

        ${
  hasSpatialIndex && this.config.showSpatialGrid
    ? `
        `
    : ''
}
      </div>
    `;
  }

  /**
   * Build the detailed view structure once (called on expand or tab change)
   */
  private buildDetailedViewStructure(): void {
    if (!this.panel) return;

    this.panel.innerHTML = `
      <div class="luxar-monitor-detailed">
        <!-- Header -->
        <div class="luxar-data-monitor__header">
          <h3 style="margin: 0; font-size: 14px;">Data Loading Monitor</h3>
          <div class="header-actions" style="display: flex; gap: 8px;">
            <button class="header-btn minimize-btn" data-action="minimize" title="Minimize">—</button>
            <button class="header-btn close-btn" data-action="hide" title="Close">×</button>
          </div>
        </div>

        <!-- Tabs -->
        <div class="luxar-data-monitor__tabs">
          ${this.renderTabs()}
        </div>

        <!-- Content (updated frequently) -->
        <div class="luxar-data-monitor__content">
          ${this.renderTabContent()}
        </div>
      </div>
    `;

    // Cache reference to content container for efficient updates
    this.contentContainer = this.panel.querySelector('.luxar-data-monitor__content');

    // Reinitialize component canvases if needed
    if (this.uiState.activeTab === 'performance') {
      this.timeline.initializeCanvas('timeline-canvas');
    }

    // Add hover effects to header buttons (only once)
    this.addHeaderButtonHoverEffects();
  }

  /**
   * Update detailed view (optimized to only update content, not structure)
   */
  private updateDetailedView(): void {
    if (!this.panel) return;

    // Update metrics from all loaders first
    for (const [path, loader] of this.loaders) {
      const metrics = loader.getMetrics();
      this.metrics.set(path, metrics);
    }

    // If structure doesn't exist yet, build it
    if (!this.contentContainer) {
      this.buildDetailedViewStructure();
      return;
    }

    // Only update the content area (much faster than rebuilding everything)
    this.contentContainer.innerHTML = this.renderTabContent();

    // Reinitialize canvas if on performance tab
    if (this.uiState.activeTab === 'performance') {
      this.timeline.initializeCanvas('timeline-canvas');
    }
  }

  /**
   * Add hover effects to header buttons
   */
  private addHeaderButtonHoverEffects(): void {
    if (!this.panel) return;

    // Button hover effects now handled by CSS :hover pseudo-class
  }

  /**
   * Render tabs
   */
  private renderTabs(): string {
    const tabs = [
      { id: 'overview', label: 'Overview', icon: '📊' },
      { id: 'cache', label: 'Cache', icon: '💾' },
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
   * Render performance tab
   */
  private renderPerformanceTab(): string {
    return `
      <div class="performance-content" style="width: 100%; overflow: hidden;">
        <canvas id="timeline-canvas" style="width: 100%; height: 200px;"></canvas>
        
        <div class="timeline-controls" style="margin-top: 10px; font-size: 11px; color: ${MonitorColors.muted};">
          Performance timeline (1 minute window)
        </div>
        
        <!-- Recent events -->
        <div class="event-log" style="margin-top: 15px; max-height: 150px; overflow-y: auto;">
          <h4 style="margin: 0 0 10px 0; font-size: 12px; opacity: 0.7;">Recent Events</h4>
          ${this.renderRecentEvents()}
        </div>
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
   * Render loader list
   */
  // Removed unused renderLoaderList method - functionality moved to renderCompactLoaderList

  /**
   * Render recent events
   */
  private renderRecentEvents(): string {
    const recentEvents = this.events.slice(-10).reverse();

    return recentEvents
      .map(
        (event) => `
      <div class="event-item" style="${this.getEventItemStyles()}">
        <span class="event-time">${new Date(event.timestamp).toLocaleTimeString()}</span>
        <span class="event-type">${this.getEventIcon(event.type)}</span>
        <span class="event-desc">${this.getEventDescription(event)}</span>
      </div>
    `
      )
      .join('');
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
    // For lines, visible typically equals total (no spatial filtering like points)
    const visibleSegments = this.sceneGraphState.totalSegments;

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

    // Get L1/L2/network breakdown from cache stats provider if available
    let l1Stats: CacheMetrics['l1'] | undefined;
    let l2Stats: CacheMetrics['l2'] | undefined;
    let networkStats: CacheMetrics['network'] | undefined;
    let cacheEnabled = true;

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

      // Update totals from cache stats
      totalCacheMemory = l1Stats.size + l2Stats.size;
      totalEntries = l1Stats.count + l2Stats.count;
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
      // L1/L2/Network breakdown
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

  // Individual rate calculation methods removed - use calculateRates() and cachedRates directly

  private renderCompactLoaderList(): string {
    const loaderEntries = Array.from(this.metrics.entries());

    if (loaderEntries.length === 0) {
      return '<div style="color: rgba(255,255,255,0.4); font-size: 10px;">No active loaders</div>';
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
    this.timeline.setTimeRange(this.uiState.timeRange);
    this.updateUI();
  }

  private startUpdating(): void {
    this.updateUI();
  }

  private stopUpdating(): void {
    if (this.updateTimer !== null) {
      clearTimeout(this.updateTimer);
      this.updateTimer = null;
    }
  }

  // Utility methods

  private formatNumber(n: number): string {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n.toString();
  }

  private formatBytes(bytes: number): string {
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + 'GB';
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + 'MB';
    if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + 'KB';
    return bytes.toFixed(0) + 'B';
  }

  private truncatePath(path: string): string {
    if (path.length <= 30) return path;
    const parts = path.split('/');
    if (parts.length > 2) {
      return `.../${parts.slice(-2).join('/')}`;
    }
    return '...' + path.slice(-27);
  }

  private getEventIcon(type: MonitorEventType): string {
    const icons: Record<MonitorEventType, string> = {
      query: '🔍',
      load: '📥',
      'cache-hit': '✅',
      'cache-miss': '❌',
      evict: '🗑️',
      error: '⚠️',
      prefetch: '🔮',
    };
    return icons[type] || '•';
  }

  private getEventDescription(event: MonitorEvent): string {
    const path = event.data.path ? this.truncatePath(event.data.path) : 'unknown';

    switch (event.type) {
      case 'query':
        return `Query ${path}: ${event.data.points || 0} points`;
      case 'load':
        return `Loaded ${path}: ${this.formatBytes(event.data.memory || 0)}`;
      case 'cache-hit':
        return `Cache hit: ${path}`;
      case 'cache-miss':
        return `Cache miss: ${path}`;
      case 'evict':
        return `Evicted: ${path}`;
      case 'error':
        return `Error: ${event.data.error || 'Unknown'}`;
      default:
        return event.type;
    }
  }

  // Styles

  // All style getter methods removed - styling now in CSS files

  private getEventItemStyles(): string {
    return `
      display: flex;
      gap: ${MonitorSpacing.elementGap}px;
      padding: ${MonitorSpacing.tinyGap}px 0;
      font-size: ${MonitorTypography.small.fontSize};
      opacity: 0.7;
    `;
  }

  public dispose(): void {
    const errors: Error[] = [];

    // Step 1: Stop update timer (safe operation)
    try {
      this.stopUpdating();
    } catch (error) {
      errors.push(new Error(`Failed to stop update timer: ${error}`));
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

    // Step 4: Clean up components (important for animation frame cleanup)
    try {
      if (this.timeline && typeof this.timeline.dispose === 'function') {
        this.timeline.dispose();
      }
    } catch (error) {
      errors.push(new Error(`Failed to dispose timeline: ${error}`));
    }

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
