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
} from './data-monitor-types';

import { SpatialQueryVisualizer } from './components/spatial-query-visualizer';
import { PerformanceTimeline } from './components/performance-timeline';
import { LoadingAdvisor } from './components/loading-advisor';
import { log, Modules } from '../utils/log';

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
  private spatialViz: SpatialQueryVisualizer;
  private timeline: PerformanceTimeline;
  private advisor: LoadingAdvisor;

  // Configuration
  private config: MonitorConfig;

  // UI State
  private uiState: MonitorUIState = {
    isVisible: false,
    isExpanded: false,
    activeTab: 'overview',
    timeRange: 60, // Last 60 seconds
  };

  // Update tracking
  private updateTimer: number | null = null;
  private lastUpdateTime = 0;

  // Event listener for external updates
  private eventListener = this.handleLoaderEvent.bind(this);

  // UI event handler bound to this instance
  private uiEventHandler = this.handleUIEvent.bind(this);

  constructor(container: HTMLElement, config?: Partial<MonitorConfig>) {
    this.container = container;
    this.config = {
      position: 'top-right',
      theme: 'dark',
      defaultView: 'compact',
      updateInterval: 100, // 10Hz updates
      maxEvents: 1000,
      showSpatialGrid: true,
      showTimeline: true,
      showRecommendations: true,
      autoExpand: false,
      enableProfiling: true,
      sampleRate: 1,
      ...config,
    };

    // Initialize components
    this.spatialViz = new SpatialQueryVisualizer();
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

    // Update UI if visible
    if (this.uiState.isVisible) {
      this.updateUI();
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
   * Handle events from loaders
   */
  private handleLoaderEvent(event: MonitorEvent): void {
    // Store event
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

    // Update components
    this.spatialViz.handleEvent(event);
    this.timeline.addEvent(event);

    // Check for issues
    if (this.config.showRecommendations) {
      this.advisor.analyzeEvent(event);

      // Auto-expand on warnings if configured
      if (this.config.autoExpand && this.advisor.hasWarnings()) {
        this.expand();
      }
    }

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

      case 'cache-hit':
        metrics.cacheHits++;
        break;

      case 'cache-miss':
        metrics.cacheMisses++;
        break;

      case 'evict':
        metrics.evictions++;
        break;

      case 'error':
        metrics.errors++;
        break;
    }

    // Update cache hit rate
    const totalCacheAccess = metrics.cacheHits + metrics.cacheMisses;
    metrics.cacheHitRate = totalCacheAccess > 0 ? (metrics.cacheHits / totalCacheAccess) * 100 : 0;

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

    // Clean up old queries
    const cutoff = Date.now() - 60000; // Keep last minute
    for (const [id, query] of this.queries) {
      if (query.startTime < cutoff) {
        this.queries.delete(id);
      }
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
      cacheHits: 0,
      cacheMisses: 0,
      evictions: 0,
      errors: 0,
      pointsLoaded: 0,
      bytesLoaded: 0,
      avgQueryTime: 0,
      avgLoadTime: 0,
      cacheHitRate: 0,
      memoryUsed: 0,
      memoryLimit: 500 * 1024 * 1024, // 500MB default
    };
  }

  /**
   * Schedule UI update
   */
  private scheduleUpdate(): void {
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
    }
  }

  /**
   * Create UI elements
   */
  private createUI(): void {
    // Create main panel
    this.panel = document.createElement('div');
    this.panel.className = 'luxar-data-monitor';
    this.panel.style.cssText = this.getPanelStyles();

    // Add event delegation listeners
    this.panel.addEventListener('click', this.uiEventHandler);
    this.panel.addEventListener('change', this.uiEventHandler);

    // Add to container
    this.container.appendChild(this.panel);

    // Initially hidden
    this.hide();
  }

  /**
   * Get panel styles based on config
   */
  private getPanelStyles(): string {
    const positions: Record<string, string> = {
      'top-left': 'top: 20px; left: 20px;',
      'top-right': 'top: 20px; right: 20px;',
      'bottom-left': 'bottom: 20px; left: 20px;',
      'bottom-right': 'bottom: 20px; right: 20px;',
    };

    return `
      position: fixed;
      ${positions[this.config.position]}
      z-index: 10000;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 12px;
      user-select: none;
      pointer-events: auto;
    `;
  }

  /**
   * Update compact view
   */
  private updateCompactView(): void {
    if (!this.panel) return;

    const stats = this.getGlobalStats();
    const hasSpatialIndex = stats.activeSpatialLoaders > 0;
    const recommendations = this.advisor.getRecommendations();
    const hasWarnings = recommendations.some((r) => r.severity === 'warning');
    const hasErrors = recommendations.some((r) => r.severity === 'error');

    this.panel.innerHTML = `
      <div class="monitor-compact" style="${this.getCompactStyles()}">
        <!-- Metrics bar -->
        <div class="metrics-bar" style="${this.getMetricsBarStyles()}">
          <span class="loader-type" title="Loading mode">
            ${hasSpatialIndex ? '🔍' : '📦'}
          </span>
          
          <span class="points" title="Total points loaded">
            ${this.formatNumber(stats.totalPoints)}
          </span>
          
          <span class="cache" title="Cache hit rate">
            ${stats.globalCacheHitRate.toFixed(0)}%
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
          <div class="mini-spatial" style="${this.getMiniSpatialStyles()}">
            ${this.spatialViz.renderMiniGrid()}
          </div>
        `
            : ''
        }
      </div>
    `;
  }

  /**
   * Update detailed view
   */
  private updateDetailedView(): void {
    if (!this.panel) return;

    this.panel.innerHTML = `
      <div class="monitor-detailed" style="${this.getDetailedStyles()}">
        <!-- Header -->
        <div class="monitor-header" style="${this.getHeaderStyles()}">
          <h3 style="margin: 0; font-size: 14px;">Data Loading Monitor</h3>
          <div class="header-actions">
            <button data-action="minimize" title="Minimize">_</button>
            <button data-action="hide" title="Close">×</button>
          </div>
        </div>
        
        <!-- Tabs -->
        <div class="monitor-tabs" style="${this.getTabStyles()}">
          ${this.renderTabs()}
        </div>
        
        <!-- Content -->
        <div class="monitor-content" style="${this.getContentStyles()}">
          ${this.renderTabContent()}
        </div>
      </div>
    `;

    // Reinitialize component canvases if needed
    if (this.uiState.activeTab === 'spatial') {
      this.spatialViz.initializeCanvas('spatial-grid-canvas');
    } else if (this.uiState.activeTab === 'performance') {
      this.timeline.initializeCanvas('timeline-canvas');
    }
  }

  /**
   * Render tabs
   */
  private renderTabs(): string {
    const tabs = [
      { id: 'overview', label: 'Overview', icon: '📊' },
      { id: 'spatial', label: 'Spatial', icon: '🔍' },
      { id: 'performance', label: 'Performance', icon: '⚡' },
      { id: 'insights', label: 'Insights', icon: '💡' },
    ];

    return tabs
      .map(
        (tab) => `
      <button 
        class="tab ${this.uiState.activeTab === tab.id ? 'active' : ''}"
        data-action="setTab" data-tab-id="${tab.id}"
        style="${this.getTabButtonStyles(this.uiState.activeTab === tab.id)}"
      >
        ${tab.icon} ${tab.label}
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
      case 'spatial':
        return this.renderSpatialTab();
      case 'performance':
        return this.renderPerformanceTab();
      case 'insights':
        return this.renderInsightsTab();
      default:
        return '';
    }
  }

  /**
   * Render overview tab
   */
  private renderOverviewTab(): string {
    const stats = this.getGlobalStats();

    return `
      <div class="overview-content">
        <div class="stats-grid" style="${this.getStatsGridStyles()}">
          <div class="stat-card">
            <label>Active Loaders</label>
            <value>${stats.totalLoaders}</value>
            <detail>${stats.activeSpatialLoaders} spatial index loaders</detail>
          </div>
          
          <div class="stat-card">
            <label>Total Points</label>
            <value>${this.formatNumber(stats.totalPoints)}</value>
            <detail>${this.formatBytes(stats.totalMemory)} in memory</detail>
          </div>
          
          <div class="stat-card">
            <label>Cache Performance</label>
            <value>${stats.globalCacheHitRate.toFixed(1)}%</value>
            <detail>Hit rate</detail>
          </div>
          
          <div class="stat-card">
            <label>Query Performance</label>
            <value>${stats.avgQueryTime.toFixed(0)}ms</value>
            <detail>${stats.queriesPerSecond.toFixed(1)} queries/sec</detail>
          </div>
        </div>
        
        <!-- Loader breakdown -->
        <div class="loader-list" style="margin-top: 15px;">
          <h4 style="margin: 0 0 10px 0; font-size: 12px; opacity: 0.7;">Active Loaders</h4>
          ${this.renderLoaderList()}
        </div>
      </div>
    `;
  }

  /**
   * Render spatial tab
   */
  private renderSpatialTab(): string {
    const spatialMetrics = this.getSpatialMetrics();

    if (!spatialMetrics) {
      return `
        <div class="no-spatial" style="text-align: center; padding: 20px; opacity: 0.5;">
          No spatial index active
        </div>
      `;
    }

    return `
      <div class="spatial-content">
        <!-- Grid visualization -->
        <div class="spatial-grid">
          <canvas id="spatial-grid-canvas" width="400" height="300"></canvas>
        </div>
        
        <!-- Spatial metrics -->
        <div class="spatial-stats" style="${this.getStatsGridStyles()}">
          <div class="stat-card">
            <label>Grid Shape</label>
            <value>${spatialMetrics.gridShape.join('×')}</value>
          </div>
          
          <div class="stat-card">
            <label>Occupancy</label>
            <value>${spatialMetrics.occupiedCells}/${spatialMetrics.totalCells}</value>
            <detail>${((spatialMetrics.occupiedCells / spatialMetrics.totalCells) * 100).toFixed(1)}%</detail>
          </div>
          
          <div class="stat-card">
            <label>Avg Cells/Query</label>
            <value>${spatialMetrics.avgCellsPerQuery.toFixed(1)}</value>
          </div>
          
          <div class="stat-card">
            <label>Query Efficiency</label>
            <value>${spatialMetrics.queryEfficiency.toFixed(1)}%</value>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Render performance tab
   */
  private renderPerformanceTab(): string {
    return `
      <div class="performance-content">
        <canvas id="timeline-canvas" width="400" height="200"></canvas>
        
        <div class="timeline-controls" style="margin-top: 10px;">
          <label>Time Range:</label>
          <select data-action="setTimeRange">
            <option value="10" ${this.uiState.timeRange === 10 ? 'selected' : ''}>10s</option>
            <option value="30" ${this.uiState.timeRange === 30 ? 'selected' : ''}>30s</option>
            <option value="60" ${this.uiState.timeRange === 60 ? 'selected' : ''}>1m</option>
            <option value="300" ${this.uiState.timeRange === 300 ? 'selected' : ''}>5m</option>
          </select>
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

    if (recommendations.length === 0) {
      return `
        <div class="no-insights" style="text-align: center; padding: 20px; opacity: 0.5;">
          ✅ No issues detected
        </div>
      `;
    }

    return `
      <div class="insights-content">
        ${recommendations
          .map(
            (rec) => `
          <div class="recommendation" style="${this.getRecommendationStyles(rec.severity)}">
            <div class="rec-header">
              ${this.getSeverityIcon(rec.severity)}
              <strong>${rec.title}</strong>
            </div>
            <div class="rec-message">${rec.message}</div>
            ${
              rec.suggestion
                ? `
              <div class="rec-suggestion">💡 ${rec.suggestion}</div>
            `
                : ''
            }
          </div>
        `
          )
          .join('')}
      </div>
    `;
  }

  /**
   * Render loader list
   */
  private renderLoaderList(): string {
    const entries = Array.from(this.metrics.entries());

    if (entries.length === 0) {
      return '<div style="opacity: 0.5;">No active loaders</div>';
    }

    return entries
      .map(
        ([path, metrics]) => `
      <div class="loader-item" style="${this.getLoaderItemStyles()}">
        <span class="loader-icon">${metrics.type === 'spatial-index' ? '🔍' : '📦'}</span>
        <span class="loader-path" style="flex: 1;">${this.truncatePath(path)}</span>
        <span class="loader-stats">
          ${this.formatNumber(metrics.pointsLoaded)} pts,
          ${metrics.cacheHitRate.toFixed(0)}% cache
        </span>
      </div>
    `
      )
      .join('');
  }

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
    let totalCacheHits = 0;
    let totalCacheAccess = 0;
    let totalQueryTime = 0;
    let activeSpatial = 0;

    for (const metrics of this.metrics.values()) {
      totalPoints += metrics.pointsLoaded;
      totalMemory += metrics.memoryUsed;
      totalQueries += metrics.queries;
      totalLoads += metrics.loads;
      totalCacheHits += metrics.cacheHits;
      totalCacheAccess += metrics.cacheHits + metrics.cacheMisses;
      totalQueryTime += metrics.avgQueryTime * metrics.queries;

      if (metrics.type === 'spatial-index') activeSpatial++;
    }

    // Calculate QPS from recent events
    const recentQueries = this.events.filter(
      (e) => e.type === 'query' && e.timestamp > Date.now() - 5000
    );
    const qps = recentQueries.length / 5;

    return {
      totalLoaders: this.loaders.size,
      activeSpatialLoaders: activeSpatial,
      activeFallbackLoaders: 0, // No more fallback loaders
      totalPoints,
      totalMemory,
      totalQueries,
      totalLoads,
      totalCacheHits,
      totalPointsLoaded: totalPoints, // Alias for compatibility
      totalMemoryUsed: totalMemory, // Alias for compatibility
      globalCacheHitRate: totalCacheAccess > 0 ? (totalCacheHits / totalCacheAccess) * 100 : 0,
      avgQueryTime: totalQueries > 0 ? totalQueryTime / totalQueries : 0,
      queriesPerSecond: qps,
      recommendations: this.advisor.getRecommendations(),
    };
  }

  /**
   * Get spatial metrics
   */
  private getSpatialMetrics(): any {
    for (const metrics of this.metrics.values()) {
      if (metrics.spatialIndex) {
        return metrics.spatialIndex;
      }
    }
    return null;
  }

  // Public API

  public show(): void {
    if (this.panel) {
      this.panel.style.display = 'block';
      this.uiState.isVisible = true;
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

  public expand(): void {
    this.uiState.isExpanded = true;
    this.updateUI();
  }

  public minimize(): void {
    this.uiState.isExpanded = false;
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
    this.uiState.activeTab = tab as any;
    this.updateUI();
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
    return bytes + 'B';
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

  private getSeverityIcon(severity: string): string {
    switch (severity) {
      case 'error':
        return '🔴';
      case 'warning':
        return '🟡';
      case 'info':
        return 'ℹ️';
      default:
        return '•';
    }
  }

  // Styles

  private getCompactStyles(): string {
    return `
      background: rgba(30, 30, 30, 0.95);
      border-radius: 6px;
      padding: 8px 12px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      backdrop-filter: blur(10px);
    `;
  }

  private getMetricsBarStyles(): string {
    return `
      display: flex;
      align-items: center;
      gap: 12px;
      color: #e0e0e0;
      font-size: 11px;
      font-weight: 500;
    `;
  }

  private getMiniSpatialStyles(): string {
    return `
      margin-top: 8px;
      height: 40px;
      background: rgba(0, 0, 0, 0.3);
      border-radius: 4px;
      overflow: hidden;
    `;
  }

  private getDetailedStyles(): string {
    return `
      background: rgba(30, 30, 30, 0.98);
      border-radius: 8px;
      width: 450px;
      max-height: 600px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
      backdrop-filter: blur(10px);
      overflow: hidden;
    `;
  }

  private getHeaderStyles(): string {
    return `
      padding: 12px 15px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: #e0e0e0;
    `;
  }

  private getTabStyles(): string {
    return `
      display: flex;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      background: rgba(0, 0, 0, 0.2);
    `;
  }

  private getTabButtonStyles(active: boolean): string {
    return `
      flex: 1;
      padding: 8px;
      background: ${active ? 'rgba(255, 255, 255, 0.1)' : 'transparent'};
      border: none;
      color: ${active ? '#fff' : '#999'};
      cursor: pointer;
      font-size: 11px;
      transition: all 0.2s;
    `;
  }

  private getContentStyles(): string {
    return `
      padding: 15px;
      max-height: 500px;
      overflow-y: auto;
      color: #e0e0e0;
    `;
  }

  private getStatsGridStyles(): string {
    return `
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      
      .stat-card {
        background: rgba(0, 0, 0, 0.3);
        padding: 10px;
        border-radius: 4px;
      }
      
      .stat-card label {
        display: block;
        font-size: 10px;
        opacity: 0.7;
        margin-bottom: 4px;
      }
      
      .stat-card value {
        display: block;
        font-size: 18px;
        font-weight: bold;
        color: #4CAF50;
      }
      
      .stat-card detail {
        display: block;
        font-size: 10px;
        opacity: 0.5;
        margin-top: 2px;
      }
    `;
  }

  private getLoaderItemStyles(): string {
    return `
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 4px 0;
      font-size: 11px;
      opacity: 0.8;
    `;
  }

  private getEventItemStyles(): string {
    return `
      display: flex;
      gap: 8px;
      padding: 2px 0;
      font-size: 10px;
      opacity: 0.7;
    `;
  }

  private getRecommendationStyles(severity: string): string {
    const colors = {
      error: 'rgba(244, 67, 54, 0.2)',
      warning: 'rgba(255, 193, 7, 0.2)',
      info: 'rgba(33, 150, 243, 0.2)',
    };

    return `
      background: ${colors[severity as keyof typeof colors] || colors.info};
      border-radius: 4px;
      padding: 10px;
      margin-bottom: 8px;
      font-size: 11px;
      
      .rec-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
      }
      
      .rec-message {
        opacity: 0.8;
        margin-bottom: 4px;
      }
      
      .rec-suggestion {
        opacity: 0.6;
        font-style: italic;
      }
    `;
  }

  /**
   * Reset monitor state for new scene
   */
  public reset(): void {
    // Clear events and metrics
    this.events = [];
    this.metrics.clear();
    this.queries.clear();

    // Reset UI components (if they have reset methods)
    if ('reset' in this.spatialViz) {
      (this.spatialViz as any).reset();
    }
    if ('reset' in this.timeline) {
      (this.timeline as any).reset();
    }
    if ('reset' in this.advisor) {
      (this.advisor as any).reset();
    }

    // Update UI if visible
    if (this.uiState.isVisible) {
      this.updateUI();
    }
  }

  public dispose(): void {
    this.stopUpdating();

    // Disconnect all loaders
    for (const [, loader] of this.loaders) {
      loader.removeEventListener(this.eventListener);
    }
    this.loaders.clear();

    // Remove UI
    if (this.panel) {
      // Remove event listeners
      this.panel.removeEventListener('click', this.uiEventHandler);
      this.panel.removeEventListener('change', this.uiEventHandler);

      this.panel.remove();
      this.panel = null;
    }

    // Clean up components
    this.spatialViz.dispose();
    this.timeline.dispose();
    this.advisor.dispose();
  }
}
