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
} from './data-monitor-types';

import { PerformanceTimeline } from './components/performance-timeline';
import { LoadingAdvisor } from './components/loading-advisor';
import { log, Modules } from '../utils/log';
import {
  MonitorColors,
  MonitorTypography,
  MonitorSpacing,
  MonitorEffects,
  MonitorStyles,
} from './data-loading-monitor-styles';

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
    timeRange: 60, // Last 60 seconds
  };

  // Update tracking
  private updateTimer: number | null = null;
  private lastUpdateTime = 0;

  // Performance optimization
  private lastEventCleanup = 0;
  private eventCleanupInterval = 30000; // Clean events every 30 seconds
  private maxEventAge = 300000; // Keep events for 5 minutes

  // Cached calculations
  private cachedRates = {
    queriesPerSec: 0,
    loadsPerSec: 0,
    hitsPerSec: 0,
    missesPerSec: 0,
    bandwidth: 0,
    lastCalculated: 0,
  };
  private ratesCacheTimeout = 1000; // Recalculate rates every second max

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

    // Clean up old queries more efficiently - only check every 10 queries
    if (this.queries.size % 10 === 0) {
      const cutoff = Date.now() - 60000; // Keep last minute
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
      memoryLimit: 1024 * 1024 * 1024, // 1GB default (will be overridden by actual loader limits)
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
    }
  }

  /**
   * Create UI elements
   */
  private createUI(): void {
    // Add global styles for the monitor if not already added
    if (!document.getElementById('luxar-monitor-styles')) {
      const style = document.createElement('style');
      style.id = 'luxar-monitor-styles';
      style.textContent = `
        .luxar-data-monitor .header-btn:hover {
          color: #fff !important;
        }
        .luxar-data-monitor .expand-btn:hover {
          color: #fff !important;
        }
      `;
      document.head.appendChild(style);
    }

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
      'top-left': `top: ${MonitorSpacing.panelMargin}px; left: ${MonitorSpacing.panelMargin}px;`,
      'top-right': `top: ${MonitorSpacing.panelMargin}px; right: ${MonitorSpacing.panelMargin}px;`,
      'bottom-left': `bottom: ${MonitorSpacing.panelMargin}px; left: ${MonitorSpacing.panelMargin}px;`,
      'bottom-right': `bottom: ${MonitorSpacing.panelMargin}px; right: ${MonitorSpacing.panelMargin}px;`,
    };

    // Set fixed width for expanded mode, auto for compact mode
    const widthStyle = this.uiState.isExpanded
      ? 'width: 480px; min-width: 480px; max-width: 480px;'
      : 'width: auto;';

    return `
      position: fixed;
      ${positions[this.config.position]}
      ${MonitorStyles.panel.base}
      ${widthStyle}
      user-select: none;
      pointer-events: auto;
      box-sizing: border-box;
    `;
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
          
          <button class="expand-btn" data-action="expand" title="Show details" style="
            background: none;
            border: none;
            color: #999;
            font-size: 18px;
            cursor: pointer;
            padding: 0 4px;
            transition: color 0.2s;
          ">
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
   * Update detailed view
   */
  private updateDetailedView(): void {
    if (!this.panel) return;

    // Update metrics from all loaders first
    for (const [path, loader] of this.loaders) {
      const metrics = loader.getMetrics();
      this.metrics.set(path, metrics);
    }

    this.panel.innerHTML = `
      <div class="monitor-detailed" style="${this.getDetailedStyles()}">
        <!-- Header -->
        <div class="monitor-header" style="${this.getHeaderStyles()}">
          <h3 style="margin: 0; font-size: 14px;">Data Loading Monitor</h3>
          <div class="header-actions" style="display: flex; gap: 8px;">
            <button class="header-btn minimize-btn" data-action="minimize" title="Minimize" style="
              background: none;
              border: none;
              color: #999;
              font-size: 20px;
              cursor: pointer;
              padding: 0;
              width: 30px;
              height: 30px;
              display: flex;
              align-items: center;
              justify-content: center;
              transition: color 0.2s;
            ">—</button>
            <button class="header-btn close-btn" data-action="hide" title="Close" style="
              background: none;
              border: none;
              color: #999;
              font-size: 24px;
              cursor: pointer;
              padding: 0;
              width: 30px;
              height: 30px;
              display: flex;
              align-items: center;
              justify-content: center;
              transition: color 0.2s;
            ">×</button>
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
    if (this.uiState.activeTab === 'performance') {
      this.timeline.initializeCanvas('timeline-canvas');
    }
    
    // Add hover effects to header buttons
    this.addHeaderButtonHoverEffects();
  }
  
  /**
   * Add hover effects to header buttons
   */
  private addHeaderButtonHoverEffects(): void {
    if (!this.panel) return;
    
    const headerButtons = this.panel.querySelectorAll('.header-btn');
    headerButtons.forEach((btn) => {
      const button = btn as HTMLButtonElement;
      button.addEventListener('mouseenter', () => {
        button.style.color = '#fff';
      });
      button.addEventListener('mouseleave', () => {
        button.style.color = '#999';
      });
    });
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
        class="tab ${this.uiState.activeTab === tab.id ? 'active' : ''}"
        data-action="setTab" data-tab-id="${tab.id}"
        style="${this.getTabButtonStyles(this.uiState.activeTab === tab.id)}; white-space: nowrap;"
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
    const cacheColor = this.getCacheRateColor(stats.globalCacheHitRate);
    // Get actual memory limit from cache metrics
    const cacheMetrics = this.getCacheMetrics();
    const memoryPercent =
      cacheMetrics.memoryLimit > 0 ? (stats.totalMemory / cacheMetrics.memoryLimit) * 100 : 0;

    return `
      <div class="overview-content">
        <!-- Primary metrics with large, clear values -->
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
          
          <div style="background: ${MonitorColors.sectionBg}; padding: 15px; border-radius: 6px;">
            <div style="font-size: 32px; font-weight: bold; color: ${MonitorColors.success};">
              ${this.formatNumber(stats.totalPoints)}
            </div>
            <div style="font-size: 11px; color: ${MonitorColors.muted}; margin-top: 4px;">
              POINTS LOADED
            </div>
            <div style="font-size: 10px; color: ${MonitorColors.dimmed}; margin-top: 2px;">
              ${stats.totalLoaders} loaders (${stats.activeSpatialLoaders} spatial)
            </div>
          </div>
          
          <div style="background: ${MonitorColors.sectionBg}; padding: 15px; border-radius: 6px;">
            <div style="font-size: 32px; font-weight: bold; color: ${cacheColor};">
              ${stats.globalCacheHitRate.toFixed(0)}%
            </div>
            <div style="font-size: 11px; color: ${MonitorColors.muted}; margin-top: 4px;">
              CACHE HIT RATE
            </div>
            <div style="font-size: 10px; color: ${MonitorColors.dimmed}; margin-top: 2px;">
              ${stats.totalCacheHits}/${stats.totalCacheHits + (stats.totalQueries - stats.totalCacheHits)} hits
            </div>
          </div>
          
        </div>
        
        <!-- Secondary metrics bar -->
        <div style="display: flex; gap: 20px; padding: 10px; background: ${MonitorColors.sectionBg}; border-radius: 4px; margin-bottom: 15px;">
          <div style="flex: 1;">
            <span style="color: ${MonitorColors.muted}; font-size: 10px;">MEMORY</span>
            <div style="color: ${MonitorColors.primaryText}; font-size: 14px; font-weight: 600;">
              ${this.formatBytes(stats.totalMemory)}
            </div>
            <div style="height: 2px; background: rgba(255,255,255,0.1); margin-top: 4px;">
              <div style="height: 100%; background: ${this.getCacheMemoryColor(memoryPercent)}; width: ${Math.min(100, memoryPercent)}%;"></div>
            </div>
          </div>
          
          <div style="flex: 1;">
            <span style="color: ${MonitorColors.muted}; font-size: 10px;">QUERY SPEED</span>
            <div style="color: ${MonitorColors.primaryText}; font-size: 14px; font-weight: 600;">
              ${stats.avgQueryTime.toFixed(0)}ms
            </div>
            <div style="color: ${MonitorColors.dimmed}; font-size: 10px;">
              ${stats.queriesPerSecond.toFixed(1)}/sec
            </div>
          </div>
          
          <div style="flex: 1;">
            <span style="color: ${MonitorColors.muted}; font-size: 10px;">LOAD SPEED</span>
            <div style="color: ${MonitorColors.primaryText}; font-size: 14px; font-weight: 600;">
              ${stats.totalLoads} loads
            </div>
            <div style="color: ${MonitorColors.dimmed}; font-size: 10px;">
              ${this.formatBytes(stats.totalMemoryUsed)}/s
            </div>
          </div>
        </div>
        
        <!-- Compact loader list -->
        <div class="loader-list">
          <h4 style="margin: 0 0 8px 0; font-size: 11px; color: ${MonitorColors.muted};">ACTIVE LOADERS</h4>
          <div style="max-height: 150px; overflow-y: auto;">
            ${this.renderCompactLoaderList()}
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Render comprehensive cache analytics tab
   */
  private renderCacheTab(): string {
    const stats = this.getGlobalStats();
    const cacheMetrics = this.getCacheMetrics();

    return `
      <div class="cache-content">
        <!-- Cache overview cards -->
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 15px;">
          
          <div style="background: ${MonitorColors.sectionBg}; padding: 12px; border-radius: 4px;">
            <div style="font-size: 10px; color: ${MonitorColors.muted}; margin-bottom: 4px;">CACHE MEMORY</div>
            <div style="font-size: 24px; font-weight: bold; color: ${MonitorColors.success};">
              ${this.formatBytes(cacheMetrics.totalCacheMemory)}
            </div>
            <div style="margin-top: 6px;">
              <div style="height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px;">
                <div style="height: 100%; background: ${this.getCacheMemoryColor(cacheMetrics.memoryPercent)}; width: ${cacheMetrics.memoryPercent}%; border-radius: 2px;"></div>
              </div>
              <div style="font-size: 9px; color: ${MonitorColors.dimmed}; margin-top: 2px;">
                ${cacheMetrics.memoryPercent.toFixed(0)}% of ${this.formatBytes(cacheMetrics.memoryLimit)}
              </div>
            </div>
          </div>
          
          <div style="background: ${MonitorColors.sectionBg}; padding: 12px; border-radius: 4px;">
            <div style="font-size: 10px; color: ${MonitorColors.muted}; margin-bottom: 4px;">HIT RATE</div>
            <div style="font-size: 24px; font-weight: bold; color: ${this.getCacheRateColor(stats.globalCacheHitRate)};">
              ${stats.globalCacheHitRate.toFixed(1)}%
            </div>
            <div style="font-size: 10px; color: ${MonitorColors.dimmed}; margin-top: 4px;">
              ${cacheMetrics.recentHitRate.toFixed(0)}% recent (1m)
            </div>
            <div style="font-size: 9px; color: ${MonitorColors.muted}; margin-top: 2px;">
              ${stats.totalCacheHits} hits / ${cacheMetrics.totalAccesses} total
            </div>
          </div>
          
          <div style="background: ${MonitorColors.sectionBg}; padding: 12px; border-radius: 4px;">
            <div style="font-size: 10px; color: ${MonitorColors.muted}; margin-bottom: 4px;">CACHED RANGES</div>
            <div style="font-size: 24px; font-weight: bold; color: ${MonitorColors.primaryText};">
              ${cacheMetrics.totalEntries}
            </div>
            <div style="font-size: 10px; color: ${MonitorColors.dimmed}; margin-top: 4px;">
              ${cacheMetrics.evictionsPerMin.toFixed(0)} evict/min
            </div>
          </div>
          
          <div style="background: ${MonitorColors.sectionBg}; padding: 12px; border-radius: 4px;">
            <div style="font-size: 10px; color: ${MonitorColors.muted}; margin-bottom: 4px;">AVG RANGE SIZE</div>
            <div style="font-size: 24px; font-weight: bold; color: ${MonitorColors.primaryText};">
              ${this.formatBytes(cacheMetrics.avgEntrySize)}
            </div>
            <div style="font-size: 10px; color: ${MonitorColors.dimmed}; margin-top: 4px;">
              Reuse: ${cacheMetrics.reuseRatio.toFixed(1)}x
            </div>
          </div>
          
        </div>
        
        <!-- Cache performance metrics -->
        <div style="margin-bottom: 15px;">
          <h4 style="margin: 0 0 8px 0; font-size: 11px; color: ${MonitorColors.muted};">CACHE PERFORMANCE</h4>
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
            <div style="background: ${MonitorColors.sectionBg}; padding: 8px; border-radius: 4px; text-align: center;">
              <div style="font-size: 16px; font-weight: bold; color: ${MonitorColors.info};">
                ${cacheMetrics.hitsPerSecond.toFixed(1)}/s
              </div>
              <div style="font-size: 9px; color: ${MonitorColors.muted};">Hits/sec</div>
            </div>
            <div style="background: ${MonitorColors.sectionBg}; padding: 8px; border-radius: 4px; text-align: center;">
              <div style="font-size: 16px; font-weight: bold; color: ${MonitorColors.warning};">
                ${cacheMetrics.missesPerSecond.toFixed(1)}/s
              </div>
              <div style="font-size: 9px; color: ${MonitorColors.muted};">Misses/sec</div>
            </div>
            <div style="background: ${MonitorColors.sectionBg}; padding: 8px; border-radius: 4px; text-align: center;">
              <div style="font-size: 16px; font-weight: bold; color: ${this.getAccessTimeColor(cacheMetrics.avgAccessTime)};">
                ${
                  cacheMetrics.avgAccessTime < 1
                    ? (cacheMetrics.avgAccessTime * 1000).toFixed(0) + 'μs' // Show in microseconds if < 1ms
                    : cacheMetrics.avgAccessTime.toFixed(2) + 'ms'
                }
              </div>
              <div style="font-size: 9px; color: ${MonitorColors.muted};">Avg Access</div>
            </div>
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

    // Use cached QPS calculation instead of filtering events again
    this.calculateRates();
    const qps = this.cachedRates.queriesPerSec;

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
   * Get cache metrics aggregated across all loaders
   */
  private getCacheMetrics(): CacheMetrics {
    let totalCacheMemory = 0;
    let memoryLimit = 0;
    let totalEntries = 0;
    let totalAccesses = 0;
    let totalHits = 0;
    let evictions = 0;
    let totalAccessTimeWeighted = 0; // Sum of (avgAccessTime * accessCount) for weighted average
    let totalAccessCount = 0; // Total number of cache accesses across all loaders

    // Aggregate from all loaders
    for (const [path, loader] of this.loaders) {
      // Get fresh metrics from the loader
      const metrics = loader.getMetrics();
      this.metrics.set(path, metrics);

      totalCacheMemory += metrics.memoryUsed;
      memoryLimit += metrics.memoryLimit;
      totalHits += metrics.cacheHits;
      totalAccesses += metrics.cacheHits + metrics.cacheMisses;
      evictions += metrics.evictions;

      // Get actual cached ranges from the loader's cache stats
      if (metrics.spatialIndex && metrics.spatialIndex.rangesInCache !== undefined) {
        totalEntries += metrics.spatialIndex.rangesInCache;
      }

      // Get cache stats from the loader to calculate weighted average access time
      // Note: We need to access the loader's cache directly or through a method
      // For now, we'll check if the loader has a getCacheStats method
      if ('getCacheStats' in loader && typeof loader.getCacheStats === 'function') {
        const cacheStats = (loader as any).getCacheStats();
        if (cacheStats && typeof cacheStats.avgAccessTime === 'number') {
          const accessCount = cacheStats.hits + cacheStats.misses;
          if (accessCount > 0) {
            totalAccessTimeWeighted += cacheStats.avgAccessTime * accessCount;
            totalAccessCount += accessCount;
          }
        }
      }
    }

    const hitRate = totalAccesses > 0 ? (totalHits / totalAccesses) * 100 : 0;
    const memoryPercent = memoryLimit > 0 ? (totalCacheMemory / memoryLimit) * 100 : 0;

    // Calculate weighted average access time
    const avgAccessTime = totalAccessCount > 0 ? totalAccessTimeWeighted / totalAccessCount : 0;

    // Calculate rates once and reuse
    this.calculateRates();

    return {
      totalCacheMemory,
      memoryLimit,
      memoryPercent,
      totalEntries,
      totalAccesses,
      recentHitRate: hitRate, // Simplified for now
      evictionsPerMin: evictions, // Simplified
      avgEntrySize: totalEntries > 0 ? totalCacheMemory / totalEntries : 0,
      reuseRatio: totalHits > 0 ? totalHits / Math.max(1, totalAccesses - totalHits) : 0,
      hitsPerSecond: this.cachedRates.hitsPerSec,
      missesPerSecond: this.cachedRates.missesPerSec,
      avgAccessTime: avgAccessTime,
      queriesPerSec: this.cachedRates.queriesPerSec,
      loadsPerSec: this.cachedRates.loadsPerSec,
      bandwidth: this.cachedRates.bandwidth,
    };
  }

  private getCacheRateColor(rate: number): string {
    if (rate >= 80) return MonitorColors.success;
    if (rate >= 60) return MonitorColors.warning;
    return MonitorColors.error;
  }

  private getCacheMemoryColor(percent: number): string {
    if (percent <= 60) return MonitorColors.success;
    if (percent <= 80) return MonitorColors.warning;
    return MonitorColors.error;
  }

  private getAccessTimeColor(timeMs: number): string {
    // Color based on cache access speed
    if (timeMs < 0.1) return MonitorColors.success; // < 0.1ms is excellent (memory cache)
    if (timeMs < 1.0) return MonitorColors.info; // < 1ms is good
    if (timeMs < 5.0) return MonitorColors.warning; // < 5ms is acceptable
    return MonitorColors.error; // >= 5ms is slow
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

    const cutoff5s = now - 5000;
    const cutoff1s = now - 1000;

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

    // Update cached values
    this.cachedRates.queriesPerSec = queries5s / 5;
    this.cachedRates.loadsPerSec = loads5s / 5;
    this.cachedRates.hitsPerSec = hits5s / 5;
    this.cachedRates.missesPerSec = misses5s / 5;
    this.cachedRates.bandwidth = bandwidth1s;
    this.cachedRates.lastCalculated = now;
  }

  // Individual rate calculation methods removed - use calculateRates() and cachedRates directly

  private renderCompactLoaderList(): string {
    const loaderEntries = Array.from(this.metrics.entries());

    if (loaderEntries.length === 0) {
      return '<div style="color: rgba(255,255,255,0.4); font-size: 10px;">No active loaders</div>';
    }

    return loaderEntries
      .map(([path, metrics]) => {
        const hitRate =
          metrics.cacheHits + metrics.cacheMisses > 0
            ? (metrics.cacheHits / (metrics.cacheHits + metrics.cacheMisses)) * 100
            : 0;
        const statusColor = metrics.queries > 0 ? MonitorColors.success : MonitorColors.muted;

        return `
        <div style="background: ${MonitorColors.sectionBg}; padding: 8px; margin-bottom: 6px; border-radius: 4px;">
          <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-size: 10px; color: ${statusColor}; font-family: monospace;">${path}</span>
            <span style="font-size: 9px; color: ${MonitorColors.muted};">${metrics.type}</span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 9px; color: ${MonitorColors.dimmed};">
            <span>${metrics.pointsLoaded.toLocaleString()} pts</span>
            <span>${hitRate.toFixed(0)}% cache</span>
            <span>${this.formatBytes(metrics.memoryUsed)}</span>
          </div>
        </div>
      `;
      })
      .join('');
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
        this.panel.style.cssText = this.getPanelStyles();
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
      this.panel.style.cssText = this.getPanelStyles();
    }
    this.updateUI();
  }

  public minimize(): void {
    this.uiState.isExpanded = false;
    // Update panel styles to remove fixed width
    if (this.panel) {
      this.panel.style.cssText = this.getPanelStyles();
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
    // Validate tab is a valid tab type
    const validTabs = ['overview', 'cache', 'spatial', 'performance', 'insights'] as const;
    if (validTabs.includes(tab as any)) {
      this.uiState.activeTab = tab as 'overview' | 'cache' | 'spatial' | 'performance' | 'insights';
      this.updateUI();
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
      ${MonitorStyles.panel.base}
      ${MonitorStyles.panel.compact}
      border-radius: ${MonitorEffects.borderRadiusSmall}px;
    `;
  }

  private getMetricsBarStyles(): string {
    return `
      display: flex;
      align-items: center;
      gap: ${MonitorSpacing.elementGap}px;
      color: ${MonitorColors.primaryText};
      font-size: ${MonitorTypography.body.fontSize};
      font-weight: 500;
    `;
  }

  private getDetailedStyles(): string {
    return `
      ${MonitorStyles.panel.base}
      ${MonitorStyles.panel.expanded}
      max-height: 600px;
      box-shadow: ${MonitorEffects.boxShadowStrong};
    `;
  }

  private getHeaderStyles(): string {
    return MonitorStyles.header;
  }

  private getTabStyles(): string {
    return `
      display: flex;
      border-bottom: 1px solid ${MonitorColors.separator};
      background: ${MonitorColors.sectionBg};
    `;
  }

  private getTabButtonStyles(active: boolean): string {
    return `
      ${MonitorStyles.button.tab.base}
      ${active ? MonitorStyles.button.tab.active : MonitorStyles.button.tab.inactive}
    `;
  }

  private getContentStyles(): string {
    return `
      padding: ${MonitorSpacing.panelPadding}px;
      max-height: 500px;
      overflow-y: auto;
      color: ${MonitorColors.primaryText};
    `;
  }

  // Removed unused getLoaderItemStyles - styles inlined where needed

  private getEventItemStyles(): string {
    return `
      display: flex;
      gap: ${MonitorSpacing.elementGap}px;
      padding: ${MonitorSpacing.tinyGap}px 0;
      font-size: ${MonitorTypography.small.fontSize};
      opacity: 0.7;
    `;
  }

  private getRecommendationStyles(severity: string): string {
    const styleMap = {
      error: MonitorStyles.alert.error,
      warning: MonitorStyles.alert.warning,
      info: MonitorStyles.alert.success,
    };

    return styleMap[severity as keyof typeof styleMap] || MonitorStyles.alert.success;
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
      console.warn('[DataLoadingMonitor] Disposal completed with errors:');
      errors.forEach((error, index) => {
        console.warn(`  ${index + 1}. ${error.message}`);
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
