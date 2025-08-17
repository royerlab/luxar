/**
 * Lazy Loading Monitor Panel
 *
 * Provides real-time monitoring of the lazy loading system, including:
 * - Cache statistics and memory usage
 * - Loading/eviction activity log
 * - Performance metrics
 * - Chunk visualization
 */

// LazyDataManager type is only used for documentation purposes

export interface MonitorEvent {
  timestamp: number;
  type: 'load' | 'evict' | 'hit' | 'miss' | 'preload';
  message: string;
  details?: any;
}

export class LazyLoadingMonitor {
  private container: HTMLElement;
  private panel: HTMLElement;
  private isVisible: boolean = false;
  private events: MonitorEvent[] = [];
  private maxEvents: number = 100;
  private updateInterval: number | null = null;

  // UI Elements
  private statsContainer: HTMLElement | null = null;
  private activityLog: HTMLElement | null = null;
  private performanceContainer: HTMLElement | null = null;
  private chunkMapContainer: HTMLElement | null = null;

  // Metrics
  private loadCount: number = 0;
  private evictCount: number = 0;
  private hitCount: number = 0;
  private missCount: number = 0;
  private totalLoadTime: number = 0;
  private lastUpdateTime: number = Date.now();

  // Collapse states
  private isChunkMapCollapsed: boolean = true;
  private isActivityLogCollapsed: boolean = true;

  constructor(container: HTMLElement) {
    this.container = container;
    this.panel = this.createPanel();
    this.setupEventListeners();
  }

  private createPanel(): HTMLElement {
    const panel = document.createElement('div');
    panel.id = 'lazy-loading-monitor';
    panel.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      width: 400px;
      max-width: 90vw;
      max-height: 80vh;
      background: rgba(30, 30, 30, 0.95);
      border-radius: 8px;
      color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif;
      font-size: 12px;
      z-index: 100;
      display: none;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      backdrop-filter: blur(10px);
    `;

    // Header
    const header = document.createElement('div');
    header.style.cssText = `
      padding: 15px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      justify-content: space-between;
      align-items: center;
    `;

    const title = document.createElement('h3');
    title.innerHTML =
      'Lazy Loading Monitor <span style="font-size: 10px; color: #888; font-weight: normal;">(Global)</span>';
    title.style.cssText = `
      margin: 0;
      font-size: 14px;
      font-weight: bold;
    `;
    title.title = 'Shows combined statistics for all lazy-loaded point clouds in the scene';

    const closeBtn = document.createElement('button');
    closeBtn.textContent = '×';
    closeBtn.style.cssText = `
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
    `;
    closeBtn.onmouseover = () => (closeBtn.style.color = '#fff');
    closeBtn.onmouseout = () => (closeBtn.style.color = '#999');
    closeBtn.onclick = () => this.hide();

    header.appendChild(title);
    header.appendChild(closeBtn);
    panel.appendChild(header);

    // Content container with tabs
    const content = document.createElement('div');
    content.style.cssText = `
      max-height: calc(80vh - 50px);
      overflow-y: auto;
    `;

    // Stats Section
    this.statsContainer = this.createStatsSection();
    content.appendChild(this.statsContainer);

    // Performance Section
    this.performanceContainer = this.createPerformanceSection();
    content.appendChild(this.performanceContainer);

    // Chunk Map Section
    this.chunkMapContainer = this.createChunkMapSection();
    content.appendChild(this.chunkMapContainer);

    // Activity Log Section
    this.activityLog = this.createActivityLogSection();
    content.appendChild(this.activityLog);

    panel.appendChild(content);
    this.container.appendChild(panel);

    return panel;
  }

  private createStatsSection(): HTMLElement {
    const section = document.createElement('div');
    section.style.cssText = `
      padding: 15px;
    `;

    const title = document.createElement('h4');
    title.textContent = 'Cache Statistics';
    title.style.cssText = `
      margin: 0 0 10px 0;
      font-size: 12px;
      color: #4CAF50;
      font-weight: 600;
    `;
    title.title = 'Memory management statistics for cached data chunks';
    section.appendChild(title);

    const stats = document.createElement('div');
    stats.id = 'cache-stats';
    stats.style.cssText = `
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      font-size: 11px;
    `;
    section.appendChild(stats);

    return section;
  }

  private createPerformanceSection(): HTMLElement {
    const section = document.createElement('div');
    section.style.cssText = `
      padding: 15px;
      padding-top: 0;
    `;

    const title = document.createElement('h4');
    title.textContent = 'Performance Metrics';
    title.style.cssText = `
      margin: 0 0 10px 0;
      font-size: 12px;
      color: #FFC107;
      font-weight: 600;
    `;
    title.title = 'Real-time performance indicators for the lazy loading system';
    section.appendChild(title);

    const metrics = document.createElement('div');
    metrics.id = 'performance-metrics';
    metrics.style.cssText = `
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      font-size: 11px;
    `;
    section.appendChild(metrics);

    return section;
  }

  private createChunkMapSection(): HTMLElement {
    const section = document.createElement('div');
    section.style.cssText = `
      padding: 15px;
      padding-top: 0;
    `;

    const headerContainer = document.createElement('div');
    headerContainer.style.cssText = `
      display: flex;
      align-items: center;
      cursor: pointer;
      user-select: none;
      margin-bottom: 10px;
    `;

    const arrow = document.createElement('span');
    arrow.style.cssText = `
      margin-right: 5px;
      font-size: 10px;
      color: rgba(255, 255, 255, 0.6);
      transition: transform 0.2s;
      display: inline-block;
      transform: ${this.isChunkMapCollapsed ? 'rotate(0deg)' : 'rotate(90deg)'};
    `;
    arrow.textContent = '▶';

    const title = document.createElement('h4');
    title.textContent = 'Data Slice Status';
    title.style.cssText = `
      margin: 0;
      font-size: 12px;
      color: #2196F3;
      font-weight: 600;
      flex: 1;
    `;
    title.title =
      'Visual representation of loaded data slices across all point clouds in the scene';

    headerContainer.appendChild(arrow);
    headerContainer.appendChild(title);

    headerContainer.onclick = () => {
      this.isChunkMapCollapsed = !this.isChunkMapCollapsed;
      arrow.style.transform = this.isChunkMapCollapsed ? 'rotate(0deg)' : 'rotate(90deg)';
      const map = section.querySelector('#chunk-map') as HTMLElement;
      if (map) {
        map.style.display = this.isChunkMapCollapsed ? 'none' : 'block';
      }
    };

    section.appendChild(headerContainer);

    const map = document.createElement('div');
    map.id = 'chunk-map';
    map.style.cssText = `
      min-height: 80px;
      background: rgba(0, 0, 0, 0.3);
      border-radius: 4px;
      padding: 10px;
      font-size: 11px;
      display: ${this.isChunkMapCollapsed ? 'none' : 'block'};
    `;
    section.appendChild(map);

    return section;
  }

  private createActivityLogSection(): HTMLElement {
    const section = document.createElement('div');
    section.style.cssText = `
      padding: 15px;
      padding-top: 0;
    `;

    const headerContainer = document.createElement('div');
    headerContainer.style.cssText = `
      display: flex;
      align-items: center;
      cursor: pointer;
      user-select: none;
      margin-bottom: 10px;
    `;

    const arrow = document.createElement('span');
    arrow.style.cssText = `
      margin-right: 5px;
      font-size: 10px;
      color: rgba(255, 255, 255, 0.6);
      transition: transform 0.2s;
      display: inline-block;
      transform: ${this.isActivityLogCollapsed ? 'rotate(0deg)' : 'rotate(90deg)'};
    `;
    arrow.textContent = '▶';

    const title = document.createElement('h4');
    title.textContent = 'Activity Log';
    title.style.cssText = `
      margin: 0;
      font-size: 12px;
      color: #9C27B0;
      font-weight: 600;
      flex: 1;
    `;

    headerContainer.appendChild(arrow);
    headerContainer.appendChild(title);

    headerContainer.onclick = () => {
      this.isActivityLogCollapsed = !this.isActivityLogCollapsed;
      arrow.style.transform = this.isActivityLogCollapsed ? 'rotate(0deg)' : 'rotate(90deg)';
      const log = section.querySelector('#activity-log') as HTMLElement;
      if (log) {
        log.style.display = this.isActivityLogCollapsed ? 'none' : 'block';
      }
    };

    section.appendChild(headerContainer);

    const log = document.createElement('div');
    log.id = 'activity-log';
    log.style.cssText = `
      max-height: 150px;
      overflow-y: auto;
      background: rgba(0, 0, 0, 0.3);
      border-radius: 4px;
      padding: 8px;
      font-size: 10px;
      display: ${this.isActivityLogCollapsed ? 'none' : 'block'};
    `;
    section.appendChild(log);

    return section;
  }

  private setupEventListeners(): void {
    // Listen for Escape key to close
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isVisible) {
        this.hide();
      }
    });
  }

  public show(): void {
    this.panel.style.display = 'block';
    this.isVisible = true;
    this.startUpdating();
    this.logEvent('show', 'Monitor panel opened');
  }

  public hide(): void {
    this.panel.style.display = 'none';
    this.isVisible = false;
    this.stopUpdating();
    this.logEvent('hide', 'Monitor panel closed');
  }

  public getIsVisible(): boolean {
    return this.isVisible;
  }

  public toggle(): void {
    if (this.isVisible) {
      this.hide();
    } else {
      this.show();
    }
  }

  private startUpdating(): void {
    this.updateStats();
    this.updateInterval = window.setInterval(() => {
      this.updateStats();
    }, 500); // Update every 500ms
  }

  private stopUpdating(): void {
    if (this.updateInterval !== null) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }

  private updateStats(): void {
    // Get the global lazy manager instance
    const lazyManager = (window as any).__luxarLazyManager;
    if (!lazyManager) {
      this.updateStatsDisplay({
        numChunks: 0,
        totalSizeMB: 0,
        maxSizeMB: 500,
        utilizationPercent: 0,
      });
      return;
    }

    const stats = lazyManager.getCacheStats();
    this.updateStatsDisplay(stats);
    this.updatePerformanceDisplay();
    this.updateChunkMap(lazyManager);
  }

  private updateStatsDisplay(stats: any): void {
    if (!this.statsContainer) return;

    const statsDiv = this.statsContainer.querySelector('#cache-stats');
    if (!statsDiv) return;

    statsDiv.innerHTML = `
      <div title="Number of data chunks currently stored in memory">
        <span style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Chunks Cached</span><br>
        <span style="color: #4CAF50; font-weight: bold; font-size: 14px;">${stats.numChunks}</span>
      </div>
      <div title="Memory consumed by cached chunks / Maximum allowed memory">
        <span style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Memory Used</span><br>
        <span style="color: ${stats.utilizationPercent > 80 ? '#f44336' : '#4CAF50'}; font-weight: bold; font-size: 14px;">
          ${stats.totalSizeMB.toFixed(1)} / ${stats.maxSizeMB}MB
        </span>
      </div>
      <div title="Percentage of maximum memory currently in use">
        <span style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Utilization</span><br>
        <span style="color: ${stats.utilizationPercent > 80 ? '#f44336' : '#4CAF50'}; font-weight: bold; font-size: 14px;">
          ${stats.utilizationPercent.toFixed(1)}%
        </span>
      </div>
      <div title="Total number of chunks loaded from disk since startup">
        <span style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Total Loads</span><br>
        <span style="color: #2196F3; font-weight: bold; font-size: 14px;">${this.loadCount}</span>
      </div>
    `;
  }

  private updatePerformanceDisplay(): void {
    if (!this.performanceContainer) return;

    const metricsDiv = this.performanceContainer.querySelector('#performance-metrics');
    if (!metricsDiv) return;

    const hitRate =
      this.hitCount + this.missCount > 0
        ? ((this.hitCount / (this.hitCount + this.missCount)) * 100).toFixed(1)
        : '0.0';

    const avgLoadTime = this.loadCount > 0 ? (this.totalLoadTime / this.loadCount).toFixed(0) : '0';

    metricsDiv.innerHTML = `
      <div title="Number of times requested data was found in cache (faster)">
        <span style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Cache Hits</span><br>
        <span style="color: #4CAF50; font-weight: bold; font-size: 14px;">${this.hitCount}</span>
      </div>
      <div title="Number of times data had to be loaded from disk (slower)">
        <span style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Cache Misses</span><br>
        <span style="color: #FFC107; font-weight: bold; font-size: 14px;">${this.missCount}</span>
      </div>
      <div title="Percentage of requests served from cache (higher is better)">
        <span style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Hit Rate</span><br>
        <span style="color: ${parseFloat(hitRate) > 50 ? '#4CAF50' : '#FFC107'}; font-weight: bold; font-size: 14px;">
          ${hitRate}%
        </span>
      </div>
      <div title="Average time to load a chunk from disk">
        <span style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Avg Load Time</span><br>
        <span style="color: #2196F3; font-weight: bold; font-size: 14px;">${avgLoadTime}ms</span>
      </div>
      <div title="Number of chunks removed from cache to free memory">
        <span style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Evictions</span><br>
        <span style="color: #f44336; font-weight: bold; font-size: 14px;">${this.evictCount}</span>
      </div>
      <div title="Frequency of monitor updates">
        <span style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Update Rate</span><br>
        <span style="color: #9C27B0; font-weight: bold; font-size: 14px;">
          ${(1000 / (Date.now() - this.lastUpdateTime)).toFixed(1)} Hz
        </span>
      </div>
    `;

    this.lastUpdateTime = Date.now();
  }

  private updateChunkMap(lazyManager: any): void {
    if (!this.chunkMapContainer) return;

    const mapDiv = this.chunkMapContainer.querySelector('#chunk-map');
    if (!mapDiv) return;

    // Get cache details from the manager
    const cache = lazyManager.cache;
    if (!cache || cache.size === 0) {
      mapDiv.innerHTML =
        '<span style="color: rgba(255, 255, 255, 0.4); font-style: italic;">No data slices loaded yet</span>';
      return;
    }

    // Organize chunks by array type and slice index
    const chunksByType: Map<string, Set<number>> = new Map();
    const chunksByPath: Map<string, Set<string>> = new Map();
    let currentSliceIndex = -1;
    let totalPointClouds = 0;

    cache.forEach((_entry: any, key: string) => {
      const parts = key.split(':');
      const fullPath = parts[0];
      const pathParts = fullPath.split('/');
      const arrayName = pathParts.pop() || 'unknown';
      const objectPath = pathParts.join('/');
      const indices = parts[1] || '';

      // Track unique point cloud objects
      if (!chunksByPath.has(objectPath)) {
        chunksByPath.set(objectPath, new Set());
        if (arrayName === 'positions') {
          totalPointClouds++;
        }
      }
      chunksByPath.get(objectPath)!.add(arrayName);

      // Extract the first index (the slice dimension)
      const indexMatch = indices.match(/^(\d+)/);
      if (indexMatch) {
        const sliceIdx = parseInt(indexMatch[1]);

        if (!chunksByType.has(arrayName)) {
          chunksByType.set(arrayName, new Set());
        }
        chunksByType.get(arrayName)!.add(sliceIdx);

        if (arrayName === 'positions') {
          currentSliceIndex = sliceIdx;
        }
      }
    });

    // Create a compact visualization
    const arrayTypes = Array.from(chunksByType.keys());
    const positionSlices = chunksByType.get('positions') || new Set();
    const sliceIndices = Array.from(positionSlices).sort((a, b) => a - b);

    // Create a visual timeline/slider representation
    let sliceVisualization = '';
    if (sliceIndices.length > 0) {
      const min = Math.min(...sliceIndices);
      const max = Math.max(...sliceIndices);
      const range = max - min + 1;

      // Create a simple text-based visualization
      const visWidth = 40; // characters wide
      const scale = range > visWidth ? visWidth / range : 1;

      let timeline = '';
      for (let i = 0; i < Math.min(range * scale, visWidth); i++) {
        const actualIdx = min + Math.floor(i / scale);
        if (actualIdx === currentSliceIndex) {
          timeline += '█'; // Current position
        } else if (sliceIndices.includes(actualIdx)) {
          timeline += '▓'; // Cached
        } else {
          timeline += '░'; // Not cached
        }
      }

      sliceVisualization = `
        <div style="margin-top: 10px;">
          <div style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Slice Cache Map (${min}-${max}):</div>
          <div style="font-family: monospace; font-size: 14px; letter-spacing: -2px; margin: 5px 0;">
            ${timeline}
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 9px; color: rgba(255, 255, 255, 0.4);">
            <span>${min}</span>
            <span style="color: #4CAF50;">Current: ${currentSliceIndex}</span>
            <span>${max}</span>
          </div>
        </div>
      `;
    }

    // Summary statistics
    const totalChunks = cache.size;
    const uniqueSlices = positionSlices.size;
    const cacheSpread =
      sliceIndices.length > 1
        ? `${Math.min(...sliceIndices)}-${Math.max(...sliceIndices)}`
        : currentSliceIndex.toString();

    mapDiv.innerHTML = `
      <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px;">
        <div title="Total number of data chunks across all arrays and objects">
          <div style="color: rgba(255, 255, 255, 0.4); font-size: 9px;">Total Chunks</div>
          <div style="color: #4CAF50; font-weight: bold; font-size: 12px;">${totalChunks}</div>
        </div>
        <div title="Number of distinct time/dimension slices cached">
          <div style="color: rgba(255, 255, 255, 0.4); font-size: 9px;">Unique Slices</div>
          <div style="color: #2196F3; font-weight: bold; font-size: 12px;">${uniqueSlices}</div>
        </div>
        <div title="Range of slice indices currently in cache">
          <div style="color: rgba(255, 255, 255, 0.4); font-size: 9px;">Slice Range</div>
          <div style="color: #FFC107; font-weight: bold; font-size: 12px;">${cacheSpread}</div>
        </div>
      </div>
      
      <div style="margin-top: 5px;">
        <div style="color: rgba(255, 255, 255, 0.4); font-size: 9px;">Arrays Cached:</div>
        <div style="color: #9C27B0; font-size: 11px;">${arrayTypes.join(', ')}</div>
      </div>
      
      ${
        totalPointClouds > 0
          ? `
      <div style="margin-top: 5px;">
        <div style="color: rgba(255, 255, 255, 0.4); font-size: 9px;">Point Clouds:</div>
        <div style="color: #FF9800; font-size: 11px;">${totalPointClouds} object${totalPointClouds > 1 ? 's' : ''}</div>
      </div>
      `
          : ''
      }
      
      ${sliceVisualization}
      
      <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
        <div style="font-size: 9px; color: rgba(255, 255, 255, 0.4);">
          <span>Legend: </span>
          <span style="color: #4CAF50;">█ Current</span>
          <span style="margin-left: 10px; color: rgba(255, 255, 255, 0.6);">▓ Cached</span>
          <span style="margin-left: 10px; color: rgba(255, 255, 255, 0.3);">░ Not Cached</span>
        </div>
      </div>
    `;
  }

  public logEvent(type: string, message: string, details?: any): void {
    const event: MonitorEvent = {
      timestamp: Date.now(),
      type: type as any,
      message,
      details,
    };

    this.events.unshift(event);
    if (this.events.length > this.maxEvents) {
      this.events.pop();
    }

    // Update metrics based on event type
    switch (type) {
      case 'load':
        this.loadCount++;
        this.missCount++;
        if (details?.loadTime) {
          this.totalLoadTime += details.loadTime;
        }
        break;
      case 'hit':
        this.hitCount++;
        break;
      case 'evict':
        this.evictCount++;
        break;
    }

    this.updateActivityLog();
  }

  private updateActivityLog(): void {
    if (!this.activityLog || !this.isVisible) return;

    const logDiv = this.activityLog.querySelector('#activity-log');
    if (!logDiv) return;

    const logHtml = this.events
      .slice(0, 20)
      .map((event) => {
        const time = new Date(event.timestamp).toLocaleTimeString();
        let icon = '•';
        let color = '#888';

        switch (event.type) {
          case 'load':
            icon = '⬇';
            color = '#4CAF50';
            break;
          case 'evict':
            icon = '×';
            color = '#f44336';
            break;
          case 'hit':
            icon = '✓';
            color = '#2196F3';
            break;
          case 'miss':
            icon = '•';
            color = '#FFC107';
            break;
          case 'preload':
            icon = '→';
            color = '#9C27B0';
            break;
        }

        return `
        <div style="margin-bottom: 4px; color: ${color}; font-size: 10px;">
          <span>${icon}</span>
          <span style="color: rgba(255, 255, 255, 0.4); font-size: 9px;">${time}</span>
          <span style="color: rgba(255, 255, 255, 0.8);">${event.message}</span>
        </div>
      `;
      })
      .join('');

    logDiv.innerHTML =
      logHtml ||
      '<span style="color: rgba(255, 255, 255, 0.4); font-style: italic; font-size: 10px;">No activity yet</span>';
  }

  public dispose(): void {
    this.stopUpdating();
    this.panel.remove();
  }
}
