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
  type: 'load' | 'evict' | 'hit' | 'miss' | 'preload' | 'memory' | 'clear' | 'show' | 'hide';
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

  // Selected object for chunk map visualization
  private selectedObject: string = '';

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
      'Data Loading and Caching Monitor <span style="font-size: 10px; color: #888; font-weight: normal;">(Global)</span>';
    title.style.cssText = `
      margin: 0;
      font-size: 14px;
      font-weight: bold;
    `;
    title.title = 'Shows combined statistics for all data loading and caching in the scene';

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
    title.title = 'Visual representation of loaded data slices for selected object';

    headerContainer.appendChild(arrow);
    headerContainer.appendChild(title);

    headerContainer.onclick = () => {
      this.isChunkMapCollapsed = !this.isChunkMapCollapsed;
      arrow.style.transform = this.isChunkMapCollapsed ? 'rotate(0deg)' : 'rotate(90deg)';
      const map = section.querySelector('#chunk-map') as HTMLElement;
      const selectorContainer = section.querySelector('#object-selector')
        ?.parentElement as HTMLElement;
      if (map) {
        map.style.display = this.isChunkMapCollapsed ? 'none' : 'block';
      }
      if (selectorContainer) {
        selectorContainer.style.display = this.isChunkMapCollapsed ? 'none' : 'block';
      }
    };

    section.appendChild(headerContainer);

    // Add object/array selector dropdown
    const selectorContainer = document.createElement('div');
    selectorContainer.style.cssText = `
      margin-bottom: 10px;
      display: ${this.isChunkMapCollapsed ? 'none' : 'block'};
    `;

    const selector = document.createElement('select');
    selector.id = 'object-selector';
    selector.style.cssText = `
      width: 100%;
      padding: 5px;
      background: rgba(0, 0, 0, 0.5);
      color: #e0e0e0;
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 3px;
      font-size: 11px;
    `;

    selectorContainer.appendChild(selector);
    section.appendChild(selectorContainer);

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
      <div title="Memory consumed by cached chunks / Maximum allowed memory (auto-detected)">
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
    const selector = this.chunkMapContainer.querySelector('#object-selector') as HTMLSelectElement;
    if (!mapDiv) return;

    // Get cache details from the manager
    const cache = lazyManager.cache;

    // Get dataset metadata for full range
    const totalSlices = lazyManager.getDatasetMetadata
      ? lazyManager.getDatasetMetadata('totalSlices') || 1
      : 1;

    if (!cache || cache.size === 0) {
      // Even with no cache, show the full range if we know it
      if (totalSlices > 1) {
        // When no data is cached, we use the metadata totalSlices as-is
        mapDiv.innerHTML = this.createSliceVisualization([], totalSlices);
      } else {
        mapDiv.innerHTML =
          '<span style="color: rgba(255, 255, 255, 0.4); font-style: italic;">No data slices loaded yet</span>';
      }
      if (selector) selector.innerHTML = '<option>No data loaded</option>';
      return;
    }

    // Organize chunks by object path
    const objectData: Map<
      string,
      {
        arrays: Set<string>;
        slices: Set<number>;
        currentSlice: number;
      }
    > = new Map();

    cache.forEach((_entry: any, key: string) => {
      const parts = key.split(':');
      const fullPath = parts[0];
      const pathParts = fullPath.split('/');
      const arrayName = pathParts.pop() || 'unknown';
      const objectPath = pathParts.join('/') || '/';
      const indices = parts[1] || '';

      // Initialize object data if needed
      if (!objectData.has(objectPath)) {
        objectData.set(objectPath, {
          arrays: new Set(),
          slices: new Set(),
          currentSlice: -1,
        });
      }

      const objData = objectData.get(objectPath)!;
      objData.arrays.add(arrayName);

      // Extract chunk indices - these are zarr chunk coordinates, not necessarily
      // corresponding to dimensional slices in the visualization.
      // The relationship between chunks and slices depends on the data organization.
      const indexMatch = indices.match(/^(\d+)/);
      if (indexMatch) {
        // Store the first chunk index as a "slice" for visualization purposes
        // This may not directly correspond to dimensional navigation slices
        const chunkIdx = parseInt(indexMatch[1]);
        objData.slices.add(chunkIdx);
      }
    });

    // Update dropdown selector
    if (selector) {
      const currentValue = selector.value || this.selectedObject;
      selector.innerHTML = '';

      // Add "All Objects" option
      const allOption = document.createElement('option');
      allOption.value = '__all__';
      allOption.textContent = `All Objects (${objectData.size} total)`;
      selector.appendChild(allOption);

      // Add individual object options (sorted alphabetically)
      const sortedPaths = Array.from(objectData.keys()).sort();
      sortedPaths.forEach((path) => {
        const data = objectData.get(path)!;
        const option = document.createElement('option');
        option.value = path;

        // Format the display name nicely
        let displayName: string;
        if (path === '/' || path === '') {
          displayName = '📦 Root Dataset';
        } else {
          // Extract just the object name from the path
          const objectName =
            path
              .split('/')
              .filter((p) => p)
              .pop() || path;
          // Add appropriate icon based on content
          const icon = data.arrays.has('positions') ? '🌟' : '📊';
          displayName = `${icon} ${objectName}`;
        }

        option.textContent = `${displayName} (${data.arrays.size} arrays, ${data.slices.size} chunks)`;
        selector.appendChild(option);
      });

      // Restore selection or select first object
      if (currentValue && Array.from(selector.options).some((opt) => opt.value === currentValue)) {
        selector.value = currentValue;
        this.selectedObject = currentValue;
      } else if (objectData.size > 0) {
        this.selectedObject = selector.options[1]?.value || '__all__';
        selector.value = this.selectedObject;
      }

      // Add change event listener
      selector.onchange = () => {
        this.selectedObject = selector.value;
        this.updateChunkMap(lazyManager);
      };
    }

    // Display data for selected object
    let displayData: {
      arrays: Set<string>;
      slices: Set<number>;
    };

    if (this.selectedObject === '__all__' || !this.selectedObject) {
      // Show summary for all objects
      let allSlices = new Set<number>();
      let allArrays = new Set<string>();

      objectData.forEach((data) => {
        data.slices.forEach((s) => allSlices.add(s));
        data.arrays.forEach((a) => allArrays.add(a));
      });

      displayData = {
        arrays: allArrays,
        slices: allSlices,
      };
    } else {
      displayData = objectData.get(this.selectedObject) || {
        arrays: new Set<string>(),
        slices: new Set<number>(),
      };
    }

    const sliceIndices = Array.from(displayData.slices).sort((a, b) => a - b);

    // Create a compact visualization
    const arrayTypes = Array.from(displayData.arrays) as string[];

    // Determine the actual total slices - it's either from metadata or the max slice index + 1
    // (whichever is larger, since actual data might exceed metadata range)
    const maxSliceIndex = sliceIndices.length > 0 ? Math.max(...sliceIndices) : -1;
    const actualTotalSlices = Math.max(totalSlices, maxSliceIndex + 1);

    // Create a visual slice map that always shows the full range
    const sliceVisualization = this.createSliceVisualization(sliceIndices, actualTotalSlices);

    // Summary statistics
    const totalChunks =
      this.selectedObject === '__all__'
        ? cache.size
        : Array.from(cache.keys() as IterableIterator<string>).filter((k) =>
            k.startsWith(this.selectedObject)
          ).length;
    const uniqueSlices = displayData.slices.size;
    const cacheSpread = actualTotalSlices > 1 ? `0-${actualTotalSlices - 1}` : '0';

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
        this.selectedObject === '__all__' && objectData.size > 0
          ? `
      <div style="margin-top: 5px;">
        <div style="color: rgba(255, 255, 255, 0.4); font-size: 9px;">Objects:</div>
        <div style="color: #FF9800; font-size: 11px;">${objectData.size} object${objectData.size > 1 ? 's' : ''}</div>
      </div>
      `
          : ''
      }
      
      ${sliceVisualization}
      
      <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
        <div style="font-size: 9px; color: rgba(255, 255, 255, 0.4); display: flex; align-items: center; gap: 10px;">
          <span>Legend:</span>
          <span style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; background: rgba(76, 175, 80, 0.6); border-radius: 2px; margin-right: 4px;"></span>
            Cached
          </span>
          <span style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; background: rgba(255, 255, 255, 0.1); border-radius: 2px; margin-right: 4px;"></span>
            Not Cached
          </span>
        </div>
      </div>
    `;
  }

  private createSliceVisualization(cachedSlices: number[], totalSlices: number): string {
    if (totalSlices <= 0) return '';

    const min = 0;
    const max = totalSlices - 1;
    const sliceIndices = cachedSlices;

    // Build the visual timeline using proper HTML elements
    let timeline = '';
    const maxSegments = 50; // Maximum number of visual segments

    if (totalSlices <= maxSegments) {
      // If range is small enough, show each slice individually
      for (let i = min; i <= max; i++) {
        const isCached = sliceIndices.includes(i);

        const bgColor = isCached
          ? 'rgba(76, 175, 80, 0.6)' // Green for cached
          : 'rgba(255, 255, 255, 0.1)'; // Dim for not cached

        timeline += `<div style="
          flex: 1;
          height: 100%;
          background: ${bgColor};
          border-left: 1px solid rgba(0,0,0,0.2);
        " title="Slice ${i}: ${isCached ? 'Cached' : 'Not cached'}"></div>`;
      }
    } else {
      // For large ranges, aggregate into segments
      const segmentSize = totalSlices / maxSegments;
      for (let i = 0; i < maxSegments; i++) {
        const sliceStart = Math.floor(min + i * segmentSize);
        const sliceEnd = Math.floor(min + (i + 1) * segmentSize);

        // Calculate cache density in this segment
        let cachedInSegment = 0;
        for (let j = sliceStart; j < sliceEnd && j <= max; j++) {
          if (sliceIndices.includes(j)) cachedInSegment++;
        }
        const actualSegmentSize = sliceEnd - sliceStart;
        const segmentCacheRatio = actualSegmentSize > 0 ? cachedInSegment / actualSegmentSize : 0;

        // Gradient based on cache density
        const opacity = 0.1 + segmentCacheRatio * 0.6;
        const bgColor = `rgba(76, 175, 80, ${opacity})`;

        timeline += `<div style="
          flex: 1;
          height: 100%;
          background: ${bgColor};
          border-left: 1px solid rgba(0,0,0,0.2);
        " title="Slices ${sliceStart}-${sliceEnd}: ${Math.round(segmentCacheRatio * 100)}% cached"></div>`;
      }
    }

    const cacheHitRate = sliceIndices.length / totalSlices;
    const percentCached = (cacheHitRate * 100).toFixed(1);

    return `
      <div style="margin-top: 10px; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
        <div style="color: rgba(255, 255, 255, 0.6); font-size: 10px; margin-bottom: 5px;">
          Slice Cache Map (0-${max}):
        </div>
        
        <!-- Visual timeline using flexbox -->
        <div style="
          display: flex;
          height: 20px;
          width: 100%;
          margin: 8px 0;
          border-radius: 2px;
          overflow: hidden;
          background: rgba(0, 0, 0, 0.3);
          box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3);
        ">
          ${timeline}
        </div>
        
        <div style="display: flex; justify-content: space-between; font-size: 9px; color: rgba(255, 255, 255, 0.4); margin-top: 5px;">
          <span>0</span>
          <span>Total: ${totalSlices} slices</span>
          <span>${max}</span>
        </div>
        
        <!-- Cache statistics -->
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div>
              <div style="color: rgba(255, 255, 255, 0.4); font-size: 9px;">Cache Coverage</div>
              <div style="color: #4CAF50; font-weight: bold; font-size: 12px;">${percentCached}%</div>
            </div>
            <div>
              <div style="color: rgba(255, 255, 255, 0.4); font-size: 9px;">Cached Chunks</div>
              <div style="color: #2196F3; font-weight: bold; font-size: 12px;">${sliceIndices.length}/${totalSlices}</div>
            </div>
          </div>
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
          case 'memory':
            icon = '💾';
            color = '#FF9800';
            break;
          case 'clear':
            icon = '🗑';
            color = '#607D8B';
            break;
        }

        // Extract object name from the message if it contains a cache key
        let formattedMessage = event.message;
        let objectTag = '';

        // Look for cache keys in the format "ObjectName/arrayType:indices"
        const cacheKeyMatch = event.message.match(/([^:\s]+\/[^:\s]+)(?::\S+)?/);
        if (cacheKeyMatch) {
          const fullPath = cacheKeyMatch[1];
          const pathParts = fullPath.split('/');
          if (pathParts.length >= 2) {
            const objectName = pathParts[pathParts.length - 2] || 'Root';
            const arrayType = pathParts[pathParts.length - 1];

            // Create a colored object tag
            const objectColor = this.getObjectColor(objectName);
            objectTag = `<span style="background: ${objectColor}; color: #000; padding: 1px 4px; border-radius: 2px; font-weight: bold; margin-right: 4px; font-size: 9px;">${objectName}</span>`;

            // Simplify the message to just show the array type and action
            formattedMessage = event.message.replace(cacheKeyMatch[0], arrayType);
          }
        }

        return `
        <div style="margin-bottom: 4px; color: ${color}; font-size: 10px;">
          <span>${icon}</span>
          <span style="color: rgba(255, 255, 255, 0.4); font-size: 9px;">${time}</span>
          ${objectTag}
          <span style="color: rgba(255, 255, 255, 0.8);">${formattedMessage}</span>
        </div>
      `;
      })
      .join('');

    logDiv.innerHTML =
      logHtml ||
      '<span style="color: rgba(255, 255, 255, 0.4); font-style: italic; font-size: 10px;">No activity yet</span>';
  }

  public reset(): void {
    // Clear all counters
    this.loadCount = 0;
    this.evictCount = 0;
    this.hitCount = 0;
    this.missCount = 0;
    this.totalLoadTime = 0;
    this.events = [];
    this.selectedObject = '';

    // Update displays if visible
    if (this.isVisible) {
      this.updateStats();
    }
  }

  /**
   * Generate a consistent color for an object name
   * Uses a simple hash function to ensure the same object always gets the same color
   */
  private getObjectColor(objectName: string): string {
    // Predefined palette of distinguishable colors
    const colors = [
      '#4CAF50', // Green
      '#2196F3', // Blue
      '#FF9800', // Orange
      '#9C27B0', // Purple
      '#F44336', // Red
      '#00BCD4', // Cyan
      '#FFEB3B', // Yellow
      '#795548', // Brown
      '#607D8B', // Blue Grey
      '#E91E63', // Pink
      '#8BC34A', // Light Green
      '#3F51B5', // Indigo
    ];

    // Simple hash function to get consistent index for each object name
    let hash = 0;
    for (let i = 0; i < objectName.length; i++) {
      hash = (hash << 5) - hash + objectName.charCodeAt(i);
      hash = hash & hash; // Convert to 32-bit integer
    }

    // Use absolute value and modulo to get index
    const index = Math.abs(hash) % colors.length;
    return colors[index];
  }

  public dispose(): void {
    this.stopUpdating();
    this.panel.remove();
  }
}
