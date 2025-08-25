/**
 * Performance Timeline Component
 *
 * Real-time graph showing loading performance, query patterns,
 * and cache efficiency over time.
 */

import type { MonitorEvent, TimelinePoint } from '../data-monitor-types';
import { config } from '../../config';

const MonitorTimings = config.dataLoading.monitor.timings;
const MonitorLimits = config.dataLoading.monitor.limits;

export class PerformanceTimeline {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  // Timeline data
  private points: TimelinePoint[] = [];
  private maxPoints = MonitorLimits.maxTimelinePoints;
  private timeRange: number = MonitorTimings.defaultTimeRange;

  // Metrics tracking
  private lastQueryTime = 0;
  private lastLoadTime = 0;
  private recentCacheHits = 0;
  private recentCacheMisses = 0;
  private lastCacheCounterReset = Date.now();

  // Rendering optimization
  private renderPending = false;
  private animationFrameId: number | null = null;
  private lastRenderTime = 0;
  private minRenderInterval = MonitorTimings.minRenderInterval;
  private needsRender = false;

  // Colors
  private colors = {
    background: 'rgba(0, 0, 0, 0.5)',
    grid: 'rgba(255, 255, 255, 0.1)',
    queryTime: '#2196F3',
    loadTime: '#4CAF50',
    cacheRate: '#FFC107',
    error: '#F44336',
  };

  /**
   * Add event to timeline
   */
  addEvent(event: MonitorEvent): void {
    const now = Date.now();

    // Update metrics based on event
    switch (event.type) {
      case 'query':
        if (event.data.latency) {
          this.lastQueryTime = event.data.latency;
        }
        break;

      case 'load':
        if (event.data.latency) {
          this.lastLoadTime = event.data.latency;
        }
        break;

      case 'cache-hit':
        this.recentCacheHits++;
        break;

      case 'cache-miss':
        this.recentCacheMisses++;
        break;
    }

    // Add timeline point only at reasonable intervals (aggregate events)
    const lastPoint = this.points[this.points.length - 1];
    const shouldAddPoint =
      !lastPoint || now - lastPoint.timestamp > MonitorTimings.timelinePointInterval;

    if (shouldAddPoint) {
      const point: TimelinePoint = {
        timestamp: now,
        queryTime: this.lastQueryTime,
        loadTime: this.lastLoadTime,
        cacheHitRate: this.calculateRecentCacheRate(),
        event: event.type,
        loaderType: event.loader,
      };

      this.points.push(point);

      // Efficient trimming: remove old points in one operation
      // Keep points from last 5 minutes AND respect max points limit
      const cutoff = now - MonitorLimits.maxTimelinePoints * 1000; // Convert to ms

      // Find the index of the first point to keep
      let keepFromIndex = 0;
      for (let i = 0; i < this.points.length; i++) {
        if (this.points[i].timestamp >= cutoff) {
          keepFromIndex = i;
          break;
        }
      }

      // If we need to trim by time, do it efficiently
      if (keepFromIndex > 0) {
        this.points = this.points.slice(keepFromIndex);
      }

      // Also enforce max points limit efficiently
      if (this.points.length > this.maxPoints) {
        // Keep only the most recent maxPoints
        this.points = this.points.slice(this.points.length - this.maxPoints);
      }
    }

    // Reset cache counters every 5 seconds (reliable timing)
    if (now - this.lastCacheCounterReset >= 5000) {
      this.recentCacheHits = 0;
      this.recentCacheMisses = 0;
      this.lastCacheCounterReset = now;
    }

    // Schedule render with requestAnimationFrame
    this.scheduleRender();
  }

  /**
   * Initialize canvas
   */
  initializeCanvas(canvasId: string): void {
    this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    if (this.canvas) {
      this.ctx = this.canvas.getContext('2d');

      // Get the parent container width to ensure canvas fits properly
      const parent = this.canvas.parentElement;
      const maxWidth = parent ? parent.clientWidth : 400;

      // Set canvas display size (CSS)
      this.canvas.style.width = `${maxWidth}px`;
      this.canvas.style.height = '200px';

      // Set up proper canvas resolution for retina displays
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = maxWidth * dpr;
      this.canvas.height = 200 * dpr;
      this.ctx?.scale(dpr, dpr);

      this.scheduleRender();
    }
  }

  /**
   * Set time range (no-op for now, fixed at 60s)
   */
  setTimeRange(range: number): void {
    // Update the time range
    this.timeRange = range;
    this.scheduleRender();
  }

  /**
   * Schedule render with requestAnimationFrame
   */
  private scheduleRender(): void {
    this.needsRender = true;

    if (!this.renderPending) {
      this.renderPending = true;

      // Cancel any pending frame
      if (this.animationFrameId !== null) {
        cancelAnimationFrame(this.animationFrameId);
      }

      this.animationFrameId = requestAnimationFrame(() => {
        this.performRender();
      });
    }
  }

  /**
   * Perform the actual render with throttling
   */
  private performRender(): void {
    const now = Date.now();
    const timeSinceLastRender = now - this.lastRenderTime;

    // Throttle renders to max 10 FPS
    if (timeSinceLastRender < this.minRenderInterval) {
      // Schedule another frame
      this.animationFrameId = requestAnimationFrame(() => {
        this.performRender();
      });
      return;
    }

    // Reset flags
    this.renderPending = false;
    this.animationFrameId = null;
    this.lastRenderTime = now;

    // Only render if we actually need to
    if (this.needsRender) {
      this.render();
      this.needsRender = false;
    }
  }

  /**
   * Render timeline
   */
  private render(): void {
    if (!this.canvas || !this.ctx) return;

    const ctx = this.ctx;
    // Use display size, not pixel size (for retina displays)
    const rect = this.canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background
    ctx.fillStyle = this.colors.background;
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    this.drawGrid(ctx, width, height);

    // Get points in time range
    const now = Date.now();
    const startTime = now - this.timeRange * 1000;
    const visiblePoints = this.points.filter((p) => p.timestamp >= startTime);

    if (visiblePoints.length < 2) {
      // Not enough data
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Collecting data...', width / 2, height / 2);
      return;
    }

    // Draw metrics
    this.drawMetric(ctx, width, height, visiblePoints, 'queryTime', this.colors.queryTime, 100);
    this.drawMetric(ctx, width, height, visiblePoints, 'loadTime', this.colors.loadTime, 100);
    this.drawMetric(
      ctx,
      width,
      height,
      visiblePoints,
      'cacheHitRate',
      this.colors.cacheRate,
      100,
      true
    );

    // Draw events
    this.drawEvents(ctx, width, height, visiblePoints);

    // Draw legend
    this.drawLegend(ctx, width, height);
  }

  /**
   * Draw grid lines
   */
  private drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    ctx.strokeStyle = this.colors.grid;
    ctx.lineWidth = 1;

    // Horizontal lines (every 25%)
    for (let i = 1; i < 4; i++) {
      const y = (height * i) / 4;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Vertical lines (every 10 seconds)
    const step = (width * 10) / this.timeRange;
    for (let x = step; x < width; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }

  /**
   * Draw a metric line
   */
  private drawMetric(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    points: TimelinePoint[],
    metric: keyof TimelinePoint,
    color: string,
    maxValue: number,
    isPercentage = false
  ): void {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    let firstPoint = true;
    const now = Date.now();
    const timeSpan = this.timeRange * 1000;

    for (const point of points) {
      const value = point[metric] as number;
      if (value === undefined || value === null) continue;

      const x = ((point.timestamp - (now - timeSpan)) / timeSpan) * width;
      const y = height - (value / maxValue) * height * 0.8 - height * 0.1;

      if (firstPoint) {
        ctx.moveTo(x, y);
        firstPoint = false;
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();

    // Draw current value
    const lastPoint = points[points.length - 1];
    const lastValue = lastPoint[metric] as number;
    if (lastValue !== undefined) {
      ctx.fillStyle = color;
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'right';
      const displayValue = isPercentage ? `${lastValue.toFixed(0)}%` : `${lastValue.toFixed(0)}ms`;
      ctx.fillText(displayValue, width - 5, 15 + this.getMetricOffset(metric));
    }
  }

  /**
   * Draw event markers
   */
  private drawEvents(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    points: TimelinePoint[]
  ): void {
    const now = Date.now();
    const timeSpan = this.timeRange * 1000;

    for (const point of points) {
      if (point.event === 'error') {
        const x = ((point.timestamp - (now - timeSpan)) / timeSpan) * width;

        // Draw error marker
        ctx.strokeStyle = this.colors.error;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
    }
  }

  /**
   * Draw legend
   */
  private drawLegend(ctx: CanvasRenderingContext2D, _width: number, height: number): void {
    const legends = [
      { label: 'Query', color: this.colors.queryTime },
      { label: 'Load', color: this.colors.loadTime },
      { label: 'Cache', color: this.colors.cacheRate },
    ];

    ctx.font = '10px sans-serif';
    let x = 10;

    for (const legend of legends) {
      // Draw color dot
      ctx.fillStyle = legend.color;
      ctx.beginPath();
      ctx.arc(x, height - 10, 3, 0, Math.PI * 2);
      ctx.fill();

      // Draw label
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.textAlign = 'left';
      ctx.fillText(legend.label, x + 8, height - 7);

      x += 60;
    }
  }

  /**
   * Get metric offset for label positioning
   */
  private getMetricOffset(metric: keyof TimelinePoint): number {
    switch (metric) {
      case 'queryTime':
        return 0;
      case 'loadTime':
        return 15;
      case 'cacheHitRate':
        return 30;
      default:
        return 0;
    }
  }

  /**
   * Calculate recent cache hit rate
   */
  private calculateRecentCacheRate(): number {
    const total = this.recentCacheHits + this.recentCacheMisses;
    if (total === 0) return 0;
    return (this.recentCacheHits / total) * 100;
  }

  /**
   * Clear timeline
   */
  clear(): void {
    this.points = [];
    this.lastQueryTime = 0;
    this.lastLoadTime = 0;
    this.recentCacheHits = 0;
    this.recentCacheMisses = 0;
    this.lastCacheCounterReset = Date.now();
    this.scheduleRender();
  }

  /**
   * Get current statistics
   */
  getStats(): {
    avgQueryTime: number;
    avgLoadTime: number;
    avgCacheRate: number;
  } {
    if (this.points.length === 0) {
      return { avgQueryTime: 0, avgLoadTime: 0, avgCacheRate: 0 };
    }

    let totalQuery = 0;
    let queryCount = 0;
    let totalLoad = 0;
    let loadCount = 0;
    let totalCache = 0;
    let cacheCount = 0;

    for (const point of this.points) {
      if (point.queryTime !== undefined) {
        totalQuery += point.queryTime;
        queryCount++;
      }
      if (point.loadTime !== undefined) {
        totalLoad += point.loadTime;
        loadCount++;
      }
      if (point.cacheHitRate !== undefined) {
        totalCache += point.cacheHitRate;
        cacheCount++;
      }
    }

    return {
      avgQueryTime: queryCount > 0 ? totalQuery / queryCount : 0,
      avgLoadTime: loadCount > 0 ? totalLoad / loadCount : 0,
      avgCacheRate: cacheCount > 0 ? totalCache / cacheCount : 0,
    };
  }

  /**
   * Dispose
   */
  dispose(): void {
    const errors: Error[] = [];

    // Cancel any pending animation frame
    if (this.animationFrameId !== null) {
      try {
        cancelAnimationFrame(this.animationFrameId);
      } catch (error) {
        // Non-critical - animation frame may already be cancelled
        errors.push(new Error(`Failed to cancel animation frame: ${error}`));
      }
      this.animationFrameId = null;
    }

    // Clear context and canvas references
    if (this.ctx && this.canvas) {
      try {
        // Clear the canvas before disposing
        const rect = this.canvas.getBoundingClientRect();
        if (rect && rect.width > 0 && rect.height > 0) {
          this.ctx.clearRect(0, 0, rect.width, rect.height);
        }
      } catch (error) {
        // Canvas might be detached or context lost - non-critical
        errors.push(new Error(`Failed to clear canvas: ${error}`));
      }
    }

    // Clear references to allow garbage collection
    this.canvas = null;
    this.ctx = null;
    this.points = [];

    // Reset render state
    this.renderPending = false;
    this.needsRender = false;
    this.lastRenderTime = 0;

    // Log any non-critical errors for debugging
    if (errors.length > 0) {
      console.debug('[PerformanceTimeline] Disposal completed with non-critical errors:', errors);
    }
  }
}
