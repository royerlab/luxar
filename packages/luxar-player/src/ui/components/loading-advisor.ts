/**
 * Loading Advisor Component
 *
 * Analyzes loading performance and provides actionable recommendations
 * for optimization. Detects common issues and suggests solutions.
 */

import type { MonitorEvent, Recommendation, LoaderMetrics } from '../data-monitor-types';

export class LoadingAdvisor {
  private recommendations: Map<string, Recommendation> = new Map();
  private eventHistory: MonitorEvent[] = [];
  private maxHistory = 100;

  // Performance thresholds
  private thresholds = {
    lowCacheHitRate: 30, // Below 30% is concerning
    highQueryTime: 100, // Above 100ms is slow
    highLoadTime: 500, // Above 500ms is slow
    highMemoryUsage: 0.8, // Above 80% memory usage
    highErrorRate: 0.05, // Above 5% error rate
    lowQueryEfficiency: 0.5, // Below 50% efficiency
  };

  /**
   * Analyze event for potential issues
   */
  analyzeEvent(event: MonitorEvent): void {
    this.eventHistory.push(event);
    if (this.eventHistory.length > this.maxHistory) {
      this.eventHistory.shift();
    }

    // Check for specific issues based on event
    switch (event.type) {
      case 'error':
        this.checkErrorRate();
        break;

      case 'query':
        if (event.data.latency && event.data.latency > this.thresholds.highQueryTime) {
          this.addSlowQueryRecommendation(event);
        }
        break;

      case 'load':
        if (event.data.latency && event.data.latency > this.thresholds.highLoadTime) {
          this.addSlowLoadRecommendation(event);
        }
        break;

      case 'evict':
        this.checkMemoryPressure();
        break;
    }
  }

  /**
   * Analyze metrics for issues
   */
  analyzeMetrics(metrics: LoaderMetrics): void {
    // Check cache hit rate
    if (metrics.cacheHitRate < this.thresholds.lowCacheHitRate) {
      this.addLowCacheRateRecommendation(metrics);
    }

    // Check query performance
    if (metrics.avgQueryTime > this.thresholds.highQueryTime) {
      this.addHighQueryTimeRecommendation(metrics);
    }

    // Check memory usage
    if (metrics.memoryUsed / metrics.memoryLimit > this.thresholds.highMemoryUsage) {
      this.addHighMemoryRecommendation(metrics);
    }

    // Check spatial index efficiency
    if (metrics.spatialIndex) {
      const efficiency = metrics.spatialIndex.queryEfficiency;
      if (efficiency < this.thresholds.lowQueryEfficiency) {
        this.addLowEfficiencyRecommendation(metrics);
      }
    }
  }

  /**
   * Add slow query recommendation
   */
  private addSlowQueryRecommendation(event: MonitorEvent): void {
    const rec: Recommendation = {
      id: 'slow-query',
      severity: 'warning',
      category: 'performance',
      title: 'Slow Query Performance',
      message: `Queries taking ${event.data.latency}ms on average`,
      suggestion: 'Check network latency or consider increasing cache size',
      metric: 'queryTime',
      value: event.data.latency,
      threshold: this.thresholds.highQueryTime,
    };

    this.recommendations.set(rec.id, rec);
  }

  /**
   * Add slow load recommendation
   */
  private addSlowLoadRecommendation(event: MonitorEvent): void {
    const rec: Recommendation = {
      id: 'slow-load',
      severity: 'warning',
      category: 'performance',
      title: 'Slow Data Loading',
      message: `Data loads taking ${event.data.latency}ms`,
      suggestion: 'Check network bandwidth and consider using smaller chunk sizes',
      metric: 'loadTime',
      value: event.data.latency,
      threshold: this.thresholds.highLoadTime,
    };

    this.recommendations.set(rec.id, rec);
  }

  /**
   * Add low cache rate recommendation
   */
  private addLowCacheRateRecommendation(metrics: LoaderMetrics): void {
    const rec: Recommendation = {
      id: 'low-cache-rate',
      severity: 'warning',
      category: 'performance',
      title: 'Low Cache Hit Rate',
      message: `Only ${metrics.cacheHitRate.toFixed(1)}% of requests served from cache`,
      suggestion: 'Increase cache size or adjust preload radius for better performance',
      metric: 'cacheHitRate',
      value: metrics.cacheHitRate,
      threshold: this.thresholds.lowCacheHitRate,
    };

    this.recommendations.set(rec.id, rec);
  }

  /**
   * Add high query time recommendation
   */
  private addHighQueryTimeRecommendation(metrics: LoaderMetrics): void {
    const rec: Recommendation = {
      id: 'high-avg-query',
      severity: metrics.avgQueryTime > 200 ? 'error' : 'warning',
      category: 'performance',
      title: 'High Average Query Time',
      message: `Average query time is ${metrics.avgQueryTime.toFixed(0)}ms`,
      suggestion:
        'Spatial index may be suboptimal - consider rebuilding with different grid resolution',
      metric: 'avgQueryTime',
      value: metrics.avgQueryTime,
      threshold: this.thresholds.highQueryTime,
    };

    this.recommendations.set(rec.id, rec);
  }

  /**
   * Add high memory recommendation
   */
  private addHighMemoryRecommendation(metrics: LoaderMetrics): void {
    const usage = (metrics.memoryUsed / metrics.memoryLimit) * 100;

    const rec: Recommendation = {
      id: 'high-memory',
      severity: usage > 90 ? 'error' : 'warning',
      category: 'memory',
      title: 'High Memory Usage',
      message: `Using ${usage.toFixed(1)}% of available memory`,
      suggestion: 'Consider reducing cache size or enabling more aggressive eviction',
      metric: 'memoryUsage',
      value: usage,
      threshold: this.thresholds.highMemoryUsage * 100,
    };

    this.recommendations.set(rec.id, rec);
  }

  /**
   * Add low efficiency recommendation
   */
  private addLowEfficiencyRecommendation(metrics: LoaderMetrics): void {
    if (!metrics.spatialIndex) return;

    const efficiency = metrics.spatialIndex.queryEfficiency;

    const rec: Recommendation = {
      id: 'low-efficiency',
      severity: 'info',
      category: 'configuration',
      title: 'Suboptimal Query Efficiency',
      message: `Spatial queries only ${(efficiency * 100).toFixed(1)}% efficient`,
      suggestion: 'Grid resolution may be too coarse - consider rebuilding with finer grid',
      metric: 'queryEfficiency',
      value: efficiency,
      threshold: this.thresholds.lowQueryEfficiency,
    };

    this.recommendations.set(rec.id, rec);
  }

  /**
   * Check error rate
   */
  private checkErrorRate(): void {
    const recentEvents = this.eventHistory.slice(-20);
    const errors = recentEvents.filter((e) => e.type === 'error').length;
    const errorRate = errors / recentEvents.length;

    if (errorRate > this.thresholds.highErrorRate) {
      const rec: Recommendation = {
        id: 'high-errors',
        severity: 'error',
        category: 'performance',
        title: 'High Error Rate',
        message: `${(errorRate * 100).toFixed(1)}% of recent operations failed`,
        suggestion: 'Check network connectivity and data availability',
        metric: 'errorRate',
        value: errorRate,
        threshold: this.thresholds.highErrorRate,
      };

      this.recommendations.set(rec.id, rec);
    }
  }

  /**
   * Check memory pressure
   */
  private checkMemoryPressure(): void {
    const recentEvictions = this.eventHistory.filter(
      (e) => e.type === 'evict' && e.timestamp > Date.now() - 10000
    ).length;

    if (recentEvictions > 5) {
      const rec: Recommendation = {
        id: 'memory-pressure',
        severity: 'warning',
        category: 'memory',
        title: 'Frequent Cache Evictions',
        message: `${recentEvictions} evictions in last 10 seconds`,
        suggestion: 'Memory pressure detected - consider increasing cache limit',
      };

      this.recommendations.set(rec.id, rec);
    }
  }

  /**
   * Get all current recommendations
   */
  getRecommendations(): Recommendation[] {
    // Clean up old recommendations that may no longer apply
    this.cleanupRecommendations();

    // Sort by severity
    const severityOrder = { error: 0, warning: 1, info: 2 };
    return Array.from(this.recommendations.values()).sort(
      (a, b) => severityOrder[a.severity] - severityOrder[b.severity]
    );
  }

  /**
   * Clean up old recommendations
   */
  private cleanupRecommendations(): void {
    // Remove recommendations that haven't been reinforced recently
    // const now = Date.now();
    // Check if issues are still present
    // For now, just timeout old recommendations
    // In a real implementation, we'd check if the issue is resolved
  }

  /**
   * Check if there are warnings
   */
  hasWarnings(): boolean {
    return Array.from(this.recommendations.values()).some(
      (r) => r.severity === 'warning' || r.severity === 'error'
    );
  }

  /**
   * Clear all recommendations
   */
  clear(): void {
    this.recommendations.clear();
    this.eventHistory = [];
  }

  /**
   * Dispose
   */
  dispose(): void {
    this.recommendations.clear();
    this.eventHistory = [];
  }
}
