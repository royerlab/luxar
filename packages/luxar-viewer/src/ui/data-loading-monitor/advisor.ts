/**
 * Loading Advisor Component
 *
 * Analyzes loading performance and provides actionable recommendations
 * for optimization. Detects common issues and suggests solutions.
 */

import type { MonitorEvent, Recommendation, LoaderMetrics } from '../../types/data-monitor-types';
import type { MemoryMetrics } from './templates';
import { config } from '../../config';
import { POOLED_GEOMETRY_TYPES } from '../../types/data-monitor-types';

const PerformanceThresholds = config.dataLoading.monitor.thresholds;
const MonitorLimits = config.dataLoading.monitor.limits;

export class LoadingAdvisor {
  private recommendations: Map<string, Recommendation> = new Map();
  private eventHistory: MonitorEvent[] = [];
  private maxHistory = MonitorLimits.maxAdvisorHistory;

  // Use centralized performance thresholds
  private thresholds = PerformanceThresholds;

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
    }
  }

  /**
   * Analyze metrics for issues
   */
  analyzeMetrics(metrics: LoaderMetrics): void {
    // Note: Cache hit rate analysis removed (L0 cache removed)

    // Check query performance
    if (metrics.avgQueryTime > this.thresholds.highQueryTime) {
      this.addHighQueryTimeRecommendation(metrics);
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
   * Update recommendations based on global stats
   */
  updateRecommendations(_stats: unknown): void {
    // Clear old global recommendations
    const toRemove: string[] = [];
    for (const [id, rec] of this.recommendations) {
      if (rec.category === 'performance' && id.startsWith('global-')) {
        toRemove.push(id);
      }
    }
    toRemove.forEach((id) => this.recommendations.delete(id));

    // Note: Cache hit rate analysis removed (L0 cache removed)
  }

  /**
   * Analyze memory metrics for GPU buffer pool and accumulators
   */
  analyzeMemoryMetrics(metrics: MemoryMetrics): void {
    // Analyze GPU buffer pool
    if (metrics.gpuPool) {
      const { allocations, reuses, byType } = metrics.gpuPool;
      const total = allocations + reuses;
      const reuseRate = total > 0 ? reuses / total : 1;

      // Check overall reuse rate
      if (total > 10 && reuseRate < 0.5) {
        this.addLowReuseRateRecommendation(reuseRate, 'overall');
      }

      // Check per-type reuse rates
      for (const type of POOLED_GEOMETRY_TYPES) {
        const typeStats = byType[type];
        const typeTotal = typeStats.allocations + typeStats.reuses;
        const typeReuseRate = typeTotal > 0 ? typeStats.reuses / typeTotal : 1;

        if (typeTotal > 5 && typeReuseRate < 0.2) {
          this.addLowReuseRateRecommendation(typeReuseRate, type);
        }
      }
    }

    // Analyze accumulators
    for (const type of POOLED_GEOMETRY_TYPES) {
      const stats = metrics.accumulators[type];
      if (stats && stats.growthEvents > 5) {
        this.addExcessiveGrowthRecommendation(type, stats.growthEvents);
      }
    }
  }

  /**
   * Add low GPU buffer reuse rate recommendation
   */
  private addLowReuseRateRecommendation(rate: number, type: string): void {
    const id = `low-gpu-reuse-${type}`;
    const percentage = (rate * 100).toFixed(0);
    const severity = rate < 0.2 ? 'error' : 'warning';

    const rec: Recommendation = {
      id,
      severity: severity as 'warning' | 'error',
      category: 'memory',
      title: `Low GPU Buffer Reuse (${type})`,
      message: `Only ${percentage}% buffer reuse for ${type}`,
      suggestion: 'Frequent allocations may cause frame drops. Check for buffer lifecycle issues.',
      metric: 'gpuReuseRate',
      value: rate,
      threshold: 0.5,
    };

    this.recommendations.set(rec.id, rec);
  }

  /**
   * Add excessive accumulator growth recommendation
   */
  private addExcessiveGrowthRecommendation(type: string, growthEvents: number): void {
    const id = `excessive-growth-${type}`;
    const severity = growthEvents > 10 ? 'error' : 'warning';

    const rec: Recommendation = {
      id,
      severity: severity as 'warning' | 'error',
      category: 'memory',
      title: `Excessive Accumulator Growth (${type})`,
      message: `${type} accumulator grew ${growthEvents} times`,
      suggestion: 'Initial capacity may be too small. Memory fragmentation likely.',
      metric: 'growthEvents',
      value: growthEvents,
      threshold: 5,
    };

    this.recommendations.set(rec.id, rec);
  }

  /**
   * Get all current recommendations
   */
  getRecommendations(): Recommendation[] {
    // Sort by severity
    const severityOrder = { error: 0, warning: 1, info: 2 };
    return Array.from(this.recommendations.values()).sort(
      (a, b) => severityOrder[a.severity] - severityOrder[b.severity]
    );
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
