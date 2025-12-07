/**
 * Data Loading Monitor HTML Templates
 *
 * This module provides template functions for generating HTML content
 * in the Data Loading Monitor. It extracts the HTML generation logic
 * from the main monitor class to improve code organization and maintainability.
 */

import type {
  GlobalStats,
  LoaderMetrics,
  Recommendation,
  CacheMetrics,
} from './data-monitor-types';
import { config } from '../config';

// Extract MonitorColors and component config from config
const MonitorColors = config.ui.styles.colors;
const monitorConfig = config.ui.components.dataMonitor;
const spacingConfig = config.ui.styles.spacing;

/**
 * Template for metric card component
 */
export function renderMetricCard(
  title: string,
  value: string | number,
  subtitle?: string,
  color: string = MonitorColors.primaryText,
  size: 'small' | 'medium' | 'large' = 'medium'
): string {
  const fontSize = size === 'large' ? '32px' : size === 'medium' ? '24px' : '16px';

  return `
    <div style="background: ${MonitorColors.sectionBg}; padding: ${size === 'large' ? spacingConfig.panelPadding : spacingConfig.sectionPadding + 2}px; border-radius: ${size === 'large' ? monitorConfig.borderRadius.section : monitorConfig.borderRadius.card}px;">
      ${title ? `<div style="font-size: 10px; color: ${MonitorColors.muted}; margin-bottom: 4px;">${title}</div>` : ''}
      <div style="font-size: ${fontSize}; font-weight: bold; color: ${color};">
        ${value}
      </div>
      ${subtitle ? `<div style="font-size: 10px; color: ${MonitorColors.dimmed}; margin-top: 4px;">${subtitle}</div>` : ''}
    </div>
  `;
}

/**
 * Template for progress bar component
 */
export function renderProgressBar(
  percent: number,
  color?: string,
  label?: string,
  height: number = 4
): string {
  const barColor = color || getProgressColor(percent);

  return `
    <div style="margin-top: ${label ? 6 : 0}px;">
      <div style="height: ${height}px; background: rgba(255,255,255,0.1); border-radius: ${height / 2}px;">
        <div style="height: 100%; background: ${barColor}; width: ${Math.min(100, percent)}%; border-radius: ${height / 2}px;"></div>
      </div>
      ${label ? `<div style="font-size: 9px; color: ${MonitorColors.dimmed}; margin-top: 2px;">${label}</div>` : ''}
    </div>
  `;
}

/**
 * Template for stat grid component
 */
export function renderStatGrid(
  stats: Array<{ label: string; value: string | number; color?: string }>
): string {
  return `
    <div style="display: grid; grid-template-columns: repeat(${Math.min(3, stats.length)}, 1fr); gap: 8px;">
      ${stats
        .map(
          (stat) => `
        <div style="background: ${MonitorColors.sectionBg}; padding: ${monitorConfig.padding.compact}px; border-radius: ${monitorConfig.borderRadius.card}px; text-align: center;">
          <div style="font-size: 16px; font-weight: bold; color: ${stat.color || MonitorColors.info};">
            ${stat.value}
          </div>
          <div style="font-size: 9px; color: ${MonitorColors.muted};">${stat.label}</div>
        </div>
      `
        )
        .join('')}
    </div>
  `;
}

/**
 * Template for loader list item
 */
export function renderLoaderItem(path: string, metrics: LoaderMetrics): string {
  const hitRate =
    metrics.cacheHits + metrics.cacheMisses > 0
      ? (metrics.cacheHits / (metrics.cacheHits + metrics.cacheMisses)) * 100
      : 0;
  const statusColor = metrics.queries > 0 ? MonitorColors.success : MonitorColors.muted;

  return `
    <div style="background: ${MonitorColors.sectionBg}; padding: ${monitorConfig.padding.compact}px; margin-bottom: ${spacingConfig.borderPadding}px; border-radius: ${monitorConfig.borderRadius.card}px;">
      <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span style="font-size: 10px; color: ${statusColor}; font-family: monospace;">${path}</span>
        <span style="font-size: 9px; color: ${MonitorColors.muted};">${metrics.type}</span>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 9px; color: ${MonitorColors.dimmed};">
        <span>${metrics.visiblePoints.toLocaleString()} pts</span>
        <span>${hitRate.toFixed(0)}% cache</span>
        <span>${formatBytes(metrics.memoryUsed)}</span>
      </div>
    </div>
  `;
}

/**
 * Template for secondary metrics bar
 */
export function renderSecondaryMetrics(
  memory: { used: number; limit: number },
  querySpeed: { avgTime: number; perSec: number },
  loadSpeed: { count: number; bandwidth: number }
): string {
  const memoryPercent = memory.limit > 0 ? (memory.used / memory.limit) * 100 : 0;

  return `
    <div style="display: flex; gap: ${spacingConfig.panelMargin}px; padding: ${monitorConfig.padding.default}px; background: ${MonitorColors.sectionBg}; border-radius: ${monitorConfig.borderRadius.card}px; margin-bottom: ${spacingConfig.sectionGap}px;">
      <div style="flex: 1;">
        <span style="color: ${MonitorColors.muted}; font-size: 10px;">MEMORY</span>
        <div style="color: ${MonitorColors.primaryText}; font-size: 14px; font-weight: 600;">
          ${formatBytes(memory.used)}
        </div>
        ${renderProgressBar(memoryPercent, getCacheMemoryColor(memoryPercent), '', 2)}
      </div>
      
      <div style="flex: 1;">
        <span style="color: ${MonitorColors.muted}; font-size: 10px;">QUERY SPEED</span>
        <div style="color: ${MonitorColors.primaryText}; font-size: 14px; font-weight: 600;">
          ${querySpeed.avgTime.toFixed(0)}ms
        </div>
        <div style="color: ${MonitorColors.dimmed}; font-size: 10px;">
          ${querySpeed.perSec.toFixed(1)}/sec
        </div>
      </div>
      
      <div style="flex: 1;">
        <span style="color: ${MonitorColors.muted}; font-size: 10px;">LOAD SPEED</span>
        <div style="color: ${MonitorColors.primaryText}; font-size: 14px; font-weight: 600;">
          ${loadSpeed.count} loads
        </div>
        <div style="color: ${MonitorColors.dimmed}; font-size: 10px;">
          ${formatBytes(loadSpeed.bandwidth)}/s
        </div>
      </div>
    </div>
  `;
}

/**
 * Template for overview tab content
 */
export function renderOverviewContent(stats: GlobalStats, cacheMetrics: CacheMetrics): string {
  // Calculate visible percentage of dataset
  const visiblePercent =
    stats.datasetSize > 0 ? ((stats.visiblePoints / stats.datasetSize) * 100).toFixed(1) : '0';

  return `
    <div class="overview-content">
      <!-- Primary metrics -->
      <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
        ${renderMetricCard(
          'VISIBLE POINTS',
          formatNumber(stats.visiblePoints),
          `${visiblePercent}% of ${formatNumber(stats.datasetSize)} total`,
          MonitorColors.success,
          'large'
        )}
        ${renderMetricCard(
          'CACHE HIT RATE',
          `${stats.globalCacheHitRate.toFixed(0)}%`,
          `${stats.totalCacheHits}/${stats.totalCacheHits + (stats.totalQueries - stats.totalCacheHits)} hits`,
          getCacheRateColor(stats.globalCacheHitRate),
          'large'
        )}
      </div>
      
      <!-- Secondary metrics -->
      ${renderSecondaryMetrics(
        { used: stats.totalMemory, limit: cacheMetrics.memoryLimit },
        { avgTime: stats.avgQueryTime, perSec: stats.queriesPerSecond },
        { count: stats.totalLoads, bandwidth: stats.totalMemoryUsed }
      )}
      
      <!-- Loader list -->
      <div class="loader-list">
        <h4 style="margin: 0 0 8px 0; font-size: 11px; color: ${MonitorColors.muted};">ACTIVE LOADERS</h4>
        <div style="max-height: 150px; overflow-y: auto;">
          <div id="loader-list-content"></div>
        </div>
      </div>
    </div>
  `;
}

/**
 * Template for cache tab content
 */
export function renderCacheContent(stats: GlobalStats, cacheMetrics: CacheMetrics): string {
  return `
    <div class="cache-content">
      <!-- Cache overview cards -->
      <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 15px;">
        ${renderMetricCard(
          'CACHE MEMORY',
          formatBytes(cacheMetrics.totalCacheMemory),
          `${cacheMetrics.memoryPercent.toFixed(0)}% of ${formatBytes(cacheMetrics.memoryLimit)}`,
          MonitorColors.success,
          'medium'
        )}
        ${renderMetricCard(
          'HIT RATE',
          `${stats.globalCacheHitRate.toFixed(1)}%`,
          `${cacheMetrics.recentHitRate.toFixed(0)}% recent (1m) | ${stats.totalCacheHits} hits / ${cacheMetrics.totalAccesses} total`,
          getCacheRateColor(stats.globalCacheHitRate),
          'medium'
        )}
        ${renderMetricCard(
          'CACHED RANGES',
          cacheMetrics.totalEntries.toString(),
          `${cacheMetrics.evictionsPerMin.toFixed(0)} evict/min`,
          MonitorColors.primaryText,
          'medium'
        )}
        ${renderMetricCard(
          'AVG RANGE SIZE',
          formatBytes(cacheMetrics.avgEntrySize),
          `Reuse: ${cacheMetrics.reuseRatio.toFixed(1)}x`,
          MonitorColors.primaryText,
          'medium'
        )}
      </div>
      
      <!-- Cache performance metrics -->
      <div style="margin-bottom: 15px;">
        <h4 style="margin: 0 0 8px 0; font-size: 11px; color: ${MonitorColors.muted};">CACHE PERFORMANCE</h4>
        ${renderStatGrid([
          {
            label: 'Hits/sec',
            value: `${cacheMetrics.hitsPerSecond.toFixed(1)}/s`,
            color: MonitorColors.info,
          },
          {
            label: 'Misses/sec',
            value: `${cacheMetrics.missesPerSecond.toFixed(1)}/s`,
            color: MonitorColors.warning,
          },
          {
            label: 'Avg Access',
            value: formatAccessTime(cacheMetrics.avgAccessTime),
            color: getAccessTimeColor(cacheMetrics.avgAccessTime),
          },
        ])}
      </div>
    </div>
  `;
}

/**
 * Template for recommendation item
 */
export function renderRecommendation(rec: Recommendation): string {
  const severityIcons = {
    error: '🔴',
    warning: '🟡',
    info: 'ℹ️',
  };

  const severityColors = {
    error: MonitorColors.error,
    warning: MonitorColors.warning,
    info: MonitorColors.info,
  };

  return `
    <div style="background: rgba(255,255,255,0.05); padding: ${monitorConfig.padding.default}px; margin-bottom: ${spacingConfig.elementGap}px; border-radius: ${monitorConfig.borderRadius.card}px; border-left: 3px solid ${severityColors[rec.severity]};">
      <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
        ${severityIcons[rec.severity]}
        <strong style="font-size: 11px;">${rec.title}</strong>
      </div>
      <div style="font-size: 10px; color: ${MonitorColors.primaryText}; opacity: 0.9;">
        ${rec.message}
      </div>
      ${
        rec.suggestion
          ? `
        <div style="font-size: 10px; color: ${MonitorColors.muted}; margin-top: 4px;">
          💡 ${rec.suggestion}
        </div>
      `
          : ''
      }
    </div>
  `;
}

/**
 * Template for insights tab content
 */
export function renderInsightsContent(recommendations: Recommendation[]): string {
  if (recommendations.length === 0) {
    return `
      <div style="text-align: center; padding: ${spacingConfig.panelMargin}px; opacity: 0.5;">
        ✅ No issues detected
      </div>
    `;
  }

  return `
    <div class="insights-content">
      ${recommendations.map((rec) => renderRecommendation(rec)).join('')}
    </div>
  `;
}

// Helper functions

function formatNumber(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + 'GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + 'MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + 'KB';
  return bytes.toFixed(0) + 'B';
}

function formatAccessTime(timeMs: number): string {
  if (timeMs < 1) {
    return (timeMs * 1000).toFixed(0) + 'μs';
  }
  return timeMs.toFixed(2) + 'ms';
}

function getCacheRateColor(rate: number): string {
  if (rate >= 80) return MonitorColors.success;
  if (rate >= 60) return MonitorColors.warning;
  return MonitorColors.error;
}

function getCacheMemoryColor(percent: number): string {
  if (percent <= 60) return MonitorColors.success;
  if (percent <= 80) return MonitorColors.warning;
  return MonitorColors.error;
}

function getAccessTimeColor(timeMs: number): string {
  if (timeMs < 0.1) return MonitorColors.success;
  if (timeMs < 1.0) return MonitorColors.info;
  if (timeMs < 5.0) return MonitorColors.warning;
  return MonitorColors.error;
}

function getProgressColor(percent: number): string {
  if (percent <= 60) return MonitorColors.success;
  if (percent <= 80) return MonitorColors.warning;
  return MonitorColors.error;
}
