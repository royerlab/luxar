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
  SceneGraphNode,
  SceneGraphState,
} from './data-monitor-types';
import { config } from '../config';

// Extract MonitorColors for dynamic color values (runtime-determined)
const MonitorColors = config.ui.styles.colors;

/**
 * Template for metric card component
 */
export function renderMetricCard(
  title: string,
  value: string | number,
  subtitle?: string,
  color: string = '',
  size: 'small' | 'medium' | 'large' = 'medium'
): string {
  const colorStyle = color ? `style="color: ${color}"` : '';

  return `
    <div class="luxar-metric-card luxar-metric-card--${size}">
      ${title ? `<div class="luxar-metric-card__title">${title}</div>` : ''}
      <div class="luxar-metric-card__value luxar-metric-card__value--${size}" ${colorStyle}>
        ${value}
      </div>
      ${subtitle ? `<div class="luxar-metric-card__subtitle">${subtitle}</div>` : ''}
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
  const trackStyle = `style="height: ${height}px; border-radius: ${height / 2}px;"`;
  const fillStyle = `style="background: ${barColor}; width: ${Math.min(100, percent)}%; border-radius: ${height / 2}px;"`;

  return `
    <div class="luxar-progress-bar__container">
      <div class="luxar-progress-bar__track" ${trackStyle}>
        <div class="luxar-progress-bar__fill" ${fillStyle}></div>
      </div>
      ${label ? `<div class="luxar-progress-bar__label">${label}</div>` : ''}
    </div>
  `;
}

/**
 * Template for stat grid component
 */
export function renderStatGrid(
  stats: Array<{ label: string; value: string | number; color?: string }>
): string {
  const cols = Math.min(3, stats.length);

  return `
    <div class="luxar-stat-grid luxar-stat-grid--cols-${cols}">
      ${stats
        .map(
          (stat) => {
            const colorStyle = stat.color ? `style="color: ${stat.color}"` : '';
            return `
        <div class="luxar-stat-grid__item">
          <div class="luxar-stat-grid__value" ${colorStyle}>
            ${stat.value}
          </div>
          <div class="luxar-stat-grid__label">${stat.label}</div>
        </div>
      `;
          }
        )
        .join('')}
    </div>
  `;
}

/**
 * Template for loader list item
 */
export function renderLoaderItem(path: string, metrics: LoaderMetrics): string {
  const statusColor = metrics.queries > 0 ? MonitorColors.success : MonitorColors.muted;

  return `
    <div class="luxar-loader-item">
      <div class="luxar-loader-item__header">
        <span class="luxar-loader-item__path" style="color: ${statusColor}">${path}</span>
        <span class="luxar-loader-item__status">${metrics.type}</span>
      </div>
      <div class="luxar-loader-item__metrics">
        <span>${metrics.visiblePoints.toLocaleString()} pts</span>
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
  network: { bytesTransferred: number; requestCount: number; bandwidth: number } | undefined
): string {
  const memoryPercent = memory.limit > 0 ? (memory.used / memory.limit) * 100 : 0;

  return `
    <div class="luxar-secondary-metrics">
      <div class="luxar-secondary-metrics__item">
        <span class="luxar-secondary-metrics__label">MEMORY</span>
        <div class="luxar-secondary-metrics__value">
          ${formatBytes(memory.used)}
        </div>
        ${renderProgressBar(memoryPercent, getCacheMemoryColor(memoryPercent), '', 2)}
      </div>

      <div class="luxar-secondary-metrics__item">
        <span class="luxar-secondary-metrics__label">QUERY SPEED</span>
        <div class="luxar-secondary-metrics__value">
          ${querySpeed.avgTime.toFixed(0)}ms
        </div>
        <div class="luxar-secondary-metrics__subtitle">
          ${querySpeed.perSec.toFixed(1)}/sec
        </div>
      </div>

      <div class="luxar-secondary-metrics__item">
        <span class="luxar-secondary-metrics__label">NETWORK I/O</span>
        <div class="luxar-secondary-metrics__value">
          ${network ? formatBytes(network.bytesTransferred) : '0B'}
        </div>
        <div class="luxar-secondary-metrics__subtitle">
          ${network ? `${network.requestCount} req · ${formatBytes(network.bandwidth)}/s` : '0 req'}
        </div>
      </div>
    </div>
  `;
}

/**
 * Template for overview tab content
 */
export function renderOverviewContent(stats: GlobalStats, cacheMetrics: CacheMetrics): string {
  // Calculate visible percentage of points dataset
  const visiblePointsPercent =
    stats.datasetSize > 0 ? ((stats.visiblePoints / stats.datasetSize) * 100).toFixed(1) : '0';

  // Calculate visible percentage of segments dataset
  const visibleSegmentsPercent =
    stats.datasetSegments > 0
      ? ((stats.visibleSegments / stats.datasetSegments) * 100).toFixed(1)
      : '0';

  // Determine what to show based on available data
  const hasPoints = stats.datasetSize > 0 || stats.visiblePoints > 0;
  const hasLines = stats.datasetSegments > 0 || stats.visibleSegments > 0;
  const showBoth = hasPoints && hasLines;

  // Build primary metrics section
  let primaryMetrics = '';
  if (showBoth) {
    // Show both side by side
    primaryMetrics = `
      <div class="luxar-overview-grid luxar-overview-grid--cols-2">
        ${renderMetricCard(
          'VISIBLE POINTS',
          formatNumber(stats.visiblePoints),
          `${visiblePointsPercent}% of ${formatNumber(stats.datasetSize)} total`,
          MonitorColors.success,
          'medium'
        )}
        ${renderMetricCard(
          'VISIBLE LINES',
          formatNumber(stats.visibleSegments),
          `${visibleSegmentsPercent}% of ${formatNumber(stats.datasetSegments)} total`,
          MonitorColors.warning,
          'medium'
        )}
      </div>
    `;
  } else if (hasPoints) {
    // Show only points (large)
    primaryMetrics = `
      <div class="luxar-overview-grid luxar-overview-grid--cols-1">
        ${renderMetricCard(
          'VISIBLE POINTS',
          formatNumber(stats.visiblePoints),
          `${visiblePointsPercent}% of ${formatNumber(stats.datasetSize)} total`,
          MonitorColors.success,
          'large'
        )}
      </div>
    `;
  } else if (hasLines) {
    // Show only lines (large)
    primaryMetrics = `
      <div class="luxar-overview-grid luxar-overview-grid--cols-1">
        ${renderMetricCard(
          'VISIBLE LINES',
          formatNumber(stats.visibleSegments),
          `${visibleSegmentsPercent}% of ${formatNumber(stats.datasetSegments)} total`,
          MonitorColors.warning,
          'large'
        )}
      </div>
    `;
  } else {
    // No data yet
    primaryMetrics = `
      <div class="luxar-overview-grid luxar-overview-grid--cols-1">
        ${renderMetricCard('LOADING', '...', 'Waiting for data', MonitorColors.muted, 'large')}
      </div>
    `;
  }

  return `
    <div class="overview-content">
      <!-- Primary metrics -->
      ${primaryMetrics}

      <!-- Secondary metrics -->
      ${renderSecondaryMetrics(
        { used: cacheMetrics.totalCacheMemory, limit: cacheMetrics.memoryLimit },
        { avgTime: stats.avgQueryTime, perSec: stats.queriesPerSecond },
        cacheMetrics.network
      )}

      <!-- Scene graph or loader list (injected by monitor) -->
      <div class="scene-or-loaders">
        <div id="loader-list-content"></div>
      </div>
    </div>
  `;
}

/**
 * Reusable cache section component (reduces duplication between L1/L2)
 */
function renderCacheSection(
  title: string,
  titleTooltip: string | undefined,
  clearAction: string,
  clearTooltip: string,
  metrics: Array<{
    label: string;
    value: string;
    subtitle: string;
    tooltip: string;
    color?: string;
  }>
): string {
  const cols = metrics.length === 2 ? 'cols-2' : 'cols-3';

  return `
    <div class="luxar-cache-section">
      <div class="luxar-cache-section__header">
        <span class="luxar-cache-section__title"${titleTooltip ? ` title="${titleTooltip}"` : ''}>${title}</span>
        <button data-action="${clearAction}" class="luxar-cache-section__clear-btn" title="${clearTooltip}">Clear</button>
      </div>
      <div class="luxar-cache-section__metrics luxar-cache-section__metrics--${cols}">
        ${metrics
          .map(
            (metric) => {
              const colorStyle = metric.color ? ` style="color: ${metric.color}"` : '';
              const sizeClass = metrics.length === 2 ? 'luxar-metric-card__value--medium' : 'luxar-metric-card__value--small';
              return `
          <div class="luxar-metric-card luxar-metric-card--small"${metric.tooltip ? ` title="${metric.tooltip}"` : ''}>
            <div class="luxar-metric-card__title">${metric.label}</div>
            <div class="luxar-metric-card__value ${sizeClass}"${colorStyle}>
              ${metric.value}
            </div>
            <div class="luxar-metric-card__subtitle">
              ${metric.subtitle}
            </div>
          </div>
        `;
            }
          )
          .join('')}
      </div>
    </div>
  `;
}

/**
 * Template for cache tab content with L1/L2 breakdown
 */
export function renderCacheContent(_stats: GlobalStats, cacheMetrics: CacheMetrics): string {
  // Check if caching is disabled
  if (cacheMetrics.enabled === false) {
    return `
      <div class="cache-content">
        <div class="luxar-data-monitor__empty">
          <div style="font-size: 32px; margin-bottom: 10px;">🚫</div>
          <div>Caching is disabled</div>
          <div class="luxar-metric-card__subtitle" style="margin-top: 8px;">
            Remove ?no-cache from URL to enable
          </div>
        </div>
      </div>
    `;
  }

  // Check if we have L1/L2 breakdown
  const hasL1L2 = cacheMetrics.l1 !== undefined && cacheMetrics.l2 !== undefined;

  if (!hasL1L2) {
    // Fallback to basic view if no cache stats provider connected
    return `
      <div class="cache-content">
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 15px;">
          ${renderMetricCard(
            'CACHE MEMORY',
            formatBytes(cacheMetrics.totalCacheMemory),
            `${cacheMetrics.memoryPercent.toFixed(0)}% of ${formatBytes(cacheMetrics.memoryLimit)}`,
            MonitorColors.success,
            'medium'
          )}
          ${renderMetricCard(
            'CACHED ENTRIES',
            cacheMetrics.totalEntries.toString(),
            '',
            MonitorColors.primaryText,
            'medium'
          )}
        </div>
        <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 4px; color: ${MonitorColors.muted}; font-size: 11px;">
          Loading cache statistics...
        </div>
      </div>
    `;
  }

  // L1 hit rate calculation
  const l1Total = cacheMetrics.l1!.hits + cacheMetrics.l1!.misses;
  const l1HitRate = l1Total > 0 ? (cacheMetrics.l1!.hits / l1Total) * 100 : 0;

  return `
    <div class="cache-content">
      <!-- L1 Memory Cache Section -->
      ${renderCacheSection('L1 MEMORY CACHE', undefined, 'clearL1', 'Clear L1 cache', [
        {
          label: 'SIZE',
          value: formatBytes(cacheMetrics.l1!.size),
          subtitle: `${cacheMetrics.l1!.count} entries`,
          tooltip: 'L1 memory cache: fast in-memory storage for recently accessed chunks',
          color: MonitorColors.success,
        },
        {
          label: 'HIT RATE',
          value: `${l1HitRate.toFixed(1)}%`,
          subtitle: `${formatNumber(cacheMetrics.l1!.hits)} hits · ${formatNumber(cacheMetrics.l1!.misses)} miss`,
          tooltip: `Cache hit rate: ${cacheMetrics.l1!.hits.toLocaleString()} hits out of ${l1Total.toLocaleString()} total accesses`,
          color:
            l1HitRate > 80
              ? MonitorColors.success
              : l1HitRate > 50
                ? MonitorColors.warning
                : MonitorColors.error,
        },
        {
          label: 'EVICTIONS',
          value: formatNumber(cacheMetrics.l1!.evictions),
          subtitle: 'LRU removed',
          tooltip:
            'Entries removed from cache when memory limit reached (LRU = Least Recently Used)',
          color: cacheMetrics.l1!.evictions > 0 ? MonitorColors.warning : MonitorColors.dimmed,
        },
      ])}

      <!-- L2 OPFS Cache Section -->
      ${renderCacheSection(
        'L2 OPFS CACHE',
        'Origin Private File System: persistent browser storage for cached data',
        'clearL2',
        'Clear L2 persistent cache (data will need to be re-downloaded)',
        [
          {
            label: 'SIZE',
            value: formatBytes(cacheMetrics.l2!.size),
            subtitle: `${cacheMetrics.l2!.count} entries`,
            tooltip:
              "L2 persistent cache: stored in browser's Origin Private File System, survives page reloads",
            color: MonitorColors.info,
          },
          {
            label: 'I/O',
            value: `${formatNumber(cacheMetrics.l2!.reads)} reads`,
            subtitle: `${formatNumber(cacheMetrics.l2!.writes)} writes`,
            tooltip: 'Disk I/O operations: reads from cache, writes to cache',
          },
        ]
      )}

      <!-- Combined Stats + Clear All -->
      <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 6px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <span style="font-size: 11px; color: ${MonitorColors.muted}; font-weight: 600;" title="Combined L1 + L2 cache usage">TOTAL</span>
          <button data-action="clearAll" class="luxar-cache-section__clear-btn" title="Clear both L1 and L2 caches (all cached data will be removed)">Clear All</button>
        </div>
        <div style="font-size: 14px; font-weight: bold; color: ${MonitorColors.primaryText}; margin-bottom: 6px;" title="${cacheMetrics.totalCacheMemory.toLocaleString()} bytes total cached">
          ${formatBytes(cacheMetrics.totalCacheMemory)}
        </div>
        ${renderProgressBar(cacheMetrics.memoryPercent, getCacheMemoryColor(cacheMetrics.memoryPercent), `${cacheMetrics.memoryPercent.toFixed(0)}% of ${formatBytes(cacheMetrics.memoryLimit)} limit`, 6)}
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

  return `
    <div class="luxar-recommendation luxar-recommendation--${rec.severity}">
      <div class="luxar-recommendation__header">
        <span class="luxar-recommendation__icon">${severityIcons[rec.severity]}</span>
        <strong class="luxar-recommendation__title">${rec.title}</strong>
      </div>
      <div class="luxar-recommendation__message">
        ${rec.message}
      </div>
      ${
        rec.suggestion
          ? `
        <div class="luxar-recommendation__message" style="margin-top: 4px;">
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
      <div class="luxar-data-monitor__empty" style="opacity: 0.5;">
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

function getCacheMemoryColor(percent: number): string {
  if (percent <= 60) return MonitorColors.success;
  if (percent <= 80) return MonitorColors.warning;
  return MonitorColors.error;
}

function getProgressColor(percent: number): string {
  if (percent <= 60) return MonitorColors.success;
  if (percent <= 80) return MonitorColors.warning;
  return MonitorColors.error;
}

// Scene graph tree rendering

/**
 * Get icon for scene graph node type
 */
function getNodeTypeIcon(type: string): string {
  const icons: Record<string, string> = {
    scene: '🌐',
    group: '📁',
    points: '⚬',
    lines: '╱',
    mesh: '⬡',
  };
  return icons[type] || '•';
}

/**
 * Get node type color
 */
function getNodeTypeColor(type: string): string {
  const colors: Record<string, string> = {
    scene: MonitorColors.info,
    group: MonitorColors.muted,
    points: MonitorColors.success,
    lines: MonitorColors.warning,
    mesh: '#9b59b6',
  };
  return colors[type] || MonitorColors.primaryText;
}

/**
 * Render a single scene graph tree node
 */
function renderSceneGraphNode(
  node: SceneGraphNode,
  expandedNodes: Set<string>,
  depth: number = 0
): string {
  const hasChildren = node.children.length > 0;
  const isExpanded = expandedNodes.has(node.path);
  const indent = depth * 16;

  // Node stats and tooltips
  let statsText = '';
  let statsTooltip = '';
  if (node.type === 'points' && node.pointCount !== undefined) {
    statsText = formatNumber(node.pointCount);
    statsTooltip = `${node.pointCount.toLocaleString()} points in this layer`;
  } else if (node.type === 'lines') {
    if (node.segmentCount !== undefined) {
      statsText = formatNumber(node.segmentCount);
      statsTooltip = `${node.segmentCount.toLocaleString()} line segments`;
      if (node.vertexCount !== undefined) {
        statsTooltip += `, ${node.vertexCount.toLocaleString()} vertices`;
      }
    }
  } else if (node.type === 'group' && hasChildren) {
    // For groups, show child count
    statsText = `${node.children.length}`;
    statsTooltip = `${node.children.length} child node${node.children.length !== 1 ? 's' : ''}`;
  }

  // Build tooltip for the whole node
  const nodeTypeDescriptions: Record<string, string> = {
    scene: 'Root scene node',
    group: 'Container for organizing nodes',
    points: 'Point cloud layer',
    lines: 'Line segments layer',
    mesh: 'Mesh geometry',
  };
  const nodeTooltip = `${nodeTypeDescriptions[node.type] || node.type}${node.hasSpatialIndex ? ' (indexed)' : ''}`;

  // Expand/collapse toggle
  const toggleIcon = hasChildren ? (isExpanded ? '▼' : '▶') : '•';
  const toggleStyle = hasChildren ? 'cursor: pointer; user-select: none;' : 'opacity: 0.3;';

  return `
    <div style="padding: 2px 0;">
      <div style="display: flex; align-items: center; padding-left: ${indent}px;" title="${nodeTooltip}">
        <!-- Toggle -->
        <span
          ${hasChildren ? `data-action="toggleNode" data-node-path="${node.path}"` : ''}
          style="width: 16px; font-size: 9px; color: ${MonitorColors.muted}; ${toggleStyle}"
          ${hasChildren ? `title="${isExpanded ? 'Collapse' : 'Expand'} ${node.name}"` : ''}
        >${toggleIcon}</span>

        <!-- Icon & Name -->
        <span style="font-size: 11px; margin-right: 4px;">${getNodeTypeIcon(node.type)}</span>
        <span style="font-size: 11px; color: ${getNodeTypeColor(node.type)}; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          ${node.name}
        </span>

        <!-- Stats badge -->
        ${
          statsText
            ? `<span style="font-size: 9px; color: ${MonitorColors.primaryText}; background: rgba(255,255,255,0.1); padding: 1px 5px; border-radius: 8px; margin-left: 4px;" title="${statsTooltip}">${statsText}</span>`
            : ''
        }

        <!-- Loading indicator -->
        ${node.isLoading ? '<span style="font-size: 9px; margin-left: 4px;" title="Loading data...">⏳</span>' : ''}
      </div>

      <!-- Children (if expanded) -->
      ${
        isExpanded && hasChildren
          ? node.children
              .map((child) => renderSceneGraphNode(child, expandedNodes, depth + 1))
              .join('')
          : ''
      }
    </div>
  `;
}

/**
 * Render scene graph tree component
 */
export function renderSceneGraphTree(state: SceneGraphState, expandedNodes: Set<string>): string {
  if (!state.root) {
    return `
      <div style="color: ${MonitorColors.dimmed}; font-size: 10px; padding: 8px;">
        No scene loaded
      </div>
    `;
  }

  // Header with stats
  const headerStats = [
    state.pointsNodes > 0 ? `${state.pointsNodes} points` : null,
    state.linesNodes > 0 ? `${state.linesNodes} lines` : null,
  ]
    .filter(Boolean)
    .join(', ');

  return `
    <div class="scene-graph-tree">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <h4 style="margin: 0; font-size: 11px; color: ${MonitorColors.muted};">SCENE GRAPH</h4>
        ${
          headerStats
            ? `<span style="font-size: 9px; color: ${MonitorColors.dimmed};">${headerStats}</span>`
            : ''
        }
      </div>
      <div style="max-height: 180px; overflow-y: auto; background: ${MonitorColors.sectionBg}; border-radius: 4px; padding: 4px;">
        ${renderSceneGraphNode(state.root, expandedNodes)}
      </div>
    </div>
  `;
}
