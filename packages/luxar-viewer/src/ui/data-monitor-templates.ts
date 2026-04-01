/**
 * Data Loading Monitor HTML Templates
 *
 * This module provides template functions for generating HTML content
 * in the Data Loading Monitor. It extracts the HTML generation logic
 * from the main monitor class to improve code organization and maintainability.
 *
 * STYLING: All styles are now in CSS classes (data-loading-monitor.css).
 * Dynamic colors use CSS modifier classes (luxar-color--success, etc.)
 */

import type {
  GlobalStats,
  LoaderMetrics,
  Recommendation,
  CacheMetrics,
  SceneGraphNode,
  SceneGraphState,
} from './data-monitor-types';
import { escapeHtml } from '../utils/escape-html';

/**
 * Semantic color names mapped to CSS class modifiers.
 * These are used with the luxar-color--{name} classes.
 */
type SemanticColor = 'success' | 'warning' | 'error' | 'info' | 'muted' | 'dimmed' | 'primary';

/**
 * Get CSS class for a semantic color.
 */
export function getColorClass(color: SemanticColor): string {
  return `luxar-color--${color}`;
}

/**
 * Template for metric card component
 * @param colorClass - CSS class for color (e.g., 'luxar-color--success')
 * @param dataField - Optional data-field attribute for targeted DOM patching
 */
export function renderMetricCard(
  title: string,
  value: string | number,
  subtitle?: string,
  colorClass: string = '',
  size: 'small' | 'medium' | 'large' = 'medium',
  dataField?: string
): string {
  const fieldAttr = dataField ? ` data-field="${dataField}"` : '';
  const subFieldAttr = dataField ? ` data-field="${dataField}-sub"` : '';
  return `
    <div class="luxar-metric-card luxar-metric-card--${size}">
      ${title ? `<div class="luxar-metric-card__title">${title}</div>` : ''}
      <div class="luxar-metric-card__value luxar-metric-card__value--${size} ${colorClass}"${fieldAttr}>
        ${value}
      </div>
      ${subtitle ? `<div class="luxar-metric-card__subtitle"${subFieldAttr}>${subtitle}</div>` : ''}
    </div>
  `;
}

/**
 * Template for progress bar component
 * Note: Uses CSS custom properties for dynamic sizing
 * @param colorClass - CSS color class (e.g., 'luxar-color--success')
 */
export function renderProgressBar(
  percent: number,
  colorClass?: string,
  label?: string,
  height: number = 4
): string {
  const barColorClass = colorClass || getProgressColorClass(percent);
  // Using CSS custom properties for dynamic values that can't be pure CSS
  const trackStyle = `style="--bar-height: ${height}px; height: var(--bar-height); border-radius: calc(var(--bar-height) / 2);"`;
  const fillStyle = `style="width: ${Math.min(100, percent)}%; border-radius: calc(var(--bar-height, 4px) / 2);"`;

  return `
    <div class="luxar-progress-bar__container">
      <div class="luxar-progress-bar__track" ${trackStyle}>
        <div class="luxar-progress-bar__fill ${barColorClass}" ${fillStyle}></div>
      </div>
      ${label ? `<div class="luxar-progress-bar__label">${label}</div>` : ''}
    </div>
  `;
}

/**
 * Template for stat grid component
 * @param stats - Array of stats with colorClass for CSS class-based coloring
 */
export function renderStatGrid(
  stats: Array<{ label: string; value: string | number; colorClass?: string }>
): string {
  const cols = Math.min(3, stats.length);

  return `
    <div class="luxar-stat-grid luxar-stat-grid--cols-${cols}">
      ${stats
        .map((stat) => {
          return `
        <div class="luxar-stat-grid__item">
          <div class="luxar-stat-grid__value ${stat.colorClass || ''}">
            ${stat.value}
          </div>
          <div class="luxar-stat-grid__label">${stat.label}</div>
        </div>
      `;
        })
        .join('')}
    </div>
  `;
}

/**
 * Template for loader list item
 */
export function renderLoaderItem(path: string, metrics: LoaderMetrics): string {
  const statusColorClass = metrics.queries > 0 ? getColorClass('success') : getColorClass('muted');

  return `
    <div class="luxar-loader-item">
      <div class="luxar-loader-item__header">
        <span class="luxar-loader-item__path ${statusColorClass}">${escapeHtml(path)}</span>
        <span class="luxar-loader-item__status">${escapeHtml(metrics.type)}</span>
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
        <div class="luxar-secondary-metrics__value" data-field="memory-used">
          ${formatBytes(memory.used)}
        </div>
        ${renderProgressBar(memoryPercent, getCacheMemoryColorClass(memoryPercent), '', 2)}
      </div>

      <div class="luxar-secondary-metrics__item">
        <span class="luxar-secondary-metrics__label">QUERY SPEED</span>
        <div class="luxar-secondary-metrics__value" data-field="query-speed">
          ${querySpeed.avgTime.toFixed(0)}ms
        </div>
        <div class="luxar-secondary-metrics__subtitle" data-field="query-rate">
          ${querySpeed.perSec.toFixed(1)}/sec
        </div>
      </div>

      <div class="luxar-secondary-metrics__item">
        <span class="luxar-secondary-metrics__label">NETWORK I/O</span>
        <div class="luxar-secondary-metrics__value" data-field="network-bytes">
          ${network ? formatBytes(network.bytesTransferred) : '0B'}
        </div>
        <div class="luxar-secondary-metrics__subtitle" data-field="network-detail">
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
  const hasGSplats = stats.datasetSplats > 0 || stats.visibleSplats > 0;

  // Count how many data types we have
  const dataTypes = [hasPoints, hasLines, hasGSplats].filter(Boolean).length;
  const showBoth = dataTypes === 2;
  const showAll = dataTypes === 3;

  // Calculate visible percentage for gsplats
  const visibleSplatsPercent =
    stats.datasetSplats > 0 ? ((stats.visibleSplats / stats.datasetSplats) * 100).toFixed(1) : '0';

  // Build primary metrics section
  let primaryMetrics = '';
  if (showAll) {
    // Show all three (points, lines, gsplats)
    primaryMetrics = `
      <div class="luxar-overview-grid luxar-overview-grid--cols-3">
        ${renderMetricCard(
          'VISIBLE POINTS',
          formatNumber(stats.visiblePoints),
          `${visiblePointsPercent}% of ${formatNumber(stats.datasetSize)}`,
          getColorClass('success'),
          'small',
          'visible-points'
        )}
        ${renderMetricCard(
          'VISIBLE LINES',
          formatNumber(stats.visibleSegments),
          `${visibleSegmentsPercent}% of ${formatNumber(stats.datasetSegments)}`,
          getColorClass('warning'),
          'small',
          'visible-lines'
        )}
        ${renderMetricCard(
          'VISIBLE SPLATS',
          formatNumber(stats.visibleSplats),
          `${visibleSplatsPercent}% of ${formatNumber(stats.datasetSplats)}`,
          getColorClass('info'),
          'small',
          'visible-splats'
        )}
      </div>
    `;
  } else if (showBoth) {
    // Show two data types
    const cards = [];
    if (hasPoints) {
      cards.push(
        renderMetricCard(
          'VISIBLE POINTS',
          formatNumber(stats.visiblePoints),
          `${visiblePointsPercent}% of ${formatNumber(stats.datasetSize)}`,
          getColorClass('success'),
          'medium',
          'visible-points'
        )
      );
    }
    if (hasLines) {
      cards.push(
        renderMetricCard(
          'VISIBLE LINES',
          formatNumber(stats.visibleSegments),
          `${visibleSegmentsPercent}% of ${formatNumber(stats.datasetSegments)}`,
          getColorClass('warning'),
          'medium',
          'visible-lines'
        )
      );
    }
    if (hasGSplats) {
      cards.push(
        renderMetricCard(
          'VISIBLE SPLATS',
          formatNumber(stats.visibleSplats),
          `${visibleSplatsPercent}% of ${formatNumber(stats.datasetSplats)}`,
          getColorClass('info'),
          'medium',
          'visible-splats'
        )
      );
    }
    primaryMetrics = `
      <div class="luxar-overview-grid luxar-overview-grid--cols-2">
        ${cards.join('\n')}
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
          getColorClass('success'),
          'large',
          'visible-points'
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
          getColorClass('warning'),
          'large',
          'visible-lines'
        )}
      </div>
    `;
  } else if (hasGSplats) {
    // Show only gsplats (large)
    primaryMetrics = `
      <div class="luxar-overview-grid luxar-overview-grid--cols-1">
        ${renderMetricCard(
          'VISIBLE SPLATS',
          formatNumber(stats.visibleSplats),
          `${visibleSplatsPercent}% of ${formatNumber(stats.datasetSplats)} total`,
          getColorClass('info'),
          'large',
          'visible-splats'
        )}
      </div>
    `;
  } else {
    // No data yet
    primaryMetrics = `
      <div class="luxar-overview-grid luxar-overview-grid--cols-1">
        ${renderMetricCard('LOADING', '...', 'Waiting for data', getColorClass('muted'), 'large')}
      </div>
    `;
  }

  return `
    <div class="luxar-tab-content--overview">
      <!-- Primary metrics -->
      ${primaryMetrics}

      <!-- Secondary metrics -->
      ${renderSecondaryMetrics(
        { used: cacheMetrics.totalCacheMemory, limit: cacheMetrics.memoryLimit },
        { avgTime: stats.avgQueryTime, perSec: stats.queriesPerSecond },
        cacheMetrics.network
      )}

      <!-- Scene graph or loader list (injected by monitor) -->
      <div class="luxar-overview__scene-loaders">
        <div id="loader-list-content"></div>
      </div>
    </div>
  `;
}

/**
 * Reusable cache section component (reduces duplication between L1/L2)
 * @param metrics - Each metric can have a colorClass for CSS class-based coloring
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
    colorClass?: string;
    dataField?: string;
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
          .map((metric) => {
            const sizeClass =
              metrics.length === 2
                ? 'luxar-metric-card__value--medium'
                : 'luxar-metric-card__value--small';
            const fieldAttr = metric.dataField ? ` data-field="${metric.dataField}"` : '';
            const subFieldAttr = metric.dataField ? ` data-field="${metric.dataField}-sub"` : '';
            return `
          <div class="luxar-metric-card luxar-metric-card--small"${metric.tooltip ? ` title="${metric.tooltip}"` : ''}>
            <div class="luxar-metric-card__title">${metric.label}</div>
            <div class="luxar-metric-card__value ${sizeClass} ${metric.colorClass || ''}"${fieldAttr}>
              ${metric.value}
            </div>
            <div class="luxar-metric-card__subtitle"${subFieldAttr}>
              ${metric.subtitle}
            </div>
          </div>
        `;
          })
          .join('')}
      </div>
    </div>
  `;
}

/**
 * Template for cache tab content with L0/L1/L2 breakdown
 */
export function renderCacheContent(_stats: GlobalStats, cacheMetrics: CacheMetrics): string {
  // Check if caching is disabled
  if (cacheMetrics.enabled === false) {
    return `
      <div class="luxar-tab-content--cache">
        <div class="luxar-cache-disabled">
          <div class="luxar-cache-disabled__icon">🚫</div>
          <div class="luxar-cache-disabled__message">Caching is disabled</div>
          <div class="luxar-cache-disabled__hint">
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
      <div class="luxar-tab-content--cache">
        <div class="luxar-grid-2">
          ${renderMetricCard(
            'CACHE MEMORY',
            formatBytes(cacheMetrics.totalCacheMemory),
            `${cacheMetrics.memoryPercent.toFixed(0)}% of ${formatBytes(cacheMetrics.memoryLimit)}`,
            getColorClass('success'),
            'medium'
          )}
          ${renderMetricCard(
            'CACHED ENTRIES',
            cacheMetrics.totalEntries.toString(),
            '',
            getColorClass('primary'),
            'medium'
          )}
        </div>
        <div class="luxar-cache-loading">
          Loading cache statistics...
        </div>
      </div>
    `;
  }

  // L0 hit rate calculation (if available)
  const l0Total = cacheMetrics.l0 ? cacheMetrics.l0.hits + cacheMetrics.l0.misses : 0;
  const l0HitRate = l0Total > 0 ? (cacheMetrics.l0!.hits / l0Total) * 100 : 0;
  const l0HitRateColorClass =
    l0HitRate > 80
      ? getColorClass('success')
      : l0HitRate > 50
        ? getColorClass('warning')
        : getColorClass('error');

  // L1 hit rate calculation
  const l1Total = cacheMetrics.l1!.hits + cacheMetrics.l1!.misses;
  const l1HitRate = l1Total > 0 ? (cacheMetrics.l1!.hits / l1Total) * 100 : 0;

  // Determine hit rate color class
  const l1HitRateColorClass =
    l1HitRate > 80
      ? getColorClass('success')
      : l1HitRate > 50
        ? getColorClass('warning')
        : getColorClass('error');

  return `
    <div class="luxar-tab-content--cache">
      <!-- L0 Decompressed Chunk Cache Section (fastest layer - avoids Blosc decompression) -->
      ${
        cacheMetrics.l0
          ? renderCacheSection(
              'L0 DECOMPRESSED CACHE',
              'Caches decoded zarr chunks to avoid ~2ms Blosc decompression overhead',
              'clearL0',
              'Clear L0 decompressed chunk cache',
              [
                {
                  label: 'SIZE',
                  value: formatBytes(cacheMetrics.l0.size),
                  subtitle: `${cacheMetrics.l0.count} chunks`,
                  tooltip:
                    'L0 decompressed cache: stores already-decoded TypedArrays, eliminating decompression latency',
                  colorClass: getColorClass('primary'),
                  dataField: 'l0-size',
                },
                {
                  label: 'HIT RATE',
                  value: `${l0HitRate.toFixed(1)}%`,
                  subtitle: `${formatNumber(cacheMetrics.l0.hits)} hits · ${formatNumber(cacheMetrics.l0.misses)} miss`,
                  tooltip:
                    'L0 hit: ~1μs lookup (no decompression). Miss: ~2ms decompression + L1 lookup',
                  colorClass: l0HitRateColorClass,
                  dataField: 'l0-hitrate',
                },
                {
                  label: 'EVICTIONS',
                  value: formatNumber(cacheMetrics.l0.evictions),
                  subtitle: 'LRU removed',
                  tooltip: 'Chunks removed from L0 cache when memory limit reached',
                  colorClass:
                    cacheMetrics.l0.evictions > 0
                      ? getColorClass('warning')
                      : getColorClass('dimmed'),
                  dataField: 'l0-evictions',
                },
              ]
            )
          : ''
      }

      <!-- L1 Memory Cache Section -->
      ${renderCacheSection('L1 MEMORY CACHE', undefined, 'clearL1', 'Clear L1 cache', [
        {
          label: 'SIZE',
          value: formatBytes(cacheMetrics.l1!.size),
          subtitle: `${cacheMetrics.l1!.count} entries`,
          tooltip: 'L1 memory cache: fast in-memory storage for recently accessed chunks',
          colorClass: getColorClass('success'),
          dataField: 'l1-size',
        },
        {
          label: 'HIT RATE',
          value: `${l1HitRate.toFixed(1)}%`,
          subtitle: `${formatNumber(cacheMetrics.l1!.hits)} hits · ${formatNumber(cacheMetrics.l1!.misses)} miss`,
          tooltip: `Cache hit rate: ${cacheMetrics.l1!.hits.toLocaleString()} hits out of ${l1Total.toLocaleString()} total accesses`,
          colorClass: l1HitRateColorClass,
          dataField: 'l1-hitrate',
        },
        {
          label: 'EVICTIONS',
          value: formatNumber(cacheMetrics.l1!.evictions),
          subtitle: 'LRU removed',
          tooltip:
            'Entries removed from cache when memory limit reached (LRU = Least Recently Used)',
          colorClass:
            cacheMetrics.l1!.evictions > 0 ? getColorClass('warning') : getColorClass('dimmed'),
          dataField: 'l1-evictions',
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
            colorClass: getColorClass('info'),
            dataField: 'l2-size',
          },
          {
            label: 'I/O',
            value: `${formatNumber(cacheMetrics.l2!.reads)} reads`,
            subtitle: `${formatNumber(cacheMetrics.l2!.writes)} writes`,
            tooltip: 'Disk I/O operations: reads from cache, writes to cache',
            dataField: 'l2-io',
          },
        ]
      )}

      <!-- Combined Stats + Clear All -->
      <div class="luxar-cache-total">
        <div class="luxar-cache-total__header">
          <span class="luxar-cache-total__label" title="Combined L0 + L1 + L2 cache usage">TOTAL</span>
          <button data-action="clearAll" class="luxar-cache-section__clear-btn" title="Clear all caches (L0 + L1 + L2)">Clear All</button>
        </div>
        <div class="luxar-cache-total__value" data-field="cache-total" title="${cacheMetrics.totalCacheMemory.toLocaleString()} bytes total cached">
          ${formatBytes(cacheMetrics.totalCacheMemory)}
        </div>
        ${renderProgressBar(cacheMetrics.memoryPercent, getCacheMemoryColorClass(cacheMetrics.memoryPercent), `${cacheMetrics.memoryPercent.toFixed(0)}% of ${formatBytes(cacheMetrics.memoryLimit)} limit`, 6)}
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
        <strong class="luxar-recommendation__title">${escapeHtml(rec.title)}</strong>
      </div>
      <div class="luxar-recommendation__message">
        ${escapeHtml(rec.message)}
      </div>
      ${
        rec.suggestion
          ? `
        <div class="luxar-recommendation__message luxar-suggestion">
          💡 ${escapeHtml(rec.suggestion)}
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
      <div class="luxar-data-monitor__empty luxar-data-monitor__empty--faded">
        ✅ No issues detected
      </div>
    `;
  }

  return `
    <div class="luxar-tab-content--insights">
      ${recommendations.map((rec) => renderRecommendation(rec)).join('')}
    </div>
  `;
}

/**
 * Memory metrics for GPU buffer pool (per-type)
 */
export interface GPUPoolTypeStats {
  allocations: number;
  reuses: number;
  evictions: number;
  activeBuffers: number;
  pooledBuffers: number;
}

/**
 * Memory metrics for GPU buffer pool
 */
export interface GPUPoolStats {
  allocations: number;
  reuses: number;
  evictions: number;
  capacityGrowths: number;
  activeBuffers: number;
  pooledBuffers: number;
  byType: {
    points: GPUPoolTypeStats;
    lines: GPUPoolTypeStats;
    gsplats: GPUPoolTypeStats;
  };
}

/**
 * Memory metrics for data accumulators
 */
export interface AccumulatorStats {
  capacity: number;
  allocations: number;
  growthEvents: number;
  memoryMB: number;
}

/**
 * Combined memory metrics for the Memory tab
 */
export interface MemoryMetrics {
  gpuPool: GPUPoolStats | null;
  accumulators: {
    points: AccumulatorStats | null;
    lines: AccumulatorStats | null;
    gsplats: AccumulatorStats | null;
  };
}

/**
 * Helper to calculate reuse rate percentage
 */
export function calculateReuseRate(allocations: number, reuses: number): number {
  const total = allocations + reuses;
  return total > 0 ? (reuses / total) * 100 : 0;
}

/**
 * Get color class for reuse rate
 */
export function getReuseRateColorClass(rate: number): string {
  if (rate >= 80) return getColorClass('success');
  if (rate >= 50) return getColorClass('warning');
  return getColorClass('error');
}

/**
 * Template for memory tab content with GPU buffer pool and accumulators
 */
export function renderMemoryContent(metrics: MemoryMetrics): string {
  const { gpuPool, accumulators } = metrics;

  // Calculate total memory from accumulators
  const totalAccumulatorMemory =
    (accumulators.points?.memoryMB ?? 0) +
    (accumulators.lines?.memoryMB ?? 0) +
    (accumulators.gsplats?.memoryMB ?? 0);

  // GPU Pool section
  const gpuPoolSection = gpuPool
    ? renderGPUPoolSection(gpuPool)
    : `
    <div class="luxar-memory-section">
      <div class="luxar-memory-section__header">
        <span class="luxar-memory-section__icon">⬡</span>
        <span class="luxar-memory-section__title">GPU BUFFER POOL</span>
      </div>
      <div class="luxar-memory-section__empty">Not initialized</div>
    </div>
  `;

  // Accumulators section
  const accumulatorsSection = renderAccumulatorsSection(accumulators);

  // Total summary
  const totalAllocations = gpuPool ? gpuPool.allocations : 0;
  const totalReuses = gpuPool ? gpuPool.reuses : 0;
  const overallReuseRate = calculateReuseRate(totalAllocations, totalReuses);

  return `
    <div class="luxar-tab-content--memory">
      ${gpuPoolSection}
      ${accumulatorsSection}
      <div class="luxar-memory-total">
        <span class="luxar-memory-total__label">Total:</span>
        <span class="luxar-memory-total__value" data-field="memory-total">
          ${totalAllocations} allocs · ${overallReuseRate.toFixed(0)}% reuse · ${totalAccumulatorMemory.toFixed(1)}MB
        </span>
      </div>
    </div>
  `;
}

/**
 * Render GPU buffer pool section with per-type table
 */
function renderGPUPoolSection(stats: GPUPoolStats): string {
  const types = ['points', 'lines', 'gsplats'] as const;

  const rows = types
    .map((type) => {
      const typeStats = stats.byType[type];
      const reuseRate = calculateReuseRate(typeStats.allocations, typeStats.reuses);
      const hasData =
        typeStats.allocations > 0 || typeStats.reuses > 0 || typeStats.activeBuffers > 0;
      const reuseColorClass = hasData ? getReuseRateColorClass(reuseRate) : getColorClass('dimmed');

      return `
      <tr class="luxar-memory-table__row" data-row="gpu-${type}">
        <td class="luxar-memory-table__cell luxar-memory-table__cell--type">${capitalize(type)}</td>
        <td class="luxar-memory-table__cell luxar-memory-table__cell--value ${reuseColorClass}" data-field="gpu-${type}-reuse">
          ${hasData ? `${reuseRate.toFixed(0)}%` : '—'}
        </td>
        <td class="luxar-memory-table__cell luxar-memory-table__cell--value" data-field="gpu-${type}-active">
          ${hasData ? typeStats.activeBuffers : '—'}
        </td>
        <td class="luxar-memory-table__cell luxar-memory-table__cell--value" data-field="gpu-${type}-pooled">
          ${hasData ? typeStats.pooledBuffers : '—'}
        </td>
        <td class="luxar-memory-table__cell luxar-memory-table__cell--value" data-field="gpu-${type}-allocs">
          ${hasData ? typeStats.allocations : '—'}
        </td>
      </tr>
    `;
    })
    .join('');

  return `
    <div class="luxar-memory-section">
      <div class="luxar-memory-section__header">
        <span class="luxar-memory-section__icon">⬡</span>
        <span class="luxar-memory-section__title">GPU BUFFER POOL</span>
      </div>
      <table class="luxar-memory-table">
        <thead>
          <tr class="luxar-memory-table__header-row">
            <th class="luxar-memory-table__header">TYPE</th>
            <th class="luxar-memory-table__header">REUSE %</th>
            <th class="luxar-memory-table__header">ACTIVE</th>
            <th class="luxar-memory-table__header">POOLED</th>
            <th class="luxar-memory-table__header">ALLOCS</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
      <div class="luxar-memory-section__summary" data-field="gpu-summary">
        Total: ${stats.allocations} allocs · ${stats.reuses} reuses · ${stats.evictions} evicted
      </div>
    </div>
  `;
}

/**
 * Render accumulators section with per-type table
 */
function renderAccumulatorsSection(accumulators: MemoryMetrics['accumulators']): string {
  const types = ['points', 'lines', 'gsplats'] as const;

  const rows = types
    .map((type) => {
      const stats = accumulators[type];
      const hasData = stats !== null && stats.capacity > 0;

      return `
      <tr class="luxar-memory-table__row" data-row="acc-${type}">
        <td class="luxar-memory-table__cell luxar-memory-table__cell--type">${capitalize(type)}</td>
        <td class="luxar-memory-table__cell luxar-memory-table__cell--value" data-field="acc-${type}-capacity">
          ${hasData ? formatNumber(stats.capacity) : '—'}
        </td>
        <td class="luxar-memory-table__cell luxar-memory-table__cell--value" data-field="acc-${type}-memory">
          ${hasData ? `${stats.memoryMB.toFixed(1)}MB` : '—'}
        </td>
        <td class="luxar-memory-table__cell luxar-memory-table__cell--value ${hasData && stats.growthEvents > 5 ? getColorClass('warning') : ''}" data-field="acc-${type}-grows">
          ${hasData ? stats.growthEvents : '—'}
        </td>
      </tr>
    `;
    })
    .join('');

  // Calculate totals
  const totalMemory =
    (accumulators.points?.memoryMB ?? 0) +
    (accumulators.lines?.memoryMB ?? 0) +
    (accumulators.gsplats?.memoryMB ?? 0);
  const totalAllocations =
    (accumulators.points?.allocations ?? 0) +
    (accumulators.lines?.allocations ?? 0) +
    (accumulators.gsplats?.allocations ?? 0);

  return `
    <div class="luxar-memory-section">
      <div class="luxar-memory-section__header">
        <span class="luxar-memory-section__icon">⚡</span>
        <span class="luxar-memory-section__title">DATA ACCUMULATORS</span>
      </div>
      <table class="luxar-memory-table">
        <thead>
          <tr class="luxar-memory-table__header-row">
            <th class="luxar-memory-table__header">TYPE</th>
            <th class="luxar-memory-table__header">CAPACITY</th>
            <th class="luxar-memory-table__header">MEMORY</th>
            <th class="luxar-memory-table__header">GROWS</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
      <div class="luxar-memory-section__summary" data-field="acc-summary">
        Total: ${totalMemory.toFixed(1)}MB · ${totalAllocations} allocations
      </div>
    </div>
  `;
}

/**
 * Capitalize first letter
 */
function capitalize(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// Helper functions (exported for use by value update functions in the monitor)

export function formatNumber(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + 'GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + 'MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + 'KB';
  return bytes.toFixed(0) + 'B';
}

/**
 * Get CSS color class for cache memory usage percentage
 */
export function getCacheMemoryColorClass(percent: number): string {
  if (percent <= 60) return getColorClass('success');
  if (percent <= 80) return getColorClass('warning');
  return getColorClass('error');
}

/**
 * Get CSS color class for progress percentage
 */
function getProgressColorClass(percent: number): string {
  if (percent <= 60) return getColorClass('success');
  if (percent <= 80) return getColorClass('warning');
  return getColorClass('error');
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
    gsplats: '🔮',
    mesh: '⬡',
  };
  return icons[type] || '•';
}

/**
 * Get CSS class for node type color
 */
function getNodeTypeColorClass(type: string): string {
  const typeClasses: Record<string, string> = {
    scene: 'luxar-scene-graph__name--scene',
    group: 'luxar-scene-graph__name--group',
    points: 'luxar-scene-graph__name--points',
    lines: 'luxar-scene-graph__name--lines',
    gsplats: 'luxar-scene-graph__name--gsplats',
    mesh: 'luxar-scene-graph__name--mesh',
  };
  return typeClasses[type] || '';
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
  } else if (node.type === 'gsplats' && node.splatCount !== undefined) {
    statsText = formatNumber(node.splatCount);
    statsTooltip = `${node.splatCount.toLocaleString()} Gaussian splats`;
    if (node.visibleSplatCount !== undefined && node.visibleSplatCount !== node.splatCount) {
      statsTooltip += ` (${node.visibleSplatCount.toLocaleString()} visible)`;
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
    gsplats: 'Gaussian splats layer',
    mesh: 'Mesh geometry',
  };
  const nodeTooltip = `${nodeTypeDescriptions[node.type] || node.type}${node.hasSpatialIndex ? ' (indexed)' : ''}`;

  // Expand/collapse toggle
  const toggleIcon = hasChildren ? (isExpanded ? '▼' : '▶') : '•';
  const toggleClass = hasChildren
    ? 'luxar-scene-graph__toggle--clickable'
    : 'luxar-scene-graph__toggle--disabled';

  // Indent using CSS custom property for dynamic depth
  const indentStyle = `style="--node-depth: ${depth}; padding-left: calc(var(--node-depth) * 16px);"`;

  return `
    <div class="luxar-scene-graph__node">
      <div class="luxar-scene-graph__node-row" ${indentStyle} title="${escapeHtml(nodeTooltip)}">
        <!-- Toggle -->
        <span
          class="luxar-scene-graph__toggle ${toggleClass}"
          ${hasChildren ? `data-action="toggleNode" data-node-path="${escapeHtml(node.path)}"` : ''}
          ${hasChildren ? `title="${isExpanded ? 'Collapse' : 'Expand'} ${escapeHtml(node.name)}"` : ''}
        >${toggleIcon}</span>

        <!-- Icon & Name -->
        <span class="luxar-scene-graph__icon">${getNodeTypeIcon(node.type)}</span>
        <span class="luxar-scene-graph__name ${getNodeTypeColorClass(node.type)}">
          ${escapeHtml(node.name)}
        </span>

        <!-- Stats badge -->
        ${
          statsText
            ? `<span class="luxar-scene-graph__badge" data-node-path="${escapeHtml(node.path)}" title="${escapeHtml(statsTooltip)}">${statsText}</span>`
            : ''
        }

        <!-- Loading indicator -->
        ${node.isLoading ? '<span class="luxar-scene-graph__loading" title="Loading data...">⏳</span>' : ''}
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
      <div class="luxar-scene-graph__empty">
        No scene loaded
      </div>
    `;
  }

  // Header with stats
  const headerStats = [
    state.pointsNodes > 0 ? `${state.pointsNodes} points` : null,
    state.linesNodes > 0 ? `${state.linesNodes} lines` : null,
    state.gsplatsNodes > 0 ? `${state.gsplatsNodes} gsplats` : null,
  ]
    .filter(Boolean)
    .join(', ');

  return `
    <div class="luxar-scene-graph">
      <div class="luxar-scene-graph__header">
        <h4 class="luxar-scene-graph__title">SCENE GRAPH</h4>
        ${headerStats ? `<span class="luxar-scene-graph__stats">${headerStats}</span>` : ''}
      </div>
      <div class="luxar-scene-graph__container">
        ${renderSceneGraphNode(state.root, expandedNodes)}
      </div>
    </div>
  `;
}
