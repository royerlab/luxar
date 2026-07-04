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
  CacheStatusBadge,
  SceneGraphNode,
  SceneGraphState,
  LODProgressState,
} from '../../types/data-monitor-types';
import { escapeHtml } from '../../utils/escape-html';

/**
 * Semantic color names mapped to CSS class modifiers.
 * These are used with the luxar-color--{name} classes.
 */
export type SemanticColor =
  | 'success'
  | 'warning'
  | 'error'
  | 'info'
  | 'muted'
  | 'dimmed'
  | 'primary';

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
 * @param tooltip - Optional hover tooltip describing the metric
 */
export function renderMetricCard(
  title: string,
  value: string | number,
  subtitle?: string,
  colorClass: string = '',
  size: 'small' | 'medium' | 'large' = 'medium',
  dataField?: string,
  tooltip?: string
): string {
  const fieldAttr = dataField ? ` data-field="${dataField}"` : '';
  const subFieldAttr = dataField ? ` data-field="${dataField}-sub"` : '';
  const tooltipAttr = tooltip ? ` title="${escapeHtml(tooltip)}"` : '';
  return `
    <div class="luxar-metric-card luxar-metric-card--${size}"${tooltipAttr}>
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
 * Map LoaderType identifier to its short display label / item-unit pair.
 * Keeps the loader-list rendering geometry-aware: a lines loader shows
 * "verts", a gsplats loader shows "splats", points shows "pts".
 */
function loaderDisplay(type: LoaderMetrics['type']): { label: string; unit: string } {
  switch (type) {
    case 'lines-spatial-index':
      return { label: 'lines', unit: 'verts' };
    case 'gsplats-spatial-index':
      return { label: 'gsplats', unit: 'splats' };
    case 'point-spatial-index':
    default:
      return { label: 'points', unit: 'pts' };
  }
}

/**
 * Template for loader list item
 */
export function renderLoaderItem(path: string, metrics: LoaderMetrics): string {
  const statusColorClass = metrics.queries > 0 ? getColorClass('success') : getColorClass('muted');
  const { label, unit } = loaderDisplay(metrics.type);

  return `
    <div class="luxar-loader-item" title="A loader is the component that streams one layer's data from the zarr store into the viewer — this one serves the ${escapeHtml(label)} layer at ${escapeHtml(path)}">
      <div class="luxar-loader-item__header">
        <span class="luxar-loader-item__path ${statusColorClass}" title="Path of this layer inside the dataset (green = has answered queries this session, grey = idle so far)">${escapeHtml(path)}</span>
        <span class="luxar-loader-item__status" title="Geometry type this loader streams (points, lines, or gsplats)">${escapeHtml(label)}</span>
      </div>
      <div class="luxar-loader-item__metrics">
        <span title="Elements from this layer currently on screen (inside the active nD slice)">${metrics.visiblePoints.toLocaleString()} ${escapeHtml(unit)}</span>
        <span title="CPU memory this loader currently holds for loaded chunks and index data">${formatBytes(metrics.memoryUsed)}</span>
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
  network:
    | {
        bytesTransferred: number;
        requestCount: number;
        bandwidth: number;
        totalBytesServed?: number;
        totalRequestsServed?: number;
      }
    | undefined
): string {
  const memoryPercent = memory.limit > 0 ? (memory.used / memory.limit) * 100 : 0;
  // Cumulative bytes delivered across all tiers (L1 + L2 + network).
  // Falls back to network bytes for providers predating the field.
  const dataLoaded = network ? (network.totalBytesServed ?? network.bytesTransferred) : 0;
  // Demand reads served across all tiers; falls back to the network request
  // count for providers predating the field (mirrors the bytes fallback).
  const requestsServed = network ? (network.totalRequestsServed ?? network.requestCount) : 0;

  return `
    <div class="luxar-secondary-metrics">
      <div class="luxar-secondary-metrics__item">
        <span class="luxar-secondary-metrics__label" title="Memory currently used by cached data across all cache tiers, with the bar showing usage against the configured limit. See the Cache tab for the per-tier (L0/L1/L2) breakdown">MEMORY</span>
        <div class="luxar-secondary-metrics__value" data-field="memory-used">
          ${formatBytes(memory.used)}
        </div>
        ${renderProgressBar(memoryPercent, getCacheMemoryColorClass(memoryPercent), '', 2)}
      </div>

      <div class="luxar-secondary-metrics__item">
        <span class="luxar-secondary-metrics__label" title="How long a spatial query takes on average, in milliseconds. A query asks the spatial index which data falls inside the current view/slice — it runs every time you pan, zoom, or move a dimension slider. The subtitle is how many such queries run per second">QUERY SPEED</span>
        <div class="luxar-secondary-metrics__value" data-field="query-speed">
          ${querySpeed.avgTime.toFixed(0)}ms
        </div>
        <div class="luxar-secondary-metrics__subtitle" data-field="query-rate">
          ${querySpeed.perSec.toFixed(1)}/sec
        </div>
      </div>

      <div class="luxar-secondary-metrics__item">
        <span class="luxar-secondary-metrics__label" title="Total data delivered to the renderer since load, from all sources combined (memory caches + disk cache + network). The subtitle breaks out the network share — 'net' is what was actually downloaded, followed by current download bandwidth and the total request count. A big gap between loaded and net means the cache is doing its job">DATA LOADED</span>
        <div class="luxar-secondary-metrics__value" data-field="network-bytes">
          ${network ? formatBytes(dataLoaded) : '0B'}
        </div>
        <div class="luxar-secondary-metrics__subtitle" data-field="network-detail">
          ${network ? `${formatBytes(network.bytesTransferred)} net · ${formatBytes(network.bandwidth)}/s · ${requestsServed.toLocaleString()} reqs` : '0B net'}
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
          'visible-points',
          'How many points are on screen right now versus how many the whole dataset holds. The two differ because only data inside the current nD slice is shown, and level-of-detail (LOD) streaming may not have loaded full resolution yet'
        )}
        ${renderMetricCard(
          'VISIBLE LINES',
          formatNumber(stats.visibleSegments),
          `${visibleSegmentsPercent}% of ${formatNumber(stats.datasetSegments)}`,
          getColorClass('warning'),
          'small',
          'visible-lines',
          'How many line segments are on screen right now versus how many the whole dataset holds. The two differ because only data inside the current nD slice is shown, and level-of-detail (LOD) streaming may not have loaded full resolution yet'
        )}
        ${renderMetricCard(
          'VISIBLE SPLATS',
          formatNumber(stats.visibleSplats),
          `${visibleSplatsPercent}% of ${formatNumber(stats.datasetSplats)}`,
          getColorClass('info'),
          'small',
          'visible-splats',
          'How many Gaussian splats are on screen right now versus how many the whole dataset holds. The two differ because only data inside the current nD slice is shown, and level-of-detail (LOD) streaming may not have loaded full resolution yet'
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
          'visible-points',
          'How many points are on screen right now versus how many the whole dataset holds. The two differ because only data inside the current nD slice is shown, and level-of-detail (LOD) streaming may not have loaded full resolution yet'
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
          'visible-lines',
          'How many line segments are on screen right now versus how many the whole dataset holds. The two differ because only data inside the current nD slice is shown, and level-of-detail (LOD) streaming may not have loaded full resolution yet'
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
          'visible-splats',
          'How many Gaussian splats are on screen right now versus how many the whole dataset holds. The two differ because only data inside the current nD slice is shown, and level-of-detail (LOD) streaming may not have loaded full resolution yet'
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
          'visible-points',
          'How many points are on screen right now versus how many the whole dataset holds. The two differ because only data inside the current nD slice is shown, and level-of-detail (LOD) streaming may not have loaded full resolution yet'
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
          'visible-lines',
          'How many line segments are on screen right now versus how many the whole dataset holds. The two differ because only data inside the current nD slice is shown, and level-of-detail (LOD) streaming may not have loaded full resolution yet'
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
          'visible-splats',
          'How many Gaussian splats are on screen right now versus how many the whole dataset holds. The two differ because only data inside the current nD slice is shown, and level-of-detail (LOD) streaming may not have loaded full resolution yet'
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
  // R3: support 4-card sections (L2 now includes an ERRORS card alongside
  // SIZE / HIT RATE / I/O). 2 → cols-2, 3 → cols-3, anything else → cols-4.
  const cols = metrics.length === 2 ? 'cols-2' : metrics.length === 3 ? 'cols-3' : 'cols-4';

  return `
    <div class="luxar-cache-section">
      <div class="luxar-cache-section__header">
        <span class="luxar-cache-section__title"${titleTooltip ? ` title="${escapeHtml(titleTooltip)}"` : ''}>${title}</span>
        <button data-action="${clearAction}" class="luxar-cache-section__clear-btn" title="${escapeHtml(clearTooltip)}">Clear</button>
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
          <div class="luxar-metric-card luxar-metric-card--small"${metric.tooltip ? ` title="${escapeHtml(metric.tooltip)}"` : ''}>
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
 * R3: Color class per cache status badge. Exported so the cache-tab
 * incremental updater renders the same colors as the initial template
 * and so unit tests can assert on the mapping.
 */
export const CACHE_BADGE_COLOR: Record<CacheStatusBadge, string> = {
  'cache-enabled': getColorClass('success'),
  'no-cache': getColorClass('dimmed'),
  'disabled-config': getColorClass('dimmed'),
  'opfs-unavailable': getColorClass('warning'),
  'quota-constrained': getColorClass('warning'),
  'cache-errors-detected': getColorClass('error'),
  'unvalidated-external-dataset': getColorClass('warning'),
  'provider-missing': getColorClass('error'),
};

/**
 * Didactic hover explanation per cache status badge. Each entry says
 * what the badge means, why it appears, and what (if anything) the
 * user should do about it. Exported so the cache-tab incremental
 * updater and unit tests share the exact same wording.
 */
export const CACHE_BADGE_TOOLTIP: Record<CacheStatusBadge, string> = {
  'cache-enabled':
    'Multi-level caching is active. Downloaded chunks are kept in memory (L0 decoded + L1 raw) ' +
    'and on disk (L2, browser private storage), so re-slicing and revisits are served locally ' +
    'instead of re-downloading. Nothing to do — this is the healthy state.',
  'no-cache':
    'Caching is turned off for this session by the ?no-cache URL parameter: every chunk is ' +
    'fetched from the network each time it is needed and nothing persists across reloads. ' +
    'Remove ?no-cache from the URL to re-enable caching.',
  'disabled-config':
    'Caching is turned off in the viewer configuration (cache.enabled / cache.l0Enabled): every ' +
    'chunk is fetched from the network each time it is needed. Enable it in the app config to ' +
    'speed up repeat access.',
  'opfs-unavailable':
    'The browser did not grant Origin Private File System storage, so the persistent L2 disk ' +
    'cache is off. In-memory caching (L0/L1) still works, but nothing survives a page reload. ' +
    'Common in private/incognito windows or when site storage is blocked.',
  'quota-constrained':
    'The browser storage quota is full: some chunks could not be written to the L2 disk cache ' +
    'and will have to be re-downloaded in future sessions. Free up disk space or clear other ' +
    'site data to restore full caching.',
  'cache-errors-detected':
    'The L2 disk cache hit errors: failed writes, corrupted entries, or unreadable metadata ' +
    '(see the ERRORS card for the breakdown). Corrupt entries are dropped and re-fetched ' +
    'automatically; if the count keeps growing, press Clear All to rebuild the cache.',
  'unvalidated-external-dataset':
    'Warning, not an error: this dataset carries no content_hash (it was not produced by the ' +
    'Luxar compiler) and no cache TTL is configured, so the viewer cannot detect whether the ' +
    'file changed on the server. Cached chunks are trusted indefinitely — if the data may have ' +
    'been updated, press Clear All to force a fresh download.',
  'provider-missing':
    'Internal inconsistency: telemetry reports caching as enabled, but no cache provider is ' +
    'attached, so the statistics on this tab may be incomplete. Usually transient during a ' +
    'scene switch; if it persists, reload the page.',
};

/**
 * R3: Render one HTML pill per CacheStatusBadge. Returns an empty
 * string when no badges are present so `data-field="cache-status-row"`
 * still exists in the DOM (the incremental patcher fills it).
 */
export function renderCacheStatusBadges(badges: CacheStatusBadge[] | undefined): string {
  if (!badges || badges.length === 0) return '';
  return badges
    .map(
      (b) =>
        `<span class="luxar-badge ${CACHE_BADGE_COLOR[b] ?? getColorClass('muted')}" data-badge="${b}" title="${escapeHtml(CACHE_BADGE_TOOLTIP[b] ?? b)}">${b}</span>`
    )
    .join('');
}

/**
 * Render the cache-status badge row. In the full L1/L2 cache view we
 * keep the row mounted even when empty so the incremental updater can
 * patch it in place. Disabled/fallback cache views pass `always=false`
 * so a truly empty status set doesn't add a blank spacer above the
 * explanatory disabled message.
 */
function renderCacheStatusRow(badges: CacheStatusBadge[] | undefined, always = false): string {
  const signature = (badges ?? []).join('|');
  if (!always && signature.length === 0) return '';
  return `
    <div class="luxar-cache-status" data-field="cache-status-row" data-signature="${signature}" title="Cache status badges — green is healthy, amber is a limitation to be aware of, red needs attention. Hover each badge for a full explanation">
      ${renderCacheStatusBadges(badges)}
    </div>
  `;
}

/**
 * R3: Friendly label for a validation mode. The cache-tab UI shows
 * this verbatim; null/undefined render as a neutral placeholder so
 * callers don't have to guard the value themselves.
 */
export function formatValidationMode(
  mode: 'content-hash' | 'zattrs-hash' | 'ttl' | 'none' | undefined
): string {
  switch (mode) {
    case 'content-hash':
      return 'Content Hash';
    case 'zattrs-hash':
      return 'Metadata Hash';
    case 'ttl':
      return 'TTL';
    case 'none':
      return 'None';
    default:
      return '—';
  }
}

/**
 * Didactic hover explanation for the current validation mode. Mode-
 * specific so the tooltip always explains what the *shown* value means
 * (the generic "could be any of three modes" wording read as evasive).
 * Shared by the initial template and the cache-tab incremental updater
 * so the tooltip stays correct when the mode changes after the first
 * validation completes (e.g. '—' → Content Hash).
 */
export function validationModeTooltip(
  mode: 'content-hash' | 'zattrs-hash' | 'ttl' | 'none' | undefined
): string {
  switch (mode) {
    case 'content-hash':
      return (
        'Content-hash validation (strongest): this dataset was produced by the Luxar compiler ' +
        'and publishes a content_hash fingerprint in its root metadata. At load time the viewer ' +
        're-fetches that fingerprint from the server and compares it with the one stored next ' +
        'to the disk cache. If they differ, every cache tier is cleared and the data is ' +
        're-downloaded — you can never be shown stale data.'
      );
    case 'zattrs-hash':
      return (
        'Metadata-hash validation: the dataset publishes no content_hash fingerprint, so the ' +
        'viewer fingerprints the raw root metadata (.zattrs) bytes instead. Luxar writers stamp ' +
        'a fresh timestamp on every save, so a dataset regenerated at the same URL is detected ' +
        'and every cache tier cleared. Only a producer that rewrites chunk data without touching ' +
        'root metadata could still serve stale chunks.'
      );
    case 'ttl':
      return (
        'Time-to-live validation: the dataset root metadata could not be fetched for a ' +
        'fingerprint, but a maximum cache age is configured (cache.externalDatasetTtlMs). ' +
        'Cached data older than that age is discarded and re-downloaded. Within the window, a ' +
        'change on the server is NOT detected — the TTL bounds how stale the view can get.'
      );
    case 'none':
      return (
        'No validation: the dataset has no content_hash fingerprint (it was not produced by the ' +
        'Luxar compiler — e.g. a plain external zarr) and no cache TTL is configured. Cached ' +
        'chunks are served indefinitely, so if the file changes on the server you will keep ' +
        'seeing the old data until you press Clear All to force a fresh download.'
      );
    default:
      return (
        'How the cache decides whether its stored chunks still match the dataset on the server. ' +
        'Not determined yet — the freshness check runs right after the dataset loads.'
      );
  }
}

/**
 * Didactic hover explanation for the "Last Validated" timestamp,
 * mode-aware because the timestamp means different things: under
 * content-hash it is a real confirmation; under `none` it only records
 * that the check ran and found nothing to compare.
 */
export function lastValidatedTooltip(
  mode: 'content-hash' | 'zattrs-hash' | 'ttl' | 'none' | undefined
): string {
  const base =
    'When the viewer last ran its freshness check (it re-fetches the dataset root metadata ' +
    'from the server at load time). ';
  switch (mode) {
    case 'content-hash':
      return (
        base +
        'At this moment the cached content_hash was compared against the server and the cache ' +
        'was confirmed current (or cleared if it did not match). "Never" = no check has ' +
        'completed yet, e.g. offline.'
      );
    case 'zattrs-hash':
      return (
        base +
        'At this moment the fingerprint of the dataset root metadata (.zattrs bytes) was ' +
        'compared against the server and the cache was confirmed current (or cleared if it did ' +
        'not match). "Never" = no check has completed yet, e.g. offline.'
      );
    case 'ttl':
      return (
        base +
        'This timestamp starts the TTL countdown: once the cache is older than the configured ' +
        'maximum age it is discarded and re-downloaded. "Never" = no check has completed yet, ' +
        'e.g. offline.'
      );
    case 'none':
      return (
        base +
        'Careful: with validation "None" this is only when the check last RAN — it found no ' +
        'content_hash to compare, so it does NOT confirm the cached data matches the server. ' +
        '"Never" = no check has completed yet, e.g. offline.'
      );
    default:
      return base + '"Never" = no check has completed yet, e.g. offline.';
  }
}

/**
 * Row label for the freshness timestamp, mode-aware to match the
 * timestamp's actual meaning: only content-hash mode truly VALIDATES
 * the cache against the server; under ttl/none the check merely ran
 * (found no fingerprint to compare), so "Last Validated" would
 * overstate what happened — "Last Checked" is the honest label.
 */
export function lastValidatedLabel(
  mode: 'content-hash' | 'zattrs-hash' | 'ttl' | 'none' | undefined
): string {
  // Both hash modes genuinely VALIDATE the cache against the server;
  // ttl/none merely record that the check ran.
  return mode === 'content-hash' || mode === 'zattrs-hash' ? 'Last Validated' : 'Last Checked';
}

/**
 * R3: Friendly timestamp for `health.lastValidatedAt`. null renders
 * as "Never"; valid timestamps use the browser's locale formatter.
 */
export function formatLastValidated(ts: number | null | undefined): string {
  if (ts == null || !Number.isFinite(ts)) return 'Never';
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return 'Never';
  }
}

/**
 * R3: Sum of OPFS health-failure counters surfaced on
 * `CacheMetrics.l2`. Used to decide whether the L2 "ERRORS" card
 * should render in error vs dimmed color.
 */
export function l2ErrorTotal(l2: CacheMetrics['l2']): number {
  if (!l2) return 0;
  return (
    (l2.quotaWriteSkipped ?? 0) +
    (l2.writeFailures ?? 0) +
    (l2.corruptedEntries ?? 0) +
    (l2.metadataParseFailures ?? 0)
  );
}

/**
 * Template for cache tab content with L0/L1/L2 breakdown
 */
export function renderCacheContent(_stats: GlobalStats, cacheMetrics: CacheMetrics): string {
  // Switch on the explicit telemetry state so each not-enabled
  // variant gets a faithful message. Fall back to the `enabled`
  // boolean only when telemetryState is absent.
  const stateKind =
    cacheMetrics.telemetryState?.kind ?? (cacheMetrics.enabled === false ? 'not-wired' : 'enabled');

  if (stateKind !== 'enabled') {
    let message: string;
    let hint: string;
    switch (stateKind) {
      case 'disabled-no-cache':
        message = 'Caching disabled by ?no-cache';
        hint = 'Remove ?no-cache from URL to enable';
        break;
      case 'disabled-config':
        message = 'Caching disabled by configuration';
        hint = 'Enable cache.enabled or cache.l0Enabled in app config';
        break;
      case 'not-wired':
      default:
        message = 'Cache statistics not connected';
        hint = 'Loading scene...';
        break;
    }
    return `
      <div class="luxar-tab-content--cache">
        ${renderCacheStatusRow(cacheMetrics.status)}
        <div class="luxar-cache-disabled">
          <div class="luxar-cache-disabled__icon">🚫</div>
          <div class="luxar-cache-disabled__message">${message}</div>
          <div class="luxar-cache-disabled__hint">${hint}</div>
        </div>
      </div>
    `;
  }

  // Check if we have L1/L2 breakdown
  const hasL1L2 = cacheMetrics.l1 !== undefined && cacheMetrics.l2 !== undefined;

  if (!hasL1L2) {
    // Fallback to basic view if no cache stats provider connected.
    // Keep any status badges visible here as well (for example an
    // enabled/provider-missing diagnostic) even though the full L1/L2
    // cache structure is not mounted yet.
    return `
      <div class="luxar-tab-content--cache">
        ${renderCacheStatusRow(cacheMetrics.status)}
        <div class="luxar-grid-2">
          ${renderMetricCard(
            'CACHE MEMORY',
            formatBytes(cacheMetrics.totalCacheMemory),
            `${cacheMetrics.memoryPercent.toFixed(0)}% of ${formatBytes(cacheMetrics.memoryLimit)}`,
            getColorClass('success'),
            'medium',
            undefined,
            'Memory currently used by cached data across all cache tiers, relative to the configured limit. Caching keeps downloaded chunks local so re-slicing and revisits do not re-download them'
          )}
          ${renderMetricCard(
            'CACHED ENTRIES',
            cacheMetrics.totalEntries.toString(),
            '',
            getColorClass('primary'),
            'medium',
            undefined,
            'Number of chunks/entries currently held across all cache tiers. The full per-tier breakdown (L0/L1/L2) appears once cache statistics finish connecting'
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
  // S3: dimmed on no-access, matches the incremental updater so a
  // freshly loaded session doesn't flash red on first paint.
  const l0HitRateColorClass = getCacheHitRateColorClassWithGuard(l0HitRate, l0Total);

  // L1 hit rate calculation
  const l1Total = cacheMetrics.l1!.hits + cacheMetrics.l1!.misses;
  const l1HitRate = l1Total > 0 ? (cacheMetrics.l1!.hits / l1Total) * 100 : 0;

  const l1HitRateColorClass = getCacheHitRateColorClassWithGuard(l1HitRate, l1Total);

  // R3: status pill row. In the full view it lives inline on the right
  // of the CACHE HEALTH header (rather than its own full-width row at the
  // top) to save vertical space. Always rendered (with
  // `data-field="cache-status-row"`) so the incremental patcher can
  // refresh badge sets without a full re-render.
  const statusRowHtml = renderCacheStatusRow(cacheMetrics.status, true);

  // R3: L2 error-counter card. Sums the four OPFS health counters
  // (quotaWriteSkipped + writeFailures + corruptedEntries +
  // metadataParseFailures) — shows the total + a per-counter
  // breakdown subtitle.
  const l2Errors = l2ErrorTotal(cacheMetrics.l2);
  const l2 = cacheMetrics.l2!;

  return `
    <div class="luxar-tab-content--cache">
      <!-- L0 Decompressed Chunk Cache Section (fastest layer - avoids Blosc decompression) -->
      ${
        cacheMetrics.l0
          ? renderCacheSection(
              'L0 DECOMPRESSED CACHE',
              'The fastest cache tier. Data chunks arrive compressed and must be decoded (~2ms each) before use; L0 keeps the already-decoded arrays in memory so repeat reads skip both the download AND the decode. Lookups try L0 first, then fall through L1 (memory) → L2 (disk) → network',
              'clearL0',
              'Empty the L0 decoded-chunk cache. Harmless: chunks are still in L1/L2 and will simply be re-decoded (~2ms each) on next access',
              [
                {
                  label: 'SIZE',
                  value: formatBytes(cacheMetrics.l0.size),
                  subtitle: `${cacheMetrics.l0.count} chunks`,
                  tooltip:
                    'Memory currently held by decoded (ready-to-use) chunks, and how many chunks that is. Bounded by an LRU limit — see EVICTIONS',
                  colorClass: getColorClass('primary'),
                  dataField: 'l0-size',
                },
                {
                  label: 'HIT RATE',
                  value: `${l0HitRate.toFixed(1)}%`,
                  subtitle: `${formatNumber(cacheMetrics.l0.hits)} hits · ${formatNumber(cacheMetrics.l0.misses)} miss`,
                  tooltip:
                    'Share of chunk requests answered by L0. A hit returns a decoded array in ~1μs; a miss pays ~2ms decompression after fetching the raw chunk from L1/L2/network. High is good; a low rate right after loading is normal while the cache warms up',
                  colorClass: l0HitRateColorClass,
                  dataField: 'l0-hitrate',
                },
                {
                  label: 'EVICTIONS',
                  value: formatNumber(cacheMetrics.l0.evictions),
                  subtitle: 'LRU removed',
                  tooltip:
                    'Chunks pushed out of L0 (least-recently-used first) because it reached its memory limit. Evicted chunks are not lost — they remain in L1/L2 and are re-decoded on demand. Steady growth just means the working set is larger than the L0 limit',
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
      ${renderCacheSection(
        'L1 MEMORY CACHE',
        'The in-memory tier for raw (still-compressed) chunks and metadata. Serves L0 misses from RAM with no disk or network round-trip. Cleared when the page closes — the persistent copy lives in L2. Lookup order: L0 → L1 → L2 → network',
        'clearL1',
        'Empty the L1 in-memory cache. Harmless: chunks still cached on disk (L2) are re-read from there; only uncached data goes back to the network',
        [
          {
            label: 'SIZE',
            value: formatBytes(cacheMetrics.l1!.size),
            subtitle: `${cacheMetrics.l1!.count} entries`,
            tooltip:
              'RAM currently held by raw chunks + metadata in L1, and the number of entries. Bounded by an LRU limit — see EVICTIONS',
            colorClass: getColorClass('success'),
            dataField: 'l1-size',
          },
          {
            label: 'HIT RATE',
            value: `${l1HitRate.toFixed(1)}%`,
            subtitle: `${formatNumber(cacheMetrics.l1!.hits)} hits · ${formatNumber(cacheMetrics.l1!.misses)} miss`,
            tooltip: `Share of L1 lookups served from RAM (${cacheMetrics.l1!.hits.toLocaleString()} hits of ${l1Total.toLocaleString()} accesses). A miss falls through to the L2 disk cache, and to the network only if L2 misses too. High is good; low right after loading is normal while the cache warms up`,
            colorClass: l1HitRateColorClass,
            dataField: 'l1-hitrate',
          },
          {
            label: 'EVICTIONS',
            value: formatNumber(cacheMetrics.l1!.evictions),
            subtitle: 'LRU removed',
            tooltip:
              'Entries pushed out of L1 (least-recently-used first) because it reached its memory limit. Evicted entries usually persist in the L2 disk cache, so they are re-read from disk rather than re-downloaded. Steady growth means the working set exceeds the L1 limit',
            colorClass:
              cacheMetrics.l1!.evictions > 0 ? getColorClass('warning') : getColorClass('dimmed'),
            dataField: 'l1-evictions',
          },
        ]
      )}

      <!-- L2 OPFS Cache Section -->
      ${(() => {
        const l2Total = cacheMetrics.l2!.reads + cacheMetrics.l2!.misses;
        const l2HitRate = l2Total > 0 ? (cacheMetrics.l2!.reads / l2Total) * 100 : 0;
        const l2HitRateColorClass = getCacheHitRateColorClassWithGuard(l2HitRate, l2Total);
        return renderCacheSection(
          'L2 OPFS CACHE',
          "The persistent disk tier, stored in the browser's Origin Private File System (private storage on your machine — never uploaded anywhere). Survives page reloads and browser restarts, so a revisited dataset loads from disk instead of the network. Lookup order: L0 → L1 → L2 → network",
          'clearL2',
          'Delete the on-disk (L2) cache for this dataset. Anything not held in memory will be re-downloaded from the server — use this to reclaim disk space or force a fresh copy',
          [
            {
              label: 'SIZE',
              value: formatBytes(cacheMetrics.l2!.size),
              subtitle: `${cacheMetrics.l2!.count} entries`,
              tooltip:
                "Disk space used by cached chunks in the browser's private storage, and the number of entries. Persists across sessions; counts against the browser storage quota (see the quota-constrained badge if it fills up)",
              colorClass: getColorClass('info'),
              dataField: 'l2-size',
            },
            {
              label: 'HIT RATE',
              value: l2Total > 0 ? `${l2HitRate.toFixed(1)}%` : '—',
              subtitle: `${formatNumber(cacheMetrics.l2!.reads)} hits · ${formatNumber(cacheMetrics.l2!.misses)} miss`,
              tooltip: `Of the requests that missed the memory caches and fell through to disk, the share found there (${cacheMetrics.l2!.reads.toLocaleString()} of ${l2Total.toLocaleString()}). An L2 miss is the only case that costs a network download. "—" = nothing has fallen through to L2 yet`,
              colorClass: l2HitRateColorClass,
              dataField: 'l2-hitrate',
            },
            {
              label: 'I/O',
              value: `${formatNumber(cacheMetrics.l2!.reads)} reads`,
              subtitle: `${formatNumber(cacheMetrics.l2!.writes)} writes`,
              tooltip:
                'Disk traffic: reads = chunks served from the on-disk cache; writes = freshly downloaded chunks saved to disk so future sessions can skip the download',
              dataField: 'l2-io',
            },
            {
              label: 'ERRORS',
              value: l2Errors > 0 ? formatNumber(l2Errors) : '0',
              subtitle:
                l2Errors > 0
                  ? `${formatNumber(l2.quotaWriteSkipped ?? 0)} quota · ${formatNumber(
                      l2.writeFailures ?? 0
                    )} write · ${formatNumber(l2.corruptedEntries ?? 0)} corrupt`
                  : 'no errors',
              tooltip:
                'Problems in the disk tier, summed: quota = writes skipped because browser storage is full; write = writes that failed outright; corrupt = stored entries that failed integrity checks and were dropped (auto re-fetched); plus unreadable metadata. Occasional errors self-heal; a growing count → press Clear All',
              colorClass: l2Errors > 0 ? getColorClass('error') : getColorClass('dimmed'),
              dataField: 'l2-errors',
            },
          ]
        );
      })()}

      <!-- R3: Cache Health — validation mode + last-validated timestamp +
           operational status badges (inline, right of the header).
           Always rendered so the incremental patcher can refresh values. -->
      <div class="luxar-cache-health">
        <div class="luxar-cache-health__header">
          <span class="luxar-cache-health__title" title="Is the cached data trustworthy? Shows how (and when) the viewer checks that its cached chunks still match the dataset on the server, plus status badges for anything that needs attention — hover each badge and row for details">CACHE HEALTH</span>
          ${statusRowHtml}
        </div>
        <div class="luxar-cache-health__row">
          <span class="luxar-cache-health__label" title="The strategy used to detect a dataset that changed on the server: Content Hash (fingerprint comparison, Luxar-compiled datasets), TTL (cached data expires after a configured age), or None (no change detection — external dataset)">Validation</span>
          <span class="luxar-cache-health__value" data-field="cache-health-mode" title="${escapeHtml(validationModeTooltip(cacheMetrics.health?.validationMode))}">
            ${formatValidationMode(cacheMetrics.health?.validationMode)}
          </span>
        </div>
        <div class="luxar-cache-health__row">
          <span class="luxar-cache-health__label" data-field="cache-health-validated-label" title="Timestamp of the most recent freshness check against the server — hover the value for what that means under the current validation mode">${lastValidatedLabel(cacheMetrics.health?.validationMode)}</span>
          <span class="luxar-cache-health__value" data-field="cache-health-validated" title="${escapeHtml(lastValidatedTooltip(cacheMetrics.health?.validationMode))}">
            ${formatLastValidated(cacheMetrics.health?.lastValidatedAt)}
          </span>
        </div>
      </div>

      <!-- Combined Stats + Clear All -->
      <div class="luxar-cache-total">
        <div class="luxar-cache-total__header">
          <span class="luxar-cache-total__label" title="Total space used by cached data across every tier: L0 (decoded, memory) + L1 (raw, memory) + L2 (disk). The bar below shows usage against the configured limit">TOTAL</span>
          <button data-action="clearAll" class="luxar-cache-section__clear-btn" title="Delete everything in every cache tier (L0 + L1 + L2 disk). The scene stays loaded, but data needed afterwards is re-downloaded from the server. Use this to force a fresh copy of a dataset that may have changed (especially with Validation: None), or to reclaim disk space">Clear All</button>
        </div>
        <div class="luxar-cache-total__value" data-field="cache-total" title="${cacheMetrics.totalCacheMemory.toLocaleString()} bytes total cached">
          ${formatBytes(cacheMetrics.totalCacheMemory)}
        </div>
        ${renderProgressBar(cacheMetrics.memoryPercent, getCacheMemoryColorClass(cacheMetrics.memoryPercent), cacheMetrics.memoryLimit > 0 ? `${cacheMetrics.memoryPercent.toFixed(0)}% of ${formatBytes(cacheMetrics.memoryLimit)} limit` : 'no memory limit configured', 6)}
        ${
          cacheMetrics.effectiveDemandHitRate !== undefined
            ? `<div class="luxar-cache-total__demand" data-field="cache-effective-hitrate" title="The bottom line for caching: of all data requests made by the renderer, the share answered by ANY cache tier (L0, L1, or L2) instead of the network. 100% = fully local, no downloads; low values right after first load are normal — the caches have to be filled once before they can hit">
                 EFFECTIVE HIT RATE: ${(cacheMetrics.effectiveDemandHitRate * 100).toFixed(1)}%
               </div>`
            : ''
        }
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

// Memory-metrics contracts live in `types/data-monitor-types` so the data
// layer's SceneLoaderMonitorPort can reference them precisely. Re-exported
// here for existing UI template callers; the local `import type` is needed
// because other functions in this file reference these types directly.
import type {
  GPUPoolTypeStats,
  GPUPoolStats,
  AccumulatorStats,
  MemoryMetrics,
} from '../../types/data-monitor-types';
export type { GPUPoolTypeStats, GPUPoolStats, AccumulatorStats, MemoryMetrics };

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
 * Color class for cache hit-rate metrics (L0/L1/L2). Same threshold
 * shape as `getReuseRateColorClass`, but with `>` semantics so an
 * 80% hit rate still shows as warning — caches do not spend much
 * time at exactly 80%, but any drop below the threshold is meaningful.
 */
export function getCacheHitRateColorClass(rate: number): string {
  if (rate > 80) return getColorClass('success');
  if (rate > 50) return getColorClass('warning');
  return getColorClass('error');
}

/**
 * Like {@link getCacheHitRateColorClass} but returns the dimmed color
 * when no accesses have happened yet. Keeps the initial render of
 * the cache tab consistent with the incremental cache-tab updater,
 * which already special-cases the no-data state. Without this, a
 * freshly loaded session shows hit-rate cards in red (error color)
 * on first paint, then flips to dimmed on the next 1s poll tick.
 */
export function getCacheHitRateColorClassWithGuard(rate: number, totalAccesses: number): string {
  if (totalAccesses === 0) return getColorClass('dimmed');
  return getCacheHitRateColorClass(rate);
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
        <span class="luxar-memory-section__title" title="A recycling pool for GPU buffers. Streaming constantly needs new buffers as data arrives; allocating GPU memory is slow, so finished buffers are returned to a pool and handed back out instead of reallocated. REUSE % tells you how well that is working">GPU BUFFER POOL</span>
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
      <div class="luxar-memory-total" title="One-line summary of this tab: total GPU buffer allocations since load, the share of buffer requests served by pool reuse (higher = smoother streaming), and CPU memory held by the accumulators">
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
        <span class="luxar-memory-section__title" title="A recycling pool for GPU buffers. Streaming constantly needs new buffers as data arrives; allocating GPU memory is slow, so finished buffers are returned to a pool and handed back out instead of reallocated. REUSE % tells you how well that is working">GPU BUFFER POOL</span>
      </div>
      <table class="luxar-memory-table">
        <thead>
          <tr class="luxar-memory-table__header-row">
            <th class="luxar-memory-table__header" title="Geometry type whose GPU buffers this row tracks (points, lines, or gsplats)">TYPE</th>
            <th class="luxar-memory-table__header" title="Share of buffer requests served by recycling a pooled buffer instead of allocating a fresh one. Higher is better — GPU allocation is expensive, so high reuse means smoother streaming">REUSE %</th>
            <th class="luxar-memory-table__header" title="Buffers currently checked out of the pool and holding live geometry data">ACTIVE</th>
            <th class="luxar-memory-table__header" title="Free buffers kept in the pool, ready to be handed out without a new allocation">POOLED</th>
            <th class="luxar-memory-table__header" title="Total fresh GPU buffer allocations since load — grows only on pool misses">ALLOCS</th>
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
        <span class="luxar-memory-section__title" title="CPU-side staging buffers that collect geometry attributes (positions, colors, ...) as chunks stream in, before GPU upload. They over-allocate and grow geometrically so appending stays cheap — see the GROWS column for how often growth was needed">DATA ACCUMULATORS</span>
      </div>
      <table class="luxar-memory-table">
        <thead>
          <tr class="luxar-memory-table__header-row">
            <th class="luxar-memory-table__header" title="Geometry type whose accumulator this row tracks (points, lines, or gsplats)">TYPE</th>
            <th class="luxar-memory-table__header" title="Elements the accumulator can hold before it must grow. Capacity is over-allocated ahead of demand so most incoming chunks append without a reallocation">CAPACITY</th>
            <th class="luxar-memory-table__header" title="CPU memory currently allocated by this accumulator's attribute buffers">MEMORY</th>
            <th class="luxar-memory-table__header" title="How many times the accumulator had to grow (reallocate + copy) to fit more streamed data. Frequent growth is a sign the initial capacity estimate was too small (highlighted above 5)">GROWS</th>
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
 * Pick the tree icon for a node. Specialized groups (`kind=lod` /
 * `kind=partition`) get their own glyph so they read distinctly from
 * plain containers; everything else falls back to its geometry type.
 */
function getSceneGraphIcon(node: SceneGraphNode): string {
  if (node.kind === 'lod') return '🎚️';
  if (node.kind === 'partition') return '🧩';
  return getNodeTypeIcon(node.type);
}

/**
 * Render the kind badge (`K LODs` / `N parts`) for a specialized group.
 * Mirrors the Layers-panel `--kind` badge so the two panels read
 * consistently. Returns `''` for non-specialized nodes.
 */
function renderKindBadge(node: SceneGraphNode): string {
  if (node.kind === 'lod' && (node.lodGroupChildCount ?? 0) > 0) {
    const title = `Substitutive LOD group · ${node.lodGroupChildCount} levels (one rendered at a time)`;
    return `<span class="luxar-scene-graph__badge luxar-scene-graph__badge--kind" title="${escapeHtml(title)}">${node.lodGroupChildCount} LODs</span>`;
  }
  if (node.kind === 'partition' && (node.partCount ?? 0) > 0) {
    const title = `Partition group · ${node.partCount} disjoint BSP parts — all parts are rendered; the GPU frustum-culls each part at draw time`;
    return `<span class="luxar-scene-graph__badge luxar-scene-graph__badge--kind" title="${escapeHtml(title)}">${node.partCount} parts</span>`;
  }
  return '';
}

/**
 * Render the live LOD-progress chip for a node, driven by the
 * {@link LODProgressState} snapshot:
 *   - substitutive (`kind=lod`): "L{active+1}/{count}" active-level chip.
 *   - additive (`additiveSublods`): "LOD {loaded}/{total}" with a ⏳ while
 *     refinement is in progress and a residency dot (● cached / ◌ streaming).
 * Renders a structural slot even before the first provider poll so the
 * incremental patcher (`updateSceneGraphBadges`) can fill it in-place.
 * Returns `''` for nodes with no LOD dimension.
 */
export function lodChipContent(
  node: SceneGraphNode,
  state: LODProgressState | undefined
): { text: string; title: string } | null {
  if (node.kind === 'lod') {
    const count = state?.levelCount ?? node.lodGroupChildCount ?? 0;
    if (count <= 0) return null;
    // No live registry state yet: show "–" instead of guessing level 1.
    if (!state || state.activeLevel === undefined) {
      return {
        text: `L–/${count}`,
        title: `Substitutive LOD group with ${count} levels — active level not yet reported`,
      };
    }
    const active = state.activeLevel + 1;
    const sel = state.selector && state.selector !== 'auto' ? ` (${state.selector})` : '';
    return {
      text: `L${active}/${count}`,
      title: `Active substitutive level ${active} of ${count}${sel} — only this level is rendered`,
    };
  }

  if (node.additiveSublods && node.additiveSublods > 1) {
    const total = state?.total ?? node.additiveSublods;
    // No live loader state: the node's progressive loader isn't streaming
    // (typically an inactive substitutive level). Show "–" rather than a
    // fabricated 0 so "not active" doesn't read as "stalled at zero".
    if (!state) {
      return {
        text: `LOD –/${total}`,
        title: `Additive LOD — ${total} detail levels available; not streaming (level not active)`,
      };
    }
    const loaded = state.loaded ?? 0;
    const refining = state.refining === true;
    const residency =
      state.lastAllResident === false ? ' ◌' : state.lastAllResident === true ? ' ●' : '';
    const spinner = refining ? ' ⏳' : '';
    // Always spell out what the residency dot means — the ● typically
    // appears exactly when refinement has finished, so the explanation
    // must not be gated on `refining`.
    const residencyNote =
      state.lastAllResident === true
        ? ' · ● = fully cache-resident (no network needed)'
        : state.lastAllResident === false
          ? ' · ◌ = streaming from network'
          : '';
    const base = refining
      ? `Additive LOD refining — ${loaded}/${total} levels loaded`
      : `Additive LOD — ${loaded}/${total} levels loaded`;
    return {
      text: `LOD ${loaded}/${total}${residency}${spinner}`,
      title: `${base}${residencyNote}`,
    };
  }

  return null;
}

function renderLodChip(node: SceneGraphNode, state: LODProgressState | undefined): string {
  const content = lodChipContent(node, state);
  if (!content) return '';
  return `<span class="luxar-scene-graph__lod" data-lod-path="${escapeHtml(node.path)}" title="${escapeHtml(content.title)}">${escapeHtml(content.text)}</span>`;
}

/**
 * Role of a node that is a direct child of a substitutive `kind=lod`
 * group: `active` = the level currently rendered, `inactive` = a level
 * present in the file but not rendered right now. `undefined` when the
 * node is not a substitutive level (or the active level is unknown).
 */
interface LevelContext {
  parentPath: string;
  index: number;
  role: 'active' | 'inactive' | undefined;
}

/**
 * Stats-badge content (element / child count + tooltip) for a tree node.
 * Shared by the initial render and the incremental badge patcher so text
 * and tooltip always agree. Per-type visible counts (after nD slicing)
 * are appended symmetrically for points / lines / gsplats when known and
 * different from the dataset count. Returns `null` for nodes with no
 * stats badge (leaves without counts; specialized groups, whose kind
 * badge already carries the child count).
 */
export function nodeStatsContent(node: SceneGraphNode): { text: string; title: string } | null {
  const visibleSuffix = (visible: number | undefined, total: number): string =>
    visible !== undefined && visible !== total
      ? ` (${visible.toLocaleString()} visible after slicing)`
      : '';

  if (node.type === 'points' && node.pointCount !== undefined) {
    return {
      text: formatNumber(node.pointCount),
      title: `${node.pointCount.toLocaleString()} points in this layer${visibleSuffix(node.visiblePointCount, node.pointCount)}`,
    };
  }
  if (node.type === 'lines' && node.segmentCount !== undefined) {
    let title = `${node.segmentCount.toLocaleString()} line segments`;
    if (node.vertexCount !== undefined) {
      title += `, ${node.vertexCount.toLocaleString()} vertices`;
    }
    title += visibleSuffix(node.visibleSegmentCount, node.segmentCount);
    return { text: formatNumber(node.segmentCount), title };
  }
  if (node.type === 'gsplats' && node.splatCount !== undefined) {
    return {
      text: formatNumber(node.splatCount),
      title: `${node.splatCount.toLocaleString()} Gaussian splats${visibleSuffix(node.visibleSplatCount, node.splatCount)}`,
    };
  }
  if (node.type === 'group' && node.children.length > 0 && !node.kind) {
    // Plain groups show child count. Specialized groups (kind=lod /
    // kind=partition) skip it — their kind badge ("K LODs" / "N parts")
    // already carries the same number.
    return {
      text: `${node.children.length}`,
      title: `${node.children.length} child node${node.children.length !== 1 ? 's' : ''}`,
    };
  }
  return null;
}

/** Human-readable suffix for a substitutive level's row tooltip. */
export function levelRoleTitleSuffix(role: 'active' | 'inactive' | undefined): string {
  if (role === 'active') return ' — active substitutive level (currently rendered)';
  if (role === 'inactive') return ' — inactive substitutive level (not rendered)';
  return '';
}

/**
 * Render a single scene graph tree node
 */
function renderSceneGraphNode(
  node: SceneGraphNode,
  expandedNodes: Set<string>,
  depth: number = 0,
  lodStates?: Map<string, LODProgressState>,
  levelCtx?: LevelContext
): string {
  const hasChildren = node.children.length > 0;
  const isExpanded = expandedNodes.has(node.path);

  // Node stats and tooltips (shared with the incremental badge patcher).
  const stats = nodeStatsContent(node);
  const statsText = stats?.text ?? '';
  const statsTooltip = stats?.title ?? '';

  // Build tooltip for the whole node
  const nodeTypeDescriptions: Record<string, string> = {
    scene: 'Root scene node',
    group: 'Container for organizing nodes',
    points: 'Point cloud layer',
    lines: 'Line segments layer',
    gsplats: 'Gaussian splats layer',
    mesh: 'Mesh geometry',
  };
  const kindDescriptions: Partial<Record<string, string>> = {
    lod: 'Substitutive LOD group (one level rendered at a time)',
    partition: 'Partition group (disjoint BSP parts)',
  };
  const baseDesc = node.kind
    ? kindDescriptions[node.kind] || node.kind
    : nodeTypeDescriptions[node.type] || node.type;
  const baseTooltip = `${baseDesc}${node.hasSpatialIndex ? ' (indexed)' : ''}`;
  const nodeTooltip = `${baseTooltip}${levelRoleTitleSuffix(levelCtx?.role)}`;

  const kindBadge = renderKindBadge(node);
  const lodChip = renderLodChip(node, lodStates?.get(node.path));

  // Expand/collapse toggle
  const toggleIcon = hasChildren ? (isExpanded ? '▼' : '▶') : '•';
  const toggleClass = hasChildren
    ? 'luxar-scene-graph__toggle--clickable'
    : 'luxar-scene-graph__toggle--disabled';

  // Indent using CSS custom property for dynamic depth
  const indentStyle = `style="--node-depth: ${depth}; padding-left: calc(var(--node-depth) * 16px);"`;

  // Substitutive-level rows carry data attributes so the incremental
  // patcher can re-mark active/inactive when the LOD selector switches
  // levels between structural rebuilds (`data-base-title` lets it
  // re-derive the tooltip without re-rendering).
  const levelClass =
    levelCtx?.role === 'active'
      ? ' luxar-scene-graph__node-row--active-level'
      : levelCtx?.role === 'inactive'
        ? ' luxar-scene-graph__node-row--inactive-level'
        : '';
  const levelAttrs = levelCtx
    ? ` data-level-of="${escapeHtml(levelCtx.parentPath)}" data-level-index="${levelCtx.index}" data-base-title="${escapeHtml(baseTooltip)}"`
    : '';

  return `
    <div class="luxar-scene-graph__node">
      <div class="luxar-scene-graph__node-row${levelClass}" ${indentStyle} title="${escapeHtml(nodeTooltip)}"${levelAttrs}>
        <!-- Toggle -->
        <span
          class="luxar-scene-graph__toggle ${toggleClass}"
          ${hasChildren ? `data-action="toggleNode" data-node-path="${escapeHtml(node.path)}"` : ''}
          ${hasChildren ? `title="${isExpanded ? 'Collapse' : 'Expand'} ${escapeHtml(node.name)}"` : ''}
        >${toggleIcon}</span>

        <!-- Icon & Name -->
        <span class="luxar-scene-graph__icon">${getSceneGraphIcon(node)}</span>
        <span class="luxar-scene-graph__name ${getNodeTypeColorClass(node.displayType ?? node.type)}">
          ${escapeHtml(node.name)}
        </span>

        <!-- Stats badge -->
        ${
          statsText
            ? `<span class="luxar-scene-graph__badge" data-node-path="${escapeHtml(node.path)}" title="${escapeHtml(statsTooltip)}">${statsText}</span>`
            : ''
        }

        <!-- Kind badge (K LODs / N parts) -->
        ${kindBadge}

        <!-- Live LOD-progress chip (also carries the ⏳ refining marker) -->
        ${lodChip}
      </div>

      <!-- Children (if expanded) -->
      ${
        isExpanded && hasChildren
          ? node.children
              .map((child, i) =>
                renderSceneGraphNode(
                  child,
                  expandedNodes,
                  depth + 1,
                  lodStates,
                  node.kind === 'lod'
                    ? {
                        parentPath: node.path,
                        index: i,
                        role: activeLevelRole(lodStates?.get(node.path), i),
                      }
                    : undefined
                )
              )
              .join('')
          : ''
      }
    </div>
  `;
}

/**
 * Resolve a substitutive level's role from its parent group's live state.
 * Unknown (no state yet) → `undefined`, so rows aren't mis-marked before
 * the first provider poll. Shared by the initial render and the monitor's
 * incremental level-row patcher so both derive the role identically.
 */
export function activeLevelRole(
  state: LODProgressState | undefined,
  index: number
): 'active' | 'inactive' | undefined {
  const active = state?.kind === 'lod' ? state.activeLevel : undefined;
  if (active === undefined) return undefined;
  return index === active ? 'active' : 'inactive';
}

/**
 * Render scene graph tree component
 */
export function renderSceneGraphTree(
  state: SceneGraphState,
  expandedNodes: Set<string>,
  lodStates?: Map<string, LODProgressState>
): string {
  if (!state.root) {
    return `
      <div class="luxar-scene-graph__empty">
        No scene loaded
      </div>
    `;
  }

  // Header with stats. "layers" (node counts), not element counts — the
  // tooltip disambiguates, since "5 gsplats" otherwise reads as 5 splats.
  const headerStats = [
    state.pointsNodes > 0 ? `${state.pointsNodes} points` : null,
    state.linesNodes > 0 ? `${state.linesNodes} lines` : null,
    state.gsplatsNodes > 0 ? `${state.gsplatsNodes} gsplats` : null,
  ]
    .filter(Boolean)
    .join(', ');
  const headerStatsTooltip =
    'Layer (node) counts per geometry type — not element counts. ' +
    'Each level of a substitutive LOD group counts as its own layer, ' +
    'even though only one level renders at a time.';

  // Summarise LOD/partition activity across the scene so the user sees it
  // without expanding the tree: how many substitutive-LOD groups, how many
  // additive nodes streaming/refining, out of how many additive nodes total.
  const lodSummary = summariseLodStates(lodStates, countAdditiveNodes(state.root));
  const lodSummaryTooltip =
    'LOD activity: substitutive = groups that swap between K resolutions of the same data ' +
    '(one rendered at a time) · additive = layers refined by streaming extra detail levels ' +
    'on top of a base ("x/y active" = levels with a live streaming loader out of all additive layers) · ' +
    'partition = groups of disjoint spatial parts (all rendered, frustum-culled per part) · ' +
    'refining = additive layers still loading detail.';

  return `
    <div class="luxar-scene-graph">
      <div class="luxar-scene-graph__header">
        <h4 class="luxar-scene-graph__title" title="Hierarchy of scene nodes (groups, points, lines, gsplats) with per-node element counts">SCENE GRAPH</h4>
        ${headerStats ? `<span class="luxar-scene-graph__stats" title="${escapeHtml(headerStatsTooltip)}">${headerStats}</span>` : ''}
      </div>
      ${lodSummary ? `<div class="luxar-scene-graph__lod-summary" data-field="lod-summary" title="${escapeHtml(lodSummaryTooltip)}">${lodSummary}</div>` : ''}
      <div class="luxar-scene-graph__container">
        ${renderSceneGraphNode(state.root, expandedNodes, 0, lodStates)}
      </div>
    </div>
  `;
}

/**
 * Count additive-LOD layers (nodes with `additiveSublods > 1`) in the tree.
 * Used to reconcile the header summary (which counts *live* streaming
 * loaders) with the tree (which renders a chip slot for every additive
 * layer): "1/5 additive active" instead of a bare "1 additive" that
 * contradicts five visible chips.
 */
export function countAdditiveNodes(root: SceneGraphNode | null | undefined): number {
  if (!root) return 0;
  let count = (root.additiveSublods ?? 0) > 1 ? 1 : 0;
  for (const child of root.children) count += countAdditiveNodes(child);
  return count;
}

/**
 * One-line summary of LOD/partition activity for the scene-graph header,
 * e.g. "2 substitutive · 1/5 additive active · refining 1". Returns `''`
 * when no LOD/partition state is present.
 *
 * @param additiveTotal - total additive layers in the *tree* (see
 *   {@link countAdditiveNodes}). The live count only covers layers with a
 *   streaming loader attached (typically just the active substitutive
 *   level); showing "live/total" keeps the summary consistent with the
 *   number of additive chips visible in the tree.
 */
export function summariseLodStates(
  lodStates?: Map<string, LODProgressState>,
  additiveTotal?: number
): string {
  if (!lodStates || lodStates.size === 0) return '';
  let lod = 0;
  let additive = 0;
  let partition = 0;
  let refining = 0;
  for (const s of lodStates.values()) {
    if (s.kind === 'lod') lod++;
    else if (s.kind === 'additive') {
      additive++;
      if (s.refining) refining++;
    } else if (s.kind === 'partition') partition++;
  }
  const parts: string[] = [];
  // "substitutive" (not the generic "LOD group") so it reads in parallel
  // with "additive" — both are LOD kinds; naming only one "LOD" was the
  // ambiguous wording.
  if (lod > 0) parts.push(`${lod} substitutive`);
  if (additiveTotal !== undefined && additiveTotal > additive) {
    parts.push(`${additive}/${additiveTotal} additive active`);
  } else if (additive > 0) {
    parts.push(`${additive} additive`);
  }
  if (partition > 0) parts.push(`${partition} partition${partition !== 1 ? 's' : ''}`);
  if (refining > 0) parts.push(`refining ${refining}`);
  return parts.join(' · ');
}
