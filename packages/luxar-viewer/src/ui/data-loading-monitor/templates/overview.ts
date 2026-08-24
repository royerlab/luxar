/**
 * Overview-tab templates for loader and aggregate metrics.
 */

import type { CacheMetrics, GlobalStats, LoaderMetrics } from '../../../types/data-monitor-types';
import { escapeHtml } from '../../../utils/escape-html';
import {
  MONITOR_ICONS,
  countColorClass,
  getColorClass,
  renderMetricCard,
  renderProgressBar,
} from './primitives';
import { formatBytes, formatNumber, getCacheMemoryColorClass } from './format';

/**
 * Map LoaderType identifier to its short display label / item-unit pair.
 * Keeps the loader-list rendering geometry-aware: a lines loader shows
 * "segs" (visibleElements counts visible segments — the queried unit for
 * lines), a gsplats loader shows "splats", points shows "pts".
 */
function loaderDisplay(type: LoaderMetrics['type']): { label: string; unit: string } {
  switch (type) {
    case 'lines-spatial-index':
      return { label: 'lines', unit: 'segs' };
    case 'gsplats-spatial-index':
      return { label: 'gsplats', unit: 'splats' };
    case 'point-spatial-index':
      return { label: 'points', unit: 'pts' };
    default:
      // A new `LoaderType` member is a COMPILE error here, not a silent mislabel.
      // `point-spatial-index` used to share this arm, so any future loader type fell
      // through and was rendered as "points / pts" — and mesh is the concrete case
      // waiting to hit it: `MeshWholeNodeLoader` has no `getMetrics` yet, so it is absent from
      // the union today, and whichever phase adds mesh metrics needs a label here.
      // Runtime behaviour is unchanged (an unknown type still renders as points) —
      // only the silence is gone.
      void (type satisfies never);
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
        <span title="Elements from this layer currently on screen (inside the active nD slice)">${metrics.visibleElements.toLocaleString()} ${escapeHtml(unit)}</span>
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
 * Warning banner for recorded load failures, with a Retry action. Rendered
 * at the top of the Overview tab while `SceneLoader.getFailedLoaders()` is
 * non-empty; empty string when nothing failed (the common case). The
 * button carries `data-action="retryFailedLoads"` for the monitor's event
 * delegation and is disabled while a retry batch is in flight.
 */
export function renderFailedLoadsBanner(
  failedPaths: readonly string[],
  retryInFlight: boolean
): string {
  if (failedPaths.length === 0) return '';
  const n = failedPaths.length;
  const pathList = failedPaths.join('\n');
  return `
    <div class="luxar-failed-loads" title="${escapeHtml(
      `These nodes failed to load (network error, missing data, …) and may render incomplete:
${pathList}

Retry re-runs each failed load with the current view state. Failed loads are also retried automatically when the connection comes back online.`
    )}">
      <span class="luxar-failed-loads__label">${MONITOR_ICONS.alert} ${n} failed load${n === 1 ? '' : 's'}</span>
      <button data-action="retryFailedLoads" class="luxar-cache-section__clear-btn" ${
        retryInFlight ? 'disabled' : ''
      } title="Re-run every failed load with the current view state">${retryInFlight ? 'Retrying…' : 'Retry'}</button>
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
  let primaryMetrics: string;
  if (showAll) {
    // Show all three (points, lines, gsplats)
    primaryMetrics = `
      <div class="luxar-overview-grid luxar-overview-grid--cols-3">
        ${renderMetricCard(
          'VISIBLE POINTS',
          formatNumber(stats.visiblePoints),
          `${visiblePointsPercent}% of ${formatNumber(stats.datasetSize)}`,
          countColorClass(stats.visiblePoints),
          'small',
          'visible-points',
          'How many points are on screen right now versus how many the whole dataset holds. The two differ because only data inside the current nD slice is shown, and level-of-detail (LOD) streaming may not have loaded full resolution yet'
        )}
        ${renderMetricCard(
          'VISIBLE LINES',
          formatNumber(stats.visibleSegments),
          `${visibleSegmentsPercent}% of ${formatNumber(stats.datasetSegments)}`,
          countColorClass(stats.visibleSegments),
          'small',
          'visible-lines',
          'How many line segments are on screen right now versus how many the whole dataset holds. The two differ because only data inside the current nD slice is shown, and level-of-detail (LOD) streaming may not have loaded full resolution yet'
        )}
        ${renderMetricCard(
          'VISIBLE SPLATS',
          formatNumber(stats.visibleSplats),
          `${visibleSplatsPercent}% of ${formatNumber(stats.datasetSplats)}`,
          countColorClass(stats.visibleSplats),
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
          countColorClass(stats.visiblePoints),
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
          countColorClass(stats.visibleSegments),
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
          countColorClass(stats.visibleSplats),
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
          countColorClass(stats.visiblePoints),
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
          countColorClass(stats.visibleSegments),
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
          countColorClass(stats.visibleSplats),
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
