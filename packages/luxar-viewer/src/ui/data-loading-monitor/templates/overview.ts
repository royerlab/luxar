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
import { headlineTooltip, presentHeadlineCounts } from '../headline-counts';
import { formatBytes, formatNumber, getCacheMemoryColorClass, networkSummary } from './format';

/**
 * Map LoaderType identifier to its short display label / item-unit pair.
 * Keeps the loader-list rendering geometry-aware: a lines loader shows
 * "segs" (visibleElements counts visible segments — the queried unit for
 * lines), a gsplats loader shows "splats", points shows "pts", and a mesh
 * loader shows "tris" (the drawn primitive, as everywhere else for mesh).
 */
function loaderDisplay(type: LoaderMetrics['type']): { label: string; unit: string } {
  switch (type) {
    case 'lines-spatial-index':
      return { label: 'lines', unit: 'segs' };
    case 'gsplats-spatial-index':
      return { label: 'gsplats', unit: 'splats' };
    case 'point-spatial-index':
      return { label: 'points', unit: 'pts' };
    case 'mesh-whole-node':
      // "mesh", not "mesh-whole-node": the row's job is to say which LAYER this
      // loader serves. That it is whole-node rather than spatially indexed is
      // visible in the row's own metrics (loads but no queries) and in the
      // compact badge's loading-mode label.
      return { label: 'mesh', unit: 'tris' };
    default:
      // A new `LoaderType` member is a COMPILE error here, not a silent mislabel.
      // `point-spatial-index` used to share this arm, so any future loader type fell
      // through and was rendered as "points / pts" — which is exactly how mesh would
      // have entered: it grew `getMetrics` and joined the union, and without an arm
      // here every mesh row would have read "points / pts".
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
  const statusColorClass =
    metrics.queries > 0 || metrics.loads > 0 ? getColorClass('success') : getColorClass('muted');
  const { label, unit } = loaderDisplay(metrics.type);

  return `
    <div class="luxar-loader-item" title="A loader is the component that streams one layer's data from the zarr store into the viewer — this one serves the ${escapeHtml(label)} layer at ${escapeHtml(path)}">
      <div class="luxar-loader-item__header">
        <span class="luxar-loader-item__path ${statusColorClass}" title="Path of this layer inside the dataset (green = has loaded data or answered queries this session, grey = idle so far)">${escapeHtml(path)}</span>
        <span class="luxar-loader-item__status" title="Geometry type this loader streams (points, lines, gsplats, or mesh)">${escapeHtml(label)}</span>
      </div>
      <div class="luxar-loader-item__metrics">
        <span title="Elements from this layer currently on screen (inside the active nD slice)">${metrics.visibleElements.toLocaleString()} ${escapeHtml(unit)}</span>
        <span title="CPU memory this loader currently holds for loaded data and supporting structures">${formatBytes(metrics.memoryUsed)}</span>
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
  network: NonNullable<CacheMetrics['network']> | undefined
): string {
  const memoryPercent = memory.limit > 0 ? (memory.used / memory.limit) * 100 : 0;
  const networkMetrics = networkSummary(network);

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
          ${networkMetrics.dataLoaded}
        </div>
        <div class="luxar-secondary-metrics__subtitle" data-field="network-detail">
          ${networkMetrics.detail}
        </div>
      </div>
    </div>
  `;
}

/**
 * Warning banner for recorded loader or lazy-LOD failures, with Retry.
 * Rendered at the top of the Overview tab while the injected provider has
 * failed paths; empty string when nothing failed (the common case). The
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
      `These loads failed (network error, missing data, …) and may render incomplete:
${pathList}

Retry re-runs each failed load with the current view state. Retryable loader failures and latched LOD branches are also retried automatically when the connection comes back online.`
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
  // One hero card per geometry type PRESENT in the scene, from the shared
  // headline table — all four types, so a mesh-only scene gets a real count
  // instead of the "LOADING …" placeholder and a mixed scene shows its
  // triangles next to its points/segments/splats.
  const present = presentHeadlineCounts(stats);

  // Card size shrinks as the row fills, and the grid is `auto-fit` rather than
  // a fixed column count: one row while the cards fit, wrapping to 2x2 in a
  // narrow panel. (The old fixed classes also emitted a `--cols-3` that had no
  // CSS rule at all, so three cards silently stacked in one column.)
  const size = present.length === 1 ? 'large' : present.length === 2 ? 'medium' : 'small';
  const gridClass =
    present.length <= 1 ? 'luxar-overview-grid--cols-1' : 'luxar-overview-grid--auto';

  const cards = present.map((c) => {
    const percent = c.total > 0 ? ((c.visible / c.total) * 100).toFixed(1) : '0';
    // Single-card layouts spell out " total"; a filled row has no room for it.
    const suffix = present.length === 1 ? ' total' : '';
    return renderMetricCard(
      c.label,
      formatNumber(c.visible),
      `${percent}% of ${formatNumber(c.total)}${suffix}`,
      countColorClass(c.visible),
      size,
      c.field,
      headlineTooltip(c.noun)
    );
  });

  const primaryMetrics =
    cards.length > 0
      ? `
      <div class="luxar-overview-grid ${gridClass}">
        ${cards.join('\n')}
      </div>
    `
      : `
      <div class="luxar-overview-grid luxar-overview-grid--cols-1">
        ${renderMetricCard('LOADING', '...', 'Waiting for data', getColorClass('muted'), 'large')}
      </div>
    `;

  return `
    <div class="luxar-tab-content--overview">
      <!-- Primary metrics -->
      ${primaryMetrics}

      <div class="luxar-overview-grid luxar-overview-grid--cols-1">
        ${renderMetricCard(
          'DROPPED ELEMENTS',
          formatNumber(stats.droppedElements),
          'renderer capacity clamp — partition oversized nodes',
          stats.droppedElements > 0 ? getColorClass('error') : getColorClass('success'),
          'small',
          'dropped-elements',
          'Elements requested by visible points, lines, and Gaussian-splat nodes but omitted because a node exceeded the GPU element-texture capacity. Split the dataset into multiple nodes or partition it; mesh is not element-texture backed.'
        )}
      </div>

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
