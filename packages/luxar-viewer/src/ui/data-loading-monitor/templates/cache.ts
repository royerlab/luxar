/**
 * Cache-tab templates and shared cache display helpers.
 */

import type { CacheValidationMode } from '../../../cache/types';
import type {
  CacheMetrics,
  CacheStatusBadge,
  GlobalStats,
} from '../../../types/data-monitor-types';
import { escapeHtml } from '../../../utils/escape-html';
import {
  MONITOR_ICONS,
  countColorClass,
  getColorClass,
  renderMetricCard,
  renderProgressBar,
} from './primitives';
import { formatBytes, formatNumber, getCacheMemoryColorClass } from './format';

function formatOptionalCount(value: number | undefined): string {
  return formatNumber(value === undefined ? 0 : value);
}

/**
 * Color class for cache hit-rate metrics (L0/L1/L2). Same threshold
 * shape as the reuse-rate helper in `memory.ts`, but with `>` semantics
 * so an 80% hit rate still shows as warning — caches do not spend much
 * time at exactly 80%, but any drop below the threshold is meaningful.
 */
export function getCacheHitRateColorClass(rate: number): string {
  if (rate > 80) return getColorClass('success');
  if (rate > 50) return getColorClass('warning');
  return getColorClass('error');
}

/**
 * Accesses below this count are the warm-up phase: a low hit rate on a
 * handful of first-touch lookups is expected (the cache HAS to miss
 * before it can hit) and should read as "no signal yet", not as a
 * red-alert failure.
 */
export const CACHE_WARMUP_ACCESSES = 25;

/**
 * Like {@link getCacheHitRateColorClass} but returns the dimmed color
 * while the cache is still warming up (fewer than
 * {@link CACHE_WARMUP_ACCESSES} accesses, including zero). Keeps the
 * initial render of the cache tab consistent with the incremental
 * cache-tab updater. Without this, a freshly loaded session shows
 * hit-rate cards in alarm red when nothing is wrong — the first
 * lookups are unavoidable misses.
 */
export function getCacheHitRateColorClassWithGuard(rate: number, totalAccesses: number): string {
  if (totalAccesses < CACHE_WARMUP_ACCESSES) return getColorClass('dimmed');
  return getCacheHitRateColorClass(rate);
}

/**
 * Cache-section keys accepted by the collapse toggle. Sections start
 * collapsed (compact one-line summary) — the Cache tab holds 4 stacked
 * sections and would otherwise overflow the panel with a scrollbar.
 */
export const CACHE_SECTION_KEYS = ['slice', 'l0', 'l1', 'l2'] as const;
export type CacheSectionKey = (typeof CACHE_SECTION_KEYS)[number];

/**
 * Reusable cache section component (reduces duplication between L1/L2).
 * Collapsible: the header always carries a compact inline summary
 * (hidden by CSS while expanded) that mirrors every metric — same
 * value/subtitle text and the SAME `data-field` keys as the full cards,
 * so the per-tick patcher in `tabs/cache.ts` updates both views with
 * one pass and no state is lost by collapsing.
 *
 * The title renders in two variants: `title` (full, e.g. "L0 DECOMPRESSED
 * CACHE") shown while expanded, and `shortTitle` (e.g. "L0 cache") shown in
 * the collapsed one-line summary. The short titles are near-uniform width,
 * which lets the collapsed summaries grid-align their value columns across
 * all four cache rows; the didactic titleTooltip carries the full story in
 * both states.
 * @param metrics - Each metric can have a colorClass for CSS class-based coloring
 */
function renderCacheSection(
  title: string,
  shortTitle: string,
  titleTooltip: string | undefined,
  clearAction: string,
  clearTooltip: string,
  sectionKey: CacheSectionKey,
  collapsed: boolean,
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

  const summaryHtml = metrics
    .map((metric) => {
      const fieldAttr = metric.dataField ? ` data-field="${metric.dataField}"` : '';
      const subFieldAttr = metric.dataField ? ` data-field="${metric.dataField}-sub"` : '';
      // The one-line summary only has room for label + value per metric;
      // the sole subtitle carried along is the hit-rate hits·miss split
      // (live data not visible anywhere else while collapsed). The other
      // subtitles (entry counts, "LRU removed", error breakdowns) stay on
      // the expanded cards and in each item's hover tooltip.
      const withSub = metric.dataField?.includes('hitrate');
      const subHtml = withSub
        ? `<span class="luxar-cache-section__summary-sub"${subFieldAttr}>${metric.subtitle}</span>`
        : '';
      return `
        <span class="luxar-cache-section__summary-item"${metric.tooltip ? ` title="${escapeHtml(metric.tooltip)}"` : ''}>
          <span class="luxar-cache-section__summary-label">${metric.label}</span>
          <span class="luxar-cache-section__summary-value ${metric.colorClass || ''}"${fieldAttr}>${metric.value}</span>
          ${subHtml}
        </span>
      `;
    })
    .join('');

  return `
    <div class="luxar-cache-section${collapsed ? ' luxar-cache-section--collapsed' : ''}" data-section="${sectionKey}">
      <div class="luxar-cache-section__header" data-action="toggleCacheSection" data-section-key="${sectionKey}" role="button" tabindex="0" aria-expanded="${!collapsed}" title="Click to ${collapsed ? 'expand' : 'collapse'} this section">
        <span class="luxar-cache-section__chevron" aria-hidden="true">▾</span>
        <span class="luxar-cache-section__title"${titleTooltip ? ` title="${escapeHtml(titleTooltip)}"` : ''}><span class="luxar-cache-section__title-full">${title}</span><span class="luxar-cache-section__title-short">${shortTitle}</span></span>
        <span class="luxar-cache-section__summary">${summaryHtml}</span>
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
    'Multi-level caching is active. Decoded slices (S-cache) and downloaded chunks (L0 decoded + ' +
    'L1 raw in memory, L2 on disk in browser private storage) are retained, so re-slicing and ' +
    'revisits are served locally instead of re-downloading. Nothing to do — this is the healthy state.',
  'no-cache':
    'Caching is turned off for this session by the ?noCache URL parameter: every chunk is ' +
    'fetched from the network each time it is needed and nothing persists across reloads. ' +
    'Remove ?noCache from the URL to re-enable caching.',
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
export function formatValidationMode(mode: CacheValidationMode | undefined): string {
  switch (mode) {
    case 'content-hash':
      return 'Content Hash';
    case 'zattrs-hash':
      return 'Metadata Hash';
    case 'archive-etag':
      return 'Archive ETag';
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
export function validationModeTooltip(mode: CacheValidationMode | undefined): string {
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
    case 'archive-etag':
      return (
        'Archive-ETag validation: this dataset is a single zipped store (.zarr.zip), whose root ' +
        'metadata lives INSIDE the archive and so cannot be re-fetched on its own. The viewer ' +
        "instead asks the server for the archive's ETag (or its modification time and size) and " +
        'compares that with the one stored next to the disk cache. This covers the whole store at ' +
        'once rather than one document, so any change to the archive clears every cache tier.'
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
        'No validation: the dataset root metadata could not be fetched (offline or a store with ' +
        'no root .zattrs), no cache TTL is configured, and no cached hash was available to fall ' +
        'back on. Cached chunks are served indefinitely, so if the file changes on the server you ' +
        'will keep seeing the old data until you press Clear All to force a fresh download.'
      );
    default:
      return (
        'How the cache decides whether its stored chunks still match the dataset on the server. ' +
        'Not determined yet — the freshness check runs right after the dataset loads.'
      );
  }
}

/**
 * Didactic hover explanation for the "Last Validated" / "Cached Since"
 * timestamp, mode-aware because the timestamp means different things:
 * under the source-validation modes it is a real confirmation that updates on each
 * successful online check, while under ttl/none it is a fixed known-good
 * baseline (when the cache was established) that does NOT advance on
 * repeat offline checks.
 */
export function lastValidatedTooltip(mode: CacheValidationMode | undefined): string {
  const base =
    'At load time the viewer checks the dataset root metadata, or the archive itself for a ' +
    'zipped store. For the content-hash, .zattrs-hash, and archive-etag modes this timestamp ' +
    'updates on each successful check; for ttl/none it is a fixed baseline that does not ' +
    'advance on repeat checks. ';
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
    case 'archive-etag':
      return (
        base +
        "At this moment the archive's ETag (or modification time and size) was compared against " +
        'the server and the cache was confirmed current (or cleared if it did not match). ' +
        '"Never" = no check has completed yet, e.g. offline.'
      );
    case 'ttl':
      return (
        base +
        'This timestamp marks when the cache baseline was established and starts the TTL ' +
        'countdown: once the cache is older than the configured maximum age it is discarded ' +
        'and re-downloaded. Repeat offline checks do NOT push it forward. "Never" = no check ' +
        'has completed yet, e.g. offline.'
      );
    case 'none':
      return (
        base +
        'Careful: with validation "None" this marks when the cache baseline was established — ' +
        'it found no content_hash to compare, so it does NOT confirm the cached data matches ' +
        'the server and does NOT advance on repeat checks. "Never" = no check has completed ' +
        'yet, e.g. offline.'
      );
    default:
      return base + '"Never" = no check has completed yet, e.g. offline.';
  }
}

/**
 * Row label for the freshness timestamp, mode-aware to match the
 * timestamp's actual meaning: only the source-validation modes truly validate the
 * cache against the server on each check ("Last Validated"). Under
 * ttl/none the timestamp is a fixed baseline marking when the cache was
 * established, not a per-check event, so "Cached Since" is the honest label.
 */
export function lastValidatedLabel(mode: CacheValidationMode | undefined): string {
  // All three validation modes genuinely validate the cache against the server on each
  // successful check; ttl/none record a fixed baseline that does not advance.
  return mode === 'content-hash' || mode === 'zattrs-hash' || mode === 'archive-etag'
    ? 'Last Validated'
    : 'Cached Since';
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
 * Template for cache tab content with L0/L1/L2 breakdown.
 * `collapsedSections` holds the section keys currently collapsed to
 * their compact one-line summary (default: all — the compact state is
 * the resting state so the tab fits without a scrollbar).
 */
export function renderCacheContent(
  _stats: GlobalStats,
  cacheMetrics: CacheMetrics,
  collapsedSections: ReadonlySet<string> = new Set(CACHE_SECTION_KEYS)
): string {
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
        message = 'Caching disabled by ?noCache';
        hint = 'Remove ?noCache from URL to enable';
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
          <div class="luxar-cache-disabled__icon">${MONITOR_ICONS.blocked}</div>
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

  // SliceCache ("S-cache") hit rate calculation (if available)
  const sliceTotal = cacheMetrics.slice ? cacheMetrics.slice.hits + cacheMetrics.slice.misses : 0;
  const sliceHitRate = sliceTotal > 0 ? (cacheMetrics.slice!.hits / sliceTotal) * 100 : 0;
  const sliceHitRateColorClass = getCacheHitRateColorClassWithGuard(sliceHitRate, sliceTotal);

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
      <!-- SliceCache ("S-cache") — per-(node,view) decoded-slice cache, above L0 -->
      ${
        cacheMetrics.slice
          ? renderCacheSection(
              'SLICE CACHE (S-CACHE)',
              'S-cache',
              'Caches the fully decoded geometry of a whole slice (a node at one view: slice position + displayed dims), keyed per node. Revisiting a slice — e.g. scrubbing back to a timepoint — restores it instantly, skipping the query + fetch + decode entirely (only the cheap nD→3D projection re-runs). Cleared on dataset change',
              'clearSlice',
              'Empty the SliceCache. Harmless: the next visit to each slice re-loads and re-decodes it from L0/L1/L2/network',
              'slice',
              collapsedSections.has('slice'),
              [
                {
                  label: 'SIZE',
                  value: formatBytes(cacheMetrics.slice.size),
                  subtitle: `${cacheMetrics.slice.count} slices`,
                  tooltip:
                    'Memory held by cached decoded slices, and how many slices that is. Bounded by an LRU byte budget — see EVICTIONS',
                  colorClass: getColorClass('primary'),
                  dataField: 's-size',
                },
                {
                  label: 'HIT RATE',
                  value: `${sliceHitRate.toFixed(1)}%`,
                  subtitle: `${formatNumber(cacheMetrics.slice.hits)} hits · ${formatNumber(cacheMetrics.slice.misses)} miss`,
                  tooltip:
                    'Share of slice revisits served instantly from the cache. A hit skips the whole load+decode; a miss loads the slice normally. Rises as you revisit / oscillate over slices (e.g. scrubbing back and forth in time)',
                  colorClass: sliceHitRateColorClass,
                  dataField: 's-hitrate',
                },
                {
                  label: 'EVICTIONS',
                  value: formatNumber(cacheMetrics.slice.evictions),
                  subtitle: 'LRU removed',
                  tooltip:
                    'Slices pushed out of the cache (least-recently-used first) once it hit its byte budget. Evicted slices are simply re-loaded on next visit. Steady growth means more distinct slices were visited than the budget holds',
                  colorClass:
                    cacheMetrics.slice.evictions > 0
                      ? getColorClass('warning')
                      : getColorClass('dimmed'),
                  dataField: 's-evictions',
                },
              ]
            )
          : ''
      }

      <!-- L0 Decompressed Chunk Cache Section (fastest layer - avoids Blosc decompression) -->
      ${
        cacheMetrics.l0
          ? renderCacheSection(
              'L0 DECOMPRESSED CACHE',
              'L0 cache',
              'The fastest cache tier. Data chunks arrive compressed and must be decoded (~2ms each) before use; L0 keeps the already-decoded arrays in memory so repeat reads skip both the download AND the decode. Lookups try L0 first, then fall through L1 (memory) → L2 (disk) → network',
              'clearL0',
              'Empty the L0 decoded-chunk cache. Harmless: chunks are still in L1/L2 and will simply be re-decoded (~2ms each) on next access',
              'l0',
              collapsedSections.has('l0'),
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
        'L1 cache',
        'The in-memory tier for raw (still-compressed) chunks and metadata. Serves L0 misses from RAM with no disk or network round-trip. Cleared when the page closes — the persistent copy lives in L2. Lookup order: L0 → L1 → L2 → network',
        'clearL1',
        'Empty the L1 in-memory cache. Harmless: chunks still cached on disk (L2) are re-read from there; only uncached data goes back to the network',
        'l1',
        collapsedSections.has('l1'),
        [
          {
            label: 'SIZE',
            value: formatBytes(cacheMetrics.l1!.size),
            subtitle: `${cacheMetrics.l1!.count} entries`,
            tooltip:
              'RAM currently held by raw chunks + metadata in L1, and the number of entries. Bounded by an LRU limit — see EVICTIONS',
            colorClass: getColorClass('primary'),
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
          'L2 cache',
          "The persistent disk tier, stored in the browser's Origin Private File System (private storage on your machine — never uploaded anywhere). Survives page reloads and browser restarts, so a revisited dataset loads from disk instead of the network. Lookup order: L0 → L1 → L2 → network",
          'clearL2',
          'Delete the on-disk (L2) cache for this dataset. Anything not held in memory will be re-downloaded from the server — use this to reclaim disk space or force a fresh copy',
          'l2',
          collapsedSections.has('l2'),
          [
            {
              label: 'SIZE',
              value: formatBytes(cacheMetrics.l2!.size),
              subtitle: `${cacheMetrics.l2!.count} entries`,
              tooltip:
                "Disk space used by cached chunks in the browser's private storage, and the number of entries. Persists across sessions; counts against the browser storage quota (see the quota-constrained badge if it fills up)",
              colorClass: getColorClass('primary'),
              dataField: 'l2-size',
            },
            {
              label: 'HIT RATE',
              value: l2Total > 0 ? `${l2HitRate.toFixed(1)}%` : '—',
              subtitle: `${formatNumber(cacheMetrics.l2!.reads)} hits · ${formatNumber(cacheMetrics.l2!.misses)} miss · ${formatOptionalCount(cacheMetrics.l2!.canceledReads)} canceled`,
              tooltip: `Of the requests that missed the memory caches and fell through to disk, the share found there (${cacheMetrics.l2!.reads.toLocaleString()} of ${l2Total.toLocaleString()}). An L2 miss is the only case that costs a network download. "—" = nothing has fallen through to L2 yet`,
              colorClass: l2HitRateColorClass,
              dataField: 'l2-hitrate',
            },
            {
              label: 'I/O',
              value: `${formatNumber(cacheMetrics.l2!.reads)} reads`,
              subtitle: `${formatNumber(cacheMetrics.l2!.writes)} writes · ${formatOptionalCount(cacheMetrics.l2!.activeReads)} active · ${formatOptionalCount(cacheMetrics.l2!.queuedReads)} queued`,
              tooltip:
                'Disk traffic and live read backpressure: reads = chunks served from the on-disk cache; writes = freshly downloaded chunks saved to disk; active/queued show the page-wide OPFS read gate',
              colorClass: countColorClass(cacheMetrics.l2!.reads + cacheMetrics.l2!.writes),
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
          <span class="luxar-cache-health__label" title="The strategy used to detect a dataset that changed on the server: Content Hash, Metadata Hash, Archive ETag, TTL (cached data expires after a configured age), or None (no change detection)">Validation</span>
          <span class="luxar-cache-health__value" data-field="cache-health-mode" title="${escapeHtml(validationModeTooltip(cacheMetrics.health?.validationMode))}">
            ${formatValidationMode(cacheMetrics.health?.validationMode)}
          </span>
        </div>
        <div class="luxar-cache-health__row">
          <span class="luxar-cache-health__label" data-field="cache-health-validated-label" title="When the cache's freshness was last established — for source-validated datasets the last successful server check, for TTL/none the known-good baseline; hover the value for details under the current mode">${lastValidatedLabel(cacheMetrics.health?.validationMode)}</span>
          <span class="luxar-cache-health__value" data-field="cache-health-validated" title="${escapeHtml(lastValidatedTooltip(cacheMetrics.health?.validationMode))}">
            ${formatLastValidated(cacheMetrics.health?.lastValidatedAt)}
          </span>
        </div>
      </div>

      <!-- Combined Stats + Clear All -->
      <div class="luxar-cache-total">
        <div class="luxar-cache-total__header">
          <span class="luxar-cache-total__label" title="Total space used by cached data across every tier: S-cache (decoded slices) + L0 (decoded chunks, memory) + L1 (raw, memory) + L2 (disk). The bar below shows usage against the configured limit">TOTAL</span>
          <button data-action="clearAll" class="luxar-cache-section__clear-btn" title="Delete everything in every cache tier (S-cache + L0 + L1 + L2 disk). The scene stays loaded, but data needed afterwards is re-downloaded from the server. Use this to force a fresh copy of a dataset that may have changed (especially with Validation: None), or to reclaim disk space">Clear All</button>
        </div>
        <div class="luxar-cache-total__value" data-field="cache-total" title="${cacheMetrics.totalCacheMemory.toLocaleString()} bytes total cached">
          ${formatBytes(cacheMetrics.totalCacheMemory)}
        </div>
        ${renderProgressBar(cacheMetrics.memoryPercent, getCacheMemoryColorClass(cacheMetrics.memoryPercent), cacheMetrics.memoryLimit > 0 ? `${cacheMetrics.memoryPercent.toFixed(0)}% of ${formatBytes(cacheMetrics.memoryLimit)} limit` : 'no memory limit configured', 6)}
        ${
          cacheMetrics.effectiveDemandHitRate !== undefined
            ? `<div class="luxar-cache-total__demand" title="The bottom line for caching: of all data requests made by the renderer, the share answered by ANY cache tier (S-cache, L0, L1, or L2) instead of the network. 100% = fully local, no downloads; low values right after first load are normal — the caches have to be filled once before they can hit">
                 <span class="luxar-cache-total__demand-label">EFFECTIVE HIT RATE</span>
                 <span class="luxar-cache-total__demand-value" data-field="cache-effective-hitrate">${(cacheMetrics.effectiveDemandHitRate * 100).toFixed(1)}%</span>
               </div>`
            : ''
        }
      </div>
    </div>
  `;
}
