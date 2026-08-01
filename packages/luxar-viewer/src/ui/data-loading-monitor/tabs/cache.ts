/**
 * Incremental Cache-tab updater. The Cache tab paints the SliceCache
 * ("S-cache") + L0/L1/L2 stats + total cache memory bar + eviction
 * counts using the data-field selector pattern. The renderer (in
 * `../templates.ts`) paints the static structure once; this module
 * handles per-tick value patching.
 *
 * Inputs are the container element and the pre-aggregated
 * CacheMetrics (built by `../metrics/cache.ts`). Returns `true` when
 * the container had the expected structure; `false` tells the monitor
 * it needs a full rebuild.
 */

import type { CacheMetrics } from '../../../types/data-monitor-types';
import {
  formatNumber as templateFormatNumber,
  formatBytes as templateFormatBytes,
  getColorClass,
  getCacheHitRateColorClassWithGuard,
  getCacheMemoryColorClass,
  countColorClass,
  renderCacheStatusBadges,
  formatValidationMode,
  formatLastValidated,
  validationModeTooltip,
  lastValidatedTooltip,
  lastValidatedLabel,
  l2ErrorTotal,
} from '../templates';
import { patchField, updateColorClass, updateColorClassByField } from './dom-helpers';

/**
 * Patch every value cell on the Cache tab in place. Idempotent
 * within a single tick (each call rewrites all fields); returns
 * `false` if the tab structure isn't present in the container yet.
 */
export function updateCacheTab(container: HTMLElement | null, cacheMetrics: CacheMetrics): boolean {
  if (!container) return false;
  // Structure validation: cache-total is always present in the
  // L1/L2 view; if missing the structure hasn't been rendered yet.
  if (!container.querySelector('[data-field="cache-total"]')) return false;

  // L0 stats
  if (cacheMetrics.l0) {
    const l0Total = cacheMetrics.l0.hits + cacheMetrics.l0.misses;
    const l0HitRate = l0Total > 0 ? (cacheMetrics.l0.hits / l0Total) * 100 : 0;

    patchField(container, 'l0-size', templateFormatBytes(cacheMetrics.l0.size));
    patchField(container, 'l0-size-sub', `${cacheMetrics.l0.count} chunks`);
    patchField(container, 'l0-hitrate', `${l0HitRate.toFixed(1)}%`);
    patchField(
      container,
      'l0-hitrate-sub',
      `${templateFormatNumber(cacheMetrics.l0.hits)} hits · ${templateFormatNumber(cacheMetrics.l0.misses)} miss`
    );
    patchField(container, 'l0-evictions', templateFormatNumber(cacheMetrics.l0.evictions));

    // Same warm-up-aware coloring as the initial render (dimmed until
    // the cache has seen enough accesses to have a meaningful rate).
    updateColorClassByField(
      container,
      'l0-hitrate',
      getCacheHitRateColorClassWithGuard(l0HitRate, l0Total)
    );
    updateColorClassByField(
      container,
      'l0-evictions',
      cacheMetrics.l0.evictions > 0 ? getColorClass('warning') : getColorClass('dimmed')
    );
  }

  // SliceCache ("S-cache") stats
  if (cacheMetrics.slice) {
    const sliceTotal = cacheMetrics.slice.hits + cacheMetrics.slice.misses;
    const sliceHitRate = sliceTotal > 0 ? (cacheMetrics.slice.hits / sliceTotal) * 100 : 0;

    patchField(container, 's-size', templateFormatBytes(cacheMetrics.slice.size));
    patchField(container, 's-size-sub', `${cacheMetrics.slice.count} slices`);
    patchField(container, 's-hitrate', `${sliceHitRate.toFixed(1)}%`);
    patchField(
      container,
      's-hitrate-sub',
      `${templateFormatNumber(cacheMetrics.slice.hits)} hits · ${templateFormatNumber(cacheMetrics.slice.misses)} miss`
    );
    patchField(container, 's-evictions', templateFormatNumber(cacheMetrics.slice.evictions));

    updateColorClassByField(
      container,
      's-hitrate',
      getCacheHitRateColorClassWithGuard(sliceHitRate, sliceTotal)
    );
    updateColorClassByField(
      container,
      's-evictions',
      cacheMetrics.slice.evictions > 0 ? getColorClass('warning') : getColorClass('dimmed')
    );
  }

  // L1 stats
  if (cacheMetrics.l1) {
    const l1Total = cacheMetrics.l1.hits + cacheMetrics.l1.misses;
    const l1HitRate = l1Total > 0 ? (cacheMetrics.l1.hits / l1Total) * 100 : 0;

    patchField(container, 'l1-size', templateFormatBytes(cacheMetrics.l1.size));
    patchField(container, 'l1-size-sub', `${cacheMetrics.l1.count} entries`);
    patchField(container, 'l1-hitrate', `${l1HitRate.toFixed(1)}%`);
    patchField(
      container,
      'l1-hitrate-sub',
      `${templateFormatNumber(cacheMetrics.l1.hits)} hits · ${templateFormatNumber(cacheMetrics.l1.misses)} miss`
    );
    patchField(container, 'l1-evictions', templateFormatNumber(cacheMetrics.l1.evictions));

    updateColorClassByField(
      container,
      'l1-hitrate',
      getCacheHitRateColorClassWithGuard(l1HitRate, l1Total)
    );
    updateColorClassByField(
      container,
      'l1-evictions',
      cacheMetrics.l1.evictions > 0 ? getColorClass('warning') : getColorClass('dimmed')
    );
  }

  // L2 stats. Note: L2 "reads" already means successful gets (= hits);
  // L2 hit rate is reads / (reads + misses). R2: the initial template
  // rendered the hit-rate cell but the per-tick patcher never updated
  // it, so the value froze after first render — patch parity with L0/L1
  // here.
  if (cacheMetrics.l2) {
    const l2Total = cacheMetrics.l2.reads + cacheMetrics.l2.misses;
    const l2HitRate = l2Total > 0 ? (cacheMetrics.l2.reads / l2Total) * 100 : 0;

    patchField(container, 'l2-size', templateFormatBytes(cacheMetrics.l2.size));
    patchField(container, 'l2-size-sub', `${cacheMetrics.l2.count} entries`);
    patchField(container, 'l2-hitrate', l2Total > 0 ? `${l2HitRate.toFixed(1)}%` : '—');
    patchField(
      container,
      'l2-hitrate-sub',
      `${templateFormatNumber(cacheMetrics.l2.reads)} hits · ${templateFormatNumber(cacheMetrics.l2.misses)} miss`
    );
    patchField(container, 'l2-io', `${templateFormatNumber(cacheMetrics.l2.reads)} reads`);
    patchField(container, 'l2-io-sub', `${templateFormatNumber(cacheMetrics.l2.writes)} writes`);
    // Idle-dimming parity with the initial render: bright once disk
    // traffic exists, dimmed while the tier is untouched.
    updateColorClassByField(
      container,
      'l2-io',
      countColorClass(cacheMetrics.l2.reads + cacheMetrics.l2.writes)
    );

    updateColorClassByField(
      container,
      'l2-hitrate',
      getCacheHitRateColorClassWithGuard(l2HitRate, l2Total)
    );
  }

  // R3: status pill row. Avoid full innerHTML replace when the badge
  // set is stable across ticks — cheap join('|') signature compare.
  const statusRow = container.querySelector(
    '[data-field="cache-status-row"]'
  ) as HTMLElement | null;
  if (statusRow) {
    const badges = cacheMetrics.status ?? [];
    const signature = badges.join('|');
    if (statusRow.dataset.signature !== signature) {
      statusRow.dataset.signature = signature;
      statusRow.innerHTML = renderCacheStatusBadges(badges);
    }
  }

  // R3: Cache Health — validation mode + last-validated timestamp.
  // Tooltips are mode-specific (e.g. under mode 'none' the timestamp
  // records a check attempt, not a confirmation), so they are patched
  // alongside the values: the mode changes at runtime ('—' → actual
  // mode once the first validation completes) and a stale tooltip
  // would then describe the wrong mode.
  const mode = cacheMetrics.health?.validationMode;
  patchField(container, 'cache-health-mode', formatValidationMode(mode));
  patchField(
    container,
    'cache-health-validated',
    formatLastValidated(cacheMetrics.health?.lastValidatedAt)
  );
  // The row label is mode-aware too ("Last Validated" under content-hash
  // AND zattrs-hash; "Cached Since" otherwise) — see lastValidatedLabel.
  patchField(container, 'cache-health-validated-label', lastValidatedLabel(mode));
  const modeEl = container.querySelector('[data-field="cache-health-mode"]');
  if (modeEl) modeEl.setAttribute('title', validationModeTooltip(mode));
  const validatedEl = container.querySelector('[data-field="cache-health-validated"]');
  if (validatedEl) validatedEl.setAttribute('title', lastValidatedTooltip(mode));

  // R3: L2 error counters card.
  if (cacheMetrics.l2) {
    const errTotal = l2ErrorTotal(cacheMetrics.l2);
    patchField(container, 'l2-errors', errTotal > 0 ? templateFormatNumber(errTotal) : '0');
    patchField(
      container,
      'l2-errors-sub',
      errTotal > 0
        ? `${templateFormatNumber(cacheMetrics.l2.quotaWriteSkipped ?? 0)} quota · ${templateFormatNumber(cacheMetrics.l2.writeFailures ?? 0)} write · ${templateFormatNumber(cacheMetrics.l2.corruptedEntries ?? 0)} corrupt`
        : 'no errors'
    );
    updateColorClassByField(
      container,
      'l2-errors',
      errTotal > 0 ? getColorClass('error') : getColorClass('dimmed')
    );
  }

  // Total
  patchField(container, 'cache-total', templateFormatBytes(cacheMetrics.totalCacheMemory));

  // Effective demand hit-rate. The label is static in the template;
  // only the value span is patched. The field is absent in the
  // rendered template when `effectiveDemandHitRate` is undefined;
  // `patchField` no-ops when the selector misses.
  if (cacheMetrics.effectiveDemandHitRate !== undefined) {
    patchField(
      container,
      'cache-effective-hitrate',
      `${(cacheMetrics.effectiveDemandHitRate * 100).toFixed(1)}%`
    );
  }

  // Total progress bar
  const barFill = container.querySelector(
    '.luxar-cache-total .luxar-progress-bar__fill'
  ) as HTMLElement | null;
  if (barFill) {
    barFill.style.width = `${Math.min(100, cacheMetrics.memoryPercent)}%`;
    updateColorClass(barFill, getCacheMemoryColorClass(cacheMetrics.memoryPercent));
  }
  const barLabel = container.querySelector('.luxar-cache-total .luxar-progress-bar__label');
  if (barLabel) {
    barLabel.textContent =
      cacheMetrics.memoryLimit > 0
        ? `${cacheMetrics.memoryPercent.toFixed(0)}% of ${templateFormatBytes(cacheMetrics.memoryLimit)} limit`
        : 'no memory limit configured';
  }

  return true;
}
