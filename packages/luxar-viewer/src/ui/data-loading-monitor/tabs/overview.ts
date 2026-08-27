/**
 * Incremental Overview-tab updater. The renderer paints the stable card,
 * progress-bar, and scene-graph structure once; this module patches the
 * per-tick values and invokes the caller-owned badge refresh only after the
 * expected structure is present.
 */

import type { CacheMetrics, GlobalStats } from '../../../types/data-monitor-types';
import {
  formatBytes,
  formatNumber,
  getCacheMemoryColorClass,
  networkSummary,
} from '../templates/format';
import { countColorClass, getColorClass } from '../templates/primitives';
import { patchField, updateColorClass } from './dom-helpers';
import { presentHeadlineCounts } from '../headline-counts';

export function updateOverviewTab(
  container: HTMLElement | null,
  stats: GlobalStats,
  cacheMetrics: CacheMetrics,
  badges: () => void
): boolean {
  if (!container) return false;
  // Same shared table the renderer painted from, so the patcher can never
  // disagree with it about which types are present or where their fields live.
  const present = presentHeadlineCounts(stats);
  // EVERY present type must already have a card, not just one of them: a type
  // that appears mid-session (a mesh node finishing its load after the first
  // paint) has no card yet, and patching it would silently no-op forever.
  // Reporting "not patchable" hands the tick to the caller's full rebuild,
  // which paints the new card. `present.length === 0` (nothing loaded) also
  // lands here, exactly as the old any-field probe did.
  const painted =
    present.length > 0 &&
    present.every((c) => container.querySelector(`[data-field="${c.field}"]`) !== null);
  if (!painted) return false;

  // Single-type layouts use " total" suffix in subtitle (matches template rendering)
  const suffix = present.length === 1 ? ' total' : '';

  // Count cards: value text plus the state color (neutral with data,
  // dimmed at zero — matches `countColorClass` in the initial render,
  // so a card doesn't stay dimmed after points scroll into view).
  for (const c of present) {
    const percent = c.total > 0 ? ((c.visible / c.total) * 100).toFixed(1) : '0';
    patchField(container, c.field, formatNumber(c.visible));
    patchField(container, `${c.field}-sub`, `${percent}% of ${formatNumber(c.total)}${suffix}`);
    const element = container.querySelector(`[data-field="${c.field}"]`);
    if (element) updateColorClass(element as HTMLElement, countColorClass(c.visible));
  }

  // Dropped-element card: its own field, patched unconditionally (it is
  // rendered whenever the overview grid is painted, independent of which
  // geometry types are present).
  const droppedElement = container.querySelector('[data-field="dropped-elements"]');
  patchField(container, 'dropped-elements', formatNumber(stats.droppedElements));
  if (droppedElement) {
    updateColorClass(
      droppedElement as HTMLElement,
      stats.droppedElements > 0 ? getColorClass('error') : getColorClass('success')
    );
  }

  patchField(container, 'memory-used', formatBytes(cacheMetrics.totalCacheMemory));
  patchField(container, 'query-speed', `${stats.avgQueryTime.toFixed(0)}ms`);
  patchField(container, 'query-rate', `${stats.queriesPerSecond.toFixed(1)}/sec`);
  const networkMetrics = networkSummary(cacheMetrics.network);
  patchField(container, 'network-bytes', networkMetrics.dataLoaded);
  patchField(container, 'network-detail', networkMetrics.detail);

  const memoryPercent =
    cacheMetrics.memoryLimit > 0
      ? (cacheMetrics.totalCacheMemory / cacheMetrics.memoryLimit) * 100
      : 0;
  const barFill = container.querySelector(
    '.luxar-secondary-metrics .luxar-progress-bar__fill'
  ) as HTMLElement | null;
  if (barFill) {
    barFill.style.width = `${Math.min(100, memoryPercent)}%`;
    updateColorClass(barFill, getCacheMemoryColorClass(memoryPercent));
  }
  badges();
  return true;
}
