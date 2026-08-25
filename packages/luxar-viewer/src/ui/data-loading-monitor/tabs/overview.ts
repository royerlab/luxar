import type { CacheMetrics, GlobalStats } from '../../../types/data-monitor-types';
import { formatBytes, formatNumber, getCacheMemoryColorClass } from '../templates/format';
import { countColorClass } from '../templates/primitives';
import { patchField, updateColorClass } from './dom-helpers';

export function updateOverviewTab(
  container: HTMLElement | null,
  stats: GlobalStats,
  cacheMetrics: CacheMetrics,
  badges: () => void
): boolean {
  if (!container) return false;
  const hasPoints = stats.datasetSize > 0 || stats.visiblePoints > 0;
  const hasLines = stats.datasetSegments > 0 || stats.visibleSegments > 0;
  const hasGSplats = stats.datasetSplats > 0 || stats.visibleSplats > 0;
  const anyPrimaryField =
    container.querySelector('[data-field="visible-points"]') ||
    container.querySelector('[data-field="visible-lines"]') ||
    container.querySelector('[data-field="visible-splats"]');
  if (!anyPrimaryField) return false;

  const suffix = [hasPoints, hasLines, hasGSplats].filter(Boolean).length === 1 ? ' total' : '';
  const patchCount = (field: string, visible: number, dataset: number) => {
    const percent = dataset > 0 ? ((visible / dataset) * 100).toFixed(1) : '0';
    patchField(container, field, formatNumber(visible));
    patchField(container, `${field}-sub`, `${percent}% of ${formatNumber(dataset)}${suffix}`);
    const element = container.querySelector(`[data-field="${field}"]`);
    if (element) updateColorClass(element as HTMLElement, countColorClass(visible));
  };
  if (hasPoints) patchCount('visible-points', stats.visiblePoints, stats.datasetSize);
  if (hasLines) patchCount('visible-lines', stats.visibleSegments, stats.datasetSegments);
  if (hasGSplats) patchCount('visible-splats', stats.visibleSplats, stats.datasetSplats);

  patchField(container, 'memory-used', formatBytes(cacheMetrics.totalCacheMemory));
  patchField(container, 'query-speed', `${stats.avgQueryTime.toFixed(0)}ms`);
  patchField(container, 'query-rate', `${stats.queriesPerSecond.toFixed(1)}/sec`);
  const network = cacheMetrics.network;
  const dataLoaded = network ? (network.totalBytesServed ?? network.bytesTransferred) : 0;
  patchField(container, 'network-bytes', network ? formatBytes(dataLoaded) : '0B');
  patchField(
    container,
    'network-detail',
    network
      ? `${formatBytes(network.bytesTransferred)} net · ${formatBytes(network.bandwidth)}/s`
      : '0B net'
  );

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
