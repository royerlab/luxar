import type { MemoryMetrics } from '../../../types/data-monitor-types';
import { POOLED_GEOMETRY_TYPES } from '../../../types/data-monitor-types';
import { formatNumber } from '../templates/format';
import { calculateReuseRate, getReuseRateColorClass } from '../templates/memory';
import { getColorClass } from '../templates/primitives';
import { patchField, updateColorClass } from './dom-helpers';

export function updateMemoryTab(container: HTMLElement | null, metrics: MemoryMetrics): boolean {
  if (!container) return false;
  if (!container.querySelector('[data-field="memory-total"]')) return false;

  if (metrics.gpuPool) {
    for (const type of POOLED_GEOMETRY_TYPES) {
      const typeStats = metrics.gpuPool.byType[type];
      const reuseRate = calculateReuseRate(typeStats.allocations, typeStats.reuses);
      const hasData =
        typeStats.allocations > 0 || typeStats.reuses > 0 || typeStats.activeBuffers > 0;
      const reuseElement = container.querySelector(`[data-field="gpu-${type}-reuse"]`);
      if (reuseElement) {
        reuseElement.textContent = hasData ? `${reuseRate.toFixed(0)}%` : '—';
        updateColorClass(
          reuseElement as HTMLElement,
          hasData ? getReuseRateColorClass(reuseRate) : getColorClass('dimmed')
        );
      }
      patchField(container, `gpu-${type}-active`, hasData ? `${typeStats.activeBuffers}` : '—');
      patchField(container, `gpu-${type}-pooled`, hasData ? `${typeStats.pooledBuffers}` : '—');
      patchField(container, `gpu-${type}-allocs`, hasData ? `${typeStats.allocations}` : '—');
    }
    patchField(
      container,
      'gpu-summary',
      `Total: ${metrics.gpuPool.allocations} allocs · ${metrics.gpuPool.reuses} reuses · ${metrics.gpuPool.evictions} evicted`
    );
  }

  for (const type of POOLED_GEOMETRY_TYPES) {
    const stats = metrics.accumulators[type];
    const hasData = stats !== null && stats.capacity > 0;
    patchField(container, `acc-${type}-capacity`, hasData ? formatNumber(stats.capacity) : '—');
    patchField(container, `acc-${type}-memory`, hasData ? `${stats.memoryMB.toFixed(1)}MB` : '—');
    const growsElement = container.querySelector(`[data-field="acc-${type}-grows"]`);
    if (growsElement) {
      growsElement.textContent = hasData ? `${stats.growthEvents}` : '—';
      updateColorClass(
        growsElement as HTMLElement,
        hasData && stats.growthEvents > 5 ? getColorClass('warning') : ''
      );
    }
  }

  const totalAccumulatorMemory = POOLED_GEOMETRY_TYPES.reduce(
    (total, type) => total + (metrics.accumulators[type]?.memoryMB ?? 0),
    0
  );
  const totalAccumulatorAllocations = POOLED_GEOMETRY_TYPES.reduce(
    (total, type) => total + (metrics.accumulators[type]?.allocations ?? 0),
    0
  );
  patchField(
    container,
    'acc-summary',
    `Total: ${totalAccumulatorMemory.toFixed(1)}MB · ${totalAccumulatorAllocations} allocations`
  );
  const totalAllocations = metrics.gpuPool?.allocations ?? 0;
  const overallReuseRate = calculateReuseRate(totalAllocations, metrics.gpuPool?.reuses ?? 0);
  patchField(
    container,
    'memory-total',
    `${totalAllocations} allocs · ${overallReuseRate.toFixed(0)}% reuse · ${totalAccumulatorMemory.toFixed(1)}MB`
  );
  return true;
}
