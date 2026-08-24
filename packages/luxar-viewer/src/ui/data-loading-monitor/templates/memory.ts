/**
 * Memory-tab templates and reuse-rate helpers.
 */

// Memory-metrics contracts live in `types/data-monitor-types` so the data
// layer's SceneLoaderMonitorPort can reference them precisely. Re-exported
// here for existing UI template callers; the local `import type` is needed
// because other functions in this file reference these types directly.
import type {
  GPUPoolTypeStats,
  GPUPoolStats,
  AccumulatorStats,
  MemoryMetrics,
} from '../../../types/data-monitor-types';
import { POOLED_GEOMETRY_TYPES } from '../../../types/data-monitor-types';
import { formatNumber } from './format';
import { getColorClass } from './primitives';

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
  const rows = POOLED_GEOMETRY_TYPES.map((type) => {
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
  }).join('');

  return `
    <div class="luxar-memory-section">
      <div class="luxar-memory-section__header">
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
  const rows = POOLED_GEOMETRY_TYPES.map((type) => {
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
  }).join('');

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
