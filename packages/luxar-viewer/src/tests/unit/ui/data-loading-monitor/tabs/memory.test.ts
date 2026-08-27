// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { updateMemoryTab } from '../../../../../ui/data-loading-monitor/tabs/memory';
import type { MemoryMetrics } from '../../../../../types/data-monitor-types';

describe('updateMemoryTab', () => {
  it('patches populated GPU pool rows and dims empty types', () => {
    const container = document.createElement('div');
    container.innerHTML = `
      <span data-field="memory-total"></span>
      <span data-field="gpu-points-reuse"></span>
      <span data-field="gpu-points-active"></span>
      <span data-field="gpu-points-pooled"></span>
      <span data-field="gpu-points-allocs"></span>
      <span data-field="gpu-lines-reuse" class="luxar-color--warning"></span>
      <span data-field="gpu-lines-active"></span>
      <span data-field="gpu-lines-pooled"></span>
      <span data-field="gpu-lines-allocs"></span>
      <span data-field="gpu-summary"></span>
      <span data-field="acc-summary"></span>
    `;
    const metrics: MemoryMetrics = {
      gpuPool: {
        allocations: 4,
        reuses: 6,
        evictions: 2,
        capacityGrowths: 1,
        activeBuffers: 3,
        pooledBuffers: 5,
        byType: {
          points: {
            allocations: 4,
            reuses: 6,
            evictions: 2,
            activeBuffers: 3,
            pooledBuffers: 5,
          },
          lines: {
            allocations: 0,
            reuses: 0,
            evictions: 0,
            activeBuffers: 0,
            pooledBuffers: 0,
          },
          gsplats: {
            allocations: 0,
            reuses: 0,
            evictions: 0,
            activeBuffers: 0,
            pooledBuffers: 0,
          },
        },
      },
      accumulators: { points: null, lines: null, gsplats: null },
    };

    expect(updateMemoryTab(container, metrics)).toBe(true);
    expect(container.querySelector('[data-field="gpu-points-reuse"]')?.textContent).toBe('60%');
    expect(container.querySelector('[data-field="gpu-points-reuse"]')?.className).toContain(
      'warning'
    );
    expect(container.querySelector('[data-field="gpu-points-active"]')?.textContent).toBe('3');
    expect(container.querySelector('[data-field="gpu-points-pooled"]')?.textContent).toBe('5');
    expect(container.querySelector('[data-field="gpu-points-allocs"]')?.textContent).toBe('4');
    expect(container.querySelector('[data-field="gpu-lines-reuse"]')?.textContent).toBe('—');
    expect(container.querySelector('[data-field="gpu-lines-reuse"]')?.className).toContain(
      'dimmed'
    );
    expect(container.querySelector('[data-field="gpu-summary"]')?.textContent).toBe(
      'Total: 4 allocs · 6 reuses · 2 evicted'
    );
  });

  it('patches summaries and clears a stale growth warning', () => {
    const container = document.createElement('div');
    container.innerHTML =
      '<span data-field="memory-total"></span><span data-field="acc-summary"></span><span data-field="acc-points-capacity"></span><span data-field="acc-points-memory"></span><span data-field="acc-points-grows" class="luxar-color--warning"></span>';
    const metrics = {
      gpuPool: null,
      accumulators: {
        points: { capacity: 10, allocations: 2, growthEvents: 1, memoryMB: 3 },
        lines: null,
        gsplats: null,
      },
    } as MemoryMetrics;

    expect(updateMemoryTab(container, metrics)).toBe(true);
    expect(container.querySelector('[data-field="acc-summary"]')?.textContent).toBe(
      'Total: 3.0MB · 2 allocations'
    );
    expect(container.querySelector('[data-field="acc-points-grows"]')?.className).not.toContain(
      'warning'
    );
  });
});
