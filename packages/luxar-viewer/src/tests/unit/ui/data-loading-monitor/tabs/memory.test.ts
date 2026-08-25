// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { updateMemoryTab } from '../../../../../ui/data-loading-monitor/tabs/memory';
import type { MemoryMetrics } from '../../../../../types/data-monitor-types';

describe('updateMemoryTab', () => {
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
