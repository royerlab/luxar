// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest';
import { updateOverviewTab } from '../../../../../ui/data-loading-monitor/tabs/overview';
import type { CacheMetrics, GlobalStats } from '../../../../../types/data-monitor-types';

describe('updateOverviewTab', () => {
  it('patches values before invoking the badge callback', () => {
    const container = document.createElement('div');
    container.innerHTML =
      '<span data-field="visible-points"></span><span data-field="visible-points-sub"></span><span data-field="memory-used"></span><span data-field="query-speed"></span><span data-field="query-rate"></span><span data-field="network-bytes"></span><span data-field="network-detail"></span><div class="luxar-secondary-metrics"><div class="luxar-progress-bar__fill"></div></div>';
    const stats = {
      datasetSize: 100,
      visiblePoints: 25,
      datasetSegments: 0,
      visibleSegments: 0,
      datasetSplats: 0,
      visibleSplats: 0,
      avgQueryTime: 12,
      queriesPerSecond: 3,
    } as GlobalStats;
    const cache = { totalCacheMemory: 50, memoryLimit: 100 } as CacheMetrics;
    const badges = vi.fn(() =>
      expect(container.querySelector('[data-field="visible-points"]')?.textContent).toBe('25')
    );

    expect(updateOverviewTab(container, stats, cache, badges)).toBe(true);
    expect(container.querySelector('[data-field="visible-points-sub"]')?.textContent).toBe(
      '25.0% of 100 total'
    );
    expect(badges).toHaveBeenCalledOnce();
  });

  it('does not patch badges when the rendered structure is absent', () => {
    const badges = vi.fn();
    expect(
      updateOverviewTab(
        document.createElement('div'),
        {} as GlobalStats,
        {} as CacheMetrics,
        badges
      )
    ).toBe(false);
    expect(badges).not.toHaveBeenCalled();
  });
});
