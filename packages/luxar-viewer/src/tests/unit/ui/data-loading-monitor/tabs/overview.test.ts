// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest';
import { updateOverviewTab } from '../../../../../ui/data-loading-monitor/tabs/overview';
import type { CacheMetrics, GlobalStats } from '../../../../../types/data-monitor-types';

describe('updateOverviewTab', () => {
  it('patches values before invoking the badge callback', () => {
    const container = document.createElement('div');
    container.innerHTML =
      '<span data-field="visible-points"></span><span data-field="visible-points-sub"></span><span data-field="dropped-elements" class="luxar-color--success"></span><span data-field="memory-used"></span><span data-field="query-speed"></span><span data-field="query-rate"></span><span data-field="network-bytes"></span><span data-field="network-detail"></span><div class="luxar-secondary-metrics"><div class="luxar-progress-bar__fill"></div></div>';
    const stats = {
      datasetSize: 100,
      visiblePoints: 25,
      datasetSegments: 0,
      visibleSegments: 0,
      datasetSplats: 0,
      visibleSplats: 0,
      droppedElements: 5000,
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
    const dropped = container.querySelector('[data-field="dropped-elements"]');
    expect(dropped?.textContent).toBe('5.0K');
    expect(dropped?.classList.contains('luxar-color--error')).toBe(true);
    expect(dropped?.classList.contains('luxar-color--success')).toBe(false);
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

  it('keeps the request count in the patched network detail', () => {
    const container = document.createElement('div');
    container.innerHTML =
      '<span data-field="visible-points"></span><span data-field="network-bytes"></span><span data-field="network-detail"></span>';
    const stats = {
      datasetSize: 1,
      visiblePoints: 1,
      datasetSegments: 0,
      visibleSegments: 0,
      datasetSplats: 0,
      visibleSplats: 0,
      droppedElements: 0,
      avgQueryTime: 0,
      queriesPerSecond: 0,
    } as GlobalStats;
    const badges = vi.fn();

    updateOverviewTab(
      container,
      stats,
      {
        totalCacheMemory: 0,
        memoryLimit: 1,
        network: {
          bytesTransferred: 500,
          requestCount: 7,
          bandwidth: 50,
          totalBytesServed: 1000,
          totalRequestsServed: 42,
        },
      } as CacheMetrics,
      badges
    );
    expect(container.querySelector('[data-field="network-detail"]')?.textContent).toBe(
      '500B net · 50B/s · 42 reqs'
    );

    updateOverviewTab(
      container,
      stats,
      {
        totalCacheMemory: 0,
        memoryLimit: 1,
        network: {
          bytesTransferred: 500,
          requestCount: 7,
          bandwidth: 50,
          totalBytesServed: 1000,
        },
      } as CacheMetrics,
      badges
    );
    expect(container.querySelector('[data-field="network-detail"]')?.textContent).toBe(
      '500B net · 50B/s · 7 reqs'
    );
  });
});
