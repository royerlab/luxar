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

describe('updateOverviewTab — mesh parity', () => {
  /** The card structure the renderer paints for the given fields. */
  function paint(fields: string[]): HTMLElement {
    const container = document.createElement('div');
    container.innerHTML =
      fields
        .flatMap((f) => [`<span data-field="${f}"></span>`, `<span data-field="${f}-sub"></span>`])
        .join('') +
      '<span data-field="memory-used"></span><span data-field="query-speed"></span>' +
      '<span data-field="query-rate"></span><span data-field="network-bytes"></span>' +
      '<span data-field="network-detail"></span>';
    return container;
  }

  const CACHE = { totalCacheMemory: 0, memoryLimit: 100 } as CacheMetrics;

  it('patches the mesh card like any other type', () => {
    const container = paint(['visible-triangles']);
    const stats = {
      datasetTriangles: 4000,
      visibleTriangles: 1000,
      avgQueryTime: 0,
      queriesPerSecond: 0,
    } as GlobalStats;

    expect(updateOverviewTab(container, stats, CACHE, vi.fn())).toBe(true);
    expect(container.querySelector('[data-field="visible-triangles"]')?.textContent).toBe('1.0K');
    expect(container.querySelector('[data-field="visible-triangles-sub"]')?.textContent).toBe(
      '25.0% of 4.0K total'
    );
  });

  it('asks for a rebuild when a present type has no card yet', () => {
    // A mesh node finishing its load after the first paint has no card. Patching
    // it would silently no-op forever, so the tick is handed back to the caller's
    // full rebuild instead — which paints the missing card.
    const container = paint(['visible-points']);
    const stats = {
      datasetSize: 100,
      visiblePoints: 10,
      datasetTriangles: 4000,
      visibleTriangles: 1000,
      avgQueryTime: 0,
      queriesPerSecond: 0,
    } as GlobalStats;
    const badges = vi.fn();

    expect(updateOverviewTab(container, stats, CACHE, badges)).toBe(false);
    expect(badges).not.toHaveBeenCalled();
  });

  it('drops the " total" suffix once a second type appears', () => {
    const container = paint(['visible-points', 'visible-triangles']);
    const stats = {
      datasetSize: 100,
      visiblePoints: 10,
      datasetTriangles: 4000,
      visibleTriangles: 1000,
      avgQueryTime: 0,
      queriesPerSecond: 0,
    } as GlobalStats;

    expect(updateOverviewTab(container, stats, CACHE, vi.fn())).toBe(true);
    expect(container.querySelector('[data-field="visible-points-sub"]')?.textContent).toBe(
      '10.0% of 100'
    );
    expect(container.querySelector('[data-field="visible-triangles-sub"]')?.textContent).toBe(
      '25.0% of 4.0K'
    );
  });
});
