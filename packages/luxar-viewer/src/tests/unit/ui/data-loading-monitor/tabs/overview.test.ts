// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest';
import { updateOverviewTab } from '../../../../../ui/data-loading-monitor/tabs/overview';
import { renderSecondaryMetrics } from '../../../../../ui/data-loading-monitor/templates/overview';
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

  it('keeps the rendered and patched network detail identical', () => {
    const container = document.createElement('div');
    const network = {
      bytesTransferred: 500,
      requestCount: 7,
      bandwidth: 50,
      totalBytesServed: 1000,
      totalRequestsServed: 42,
    };
    container.innerHTML = `<span data-field="visible-points"></span>${renderSecondaryMetrics(
      { used: 0, limit: 1 },
      { avgTime: 0, perSec: 0 },
      network
    )}`;
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
    const renderedDetail = container
      .querySelector('[data-field="network-detail"]')
      ?.textContent?.trim();

    updateOverviewTab(
      container,
      stats,
      {
        totalCacheMemory: 0,
        memoryLimit: 1,
        network,
      } as CacheMetrics,
      badges
    );
    const patchedDetail = container
      .querySelector('[data-field="network-detail"]')
      ?.textContent?.trim();
    expect(patchedDetail).toBe(renderedDetail);
    expect(patchedDetail).toBe('500B net · 50B/s · 42 reqs');
  });

  it('falls back to requestCount for legacy providers', () => {
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
      vi.fn()
    );
    expect(container.querySelector('[data-field="network-detail"]')?.textContent).toBe(
      '500B net · 50B/s · 7 reqs'
    );
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
      droppedElements: 0,
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
      droppedElements: 0,
      avgQueryTime: 0,
      queriesPerSecond: 0,
    } as GlobalStats;
    const badges = vi.fn();

    expect(updateOverviewTab(container, stats, CACHE, badges)).toBe(false);
    expect(badges).not.toHaveBeenCalled();
  });

  it('drops the " total" suffix once a second type appears', () => {
    const container = paint(['visible-points', 'visible-triangles']);
    // The dropped-element card sits in the same grid and is patched
    // unconditionally: a mixed points+mesh scene must keep it live, not freeze
    // it at its first-paint value because a second type showed up.
    container.insertAdjacentHTML('beforeend', '<span data-field="dropped-elements"></span>');
    const stats = {
      datasetSize: 100,
      visiblePoints: 10,
      datasetTriangles: 4000,
      visibleTriangles: 1000,
      droppedElements: 5000,
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
    expect(container.querySelector('[data-field="dropped-elements"]')?.textContent).toBe('5.0K');
  });

  it('clears the dropped-element error color once the count returns to zero', () => {
    // The recovery direction: a card that went red on one dataset must clear
    // when a switch brings the count back to 0, which is the whole point of
    // having a colored dropped-element card.
    const container = paint(['visible-points']);
    container.insertAdjacentHTML(
      'beforeend',
      '<span data-field="dropped-elements" class="luxar-color--error"></span>'
    );
    const stats = {
      datasetSize: 100,
      visiblePoints: 10,
      droppedElements: 0,
      avgQueryTime: 0,
      queriesPerSecond: 0,
    } as GlobalStats;

    expect(updateOverviewTab(container, stats, CACHE, vi.fn())).toBe(true);
    const dropped = container.querySelector('[data-field="dropped-elements"]');
    expect(dropped?.textContent).toBe('0');
    expect(dropped?.classList.contains('luxar-color--success')).toBe(true);
    expect(dropped?.classList.contains('luxar-color--error')).toBe(false);
  });

  it('clears a dimmed count card once its elements scroll into view', () => {
    // Presence is `total > 0 || visible > 0`, so a mesh sliced fully out is
    // painted dimmed. When the slider moves, the count must not appear in a
    // still-greyed card.
    const container = paint(['visible-triangles']);
    const triangles = container.querySelector('[data-field="visible-triangles"]') as HTMLElement;
    triangles.className = 'luxar-color--dimmed';
    const stats = {
      datasetTriangles: 4000,
      visibleTriangles: 1000,
      droppedElements: 0,
      avgQueryTime: 0,
      queriesPerSecond: 0,
    } as GlobalStats;

    expect(updateOverviewTab(container, stats, CACHE, vi.fn())).toBe(true);
    expect(triangles.textContent).toBe('1.0K');
    expect(triangles.classList.contains('luxar-color--dimmed')).toBe(false);
  });

  it('dims a count card whose type is sliced fully out', () => {
    const container = paint(['visible-triangles']);
    const stats = {
      datasetTriangles: 4000,
      visibleTriangles: 0,
      droppedElements: 0,
      avgQueryTime: 0,
      queriesPerSecond: 0,
    } as GlobalStats;

    expect(updateOverviewTab(container, stats, CACHE, vi.fn())).toBe(true);
    const triangles = container.querySelector('[data-field="visible-triangles"]');
    expect(triangles?.textContent).toBe('0');
    expect(triangles?.classList.contains('luxar-color--dimmed')).toBe(true);
  });
});
