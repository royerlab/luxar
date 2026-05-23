/**
 * Unit tests for core/app/debug/cache-stats-view.ts (G6).
 *
 * `openCacheStatsView` opens the data-loading monitor in expanded mode
 * on the Cache tab. Contract:
 *   - Skips silently when DataMonitorManager.getInstance().getDefaultMonitor()
 *     returns nullish (embedded context with no monitor).
 *   - When a monitor exists, calls show() → expand() → setActiveTab('cache')
 *     in that exact order so the panel is visible *before* expand/tab
 *     changes (avoids a flash of collapsed state).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

const mocks = vi.hoisted(() => ({
  getDefaultMonitor: vi.fn(),
  getInstance: vi.fn(),
}));

vi.mock('../../../../../ui/data-monitor-manager', () => ({
  DataMonitorManager: {
    getInstance: () => ({
      getDefaultMonitor: mocks.getDefaultMonitor,
    }),
  },
}));

import { openCacheStatsView } from '../../../../../core/app/debug/cache-stats-view';

describe('openCacheStatsView', () => {
  beforeEach(() => {
    mocks.getDefaultMonitor.mockReset();
  });

  it('is a no-op when no monitor is registered (embedded context)', () => {
    mocks.getDefaultMonitor.mockReturnValue(null);

    // Must not throw — embedded callers with monitor disabled rely on
    // this guard.
    expect(() => openCacheStatsView()).not.toThrow();
    expect(mocks.getDefaultMonitor).toHaveBeenCalledOnce();
  });

  it('is a no-op when getDefaultMonitor returns undefined', () => {
    mocks.getDefaultMonitor.mockReturnValue(undefined);
    expect(() => openCacheStatsView()).not.toThrow();
  });

  it('show → expand → setActiveTab fires when a monitor is registered', () => {
    const monitor = {
      show: vi.fn(),
      expand: vi.fn(),
      setActiveTab: vi.fn(),
    };
    mocks.getDefaultMonitor.mockReturnValue(monitor);

    openCacheStatsView();

    expect(monitor.show).toHaveBeenCalledOnce();
    expect(monitor.expand).toHaveBeenCalledOnce();
    expect(monitor.setActiveTab).toHaveBeenCalledExactlyOnceWith('cache');
  });

  it('preserves the show → expand → setActiveTab order', () => {
    // The order matters because expand() and setActiveTab() depend on
    // the panel being visible. A regression that swapped them would
    // surface as a one-frame flash of the old tab on first open.
    const order: string[] = [];
    const monitor = {
      show: vi.fn(() => order.push('show')),
      expand: vi.fn(() => order.push('expand')),
      setActiveTab: vi.fn(() => order.push('setActiveTab')),
    };
    mocks.getDefaultMonitor.mockReturnValue(monitor);

    openCacheStatsView();

    expect(order).toEqual(['show', 'expand', 'setActiveTab']);
  });

  it('passes the literal "cache" tab id to setActiveTab', () => {
    // Mutation guard: a refactor that introduced a typo would still
    // pass a generic `toHaveBeenCalled()` check. Pin the literal.
    const monitor = {
      show: vi.fn(),
      expand: vi.fn(),
      setActiveTab: vi.fn(),
    };
    mocks.getDefaultMonitor.mockReturnValue(monitor);

    openCacheStatsView();

    expect(monitor.setActiveTab).toHaveBeenCalledWith('cache');
  });
});
