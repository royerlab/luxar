/**
 * Unit tests for PerformanceMonitor
 *
 * Mocks stats.js (a class with begin/end/showPanel/dom). Verifies the
 * monitor's setup of the host element (id, role, ARIA, position),
 * begin/end gating by visibility, panel cycling, show/hide/toggle, and
 * disposal cleanup of the injected style block.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

interface MockStatsDom extends HTMLElement {
  panel?: number;
}

interface MockStatsLike {
  dom: MockStatsDom;
  showPanel: ReturnType<typeof vi.fn>;
  begin: ReturnType<typeof vi.fn>;
  end: ReturnType<typeof vi.fn>;
}

vi.mock('stats.js', () => {
  class MockStats implements MockStatsLike {
    dom: MockStatsDom = document.createElement('div') as MockStatsDom;
    showPanel = vi.fn((id: number) => {
      this.dom.panel = id;
    });
    begin = vi.fn();
    end = vi.fn();
  }
  return { default: MockStats };
});

import { PerformanceMonitor } from '../../../ui/monitors/performance-monitor';

describe('PerformanceMonitor', () => {
  let monitor: PerformanceMonitor;

  beforeEach(() => {
    document.head.innerHTML = '';
    document.body.innerHTML = '';
    monitor = new PerformanceMonitor();
  });

  afterEach(() => {
    monitor.dispose();
  });

  describe('setup', () => {
    it('appends the stats element to body', () => {
      const el = document.body.querySelector('#luxar-stats');
      expect(el).toBeTruthy();
    });

    it('sets WCAG accessibility attributes', () => {
      const el = document.body.querySelector('#luxar-stats')!;
      expect(el.getAttribute('role')).toBe('status');
      expect(el.getAttribute('aria-label')).toContain('Performance metrics');
    });

    it('starts hidden', () => {
      const el = document.body.querySelector<HTMLElement>('#luxar-stats')!;
      expect(el.style.display).toBe('none');
      expect(monitor.visible).toBe(false);
    });

    it('positions panel in bottom-left with fixed positioning', () => {
      const el = document.body.querySelector<HTMLElement>('#luxar-stats')!;
      expect(el.style.position).toBe('fixed');
      expect(el.style.bottom).toBe('20px');
      expect(el.style.left).toBe('20px');
    });

    it('injects scoped style block once for the panel', () => {
      const styleEls = document.head.querySelectorAll('#luxar-stats-custom-styles');
      expect(styleEls.length).toBe(1);
    });

    it('does not duplicate the style block on a second instance', () => {
      const m2 = new PerformanceMonitor();
      try {
        const styleEls = document.head.querySelectorAll('#luxar-stats-custom-styles');
        expect(styleEls.length).toBe(1);
      } finally {
        m2.dispose();
      }
    });
  });

  describe('visibility', () => {
    it('show() sets visible and reveals the element', () => {
      monitor.show();
      expect(monitor.visible).toBe(true);
      const el = document.body.querySelector<HTMLElement>('#luxar-stats')!;
      expect(el.style.display).toBe('block');
    });

    it('hide() resets visibility and hides the element', () => {
      monitor.show();
      monitor.hide();
      expect(monitor.visible).toBe(false);
      const el = document.body.querySelector<HTMLElement>('#luxar-stats')!;
      expect(el.style.display).toBe('none');
    });

    it('show() is idempotent', () => {
      monitor.show();
      monitor.show();
      expect(monitor.visible).toBe(true);
    });

    it('hide() is idempotent', () => {
      monitor.hide();
      expect(monitor.visible).toBe(false);
    });

    it('toggle() flips visibility', () => {
      monitor.toggle();
      expect(monitor.visible).toBe(true);
      monitor.toggle();
      expect(monitor.visible).toBe(false);
    });
  });

  describe('frame-timing subscriptions', () => {
    it('does not forward bus events to stats while hidden (no subscription)', async () => {
      const stats = (monitor as unknown as { stats: MockStatsLike }).stats;
      const { eventBus } = await import('../../../utils/event-bus');
      eventBus.emit('frame-start', {});
      eventBus.emit('frame-end', {});
      expect(stats.begin).not.toHaveBeenCalled();
      expect(stats.end).not.toHaveBeenCalled();
    });

    it('forwards bus events to stats.begin/end while visible', async () => {
      const stats = (monitor as unknown as { stats: MockStatsLike }).stats;
      const { eventBus } = await import('../../../utils/event-bus');
      monitor.show();
      eventBus.emit('frame-start', {});
      eventBus.emit('frame-end', {});
      expect(stats.begin).toHaveBeenCalledTimes(1);
      expect(stats.end).toHaveBeenCalledTimes(1);
    });

    it('hide() unsubscribes so subsequent bus events stop driving stats', async () => {
      const stats = (monitor as unknown as { stats: MockStatsLike }).stats;
      const { eventBus } = await import('../../../utils/event-bus');
      monitor.show();
      eventBus.emit('frame-start', {});
      eventBus.emit('frame-end', {});
      monitor.hide();
      eventBus.emit('frame-start', {});
      eventBus.emit('frame-end', {});
      expect(stats.begin).toHaveBeenCalledTimes(1);
      expect(stats.end).toHaveBeenCalledTimes(1);
    });
  });

  describe('cyclePanels', () => {
    it('does nothing while hidden', () => {
      const stats = (monitor as unknown as { stats: MockStatsLike }).stats;
      monitor.cyclePanels();
      // Constructor calls showPanel(0); cycle while hidden adds nothing
      expect(stats.showPanel).toHaveBeenCalledTimes(1);
    });

    it('cycles 0 → 1 → 2 → 0 while visible', () => {
      const stats = (monitor as unknown as { stats: MockStatsLike }).stats;
      monitor.show();
      // Setup already chose panel 0 in constructor; first cycle goes to 1
      monitor.cyclePanels();
      expect(stats.showPanel).toHaveBeenLastCalledWith(1);
      monitor.cyclePanels();
      expect(stats.showPanel).toHaveBeenLastCalledWith(2);
      monitor.cyclePanels();
      expect(stats.showPanel).toHaveBeenLastCalledWith(0);
    });

    it('treats missing panel index as 0', () => {
      const stats = (monitor as unknown as { stats: MockStatsLike }).stats;
      stats.dom.panel = undefined;
      monitor.show();
      monitor.cyclePanels();
      expect(stats.showPanel).toHaveBeenLastCalledWith(1);
    });
  });

  describe('dispose', () => {
    it('removes the stats element from DOM', () => {
      monitor.dispose();
      expect(document.body.querySelector('#luxar-stats')).toBeNull();
    });

    it('removes the injected style block', () => {
      monitor.dispose();
      expect(document.head.querySelector('#luxar-stats-custom-styles')).toBeNull();
    });

    it('is a no-op if element is already detached', () => {
      const el = document.body.querySelector('#luxar-stats')!;
      el.remove();
      // Should not throw
      monitor.dispose();
    });
  });
});
