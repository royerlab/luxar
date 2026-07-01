/**
 * Unit tests for PerformanceMonitor.
 *
 * The monitor renders a compact, theme-matched FPS / frame-time readout
 * (no stats.js) driven by the animation loop's `frame-start` / `frame-end`
 * bus events. Tests cover element setup, visibility gating of the bus
 * subscription, the memory-detail cycle, and disposal.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { PerformanceMonitor } from '../../../ui/performance-monitor';
import { eventBus } from '../../../utils/cross-layer/event-bus';

function frame(): void {
  eventBus.emit('frame-start', {});
  eventBus.emit('frame-end', {});
}

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
    it('appends the readout element to the container', () => {
      const el = document.body.querySelector('#luxar-stats');
      expect(el).toBeTruthy();
      expect(el!.classList.contains('luxar-perf')).toBe(true);
    });

    it('sets WCAG accessibility attributes', () => {
      const el = document.body.querySelector('#luxar-stats')!;
      expect(el.getAttribute('role')).toBe('status');
      expect(el.getAttribute('aria-label')).toContain('Performance metrics');
    });

    it('starts hidden with placeholder text', () => {
      const el = document.body.querySelector<HTMLElement>('#luxar-stats')!;
      expect(el.style.display).toBe('none');
      expect(monitor.visible).toBe(false);
      expect(el.querySelector('.luxar-perf__fps')?.textContent).toContain('fps');
    });
  });

  describe('visibility', () => {
    it('show() sets visible and reveals the element', () => {
      monitor.show();
      expect(monitor.visible).toBe(true);
      const el = document.body.querySelector<HTMLElement>('#luxar-stats')!;
      expect(el.style.display).not.toBe('none');
    });

    it('hide() resets visibility and hides the element', () => {
      monitor.show();
      monitor.hide();
      expect(monitor.visible).toBe(false);
      expect(document.body.querySelector<HTMLElement>('#luxar-stats')!.style.display).toBe('none');
    });

    it('show()/hide() are idempotent and toggle() flips', () => {
      monitor.show();
      monitor.show();
      expect(monitor.visible).toBe(true);
      monitor.toggle();
      expect(monitor.visible).toBe(false);
      monitor.hide();
      expect(monitor.visible).toBe(false);
    });
  });

  describe('frame-timing subscription', () => {
    const fpsText = () =>
      document.body.querySelector('.luxar-perf__ms')?.textContent ?? '';

    it('does not update while hidden (no subscription)', () => {
      const before = fpsText();
      frame();
      expect(fpsText()).toBe(before); // untouched placeholder
    });

    it('updates the readout from bus events while visible', () => {
      monitor.show();
      frame();
      // The ms readout advances off its placeholder once a frame is measured.
      expect(fpsText()).toMatch(/ms$/);
      expect(fpsText()).not.toContain('––');
    });

    it('hide() unsubscribes so later frames no longer update it', () => {
      monitor.show();
      frame();
      const afterShow = fpsText();
      monitor.hide();
      frame();
      expect(fpsText()).toBe(afterShow); // frozen after unsubscribe
    });
  });

  describe('cyclePanels (memory detail)', () => {
    it('does nothing while hidden', () => {
      monitor.cyclePanels();
      const mem = document.body.querySelector<HTMLElement>('.luxar-perf__mem')!;
      expect(mem.style.display).toBe('none');
    });

    it('toggles the memory readout while visible', () => {
      monitor.show();
      const mem = document.body.querySelector<HTMLElement>('.luxar-perf__mem')!;
      expect(mem.style.display).toBe('none');
      monitor.cyclePanels();
      expect(mem.style.display).not.toBe('none');
      monitor.cyclePanels();
      expect(mem.style.display).toBe('none');
    });
  });

  describe('dispose', () => {
    it('removes the element from the DOM', () => {
      monitor.dispose();
      expect(document.body.querySelector('#luxar-stats')).toBeNull();
    });

    it('is a no-op if the element is already detached', () => {
      document.body.querySelector('#luxar-stats')!.remove();
      expect(() => monitor.dispose()).not.toThrow();
    });
  });
});
