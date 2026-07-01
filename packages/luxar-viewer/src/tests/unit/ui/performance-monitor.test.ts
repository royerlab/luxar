/**
 * Unit tests for PerformanceMonitor.
 *
 * A compact square readout (no stats.js) driven by the animation loop's
 * `frame-start` / `frame-end` bus events. It shows one metric at a time and
 * cycles FPS → ms → graph on click. Tests cover element setup, visibility
 * gating of the bus subscription, the metric cycle, and disposal.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { PerformanceMonitor } from '../../../ui/performance-monitor';
import { eventBus } from '../../../utils/cross-layer/event-bus';

function frame(): void {
  eventBus.emit('frame-start', {});
  eventBus.emit('frame-end', {});
}

const el = () => document.body.querySelector<HTMLElement>('#luxar-stats')!;
const num = () => document.body.querySelector('.luxar-perf__num')?.textContent ?? '';
const unit = () => document.body.querySelector('.luxar-perf__unit')?.textContent ?? '';

describe('PerformanceMonitor', () => {
  let monitor: PerformanceMonitor;

  beforeEach(() => {
    document.head.innerHTML = '';
    document.body.innerHTML = '';
    monitor = new PerformanceMonitor();
    // The monitor no longer self-mounts (the rail docks its element); mount it
    // here so the DOM-based assertions below can find it.
    document.body.appendChild(monitor.element);
  });

  afterEach(() => {
    monitor.dispose();
  });

  describe('setup', () => {
    it('appends a square readout element with FPS as the default mode', () => {
      expect(el()).toBeTruthy();
      expect(el().classList.contains('luxar-perf')).toBe(true);
      expect(el().dataset.mode).toBe('fps');
      expect(unit()).toBe('fps');
    });

    it('sets accessibility attributes', () => {
      expect(el().getAttribute('role')).toBe('status');
      expect(el().getAttribute('aria-label')).toContain('Performance');
    });

    it('starts hidden', () => {
      expect(el().classList.contains('is-hidden')).toBe(true);
      expect(monitor.visible).toBe(false);
    });
  });

  describe('visibility', () => {
    it('show() reveals and hide() hides', () => {
      monitor.show();
      expect(monitor.visible).toBe(true);
      expect(el().classList.contains('is-hidden')).toBe(false);
      monitor.hide();
      expect(monitor.visible).toBe(false);
      expect(el().classList.contains('is-hidden')).toBe(true);
    });

    it('toggle() flips, show()/hide() idempotent', () => {
      monitor.toggle();
      expect(monitor.visible).toBe(true);
      monitor.show();
      expect(monitor.visible).toBe(true);
      monitor.hide();
      monitor.hide();
      expect(monitor.visible).toBe(false);
    });
  });

  describe('frame-timing subscription', () => {
    it('does not update while hidden (no subscription)', () => {
      const before = num();
      frame();
      expect(num()).toBe(before);
    });

    it('updates the readout from bus events while visible', () => {
      monitor.show();
      frame();
      expect(num()).not.toBe('––');
    });

    it('hide() unsubscribes so later frames no longer update it', () => {
      monitor.show();
      frame();
      const after = num();
      monitor.hide();
      frame();
      expect(num()).toBe(after);
    });
  });

  describe('cycleMode', () => {
    it('cycles FPS → ms → graph → FPS', () => {
      expect(el().dataset.mode).toBe('fps');
      monitor.cycleMode();
      expect(el().dataset.mode).toBe('ms');
      expect(unit()).toBe('ms');
      monitor.cycleMode();
      expect(el().dataset.mode).toBe('graph');
      monitor.cycleMode();
      expect(el().dataset.mode).toBe('fps');
      expect(unit()).toBe('fps');
    });

    it('cycles on click', () => {
      el().click();
      expect(el().dataset.mode).toBe('ms');
    });
  });

  describe('dispose', () => {
    it('removes the element from the DOM', () => {
      monitor.dispose();
      expect(document.body.querySelector('#luxar-stats')).toBeNull();
    });

    it('is a no-op if the element is already detached', () => {
      el().remove();
      expect(() => monitor.dispose()).not.toThrow();
    });
  });
});
