/**
 * Unit tests for ResolutionIndicator component
 *
 * Verifies show/hide lifecycle, "show once per mode activation" gating,
 * auto-hide timing, text formatting, and resource cleanup.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

import { ResolutionIndicator } from '../../../ui/resolution-indicator';

describe('ResolutionIndicator', () => {
  let indicator: ResolutionIndicator;

  beforeEach(() => {
    document.body.innerHTML = '';
    vi.useFakeTimers();
    indicator = new ResolutionIndicator();
  });

  afterEach(() => {
    indicator.dispose();
    vi.useRealTimers();
  });

  describe('lifecycle', () => {
    it('starts hidden (no element added until show)', () => {
      expect(document.body.children.length).toBe(0);
      expect(indicator.getIsVisible()).toBe(false);
    });

    it('lazily creates the element on first show()', () => {
      indicator.show(0.75);
      const el = document.body.querySelector('.luxar-resolution-indicator');
      expect(el).toBeTruthy();
      expect(indicator.getIsVisible()).toBe(true);
    });

    it('removes the element on dispose()', () => {
      indicator.show(0.75);
      indicator.dispose();
      expect(document.body.querySelector('.luxar-resolution-indicator')).toBeNull();
      expect(indicator.getIsVisible()).toBe(false);
    });
  });

  describe('show / hide gating', () => {
    it('only shows once per mode activation (second show is suppressed)', () => {
      indicator.show(0.75);
      indicator.hide();
      // Second show in the same mode activation should be a no-op
      indicator.show(0.5);
      expect(indicator.getIsVisible()).toBe(false);
    });

    it('reset() re-arms show after a previous activation', () => {
      indicator.show(0.75);
      indicator.hide();
      indicator.reset();
      indicator.show(0.5);
      expect(indicator.getIsVisible()).toBe(true);
    });

    it('refreshes the text in place while visible (no re-show) on later show() calls', () => {
      indicator.show(0.75);
      const el = document.body.querySelector('.luxar-resolution-indicator__text')!;
      expect(el.textContent).toContain('75%');
      // The AdaptiveDPR caller streams every DPR step; while the toast
      // from this activation is still up, the percentage tracks the
      // latest value instead of freezing on the first step.
      indicator.show(0.5);
      expect(el.textContent).toContain('50%');
      expect(indicator.getIsVisible()).toBe(true);
    });

    it('ignores show() after auto-hide in the same activation (no re-show, no text change)', () => {
      indicator.show(0.75);
      const el = document.body.querySelector('.luxar-resolution-indicator__text')!;
      vi.advanceTimersByTime(4000); // auto-hide fires
      expect(indicator.getIsVisible()).toBe(false);

      indicator.show(0.5);
      expect(indicator.getIsVisible()).toBe(false);
      expect(el.textContent).toContain('75%');
    });

    it('hide() is a no-op when already hidden', () => {
      indicator.hide();
      expect(indicator.getIsVisible()).toBe(false);
    });
  });

  describe('text formatting', () => {
    it('shows percentage when dpr is provided', () => {
      indicator.show(0.625);
      const text = document.body.querySelector('.luxar-resolution-indicator__text')!;
      expect(text.textContent).toBe('Resolution Scaled to 63% to maintain 60 fps');
    });

    it('omits percentage when dpr is undefined', () => {
      indicator.show();
      const text = document.body.querySelector('.luxar-resolution-indicator__text')!;
      expect(text.textContent).toBe('Resolution Scaled to maintain 60 fps');
    });

    it('uses configured target FPS in text', () => {
      indicator.setTargetFPS(120);
      indicator.show(0.5);
      const text = document.body.querySelector('.luxar-resolution-indicator__text')!;
      expect(text.textContent).toContain('120 fps');
    });
  });

  describe('auto-hide', () => {
    it('auto-hides after 4 seconds', () => {
      indicator.show(0.75);
      expect(indicator.getIsVisible()).toBe(true);
      vi.advanceTimersByTime(4000);
      expect(indicator.getIsVisible()).toBe(false);
    });

    it('does not auto-hide before 4 seconds', () => {
      indicator.show(0.75);
      vi.advanceTimersByTime(3999);
      expect(indicator.getIsVisible()).toBe(true);
    });

    it('hide animation removes display:none after 300ms', () => {
      indicator.show(0.75);
      indicator.hide();
      const el = document.body.querySelector<HTMLElement>('.luxar-resolution-indicator')!;
      expect(el.style.display).not.toBe('none');
      vi.advanceTimersByTime(300);
      expect(el.style.display).toBe('none');
    });
  });

  describe('dispose cleanup', () => {
    it('clears pending auto-hide timer', () => {
      indicator.show(0.75);
      indicator.dispose();
      // Advance well past auto-hide; nothing should throw or reappear
      vi.advanceTimersByTime(10_000);
      expect(document.body.querySelector('.luxar-resolution-indicator')).toBeNull();
    });

    it('clears pending hide-animation timer', () => {
      indicator.show(0.75);
      indicator.hide();
      indicator.dispose();
      vi.advanceTimersByTime(1000);
      // No errors and element gone
      expect(document.body.querySelector('.luxar-resolution-indicator')).toBeNull();
    });

    it('resets hasShownForCurrentMode so a new instance can show', () => {
      indicator.show(0.75);
      indicator.dispose();
      const fresh = new ResolutionIndicator();
      fresh.show(0.5);
      expect(fresh.getIsVisible()).toBe(true);
      fresh.dispose();
    });
  });
});
