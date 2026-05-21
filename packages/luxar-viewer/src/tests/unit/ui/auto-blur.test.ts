/**
 * Unit tests for the auto-blur dispatcher.
 *
 * The dispatcher chooses different blur triggers per input type
 * (checkbox: change/click; range: mouseup/touchend/wheel-debounced;
 * number/text: Enter/Escape; select: change). Tests use a real
 * EventManager and real DOM events with fake timers — no mocks.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { applyAutoBlur } from '../../../ui/gui/format/auto-blur';
import { EventManager } from '../../../ui/gui/dom/event-manager';

describe('applyAutoBlur', () => {
  let manager: EventManager;

  beforeEach(() => {
    manager = new EventManager();
    vi.useFakeTimers();
    document.body.innerHTML = '';
  });

  afterEach(() => {
    manager.removeAll();
    vi.useRealTimers();
  });

  function makeInput(type: string): HTMLInputElement {
    const el = document.createElement('input');
    el.type = type;
    document.body.appendChild(el);
    return el;
  }

  function makeSelect(): HTMLSelectElement {
    const el = document.createElement('select');
    document.body.appendChild(el);
    return el;
  }

  describe('checkbox', () => {
    it('registers change + click listeners', () => {
      const el = makeInput('checkbox');
      applyAutoBlur(el, manager);
      expect(manager.count()).toBe(2);
    });

    it('blurs after change (with the 10ms setTimeout delay)', () => {
      const el = makeInput('checkbox');
      applyAutoBlur(el, manager);
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new Event('change'));
      // Listener schedules blur via setTimeout; flush 10ms.
      vi.advanceTimersByTime(10);
      expect(blur).toHaveBeenCalledTimes(1);
    });

    it('also blurs after click', () => {
      const el = makeInput('checkbox');
      applyAutoBlur(el, manager);
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new Event('click'));
      vi.advanceTimersByTime(10);
      expect(blur).toHaveBeenCalledTimes(1);
    });
  });

  describe('range', () => {
    it('registers mouseup + touchend + wheel listeners', () => {
      const el = makeInput('range');
      applyAutoBlur(el, manager);
      expect(manager.count()).toBe(3);
    });

    it('blurs on mouseup', () => {
      const el = makeInput('range');
      applyAutoBlur(el, manager);
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new Event('mouseup'));
      vi.advanceTimersByTime(10);
      expect(blur).toHaveBeenCalledTimes(1);
    });

    it('blurs on touchend', () => {
      const el = makeInput('range');
      applyAutoBlur(el, manager);
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new Event('touchend'));
      vi.advanceTimersByTime(10);
      expect(blur).toHaveBeenCalledTimes(1);
    });

    it('blurs on wheel after a 200ms debounce', () => {
      const el = makeInput('range');
      applyAutoBlur(el, manager);
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new Event('wheel'));
      vi.advanceTimersByTime(199);
      expect(blur).not.toHaveBeenCalled();
      vi.advanceTimersByTime(1);
      expect(blur).toHaveBeenCalledTimes(1);
    });

    it('debounces consecutive wheel events into a single blur', () => {
      const el = makeInput('range');
      applyAutoBlur(el, manager);
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new Event('wheel'));
      vi.advanceTimersByTime(100);
      el.dispatchEvent(new Event('wheel'));
      vi.advanceTimersByTime(100);
      el.dispatchEvent(new Event('wheel'));
      vi.advanceTimersByTime(199);
      expect(blur).not.toHaveBeenCalled();
      vi.advanceTimersByTime(1);
      expect(blur).toHaveBeenCalledTimes(1);
    });
  });

  describe('text/number inputs', () => {
    it('registers a single keydown listener for text inputs', () => {
      const el = makeInput('text');
      applyAutoBlur(el, manager);
      expect(manager.count()).toBe(1);
    });

    it('blurs on Enter (commit)', () => {
      const el = makeInput('text');
      applyAutoBlur(el, manager);
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
      expect(blur).toHaveBeenCalledTimes(1);
    });

    it('blurs on Escape (cancel)', () => {
      const el = makeInput('text');
      applyAutoBlur(el, manager);
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
      expect(blur).toHaveBeenCalledTimes(1);
    });

    it('does not blur on other keys (e.g. typing)', () => {
      const el = makeInput('text');
      applyAutoBlur(el, manager);
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }));
      el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab' }));
      el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown' }));
      expect(blur).not.toHaveBeenCalled();
    });

    it('also handles number inputs (same Enter/Escape contract)', () => {
      const el = makeInput('number');
      applyAutoBlur(el, manager);
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
      expect(blur).toHaveBeenCalledTimes(1);
    });
  });

  describe('unsupported input types', () => {
    it('registers no listeners for hidden inputs', () => {
      const el = makeInput('hidden');
      applyAutoBlur(el, manager);
      expect(manager.count()).toBe(0);
    });

    it('registers no listeners for file inputs', () => {
      const el = makeInput('file');
      applyAutoBlur(el, manager);
      expect(manager.count()).toBe(0);
    });
  });

  describe('select', () => {
    it('registers a single change listener', () => {
      const el = makeSelect();
      applyAutoBlur(el, manager);
      expect(manager.count()).toBe(1);
    });

    it('blurs after change', () => {
      const el = makeSelect();
      applyAutoBlur(el, manager);
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new Event('change'));
      vi.advanceTimersByTime(10);
      expect(blur).toHaveBeenCalledTimes(1);
    });
  });

  describe('removeAll cleanup', () => {
    it('manager.removeAll() drops every listener', () => {
      const el = makeInput('range');
      applyAutoBlur(el, manager);
      expect(manager.count()).toBe(3);
      manager.removeAll();
      expect(manager.count()).toBe(0);
      // After removeAll, dispatched events should no longer fire blur.
      const blur = vi.spyOn(el, 'blur');
      el.dispatchEvent(new Event('mouseup'));
      vi.advanceTimersByTime(10);
      expect(blur).not.toHaveBeenCalled();
    });
  });
});
