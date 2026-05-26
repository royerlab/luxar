/**
 * EventManager Tests - CRITICAL for memory leak prevention
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { EventManager } from '../../../../../ui/gui/dom/event-manager';

describe('EventManager', () => {
  let manager: EventManager;
  let element: HTMLElement;

  beforeEach(() => {
    manager = new EventManager();
    element = document.createElement('div');
    document.body.appendChild(element);
  });

  describe('add()', () => {
    it('should add event listener to element', () => {
      const handler = vi.fn();

      manager.add(element, 'click', handler);

      element.click();
      expect(handler).toHaveBeenCalledOnce();
    });

    it('should track added listeners', () => {
      const handler = vi.fn();

      manager.add(element, 'click', handler);

      expect(manager.count()).toBe(1);
    });

    it('should support multiple listeners', () => {
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      manager.add(element, 'click', handler1);
      manager.add(element, 'mouseover', handler2);

      expect(manager.count()).toBe(2);
    });

    it('should support event listener options', () => {
      const handler = vi.fn();

      manager.add(element, 'click', handler, { once: true });

      element.click();
      element.click();

      // Should only be called once due to { once: true }
      expect(handler).toHaveBeenCalledOnce();
    });
  });

  describe('remove()', () => {
    it('should remove specific event listener', () => {
      const handler = vi.fn();

      manager.add(element, 'click', handler);
      manager.remove(element, 'click', handler);

      element.click();
      expect(handler).not.toHaveBeenCalled();
      expect(manager.count()).toBe(0);
    });

    it('should only remove matching listener', () => {
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      manager.add(element, 'click', handler1);
      manager.add(element, 'click', handler2);

      manager.remove(element, 'click', handler1);

      element.click();
      expect(handler1).not.toHaveBeenCalled();
      expect(handler2).toHaveBeenCalledOnce();
      expect(manager.count()).toBe(1);
    });

    it('removes capture-phase listeners by replaying the original options', () => {
      // addEventListener treats capture as part of the listener identity, so
      // a listener registered with `{ capture: true }` is *not* the same as
      // one registered without options. EventManager.remove() must look up
      // the stored options to make the removal stick.
      const removeSpy = vi.spyOn(element, 'removeEventListener');
      const handler = vi.fn();

      manager.add(element, 'click', handler, { capture: true });
      manager.remove(element, 'click', handler);

      // The third arg should be the stored options object.
      expect(removeSpy).toHaveBeenCalledWith(
        'click',
        handler,
        expect.objectContaining({ capture: true })
      );
      expect(manager.count()).toBe(0);
      removeSpy.mockRestore();
    });
  });

  describe('removeAll()', () => {
    it('should remove all tracked listeners', () => {
      const handler1 = vi.fn();
      const handler2 = vi.fn();
      const handler3 = vi.fn();

      manager.add(element, 'click', handler1);
      manager.add(element, 'mouseover', handler2);
      manager.add(element, 'mouseout', handler3);

      expect(manager.count()).toBe(3);

      manager.removeAll();

      expect(manager.count()).toBe(0);

      // None should fire after removeAll
      element.click();
      element.dispatchEvent(new Event('mouseover'));
      element.dispatchEvent(new Event('mouseout'));

      expect(handler1).not.toHaveBeenCalled();
      expect(handler2).not.toHaveBeenCalled();
      expect(handler3).not.toHaveBeenCalled();
    });

    it('should be safe to call multiple times', () => {
      const handler = vi.fn();
      manager.add(element, 'click', handler);

      manager.removeAll();
      manager.removeAll(); // Should not throw

      expect(manager.count()).toBe(0);
    });

    // ui.md O9 / Phase E15: previously `'should work with no listeners added'`
    // — vague (P9). Rename to surface the actual contract: removeAll()
    // on an empty manager is a no-throw no-op and leaves the count at 0.
    it('removeAll() is a no-throw no-op on a freshly-constructed manager', () => {
      expect(() => {
        manager.removeAll();
      }).not.toThrow();

      expect(manager.count()).toBe(0);
    });
  });

  describe('count()', () => {
    it('should return 0 initially', () => {
      expect(manager.count()).toBe(0);
    });

    it('should track listener count accurately', () => {
      const handler = vi.fn();

      expect(manager.count()).toBe(0);

      manager.add(element, 'click', handler);
      expect(manager.count()).toBe(1);

      manager.add(element, 'mouseover', handler);
      expect(manager.count()).toBe(2);

      manager.remove(element, 'click', handler);
      expect(manager.count()).toBe(1);

      manager.removeAll();
      expect(manager.count()).toBe(0);
    });
  });

  describe('Memory Leak Prevention', () => {
    it('should prevent memory leaks with bind()', () => {
      const obj = {
        value: 0,
        handler() {
          this.value++;
        },
      };

      // This is the pattern that causes leaks if not tracked properly
      manager.add(element, 'click', obj.handler.bind(obj));

      element.click();
      expect(obj.value).toBe(1);

      // Critical: removeAll should remove the bound handler
      manager.removeAll();

      // After removeAll, handler should not fire
      element.click();
      expect(obj.value).toBe(1); // Still 1, not 2
    });

    it('should handle multiple elements', () => {
      const element2 = document.createElement('div');
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      manager.add(element, 'click', handler1);
      manager.add(element2, 'click', handler2);

      expect(manager.count()).toBe(2);

      manager.removeAll();

      element.click();
      element2.click();

      expect(handler1).not.toHaveBeenCalled();
      expect(handler2).not.toHaveBeenCalled();
      expect(manager.count()).toBe(0);
    });

    it('should handle window and document listeners', () => {
      const windowHandler = vi.fn();
      const documentHandler = vi.fn();

      manager.add(window, 'resize', windowHandler);
      manager.add(document, 'keydown', documentHandler);

      expect(manager.count()).toBe(2);

      window.dispatchEvent(new Event('resize'));
      document.dispatchEvent(new KeyboardEvent('keydown'));

      expect(windowHandler).toHaveBeenCalledOnce();
      expect(documentHandler).toHaveBeenCalledOnce();

      manager.removeAll();

      window.dispatchEvent(new Event('resize'));
      document.dispatchEvent(new KeyboardEvent('keydown'));

      // Should still be called only once (not twice)
      expect(windowHandler).toHaveBeenCalledOnce();
      expect(documentHandler).toHaveBeenCalledOnce();
      expect(manager.count()).toBe(0);
    });
  });
});
