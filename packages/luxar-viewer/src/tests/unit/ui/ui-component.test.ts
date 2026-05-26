/**
 * Unit tests for UIComponent base class
 * Tests lifecycle management, event listeners, theme subscription, and disposal
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { UIComponent } from '../../../ui/overlay-widgets/ui-component';
import { ThemeManager } from '../../../themes/theme-manager';
import type { Theme } from '../../../themes/types';

// Mock component for testing (using instance properties set in constructor)
class TestComponent extends UIComponent<{ title: string }> {
  public renderCalled = false;
  public attachEventListenersCalled = false;
  public onThemeChangeCalled = false;
  public onDisposeCalled = false;
  public lastTheme: Theme | null = null;

  protected getClassName(): string {
    return 'test-component';
  }

  protected render(): HTMLElement {
    // Set flag when render is called
    (this as any)._renderCalled = true;

    const el = document.createElement('div');
    el.className = this.getClassName();
    el.textContent = this.config.title;
    return el;
  }

  protected attachEventListeners(): void {
    // Set flag when attachEventListeners is called
    (this as any)._attachCalled = true;
  }

  protected onThemeChange(theme: Theme): void {
    this.onThemeChangeCalled = true;
    this.lastTheme = theme;
  }

  protected onDispose(): void {
    this.onDisposeCalled = true;
  }

  // Getter to check if render was called (works around initialization order)
  public get wasRenderCalled(): boolean {
    return (this as any)._renderCalled === true;
  }

  public get wasAttachEventListenersCalled(): boolean {
    return (this as any)._attachCalled === true;
  }

  // Expose protected methods for testing
  public addTestEventListener<K extends keyof HTMLElementEventMap>(
    target: EventTarget,
    type: K,
    listener: (ev: HTMLElementEventMap[K]) => void
  ): void {
    this.addEventListener(target, type, listener);
  }

  public removeTestEventListener<K extends keyof HTMLElementEventMap>(
    target: EventTarget,
    type: K
  ): void {
    this.removeEventListener(target, type);
  }
}

describe('UIComponent', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
    ThemeManager.disposeInstance();
  });

  afterEach(() => {
    document.body.innerHTML = '';
    ThemeManager.disposeInstance();
  });

  describe('Lifecycle', () => {
    it('should call render() during construction', () => {
      const component = new TestComponent({ title: 'Test' });
      // Verify element exists (proof that render was called)
      expect(component.getElement()).toBeInstanceOf(HTMLElement);
      expect(component.getElement().textContent).toBe('Test');
      expect(component.wasRenderCalled).toBe(true);
    });

    it('should call attachEventListeners() after render()', () => {
      const component = new TestComponent({ title: 'Test' });
      // Verify attachEventListeners was called
      expect(component.wasAttachEventListenersCalled).toBe(true);
    });

    it('should create element with correct className', () => {
      const component = new TestComponent({ title: 'Test' });
      const element = component.getElement();

      expect(element).toBeInstanceOf(HTMLElement);
      expect(element.className).toContain('test-component');
    });

    it('should store config', () => {
      const config = { title: 'Test Component' };
      const component = new TestComponent(config);

      expect((component as any).config).toBe(config);
    });
  });

  describe('Visibility Management', () => {
    it('should not be visible initially', () => {
      const component = new TestComponent({ title: 'Test' });
      expect(component.isVisible()).toBe(false);
    });

    it('should be visible after show()', () => {
      const component = new TestComponent({ title: 'Test' });
      component.show();

      expect(component.isVisible()).toBe(true);
      expect(component.getElement().classList.contains('test-component--visible')).toBe(true);
    });

    it('should append element to body when show() is called', () => {
      const component = new TestComponent({ title: 'Test' });
      component.show();

      expect(document.body.contains(component.getElement())).toBe(true);
    });

    it('should hide after hide() is called', () => {
      const component = new TestComponent({ title: 'Test' });
      component.show();
      component.hide();

      expect(component.isVisible()).toBe(false);
      expect(component.getElement().classList.contains('test-component--visible')).toBe(false);
    });

    it('toggle() flips visibility AND mirrors it in the visible-modifier class + body-attached status', () => {
      // [ui.md/W6][P2] Previously asserted only isVisible() boolean flip.
      // Strengthen by pinning the three observables in lockstep:
      // (1) isVisible() return
      // (2) CSS class 'test-component--visible' presence
      // (3) DOM attachment to document.body
      const component = new TestComponent({ title: 'Test' });
      const el = component.getElement();
      const visibleClass = 'test-component--visible';

      expect(component.isVisible()).toBe(false);
      expect(el.classList.contains(visibleClass)).toBe(false);
      expect(document.body.contains(el)).toBe(false);

      component.toggle();
      expect(component.isVisible()).toBe(true);
      expect(el.classList.contains(visibleClass)).toBe(true);
      expect(document.body.contains(el)).toBe(true);

      component.toggle();
      expect(component.isVisible()).toBe(false);
      expect(el.classList.contains(visibleClass)).toBe(false);
      // Element reference identity preserved across toggle cycles.
      expect(component.getElement()).toBe(el);
    });
  });

  describe('Event Listener Management', () => {
    it('should add event listeners', () => {
      const component = new TestComponent({ title: 'Test' });
      const handler = vi.fn();

      component.addTestEventListener(component.getElement(), 'click', handler);
      component.getElement().click();

      expect(handler).toHaveBeenCalledTimes(1);
    });

    it('should remove event listeners', () => {
      const component = new TestComponent({ title: 'Test' });
      const handler = vi.fn();

      component.addTestEventListener(component.getElement(), 'click', handler);
      component.removeTestEventListener(component.getElement(), 'click');
      component.getElement().click();

      expect(handler).not.toHaveBeenCalled();
    });

    it('should prevent duplicate event listeners', () => {
      const component = new TestComponent({ title: 'Test' });
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      // Add first listener
      component.addTestEventListener(component.getElement(), 'click', handler1);

      // Add second listener for same event (should replace first)
      component.addTestEventListener(component.getElement(), 'click', handler2);

      component.getElement().click();

      // Only handler2 should be called (handler1 was replaced)
      expect(handler1).not.toHaveBeenCalled();
      expect(handler2).toHaveBeenCalledTimes(1);
    });

    it('should clean up all event listeners on dispose', () => {
      const component = new TestComponent({ title: 'Test' });
      const clickHandler = vi.fn();
      const keydownHandler = vi.fn();

      component.addTestEventListener(component.getElement(), 'click', clickHandler);
      component.addTestEventListener(component.getElement(), 'keydown', keydownHandler);

      component.dispose();

      // Dispatch events manually since element is removed from DOM
      const clickEvent = new MouseEvent('click', { bubbles: true });
      const keyEvent = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true });

      component.getElement().dispatchEvent(clickEvent);
      component.getElement().dispatchEvent(keyEvent);

      // Handlers should not be called after disposal
      expect(clickHandler).not.toHaveBeenCalled();
      expect(keydownHandler).not.toHaveBeenCalled();
    });
  });

  describe('Theme Subscription', () => {
    it('should call onThemeChange when theme switches', () => {
      const component = new TestComponent({ title: 'Test' });
      const themeManager = ThemeManager.getInstance();

      component.onThemeChangeCalled = false; // Reset flag after construction

      themeManager.setTheme('light');

      expect(component.onThemeChangeCalled).toBe(true);
      expect(component.lastTheme?.id).toBe('light');
    });

    it('should unsubscribe from theme changes on dispose', () => {
      const component = new TestComponent({ title: 'Test' });
      const themeManager = ThemeManager.getInstance();

      component.dispose();
      component.onThemeChangeCalled = false;

      themeManager.setTheme('light');

      // Should not be called after disposal
      expect(component.onThemeChangeCalled).toBe(false);
    });
  });

  describe('Disposal', () => {
    it('should call onDispose hook', () => {
      const component = new TestComponent({ title: 'Test' });
      component.dispose();

      expect(component.onDisposeCalled).toBe(true);
    });

    it('should remove element from DOM', () => {
      const component = new TestComponent({ title: 'Test' });
      component.show();

      expect(document.body.contains(component.getElement())).toBe(true);

      component.dispose();

      expect(document.body.contains(component.getElement())).toBe(false);
    });

    it('should be safe to call dispose multiple times', () => {
      const component = new TestComponent({ title: 'Test' });

      expect(() => {
        component.dispose();
        component.dispose();
      }).not.toThrow();
    });
  });

  describe('Integration', () => {
    // ui.md O8 / Phase E15: previously `'should work with event handlers
    // and maintain proper cleanup'` — vague (P9). Rename to surface the
    // actual contract: dispose() detaches addTestEventListener-registered
    // handlers so subsequent clicks on the same element no longer fire
    // them.
    it('dispose() detaches event listeners so subsequent clicks no longer fire registered handlers', () => {
      const component = new TestComponent({ title: 'Integration Test' });
      const clickHandler = vi.fn();

      // Add event listener
      component.addTestEventListener(component.getElement(), 'click', clickHandler);

      // Show component (add to DOM)
      component.show();

      // Trigger click
      component.getElement().click();
      expect(clickHandler).toHaveBeenCalledTimes(1);

      // Dispose should remove all listeners
      component.dispose();

      // Click after dispose should not trigger handler
      const el = component.getElement();
      el.click();
      expect(clickHandler).toHaveBeenCalledTimes(1); // Still 1, not 2
    });
  });
});
