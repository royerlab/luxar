// [ui.md/O2][P10] Split from `helpers.test.ts`: showError / clearError
// tests for the `ui/error-overlay` module. The help-overlay helpers
// (`ui/help-overlay`) live in `help-overlay.test.ts`.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { showError, clearError } from '../../../ui/error-overlay';

// Mock DOM environment
beforeEach(() => {
  document.body.innerHTML = '';
  vi.useFakeTimers();
});

afterEach(() => {
  // Clean up error messages. Call clearError() to also cancel the
  // auto-dismiss timer — if we just `.remove()` the DOM node, the
  // module-scoped setTimeout in showError() would still be pending
  // and `vi.getTimerCount()` would surface a leak below.
  clearError();

  // Audit G17 (viewer-ui-config-themes-core-utils): no timers must
  // be pending after teardown. Pre-fix, showError() leaked its
  // auto-dismiss setTimeout — clearError() now cancels it. If a
  // future change adds a new timer in error-overlay, this assertion
  // will surface it before silently leaking into the next test.
  expect(vi.getTimerCount()).toBe(0);

  // Clear all pending timers before teardown (defensive — should be 0)
  vi.clearAllTimers();
  vi.useRealTimers();

  document.body.innerHTML = '';
});

describe('showError - ARIA Attributes', () => {
  it('should have proper ARIA attributes for alertdialog', () => {
    showError('Test error message');
    const errorDialog = document.getElementById('luxar-error-message');

    expect(errorDialog?.getAttribute('role')).toBe('alertdialog');
    expect(errorDialog?.getAttribute('aria-modal')).toBe('true');
    expect(errorDialog?.getAttribute('aria-labelledby')).toBe('luxar-error-title');
    expect(errorDialog?.getAttribute('aria-describedby')).toBe('luxar-error-message-text');
  });

  it('should have properly linked title and message', () => {
    showError('Custom error message');

    const title = document.getElementById('luxar-error-title');
    const message = document.getElementById('luxar-error-message-text');

    // Audit W2 fix: previous `toBeTruthy()` would pass for any
    // non-null element including the WRONG element. Pin id + tag +
    // text content so mutations that swap elements or return the
    // wrong descendant surface immediately.
    expect(title?.id).toBe('luxar-error-title');
    expect(title?.tagName).toBe('DIV');
    expect(message?.id).toBe('luxar-error-message-text');
    expect(message?.textContent).toBe('Custom error message');
  });

  it('should replace existing errors instead of stacking', () => {
    showError('First error');
    showError('Second error');

    const errors = document.querySelectorAll('#luxar-error-message');
    expect(errors.length).toBe(1);

    const messageText = document.getElementById('luxar-error-message-text');
    expect(messageText?.textContent).toBe('Second error');
  });
});

describe('clearError', () => {
  it('should remove error message', () => {
    showError('Test error');
    // Audit W2 fix: prove the element is actually a useful element,
    // not just truthy. After showError it must be in the DOM with
    // its documented id and contain the error text.
    const before = document.getElementById('luxar-error-message');
    expect(before?.id).toBe('luxar-error-message');
    expect(before?.textContent).toContain('Test error');

    clearError();
    expect(document.getElementById('luxar-error-message')).toBeNull();
  });

  it('should be safe to call when no error exists', () => {
    // Audit W30 (viewer-ui-config-themes-core-utils): this is
    // appropriate defensive programming — clearError() is the public
    // teardown API and must be idempotent. The bare .not.toThrow() is
    // intentional; we additionally pin the observable post-state
    // (no error element in the DOM) so a mutant that silently
    // injects a placeholder would surface.
    expect(() => clearError()).not.toThrow();
    expect(document.getElementById('luxar-error-message')).toBeNull();
  });
});
