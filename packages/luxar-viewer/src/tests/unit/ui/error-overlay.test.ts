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
  // Clean up error messages
  const errorMessage = document.getElementById('luxar-error-message');
  if (errorMessage) {
    errorMessage.remove();
  }

  // Audit G17 (viewer-ui-config-themes-core-utils): pinning a
  // strict `vi.getTimerCount() === 0` here surfaces a real timer
  // leak in `ui/error-overlay` (showError schedules a setTimeout
  // for the auto-clear flow that is never cancelled when the
  // overlay is removed). That's a production issue, not a test
  // issue. The leak is tracked in `delme/audit-tracking.md` as
  // an OPEN G-tier item. Pinning the strict count here would
  // fail every test in this file; defer to a follow-up that fixes
  // the production teardown.

  // Clear all pending timers before teardown
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
    expect(() => clearError()).not.toThrow();
  });
});
