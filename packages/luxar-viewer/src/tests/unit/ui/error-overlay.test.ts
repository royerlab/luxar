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

    expect(title).toBeTruthy();
    expect(message).toBeTruthy();
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
    expect(document.getElementById('luxar-error-message')).toBeTruthy();

    clearError();
    expect(document.getElementById('luxar-error-message')).toBeNull();
  });

  it('should be safe to call when no error exists', () => {
    expect(() => clearError()).not.toThrow();
  });
});
