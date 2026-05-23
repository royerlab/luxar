// [ui.md/O2][P10] Split from `helpers.test.ts`: showHelpOverlay /
// hideHelpOverlay tests for the `ui/help-overlay` module. The error-overlay
// helpers (`ui/error-overlay`) live in `error-overlay.test.ts`.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { showHelpOverlay, hideHelpOverlay } from '../../../ui/help-overlay';

// Mock DOM environment
beforeEach(() => {
  document.body.innerHTML = '';
  vi.useFakeTimers();
});

afterEach(() => {
  // Clean up any help overlays
  const helpOverlay = document.getElementById('luxar-help-overlay');
  if (helpOverlay) {
    helpOverlay.remove();
  }

  // Clear all pending timers before teardown
  vi.clearAllTimers();
  vi.useRealTimers();

  document.body.innerHTML = '';
});

describe('showHelpOverlay - Memory Leak Prevention', () => {
  it('should not create multiple overlays when called repeatedly', () => {
    showHelpOverlay();
    const firstOverlay = document.getElementById('luxar-help-overlay');
    expect(firstOverlay).toBeTruthy();

    // Try to create another one
    showHelpOverlay();
    const allOverlays = document.querySelectorAll('#luxar-help-overlay');

    // Should still be only one
    expect(allOverlays.length).toBe(1);
    expect(document.getElementById('luxar-help-overlay')).toBe(firstOverlay);
  });

  it('should properly clean up global click listener when closed via hideHelpOverlay', () => {
    const removeEventSpy = vi.spyOn(document, 'removeEventListener');

    showHelpOverlay();
    const overlay = document.getElementById('luxar-help-overlay');
    expect(overlay).toBeTruthy();

    // Advance timers to trigger the click listener addition
    vi.advanceTimersByTime(150);

    // Close via hideHelpOverlay (how InputHandler closes it on Escape/H)
    hideHelpOverlay();

    // Check overlay is removed
    expect(document.getElementById('luxar-help-overlay')).toBeNull();

    // Verify removeEventListener was called for 'click'
    const clickRemovals = removeEventSpy.mock.calls.filter((call) => call[0] === 'click');
    expect(clickRemovals.length).toBeGreaterThanOrEqual(1);

    removeEventSpy.mockRestore();
  });

  it('should properly clean up when close button is clicked', () => {
    const removeEventSpy = vi.spyOn(document, 'removeEventListener');

    showHelpOverlay();

    // Find the close button
    const closeBtn = document.querySelector(
      'button[title="Close (Escape)"]'
    ) as HTMLButtonElement;
    expect(closeBtn).toBeTruthy();

    // Advance timers to trigger the click listener addition
    vi.advanceTimersByTime(150);

    // Click close button
    closeBtn?.click();

    // Overlay should be removed
    expect(document.getElementById('luxar-help-overlay')).toBeNull();

    // Verify removeEventListener was called for 'click'
    const clickRemovals = removeEventSpy.mock.calls.filter((call) => call[0] === 'click');
    expect(clickRemovals.length).toBeGreaterThanOrEqual(1);

    removeEventSpy.mockRestore();
  });

  it('should have proper ARIA attributes', () => {
    showHelpOverlay();
    const overlay = document.getElementById('luxar-help-overlay');

    expect(overlay?.getAttribute('role')).toBe('dialog');
    expect(overlay?.getAttribute('aria-modal')).toBe('true');
    expect(overlay?.getAttribute('aria-labelledby')).toBe('luxar-help-overlay-title');

    const title = document.getElementById('luxar-help-overlay-title');
    expect(title).toBeTruthy();
    expect(title?.textContent).toContain('Luxar Controls');
  });
});

describe('hideHelpOverlay', () => {
  it('should remove help overlay', () => {
    showHelpOverlay();
    expect(document.getElementById('luxar-help-overlay')).toBeTruthy();

    hideHelpOverlay();
    expect(document.getElementById('luxar-help-overlay')).toBeNull();
  });

  it('should be safe to call when no overlay exists', () => {
    expect(() => hideHelpOverlay()).not.toThrow();
  });
});
