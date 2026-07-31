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
  // Use the public teardown so module-level listener/timer state cannot leak
  // across tests even when a case fails before its own cleanup.
  hideHelpOverlay();

  // Clear any unrelated pending timers before teardown.
  vi.clearAllTimers();
  vi.useRealTimers();

  document.body.innerHTML = '';
});

describe('showHelpOverlay - Memory Leak Prevention', () => {
  it('should not create multiple overlays when called repeatedly', () => {
    showHelpOverlay();
    const firstOverlay = document.getElementById('luxar-help-overlay');
    // Audit W3 fix: pin id + role so a mutant that returns the wrong
    // element from getElementById would surface here.
    expect(firstOverlay?.id).toBe('luxar-help-overlay');
    expect(firstOverlay?.getAttribute('role')).toBe('dialog');

    // Try to create another one
    showHelpOverlay();
    const allOverlays = document.querySelectorAll('#luxar-help-overlay');

    // Should still be only one
    expect(allOverlays.length).toBe(1);
    expect(document.getElementById('luxar-help-overlay')).toBe(firstOverlay);
  });

  // [ui.md/C6 / W9] Verifies the OBSERVABLE outside-click-cleanup
  // contract rather than spying on `removeEventListener` for the literal
  // string 'click'. The two tests below force the strongest case: after
  // hide-then-show, an outside click within the post-show delay must
  // NOT remove the new overlay. If the old overlay's `handleDocumentClick`
  // closure had leaked, it would fire on the body click and call
  // `closeHelp` which removes the new overlay by id. The new overlay
  // staying present therefore proves the prior listener was removed.
  //
  // A regression that switched outside-click dismissal from 'click' to
  // 'mousedown' / 'pointerdown' would NOT slip through here the way the
  // previous removeEventListener('click', ...) spy allowed.
  it('hideHelpOverlay cancels delayed listener registration', () => {
    showHelpOverlay();
    hideHelpOverlay();

    expect(vi.getTimerCount()).toBe(0);
  });

  it('close button cancels delayed listener registration', () => {
    showHelpOverlay();
    const closeBtn = document.querySelector('button[title="Close (Escape)"]') as HTMLButtonElement;

    closeBtn.click();

    expect(document.getElementById('luxar-help-overlay')).toBeNull();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('rapid hide then show cannot attach the stale overlay listener', () => {
    showHelpOverlay();
    vi.advanceTimersByTime(20);
    hideHelpOverlay();

    showHelpOverlay();
    const reopened = document.getElementById('luxar-help-overlay');
    expect(reopened).toBeTruthy();

    // Reach the first overlay's original 100 ms deadline, but not the
    // reopened overlay's. A stale handler would treat this inside click as
    // outside its detached element and close the newly opened overlay.
    vi.advanceTimersByTime(80);
    reopened?.click();

    expect(document.getElementById('luxar-help-overlay')).toBe(reopened);
  });

  it('only outside clicks dismiss the current overlay after the delay', () => {
    showHelpOverlay();
    const overlay = document.getElementById('luxar-help-overlay');
    expect(overlay).toBeTruthy();
    vi.advanceTimersByTime(150);

    overlay?.click();
    expect(document.getElementById('luxar-help-overlay')).toBe(overlay);

    document.body.click();
    expect(document.getElementById('luxar-help-overlay')).toBeNull();
  });

  it('hideHelpOverlay removes the outside-click listener (observable contract)', () => {
    showHelpOverlay();
    // Audit W3 fix: pin id so a wrong-element bug surfaces here.
    expect(document.getElementById('luxar-help-overlay')?.id).toBe('luxar-help-overlay');
    vi.advanceTimersByTime(150); // Listener for overlay-1 installed.

    hideHelpOverlay();
    expect(document.getElementById('luxar-help-overlay')).toBeNull();

    // Open a fresh overlay. If overlay-1's listener leaked, the next
    // body click (before overlay-2's own delayed listener installs)
    // will invoke overlay-1's `closeHelp` and remove overlay-2 by id.
    showHelpOverlay();
    expect(document.getElementById('luxar-help-overlay')).toBeTruthy();

    // Click BEFORE the 150ms delay so overlay-2's own listener is not
    // yet installed — any close that happens MUST be from a leaked
    // overlay-1 handler.
    document.body.click();

    expect(document.getElementById('luxar-help-overlay')).toBeTruthy();
  });

  it('close-button click removes the outside-click listener (observable contract)', () => {
    showHelpOverlay();
    vi.advanceTimersByTime(150); // Listener installed for overlay-1.

    const closeBtn = document.querySelector('button[title="Close (Escape)"]') as HTMLButtonElement;
    // Audit W3 fix: pin tag + title so a wrong-target query won't pass.
    expect(closeBtn?.tagName).toBe('BUTTON');
    expect(closeBtn?.getAttribute('title')).toBe('Close (Escape)');
    closeBtn.click(); // Close via close-button path.

    expect(document.getElementById('luxar-help-overlay')).toBeNull();

    // Fresh overlay; same leakage probe as above.
    showHelpOverlay();
    expect(document.getElementById('luxar-help-overlay')).toBeTruthy();
    document.body.click();
    expect(document.getElementById('luxar-help-overlay')).toBeTruthy();
  });

  it('should have proper ARIA attributes', () => {
    showHelpOverlay();
    const overlay = document.getElementById('luxar-help-overlay');

    expect(overlay?.getAttribute('role')).toBe('dialog');
    expect(overlay?.getAttribute('aria-modal')).toBe('true');
    expect(overlay?.getAttribute('aria-labelledby')).toBe('luxar-help-overlay-title');

    const title = document.getElementById('luxar-help-overlay-title');
    // Audit W3 fix: pin id + non-empty text so a returned-wrong-id
    // mutation would surface here.
    expect(title?.id).toBe('luxar-help-overlay-title');
    expect(title?.textContent).toContain('Luxar Controls');
  });
});

describe('hideHelpOverlay', () => {
  it('should remove help overlay', () => {
    showHelpOverlay();
    // Audit W3 fix: assert the element is the expected element by id
    // before tearing it down — a wrong-element bug would surface.
    const opened = document.getElementById('luxar-help-overlay');
    expect(opened?.id).toBe('luxar-help-overlay');

    hideHelpOverlay();
    expect(document.getElementById('luxar-help-overlay')).toBeNull();
  });

  it('should be safe to call when no overlay exists', () => {
    expect(() => hideHelpOverlay()).not.toThrow();
  });
});
