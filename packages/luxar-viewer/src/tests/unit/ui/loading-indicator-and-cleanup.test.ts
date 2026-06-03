/**
 * Unit tests for the small orphan modules `ui/loading-indicator.ts` and
 * `ui/ui-cleanup.ts`.
 *
 * Audit reference: ui.md G11 — both modules previously had no tests.
 * They are tiny DOM helpers used at app startup/teardown; we verify the
 * observable DOM mutation contract (element with the right id appears /
 * disappears) and idempotency under repeated calls.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// help-overlay is a sibling module — we don't mock it, but it has its own
// click listener teardown when not currently shown; calling hideHelpOverlay
// when no overlay exists is a no-op (per its README).
vi.mock('../../../ui/help-overlay', () => ({
  hideHelpOverlay: vi.fn(),
}));

import { showLoadingIndicator, hideLoadingIndicator } from '../../../ui/loading-indicator';
import { cleanupUI } from '../../../ui/ui-cleanup';
import { hideHelpOverlay } from '../../../ui/help-overlay';

beforeEach(() => {
  document.body.innerHTML = '';
  vi.mocked(hideHelpOverlay).mockClear();
});

afterEach(() => {
  document.body.innerHTML = '';
});

describe('loading-indicator', () => {
  it('showLoadingIndicator appends a uniquely-id-ed element with spinner + text', () => {
    const el = showLoadingIndicator();

    expect(el.id).toBe('luxar-loading-indicator');
    expect(el.className).toBe('luxar-loading-indicator');
    expect(el.parentNode).toBe(document.body);

    const spinner = el.querySelector('.luxar-loading-indicator__spinner');
    const text = el.querySelector('.luxar-loading-indicator__text');
    expect(spinner).toBeTruthy();
    expect(text?.textContent).toBe('Loading scene...');
    expect(text?.id).toBe('luxar-loading-text');

    // The returned reference IS the DOM element (used by callers to swap text).
    expect(document.getElementById('luxar-loading-indicator')).toBe(el);
  });

  it('hideLoadingIndicator removes the element', () => {
    showLoadingIndicator();
    expect(document.getElementById('luxar-loading-indicator')).not.toBeNull();

    hideLoadingIndicator();
    expect(document.getElementById('luxar-loading-indicator')).toBeNull();
  });

  it('hideLoadingIndicator is a no-op when no indicator exists', () => {
    expect(() => hideLoadingIndicator()).not.toThrow();
    expect(document.getElementById('luxar-loading-indicator')).toBeNull();
  });

  it('repeated show calls append multiple indicators (caller responsibility)', () => {
    // P5 boundary: the source does not deduplicate. We pin the actual
    // contract so a future "auto-dedupe" change is a conscious choice.
    showLoadingIndicator();
    showLoadingIndicator();
    expect(document.querySelectorAll('.luxar-loading-indicator').length).toBe(2);
  });
});

describe('cleanupUI', () => {
  it('removes the luxar-loading-indicator if present', () => {
    showLoadingIndicator();
    expect(document.getElementById('luxar-loading-indicator')).not.toBeNull();

    cleanupUI();

    expect(document.getElementById('luxar-loading-indicator')).toBeNull();
  });

  it('removes the luxar-error-message if present', () => {
    const errEl = document.createElement('div');
    errEl.id = 'luxar-error-message';
    document.body.appendChild(errEl);

    cleanupUI();

    expect(document.getElementById('luxar-error-message')).toBeNull();
  });

  it('invokes hideHelpOverlay', () => {
    cleanupUI();
    expect(hideHelpOverlay).toHaveBeenCalledTimes(1);
  });

  it('is idempotent — second call with nothing present is a no-op', () => {
    cleanupUI();
    cleanupUI();

    // hideHelpOverlay invoked once per cleanupUI call.
    expect(hideHelpOverlay).toHaveBeenCalledTimes(2);
    expect(document.getElementById('luxar-loading-indicator')).toBeNull();
    expect(document.getElementById('luxar-error-message')).toBeNull();
  });

  it('clears both indicator and error-message in the same pass', () => {
    showLoadingIndicator();
    const errEl = document.createElement('div');
    errEl.id = 'luxar-error-message';
    document.body.appendChild(errEl);

    // Add some unrelated DOM that must NOT be cleaned up.
    const sentinel = document.createElement('div');
    sentinel.id = 'unrelated';
    document.body.appendChild(sentinel);

    cleanupUI();

    expect(document.getElementById('luxar-loading-indicator')).toBeNull();
    expect(document.getElementById('luxar-error-message')).toBeNull();
    expect(document.getElementById('unrelated')).toBe(sentinel);
  });
});
