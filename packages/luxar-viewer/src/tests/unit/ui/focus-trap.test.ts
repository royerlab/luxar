// @vitest-environment jsdom
/**
 * Unit tests for `ui/help-overlay/focus-trap.ts`.
 *
 * The trap had no direct coverage at all — it was only ever exercised
 * indirectly through the help overlay and the dataset browser, neither of
 * which pins the wrap branches, the `autoFocusFirst` toggle, or the
 * visibility filtering. Everything here is real DOM: a real container, real
 * `KeyboardEvent`s, and assertions on `document.activeElement`.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { trapFocus } from '../../../ui/help-overlay/focus-trap';

interface Panel {
  container: HTMLElement;
  first: HTMLButtonElement;
  last: HTMLButtonElement;
  /** Focusable but inside a `display: none` row, and LAST in document order. */
  hiddenInput: HTMLInputElement;
  hiddenRow: HTMLElement;
  /** Inside the panel, focusable only programmatically (not in the trap's list). */
  inert: HTMLElement;
  outside: HTMLButtonElement;
}

/**
 * Build a panel shaped like the dataset browser while a directory loads: two
 * visible buttons, then a hidden search row whose `<input>` still matches the
 * focusable selector and is the LAST match.
 */
function buildPanel(): Panel {
  const outside = document.createElement('button');
  outside.textContent = 'outside';
  document.body.appendChild(outside);

  const container = document.createElement('div');
  container.tabIndex = -1;

  const first = document.createElement('button');
  first.textContent = 'first';
  const last = document.createElement('button');
  last.textContent = 'last';

  const hiddenRow = document.createElement('div');
  hiddenRow.style.display = 'none';
  const hiddenInput = document.createElement('input');
  hiddenInput.type = 'text';
  hiddenRow.appendChild(hiddenInput);

  const inert = document.createElement('div');
  inert.tabIndex = -1;

  container.append(first, last, hiddenRow, inert);
  document.body.appendChild(container);

  return { container, first, last, hiddenInput, hiddenRow, inert, outside };
}

function pressTab(target: HTMLElement, shiftKey = false): KeyboardEvent {
  const event = new KeyboardEvent('keydown', {
    key: 'Tab',
    shiftKey,
    bubbles: true,
    cancelable: true,
  });
  target.dispatchEvent(event);
  return event;
}

describe('trapFocus', () => {
  let panel: Panel;
  let release: (() => void) | undefined;

  beforeEach(() => {
    document.body.innerHTML = '';
    vi.useFakeTimers();
    panel = buildPanel();
  });

  afterEach(() => {
    release?.();
    release = undefined;
    vi.clearAllTimers();
    vi.useRealTimers();
    document.body.innerHTML = '';
  });

  describe('initial focus', () => {
    it('focuses the first rendered focusable by default', () => {
      release = trapFocus(panel.container);

      // Scheduled on a 0 ms timer, so nothing has moved yet.
      expect(document.activeElement).not.toBe(panel.first);
      vi.advanceTimersByTime(0);
      expect(document.activeElement).toBe(panel.first);
    });

    it('skips a hidden element when picking the first focusable', () => {
      // Hide the first button: the initial pick must fall through to the next
      // RENDERED focusable rather than focusing something invisible.
      panel.first.style.display = 'none';

      release = trapFocus(panel.container);
      vi.advanceTimersByTime(0);

      expect(document.activeElement).toBe(panel.last);
    });

    it('moves nothing when autoFocusFirst is false', () => {
      panel.outside.focus();

      release = trapFocus(panel.container, { autoFocusFirst: false });
      vi.advanceTimersByTime(50);

      // The filtered panels park focus on the container themselves; the trap
      // must not fight them by grabbing the close button a tick later.
      expect(document.activeElement).toBe(panel.outside);
    });

    it('is a no-op on a container with no focusable children', () => {
      const empty = document.createElement('div');
      document.body.appendChild(empty);
      panel.outside.focus();

      release = trapFocus(empty);
      vi.advanceTimersByTime(50);
      const event = pressTab(empty);

      expect(document.activeElement).toBe(panel.outside);
      expect(event.defaultPrevented).toBe(false);
    });

    it('cancels the pending focus timer when released first (audit G17)', () => {
      panel.outside.focus();
      const before = vi.getTimerCount();

      const releaseNow = trapFocus(panel.container);
      expect(vi.getTimerCount()).toBe(before + 1);
      releaseNow();

      expect(vi.getTimerCount()).toBe(before);
      vi.advanceTimersByTime(50);
      expect(document.activeElement).toBe(panel.outside);
    });
  });

  describe('Tab cycling', () => {
    beforeEach(() => {
      release = trapFocus(panel.container, { autoFocusFirst: false });
    });

    it('wraps Tab from the last rendered focusable to the first', () => {
      panel.last.focus();

      const event = pressTab(panel.last);

      expect(event.defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(panel.first);
    });

    it('wraps Shift+Tab from the first focusable to the last RENDERED one', () => {
      panel.first.focus();

      const event = pressTab(panel.first, true);

      // Not `hiddenInput`, even though it is the last selector match: it lives
      // in a `display: none` row, and `.focus()` on it is a no-op in a real
      // browser, which would strand focus.
      expect(event.defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(panel.last);
      expect(document.activeElement).not.toBe(panel.hiddenInput);
    });

    it('leaves a mid-list Tab to the browser', () => {
      panel.first.focus();

      const event = pressTab(panel.first);

      expect(event.defaultPrevented).toBe(false);
      expect(document.activeElement).toBe(panel.first);
    });

    it('ignores non-Tab keys', () => {
      panel.last.focus();

      const event = new KeyboardEvent('keydown', {
        key: 'Enter',
        bubbles: true,
        cancelable: true,
      });
      panel.last.dispatchEvent(event);

      expect(event.defaultPrevented).toBe(false);
      expect(document.activeElement).toBe(panel.last);
    });

    it('picks up focusables revealed after install', () => {
      // The listing arrives asynchronously: the trap must re-read the DOM on
      // every Tab rather than caching the list at install time.
      panel.hiddenRow.style.display = '';
      panel.first.focus();

      pressTab(panel.first, true);

      expect(document.activeElement).toBe(panel.hiddenInput);
    });
  });

  describe('steering inward from a non-tabbable element', () => {
    beforeEach(() => {
      release = trapFocus(panel.container, { autoFocusFirst: false });
    });

    it('Tab from the container enters at the first focusable', () => {
      panel.container.focus();

      const event = pressTab(panel.container);

      expect(event.defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(panel.first);
    });

    it('Shift+Tab from the container enters at the last RENDERED focusable', () => {
      panel.container.focus();

      const event = pressTab(panel.container, true);

      // The whole point: the browser default would walk Shift+Tab straight out
      // of the modal, and an unfiltered list would aim it at a hidden input.
      expect(event.defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(panel.last);
    });

    it('steers a non-tabbable DESCENDANT inward too', () => {
      panel.inert.focus();

      const forward = pressTab(panel.inert);
      expect(forward.defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(panel.first);

      panel.inert.focus();
      const backward = pressTab(panel.inert, true);
      expect(backward.defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(panel.last);
    });
  });

  describe('cleanup', () => {
    it('restores focus to the previously-focused element', () => {
      panel.outside.focus();

      const releaseNow = trapFocus(panel.container, { autoFocusFirst: false });
      panel.last.focus();
      releaseNow();

      expect(document.activeElement).toBe(panel.outside);
    });

    it('removes the Tab listener', () => {
      const releaseNow = trapFocus(panel.container, { autoFocusFirst: false });
      releaseNow();

      panel.last.focus();
      const event = pressTab(panel.last);

      expect(event.defaultPrevented).toBe(false);
      expect(document.activeElement).toBe(panel.last);
    });
  });
});
