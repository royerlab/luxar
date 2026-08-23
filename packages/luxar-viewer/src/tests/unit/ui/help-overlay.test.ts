// @vitest-environment jsdom
// [ui.md/O2][P10] Split from `helpers.test.ts`: showHelpOverlay /
// hideHelpOverlay tests for the `ui/help-overlay` module. The error-overlay
// helpers (`ui/error-overlay`) live in `error-overlay.test.ts`.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { showHelpOverlay, hideHelpOverlay } from '../../../ui/help-overlay';
import { isTypingInInput } from '../../../input/input-handler/commands/focus-utils';

/**
 * Cost, in pending 0 ms timers, of ONE focus transition in this environment.
 * jsdom queues a `selectionchange` tick on every `focus()`
 * (`Selection-impl._associateRange`); a real browser queues nothing. Measured
 * per test rather than hardcoded, so the timer-leak guards below stay exact
 * without draining timers — draining a 0 ms tick would also retire the
 * untracked 0 ms timer those guards exist to catch (audit G17).
 */
let focusTickCost = 0;

/**
 * Timer count with the overlay closed and focus parked on {@link focusAnchor}.
 * The guards assert `baseline + focusTransitions * focusTickCost`.
 */
let baselineTimerCount = 0;

/** Element that holds focus before an overlay opens, so `trapFocus` has a real element to restore to. */
let focusAnchor: HTMLElement;

// Mock DOM environment
beforeEach(() => {
  document.body.innerHTML = '';
  vi.useFakeTimers();

  focusAnchor = document.createElement('div');
  focusAnchor.tabIndex = -1;
  document.body.appendChild(focusAnchor);
  focusAnchor.focus();
  focusTickCost = vi.getTimerCount();
  baselineTimerCount = focusTickCost;
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

/**
 * Assert the overlay left NO timer of its own pending.
 *
 * An open-then-close cycle performs `focusTransitions` focus moves (the
 * container on open, the restore on close), each of which costs
 * {@link focusTickCost} environment timers. Anything above that is the
 * overlay's — the 100 ms click-listener timer if it stopped being cancelled,
 * or an untracked 0 ms timer (audit G17). Deliberately does NOT advance the
 * clock: `advanceTimersByTime(0)` retires the environment tick, but it retires
 * a leaked 0 ms timer with it and the guard stops biting.
 */
function expectNoOverlayTimersPending(focusTransitions: number): void {
  expect(vi.getTimerCount()).toBe(baselineTimerCount + focusTransitions * focusTickCost);
}

describe('showHelpOverlay - Memory Leak Prevention', () => {
  it('explains that digit keys address non-displayed dimensions', () => {
    showHelpOverlay();

    expect(document.getElementById('luxar-help-overlay')?.textContent).toContain(
      'Select a non-displayed dimension (panel header shows target)'
    );
  });

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

    // Two focus transitions: onto the overlay container, then back to the
    // anchor when the trap restores focus.
    expectNoOverlayTimersPending(2);
  });

  it('close button cancels delayed listener registration', () => {
    showHelpOverlay();
    const closeBtn = document.querySelector('button[title="Close (Escape)"]') as HTMLButtonElement;

    closeBtn.click();

    expect(document.getElementById('luxar-help-overlay')).toBeNull();
    expectNoOverlayTimersPending(2);
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

  it('lists the pointer and keyboard element actions', () => {
    showHelpOverlay();
    const text = document.getElementById('luxar-help-overlay')?.textContent ?? '';

    expect(text).toContain('Open the hovered element link');
    expect(text).toContain('Actions for the hovered element');
    expect(text).toContain('Context menu for the hovered element');
  });
});

describe('showHelpOverlay - initial focus and type-to-filter (#1922)', () => {
  const overlayEl = (): HTMLElement => document.getElementById('luxar-help-overlay') as HTMLElement;
  const filterEl = (): HTMLInputElement =>
    overlayEl().querySelector('.luxar-panel-filter__input') as HTMLInputElement;
  const visibleRowText = (): string[] =>
    Array.from(overlayEl().querySelectorAll<HTMLElement>('.luxar-help-overlay__row'))
      .filter((row) => row.style.display !== 'none')
      .map((row) => (row.textContent ?? '').toLowerCase());

  function pressOnOverlay(init: KeyboardEventInit & { key: string }): KeyboardEvent {
    const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
    (document.activeElement ?? overlayEl()).dispatchEvent(event);
    return event;
  }

  it('parks focus on the overlay container, never on the filter field', () => {
    showHelpOverlay();
    // Run out every pending timer: a focus timer (the old autofocus, or the
    // focus trap's own first-focusable one) would move focus here.
    vi.advanceTimersByTime(200);

    expect(document.activeElement).toBe(overlayEl());
    // The exact predicate `InputHandler.onKeyDown` guards on. If this were
    // true, the second `H` would be swallowed as typing and the overlay
    // could not be closed from the keyboard.
    expect(isTypingInInput(document.activeElement)).toBe(false);
  });

  it('the first printable keystroke lands in the filter and narrows the list', () => {
    showHelpOverlay();
    const rowsBefore = visibleRowText().length;
    expect(rowsBefore).toBeGreaterThan(1);

    const event = pressOnOverlay({ key: 'r' });

    expect(document.activeElement).toBe(filterEl());
    expect(filterEl().value).toBe('r');
    // The filter's own `input` handler ran — rows are actually narrowed, and
    // every survivor matches. Asserting the rendered result (not just focus)
    // is what pins the dispatched `input` event.
    const rows = visibleRowText();
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.length).toBeLessThan(rowsBefore);
    expect(rows.every((text) => text.includes('r'))).toBe(true);
    // Not double-inserted, and not leaked to the global bindings.
    expect(event.defaultPrevented).toBe(true);
  });

  it('leaves `H` to the global binding so the overlay stays a toggle', () => {
    const seen: string[] = [];
    const listener = (e: KeyboardEvent) => seen.push(e.key);
    document.addEventListener('keydown', listener);
    try {
      showHelpOverlay();

      const event = pressOnOverlay({ key: 'h' });

      // `H` must reach the document-level handler (which toggles the panel
      // shut) rather than being consumed as the first filter character.
      expect(seen).toEqual(['h']);
      expect(event.defaultPrevented).toBe(false);
      expect(filterEl().value).toBe('');
      expect(document.activeElement).toBe(overlayEl());
    } finally {
      document.removeEventListener('keydown', listener);
    }
  });

  it('a forwarded printable never reaches the global bindings', () => {
    const seen: string[] = [];
    const listener = (e: KeyboardEvent) => seen.push(e.key);
    document.addEventListener('keydown', listener);
    try {
      showHelpOverlay();

      // `v` cycles the camera mode globally. Typing it into the filter must
      // not also switch to fly mode behind the overlay.
      pressOnOverlay({ key: 'v' });

      expect(filterEl().value).toBe('v');
      expect(seen).toEqual([]);
    } finally {
      document.removeEventListener('keydown', listener);
    }
  });

  it('keeps non-printable global shortcuts off the scene while it is modal', () => {
    const seen: string[] = [];
    const listener = (e: KeyboardEvent) => seen.push(e.key);
    document.addEventListener('keydown', listener);
    try {
      showHelpOverlay();

      // `Home`/`End` jump the selected dimension and Shift+arrows change the
      // animation speed (`animation-shortcuts.ts`). No panel pushes an
      // InputContext, so with focus on a `tabindex="-1"` container the
      // container listener is the only thing containing them.
      pressOnOverlay({ key: 'Home' });
      pressOnOverlay({ key: 'End' });
      pressOnOverlay({ key: 'ArrowUp', shiftKey: true });

      expect(seen).toEqual([]);
      // Escape is the one key that must still get out — it closes the panel.
      pressOnOverlay({ key: 'Escape' });
      expect(seen).toEqual(['Escape']);
    } finally {
      document.removeEventListener('keydown', listener);
    }
  });

  it('does not double-insert once the filter holds focus', () => {
    showHelpOverlay();
    pressOnOverlay({ key: 'r' });
    expect(document.activeElement).toBe(filterEl());

    // The browser inserts the character itself now; the forwarder must keep
    // its hands off or the field would read "rr".
    const event = pressOnOverlay({ key: 'e' });

    expect(event.defaultPrevented).toBe(false);
    expect(filterEl().value).toBe('r');
  });

  it('types Shift+H into the filter instead of dropping it', () => {
    showHelpOverlay();

    // The global lookup spells this "h+shift", which no binding registers, so
    // passing it through would make Shift+H a dead key.
    const event = pressOnOverlay({ key: 'H', shiftKey: true });

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(filterEl());
    expect(filterEl().value).toBe('H');
  });

  it('hideHelpOverlay releases the type-to-filter listener', () => {
    showHelpOverlay();
    const overlay = overlayEl();
    const filter = filterEl();

    hideHelpOverlay();

    // The detached container must no longer forward keystrokes.
    const event = new KeyboardEvent('keydown', { key: 'r', bubbles: true, cancelable: true });
    overlay.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    expect(filter.value).toBe('');
  });

  it('a re-opened overlay starts type-to-filter fresh', () => {
    showHelpOverlay();
    pressOnOverlay({ key: 'r' });
    expect(filterEl().value).toBe('r');

    hideHelpOverlay();
    showHelpOverlay();

    expect(document.activeElement).toBe(overlayEl());
    expect(filterEl().value).toBe('');
    pressOnOverlay({ key: 'z' });
    expect(filterEl().value).toBe('z');
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
