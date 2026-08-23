/**
 * Focus-trap utility for modal dialogs.
 *
 * Keeps Tab/Shift+Tab cycling within the container's focusable elements
 * so focus can't escape to the background while a modal is open. The
 * cleanup function restores focus to the previously-focused element.
 *
 * Shared by error-overlay, help-overlay and dataset-browser.
 */

/** Options for {@link trapFocus}. */
export interface TrapFocusOptions {
  /**
   * Park initial focus on the container's first focusable element (default
   * `true`). Panels that use `installTypeToFilter` pass `false`: they focus
   * the CONTAINER instead, so no text field holds focus and the panel's own
   * toggle key survives `InputHandler`'s typing guard (issue #1922). Leaving
   * this at `true` there would make the two mechanisms fight — the trap's
   * 0 ms timer fires after the panel's synchronous container focus and would
   * silently win.
   */
  autoFocusFirst?: boolean;
}

export function trapFocus(container: HTMLElement, options: TrapFocusOptions = {}): () => void {
  const { autoFocusFirst = true } = options;
  const focusableSelectors =
    'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])';
  const previouslyFocused = document.activeElement as HTMLElement;

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;
    const focusable = Array.from(container.querySelectorAll<HTMLElement>(focusableSelectors));
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (!focusable.includes(document.activeElement as HTMLElement)) {
      // Focus is on the container itself (a `tabindex="-1"` panel parks it
      // there — see `installTypeToFilter`) or on some other non-tabbable
      // descendant. The browser default would walk OUT of the modal on
      // Shift+Tab, so steer explicitly into the trap's own ends.
      e.preventDefault();
      (e.shiftKey ? last : first).focus();
      return;
    }
    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  };

  container.addEventListener('keydown', handleKeyDown);

  // Focus first focusable element. The timer id is captured so the
  // cleanup function can cancel it — otherwise a trap that's released
  // before the next tick leaks a pending timer (audit G17).
  const firstFocusable = autoFocusFirst
    ? container.querySelector<HTMLElement>(focusableSelectors)
    : null;
  let focusTimerId: ReturnType<typeof setTimeout> | null = null;
  if (firstFocusable) {
    focusTimerId = setTimeout(() => {
      focusTimerId = null;
      firstFocusable.focus();
    }, 0);
  }

  return () => {
    if (focusTimerId !== null) {
      clearTimeout(focusTimerId);
      focusTimerId = null;
    }
    container.removeEventListener('keydown', handleKeyDown);
    if (previouslyFocused) previouslyFocused.focus();
  };
}
