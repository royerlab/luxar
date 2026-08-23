/**
 * Focus-trap utility for modal dialogs.
 *
 * Keeps Tab/Shift+Tab cycling within the container's focusable elements
 * so focus can't escape to the background while a modal is open. The
 * cleanup function restores focus to the previously-focused element.
 *
 * Shared by error-overlay, help-overlay and dataset-browser.
 */

/** CSS selector matching everything the browser considers tabbable here. */
const FOCUSABLE_SELECTORS =
  'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])';

/**
 * `true` when `el` is actually rendered, i.e. a real browser would move focus
 * to it.
 *
 * A raw `querySelectorAll` also returns elements inside a `display: none`
 * subtree — the dataset browser hides its whole search row while a directory
 * loads, but its `<input>` still matches the selector and is the LAST match,
 * so `Shift+Tab` from the container would `.focus()` a hidden input, which is
 * a no-op in a browser and leaves focus stranded (issue #1922 follow-up).
 *
 * `getClientRects()` is the browser-accurate test (see
 * `ui/control-rail/dom-helpers.ts`), but jsdom has no layout engine and
 * reports zero rects for *everything*, so it needs a fallback: walk the
 * ancestor chain for an explicit `display: none` / `visibility: hidden` /
 * `hidden`. That fallback is conservative — anything not explicitly hidden
 * counts as visible — which keeps a detached-but-about-to-be-mounted panel
 * working.
 */
function isRendered(el: HTMLElement): boolean {
  if (typeof el.getClientRects === 'function' && el.getClientRects().length > 0) return true;

  for (let node: HTMLElement | null = el; node; node = node.parentElement) {
    if (node.hidden) return false;
    if (node.style.display === 'none' || node.style.visibility === 'hidden') return false;
    const computed = typeof getComputedStyle === 'function' ? getComputedStyle(node) : null;
    if (computed && (computed.display === 'none' || computed.visibility === 'hidden')) return false;
  }
  return true;
}

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
  const previouslyFocused = document.activeElement as HTMLElement;

  /** Tabbable descendants that are actually rendered, in document order. */
  const getFocusable = (): HTMLElement[] =>
    Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS)).filter(isRendered);

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;
    const focusable = getFocusable();
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
  const firstFocusable = autoFocusFirst ? (getFocusable()[0] ?? null) : null;
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
