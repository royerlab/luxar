/**
 * Type-to-filter for modal panels that carry a filter field.
 *
 * ## Why this exists
 *
 * `InputHandler.onKeyDown` drops every key except `Escape` while focus is on
 * a typing surface, so a panel that autofocuses its own filter input makes its
 * own global toggle key one-way: `H` opens the help overlay, but the second
 * `H` is swallowed as typing and the overlay never closes (issue #1922).
 *
 * The fix is to stop autofocusing the filter. Focus lands on the panel
 * *container* instead — which is not a typing surface, so the guard never
 * engages and the toggle key keeps working — and the first printable
 * keystroke is forwarded into the filter, so type-to-filter still starts on
 * the very first key the user presses.
 *
 * Shared by `ui/help-overlay.ts` and `ui/dataset-browser.ts`; `trapFocus` +
 * `installTypeToFilter` is the house idiom for a filtered modal panel.
 *
 * ## The `h`-toggles vs `h`-filters conflict
 *
 * Both requirements cannot hold for one character: if the very first `h`
 * inside the help overlay is forwarded to the filter, `H` no longer closes
 * the panel — which is the bug this module exists to fix. The panel's own
 * toggle key is therefore listed in `passthroughKeys` and is *not* captured
 * as the first keystroke; it propagates to the global binding and toggles the
 * panel shut. Every other printable character starts filtering, and once the
 * filter has focus the toggle key types normally like any other letter (the
 * forwarder only ever looks at keys that arrive while focus is NOT on a
 * typing surface). Filtering is case-insensitive, so nothing is unreachable —
 * "help" is found by typing `elp` first, or by pressing any other key first.
 */

import { isTypingInInput } from '../../input/input-handler/commands/focus-utils';

/** Options for {@link installTypeToFilter}. */
export interface TypeToFilterOptions {
  /**
   * Keys that must reach the global bindings instead of starting
   * type-to-filter — in practice the panel's own toggle key (`h`, `o`).
   * Matched case-insensitively, so `Shift`+the key behaves the same way the
   * global binding does.
   */
  passthroughKeys?: readonly string[];
}

/**
 * Park focus on `container` and start type-to-filter on the first printable
 * keystroke.
 *
 * The container is made programmatically focusable (`tabindex="-1"`, only if
 * it doesn't already declare one) and focused synchronously — it is already
 * mounted by the time a panel installs this, so there is no timer to schedule
 * and therefore none to leak. Call this AFTER `trapFocus` and pass
 * `{ autoFocusFirst: false }` there, otherwise the trap's own 0 ms
 * first-focusable timer fires later and steals focus to the close button.
 *
 * A keydown listener on the container forwards a printable key into the
 * filter: it focuses the input, appends the character, and dispatches the
 * `input` event the panel's existing filter handler listens for. The key is
 * both `preventDefault`ed (so the character isn't inserted twice once focus
 * lands) and `stopPropagation`ed (so a forwarded `v` doesn't also cycle the
 * camera mode on its way to the document-level handler).
 *
 * Ignored — i.e. left to propagate untouched:
 * - anything with Ctrl/Meta/Alt held (those are application shortcuts),
 * - `event.key.length !== 1`, which covers `Escape`, `Tab` (must keep
 *   reaching the focus trap), `Enter`, `Backspace`, `ArrowDown` (documented
 *   as moving into the dataset list), `F1`, `Home`/`End`, and every dead key,
 * - `Space`: it is a global shortcut (fullscreen), it scrolls a
 *   `tabindex="-1"` container in some browsers, and a leading space is
 *   meaningless to a substring filter. Once the filter has focus Space types
 *   normally, since the forwarder no longer sees the event.
 * - keys mid-IME-composition (`isComposing`, or the legacy `keyCode === 229`
 *   Safari/Chromium spelling), whose committed text arrives via `input`,
 * - keys already handled by an inner listener (`defaultPrevented`),
 * - keys typed while focus is already on a typing surface, so a character is
 *   never handled twice,
 * - keys arriving while `resolveFilterInput()` returns `null` — a panel whose
 *   filter is currently hidden (the dataset browser hides its search bar
 *   while a directory is loading) has nothing to filter, so the key keeps its
 *   normal global meaning.
 *
 * @param container - The panel element. Gains `tabindex="-1"` and keyboard focus.
 * @param resolveFilterInput - Looks up the filter field at keystroke time.
 *   Return `null` when filtering is currently unavailable.
 * @param options - See {@link TypeToFilterOptions}.
 * @returns An idempotent release function that removes the listener.
 */
export function installTypeToFilter(
  container: HTMLElement,
  resolveFilterInput: () => HTMLInputElement | null,
  options: TypeToFilterOptions = {}
): () => void {
  const passthrough = new Set((options.passthroughKeys ?? []).map((key) => key.toLowerCase()));

  if (!container.hasAttribute('tabindex')) {
    container.setAttribute('tabindex', '-1');
  }
  container.focus({ preventScroll: true });

  const handleKeyDown = (event: KeyboardEvent): void => {
    if (event.defaultPrevented) return;
    // `keyCode === 229` is the pre-`isComposing` spelling still emitted by
    // some IME paths; both mean "this keystroke belongs to the composer".
    if (event.isComposing || event.keyCode === 229) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key.length !== 1) return;
    if (event.key === ' ') return;
    if (passthrough.has(event.key.toLowerCase())) return;
    if (isTypingInInput(event.target as Element | null)) return;

    const filterInput = resolveFilterInput();
    if (!filterInput) return;

    event.preventDefault();
    event.stopPropagation();
    filterInput.focus({ preventScroll: true });
    // Append rather than replace: focus may have been parked on a row or the
    // close button while the filter already held a query.
    filterInput.value += event.key;
    filterInput.dispatchEvent(new Event('input', { bubbles: true }));
  };

  container.addEventListener('keydown', handleKeyDown);

  let released = false;
  return () => {
    if (released) return;
    released = true;
    container.removeEventListener('keydown', handleKeyDown);
  };
}
