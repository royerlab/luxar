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
 * ## Modal containment
 *
 * The autofocused text field was, de facto, the only thing keeping global
 * shortcuts from reaching the scene *behind* an `aria-modal` dialog: no panel
 * pushes an `InputContext`, so with focus parked on a `tabindex="-1"`
 * container `Home`/`End` would jump the selected dimension, `Shift`+arrows
 * would change the animation speed, and arrows would steer the fly camera —
 * all while a modal is open. The container listener therefore stops any key
 * that originates AT THE CONTAINER ITSELF from propagating to the global
 * bindings, with three deliberate exceptions:
 *
 * - `Escape` — panel dismissal goes through the global handler.
 * - `Tab` — must reach the focus trap.
 * - the panel's own passthrough toggle key — the entire point of this module.
 *
 * The gate is the event *target*, not the key: an event that originates at an
 * inner control (a listing row's `Enter`/arrow navigation, the filter's own
 * keydown) merely bubbles through the container and is left alone. Containment
 * is `stopPropagation` only, never `preventDefault`, so native in-panel
 * behaviour (scrolling the panel with `Home`/`PageDown`, browser-level
 * shortcuts) is untouched.
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

/**
 * Keys that must always reach the document-level handler, even while the
 * modal holds focus: `Escape` closes the panel, `Tab` drives the focus trap.
 */
const ALWAYS_GLOBAL_KEYS: ReadonlySet<string> = new Set(['Escape', 'Tab']);

/** Options for {@link installTypeToFilter}. */
export interface TypeToFilterOptions {
  /**
   * Keys that must reach the global bindings instead of starting
   * type-to-filter — in practice the panel's own toggle key (`h`, `o`).
   * Matched case-insensitively so a CapsLock'd `H` still toggles, but only
   * while Shift is NOT held: the global lookup spells a shifted key
   * `"h+shift"`, which no binding registers, so `Shift`+the key is forwarded
   * into the filter like any other printable character.
   */
  passthroughKeys?: readonly string[];
  /**
   * Optional lookup for the first item of a navigable list inside the panel
   * (the dataset browser's first listing row). When supplied, `ArrowDown`
   * pressed while focus is parked on the container moves focus there — the
   * affordance the search field's own `ArrowDown` handler provides once the
   * field has focus. Return `null` when there is no list to enter.
   */
  resolveFirstItem?: () => HTMLElement | null;
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
 * Not forwarded into the filter:
 * - anything with Ctrl/Meta/Alt held (those are application shortcuts) —
 *   except the AltGr signature (`ctrlKey && altKey` with a one-character
 *   `key`), which is how Windows reports `AltGr+E` = `€` or `AltGr+2` = `@`
 *   and is a perfectly ordinary printable character,
 * - `event.key.length !== 1`, which covers `Escape`, `Tab` (must keep
 *   reaching the focus trap), `Enter`, `Backspace`, `ArrowDown` (steered into
 *   the listing when `resolveFirstItem` supplies one), `F1`, `Home`/`End`,
 *   and every dead key,
 * - `Space`: it scrolls a `tabindex="-1"` container in some browsers, and a
 *   leading space is meaningless to a substring filter. Once the filter has
 *   focus Space types normally, since the forwarder no longer sees the event.
 * - keys already handled by an inner listener (`defaultPrevented`),
 * - keys typed while focus is already on a typing surface, so a character is
 *   never handled twice,
 * - keys arriving while `resolveFilterInput()` returns `null` — a panel whose
 *   filter is currently hidden (the dataset browser hides its search bar
 *   while a directory is loading) has nothing to filter.
 *
 * A keystroke that opens an IME composition (`isComposing`, or the legacy
 * `keyCode === 229` Safari/Chromium spelling) is a special case: the filter is
 * focused but the event is NOT `preventDefault`ed, so the composition
 * retargets to the input and its committed text lands there. Composing
 * against the non-editable container would drop the first character outright,
 * which is every CJK/IME and European dead-key layout.
 *
 * Everything that is neither forwarded nor an exception is still contained —
 * see "Modal containment" in the module header.
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

    // Only a key that originates AT the container means "focus is parked on
    // the modal shell"; anything bubbling up from an inner control belongs to
    // that control.
    const atContainer = event.target === container;
    const typingSurface = isTypingInInput(event.target as Element | null);

    // `keyCode === 229` is the pre-`isComposing` spelling still emitted by
    // some IME paths; both mean "this keystroke opens a composition". Hand it
    // to the filter WITHOUT preventDefault so the composer retargets there —
    // composing against a non-editable container loses the character.
    if (event.isComposing || event.keyCode === 229) {
      if (!typingSurface) resolveFilterInput()?.focus({ preventScroll: true });
      return;
    }

    // Windows AltGr sets ctrlKey AND altKey while producing a normal
    // printable character, so it must not be mistaken for a shortcut.
    const isAltGraph = event.ctrlKey && event.altKey && event.key.length === 1;
    const isModified = (event.ctrlKey || event.metaKey || event.altKey) && !isAltGraph;

    if (!isModified) {
      // Shift is deliberately excluded: `Shift+H` spells `"h+shift"` in the
      // global lookup and matches no binding, so passing it through would
      // make it a dead key.
      if (!event.shiftKey && passthrough.has(event.key.toLowerCase())) return;

      if (atContainer && event.key === 'ArrowDown' && !event.shiftKey) {
        const firstItem = options.resolveFirstItem?.() ?? null;
        if (firstItem) {
          event.preventDefault();
          event.stopPropagation();
          firstItem.focus();
          return;
        }
      }

      if (event.key.length === 1 && event.key !== ' ' && !typingSurface) {
        const filterInput = resolveFilterInput();
        if (filterInput) {
          event.preventDefault();
          event.stopPropagation();
          filterInput.focus({ preventScroll: true });
          // Append rather than replace: focus may have been parked on a row
          // or the close button while the filter already held a query.
          filterInput.value += event.key;
          filterInput.dispatchEvent(new Event('input', { bubbles: true }));
          return;
        }
      }
    }

    // Modal containment: nothing behind an `aria-modal` panel should hear a
    // key the user aimed at the panel itself.
    if (atContainer && !ALWAYS_GLOBAL_KEYS.has(event.key)) {
      event.stopPropagation();
    }
  };

  container.addEventListener('keydown', handleKeyDown);

  let released = false;
  return () => {
    if (released) return;
    released = true;
    container.removeEventListener('keydown', handleKeyDown);
  };
}
