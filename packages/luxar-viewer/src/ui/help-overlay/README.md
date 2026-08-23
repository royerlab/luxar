# help-overlay

Shared modal-panel helpers, hosted under the folder of their first
consumer (`ui/help-overlay.ts`). Both files are also imported by
`ui/dataset-browser.ts`, and `focus-trap.ts` additionally by
`ui/error-overlay.ts`.

## Files

| File                | Purpose                                                                      |
| ------------------- | ---------------------------------------------------------------------------- |
| `focus-trap.ts`     | `trapFocus(container, options?)` — Tab/Shift+Tab cycling + focus restoration |
| `type-to-filter.ts` | `installTypeToFilter(...)` — container focus + first-keystroke filtering     |

## `trapFocus(container, options?)`

```typescript
import { trapFocus } from './help-overlay/focus-trap';

const release = trapFocus(modalDiv);
// ...later, when closing the modal:
release();
```

- Records the currently-focused element on entry.
- Attaches a `keydown` listener that intercepts `Tab` and `Shift+Tab` to
  cycle focus across the container's focusable descendants
  (`a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])`).
  When focus is on something outside that list — typically the
  `tabindex="-1"` container itself — Tab steers into the first focusable
  and Shift+Tab into the last, instead of letting the browser default
  walk out of the modal.
- Asynchronously focuses the first focusable element via
  `setTimeout(..., 0)` so it runs after the caller has finished mounting
  DOM. The timer id is captured so it can be cancelled on cleanup. Pass
  `{ autoFocusFirst: false }` to skip this entirely — what the panels
  using `installTypeToFilter` do, since they focus the container instead
  and the trap's timer would otherwise fire later and win.
- Returns a cleanup function that cancels any still-pending focus timer,
  removes the listener, and restores focus to the previously-focused
  element. Cancelling the timer prevents a leaked tick when the trap is
  released before the next event loop turn (audit G17).

The trap is a no-op when the container has no focusable children
(neither the initial focus call nor the Tab interception fire), so
calling it on a partially-built modal is safe.

## `installTypeToFilter(container, resolveFilterInput, options?)`

```typescript
import { installTypeToFilter } from './help-overlay/type-to-filter';

const release = trapFocus(panel, { autoFocusFirst: false });
const releaseFilter = installTypeToFilter(panel, () => filterInput, {
  passthroughKeys: [config.input.keyboard.shortcuts.toggleHelp],
});
```

`InputHandler.onKeyDown` drops every key but `Escape` while focus is on a
typing surface, so a panel that autofocuses its filter field makes its own
toggle key one-way: `H` opened the help overlay but the second `H` was
swallowed as typing (issue #1922). This helper is the fix, applied at the
mechanism level so both filtered panels behave identically:

- Makes the container programmatically focusable (`tabindex="-1"`, unless
  it already declares one) and focuses it **synchronously** — the panel is
  already mounted, so there is no timer to schedule and none to leak.
- Forwards the first printable keystroke into the filter: focus the input,
  append the character, dispatch `input`, then `preventDefault` (no double
  insertion) and `stopPropagation` (a forwarded `v` must not also cycle the
  camera mode).
- Passes through anything with Ctrl/Meta/Alt, any `key.length !== 1`
  (`Escape`, `Tab`, `Enter`, `ArrowDown`, `F1`, …), `Space` (a global
  shortcut, and it scrolls a `tabindex="-1"` container), IME composition
  keystrokes, keys already `defaultPrevented`, keys typed while focus is
  already on a typing surface, and keys arriving while
  `resolveFilterInput()` returns `null` (nothing to filter).
- Returns an idempotent release function that removes the listener.

**`passthroughKeys` and the `h`-toggles-vs-`h`-filters trade-off.** Both
cannot hold for one character: forwarding the first `h` into the help
overlay's filter would re-break the toggle. The panel's own toggle key is
therefore listed in `passthroughKeys` and cannot be the _first_ character
of a query; it types normally once the filter has focus, and filtering is
case-insensitive, so nothing is unreachable.

## Why it lives here

`ui/` follows a public-API-at-root layout (see [`../README.md`](../README.md)).
The original consumer of `focus-trap.ts` was `help-overlay.ts`, so it was
placed in its sibling folder; `error-overlay.ts` and `dataset-browser.ts`
later picked up the same dependency, and `type-to-filter.ts` was added
beside it because it is the other half of the same panel idiom.

The folder name is therefore no longer accurate — these are shared modal
helpers, not the help overlay's private ones. A rename (something like
`ui/panel/`) would be tidier but is pure import churn; it is left for a
dedicated move, and the cross-modal sharing is documented in each file
header meanwhile.

## See also

- [`../help-overlay.ts`](../help-overlay.ts) — primary consumer (`H` key panel)
- [`../error-overlay.ts`](../error-overlay.ts) — second consumer (error dialog)
- [`../dataset-browser.ts`](../dataset-browser.ts) — third consumer (`O` key panel)
