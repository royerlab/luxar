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
- Filters that list down to elements that are actually **rendered**. A raw
  `querySelectorAll` also matches inputs inside a `display: none` subtree
  (the dataset browser hides its whole search row while a directory
  loads), and `.focus()` on one of those is a no-op in a real browser — so
  `Shift+Tab` from the container would strand focus. Visibility is tested
  with `getClientRects()`, falling back to an explicit
  `display`/`visibility`/`hidden` walk up the ancestor chain because jsdom
  has no layout and reports zero rects for everything.
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
  // Every shortcut this panel advertises while open — see passthroughKeys below.
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
  camera mode). `Shift`+a printable is forwarded like any other character —
  including `Shift`+the panel's own toggle key, since the global lookup
  spells that `"h+shift"` and no binding registers it.
- Does not forward: anything with Ctrl/Meta/Alt **except** the AltGr
  signature (`ctrlKey && altKey` with a one-character `key`, which is how
  Windows reports `AltGr+E` = `€`), any `key.length !== 1` (`Escape`,
  `Tab`, `Enter`, `F1`, …), `Space` (a leading space matches nothing and it
  scrolls a `tabindex="-1"` container), keys already `defaultPrevented`,
  keys typed while focus is already on a typing surface, and keys arriving
  while `resolveFilterInput()` returns `null` (nothing to filter — the
  dataset browser hides its search bar while loading, on an error and in an
  empty directory). None of those are released to the globals; they are
  contained like any other key.
- Focuses the filter **without** `preventDefault` for a keystroke that opens
  an IME composition (`isComposing`, or the legacy `keyCode === 229`), so
  the composition retargets to the input; composing against the
  non-editable container would drop the first character outright, which is
  every CJK/IME and European dead-key layout.
- Steers `ArrowDown` into the panel's list when `resolveFirstItem` supplies
  one (the dataset browser's first listing row) and focus is still on the
  container — with focus parked there, the search field's own `ArrowDown`
  handler never sees the key. Once focus is inside the list, the list's own
  arrow navigation takes over.
- **Contains everything else while the panel is modal.** Every key that
  bubbles to the container and is neither forwarded nor `Escape`/`Tab`/an
  eligible passthrough key gets `stopPropagation()` — no `preventDefault`,
  so native behaviour survives: scrolling the panel with `Home`/`PageDown`,
  and browser-level shortcuts (`Ctrl+F`/`Ctrl+A`/`Ctrl+C`, `Cmd+W`, `F5`,
  `F1`–`F12`). Without this, `Home`/`End` would jump the selected dimension
  and `Shift`+arrows would change the animation speed _behind_ an
  `aria-modal` dialog: no panel pushes an `InputContext`, so the autofocused
  text field used to be the only thing suppressing global shortcuts.
  Containment does **not** look at the event target — a key from a listing
  row, the filter or a breadcrumb is contained too, or a single `Tab` would
  hand the scene back its shortcuts. Inner handlers still run: they fire
  first on the way up, and a key one of them consumed
  (`defaultPrevented`) is contained rather than released.
- Returns an idempotent release function that removes the listener.

**`passthroughKeys` and the `h`-toggles-vs-`h`-filters trade-off.** Both
cannot hold for one character: forwarding the first `h` into the help
overlay's filter would re-break the toggle. The panel's own toggle key is
therefore listed in `passthroughKeys` and cannot be the _first_ character
of a query; it types normally once the filter has focus, `Shift`+that key
types it even as the first character, and filtering is case-insensitive, so
nothing is unreachable. Matching is case-insensitive but Shift-sensitive:
CapsLock (`H` with `shiftKey === false`) still toggles, because the global
lookup lowercases the key.

`passthroughKeys` is the panel's whole **advertised** key surface, not just
its toggle: containment silences everything else, so a shortcut chip the
panel renders (the dataset browser's banner shows `H Help`) must be listed
or it is dead. The dataset browser therefore declares `o` and `h`; the help
overlay declares `h`. The exemption is gated on `event.target === container`
— once focus has moved into a field or a row those keys are ordinary
characters again, so typing a path beginning with `o` into the dataset
browser cannot close it.

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
