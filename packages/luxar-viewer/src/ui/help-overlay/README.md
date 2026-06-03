# help-overlay

Private helper for the top-level `ui/help-overlay.ts` public-API file.
The folder currently holds a single utility — `focus-trap.ts` — which is
also imported by `ui/error-overlay.ts` (both modals need the same Tab
cycling and focus-restore behaviour).

## Files

| File            | Purpose                                                            |
| --------------- | ------------------------------------------------------------------ |
| `focus-trap.ts` | `trapFocus(container)` — Tab/Shift+Tab cycling + focus restoration |

## `trapFocus(container)`

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
- Asynchronously focuses the first focusable element via
  `setTimeout(..., 0)` so it runs after the caller has finished mounting
  DOM.
- Returns a cleanup function that removes the listener and restores
  focus to the previously-focused element.

The trap is a no-op when the container has no focusable children
(neither the initial focus call nor the Tab interception fire), so
calling it on a partially-built modal is safe.

## Why it lives here

`ui/` follows a public-API-at-root layout (see [`../README.md`](../README.md)).
The single consumer of this helper was originally `help-overlay.ts`, so
it was placed in its sibling folder. `error-overlay.ts` later picked up
the same dependency; the file stays under `help-overlay/` because moving
it now would only churn imports — the cross-modal sharing is documented
in the file header.

## See also

- [`../help-overlay.ts`](../help-overlay.ts) — primary consumer (`H` key panel)
- [`../error-overlay.ts`](../error-overlay.ts) — second consumer (error dialog)
