# GUI Focus Helpers

Small helpers shared by the custom GUI controllers to keep input focus from
sticking after an interaction. Slider number formatting now lives in
[`../../slider-kit/`](../../slider-kit/README.md).

## Files

- `auto-blur.ts` — Applies type-aware blur behavior to `<input>` and
  `<select>` elements. Different input types use different triggers so
  focus is released as soon as the interaction is meaningfully done:
  - Checkboxes blur on `change` and `click` (10 ms delay).
  - Range sliders blur on `mouseup`, `touchend`, and after a 200 ms
    debounce on `wheel`.
  - Number and text inputs blur on `Enter` or `Escape` only — never on
    `mouseup`, which would prevent the user from clicking into the
    field to edit.
  - Selects blur on `change` (10 ms delay).

  Listeners are registered through the controller's
  [`EventManager`](../dom/event-manager.ts) so they are cleaned up when
  the controller is disposed.

## Why auto-blur matters

The viewer relies on global keyboard shortcuts (dimension navigation,
playback, etc.). If a GUI input keeps focus after the user finishes
interacting with it, keystrokes go to the input instead of the viewer.
Auto-blur restores the expected "click control, then keep navigating"
flow without forcing users to click outside the panel.

## See Also

- [`../README.md`](../README.md) — Custom GUI library overview.
- [`../dom/event-manager.ts`](../dom/event-manager.ts) — Listener
  tracking used by `applyAutoBlur`.
- [`../controllers/`](../controllers/) — Concrete controllers that
  consume these helpers.
