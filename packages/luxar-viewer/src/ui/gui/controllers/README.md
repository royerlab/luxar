# GUI Controllers

Concrete widget classes for the custom GUI library — one file per input type. Each subclass extends the abstract `Controller<T>` base (in `../controller.ts`), builds its DOM in `createDOMElement()`, and routes DOM events through the parent's shared `EventManager` for guaranteed cleanup on `dispose()`.

## Files

```
controllers/
├── boolean-controller.ts   # Checkbox for boolean values
├── number-controller.ts    # Range slider + number input (with wheel, dblclick reset, alt+click)
├── string-controller.ts    # Text input for strings
├── option-controller.ts    # Dropdown <select> for enumerated choices
└── function-controller.ts  # Button that invokes a callable property
```

All five controllers follow the same skeleton:

1. Constructor calls `super(object, property)`, sets type-specific state, then `this.initializeDOMElement()`.
2. `createDOMElement()` builds the widget DOM using `this.createBaseElement()` from the base (except `BooleanController` and `FunctionController`, which build their own minimal containers).
3. Event listeners are registered via `this.eventManager.add(...)` so `super.dispose()` removes them.
4. `applyAutoBlur(element, eventManager)` is applied to every focusable element so keyboard shortcuts resume after interaction.
5. `updateDisplay()` syncs the DOM from the bound `object[property]`.
6. The native input element is exposed as `this.$input` for callers that need direct DOM access (e.g. `rendering-controls/` overrides).

## Controllers

### BooleanController

Wraps `<input type="checkbox">` inside a `<label>` so the label text is part of the click area. Fires both `triggerChange()` and `triggerFinishChange()` on every `change` event (checkboxes have no separate "committed" event). Overrides `name(label)` to update the text node sitting next to the checkbox.

### NumberController

The most complex controller. Optionally renders a range slider (when both `min` and `max` are set) alongside a text-mode number input.

**Slider behavior**

| Interaction | Effect |
|-------------|--------|
| Drag | Live `change` events; `finishChange` on `mouseup` / `touchend`. |
| Wheel scroll | Fine-tune by `step × 0.1`; `Shift+Scroll` for `step × 10`. Debounced `finishChange` 150 ms after scrolling stops. |
| Double-click | Reset to the value captured at construction (`initialValue`). |
| Alt+click | Focus and select the companion number input for keyboard entry. |

The slider's `title` tooltip advertises these shortcuts.

**Number input behavior**

Live `change` events on every keystroke; `finishChange` on commit (`change` event). Invalid input reverts to the current value via `updateDisplay()`.

**Constraints**

- `min(value)`, `max(value)`, `step(value)` — chainable; mutate both slider and input attributes.
- `constrainValue(v)` clamps to `[minValue, maxValue]` using `clamp` from `../format/value-formatting`.
- Default step (when only `min`/`max` given) is `1%` of the range.
- Display formatting goes through `formatNumber(value, step)`.

**Custom display override**

`setCustomUpdateDisplay(fn)` lets consumers (notably `rendering-controls/`) install a custom `updateDisplay` implementation — used to render logarithmic sliders whose internal slider value is `log(actual)`.

**Disposal**

Overrides `dispose()` to clear the wheel debounce timer before delegating to `super.dispose()`.

### StringController

Plain `<input type="text">`. Both `input` and `change` events update the bound property and call `triggerChange()`; only `change` triggers `triggerFinishChange()`.

### OptionController

Builds a `<select>` from either an array (`['Low', 'Medium', 'High']`) or a label→value record (`{Low: 1, Medium: 2, High: 3}`). The DOM only stores string labels, so an internal `optionsMap: Map<string, unknown>` round-trips between displayed labels and the actual JS values bound to the property. Throws if constructed without an `options` field. `change` triggers both change events.

### FunctionController

Renders a single `<button>` whose textContent is the controller label. Clicks invoke `object[property]` as a function with `this === object` (so methods see their own receiver), then fire change/finishChange. `name(label)` updates the button text. `updateDisplay()` is a no-op — functions have nothing to display.

## Conventions

- **Type discriminator**: each subclass sets `protected type = ControllerType.{BOOLEAN|NUMBER|STRING|OPTION|FUNCTION}` (see `../types.ts`). This becomes a CSS class suffix (`luxar-gui__controller--number`) so styles in `../styles/` can target widget kinds.
- **Auto-blur**: every focusable element passes through `applyAutoBlur` (`../format/auto-blur`) so the viewer's keyboard shortcut layer regains focus after a user edits a value.
- **Safety checks**: every event handler guards `if (!this.<element>) return` to tolerate post-`dispose()` callbacks that may still be queued.
- **Event ownership**: no controller calls `addEventListener` directly — everything goes through `this.eventManager.add(...)` so `Controller.dispose()` can remove every listener centrally.

## See Also

- `../controller.ts` — Abstract `Controller<T>` base (target/property binding, event callbacks, `createBaseElement`, `dispose`).
- `../types.ts` — `ControllerType` enum and `ControllerOptions` shape.
- `../format/auto-blur.ts` — `applyAutoBlur` helper.
- `../format/value-formatting.ts` — `clamp` and `formatNumber` (used by `NumberController`).
- `../gui.ts` / `../folder.ts` — `Folder.add(object, property, ...args)` selects the appropriate controller subclass based on the value's runtime type and the optional args.
- `../../rendering-controls/` — Primary consumer; uses `setCustomUpdateDisplay` and `$input` for logarithmic sliders.
