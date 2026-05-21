# ui/overlay-widgets

Shared base class for screen-space overlay widgets in `ui/`. Provides the
managed-lifecycle scaffolding (event listeners, theme subscription, show/
hide, disposal) that overlay widgets like the scale bar and the colormap
legend extend instead of re-implementing.

## Files

```
overlay-widgets/
└── ui-component.ts   # UIComponent<TConfig>: abstract base class
```

## What `UIComponent` provides

`UIComponent<TConfig>` is an abstract class whose constructor calls
`render()` (which subclasses must implement), then `attachEventListeners()`,
then `subscribeToTheme()`. Subclasses get:

- **Managed event listeners.** `this.addEventListener(target, type, fn)`
  records each `(target, type)` pair in an internal `Map` and removes any
  existing listener for the same key first to prevent duplicates. A
  generic variant `addEventListenerGeneric()` is available for custom
  event types not in `HTMLElementEventMap`. `dispose()` removes every
  tracked listener — so a subclass that uses this API exclusively never
  needs to worry about leaks.
- **Theme subscription.** The constructor subscribes to
  `ThemeManager.getInstance().onChange(...)` and forwards changes to a
  protected `onThemeChange(theme)` hook (default: no-op). The
  unsubscribe function is stored and called on `dispose()`.
- **Visibility via BEM modifier.** `show()` appends the root element to
  `document.body` if it has no parent and adds the `<className>--visible`
  modifier class; `hide()` removes the modifier (the element stays in
  the DOM); `toggle()` flips between them; `isVisible()` reads the class
  state. The base class assumes a BEM naming convention — subclasses
  must return their base class name from the abstract `getClassName()`.
- **Disposal pattern.** `dispose()` removes all tracked listeners,
  unsubscribes from theme changes, calls `onDispose()` for custom
  cleanup, then removes the root element from the DOM.

The abstract surface a subclass must implement is small: `render()`
returns the root `HTMLElement`, and `getClassName()` returns the base
BEM class (e.g. `'luxar-scale-bar'`). Optional overrides:
`attachEventListeners()`, `onThemeChange()`, `onDispose()`.

## Subclass contract

Per the doc comments in `ui-component.ts`:

- **No inline styles** — subclasses style via CSS classes only, so the
  theming system stays the single source of truth (see `../../styles/`
  and `../../themes/`).
- **Arrow-function handlers** — instance-arrow methods give stable
  references and keep `this` bound to the component, which is what
  `addEventListener()` expects (it stores the listener verbatim, no
  binding).

## Consumers

- [`../scale-bar.ts`](../scale-bar.ts) — `ScaleBar extends UIComponent<ScaleBarConfig>`
- [`../colormap-legend.ts`](../colormap-legend.ts) — `ColormapLegend extends UIComponent<ColormapLegendConfig>`

These are the two scene-dependent HUD overlays constructed and torn down
by [`../../core/app/overlays/`](../../core/app/overlays) on each
`loadDataset()` call.

## See also

- [`../README.md`](../README.md) — full UI package layout (notes that
  `overlay-widgets/` holds the shared base for scale-bar / colormap-legend)
- [`../../themes/theme-manager.ts`](../../themes/theme-manager.ts) — the
  singleton subscribed to in the constructor
- [`../../core/app/overlays/README.md`](../../core/app/overlays/README.md)
  — overlay lifecycle on the app side
