# Custom GUI Library Styles

Scoped CSS for the custom GUI library, split across three files so each
visual concern (panel chrome, individual controllers, nested folders)
stays self-contained. Every rule is namespaced under `.luxar-gui` /
`.luxar-gui__*` so the stylesheets are safe to ship inside the
embed-safe `src/styles/index.css` bundle.

All styling references CSS custom properties from the Luxar theme
system (`--luxar-bg-*`, `--luxar-text-*`, `--luxar-spacing-*`, etc.), so
theme changes are picked up automatically for all four themes (`dark`,
`light`, `frosted-glass`, `liquid-glass`); the `[data-theme='light']`
selectors here only cover the cases needing a per-theme override. No
`!important` anywhere. Slider thumbs and `accent-color` use
`--luxar-highlight`, focus goes through `--luxar-border-focus`, and the mono
values are `text-primary`; the raw `rgba(0, 0, 0, …)` control fills in
`controller.css` are known drift — see the
[UI Design Guide](../../../../../../docs/guides/developer/UI_DESIGN_GUIDE.md)
§6/§15.1.

## Files

- `gui.css` — Root panel chrome. Defines `.luxar-gui` (outer fixed
  container, no overflow so liquid-glass pseudo-elements can paint
  outside the box), `.luxar-gui__scroll` (inner scrollable container
  with custom thin scrollbar), and the header / title / close-button /
  children-container subblocks. Deliberately does **not** override
  `--luxar-shadow-lg` per theme — it uses the same shadow token the control
  rail does; the only `[data-theme]` block here is a light-theme background
  override, and the glass themes live in `src/styles/themes/`.
- `controller.css` — Per-controller widget styles. Covers
  `.luxar-gui__controller` (flex row with label + widget), the range
  slider (WebKit and Firefox thumb/track variants), number input
  (fixed 52px width, monospace, `text-primary`, spinner-stripped),
  generic string input, checkbox, `<select>` (its `option` colors are
  centralized in `styles/components/select-menu.css`, the single sanctioned
  place for them), and primary button. Light-theme overrides at the bottom.
- `folder.css` — Nested folder styles. Defines `.luxar-gui__folder`
  (column flex), `.luxar-gui__folder-title` (clickable header),
  `.luxar-gui__folder-caret` (expand/collapse glyph), the indented
  child container with a left border, and the `--open` / `--closed`
  modifier states.

## How these files are loaded

The library does not import its own CSS — the application stylesheet
does. `src/styles/index.css` pulls all three files in via
`@import '../ui/gui/styles/<file>.css'`, so they ship as part of the
embed-safe library bundle alongside `components/*.css` and the theme
overrides.

## BEM-ish class naming

Selectors follow a light BEM convention:

- Block: `.luxar-gui`
- Element: `.luxar-gui__scroll`, `.luxar-gui__controller-name`,
  `.luxar-gui__folder-caret`
- Modifier: `.luxar-gui__folder--open`,
  `.luxar-gui__controller--boolean`,
  `.luxar-gui__input--number`

The DOM nodes carrying these classes are produced by `../gui.ts`,
`../folder.ts`, `../controller.ts`, and the concrete controllers in
`../controllers/`.

## See Also

- [`../README.md`](../README.md) — Custom GUI library overview and API.
- [`../../../styles/README.md`](../../../styles/README.md) — Overall
  viewer CSS architecture and where these files plug in.
- [`../../../styles/themes/`](../../../styles/themes/) — Frosted-
  and liquid-glass theme overrides that target `.luxar-gui`.
