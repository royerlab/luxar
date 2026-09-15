# luxar-viewer/src/styles

CSS architecture for the Luxar viewer, split between an **embed-safe library entry** (`index.css`) and a **standalone-app entry** (`standalone.css`). The split exists so third-party host pages can import the library without having their `body`, `*`, scrollbar, or focus styles clobbered.

## Architecture

Two entry stylesheets:

- `index.css` — embed-safe. Every rule is scoped to a `.luxar-*` class, a
  `[data-theme=...]` attribute, or a component-local selector, except `:root`
  blocks that declare only `--luxar-*` custom properties. Imports
  `base/utilities.css`, every `components/*.css`, the custom GUI library styles
  from `../ui/gui/styles/`, and the `themes/*.css` overrides.
- `standalone.css` — global host-page chrome. Imports `reset.css`,
  `base/typography.css`, and `base/layout.css`. The standalone app's
  `main.ts` imports **both** files; embedders import only `index.css`.

```
standalone.css (standalone app only)
  ├── reset.css            — Modern CSS reset (Josh Comeau-style)
  └── base/
      ├── typography.css   — Font families, sizes, weights
      └── layout.css       — Grid, viewport, container widths

index.css (library, embed-safe)
  ├── base/utilities.css   — Tailwind-like .luxar-* utility classes
  ├── components/*.css     — Per-UI-surface component styles
  ├── ../ui/gui/styles/    — Custom GUI library (gui, controller, folder)
  ├── themes/*.css         — Named theme overrides
  └── components/coarse-pointer.css — Touch / coarse-pointer adaptations (media-gated only, applied last)
```

## Touch and coarse pointers

Every phone/tablet adaptation lives in **one** file, `components/coarse-pointer.css`,
whose top level contains only `@media` blocks keyed on the pointer/hover media
features (`(pointer: coarse)`, `(any-hover: none)`, `(any-hover: hover)`) — never on
viewport width alone. A mouse-and-keyboard machine therefore receives byte-identical
CSS; `tests/unit/styles/coarse-pointer-css.test.ts` enforces the contract (media-only
top level, pointer-feature preludes, no pointer/hover features anywhere else in the
tree, the load-bearing clamps present). What it does: width clamps via
`min(<desktop width>, calc(100vw - margins))`, `vh` → `dvh` heights behind
`@supports (height: 100dvh)` (iOS Safari's dynamic toolbar), `env(safe-area-inset-*)`
on the rail gutter and the bottom strip (`index.html` declares `viewport-fit=cover`),
`touch-action: manipulation` on buttons, a scrolling items box for the control rail,
and no idle-dimming of the rail where nothing can hover. The rail's items wrapper
(`.luxar-control-rail__items`) is `display: contents` in `control-rail.css` so it is
layout-transparent on fine pointers. Ergonomics, same file: 44px primary targets via
a local `--luxar-hit-min` (36px secondary, 24px thumbs on a 24px hit band,
28px dual-range thumbs on a 28px band),
16px panel inputs (iOS focus-zoom), `:active` press fills with sticky post-tap
`:hover` resets for rail controls, layer rows and play buttons under `(any-hover: none)`, tips on keyboard focus only,
and the base look of the two elements that exist only on coarse pointers (the
dimension sliders' `‹ ›` step buttons and name chip). See the UI Design Guide §11.5.

## CSS Custom Properties

All colors, spacing, and effects are defined as CSS custom properties injected at runtime by the `ThemeManager`. Component styles reference these variables instead of hardcoded values:

```css
/* All properties use the --luxar- prefix */
var(--luxar-bg-primary)
var(--luxar-text-primary)
var(--luxar-spacing-4)
var(--luxar-border-default)
var(--luxar-shadow-md)
```

See `src/themes/` for the full set of available properties.

## Base Styles

### Layout (`base/layout.css`)

- Viewport: Canvas fills the entire window (`width: 100%`, `height: 100%`)
- Container: Max-width 1200px with auto margins
- Grid: `.luxar-grid` with responsive column support
- Flexbox: `.luxar-flex`, `.luxar-flex-col`, `.luxar-flex-row`

### Utilities (`base/utilities.css`)

Tailwind-like utility classes, all prefixed with `luxar-`:

- **Flex alignment**: `luxar-items-center`, `luxar-items-start`, `luxar-justify-between`
- **Spacing**: `luxar-gap-2`, `luxar-p-4`, `luxar-m-2`
- **Display**: `luxar-hidden`, `luxar-block`, `luxar-inline-block`

Plus two shared **panel recipes** (not atomic utilities — add them alongside a
panel's own BEM class, and do not restate them per-panel):

- `luxar-panel-header` — the title row: flex, standard rhythm, hairline rule
- `luxar-panel-close` — the 28x28 close affordance with the stroke SVG contract

## Component Styles

Each UI component has a dedicated CSS file:

| File                       | Component                              |
| -------------------------- | -------------------------------------- |
| `colormap-legend.css`      | Colormap legend display                |
| `context-menu.css`         | Shared right-click context menus       |
| `control-rail.css`         | Left-edge activity rail and popovers   |
| `data-loading-monitor.css` | Loading progress indicators            |
| `dataset-browser.css`      | Directory listing and dataset selector |
| `debug-console.css`        | In-viewer debug console overlay        |
| `dimension-sliders.css`    | nD dimension navigation sliders        |
| `error-dialog.css`         | Error display dialogs                  |
| `help-overlay.css`         | Keyboard shortcut help overlay         |
| `layers-panel.css`         | Per-node layer visibility controls     |
| `overlay-layer.css`        | Generic overlay layer container        |
| `performance-monitor.css`  | Rail-docked performance readout        |
| `recording-panel.css`      | Screenshot and video capture panel     |
| `resolution-indicator.css` | Adaptive DPR resolution badge          |
| `scale-bar.css`            | Physical unit scale bar                |
| `select-menu.css`          | Native `<select>` option styling       |
| `toast.css`                | Toast notification messages            |

## Theme Overrides

Theme-specific CSS files provide visual enhancements:

- `themes/frosted-glass.css` — Backdrop blur and glass morphism effects
- `themes/liquid-glass.css` — Geometry-aware refraction with SVG filters

These files are activated by the `ThemeManager` when the corresponding theme is selected.

## Conventions

- All custom classes use the `luxar-` prefix to avoid conflicts
- Component styles reference CSS custom properties rather than hardcoded
  colors; the literals still in the tree are inventoried in the
  [UI Design Guide](../../../../docs/guides/developer/UI_DESIGN_GUIDE.md)
  §15.1–15.5 (drift, migrate it) and §15.6 (sanctioned, leave it)
- Scrollbar styling is dark-theme optimized via `::-webkit-scrollbar`
- `!important` only in the three sanctioned categories (reduced-motion
  overrides, control-rail docking overrides, overlay state-forcing) — see the
  [UI Design Guide](../../../../docs/guides/developer/UI_DESIGN_GUIDE.md) §13

## See Also

- [UI Design Guide](../../../../docs/guides/developer/UI_DESIGN_GUIDE.md) —
  **the authoritative visual-design ground truth** (tokens, surface recipes,
  color semantics, iconography, motion, a11y, drift inventory)
- [base/](./base/README.md) — Layout primitives, typography, utility classes
- [components/](./components/README.md) — Per-UI-surface component styles
- [themes/](./themes/) — Named theme overrides (`frosted-glass.css`, `liquid-glass.css`)
- [`../themes/`](../themes/README.md) — `ThemeManager` and the `--luxar-*` custom properties consumed by every file here
