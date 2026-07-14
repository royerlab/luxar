# luxar-viewer/src/styles

CSS architecture for the Luxar viewer, split between an **embed-safe library entry** (`index.css`) and a **standalone-app entry** (`standalone.css`). The split exists so third-party host pages can import the library without having their `body`, `*`, scrollbar, or focus styles clobbered.

## Architecture

Two entry stylesheets:

- `index.css` — embed-safe. Every rule is scoped to a `.luxar-*` class, a
  `[data-theme=...]` attribute, or a component-local selector. Imports
  `base/utilities.css`, every `components/*.css`, the custom GUI library
  styles from `../ui/gui/styles/`, and the `themes/*.css` overrides.
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
  └── themes/*.css         — Named theme overrides (applied last)
```

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
- **Display**: `luxar-hidden`, `luxar-block`, `luxar-inline-flex`

## Component Styles

Each UI component has a dedicated CSS file:

| File                       | Component                              |
| -------------------------- | -------------------------------------- |
| `colormap-legend.css`      | Colormap legend display                |
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
- Component styles reference CSS custom properties, never hardcoded colors
- Scrollbar styling is dark-theme optimized via `::-webkit-scrollbar`
- No `!important` overrides — cascade order handles specificity

## See Also

- [base/](./base/README.md) — Layout primitives, typography, utility classes
- [components/](./components/README.md) — Per-UI-surface component styles
- [themes/](./themes/) — Named theme overrides (`frosted-glass.css`, `liquid-glass.css`)
- [`../themes/`](../themes/README.md) — `ThemeManager` and the `--luxar-*` custom properties consumed by every file here
