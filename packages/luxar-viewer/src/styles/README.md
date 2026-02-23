# luxar-viewer/src/styles

CSS architecture for the Luxar viewer, organized as a four-tier cascade: reset, base, components, and theme overrides.

## Architecture

All styles are imported through `index.css` in strict cascade order:

```
1. reset.css          — Browser normalization (box-sizing, margins, scrollbars)
2. base/              — Foundational styles applied globally
   ├── typography.css — Font families, sizes, and weights
   ├── layout.css     — Grid system, viewport setup, container widths
   └── utilities.css  — Reusable utility classes (flex, alignment, spacing)
3. components/        — Scoped styles for individual UI components
4. themes/            — Theme-specific visual overrides (glass effects)
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

| File | Component |
|------|-----------|
| `dimension-sliders.css` | nD dimension navigation sliders |
| `dataset-browser.css` | Directory listing and dataset selector |
| `debug-console.css` | In-viewer debug console overlay |
| `data-loading-monitor.css` | Loading progress indicators |
| `error-dialog.css` | Error display dialogs |
| `help-overlay.css` | Keyboard shortcut help overlay |
| `resolution-indicator.css` | Adaptive DPR resolution badge |

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
