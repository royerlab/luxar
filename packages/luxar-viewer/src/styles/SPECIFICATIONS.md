# luxar-viewer.styles - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-05-10

## Purpose

The `luxar-viewer.styles` package defines the viewer-owned CSS cascade. It keeps layout, component styling, and theme-specific visual effects deterministic while avoiding collisions with host pages that embed the viewer.

---

## Core Concepts

### Cascade Layers

Styles are loaded through `index.css` in this order:

1. `reset.css` normalizes browser defaults and box sizing.
2. `base/` provides typography, layout primitives, and utility classes.
3. `components/` styles viewer UI components.
4. `themes/` adds theme-specific visual treatments such as frosted or liquid glass effects.

Later files may refine earlier variables or component rules, but component CSS should not rely on import side effects outside this order.

### Viewer Namespace

All viewer-owned selectors use the `luxar-` prefix. Component CSS follows the BEM-style conventions documented in `../../CONVENTIONS.md`:

```text
luxar-block
luxar-block__element
luxar-block--modifier
luxar-block__element--state
```

This namespace is required because the viewer uses regular DOM rather than Shadow DOM.

### Theme Variables

Component styles read CSS custom properties injected by `src/themes/ThemeManager` instead of hardcoded theme colors:

```css
.luxar-panel {
  background: var(--luxar-bg-primary);
  color: var(--luxar-text-primary);
  border: 1px solid var(--luxar-border-default);
}
```

Theme files may add visual effects, but they should preserve the same semantic variables so components remain theme-agnostic.

---

## Data Structures

### CSS File Groups

```text
styles/
  index.css              # Single import entry point
  reset.css              # Browser normalization
  base/
    typography.css       # Font family, size, line-height rules
    layout.css           # Viewport, container, grid, flex primitives
    utilities.css        # Prefixed utility classes
  components/*.css       # Component-scoped UI styles
  themes/*.css           # Theme-specific overrides/effects
```

**Invariants**:

- `index.css` is the only package entry point imported by application bootstrap.
- Component selectors are prefixed with `luxar-`.
- Theme files override variables or add theme-scoped effects; they do not redefine component structure.
- No Tailwind-style unprefixed utility classes are introduced.

---

## Algorithms

### Style Resolution

**Purpose**: Apply a deterministic visual style to every viewer component.

**Inputs**:

- `index.css` import order.
- Theme variables injected by `ThemeManager`.
- Optional theme override file for the selected theme.

**Outputs**:

- Browser-computed CSS for viewer elements.

**Algorithm**:

```text
1. Load reset styles.
2. Load base typography/layout/utilities.
3. Load component CSS.
4. ThemeManager injects semantic CSS variables on the viewer root/document.
5. If the selected theme has an override file, load its theme-scoped effects.
6. Components resolve colors, spacing, shadows, and borders through variables.
```

**Complexity**: CSS resolution is handled by the browser. Runtime theme changes are O(number of theme variables + affected elements).

**Edge Cases**:

- Missing variables should fall back to safe values in component CSS when practical.
- Embedded host pages may define similarly named classes; the `luxar-` prefix limits accidental matches.
- Reduced-motion or low-power contexts should prefer disabling expensive visual effects in theme overrides rather than changing component markup.

---

## Validation Rules

- **Prefixing**: New selectors must be under the `luxar-` namespace unless they target browser pseudo-elements scoped beneath a `luxar-` selector.
- **Theme coupling**: Components must use CSS variables for theme-dependent color, spacing, borders, and shadows.
- **Cascade order**: Add new component files through `index.css` after base styles and before theme overrides.
- **No `!important`**: Resolve specificity through selector design and import order.
- **No unscoped globals**: Global element selectors belong in reset/base files only.

---

## Cross-Language Compatibility

Not applicable. CSS consumes runtime variables produced by TypeScript theme code; the contract is the variable names documented by `src/themes/SPECIFICATIONS.md`.

---

## Related Specifications

- `luxar-viewer.themes` - Theme token generation and ThemeManager behavior (see `../themes/SPECIFICATIONS.md`).
- `luxar-viewer.ui` - UI components that consume these styles (see `../ui/SPECIFICATIONS.md`).
- Viewer conventions - BEM naming and CSS namespace rules (see `../../CONVENTIONS.md`).

---

## Changelog

- **v1.0.0** (2026-05-10): Initial specification for the styles package.
