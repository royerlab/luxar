# luxar-viewer.themes - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-05-06

## Purpose

The `luxar-viewer.themes` package provides a runtime theming system for the
viewer's HTML/CSS UI surface. Themes are JavaScript objects that get
applied as CSS custom properties on `:root`, propagated through the cascade
to all component styles. Producers can extend the registry with custom
themes; users get persistent theme selection across sessions.

**Core Responsibility**: Decouple visual appearance from component styling
so the same UI can render in dark / light / glass modes without per-component
forks, while keeping a single source of truth (the active `Theme` object)
that all consumers — components, dashboards, shaders that read DOM colors —
can query.

---

## Table of Contents

1. [Theme Object Model](#1-theme-object-model)
2. [ThemeManager Singleton](#2-thememanager-singleton)
3. [CSS Variable Injection](#3-css-variable-injection)
4. [Persistence](#4-persistence)
5. [Glass Filter Pipeline](#5-glass-filter-pipeline)
6. [Adding a New Theme](#6-adding-a-new-theme)

---

## 1. Theme Object Model

A `Theme` is a plain serialisable JavaScript object. It has six top-level
sections (see `types.ts`):

| Section      | Purpose                                                |
| ------------ | ------------------------------------------------------ |
| `colors`     | All chromatic values (text, background, semantic, viz) |
| `typography` | Font families, sizes, weights, line heights            |
| `spacing`    | A 12-step pixel scale used for padding/margin tokens   |
| `effects`    | Border radius, shadows, backdrop blur, transitions     |
| `zIndex`     | Layered stacking values (`base / dropdown / modal …`)  |
| `meta`       | `id`, `name`, optional `description`                   |

The full type contract is defined in `src/themes/types.ts`. Every field is
typed (no `any`) so a typo in a custom theme is a compile-time error.

### 1.1 Color Tokens

Colors group by **purpose**, not appearance — a "background" token has
the same name in dark and light themes; only its value differs. This is
what lets components stay theme-agnostic.

Groups:

- **Background**: `primary` / `secondary` / `tertiary` / `overlay`
- **Text**: `primary` / `secondary` / `muted` / `disabled` / `inverse`
- **Semantic**: `success` / `warning` / `error` / `info` / `highlight`
- **Interactive**: `default` / `hover` / `active` / `focus` / `disabled`
- **Border**: `default` / `subtle` / `strong` / `focus`
- **Visualization**: `hot` / `warm` / `cold` / `neutral`

### 1.2 Typography Tokens

| Field          | Values                                            |
| -------------- | ------------------------------------------------- |
| `family.base`  | UI font stack                                     |
| `family.mono`  | Code/monospace stack                              |
| `family.display` | Headings                                        |
| `size`         | `xs / sm / md / lg / xl / 2xl / 3xl`              |
| `weight`       | `normal / medium / semibold / bold`               |
| `lineHeight`   | `tight / normal / relaxed`                        |

### 1.3 Spacing Scale

Twelve discrete values mapped from a key set to pixel values:

```
[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
   →
[0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40] px
```

This is the same scale Tailwind uses, intentionally — components that
already follow it work without further translation.

---

## 2. ThemeManager Singleton

The `ThemeManager` is a process-level singleton. Instantiation is private;
consumers always go through `ThemeManager.getInstance()`.

### 2.1 Public API

| Method                         | Returns / Side effect                                  |
| ------------------------------ | ------------------------------------------------------ |
| `getInstance()`                | The singleton                                          |
| `setTheme(id)`                 | Activates a theme + persists choice + notifies listeners |
| `getCurrentTheme()`            | The active `Theme`                                     |
| `getTheme(id)`                 | Lookup by id (returns `undefined` if not registered)   |
| `getAllThemes()`               | Array of registered themes                             |
| `registerTheme(theme)`         | Adds to the registry; idempotent on duplicate id       |
| `onChange(listener)`           | Subscribe; returns unsubscribe function                |
| `resetInstance()`              | **Tests only** — wipe the singleton                    |

### 2.2 Initialisation Order

1. Built-in themes are registered statically when the module loads.
2. On first `getInstance()` call:
   - Read `localStorage[luxar.theme]`.
   - If present and resolves to a registered id, activate it.
   - Otherwise activate the default (`frosted-glass`).
3. The active theme's CSS variables are injected on `:root`.
4. Listeners are not yet wired — callers attach them via `onChange`.

### 2.3 Listener Semantics

- Listeners fire on every `setTheme(id)` where `id !== currentTheme.id`.
- They do **not** fire on initial activation — only on subsequent changes.
- The `unsubscribe` returned by `onChange` is idempotent.
- Listener order is registration order; exceptions thrown by one listener
  do not prevent later listeners from running (caught + logged via `log.warning`).

---

## 3. CSS Variable Injection

Every Theme field is flattened to a CSS custom property on `:root`:

```css
:root {
  --luxar-color-bg-primary:   #111111;
  --luxar-color-text-primary: #f5f5f5;
  --luxar-spacing-md:         8px;
  --luxar-radius-md:          6px;
  --luxar-z-modal:            2000;
  /* ... */
}
```

Component CSS reads these directly:

```css
.luxar-panel {
  background: var(--luxar-color-bg-secondary);
  color:      var(--luxar-color-text-primary);
  padding:    var(--luxar-spacing-md);
  border-radius: var(--luxar-radius-md);
}
```

Naming convention: `--luxar-<section>-<group>-<field>` (kebab-case),
matching the dotted property path. The flattener lives in
`theme-manager.ts::applyToCssVariables`.

---

## 4. Persistence

- Storage key: `StorageKeys.theme` → `luxar.theme`.
- On set, the theme id (string) is written.
- On boot, the id is read; unknown ids fall back to the default and the
  invalid stored value is **left in place** so a later registration can
  pick it up. This handles the case where a producer registers a custom
  theme after page load.
- Quota / disabled-storage failures (private mode strict, sandboxed iframes)
  are logged as warnings and silently ignored — the theme still applies for
  the current session.

---

## 5. Glass Filter Pipeline

The `liquid-glass` theme uses an SVG filter chain to simulate physical
glass behaviour. Pipeline:

1. **Backdrop sample**: `<feGaussianBlur stdDeviation="…">` blurs the
   composited backdrop the panel sits on.
2. **Sobel gradient**: convolution kernel computes ∂L/∂x and ∂L/∂y
   on the blurred sample (used as the displacement map's direction).
3. **Displacement**: `<feDisplacementMap>` shifts pixels by `refractionScale`
   in the gradient direction — geometry-aware refraction (panel edges
   bend more than the centre).
4. **Chromatic aberration**: per-channel translation by
   `±chromaticStrength` along the gradient — simulates wavelength-dependent
   refraction.
5. **Specular highlights**: a `<feSpecularLighting>` pass adds a soft
   reflection seeded from a virtual light source above the panel.

All five parameters are tunable in the theme object's
`effects.glassFilter` block (only present on glass themes; unused fields
are ignored by non-glass themes).

---

## 6. Adding a New Theme

```typescript
import { ThemeManager } from './themes';
import type { Theme } from './themes/types';

const cyberpunk: Theme = {
  meta: { id: 'cyberpunk', name: 'Cyberpunk' },
  colors: { /* ... */ },
  typography: { /* ... */ },
  spacing: { /* ... */ },
  effects: { /* ... */ },
  zIndex: { /* ... */ },
};

ThemeManager.getInstance().registerTheme(cyberpunk);
ThemeManager.getInstance().setTheme('cyberpunk');
```

Validation runs at register time:

- `meta.id` must match `[a-z0-9_-]+` (kebab-case) and be unique.
- All required color groups must be present.
- Numeric scales must be monotonic (size, weight) and finite.

Validation failures throw — the registry stays in its previous state.

---

## File Structure

```
themes/
├── index.ts              — Public exports
├── types.ts              — Theme, ThemeColors, ThemeTypography, etc.
├── theme-manager.ts      — Singleton manager + CSS var injector
├── glass-filters.ts      — SVG filter pipeline for glass themes
└── themes/
    ├── dark.theme.ts
    ├── light.theme.ts
    ├── frosted-glass.theme.ts
    └── liquid-glass.theme.ts
```
