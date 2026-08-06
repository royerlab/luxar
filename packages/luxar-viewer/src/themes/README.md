# luxar-viewer/src/themes

Runtime theming system with CSS custom property injection, persistent user preferences, and advanced glass-effect themes.

## Architecture

```
ThemeManager (singleton)
├── Theme Registry — stores registered themes by ID
├── CSS Variable Injector — applies theme as --luxar-* properties on :root
├── LocalStorage Persistence — remembers user's chosen theme
└── Observer Pattern — notifies listeners on theme change
```

## Built-in Themes

| Theme         | ID              | Description                                |
| ------------- | --------------- | ------------------------------------------ |
| Dark          | `dark`          | Classic dark theme (`#111111` background)  |
| Light         | `light`         | Light theme with bright backgrounds        |
| Frosted Glass | `frosted-glass` | Backdrop blur and glass morphism           |
| Liquid Glass  | `liquid-glass`  | Geometry-aware refraction with SVG filters |

**Default**: `frosted-glass` (unless a saved user preference or `?theme=<id>` overrides it)

## Usage

```typescript
import { ThemeManager } from './theme-manager';

const tm = ThemeManager.getInstance();

// Switch theme
tm.setTheme('dark');

// Get current theme
const current = tm.getCurrentTheme();

// List all registered themes
const allThemes = tm.getAllThemes();

// Register a custom theme
tm.registerTheme(myCustomTheme);

// Listen for changes (returns unsubscribe function)
const unsubscribe = tm.onChange((theme) => {
  console.log('Theme changed to:', theme.name);
});

// Get a specific theme by ID
const dark = tm.getTheme('dark');

// Later: stop listening
unsubscribe();

// For shutdown or test isolation: dispose the singleton
ThemeManager.disposeInstance();
```

The `disposeInstance()` static method tears down observers, CSS variables,
glass filter DOM, and the refraction `MutationObserver`, then clears the
singleton slot so the next `getInstance()` constructs a fresh manager.

## Theme Structure

Each theme implements the `Theme` interface with these sections:

### Colors

- **Background**: primary, secondary, tertiary, overlay
- **Text**: primary, secondary, muted, disabled, inverse
- **Semantic**: success, warning, error, info, highlight
- **Interactive**: default, hover, active, focus, disabled
- **Border**: default, subtle, strong, focus
- **Visualization**: hot, warm, cold, neutral

### Typography

- **Font families**: base, mono, display
- **Font sizes**: xs through 3xl
- **Font weights**: normal, medium, semibold, bold
- **Line heights**: tight, normal, relaxed

### Spacing

Scale keys `[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]` mapping to `[0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40]` px

### Effects

- **Border radius**: none, sm, md, lg, full (pill)
- **Shadows**: sm, md, lg, xl
- **Backdrop blur**: none, sm, md, lg
- **Opacity**: disabled, secondary, hover, full
- **Transitions**: fast (0.1s), normal (0.2s), slow (0.3s)

### Z-Index Layers

| Layer    | Value |
| -------- | ----- |
| base     | 100   |
| dropdown | 1000  |
| modal    | 2000  |
| popover  | 3000  |
| tooltip  | 4000  |

## Glass Filter System

The liquid glass theme uses an advanced SVG filter pipeline:

1. **Geometry-aware refraction** — elements act as convex lenses
2. **Sobel edge detection** — computes gradient for distortion direction
3. **Chromatic aberration** — subtle RGB channel separation
4. **Specular highlights** — simulated light reflections

Filter parameters are configurable: `blurRadius`, `refractionScale`, `chromaticStrength`, `specularIntensity`.

### Opting a panel into the glass themes

Both glass themes key off a **single marker class, `luxar-glass-surface`**. A panel
opts in by adding that class to its root element at creation (e.g.
`el.className = 'luxar-my-panel luxar-glass-surface'`). Three consumers read that
one class — with no per-panel list to keep in sync:

- `frosted-glass.css` — `[data-theme='frosted-glass'] .luxar-glass-surface { … }`
- `liquid-glass.css` — the base / `::before` / `::after` glass layers
- `glass-filters.ts` — `injectGlassRefractionLayers()` and the refraction
  `MutationObserver` (via `GLASS_SURFACE_SELECTOR`)

To make a new panel glass-aware, add the class at its creation site; nothing in
the theme CSS or `glass-filters.ts` needs to change.

## Persistence

User theme preference is stored in `localStorage` under the key `luxar.theme`
(see `StorageKeys.theme` in `src/utils/storage-keys.ts`). Gracefully degrades
in private/incognito mode (logs a warning, continues without persistence).

## File Structure

```
themes/
├── types.ts              — Theme, ThemeColors, ThemeTypography, ThemeEffects, etc.
├── theme-manager.ts      — ThemeManager singleton
├── glass-filters.ts      — SVG filter pipeline + refraction-layer injector for liquid-glass
└── themes/               — Built-in Theme definitions (see Subpackages below)
    ├── dark.theme.ts
    ├── light.theme.ts
    ├── frosted-glass.theme.ts
    └── liquid-glass.theme.ts
```

## Subpackages

- [`themes/`](./themes/README.md) — The four built-in `Theme` objects
  (`darkTheme`, `lightTheme`, `frostedGlassTheme`, `liquidGlassTheme`),
  registered by `ThemeManager`'s constructor.

## Public API

There is no barrel; import each symbol directly from its module:

- `class ThemeManager` from `./theme-manager` — singleton; see Usage above
- `type Theme`, `type ThemeChangeHandler` from `./types`
- `darkTheme`, `lightTheme`, `frostedGlassTheme`, `liquidGlassTheme` from
  `./themes/{dark,light,frosted-glass,liquid-glass}.theme`

Internal, imported directly by
`theme-manager.ts`: `injectGlassFilters`, `removeGlassFilters`,
`injectGlassRefractionLayers`, `removeGlassRefractionLayers`,
`setupGlassRefractionObserver` from `./glass-filters`.
