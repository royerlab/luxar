# luxar-viewer/src/themes/themes

Built-in `Theme` definitions registered by the `ThemeManager`. Each file exports a single typed `Theme` object describing colors, typography, spacing, effects, and z-index layers.

## Themes

| Theme         | Export              | File                     | Description                                                                 |
| ------------- | ------------------- | ------------------------ | --------------------------------------------------------------------------- |
| Dark          | `darkTheme`         | `dark.theme.ts`          | Classic high-contrast dark theme (`#111111` background, `#e0e0e0` text).    |
| Light         | `lightTheme`        | `light.theme.ts`         | Inverted-palette light theme; inherits typography/spacing from `darkTheme`. |
| Frosted Glass | `frostedGlassTheme` | `frosted-glass.theme.ts` | Modern glassmorphism with heavy backdrop blur (32-64px) and soft shadows.   |
| Liquid Glass  | `liquidGlassTheme`  | `liquid-glass.theme.ts`  | True glassmorphism with inner-glow `inset` shadows and light (1-3px) blur.  |

## Theme Shape

Every theme conforms to the `Theme` interface declared in `../types.ts`:

```typescript
import type { Theme } from '../types';

export const myTheme: Theme = {
  id: 'my-theme',
  name: 'My Theme',
  description: '...',
  colors: { background, text, semantic, interactive, border, visualization },
  typography: { fontFamily, fontSize, fontWeight, lineHeight },
  spacing: { 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20 },
  effects: { borderRadius, shadow, blur, opacity, transition },
  zIndex: { base, dropdown, modal, popover, tooltip },
};
```

The `ThemeManager` flattens each theme into `--luxar-*` CSS custom properties on `:root` when activated. See the parent [themes/README.md](../README.md) for the full property catalog.

## Design Notes

- **`darkTheme`** is the canonical reference. `lightTheme` imports it and spreads `typography`, `spacing`, `effects.borderRadius`, `effects.blur`, `effects.opacity`, `effects.transition`, and `zIndex` to stay structurally identical; only colors and shadows differ.
- **Glass themes** (`frostedGlassTheme`, `liquidGlassTheme`) use translucent `rgba(255, 255, 255, …)` backgrounds and are intended to overlay arbitrary canvas content. They share an Apple-system font stack and cubic-bezier transitions.
- **Liquid Glass** encodes inner-glow highlights directly in `effects.shadow` via `inset` box-shadows (e.g. `inset 0 0 20px -5px rgba(255, 255, 255, 0.7)`), and pairs them with light backdrop blur (1-3px) — in contrast to Frosted Glass which relies on heavy blur (32-64px) without inner shadows.
- Spacing scale, font-weight scale, and z-index layers are identical across all four themes to keep component styling portable.

## Adding a Theme

1. Create `my-theme.theme.ts` in this folder exporting a `Theme`.
2. Import and register it from `../theme-manager.ts` (or via `ThemeManager.registerTheme()` at runtime).
3. If the theme needs companion CSS overrides (e.g. backdrop-filter rules), add a `my-theme.css` under `src/styles/themes/` and ensure it is imported by `src/styles/index.css`.

## See Also

- [../README.md](../README.md) — `ThemeManager`, runtime injection, persistence
- [../types.ts](../types.ts) — `Theme` and related interfaces
- [../../styles/README.md](../../styles/README.md) — CSS that consumes the injected variables
