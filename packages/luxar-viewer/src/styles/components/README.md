# components/

Per-UI-surface component styles for the Luxar viewer. Each file in this
folder targets one concrete UI element (a panel, an overlay, a dialog,
a badge) and is imported in bulk by the embed-safe library entry
`../index.css`.

Every selector here is scoped under a `.luxar-*` class so the rules are
safe to ship inside a host page without leaking into the embedder's
chrome. Colors, spacing, radii, and shadows reference the
`--luxar-*` CSS custom properties injected at runtime by the
`ThemeManager` (see `../../themes/`) — no hardcoded palette values, no
`!important`.

## Files

| File                       | UI surface                                                                                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `colormap-legend.css`      | Bottom-right overlay showing per-layer colormap gradients (`ui/colormap-legend.ts`).                                                                              |
| `control-rail.css`         | Slim left-edge activity rail, its flyout, and panel popovers (`ui/control-rail/`).                                                                                |
| `data-loading-monitor.css` | Real-time data-loading performance monitor with multi-state UI (`ui/data-loading-monitor.ts`).                                                                    |
| `dataset-browser.css`      | Modal panel for navigating and selecting Zarr datasets (`ui/dataset-browser.ts`).                                                                                 |
| `debug-console.css`        | In-app developer console for viewing intercepted browser console output (`ui/debug-console.ts`).                                                                  |
| `dimension-sliders.css`    | Napari-inspired slider controls for nD dataset navigation (`ui/dimension-sliders.ts`).                                                                            |
| `error-dialog.css`         | Styled error messages with guidance plus loading-spinner indicator (`ui/error-overlay.ts`, `ui/loading-indicator.ts`).                                            |
| `help-overlay.css`         | Keyboard-shortcuts and controls reference panel (`ui/help-overlay.ts`).                                                                                           |
| `layers-panel.css`         | Napari-inspired per-layer controls for visibility, display range, gamma, and blending mode (`ui/layers/layers-panel.ts`).                                         |
| `overlay-layer.css`        | Screen-space overlay container for text / image / HTML elements painted over the 3D canvas.                                                                       |
| `performance-monitor.css`  | Compact square performance readout (FPS / ms / graph) docked into the control rail as its footer (`ui/performance-monitor.ts`).                                   |
| `recording-panel.css`      | Screenshot and video recording panel, recording indicator, and confirmation dialog (`ui/recording-panel.ts`).                                                     |
| `resolution-indicator.css` | Subtle badge that appears when adaptive DPR has reduced rendering resolution.                                                                                     |
| `scale-bar.css`            | Physical-unit scale-bar overlay for microscopy figure generation (bar + label, bottom-left by default).                                                           |
| `select-menu.css`          | Native `<select>` option styling — single source of truth, colored from the theme's `--luxar-menu-*` tokens (OS-layer option popups can't use translucent glass). |
| `toast.css`                | Brief auto-dismissing notification shown at bottom-center of the viewport (`ui/toast.ts`).                                                                        |

## Conventions

- Class names are prefixed with `luxar-` (e.g. `.luxar-colormap-legend`,
  `.luxar-layers-panel`) so the stylesheet is embed-safe.
- Modifier states use `--` suffixes (e.g. `.luxar-toast--visible`),
  matching the BEM-ish style used by the custom GUI library in
  `../../ui/gui/styles/`.
- Colors and spacing reference `var(--luxar-*)` properties from the
  active theme. The dark/light split is handled by
  `[data-theme='dark' | 'light']` attribute selectors at the document
  root, not by per-file forks.
- Scrollbars use `::-webkit-scrollbar` with theme-aware track/thumb
  colors.

## See Also

- [`../README.md`](../README.md) — Overall viewer CSS architecture,
  including how this folder plugs into `index.css` (embed-safe) versus
  `standalone.css` (host-page chrome).
- [`../base/`](../base/) — Layout primitives, typography, utility
  classes consumed by these component sheets.
- [`../themes/`](../themes/) — Named theme overrides (frosted-glass,
  liquid-glass) that target the same `.luxar-*` selectors.
