# ui

UI configuration slice. Owns the inline z-index of the panels that set one, timing constants for transient UI (error auto-dismiss, help-close debounce), the debug console's drag-resize limits, and the scale-bar overlay.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal and `types.ts` defines the interface. This section has no `validate.ts` — its values are unconstrained UI tokens and are not invoked by the central dispatcher.

Styling beyond these numeric tokens lives in CSS variables and classes under `src/styles/` (see the theming system in `src/themes/`); this slice deliberately keeps colors and typography out of TypeScript.

## Contents

- `data.ts` — `uiConfig: UIConfig`. Groups: `zIndex` (`renderingControls`, `recordingPanel`, `layersPanel` — the left-docked panels, below the control rail — and `statsMonitor`; every other layer is stacked by CSS through the theme's `--luxar-z-*` scale), `timings` (`errorAutoDismissMs`, `helpClickDelayMs`), `debugConsole` (`panel` min/max width and height for drag-resizing, `resize.borderWidth`; its default size, position and look live in `debug-console.css`), and `scaleBar` (`targetWidthPx`, `position`).
- `types.ts` — `UIConfig` interface plus the nested `DebugConsoleConfig` interface. `scaleBar.position` is a `'bottom-left' | 'bottom-right'` union.

## Public API

- `uiConfig` — re-exported through `../../index.ts` into `AppConfig.ui`.
- `UIConfig`, `DebugConsoleConfig` — re-exported through `../../types.ts`.
