# ui

UI configuration slice. Owns z-index layering for overlays, timing constants for transient UI (error auto-dismiss, help-close debounce), loading-spinner geometry, the debug-console panel/style settings, the scale-bar overlay, and per-component border-radius/padding tokens for the dataset browser, debug console, rendering-controls panel, and data monitor.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal and `types.ts` defines the interface. This section has no `validate.ts` — its values are unconstrained UI tokens and are not invoked by the central dispatcher.

Styling beyond these numeric tokens lives in CSS variables and classes under `src/styles/` (see the theming system in `src/themes/`); this slice deliberately keeps colors and typography out of TypeScript.

## Contents

- `data.ts` — `uiConfig: UIConfig`. Groups: `zIndex` (base 100–199, mid 1000–1999, top 2000+), `timings` (`errorAutoDismissMs`, `helpClickDelayMs`), `spinner` (`size`, `borderWidth`), `debugConsole` (panel size/limits, resize border width, style colors/blur/shadow), `scaleBar` (`targetWidthPx`, `position`), and `components` (per-component `borderRadius`/`padding` tokens for `datasetBrowser`, `debugConsole`, `renderingControls`, `dataMonitor`).
- `types.ts` — `UIConfig` interface plus the nested `DebugConsoleConfig` and `UIComponentsConfig` interfaces. `scaleBar.position` is a `'bottom-left' | 'bottom-right'` union.

## Public API

- `uiConfig` — re-exported through `../../index.ts` into `AppConfig.ui`.
- `UIConfig`, `DebugConsoleConfig`, `UIComponentsConfig` — re-exported through `../../types.ts`.
