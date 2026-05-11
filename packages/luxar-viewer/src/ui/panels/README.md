# Panels

Modal / floating panels rendered above the WebGL canvas: dataset
browser, dimension sliders, debug console.

## Files

| File | Role |
|------|------|
| `dataset-browser.ts` | "Open dataset" modal — file-system browse + URL paste + recent list |
| `dataset-url-utils.ts` | Pure helpers for URL normalization and validation used by the browser |
| `dimension-sliders.ts` | nD navigation UI — per-dimension slider with step / range / unit |
| `debug-console.ts` | Floating debug console panel (toggled by `Ctrl+L`) |
| `debug-console-formatters.ts` | Pure formatters for log lines (timestamp, level coloring, etc.) |

## Public surface

Panels are constructed by `core/app.ts` and exposed through
factories where lower-layer code needs to mount them (see
`DimensionSlidersFactory` in `src/input/input-handler.ts` for the
dependency-inversion pattern).

## Invariants

- All DOM listeners go through `EventGroup` for bulk teardown.
- Panels are idempotent on `dispose()` and tolerate being disposed
  before `init()` runs (e.g. when scene load fails early).
- Dataset-URL parsing is pure (no DOM, no network). The network
  preflight lives in `dataset-browser.ts` and uses `AbortController`
  for cancellation.
