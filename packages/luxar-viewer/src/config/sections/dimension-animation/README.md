# dimension-animation

Dimension-animation configuration slice. Owns defaults and presets for FPS-based playback through dimension ranges (e.g. animating a time or Z axis), including loop mode, direction, quick-pick rates (as fps and as periods), frame-time floors, the slideshow threshold, and target-vs-actual FPS feedback thresholds.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal and `types.ts` defines the interface. This section has no `validate.ts` — its values are simple literals with no cross-field invariants for the dispatcher to enforce.

## Contents

- `data.ts` — `dimensionAnimationConfig: DimensionAnimationConfig`. Defines `defaults` (`targetFPS: 10`, `loop: 'loop'`, `direction: 'forward'`), `presets.fps` (`[0.5, 1, 2, 5, 10, 15, 30, 60, 120]`) and `presets.periodSeconds` (`[1, 2, 5, 10, 20, 30, 60, 120]`, the Speed control's period view, seconds per step, one row of chips; anything slower is typed in its Custom field) with `customMin: 1 / MAX_ANIMATION_PERIOD_S` (one step every 30 days, the slowest period that can be typed; the bound is declared in `types/zarr.ts` and checked against Python's `MAX_ANIMATION_PERIOD_S`) / `customMax: 120`, `timing` (`minFrameTimeMs: 8` ~ 120 fps cap, reached only on a ≥120 Hz display; `continuousTraverseSeconds: 10` for continuous-axis traversal; `slideshowPeriodMs: 2000` — playback at this period or slower is a SLIDESHOW: each step loads like a manual one, outside the playback directives, and its clock is held while a reveal-on-arrival waypoint flight is in the air), `ui` feedback (`showFPSFeedback: true`, `feedbackThreshold: 0.8`), and `playback` (`budgetFraction: 0.6`, `minBudgetMs: 8`, `overheadReserveMs: 50`) — the per-tick LOD time budget handed to the progressive loaders during playback: `max(window × fraction, window − reserve, floor)`.
- `types.ts` — `DimensionAnimationConfig` interface. `loop` is `'once' | 'loop' | 'bounce'`; `direction` is `'forward' | 'backward'`; `playback` documents the budget semantics (loaders stream sub-LODs until the budget runs out, then commit — see `DimensionAnimationManager.getFrameBudgetMs`).

## Loop modes

- **once** — play through the range and stop at the end.
- **loop** — restart from the beginning on reaching the end.
- **bounce** — reverse direction at each end (ping-pong).

## Public API

- `dimensionAnimationConfig` — re-exported through `../../index.ts` into `AppConfig.dimensionAnimation`.
- `DimensionAnimationConfig` — re-exported through `../../types.ts`.
