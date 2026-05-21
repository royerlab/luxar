# dimension-animation

Dimension-animation configuration slice. Owns defaults and presets for FPS-based playback through dimension ranges (e.g. animating a time or Z axis), including loop mode, direction, quick-pick FPS values, frame-time floors, and target-vs-actual FPS feedback thresholds.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal and `types.ts` defines the interface. This section has no `validate.ts` — its values are simple literals with no cross-field invariants for the dispatcher to enforce.

## Contents

- `data.ts` — `dimensionAnimationConfig: DimensionAnimationConfig`. Defines `defaults` (`targetFPS: 10`, `loop: 'loop'`, `direction: 'forward'`), `presets.fps` (`[1, 2, 5, 10, 15, 30, 60]`) with `customMin: 0.1` / `customMax: 120`, `timing` (`minFrameTimeMs: 16` ~ 60 fps cap, `continuousTraverseSeconds: 10` for continuous-axis traversal), and `ui` feedback (`showFPSFeedback: true`, `feedbackThreshold: 0.8`).
- `types.ts` — `DimensionAnimationConfig` interface. `loop` is `'once' | 'loop' | 'bounce'`; `direction` is `'forward' | 'backward'`.

## Loop modes

- **once** — play through the range and stop at the end.
- **loop** — restart from the beginning on reaching the end.
- **bounce** — reverse direction at each end (ping-pong).

## Public API

- `dimensionAnimationConfig` — re-exported through `../../index.ts` into `AppConfig.dimensionAnimation`.
- `DimensionAnimationConfig` — re-exported through `../../types.ts`.
