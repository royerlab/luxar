# `ui/dimension-sliders/`

Pure helpers for the dimension sliders panel: slider math and the
context menu's Speed choices.

This folder is the private-helpers sibling of the public-API file
`../dimension-sliders.ts`. It contains no DOM, no THREE, no scene state
— just the value/fraction/wrap arithmetic that drives the slider
thumbs, factored out so it can be unit-tested without a browser and
reused by any range-style UI that needs the same math.

## Files

| File               | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `slider-math.ts`   | Framework-free clamp/wrap/fraction helpers (see below).  |
| `speed-choices.ts` | The Speed section's rate chips in its two views (below). |

## Exports (`slider-math.ts`)

| Function                                                    | Purpose                                                                                                |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `clampWithCyclicWrap(value, min, max, isCyclic)`            | One-step clamp; when `isCyclic`, underflow wraps to `max` and overflow wraps to `min`.                 |
| `valueToFraction(value, min, max)`                          | Project `value` onto `[0, 1]`. Degenerate range (`min === max`) reports `0.5` so the thumb sits mid.   |
| `fractionToValue(fraction, min, max)`                       | Inverse of `valueToFraction`. Degenerate range returns `min`.                                          |
| `fractionToThumbLeft(fraction, containerWidth, thumbWidth)` | Pixel offset for the thumb's left edge; accounts for `thumbWidth` so the right edge doesn't overshoot. |
| `clampInteger(value, lo, hi)`                               | Integer clamp used for the 0-1000 internal range of `<input type=range>`.                              |

`clampWithCyclicWrap` is intentionally one-step: it does not normalize
multi-period overshoot (e.g. `value = max + 5` does not become
`max + 5 - period`). Callers that need full modular wrap should compose
this helper with a normalization pass.

`clampInteger` delegates to the canonical `clamp` in `../../utils/clamp`
so the codebase shares one clamp implementation.

## Consumers

- `../dimension-sliders.ts` — the public-API panel; imports all five
  helpers for thumb positioning and value/wrap handling.
- `../../tests/unit/ui/slider-math-utils.test.ts` — unit tests covering
  the cyclic-wrap, degenerate-range, and thumb-offset edge cases.

## Exports (`speed-choices.ts`)

The context menu's Speed section reads a playback rate either as steps
per second (`fps`) or as seconds per step (`period`), switched by a
two-segment toggle in the section header. The animation manager keeps the
choice per dimension (`setSpeedView` / `getSpeedView`; unset = `period`
below 1 fps). Either view sets the same `targetFPS`.

| Export                      | Purpose                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| `speedChoices(view)`        | The chips for a view: `presets.fps` (`10`, sub-1 as `1/2`) or `presets.periodSeconds` (`20s`, `1m`). |
| `SPEED_VIEWS`               | The two views in toggle order, with their segment labels and tooltips.                               |
| `formatPeriod(s)`           | `20s`, `1m`, `90s`, `2h` — the largest unit the period is a whole number of.                         |
| `describePeriod(s)`         | The long form for tooltips: `1 step every 2 min`.                                                    |
| `PERIOD_UNITS`              | The units a rate is typed in: s, min, h, d (per step in the period view, per unit time in fps).      |
| `periodInBestUnit(s)`       | A period as `{ amount, unit }` in the unit it reads best in.                                         |
| `rateInBestUnit(view, fps)` | A rate as the view types it (`0.05` fps is `20 s` per step, or `3 /min`): seeds the Custom field.    |
| `rateUnitLabel(view, unit)` | The unit picker's label: `min` (period) or `/min` (fps).                                             |
| `customRate(view, n, unit)` | The fps for a typed rate, or null outside what the viewer plays (30 days to 120 a second).           |
| `isPresetRate(view, fps)`   | Whether a rate is one of the view's chips (else the Custom row is the current choice).               |
