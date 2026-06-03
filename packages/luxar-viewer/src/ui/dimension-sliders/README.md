# `ui/dimension-sliders/`

Pure math helpers for the dimension sliders panel.

This folder is the private-helpers sibling of the public-API file
`../dimension-sliders.ts`. It contains no DOM, no THREE, no scene state
— just the value/fraction/wrap arithmetic that drives the slider
thumbs, factored out so it can be unit-tested without a browser and
reused by any range-style UI that needs the same math.

## Files

| File             | Purpose                                                 |
| ---------------- | ------------------------------------------------------- |
| `slider-math.ts` | Framework-free clamp/wrap/fraction helpers (see below). |

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

`clampInteger` delegates to `clamp` from `../gui/format/value-formatting`
so the codebase shares one clamp implementation.

## Consumers

- `../dimension-sliders.ts` — the public-API panel; imports all five
  helpers for thumb positioning and value/wrap handling.
- `../../tests/unit/ui/slider-math-utils.test.ts` — unit tests covering
  the cyclic-wrap, degenerate-range, and thumb-offset edge cases.
