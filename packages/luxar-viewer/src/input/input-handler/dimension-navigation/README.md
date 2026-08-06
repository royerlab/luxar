# `input-handler/dimension-navigation/`

nD dimension navigation math + UI lifecycle.

- `compute-step.ts` — the call-site helpers used by the `[`/`]` and digit
  bindings: `computeDimensionStep`, `getSelectedDimensionIndex`,
  `resolveSelectedDimension`. Pure functions returning result records.
- `step-math.ts` — `calculateStepSize` + `calculateNextPosition` plus
  `NavigationConfig` / `DEFAULT_NAV_CONFIG`. Pure step + boundary math.
- `selection.ts` — `getNonDisplayedDimensions`, `mapKeyToDimension`,
  `getNextDimensionIndex`. Pure dim-index helpers.
- `setup.ts` — the four lifecycle bodies (`initDimensionSliders`,
  `initAnimationManager`, `clearDimensionUI`, `updateAllNDNodes`)
  extracted from the orchestrator, plus the `DimensionSlidersFactory`
  type (re-exported from `input/input-handler.ts`) — injected from
  `core/app.ts` so the input layer only knows the slider panel via
  `import type`, respecting the `input` → `ui` dep-cruiser layer rule.
  The orchestrator owns mutable state and exposes it through
  getter/setter callbacks on `DimNavSetupCtx`; these functions never
  carry `this`.
