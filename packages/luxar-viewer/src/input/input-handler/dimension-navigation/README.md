# `input-handler/dimension-navigation/`

nD input coordination + UI lifecycle.

- `compute-step.ts` — the call-site helpers used by the `[`/`]` and digit
  bindings: `computeDimensionStep`, `getSelectedDimensionIndex`,
  `resolveSelectedDimension`. Pure functions returning result records.
- Pure step and selection helpers live with scene dimension state under
  `scene/dims/`.
- `setup.ts` — the input/UI lifecycle bodies (`initDimensionSliders`,
  `initAnimationManager`, `clearDimensionUI`) extracted from the orchestrator.
  Scene loading for the selected slice lives in `scene/dimension-loading.ts`.
  The `DimensionSlidersFactory` capability
  type lives in `panel-capabilities.ts` and is exported through `input/index.ts`.
  The orchestrator owns mutable state and exposes it through
  getter/setter callbacks on `DimNavSetupCtx`; these functions never
  carry `this`.
