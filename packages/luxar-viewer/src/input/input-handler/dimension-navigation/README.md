# `input-handler/dimension-navigation/`

nD input coordination + UI lifecycle.

- `compute-step.ts` — the call-site helpers used by the `[`/`]` and digit
  bindings: `computeDimensionStep`, `getSelectedDimensionIndex`,
  `resolveSelectedDimension`. Pure functions returning result records.
- Pure step and selection helpers live with scene dimension state under
  `scene/dims/`.
- `setup.ts` — the four lifecycle bodies (`initDimensionSliders`,
  `initAnimationManager`, `clearDimensionUI`, `updateAllNDNodes`)
  extracted from the orchestrator. The `DimensionSlidersFactory` capability
  type lives in `panel-capabilities.ts` and is exported through `input/index.ts`.
  The orchestrator owns mutable state and exposes it through
  getter/setter callbacks on `DimNavSetupCtx`; these functions never
  carry `this`.
