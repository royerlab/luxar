# scene

3D scene visual configuration slice. Owns the canvas background color and the default fit-to-bounds framing ratio, plus a placeholder shader-config object reserved for future per-geometry shader knobs.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher.

Note: global exposure / offset / gamma are intentionally **not** in this section — they live in `renderingControls.defaults` and are applied inside the mega-shader post-processing pass. This slice only owns scene-level constants that are independent of the user-adjustable rendering settings.

## Contents

- `data.ts` — `sceneConfig: SceneConfig` (`backgroundColor` as a 24-bit hex int, `defaultFitRatio` in `(0, 1]`) and `shaderConfig: ShaderConfig` (currently an empty `points: {}` placeholder; per-node color adjustments live on each material).
- `types.ts` — `SceneConfig` interface and `ShaderConfig` interface (the latter is a forward-compatible stub: `points: Record<string, never>`).
- `validate.ts` — `validateScene(config, errors, warnings)`. Requires `backgroundColor` to be an integer in `[0, 0xffffff]` (NaN-hardened via `Number.isInteger`), and `defaultFitRatio` to be a finite number in `(0, 1]`.

## Public API

- `sceneConfig`, `shaderConfig` — re-exported through `../../index.ts` into `AppConfig.scene` and `AppConfig.shader`.
- `SceneConfig`, `ShaderConfig` — re-exported through `../../types.ts`.
- `validateScene` — called from `../../validation.ts`.
