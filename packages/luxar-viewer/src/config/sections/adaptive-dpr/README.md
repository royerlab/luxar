# adaptive-dpr

Adaptive device-pixel-ratio configuration slice. Owns the construction-time defaults for the FPS-driven DPR scaler: scale-up/down FPS thresholds, multiplicative scaling factors, the floor on DPR, and the hysteresis / evaluation timings used to avoid rapid quality oscillation.

Conforms to the section-trio pattern documented in [../README.md](../README.md): `data.ts` exports the literal and `types.ts` defines the interface. This slice has no `validate.ts` — it carries only numeric defaults consumed by the renderer's DPR controller; the runtime on/off toggle lives in `renderingControls.defaults.adaptiveDPREnabled`, not here.

## Contents

- `data.ts` — `adaptiveDPRConfig: AdaptiveDPRConfig`. Defaults: `enabled: true`, `minFPS: 50`, `maxFPS: 58`, `minDPR: 0.5`, `scaleDownFactor: 0.9`, `scaleUpFactor: 1.05`, `hysteresisSeconds: 3`, `evaluationIntervalMs: 500`. Asymmetric scaling (slower up, faster down) prevents quality flicker.
- `types.ts` — `AdaptiveDPRConfig` interface. `enabled` is the construction-time default; runtime toggling is delegated to `renderingControls.defaults.adaptiveDPREnabled`.

## Public API

- `adaptiveDPRConfig` — re-exported through `../../index.ts` into `AppConfig.adaptiveDPR`.
- `AdaptiveDPRConfig` — re-exported through `../../types.ts`.
