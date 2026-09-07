# controls

Camera-controls configuration slice. Owns defaults for the three control modes — `fly`, `orbit`, and the scene-scale multipliers used to adapt control parameters to the bounding-box diagonal.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal, `types.ts` defines the interfaces, `validate.ts` exports a section validator invoked by the central dispatcher.

Note: the active control mode (`controlType: 'orbit' | 'fly' | 'ortho'`) and `autoRotate` toggle live in `renderingControls.defaults` — this slice only owns the per-mode knobs (speeds, damping, zoom limits, scale factors).

## Contents

- `data.ts` — `controlsConfig: ControlsConfig`. Defines `scaleMultipliers` (`minDistanceFactor`, `maxDistanceFactor`, `flySpeedFactor`), the `fly` block (`inertialMode`, `movement` speed/acceleration/damping, `rotation` speed/damping, `look.mouseSpeed`, `physics` thresholds and `dampingPower`), and the `orbit` block (`autoRotate.speed`, `zoom` min/max distance + speed, `damping` enabled flag + factor). Also `wheelZoomSensitivity` (default `1.0`): a global multiplier on every mouse-wheel zoom step (orbit/ortho dolly and fly forward/back) that the Settings popover's **Input > Zoom Sensitivity** slider mutates live through `config/user-settings.ts`. It is per-machine, not per-scene — some mouse drivers deliver oversized wheel deltas — and multiplies with the per-scene orbit `zoom.speed` rather than replacing it.
- `types.ts` — `ControlsConfig`, `FlyControlsConfig`, `OrbitControlsConfig`, `ScaleMultipliers`, plus the shared `ConfigRange` (`min`/`max`/`default`/optional `step`) helper interface used across UI-bound numeric settings.
- `validate.ts` — `validateControls(config, errors, warnings)`. Iterates the nine `ConfigRange` fields under `fly.*` and `orbit.*` (incl. `fly.look.mouseSpeed`); rejects non-finite `min`/`max`/`default`, enforces `min < max`, and enforces `min <= default <= max`.

## Public API

- `controlsConfig` — re-exported through `../../index.ts` into `AppConfig.controls`.
- `ControlsConfig`, `FlyControlsConfig`, `OrbitControlsConfig`, `ScaleMultipliers`, `ConfigRange` — re-exported through `../../types.ts`.
- `validateControls` — called from `../../validation.ts`.
