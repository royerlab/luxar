# camera

Camera configuration slice. Owns defaults for 3D perspective navigation: initial position, FOV zoom limits and sensitivity, and the photography-style FOV / lens-distortion preset tables.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher.

Note: `fov`, `near`, and `far` are intentionally **not** in this section — they live in `renderingControls.defaults` as the single source of truth. This slice only owns camera knobs that are independent of the user-adjustable rendering settings.

## Contents

- `data.ts` — `cameraConfig: CameraConfig`. Defines `initialPosition`, `fovMin`/`fovMax`/`fovSensitivity`, and the `fovPresets` / `lensDistortionPresets` tables keyed by 35mm-equivalent focal-length names (`28mm Wide`, `35mm`, `50mm Normal`, `85mm Portrait`, `135mm Tele`, plus `Custom` for FOV).
- `types.ts` — `CameraConfig` interface, including the per-preset lens-distortion shape (`distortionX/Y`, `principalPointX/Y`, `focalLengthX/Y`, `skew`, `dispersion`).
- `validate.ts` — `validateCamera(config, errors, warnings)`. Reads `fov`/`near`/`far` from `renderingControls.defaults`; rejects non-finite values, enforces `1 ≤ fov ≤ 180`, `near > 0`, `far > near`, and `fovMin < fovMax`; warns on `fovSensitivity` outside `(0, 1]`.

## Public API

- `cameraConfig` — re-exported through `../../index.ts` into `AppConfig.camera`.
- `CameraConfig` — re-exported through `../../types.ts`.
- `validateCamera` — called from `../../validation.ts`.
