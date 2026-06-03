# rendering-controls

User-adjustable rendering settings — the single source of truth for camera FOV / clipping, bloom, global EOG (exposure / offset / gamma), anti-aliasing, tone mapping, vignette, physics-based detector noise, chromatic lens distortion, navigation mode, and adaptive DPR / cinematic toggles. Persisted per-scene via the rendering-controls persistence layer and overridable through the zarr `viewer_config` bridge.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports section validators invoked by the central dispatcher.

Note: `fov`, `near`, and `far` live here (not in `camera/`) because they are user-adjustable at runtime; the `camera/` slice owns only the camera knobs that stay constant across user sessions.

## Contents

- `data.ts` — `renderingControlsConfig: RenderingControlsConfig`. Defines `defaults` for every user-adjustable rendering setting. Notable defaults: `fov: 47` (`50mm Normal`), `near: 0.1`, `far: 1000`, `dynamicClippingEnabled: true`, `bloomEnabled: false`, `toneMapping: 'ACES'`, `controlType: 'orbit'`, and `naturalDrag: isMacPlatform()` (touchpad-friendly LEFT-rotate / RIGHT-pan on macOS; runtime persistence layer overrides this with the user's stored choice).
- `types.ts` — `RenderingSettings` (the full user-adjustable surface) and `RenderingControlsConfig` (`{ defaults: RenderingSettings }`). `RenderingSettings` also declares optional runtime-injected `fly*` fields populated from `config.controls.fly` to avoid duplication.
- `validate.ts` — two validators:
  - `validateRendering` rejects non-finite `exposure` / `globalOffset` / `globalGamma` and enforces `exposure ∈ [-5, 5]`, `globalOffset ∈ [-1, 1]`, `globalGamma ∈ [0.1, 10]`.
  - `validateBloomConsistency` rejects non-finite bloom values, enforces `bloomThreshold ∈ [0, 1]` and integer `bloomLevels ∈ [1, 12]`, and warns on `bloomStrength` / `bloomRadius` outside the typical 0-2 range (hard cap 0-10).

## Public API

- `renderingControlsConfig` — re-exported through `../../index.ts` into `AppConfig.renderingControls`.
- `RenderingSettings`, `RenderingControlsConfig` — re-exported through `../../types.ts`.
- `validateRendering`, `validateBloomConsistency` — called from `../../validation.ts`.
