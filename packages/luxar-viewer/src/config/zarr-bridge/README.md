# zarr-bridge

Bridge between the viewer's TypeScript configuration (camelCase `RenderingSettings`, `CameraConfig`, scene state) and the zarr `viewer_config` JSON blob (snake_case, written by Python). Provides the round-trip used for `Python → zarr → viewer → export → Python`.

Unlike the `sections/<name>/` slices, this folder does not contribute a section to `AppConfig` — it converts between zarr-side and viewer-side representations and captures live viewer state for clipboard export (`Ctrl+Shift+S`).

## Contents

- `viewer-config-utils.ts` — Direction-conversion utilities and key mappings.
  - `RENDERING_SETTINGS_MAP` — snake_case zarr key → camelCase `RenderingSettings` key, covering tone mapping, EOG (exposure / global offset / global gamma), bloom, navigation, cinematic mode, vignette, detector noise, anti-aliasing (FXAA/MSAA/SSAA), chromatic lens distortion, fly controls, dynamic clipping, and adaptive DPR.
  - `REVERSE_SETTINGS_MAP` — built at module load by inverting `RENDERING_SETTINGS_MAP`.
  - `extractRenderingOverrides(zarrConfig)` — returns a `Partial<RenderingSettings>` containing only fields set in zarr. Also pulls `camera.fov`, `camera.fov_preset`, `camera.near`, and `camera.far` because those properties live on `RenderingSettings` even though zarr stores them under `camera`.
  - `renderingSettingsToZarr(settings)` — inverse mapping. Camera-related fields (`fov`, `fovPreset`, `near`, `far`) are intentionally not emitted here — `captureViewerState` writes them under `camera.*`. Iterates the input keys (not `REVERSE_SETTINGS_MAP`) so a `RenderingSettings` field missing from the bridge map is still dropped (round-trip safety with older zarr files) but logs a single warning per unknown key via the module-local `_warnedUnknownRenderingKeys` set (exported so tests can reset it).
  - `CameraOverrides` interface — spatial state applied to the camera on every scene load: `position`, `target`, `up`, `targetNode` (a named node whose bounding-box center becomes the camera target). Not part of `RenderingSettings`.
  - `extractCameraOverrides(zarrConfig)` — pulls `camera.position`/`target`/`up`/`target_node` from the zarr blob.
  - `extractBackgroundColor(zarrConfig)` — returns the hex `background_color` string or `undefined`.

- `viewer-state-capture.ts` — `captureViewerState(sceneManager, renderingControls, sceneDimsManager, animationManager?, themeManager?)`. Snapshots the live viewer as a `ZarrViewerConfig` JSON object compatible with Python's `ViewerConfig.from_json()`. Emits `camera` (position/target/up + fov/fov_preset, plus `near`/`far` **only when dynamic clipping is off** — while it is on those two settings hold transient live-camera readouts, not authored values, so capturing them would export a zoomed-in pose's planes as if they had been chosen), `background_color` (read from `scene.background` via a duck-typed `isColor`/`getHexString` narrowing to avoid importing Three's full `Color` type), snake_case rendering settings via `REVERSE_SETTINGS_MAP`, the current theme id (falls back silently if `ThemeManager` is not initialized — used in tests), `dimensions.current_step`, and a per-dimension `animation` array (only included when at least one dimension has a recorded animation state).

## Public API

- `RENDERING_SETTINGS_MAP`, `REVERSE_SETTINGS_MAP` — key tables.
- `extractRenderingOverrides`, `renderingSettingsToZarr` — settings conversion.
- `_warnedUnknownRenderingKeys` — module-local warned-key set, exported for tests.
- `CameraOverrides`, `extractCameraOverrides`, `extractBackgroundColor` — scene-state extraction.
- `captureViewerState` — live snapshot for clipboard export.

## Consumers

- `ui/rendering-controls.ts` and `ui/rendering-controls/settings-persistence.ts` call `extractRenderingOverrides` to seed user settings from zarr.
- `scene/scene-manager.ts` and `scene/scene-manager/camera/camera-setup.ts` call `extractCameraOverrides` to position the camera on scene load.
- `input/input-handler/commands/viewer-state-export.ts` calls `captureViewerState` (wired to `Ctrl+Shift+S` in `input/input-handler/key-bindings/navigation-bindings.ts`) to copy a `ZarrViewerConfig` to the clipboard.
- Unit tests in `src/tests/unit/config/viewer-config-utils.test.ts` and `src/tests/unit/config/viewer-state-capture.test.ts`.

## See Also

- [../README.md](../README.md) — parent config package overview.
- [../../types/zarr.ts](../../types/zarr.ts) — `ZarrViewerConfig` type definition (zarr-side schema).
