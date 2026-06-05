# Rendering Controls Setup Modules

> Per-category builder functions that populate the rendering-controls GUI panel.

The rendering-controls panel is assembled by composing one setup function per logical category (navigation, camera, HDR, anti-aliasing, post-processing, performance, theme). Each builder owns its folder, creates its controllers, wires `onChange` callbacks to the live subsystems (post-processing manager, scene manager, adaptive-DPR manager, theme manager), and returns the controller references the parent class needs for cross-module sync and persistence.

Splitting setup this way keeps each builder under a few hundred lines and keeps the parent `rendering-controls.ts` class focused on orchestration rather than UI wiring.

## File Structure

```
setup/
├── navigation-setup.ts        # 🕹️ Navigation (orbit / fly / ortho + sub-folders)
├── camera-setup.ts            # 🎥 Camera (FOV presets, FOV slider, clipping planes)
├── hdr-setup.ts               # ☀️ HDR (exposure, offset, gamma, tone mapping)
├── anti-aliasing-setup.ts     # ✨ Anti-Aliasing (SSAA, FXAA, MSAA)
├── post-processing-setup.ts   # 🎬 Post-Processing (bloom, detector noise, vignette, chromatic lens)
├── performance-setup.ts       # ⚡ Performance (adaptive DPR + manual DPR + FPS/DPR readouts)
└── theme-setup.ts             # 🎨 Theme (theme picker)
```

## Builder Contract

Most builders use the shared `SetupContext` / `SetupResult` interfaces from `../types.ts`:

```typescript
function setupXxxControls(
  context: SetupContext,
  controllersRef?: SetupResult['controllers'] // when cross-module sync is needed
): SetupResult;
```

`SetupContext` carries the GUI instance, current `RenderingSettings`, the post-processing and scene managers, the optional animation controller, and the `saveSettings`/`triggerAnimation`/`updateClippingControlsState`/`updateNavigationControls` callbacks. Each builder returns `{ controllers, folders?, shadowObjects? }`; the parent class merges all `controllers` into a single map.

Two builders use bespoke contexts because they own larger pieces of state:

- **`performance-setup.ts`** — `PerformanceSetupContext` / `PerformanceSetupResult`. Owns the read-only "Current DPR" / "Current FPS" display rows, a `setInterval` poll of the `AdaptiveDPRManager`, and exposes `updateVisibility` (re-applied after `loadSettings` flips the adaptive flag) and `cleanup` (clears the interval).
- **`theme-setup.ts`** — `ThemeSetupContext` only. Theme persistence lives inside `ThemeManager`, so the builder is fire-and-forget: no returned controllers.

## Builders

### `navigation-setup.ts`

Creates the **🕹️ Navigation** folder with:

- Control-type selector (`orbit` | `fly` | `ortho`) wired to `sceneManager.setControlType`.
- **Orbit Controls** sub-folder: auto-rotate toggle, rotation speed, `naturalDrag` toggle (defaults to ON on macOS for touchpad ergonomics).
- **Fly Controls** sub-folder: movement speed, rotation speed, inertial-mode toggle, translation damping, rotation damping. Damping sliders only show when inertial mode is on; movement-speed, rotation-speed, and translation-damping ranges are pulled from `config.controls.fly.*` (rotation damping is hardcoded `0.9 … 0.9999`).

Returns `folders.orbitFolder` and `folders.flyFolder` so the parent class can show/hide them from `updateNavigationControls(type)` when the user switches control type. In `ortho` mode both sub-folders are hidden (panning + zooming need no settings), and the parent also hides the FOV slider and preset since orthographic projection has no perspective.

### `camera-setup.ts`

Creates the **🎥 Camera** folder with:

- **FOV Preset** dropdown sourced from `config.camera.fovPresets` (28mm, 35mm, 50mm, 85mm, 135mm, Custom). Selecting a preset:
  1. Sets `settings.fov` and drives `sceneManager.updateFOV(delta)` (delta-based to share code with `Ctrl+Wheel`).
  2. When chromatic lens distortion is enabled, applies the matching `config.camera.lensDistortionPresets[preset]` (distortion X/Y, dispersion, principal point, focal length, skew) and refreshes the corresponding controllers via `controllersRef`.
- **Field of View** slider — driving the FOV directly flips the preset dropdown to `Custom` and rewrites its option label to `~{focalLength}mm` (via `fovToFocalLength` from `../fov-utils`).
- **Clipping Planes** sub-folder: `near`, `far`, and a `Dynamic Clipping` toggle. Manual sliders validate `near < far` (logging via `log.warning(Modules.RENDERER, ...)`) and call `sceneManager.updateClippingPlanes`. The toggle calls `sceneManager.setDynamicClipping` and notifies the parent class through `updateClippingControlsState(enabled)` so it can grey out manual sliders.

Takes `controllersRef` because the FOV preset writes back into the lens-distortion controllers owned by `post-processing-setup.ts`.

### `hdr-setup.ts`

Creates the **☀️ HDR** folder with global Exposure–Offset–Gamma (EOG) plus the tone-mapping selector. The four controls map to scene-manager calls and one post-processing call:

| Control      | Range          | Wired to                             |
| ------------ | -------------- | ------------------------------------ |
| Exposure     | -5 to +5 stops | `sceneManager.updateExposure(v)`     |
| Offset       | -1.0 to 1.0    | `sceneManager.updateGlobalOffset(v)` |
| Gamma        | 0.1 to 10.0    | `sceneManager.updateGlobalGamma(v)`  |
| Tone Mapping | seven options  | `postProcessing.setToneMapping(...)` |

The tone-mapping dropdown maps display names (`None`, `Linear`, `Reinhard`, `Cineon`, `ACES`, `AgX`, `Neutral`) to the matching `THREE.*ToneMapping` enum values. EOG is applied inside the mega-shader fragment immediately before the tone-mapping operator.

### `anti-aliasing-setup.ts`

Creates the **✨ Anti-Aliasing** folder (collapsed by default) covering all three AA paths the post-processing pipeline supports:

- **SSAA Enabled** + **SSAA Settings** sub-folder (multiplier 1.5×/2×/3×/4×).
- **FXAA Enabled** (single toggle, no sub-folder).
- **MSAA Enabled** + **MSAA Settings** sub-folder (sample count 2/4/8).

The SSAA and MSAA settings sub-folders are shown/hidden in lock-step with their enable toggle so unused settings stay out of the way.

### `post-processing-setup.ts`

Creates the **🎬 Post-Processing** folder (collapsed by default) with one sub-folder per effect:

- **Bloom** — `enabled`, `threshold`, `strength`, `radius`, `Mipmap Levels` (1–12, default 8). Toggling enabled re-runs `postProcessing.setBloomEnabled(value, strength, radius, threshold)`; sliders call `postProcessing.updateBloomSettings` and `postProcessing.setBloomLevels`.
- **Detector Noise** — physics-based, three independent sources: `Shot Noise` (Poisson photon gain), `Readout Noise` (per-frame Gaussian σ), `Fixed Pattern` (static per-pixel σ). The enable toggle starts the animation controller because temporal noise needs continuous frames.
- **Vignette** — `enabled`, `darkness`, `offset`. Sliders only emit when the effect is enabled.
- **Chromatic Lens Distortion** — combined barrel/pincushion + chromatic aberration via per-channel sampling at distorted UVs. Exposes `dispersion`, radial `distortionX`/`distortionY`, `principalPoint{X,Y}`, `focalLength{X,Y}`, and `skew`. Each control calls a single `postProcessing.updateChromaticLensDistortion({ field: value })`; the enable toggle uses `setChromaticLensDistortionEnabled(...)` with the full nine-argument signature.

Takes `_controllersRef` (currently underscored — accepted for symmetry with `camera-setup` but unused inside the builder). Stores all eight chromatic-lens controller references on the returned `controllers` map so `camera-setup.ts` can mutate them when a FOV preset is chosen.

### `performance-setup.ts`

Creates the **⚡ Performance** folder (collapsed by default) with:

- **Adaptive Resolution** toggle (`AdaptiveDPRManager.setEnabled`). When ON, manual DPR is hidden and two read-only rows show live "Current DPR" and "Current FPS" values polled every 500 ms.
- **Manual DPR** slider (range `0.25` … `getNativeDPR()`, step 0.05) using `onFinishChange` to avoid GPU resize thrash. Only applies when adaptive is OFF.
- Two hand-built display rows (`Current DPR`, `Current FPS`) injected into the folder's `.luxar-gui__children` container — these are not GUI controllers because they are read-only.

Returns:

- `adaptiveDPREnabled` — the toggle controller, so the persistence layer can re-bind it.
- `updateVisibility(adaptiveEnabled)` — re-applied by the parent class after `loadSettings()` mutates the stored flag, so the panel reflects the freshly loaded state.
- `cleanup()` — clears the `setInterval` that drives the live readouts.

### `theme-setup.ts`

Creates the **🎨 Theme** folder with a single **Active Theme** dropdown bound to `ThemeManager.getInstance()`. Themes are enumerated via `themeManager.getAllThemes()` and rendered as `name → id`. `ThemeManager` owns persistence, so the builder only needs to call `themeManager.setTheme(id)` and trigger a re-render. Returns nothing.

## Cross-Module Wiring

The parent class (`../rendering-controls.ts`) calls these builders during construction and stitches together the few inter-module dependencies:

- **FOV preset ↔ chromatic lens distortion** — `setupCameraControls(context, controllers)` is called _after_ `setupPostProcessingControls(context, controllers)` so the controller map already contains the chromatic-lens entries that the FOV preset's `onChange` will mutate. Both builders take the same `controllers` object by reference; order of insertion in the parent class is what makes the link work.
- **Navigation folder visibility** — `setupNavigationControls` returns `orbitFolder` / `flyFolder`; the parent's `updateNavigationControls(type)` toggles them when the control type changes.
- **Clipping controls enable state** — `setupCameraControls` calls back into the parent's `updateClippingControlsState(enabled)` so the parent can disable the near/far sliders when dynamic clipping is on.
- **Adaptive-DPR persistence** — after `loadSettings()` reads the stored `adaptiveDPREnabled` flag, the parent re-applies `performanceSetup.updateVisibility(flag)` so the panel matches the loaded state.

## Conventions

- **Every folder and controller gets a `title` attribute** via `domElement.setAttribute('title', ...)`. These tooltips are the only in-panel help, so they are written as multi-line strings with bullet points and ranges.
- **`saveSettings()` after every change** so refreshes restore the panel state. The persistence layer (`../settings-persistence.ts`) owns the actual storage.
- **`triggerAnimation()` after every change** that affects a rendered pixel so the on-demand renderer redraws.
- **Logging uses `log.{info,warning}(Modules.RENDERER, msg)`** from `../../../utils/log`, not `console.*`.

## Related

- [`../README.md`](../README.md) — Parent rendering-controls package overview.
- [`../types.ts`](../types.ts) — `SetupContext`, `SetupResult`, `RenderingControllers` interfaces.
- [`../fov-utils.ts`](../fov-utils.ts) — `fovToFocalLength` used by the camera builder.
- [`../../../config`](../../../config) — Source of `config.camera.fovPresets`, `config.camera.lensDistortionPresets`, and `config.controls.fly.*` ranges.
