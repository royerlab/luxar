# Rendering Controls Support Modules

> Helpers backing the `RenderingControls` facade in `../rendering-controls.ts` — settings persistence, live-state sync, cinematic-mode preset, focus management, optical math, and the per-category GUI builders under [`setup/`](./setup/README.md).

The parent class (`../rendering-controls.ts`, ~900 lines) owns construction, the GUI panel, and lifecycle. Everything else lives here as small focused modules that are individually unit-testable: pure helpers (no DOM) for math and settings I/O; small classes that own one piece of state (cinematic snapshot, RAF loop, document-level listener); and a `setup/` subpackage of builder functions that populate the GUI folders. Keeping each concern in its own file keeps `rendering-controls.ts` an orchestrator and lets the heavy machinery be tested without instantiating a SceneManager or a real GUI.

## File Structure

```
rendering-controls/
├── types.ts                   # SetupContext / SetupResult shared by the setup/ builders
├── apply-settings.ts          # applyRenderingSettings — push settings → live subsystems
├── sync-current-state.ts      # syncCurrentState — pull live state ← subsystems → GUI
├── settings-persistence.ts    # localStorage save/load + base & reset defaults builders
├── controls-utils.ts          # Pure: validate / clamp / serialize / merge / impact-score
├── cinematic-mode.ts          # CinematicModeController — film-look preset (snapshot/restore)
├── clipping-display.ts        # ClippingDisplay — RAF loop showing live near/far when dynamic
├── focus-manager.ts           # FocusManager — outside-click blur + canvas refocus
├── fov-utils.ts               # Pure: 35mm focal-length ↔ FOV conversions
└── setup/                     # Per-category GUI builders (see ./setup/README.md)
```

## Modules

### `types.ts`

Defines `SetupContext` and `SetupResult` — the shared call-shape used by every builder in `setup/`. `SetupContext` carries the `GUI`, the live `RenderingSettings`, the `PostProcessingManager`, the `SceneManager`, the optional `AnimationController`, and the `saveSettings` / `triggerAnimation` / `updateClippingControlsState` / `updateNavigationControls` callbacks. `SetupResult` returns `{ controllers, folders?, shadowObjects? }`; the parent class merges every builder's `controllers` map into one combined `RenderingControllers`.

### `apply-settings.ts`

`applyRenderingSettings(context)` pushes the current `RenderingSettings` into the post-processing pipeline and the scene manager. Used after any wholesale settings change (load from storage, apply zarr defaults, reset). No opinion about the source — only about how each field maps to a manager call. Side-effects: bloom, global EOG (exposure/offset/gamma), SSAA/FXAA/MSAA, tone mapping, detector noise (+ kicks the animation controller when enabled), vignette, chromatic lens distortion, dynamic clipping, and a final `triggerAnimation()`.

### `sync-current-state.ts`

`syncCurrentState(context)` is the inverse direction: pulls live state from the scene manager + camera + `ControlsManager` into `RenderingSettings` and refreshes every relevant `RenderingControllers` entry. Called whenever the panel becomes visible so the user sees what subsystems actually report rather than what they were last told to be. Snaps `fov` to a `config.camera.fovPresets` label when within 0.5°, mirrors fly-config from `ControlsManager` (persists across orbit↔fly switches), updates orbit `autoRotate`/`autoRotateSpeed`, and finishes with `gui.controllersRecursive().forEach(c => c.updateDisplay())` plus a cinematic-checkbox refresh and a navigation-folder visibility update.

### `settings-persistence.ts`

localStorage I/O and defaults building. No DOM, no manager calls — just structured merging:

| Export                                     | Purpose                                                                                                                                      |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `buildBaseDefaults()`                      | Hardcoded base defaults (`config.renderingControls.defaults` + `config.controls.fly.*` defaults). Returns `FullySpecifiedRenderingSettings`. |
| `buildResetDefaults(zarrViewerConfig?)`    | Base defaults overlaid with zarr `viewer_config` overrides (routed through `validateRenderingSettings` to clamp NaN/Infinity).               |
| `saveSettingsToStorage(sceneId, settings)` | Quota-safe write under `StorageKeys.rendering(sceneId)`.                                                                                     |
| `loadSettingsFromStorage(sceneId)`         | Quota-safe read; returns `{ stored, loaded }`.                                                                                               |
| `clearStoredSettings(sceneId)`             | Quota-safe `removeItem`.                                                                                                                     |

`FullySpecifiedRenderingSettings` narrows the otherwise-optional fly fields to non-optional concretes — useful for callers that don't want `| undefined` everywhere.

### `controls-utils.ts`

Pure utility functions extracted for testability — no external dependencies beyond `config`. Key exports:

- `validateRenderingSettings(partial)` — range-clamps every numeric, enum-checks `toneMapping` (`None`, `Linear`, `Reinhard`, `Cineon`, `ACES`, `AgX`, `Neutral`) and `controlType` (`orbit`, `fly`, `ortho`), and replaces non-boolean injections with defaults. Guards against NaN/Infinity in corrupted localStorage or malicious zarr config.
- `clampMSAASamples(value)` — snaps to a valid power-of-two MSAA count (0, 2, 4, 8, 16). Non-finite or non-numeric input → 0 (disabled); finite values are bucketed by `<=` comparisons.
- `serializeSettings` / `deserializeSettings` — JSON encode/decode with a null-on-parse-failure return.
- `mergeSettings(partial, defaults?)` — `{ ...defaults, ...partial }` then validate.
- `getSettingsRequiringRebuild(current, previous)` — returns the list of changed keys (`'msaa'`, `'ssaa'`, `'fxaa'`, `'toneMapping'`) that force a post-processing pipeline rebuild.
- `calculatePerformanceImpact(settings)` — heuristic 0–100 score for the AA/bloom/tone-mapping/lens-distortion/auto-rotate combo.
- `isValidColor(s)` — hex / rgb(a) / a short named-color allowlist.
- `getDefaultRenderingSettings()` — thin wrapper over `config.renderingControls.defaults` (re-exported `RenderingSettings` type for callers that prefer to import from this module).
- `settingsChanged(a, b)` — JSON-string equality (good enough for the panel's needs).

### `cinematic-mode.ts`

`CinematicModeController` owns the Cinematic-Mode preset (C-key toggle):

- **Enable** snapshots the affected `RenderingSettings` keys (tone mapping, detector noise, vignette, chromatic lens distortion, FOV, FOV preset) and applies cinematic values: ACES tone mapping, low detector noise (readout/photon/FPN), vignette on, the `35mm` lens-distortion preset, and 35 mm FOV.
- **Disable** restores from the snapshot, dirty-checked per key (a key is only restored if it still holds the cinematic value — user edits are preserved). If no snapshot exists (panel loaded with cinematic already on), falls back to `50mm Normal` defaults.
- **`updateCheckbox()`** uses a majority-vote over the four signal effects (`detectorNoiseEnabled`, `vignetteEnabled`, `chromaticLensDistortionEnabled`, `toneMapping === 'ACES'`) to drive the checkbox's display state.
- **`clearSnapshot()`** is called from `resetToDefaults`/`loadSettings` so the next toggle starts fresh.

The post-processing batch goes through `postProcessing.withDeferredRebuild(...)` so the depth counter unwinds even if a sub-setter throws. `TONE_MAPPING_MAP` (string → `THREE.ToneMapping` enum) is exported and reused by `apply-settings.ts`.

### `clipping-display.ts`

`ClippingDisplay` ties the dynamic-clipping toggle to the near/far sliders:

- When `dynamicEnabled === true` it sets `opacity: 0.5` and `pointer-events: none` on the slider containers and starts a throttled (100 ms) `requestAnimationFrame` loop that copies `camera.near` / `camera.far` into `settings.near` / `settings.far` and calls `controller.updateDisplay()` (without firing `onChange`).
- When `dynamicEnabled === false` it cancels the RAF and restores normal styling.
- `getNearPlane` / `getFarPlane` are passed as getters because the near/far controllers are created by `setup/camera-setup.ts` and may not exist when this controller is constructed.
- `dispose()` cancels the RAF; safe to call multiple times.

### `focus-manager.ts`

`FocusManager` keeps keyboard focus on the canvas when the user clicks outside the panel:

- **`onPanelShown()`** schedules a `setTimeout(100 ms)` that installs a capture-phase `mousedown` listener on `document`. The delay is so the click that opened the panel doesn't immediately trigger the handler.
- The handler blurs the active GUI element and refocuses the canvas when the click target is outside `context.panel`.
- **`onPanelHidden()`** blurs any active element, tears down the listener, and refocuses the canvas. Idempotent.
- **`dispose()`** is always sufficient to leave zero document-level listeners behind.

### `fov-utils.ts`

Pure 35mm-equivalent focal-length ↔ FOV conversions used by the camera section of the GUI:

```typescript
fovToFocalLength(fovDegrees: number): number   // sensorWidth / (2 · tan(FOV/2))
focalLengthToFov(focalLengthMm: number): number // 2 · atan(sensorWidth / (2 · f))
```

Reference sensor is the 36 mm-wide 35mm-film sensor (`FILM_35MM_SENSOR_WIDTH_MM`). `fovToFocalLength` rounds to the nearest integer mm since the GUI only displays whole-millimetre labels (e.g. `~50mm` next to the FOV slider when the preset is `Custom`).

## Subpackages

- [`setup/`](./setup/README.md) — Per-category GUI builders that populate the rendering-controls panel: `camera-setup`, `hdr-setup`, `anti-aliasing-setup`, `post-processing-setup`. Each consumes a `SetupContext` (from `./types.ts`) and returns a `SetupResult`. (`theme-setup` and `performance-setup` also live here but are now hosted in rail popovers — see [`../rail-panels/`](../rail-panels/README.md); navigation moved to `../rail-panels/navigation-popover.ts`.)

## Dependencies

- Internal: `../../config`, `../../config/zarr-bridge/viewer-config-utils`, `../../controls/types`, `../../rendering` (`PostProcessingManager`), `../../scene/scene-manager`, `../../scene/animation/animation-controller`, `../../types/zarr`, `../../utils/log`, `../../utils/storage-keys`, `../gui`.
- External: `three` (for the `THREE.ToneMapping` enum in `cinematic-mode.ts`).

## Related

- `../rendering-controls.ts` — `RenderingControls` facade that wires every module here into a single panel.
- [`./setup/README.md`](./setup/README.md) — The per-category builder functions.
