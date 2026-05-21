# Recording Panel GUI Construction

DOM-building helper extracted from `RecordingPanel.buildGUI`. Owns
the lil-gui controller tree the panel displays — mode selector,
Advanced Options folder, and the prominent Capture button at the
bottom — but knows nothing about the orchestrator: every state
mutation is expressed as a setter callback on the injected `deps`.

## Files

| File                  | Role                                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| `gui-construction.ts` | `buildRecordingGUI(deps)` + `BuildGUIDeps` / `BuildGUIResult` interfaces. Single export, no side state. |

## Contract

`buildRecordingGUI(deps: BuildGUIDeps): BuildGUIResult`

`BuildGUIDeps` carries the target `GUI` instance, the panel's
mutable `RecordingOptions`, the initial `RecordingMode`, and a small
set of callbacks (`setMode`, `updateControlVisibility`,
`getTurntableInfo`, `getNavigableDimensionOptions`,
`captureScreenshot`, `startVideoRecording`). The function builds the
panel top-to-bottom — mode toggle, `Advanced Options` folder (closed
by default) containing general / image / video / turntable rows, and
the Capture button — and returns the allocated `Controller`
references the panel needs at runtime:

- Per-control references (`formatController`, `qualityController`,
  `transparentController`, `videoCodecController`,
  `videoQualityController`, `videoDurationController`,
  `syncToggleController`, `syncDimensionController`) used by
  `RecordingPanel.updateControlVisibility` to enable/disable
  individual rows.
- Three controller arrays (`imageControllers`, `videoControllers`,
  `turntableControllers`) used by the panel to bulk show/hide
  whole control groups when the mode changes.

The visibility rules themselves (which controllers belong to which
mode + format) live in
[`../gui-builder.ts::computeControlVisibility`](../gui-builder.ts);
this builder only allocates the controllers.

## Conventions

- The `mode` toggle is captured in a local `modeObj` closure so the
  Capture button's callback always dispatches on the current mode,
  even if the user switches mode after the panel is built.
- Most controllers get a `title` attribute via
  `domElement.closest('.luxar-gui__controller')?.setAttribute('title', ...)`
  — these tooltips are the only in-panel help.
- The turntable Output row is a read-only computed display: the
  builder flips `readOnly = true` on the underlying `<input>` and
  re-runs `updateTurntableInfo()` from the speed and FPS `onChange`
  handlers.
- The Capture button gets the `luxar-recording-btn` class so
  CSS can style it as the prominent action.

## See Also

- [`../README.md`](../README.md) — Recording panel overview.
- [`../../recording-panel.ts`](../../recording-panel.ts) — The only
  caller; passes the returned controllers to its visibility +
  capture-dispatch logic.
- [`../gui-builder.ts`](../gui-builder.ts) — Pure visibility rules
  consumed by the panel after this builder has run.
