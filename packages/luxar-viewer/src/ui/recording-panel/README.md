# Recording

Video, image, and EXR-sequence export pipeline for the viewer. The
public entry point is [`RecordingPanel`](../recording-panel.ts); the
modules here are the strategies / session scaffolding / utilities
the panel delegates to. Per-mode offline encoders live one level
deeper, in [`drivers/`](./drivers/README.md).

## Architecture

```
RecordingPanel (UI + dispatch)
       │
       ▼
   RecordingSession           ── shared scaffolding (state save/restore,
       │                         mutex, confirmation dialog, REC
       │                         indicator, slider-sync)
       ▼
   CaptureStrategy  (one of)
       │
       ├─ ScreenshotStrategy       → single-frame PNG / WebP / JPEG / EXR
       ├─ VideoRecordingStrategy   → MediaRecorder real-time WebM capture
       └─ OfflineCaptureStrategy   → frame-by-frame offline loop
                                      │
                                      └─ delegates per-frame work to a
                                         driver in `drivers/`:
                                           • VideoModeDriver       (MP4/WebM/MKV)
                                           • ImageSequenceDriver   (PNG/WebP/JPEG)
                                           • ExrSequenceDriver     (HDR float)
```

`RecordingSession` is the **Session** half of a Strategy + Session
decomposition: every cross-cutting concern that all three capture
paths share — DPR save/restore, resize-lock, panel hide/restore,
auto-rotate pause/restore, the modal confirmation dialog, the REC
indicator widget, mutual-exclusion flags, slider-sync coordinator —
lives on Session. Each strategy owns its own capture pipeline plus a
reference to Session for the shared scaffolding.

## Files

| File                          | Role                                                                                                      |
| ----------------------------- | --------------------------------------------------------------------------------------------------------- |
| `session.ts`                  | `RecordingSession` — shared state save/restore, dialog, indicator, mutex                                  |
| `capture-strategy.ts`         | `CaptureStrategy` interface + `SessionState` view + `CaptureKind` union                                   |
| `screenshot-strategy.ts`      | `ScreenshotStrategy` — single-frame capture with optional transparent BG                                  |
| `video-recording-strategy.ts` | `VideoRecordingStrategy` — real-time MediaRecorder WebM capture                                           |
| `offline-capture-strategy.ts` | `OfflineCaptureStrategy` — frame-by-frame turntable / EXR loop, drives one driver                         |
| `screenshot-exporter.ts`      | `renderFrameToCanvas`, `encodeScreenshotBlob`, `normalizeScreenshotFormat`, `downloadBlob`                |
| `video-codec-selection.ts`    | `selectVideoCodec` — mediabunny codec fallback chain for the offline video path                           |
| `media-utilities.ts`          | `computeVideoBitrate`, `getSupportedMimeType`, `generateFilename`, `generateFfmpegScript`, `anchorOffset` |
| `overlay-compositor.ts`       | `compositeOverlays` + text / image / HTML overlay rasterization                                           |
| `animation-sync.ts`           | `SliderSyncCoordinator` + `getTurntableInfo` / `getNavigableDimensionOptions`                             |
| `gui-builder.ts`              | Pure mode→format and format→predicate visibility rules (`computeControlVisibility`)                       |
| `zip-sequence-capture.ts`     | `ZipSequenceCapture` — streaming ZIP writer for image / EXR sequences                                     |
| `types.ts`                    | Shared types: `RecordingMode`, `RecordingOptions`, `OutputFormat`, …                                      |

## Subpackages

- [drivers](./drivers/README.md) — Per-mode offline-capture backends
  (image sequence, EXR sequence, video). All implement the
  `OfflineCaptureDriver` interface and plug into
  `OfflineCaptureStrategy`.
- `ui/` — Internal helper for building the lil-gui controller tree
  (`gui-construction.ts`). Used only by `RecordingPanel`.

## Strategy + Session contract

Each `CaptureStrategy` implementation exposes the same tiny surface
(see `capture-strategy.ts`):

| Method                     | Purpose                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `kind`                     | `'screenshot'` \| `'video'` \| `'offline'` — used by the panel's dispatch                                           |
| `canRun(state)`            | Informational pre-check; the strategy still re-verifies inside `run`                                                |
| `run(opts, mode, session)` | The capture operation. Resolves when the capture finishes or aborts.                                                |
| `abort()`                  | Synchronous external stop. Video stops the MediaRecorder; offline fires its AbortController; screenshot is a no-op. |
| `dispose()`                | Panel-shutting-down signal. Drops long-lived state (tracks, AbortController, …).                                    |

Strategies are responsible for their own try/finally cleanup. They
read mutual-exclusion flags off `RecordingSession` (`isRecording`,
`isOfflineCaptureActive`) and call back into Session for the shared
unwind helpers (`restoreRecordingState`, `restoreAutoRotate`,
`restoreAllPanels`, …).

## Codec selection

Two codec paths, kept separate:

**Real-time video (`VideoRecordingStrategy`)** uses the browser's
`MediaRecorder`, which only supports WebM (VP9 / VP8).
`media-utilities.ts::getSupportedMimeType` picks the highest-quality
WebM mime type the platform supports, or returns `null` to disable
the option. Because the browser chooses the codec itself, the **Codec
dropdown is turntable-only** — it is hidden in Video mode (where it
would have no effect), and Video mode also hides the Format dropdown
since WebM is its only output.

**Offline video (`VideoModeDriver` in `drivers/`)** uses
[mediabunny](https://www.npmjs.com/package/mediabunny), which
supports VP9, AV1, AVC (H.264), HEVC (H.265), and VP8 across MP4 /
WebM / MKV containers. `video-codec-selection.ts::selectVideoCodec`
runs the fallback chain: map the user preference, downgrade to a
container-compatible codec if needed, then walk a fallback list
until `canEncodeVideo` reports support at the requested resolution.

## Offline-capture loop

`OfflineCaptureStrategy.runOfflineCaptureLoop` drives a deterministic
frame-by-frame capture for turntable + EXR-sequence modes:

1. Show the confirmation dialog; bail if cancelled.
2. Assign `sessionAbort` BEFORE any state mutation, so a `dispose()`
   during the early state-save / rAF window aborts cleanly.
3. Hide panels, save renderer state, disable DPR, lock resize,
   scale resolution to a 16-pixel-aligned multiple of the target.
4. Pause auto-rotate and compute per-frame angle for the turntable.
5. Build the per-mode driver (`ImageSequenceDriver` /
   `ExrSequenceDriver` / `VideoModeDriver`); call `driver.setup(ctx)`.
6. Mount the modal overlay (focus trap + Escape to cancel + preview
   canvas + counter).
7. For each frame: register a per-frame callback that orbits the
   camera one step, `await requestAnimationFrame`, then call
   `driver.captureFrame(ctx, frameIndex, progress)`. Tolerate up to
   `MAX_CONSECUTIVE_ERRORS = 3` consecutive frame failures before
   bailing.
8. Call `driver.finalize(ctx, capturedFrames, progress)`.
9. In `finally`: call `driver.abort?(ctx, reason)` if setup ran but
   finalize didn't succeed, remove per-frame callbacks, hide the
   indicator + overlay, restore auto-rotate + recording state, and
   clear the abort controller reference.

The `try { … } finally { … }` wrapping every state-mutating step is
load-bearing: a thrown error anywhere in the loop must restore the
DPR lock, resize listener, panel visibility, overlay state, and
recording flags. The driver's own abort handler runs from the same
finally so per-driver resources (ZIP streams, mediabunny encoders)
are torn down without orphan files.

## Screenshot path

`ScreenshotStrategy` is simpler: hide panels, save state, optionally
null the scene background for transparent output, capture via
`screenshot-exporter.ts::renderFrameToCanvas`, encode with
`encodeScreenshotBlob`, restore in `finally`. An `inProgress` flag
debounces the G keyboard shortcut and the Capture button so two
back-to-back captures don't race the save/restore-state pair.

For EXR screenshots, the strategy bypasses canvas-2D and calls
`postProcessing.captureHDRAsEXR()` directly to preserve the full
floating-point dynamic range.

## Public surface

External consumers (UI, embedders, tests) interact with
`RecordingPanel`. Embedders that want to drive a recording
programmatically should use `RecordingPanel.captureScreenshot()` /
`startVideoRecording()` / `stopVideoRecording()` rather than
instantiating a strategy or session directly.

`types.ts` re-exports the public option shapes (`RecordingMode`,
`RecordingOptions`, `OutputFormat`, `VideoCodecOption`,
`VideoResolution`, `VideoQuality`, `PanelStates`). `recording-panel.ts`
re-exports these so external callers have a single import surface.

## E2E coverage

`src/tests/e2e/recording-panel.spec.ts` exercises:

- Toggling the panel with the `T` key and closing it with `Escape`.
- Panel structure (mode selector, Advanced Options, Capture button).
- Triggering a screenshot via the `G` key and via the Capture button.
- Surfacing the confirmation dialog for a video recording.
- Showing the REC indicator during a video recording.
- The `Show Panels` toggle inside Advanced Options.
