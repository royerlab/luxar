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
   `ExrSequenceDriver` / `VideoModeDriver`).
6. Mount the modal overlay (focus trap + Escape to cancel + preview
   canvas + counter).
7. Enter the `try`, raise `isLoopRenderSuppressed`, call
   `driver.setup(ctx)`, then wake the rAF loop
   (`animationController.startAnimation()`) and register the
   `continuous` keep-alive callback. Only the driver's _construction_
   is pre-overlay; `setup` runs inside the `try` on purpose, so a
   throw from it is unwound by the `finally` that removes the overlay
   mounted in step 6. The suppression flag is raised _inside_ the
   `try` for the same reason, and it is the strictest case: it
   suppresses the loop's render globally, so escaping steps 3–6 with
   it stuck true would leave a dark viewport with no recovery but a
   reload.
8. For each frame: register a per-frame callback that orbits the
   camera one step, `await requestAnimationFrame`, then call
   `driver.captureFrame(ctx, frameIndex, progress)`. Tolerate up to
   `MAX_CONSECUTIVE_ERRORS = 3` consecutive frame failures before
   bailing.
9. Call `driver.finalize(ctx, capturedFrames, progress)`.
10. In `finally`: call `driver.abort?(ctx, reason)` if setup ran but
    finalize didn't succeed, remove per-frame callbacks, hide the
    indicator + overlay, restore auto-rotate + recording state, wake
    the loop once more, and clear the abort controller reference.

Step 7's wake-up is load-bearing, not belt-and-braces: the turntable's
rotation is applied from a per-frame callback, those only run while the
rAF loop is animating, and the loop idle-stops after ~2 s of no
interaction — the normal state once the user has read the panel and
confirmed the dialog. A `continuous` callback only KEEPS a running loop
alive; it never restarts a stopped one. Without the wake-up the capture
still emits N well-formed frames (the capture path renders its own
pipeline pass via `renderToImageData`, independently of the loop) —
they are simply all the same pose.

The loop runs, but its **own** render does not: `core/app/init/pipeline`
gives the animation controller a render-skip predicate keyed on
`RecordingPanel.isLoopRenderSuppressed()`. Every tick still updates the
controls and every per-frame callback — that is the whole reason the
loop has to run — but skips `postProcessing.render()`, whose output the
capture would discard anyway. It also removes a visible artifact: the
drivers' per-frame readback is asynchronous, so the loop interleaves
with it, and an EXR capture holds global raw-HDR shader flags across
that await —
a loop render landing inside the window painted a blown-out frame
through the translucent overlay, once per captured frame. The predicate
is offline-only: the real-time MediaRecorder path records the canvas the
loop paints, so suppressing its render there would yield an empty video.

What the user sees behind the scrim for the duration is a DARK viewport,
not a frozen frame: step 3's resolution scaling already resized the
render target (which clears the canvas) and nothing repaints it after
that. The overlay's preview canvas is the progress feedback — it shows
each captured frame, except on the EXR path, which never calls
`setPreview` and therefore shows the counter alone. Do not "fix" the
dark viewport by rendering into it mid-capture: that is the flicker
described above.

Step 10's second wake-up closes the tail: `restoreRecordingState()`
resizes the render target back (clearing the canvas) after the keep-alive
is gone, so without it a still-stopped loop — or one the idle timer halts
in the gap right after the resize — leaves the viewer blank until the
next mouse move. By then `isLoopRenderSuppressed` is false, so that frame
is a real render. The early-bail paths (disposed during the opening rAF
window, non-orbit controls) restore the same state and owe the same
repaint, so the non-orbit-controls bail wakes the loop too. No wake-up
ever fires on a disposed session — which is what makes the disposed bail
the exception, since a dispose is the only production way to reach it:
`runDisposePipeline` disposes the AnimationController before the
RecordingPanel, and panel dispose only ABORTS an in-flight capture, whose
finally resumes a tick later — restarting the loop there would render
against a disposed pipeline.

The `try { … } finally { … }` around steps 7–9 is load-bearing: a thrown
error anywhere in the capture loop must restore the DPR lock, resize
listener, panel visibility, overlay state, and recording flags. The
driver's own abort handler runs from the same finally so per-driver
resources (ZIP streams, mediabunny encoders) are torn down without
orphan files.

It does not reach back over steps 3–6 — the `finally` closes over
bindings those steps create — so a throw there still strands the
recording flags and the saved renderer state. That window is
synchronous DOM construction with no production-reachable throw (the
confirmation dialog in step 1 already assigns `innerHTML`, so an
environment that forbids it fails before any state is mutated), which
is why it is documented rather than guarded. The one flag whose stuck
value would be worse than a locked panel — `isLoopRenderSuppressed` —
is raised inside the `try` instead, per step 7.

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
