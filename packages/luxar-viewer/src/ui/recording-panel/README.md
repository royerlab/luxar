# Recording

Video, image, and EXR-sequence export pipeline for the viewer. The
public entry point is [`RecordingPanel`](../recording-panel.ts); the
modules here are the drivers / utilities that the panel delegates to.

## Architecture

```
RecordingPanel (UI + state)
       │
       ▼
   RecordingDriver  (one of)
       │
       ├─ video-mode-driver         → MediaRecorder real-time capture
       ├─ offline-capture-driver    → frame-by-frame render + encode
       ├─ image-sequence-driver     → PNG / JPEG per frame
       ├─ exr-sequence-driver       → 16-bit float HDR per frame
       └─ zip-sequence-capture      → zip of images for sequence export
```

Every driver implements a small state machine: `idle → recording →
finalizing → done`, with `dispose()` callable at any point to abort
cleanly.

## Files

| File                        | Role                                                                         |
| --------------------------- | ---------------------------------------------------------------------------- |
| `video-mode-driver.ts`      | Real-time MediaRecorder capture (MP4 / WebM / MKV)                           |
| `offline-capture-driver.ts` | Frame-by-frame deterministic capture; locks DPR + resize for reproducibility |
| `image-sequence-driver.ts`  | PNG/JPEG per-frame export                                                    |
| `exr-sequence-driver.ts`    | HDR float capture per frame                                                  |
| `zip-sequence-capture.ts`   | Bundles a sequence into a downloadable zip                                   |
| `screenshot-exporter.ts`    | Single-frame PNG export                                                      |
| `video-codec-selection.ts`  | Picks the best supported MediaRecorder mimeType per browser                  |
| `media-utilities.ts`        | Shared `Blob` / `ArrayBuffer` plumbing                                       |
| `overlay-compositor.ts`     | Renders the recording overlay onto the captured frame                        |
| `animation-sync.ts`         | Drives `AnimationController` deterministically during offline capture        |
| `gui-builder.ts`            | Pure mode→format and format→predicate visibility rules                       |
| `ui/gui-construction.ts`    | Builds the lil-gui controller tree for the recording panel                   |
| `types.ts`                  | Shared recording types                                                       |

## Codec selection

`video-codec-selection.ts` queries `MediaRecorder.isTypeSupported(...)`
for an ordered list of preferences:

1. MP4 (H.264) — universal playback, smallest files
2. WebM (VP9) — fallback when MP4 unavailable (most Firefox, some Linux Chrome)
3. WebM (VP8) — legacy fallback
4. MKV (H.264) — Chromium-only with `chromiumExperimental` mime

The first supported entry wins. Browsers that support none get a
disabled video-mode option with a tooltip explaining the limitation.

## Offline-capture loop

`offline-capture-driver.ts` drives a deterministic frame-by-frame
capture:

1. Lock DPR (disable adaptive-DPR; pin to the configured value).
2. Lock canvas size (disable resize listener).
3. Show the recording overlay if enabled.
4. For each frame:
   - Set `AnimationController` to the next frame's timestamp.
   - Wait for `renderer.render` to complete + composer + post-fx.
   - Read back the framebuffer (via `hdr-capture` or RGBA8 readback).
   - Encode through the selected sink (video / image / EXR).
5. Finalize the sink (flush MediaRecorder, zip the sequence, etc.).
6. Restore DPR + resize listener + overlay state.

Steps 1, 2, and 6 are non-trivial: a thrown error anywhere in the
loop must restore the locks. The driver wraps the whole loop in a
`try / finally` so the user doesn't end up with a permanently locked
DPR or hidden overlay after a partial capture.

## Offscreen-canvas teardown

EXR + image sequence drivers use offscreen canvases for compositing.
Each driver's `dispose()` runs `ctx.clearRect(0, 0, w, h)` on its
offscreen canvas (defensive) and then drops the reference. The
offscreen canvases are garbage-collected once the driver is gone.

## Public surface

External consumers (UI, embedders) interact with `RecordingPanel`,
not the drivers directly. Embedders that want to drive a recording
programmatically should use `RecordingPanel.startRecording(opts)` /
`stopRecording()` rather than instantiating a driver.

## E2E coverage

`tests/e2e/recording-panel.spec.ts` exercises:

- Codec auto-selection (verifies the panel doesn't crash on a
  browser without MP4 support).
- Video record + stop + download for a short scene.
- Image sequence export.
- Overlay compositing on / off.
