# Recording Drivers

Per-mode backends that plug into the offline-capture loop in
[`offline-capture-strategy.ts`](../offline-capture-strategy.ts). The
loop owns the shared scaffolding (state save/restore, overlay UI,
animation pump, progress display); each driver here owns its mode's
encoding pipeline.

## Files

| File                        | Role                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `offline-capture-driver.ts` | `OfflineCaptureDriver` interface + `CaptureContext` / `CaptureProgress` types         |
| `video-mode-driver.ts`      | MP4 / WebM / MKV encoder via [mediabunny](https://www.npmjs.com/package/mediabunny)   |
| `image-sequence-driver.ts`  | PNG / WebP / JPEG per-frame export, streamed into a ZIP                               |
| `exr-sequence-driver.ts`    | 16-bit half-float OpenEXR per-frame capture from the HDR post-processing buffer       |

## Driver lifecycle

All drivers implement the same four-method contract defined by
`OfflineCaptureDriver`:

```
                ┌─────────────────────────────────────────┐
                │   panel: runOfflineCaptureLoop()        │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │  setup(ctx)            → boolean        │  one-time: file picker, encoder
                ├─────────────────────────────────────────┤
                │  captureFrame(ctx, i, progress)  ×N     │  per-frame: read, encode, append
                ├─────────────────────────────────────────┤
                │  shouldAbort?()        → boolean        │  polled each iter
                ├─────────────────────────────────────────┤
                │  finalize(ctx, captured, progress)      │  flush + download (success path)
                │   ─ or ─                                │
                │  abort?(ctx, reason)                    │  partial cleanup (no artifact)
                └─────────────────────────────────────────┘
```

- `setup` returns `false` to abort the capture cleanly (e.g. no codec
  supports the requested resolution). Drivers that fail here must
  show the user an explanatory toast before returning.
- `captureFrame` may throw on transient encoder errors — the loop
  tolerates up to `MAX_CONSECUTIVE_ERRORS` before aborting. Its
  `frameIndex` is the OUTPUT sequence index (number of successful
  captures so far), so tolerated failures don't create timing gaps
  in the encoded stream.
- `shouldAbort` is polled between frames so a driver can signal
  "stop, continuing is pointless" (e.g. disk-write failure inside
  the ZIP writer) without throwing.
- `finalize` is the success path that delivers the artifact;
  `abort` is the failure / cancel path that releases encoder + file
  resources WITHOUT downloading or toasting "saved". The panel's
  `finally` block calls `abort` whenever `setup` succeeded but
  `finalize` did not run to completion.

## CaptureContext

The panel injects all driver dependencies through `CaptureContext`
so drivers stay decoupled from `RecordingPanel` internals:

| Field                  | Purpose                                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| `sceneManager`         | Access to renderer / postProcessing for pixel readback                   |
| `fps`                  | Frame rate (drives video timestamps + ffmpeg script)                     |
| `renderFrameToCanvas`  | Async readback of the live framebuffer into a 2D canvas                  |
| `generateFilename`     | Build a recording filename for a given extension                         |
| `generateFfmpegScript` | Build the bundled `encode_video.sh` for ZIP-based modes                  |
| `downloadBlob`         | Trigger a browser download                                               |
| `computeVideoBitrate`  | Compute the H.264/H.265/etc. bitrate for the canvas size                 |
| `showToast`            | Non-blocking toast notification                                          |
| `logWarning` / `logError` | Pre-tagged module loggers                                             |
| `imageQuality`         | `[0, 1]` quality for JPEG / WebP (ignored for PNG)                       |
| `videoCodec`           | User-selected codec preference                                           |
| `env`                  | Browser-environment shim (so tests can stub `showSaveFilePicker`)        |
| `signal`               | `AbortSignal` for the session — drivers MAY short-circuit on `aborted`   |

`CaptureProgress` exposes two UI hooks: `setLabel(text)` to update
the overlay status (e.g. "Packaging ZIP...") and `setPreview(canvas)`
to feed the live preview.

## Mode-specific notes

### `VideoModeDriver`

Uses [mediabunny](https://www.npmjs.com/package/mediabunny) instead
of `MediaRecorder.captureStream` because WebGL canvases with
`preserveDrawingBuffer: false` don't capture reliably via that path.
Each frame becomes a `VideoSample` with explicit `timestamp` +
`duration`; samples are `close()`'d in a `finally` so transient
encoder failures don't leak resources frame after frame.

Container selection: `Mp4OutputFormat` / `MkvOutputFormat` /
`WebMOutputFormat` based on `mode`. Codec selection itself is
delegated to [`../video-codec-selection.ts`](../video-codec-selection.ts),
which probes `canEncodeVideo` and may fall back from the user's
preferred codec with a warning toast.

mediabunny's `Output` has no public abort API: both `finalize` and
`abort` call `output.finalize()` — `abort` discards the resulting
buffer.

### `ImageSequenceDriver`

Reads each frame via `ctx.renderFrameToCanvas()` and encodes through
the canvas-native `canvas.toBlob(mime, quality)`. PNG is lossless
(quality ignored); WebP and JPEG honour `ctx.imageQuality`. Frames
are appended to a streaming
[`ZipSequenceCapture`](../zip-sequence-capture.ts); the zip's
internal success-counter handles file naming so tolerated frame
failures don't create gaps in the sequence.

### `ExrSequenceDriver`

Captures directly from the HDR post-processing buffer via
`sceneManager.postProcessing.captureHDRAsEXR()`, preserving 16-bit
half-float precision. Unlike the image and video drivers it does
NOT push pixels through a 2D canvas, so `progress.setPreview` is
not called — the panel keeps the previous frame visible during EXR
capture.

The driver also takes an `onFinalize` callback so the panel can
flip its `isEXRSequenceRecording` flag back to false from both the
success (`finalize`) and failure (`abort`) paths.

Both ZIP-based drivers (`ImageSequenceDriver`, `ExrSequenceDriver`)
implement `shouldAbort()` by delegating to `zip.hasDiskFailed()` —
once the ZIP writer fails a disk write, continuing the capture is
pointless.

## See Also

- [`../README.md`](../README.md) — Recording panel overview and full architecture
- [`../offline-capture-strategy.ts`](../offline-capture-strategy.ts) — The loop that drives these drivers
- [`../zip-sequence-capture.ts`](../zip-sequence-capture.ts) — Streaming ZIP writer used by image / EXR modes
- [`../video-codec-selection.ts`](../video-codec-selection.ts) — Codec auto-selection used by `VideoModeDriver`
