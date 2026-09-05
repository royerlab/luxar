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
auto-rotate / auto-dolly pause/restore, the modal confirmation dialog, the REC
indicator widget, mutual-exclusion flags, slider-sync coordinator —
lives on Session. Each strategy owns its own capture pipeline plus a
reference to Session for the shared scaffolding.

## Files

| File                            | Role                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| `session.ts`                    | `RecordingSession` — shared state save/restore, dialog, indicator, mutex                   |
| `capture-strategy.ts`           | `CaptureStrategy` interface + `SessionState` view + `CaptureKind` union                    |
| `screenshot-strategy.ts`        | `ScreenshotStrategy` — single-frame capture with optional transparent BG                   |
| `video-recording-strategy.ts`   | `VideoRecordingStrategy` — real-time MediaRecorder WebM capture                            |
| `offline-capture-strategy.ts`   | `OfflineCaptureStrategy` — sequences the six collaborators below into one capture          |
| `offline-capture-preflight.ts`  | Confirmation dialog, session ownership, capture resolution, turntable rotation/dolly plan  |
| `offline-capture-context.ts`    | `createCaptureDriver` (mode → driver) + `buildCaptureContext` (the driver's dependencies)  |
| `offline-capture-overlay.ts`    | The modal progress overlay: ARIA, focus trap, Escape/Cancel, live preview, frame counter   |
| `offline-capture-frame-loop.ts` | The per-frame loop: orbit → settle → grab, with the consecutive-failure bail               |
| `offline-lod-settle.ts`         | `LodSettleDrain` — the per-frame LOD quiescence wait, its latch/re-arm, and its report     |
| `offline-capture-teardown.ts`   | The one safe teardown order (flags, driver abort, callbacks, overlay, state restore)       |
| `screenshot-exporter.ts`        | `renderFrameToCanvas`, `encodeScreenshotBlob`, `normalizeScreenshotFormat`, `downloadBlob` |
| `video-codec-selection.ts`      | `selectVideoCodec` — mediabunny codec fallback chain for the offline video path            |
| `media-utilities.ts`            | `computeVideoBitrate`, `getSupportedMimeType`, `generateFilename`, `anchorOffset`          |
| `ffmpeg-script.ts`              | `generateFfmpegScript` — the bundled `encode_video.sh`, incl. the EXR display transform    |
| `overlay-compositor.ts`         | `compositeOverlays` + text / image / HTML overlay rasterization                            |
| `live-overlay-compositor.ts`    | `LiveOverlayCompositor` — mirror canvas that puts overlays into REAL-TIME WebM capture     |
| `animation-sync.ts`             | `SliderSyncCoordinator` + `getTurntableInfo` / `getNavigableDimensionOptions`              |
| `gui-builder.ts`                | Pure mode→format and format→predicate visibility rules (`computeControlVisibility`)        |
| `zip-sequence-capture.ts`       | `ZipSequenceCapture` — streaming ZIP writer for image / EXR sequences                      |
| `types.ts`                      | Shared types: `RecordingMode`, `RecordingOptions`, `OutputFormat`, …                       |

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
3. Hide panels, save renderer state, disable DPR, lock resize, scale
   resolution to the target height — "Native" being the DISPLAY size
   (`PostProcessingManager.getDisplaySize()`, _not_ `renderer.getSize()`,
   which reports the SSAA-multiplied size) times the native DPR. The
   height is aligned _down_ until the PHYSICAL frame, i.e. after the SSAA
   multiplier, is even, since that is the size the frames are written at
   and H.264/H.265 with yuv420p reject an odd dimension; the width is
   then derived from the aligned height and aligned the same way. The
   walk is best-effort: it tries eleven candidates (ten pixels of walk),
   which covers every multiplier measured except those within 0.1 of an
   even one but not on it — at 2.001 and the like the product's parity is
   locked across a thousand consecutive heights, so no bounded walk helps
   and the even floor is taken instead. Exactly 2 and 4 never need a
   step; 1 and 3 need at most one.
   The turntable's speed row is shown as **`Turn Duration (s)`** but stored as
   `RecordingOptions.turntableSpeed` in **degrees per second** — the unit the
   capture strategies and saved presets already use, converted at the GUI edge by
   `turnSecondsFromDegPerSec` / `degPerSecFromTurnSeconds` in `animation-sync.ts`.
   Same split, and same reason, as the Navigation popover's rotation period: every
   timing control in the viewer answers "how long does it take?" in seconds, while
   what is written down keeps meaning what it always meant.

4. Pause auto-rotate AND the auto-dolly, then compute the per-frame angle for
   the turntable. The rotation itself goes through
   `LuxarOrbitControls.applyOrbitRotation(angle)` with no explicit axis, which
   defaults to the configured `autoRotateAxis` — so a turntable recorded from a
   non-default-axis preview rotates the same way in the file.

   The dolly is baked the same way, through `applyOrbitDolly(phase)` at the
   configured amplitude, and for the same reason: an export should breathe like
   its preview. Two details make it a recording rather than a replay. The phase
   comes from the FRAME INDEX (offline) or the wall-clock `progress` fraction
   (live), never from `deltaTime` — offline frames wait on the LOD settle drain
   below, so their real duration says nothing about playback time. And the
   number of in-and-out cycles is ROUNDED to a whole number over the turn
   (`max(1, round(turnSeconds / period))`), so the clip loops: the last frame
   stops one step short of closing the cycle, exactly as the rotation does.

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
   camera one step, `await requestAnimationFrame`, remove the orbit
   callback, **drain on `hooks.isLODSettled`** (below), then call
   `driver.captureFrame(ctx, frameIndex, progress)`. Tolerate up to
   `MAX_CONSECUTIVE_ERRORS = 3` consecutive frame failures before
   bailing.
9. Call `driver.finalize(ctx, capturedFrames, progress)`.
10. In `finally`: call `driver.abort?(ctx, reason)` if setup ran but
    finalize didn't succeed, remove per-frame callbacks, hide the
    indicator + overlay, restore auto-rotate + auto-dolly + recording state, wake
    the loop once more, and clear the abort controller reference.

### Step 8's LOD settle drain

Because the rAF loop runs for the whole capture (that is what makes the
turntable rotate at all), the auto-LOD selector is live for the entire
sweep — and it is frustum-aware. On a `tiles` / `adaptive` / `overview`
gsplat scene, a tile whose world bbox leaves the frustum mid-orbit is
demoted to its coarsest ready level, and the resident-byte budget may
release the fine one. When it swings back into view the fine level
reloads **asynchronously**, so a loop that takes exactly one rAF per
exported frame writes those frames at the coarse level and pops back a
few frames later (#1695).

So each frame, after the orbit callback has been removed, the loop spends
extra rAF ticks until `hooks.isLODSettled()` reports every in-frame LOD
group is showing its selected level at final quality
(`LODGroupRegistry.isCaptureQuiescent()`, injected from
`core/app/init/pipeline` via `RecordingPanel.setLODSettledProvider`). The
drain sits **after** the callback removal on purpose: the camera pose is
fixed from that point on, so the extra frames only let pending loads
land. Draining before it would keep advancing the turntable while waiting
and smear the sweep.

The hook is **tri-state** — `true` / `false` / `null` — and the third
state is what keeps scenes without LOD free. The pipeline wires the
provider unconditionally, so "a hook is present" says nothing about
whether this scene has anything to wait for; `null` says there is no
`lod_group` to wait for (no scene loader, no registry, or a registry with
zero registered lod\_groups) and makes the loop skip the drain
**entirely, including the mandatory catch-up rAF below**. A `true` does
not: it means the LOD tree exists and is settled, which the loop can only
know one tick late. An absent hook is identical to `null`, which is what
the unit tests that don't supply it get. The hook is also called through a `try`/`catch`
(the same treatment `AnimationController.pacingSuspended()` gives its
injected predicate) — a throw degrades to `null`, i.e. "do not wait",
rather than being caught by the loop's outer handler and discarding the
whole sequence as "Recording failed".

The **first** of those ticks is mandatory, not part of the wait, and the
reason is per-frame callback ordering. `AnimationController.animate` runs
`controls.update()`, then every per-frame callback in Map insertion
order, then the render. The LOD selector (`lod-group-selector`) is
registered once at pipeline init, while this loop removes and re-adds its
orbit callback on every iteration — so the orbit callback is always
_last_ in that Map, and `LuxarOrbitControls.applyOrbitRotation` moves the
camera synchronously. Inside the single rAF the loop awaits for frame N,
therefore: the selector evaluates pose N−1, and only afterwards does the
camera advance to pose N. Polling immediately would read
`desiredChildIndex` / `activeChildIndex` / `offScreen` computed for the
_previous_ pose, and on the exact frame a tile re-enters the frustum the
selector has not seen the re-entry yet — the predicate would report
settled and that frame would still be filmed coarse. That is one popped
frame per re-entry, i.e. a slice of the very bug being fixed. One extra
tick puts the selector on the pose about to be captured.

The cost is one rAF per frame even on a fully settled scene — one that
has LOD groups, that is; a scene whose provider answers `null` pays
nothing. Against a full pipeline render plus an asynchronous GPU readback
plus an encode for every frame, ~16 ms is noise.

The wait is bounded by `LOD_SETTLE_TIMEOUT_MS = 2000` **and**
`LOD_SETTLE_MAX_FRAMES = 120` (≈2 s at 60 fps, inclusive of the mandatory
catch-up tick), whichever trips first. The deadline is read from
`performance.now()`, which is monotonic — `Date.now()` can step backwards
under NTP/DST and take the deadline with it. The frame cap is not
redundant with the deadline: under a stubbed clock (unit tests, fake
timers) `performance.now()` never advances and the ms bound alone would
spin forever. After `MAX_CONSECUTIVE_LOD_TIMEOUTS = 3` consecutive
timeouts the loop PAUSES draining and warns once; otherwise a scene that
can never settle (resident-byte thrash on an over-budget partition) would
multiply the capture's wall-clock by the timeout on every remaining
frame. A successful settle resets the counter.

That pause **re-arms**, and it has to. A latched frame still spends a
free, non-waiting probe of the predicate — no rAF, no poll, so it costs
exactly what an undrained frame cost before — and clears the latch as
soon as that probe reports settled; the next frame then gets the full,
correctly-timed drain. A terminal latch would make the fix inert on
precisely the scenes it targets: an over-budget `adaptive` / `overview`
partition, or simply a Capture pressed before the initial load finished,
burns the budget on frames 0–2, latches at frame 3, and would export the
remaining hundreds of frames with the pre-#1695 behaviour even though the
scene settles seconds later. (The probe's boolean describes the previous
pose, which is fine for a re-arm signal — the latched frame is captured
undrained either way.) The warning is once per _run_, not once per latch,
so a scene that keeps stalling and recovering does not log a line every
three frames.

At the end of the run, any timeouts at all produce a `log.warning` plus a
toast, so a degraded sequence is visible rather than a mystery — but the
two cases report different things on purpose. If waiting stayed on for
the whole run the count is exact ("N of M frames", M being the frames the
loop _attempted_, since a timeout is counted before the capture attempt
and a frame can still throw). If waiting was ever paused the count cannot
describe the run at all: it counts only the frames that waited and gave
up, while every frame captured while the wait was off was taken without
one — so the report says _that_ and claims no number. (Which branch runs
is keyed on a sticky "the latch fired at some point" flag, not on the
live latch, since the live one re-arms.) (The total is not pinned at
`MAX_CONSECUTIVE_LOD_TIMEOUTS`; that constant bounds the consecutive
_streak_, and a run alternating timeout/settle can reach any total before
three land in a row.) The toast for that branch says waiting was paused
rather than "LOD never settled", because a run that settled cleanly for
hundreds of frames and then hit a stall lands there too.

Both reports say the frames "may not show the level the selector had
settled on" rather than "may be coarse". The predicate is
direction-blind: `displayed !== active` also fires while the
never-downgrade gate is legitimately holding a **finer**
previously-displayed level over a coarser aspiration that is still
streaming, so a frame counted here can be better than the selection, not
worse.

The drain is offline-only, and one turntable route does not go through
it: `startVideoRecording()` sends a turntable to `VideoRecordingStrategy`
(`video-recording-strategy.ts`) whenever frame-by-frame is off **and**
the format is WebM. That path records the live canvas through
`MediaRecorder` in real time, so it cannot wait for anything — a tile
reloading its fine level on frustum re-entry can still pop in its output.
Use frame-by-frame (or any non-WebM format) to get the drain.

The drain's scope is exactly "every registered `lod_group` shows its
selected level at final quality", which is narrower than "nothing in this
scene is still loading". A `--recipe stream` scene — a single leaf with
an additive ladder and no `lod_group` — has no registered entry at all,
so the provider answers `null` and the loop does not wait, yet that
leaf's progressive refinement loop can still be climbing its ladder while
the capture runs and early frames can be exported at a partial prefix.
Same artifact class; not covered here.

Waiting rather than forcing is deliberate. `?lod-finest`
(`LODGroupRegistryDeps.getForceFinestLOD`) would pin the finest level and
skip the off-screen gate outright, but a capture visits the whole scene:
peak residency would become the entire dataset, which is exactly what the
resident-byte budget exists to prevent. Waiting costs time, not memory.

Also deliberately left alone:

- **The distance-driven coverage cross-fade.** Its weight is a function
  of projected bbox area, not wall-clock, so across a turntable it
  already spreads smoothly over consecutive exported frames — draining it
  would turn a dissolve into a hard cut.
- **The realtime WebM route** (above): `MediaRecorder` records the canvas
  as it is painted, so there is no per-frame point at which the loop
  could wait.
- **The depth-sort ordering.** `rendering/depth-sort-coordinator.ts`
  exports `resortForCapture(maxWaitMs)`, whose own doc calls it "the
  offline-capture entry point" — but the only caller is `__luxarDebug`
  (the gallery harness, which stops the rAF loop and drives each frame by
  hand). This loop keeps the rAF loop running, so the per-frame depth-sort
  scheduler does fire; with the default 3° re-sort threshold
  (`config.depthSort.angleThresholdDeg`) and a turntable stepping
  ~1°/frame, an order-dependent (`normal` / `volumetric`) gsplat
  node is nonetheless filmed with an ordering up to a few degrees stale,
  and the LOD drain makes _how_ stale vary with load timing. Wiring the
  helper in is a separate change with its own risk (it suppresses
  `requestRender` and bypasses the sorted-index apply back-pressure,
  both of which assume the loop is stopped), so it is not done here.
  Beware the name collision while reading that file: it also has a
  module-private `isCaptureQuiescent()`, unrelated to the registry
  method of the same name — it asks whether the depth-sort subsystem has
  settled (no sort RPC in flight, no queued re-sort, no chunked ordering
  apply streaming), not whether the LOD tree has.

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

The loop's **cadence** is the other half of the contract, and it is a
BROADER pairing: `core/app/init/pipeline` also gives the animation
controller a pacing-suspend predicate, keyed on this package's
`isCurrentlyRecording()` (`session.isAnyCaptureActive()`, i.e.
`session.isRecording`) rather than the narrower
`isLoopRenderSuppressed()`. While it returns true, frame pacing (#1724) is
off and every frame re-arms rAF back-to-back. Both capture families need
that, which is why the broad flag is the right key: a paced gap is a
dropped frame in a real-time MediaRecorder capture of the canvas this loop
paints, and a missed window — hence a dropped orbit step — for the offline
capture, which drives its own `await requestAnimationFrame` cadence while
registering one-shot per-frame orbit callbacks on the controller. A plain
screenshot sets neither flag and needs no suspension: it reads the canvas
after its own awaited frame rather than depending on the loop's cadence.

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

## Overlay compositing units

Overlays live in the DOM, so their sizes are authored in **viewport**
units: the overlay manager writes `font-size: <font_size × 100>vh`,
`width: <size[0] × 100>vw`, `padding: …vh`, and an `<img>` with no
configured `size` simply lays out at its natural CSS-pixel size.

The capture frame is neither the viewport nor the CSS canvas: the
offline loop renders at the chosen output height (and an embedded
viewer's canvas is a fraction of the window). `computeOverlayMetrics`
therefore derives two conversions from `glCanvas.getBoundingClientRect()`
and the viewport, and every branch uses them:

| Authored as                                            | Convert with              |
| ------------------------------------------------------ | ------------------------- |
| `font_size`, `padding`, `stroke_width`, `size[1]` (vh) | `× metrics.vh`            |
| `width`, `size[0]` (vw)                                | `× metrics.vw`            |
| natural image pixels (CSS px)                          | `× metrics.scaleX/scaleY` |
| `position` (fraction of the canvas)                    | `× canvas.width/height`   |

Resolving vh/vw against the capture canvas instead — which is what this
module used to do — is only correct when the canvas fills the viewport
AND the capture is canvas-sized. Recording breaks both halves, which is
why a logo shrank as the output resolution went up.

## Overlays in a real-time recording

`canvas.captureStream()` on the WebGL canvas sees the WebGL canvas alone,
so the real-time path used to drop every DOM overlay — "Include Overlays"
was silently a no-op for all of Video mode (which is always real-time
WebM) and for a non-smooth WebM turntable.

`live-overlay-compositor.ts` closes that: when the scene has visible
overlays and the user asked for them, `VideoRecordingStrategy` captures a
MIRROR 2D canvas instead, refreshed once per rendered frame with a blit of
the GL canvas plus the same `compositeOverlays()` the screenshot path
runs. No overlays to draw ⇒ no mirror, and the GL canvas is captured
directly as before.

The refresh is driven by the `frame-end` event, NOT by `requestAnimation-
Frame`, and that is load-bearing. The renderer runs with
`preserveDrawingBuffer: false`, so `drawImage(glCanvas)` only yields
pixels inside the same task that issued the draw calls; `frame-end` is
emitted on the line after `postProcessing.render()`. A blit from a later
task returns a fully black frame — measured on a live scene, not assumed.
Cost is 0.1 ms median / 0.3 ms max per frame on a 4.7 Mpx canvas.

The compositor is torn down in `cleanupCaptureStream()`, which every path
that ends a recording already routes through, so a per-frame listener
cannot outlive its recording.

One path still cannot composite overlays, and the confirmation dialog
says so when overlays are visible: the EXR driver writes the raw
pre-grade HDR buffer, where a display-space overlay has no meaning.
HTML overlays stay best-effort everywhere (their `foreignObject`
rasterization is async); text and image overlays are exact.

## The bundled `encode_video.sh`

Image and EXR sequences ship with a script that muxes the frames into an
MP4 (`ffmpeg-script.ts`). What it must do depends entirely on what is in
the ZIP:

- **PNG / WebP / JPEG frames are already what the viewer showed** —
  `renderToImageData` reads the framebuffer after exposure, tone mapping
  and the sRGB encode. The script applies no colour maths; adding any
  would double-grade.
- **EXR frames are scene-linear and pre-grade.** `hdr-effects-pre-tone`
  deliberately bypasses the display transform to keep unclipped HDR, so
  the script has to put it back: exposure → offset → gamma → tone map →
  sRGB. It reads the live values from
  `PostProcessingManager.getGradeSettings()` at capture time.

The tone map is emitted as an exact `geq` expression carrying three's own
constants, because ffmpeg's built-in curves are different functions.
Measured against the viewer's own PNG of the same frame:

| chain                                     | PSNR        |
| ----------------------------------------- | ----------- |
| exact `geq` (what we emit)                | **38.2 dB** |
| `tonemap=hable` (the usual ACES stand-in) | 16.0 dB     |
| no tone mapping at all                    | 23.8 dB     |
| old script (no colour handling at all)    | 15.2 dB     |
| ceiling: same maths, PNG out, no codec    | 40.3 dB     |

`hable` is _worse than doing nothing_, which is why the approximations
are not offered. AgX is the one mode with no practical closed form (four
matrix stages around a log-space polynomial), so its curve degrades to a
plain clamp and the script says so, pointing at the LDR-sequence route
instead of faking the look. Exposure, offset and gamma are still applied
there — dropping them would encode the frames at the wrong brightness,
which is the failure the whole chain exists to prevent.

`geq` evaluates its expression per pixel on the CPU, so expression _size_
is the encode cost. The cross-channel curves (ACES, Neutral) share
intermediates through `st()`/`ld()` registers rather than re-inlining
them, which is worth an order of magnitude on the same frames:

| curve   | inlined            | with registers   |
| ------- | ------------------ | ---------------- |
| ACES    | 8.0 KB, 11.0 s/fr  | 2.0 KB, 1.5 s/fr |
| Neutral | 63.2 KB, 26.1 s/fr | 1.5 KB, 1.0 s/fr |

Other things the script gets right that are easy to get wrong: `-tag:v
hvc1` (libx265 defaults to `hev1`, which QuickTime and Safari refuse),
`setparams` colour tagging (the `-color_*` output options silently failed
to reach the container on the LDR path) paired with an RGB→YUV conversion
that uses the matrix it tags (`scale=out_color_matrix=bt709`, since
swscale's default is BT.601 and the mismatch costs 40/255 on saturated
colour — the EXR chain's `zscale` already converts with `m=bt709`),
`-start_number 0`, and an HDR10
stanza that actually _converts_ to PQ/BT.2020 rather than tagging SDR
pixels as HDR. Output names come from the capture filename, and a
turntable's script notes that its frames loop seamlessly.

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
