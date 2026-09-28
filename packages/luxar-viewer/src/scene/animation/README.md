# Scene animation

Two animation drivers used by the viewer: the render-loop controller
that ticks every frame, and the dimension-animation manager that
auto-scrubs an nD slider through its range. They are independent
concerns wired together via `addPerFrameCallback` — the dimension
manager registers itself as one callback among many on the shared
render loop.

The wider scene orchestration lives in `../scene-manager.ts`; this
folder contains only the loop and the dimension scrubber.

## Files

| File                             | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `animation-controller.ts`        | The render loop's WORK and idle policy. `tick()` updates `ControlsManager`, runs registered per-frame callbacks, then renders through `PostProcessingManager`, emitting `frame-start` / `frame-end` on the event bus (for the PerformanceMonitor panel). Auto-pauses after `config.animation.idleTimeoutMs` of inactivity unless something continuous is active — a pointer gesture in progress (`ControlsManager.isGestureActive()`) counts as continuous. Frames are scheduled by `RafDriver`. |
| `committed-quality.ts`           | Walks visible, non-empty commit stamps and reports the minimum committed energy used by dimension-playback feedback.                                                                                                                                                                                                                                                                                                                                                                             |
| `dimension-animation-manager.ts` | Per-dimension FPS-throttled scrubber with `once` / `loop` / `bounce` modes. Mutates `SceneDimsManager` state and awaits `waitForUpdate()` so animation never advances faster than data loading. Extends `THREE.EventDispatcher` — emits `play`, `pause`, `complete`, `directionChange`, `speedChange`, `loopModeChange`, `fpsWarning`.                                                                                                                                                           |
| `view-signature.ts`              | The camera half of render-on-change: camera identity, `matrixWorld`, `projectionMatrix` and drawing-buffer size, compared in a preallocated `Float64Array` against the copy taken at the last RENDERED frame (no allocation per tick).                                                                                                                                                                                                                                                           |
| `render-audit.ts`                | `?renderAudit` (debug only): renders every tick and, on each tick the scheduler would have skipped, compares a 64×64 readback against the last real frame; a difference counts `render.missedDirty` and logs the last render requests.                                                                                                                                                                                                                                                           |
| `raf-driver.ts`                  | The render loop's SCHEDULING: `requestAnimationFrame`, the running flag, and frame pacing for pathologically slow frames (see the invariant below). Runs its `onFrame` (the controller's `tick()`) once per frame, AFTER arming the next frame, so the loop survives an exception thrown by the frame's work. A different driver (e.g. a WebXR session's own animation loop) can run the same `tick()`.                                                                                          |

## Public surface

**`AnimationController`**

- `startAnimation()` / `stopAnimation()` — start or pause the rAF loop. `startAnimation` is also the event-handler wired to controls, canvas input, and dimension changes; calling it on every interaction resets the idle timer. Every call also marks the next tick dirty, so each wake gets at least one rendered frame (`render.byReason.wake`).
- `requestRender(detail)` — something the next frame draws changed outside the loop (a geometry commit, a sort result, an environment landing): mark the next tick dirty and wake the loop (`render.byReason.event`; `detail` names the source in the render audit's log). `requestTick()` — keep the loop ticking (controls, callbacks, idle timer) WITHOUT asking for a render: for a subsystem that polls per frame for asynchronous work (a lazy LOD level loading), whose landing requests the render itself.
- `setRenderOnChange(enabled)` / `isRenderOnChange` — see **Render on change** below. `setViewSignatureSource({ getCamera, getDrawingBuffer })` wires the camera half of the dirty check; `setRenderAudit(audit)` the debug audit.
- `addPerFrameCallback(id, fn, { continuous?, phase?, renderEveryFrame? })` / `removePerFrameCallback(id)` / `hasPerFrameCallback(id)` — register named callbacks executed after `controls.update()` but before `postProcessing.render()`. Callbacks run by `phase` — `camera` (callbacks that move the camera: a flight, the recording turntable, the offline capture's orbit step) → `view` (the default: clipping, depth sort, density, LOD, dimension playback) → `pre-render` (the scene-captured environment) → `ui` (scale bar, Layers LOD status) — and in registration order within a phase. A camera writer therefore moves the camera before anything reads it, however late it registered: before phases, a flight (registered when it starts, after the init-time view callbacks) rendered every frame with the previous frame's near/far. `continuous: true` keeps the loop alive past the idle timeout (used by dimension animation and turntable recording); the default `false` is on-demand (e.g. dynamic clipping, scale bar). `renderEveryFrame: true` makes every tick RENDER while the callback is registered — only for a consumer of the canvas itself (the real-time recording's keep-alive), never for a state change. The callback's return value is the render-on-change contract: `true` when it changed what the next render draws, nothing / `false` when it did not. Each callback runs isolated: one that throws is logged once per id (not once per frame) and the other callbacks and the render still run; `frame-end` always pairs `frame-start`, even when the frame's work throws.
- A **pointer gesture in progress** also keeps the loop alive: `shouldContinueAnimating()` reads `ControlsManager.isGestureActive()`, true between the active controls' `start` and `end`. The controls only accumulate rotate/pan/zoom deltas on input; `update()` — a per-frame call — is what applies them. Before this, a button or finger held still for `idleTimeoutMs` paused the loop and the drag that followed moved nothing (mouse: press, hold 2 s, drag; the same with a finger).
- `setAdaptiveDPRManager(manager)` — opt-in DPR feedback: at the start of each tick the loop calls `recordFrame(tickStart)` when the PREVIOUS tick rendered (the interval between two tick starts measures the earlier tick), so the manager can downshift pixel ratio under load. Skipped after a tick that drew nothing — a skipped (unchanged) tick, or one while the context is lost or another owner drives the pipeline (see `setRenderSkipPredicate` below): a frame that draws nothing is not a fast frame. The first render after skipped ticks calls `notifyStreamBreak(tickStart)` instead, which re-bases the frame-interval clock without a sample and without clearing anything (unlike `notifyPaused`). Under `?renderAlways` every tick records, as it always did.
- `setContextLostPredicate(predicate)` — injected by `SceneManager` to suppress GPU work while the WebGL context is lost; controls and callbacks still tick so input stays responsive.
- `setRenderSkipPredicate(predicate)` — injected by `core/app/init/pipeline`, keyed on `RecordingPanel.isLoopRenderSuppressed()`: an offline capture renders its own pipeline pass per frame, so the loop's render is discarded work. Same shape as the context-lost guard — controls and callbacks still tick, but adaptive-DPR frames are not recorded (a frame that draws nothing is not a fast frame). Offline-only; the real-time recording path records the canvas the loop paints. Narrower than the capture's own mutual-exclusion flag: it is dropped before the capture teardown awaits its driver abort, so a wedged abort cannot freeze the viewport. It gates the idle-restore frame below as well — both of the controller's render call sites, since the claim is that nobody but the pipeline's current owner may draw.
- `setIdleRestorePredicate(predicate)` — consulted before the idle-pause native-DPR restore; returning false keeps the current DPR (recording resolution stays locked for a whole capture).
- `setPacingSuspendPredicate(predicate)` — injected by `core/app/init/pipeline`, keyed on the BROAD `RecordingPanel.isCurrentlyRecording()` (`session.isAnyCaptureActive()`, i.e. `session.isRecording` — the flag both the real-time MediaRecorder path and the offline capture set for the whole of their run). While it returns true, frame pacing is off and every frame re-arms rAF back-to-back: the real-time path records the canvas this loop paints, so a paced gap is a dropped frame in the video, and the offline capture drives its own `await requestAnimationFrame` cadence with one-shot per-frame orbit callbacks registered here, so a paced frame could miss its window and drop the orbit step. A plain screenshot does not set the flag and does not need it — it reads the canvas after its own awaited frame rather than depending on the loop's cadence. Deliberately wider than the `isLoopRenderSuppressed()` flag behind `setRenderSkipPredicate`. A predicate that THROWS is treated as "not suspended": `RafDriver.scheduleNextFrame()` is the loop's only re-arm point, so a throw escaping it would freeze the viewer unrecoverably.
- `tick()` — one frame of work (controls, per-frame callbacks, the render guard and render, `frame-start` / `frame-end`). Called by the driver each frame; public so another driver can run frames. It neither checks nor starts the loop.
- `get isActive` — true while the loop is running.
- `dispose()` — stops the loop and clears all per-frame callbacks. Not reusable after dispose.

**`DimensionAnimationManager`**

- `play(dimIndex, options?)` / `pause(dimIndex)` / `togglePlay(dimIndex, options?)` / `stop(dimIndex)` — control playback for one dimension. `stop` also removes the state entry; `pause` keeps it. `play` lazily calls `ensureRegistered()`, which installs the `'dimension-animation'` callback on the shared `AnimationController` with `continuous: true` and starts the loop.
- `setTargetFPS(dimIndex, fps)` / `increaseSpeed(dimIndex)` / `decreaseSpeed(dimIndex)` — set or step through the FPS presets (`config.dimensionAnimation.presets.fps`, default `[1, 2, 5, 10, 15, 30, 60]`); values clamp to `[customMin, customMax]`. The speed-step helpers jump to the next/previous preset and fall back to ±10 % multiplicative steps when already past the top/bottom preset.
- `setLoopMode(dimIndex, mode)` — `'once' | 'loop' | 'bounce'`. `setTargetFPS` and `setLoopMode` create a paused state entry if none exists, so the UI can pre-configure a dimension before the user hits play.
- `isAnimating(dimIndex)` / `getState(dimIndex)` — query playback flag and the full `DimensionAnimationState` (target/actual FPS, frame counters, direction).
- `dispose()` — pauses every dimension, clears state, removes the per-frame callback, and unsubscribes the `SceneDimsManager` listener.

`PlayOptions` (`targetFPS?`, `loopMode?`, `direction?`) is exported for
callers that build the options object dynamically; defaults come from
`config.dimensionAnimation.defaults`.

## Render on change

`config.animation.renderOnChange` (default `true`; `?renderAlways` turns it
off for a session, `LuxarAppOptions.renderAlways` for an embedder). The loop
keeps TICKING exactly as before — controls, per-frame callbacks, the idle
timer and its idle-restore frame — but a tick RENDERS only when something
changed since the last rendered frame. Measured before: one 4 px drag
rendered 181 frames over 3 s (every rAF through the 2 s idle tail), and 10 fps
hidden-dimension playback 360 frames for 54 data commits. A tick renders when,
in this precedence (each render lands in exactly one `render.byReason.*`):

| Reason        | Trigger                                                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `wake`        | `startAnimation()` since the last render (every input / state handler that wakes the loop)                                    |
| `event`       | `requestRender(detail)` since the last render (geometry commit, sort result, resize via the scene manager's `change`, …)      |
| `once`        | `renderOnce()` (draws now on a stopped loop; marks the next tick on a running one)                                            |
| `cb:<id>`     | a per-frame callback returned `true`                                                                                          |
| `camera`      | the view signature differs: camera object, `matrixWorld`, `projectionMatrix`, drawing-buffer size                             |
| `continuous`  | every tick must draw: cinematic detector noise (`needsContinuousAnimation()`), a `renderEveryFrame` callback, `?renderAlways` |
| `idleRestore` | the idle pause's ceiling-DPR frame (unchanged)                                                                                |
| `audit`       | `?renderAudit` rendering a tick the scheduler would have skipped                                                              |

The dirty flag is cleared just BEFORE the render, so a request made during it
(an `onAfterRender` hook, a sort landing) dirties the next tick. A tick blocked
by context loss or an offline capture keeps whatever was dirty. Skipped ticks
count `render.skippedTicks`; `frame-end` carries `{ rendered }` so the
performance readout and the renderer-info sampler count only drawn frames.

The pixels of a rendered frame are identical to the always-render loop's for
the same state — the change only skips re-drawing an unchanged state. Anything
that mutates drawn state must therefore say so: return `true` from its
per-frame callback, or call `requestRender` / `startAnimation` from its event.
`?debug&renderAudit` verifies it (see `render-audit.ts` and the
`render-on-change-audit.spec.ts` E2E).

The converse holds too: a geometry commit to a node that is not drawn (it or
an ancestor hidden — an LOD level the registry keeps off screen, a hidden
layer) calls `requestTick()`, not `requestRender` (`SceneLoader.setRequestRender`
and `geometry-committed` carry `drawn`). The loop still wakes, so the LOD
registry sees the commit; if it then shows that level, its visibility flip is
the redraw. During LOD timelapse playback the eager coarse level re-commits
every timepoint under the held fine level, and redrawing for it cost a second,
unchanged render per tick.

## Invariants

- **One render loop, many callbacks.** `AnimationController` is a
  singleton-style driver inside the app — multiple subsystems share it
  through `addPerFrameCallback` with unique IDs. Removing a callback
  by ID must not affect the others, so the storage is a `Map`, not an
  array.
- **Continuous vs on-demand callbacks.** Only callbacks registered
  with `continuous: true` count toward `shouldContinueAnimating()`.
  Auto-rotate, post-processing effects that self-declare via
  `needsContinuousAnimation()`, and continuous callbacks together
  decide whether idle timeout fires.
- **No GPU work while context is lost — or while another owner drives
  the pipeline.** When either injected predicate (`isContextLost`,
  `shouldSkipRender`) returns true, the loop emits `frame-end`
  (`rendered: false`) and returns before `postProcessing.render()`. Controls and per-frame
  callbacks still execute, so UI input is unaffected during the loss
  window, and the depth-sort scheduler and LOD selector keep following
  the camera through an offline capture.
- **A slow loop must still yield the main thread.** Frames re-arm
  `requestAnimationFrame` immediately, so a scene whose frames cost ~1 s
  puts the main thread at a 100 % duty cycle of long tasks and no other
  task gets a slot — not worker message delivery, not a CDP evaluate.
  That is a livelock, not just a slow render: worker replies arrived at
  ~0.5/s, every landed reply staged an ordering apply that called
  `requestRender()`, and the loop could never idle (#1724). So when a
  frame's own cost exceeds `config.animation.pacing.slowFrameMs` for TWO
  consecutive frames the next frame is scheduled after
  `min(maxCooldownMs, 25 % of the cost)` — a bounded FRACTION, because a flat
  gap cannot track the cost across the band where the fraction decides it
  (250 ms to 1 s): a fixed 100 ms over-yields at the bottom and under-yields
  at the top, 9 % of wall-clock after a 1 s frame against the fraction's 20 %.
  Past 1 s the clamp makes the shipped gap flat too, which is its job — it
  bounds the added latency of an on-demand repaint. Three properties are
  load-bearing:
  - The cooldown is armed from a zero-delay hop
    (`setTimeout(0)` → `setTimeout(cooldown)` → rAF), not directly. A slow
    frame spends its second on browser rendering work that runs after the
    rAF callback returns but inside the same main-thread task, so a timer
    armed at the frame's start is always already overdue when the thread
    frees and inserts no gap at all. The hop runs at the first event-loop
    turn after that work. Both halves of the fix then hold: no
    animation-frame request is outstanding while the frame is drawn, AND a
    genuine cooldown follows it.
  - The cost is the frame PERIOD minus the cooldown _we_ inserted (the
    wedge's second is non-JS main-thread time, so a JS-body span reads
    ~2 ms and would never fire; not subtracting our own gap would make
    pacing latch on forever), and it resets on the stopped→running edge so
    an idle rest or a tab-hide is not read as one enormous frame.
  - Only a STREAK paces. The wedge is sustained (every frame ~1 s,
    forever), so requiring a second consecutive slow frame only delays the
    first cooldown — while a lone outlier is precisely what must not be
    paced. Two MEASURED slow frames means three frames in one uninterrupted
    run (the first has no predecessor and measures nothing), and the streak
    resets on the stopped→running edge, so pacing is unreachable from a cold
    start once a frame costs more than `idleTimeoutMs / 2` — the idle timer
    stops the loop before a third frame exists. Neither costs anything: the
    wedge holds `animating=true` continuously (each landed reply calls
    `requestRender()`, pushing the idle timer out), so it paces on the third
    frame; and a loop that reaches its idle pause has already yielded the
    main thread, which is all pacing is for. Because the trigger is a period, a
    FOREIGN main-thread task of that size is charged to the loop too (a GC
    pause, a shader compile, one chunk decode); one of them can no longer
    pace anything, and a sustained run of them still does, which is what
    you want during a heavy load. An ALTERNATING slow/fast cadence is not
    paced either — the fast frames are proof the main thread is already
    getting slots. A single fast frame resets the streak, so recovery is
    immediate. See the `slowFrameMs` comment in
    `config/sections/animation/data.ts`.
  - Frames are DELAYED, never skipped, and both readouts get the REAL
    clock: each frame still emits exactly one `frame-start` / `frame-end`
    pair, and still calls `recordFrame(performance.now())` for a rendered
    predecessor (that call keeps its pre-existing context-lost /
    render-skip gate, per `setAdaptiveDPRManager` above). The achieved frame rate
    really is lower, so neither the FPS readout nor adaptive DPR is told
    otherwise (no virtual pacing clock to drift against
    `notifyContentChanged()`). A steady paced cadence is absorbed by the
    stall detector's median-based outlier test rather than read as a gap,
    and the isolated-hiccup interactions that would have mattered — a paced
    interval read off a freshly cleared window as a collapsed frame rate, a
    just-under-`gapResetMs` frame pushed just over it — cannot arise,
    because an isolated slow frame is never paced.
- **Animation never outruns data.** `DimensionAnimationManager`
  tracks `pendingUpdates` per dimension. After
  `setDimensionValue(...)` it awaits `sceneDimsManager.waitForUpdate()`
  before clearing the pending flag; until the flag clears, that
  dimension's frame work is skipped. Independent dimensions animate
  in parallel because the flag is per-index.
- **Playback feedback is informational only.** The manager emits `fpsWarning`
  when cadence falls below `feedbackThreshold * targetFPS` or visible committed
  energy is below the display threshold. Logging is warning-level while frames
  are still filling (or quality is unknown), informational when enough content
  is visible and only cadence slipped. It never throttles or stops animation.
- **Cadence is schedule-based.** Ticks fall due on a fixed grid
  (`state.nextDue += period`), and the frame within half a vsync (an EWMA of
  rAF deltas) of the due time takes the tick, so a 10 or 30 fps target is met
  on a 60 Hz display rather than drifting by up to a vsync per tick. A playhead
  more than a period late (a data stall) resyncs from now instead of bursting
  catch-up ticks.
- **Boundary semantics** (`advance-value.ts`). The range ends are frames to
  show: in `loop` and `once` a step that reaches or overshoots `max` (forward)
  or `min` (backward) lands ON it, and only the next step, taken from the
  endpoint, wraps (`loop`) or completes (`once`). `bounce` clamps and flips the
  direction on arrival. On a discrete dim whose `max` is off its grid, the last
  grid point is the endpoint. Discrete dims advance by `metadata.step` (or the
  step override), continuous dims by `range / continuousTraverseSeconds`
  scaled to the current target FPS. `peekNextValue` runs the same function, so
  the t+1 prefetch predicts the endpoint frame too.

## Events

`DimensionAnimationManager` extends `THREE.EventDispatcher` with the
`DimensionAnimationEvents` map from `src/types/animation.ts`:

| Event             | Payload                                                       | When                                                                      |
| ----------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `play`            | `{ dimIndex }`                                                | `play()` transitions a paused dim to playing                              |
| `pause`           | `{ dimIndex }`                                                | `pause()` transitions a playing dim                                       |
| `complete`        | `{ dimIndex }`                                                | `once` mode hit the far boundary                                          |
| `directionChange` | `{ dimIndex, direction }`                                     | `bounce` mode flipped at a boundary                                       |
| `speedChange`     | `{ dimIndex, fps }`                                           | `setTargetFPS` applied (after clamp)                                      |
| `loopModeChange`  | `{ dimIndex, loopMode }`                                      | `setLoopMode` applied                                                     |
| `fpsWarning`      | `{ dimIndex, targetFPS, actualFPS, committedEnergyFraction }` | Cadence slipped or a visible node is below the committed-energy threshold |

`AnimationController` does not extend `EventDispatcher`; it publishes
`frame-start` and `frame-end` on `utils/cross-layer/event-bus` so the
`PerformanceMonitor` UI panel can subscribe without
animation-controller importing UI code.

## See also

- `../scene-manager.ts` — instantiates both classes and wires them
  into controls, post-processing, resize, and WebGL-context-loss
  recovery.
- `../scene-dims-manager.ts` — the dimension state that
  `DimensionAnimationManager` mutates via `setDimensionValue` and
  observes via `addListener` / `waitForUpdate`.
- `../../rendering/adaptive-dpr-manager.ts` — receives
  `recordFrame(now)` for each rendered frame and `notifyStreamBreak(now)`
  when rendering resumes after skipped ticks, when wired through
  `setAdaptiveDPRManager`.
- `../../config/sections/animation` and
  `../../config/sections/dimension-animation` — idle timeout, pacing
  thresholds, FPS presets, min-frame time, continuous-traverse seconds,
  feedback threshold.
- `../../types/animation.ts` — `DimensionAnimationState`,
  `LoopMode`, `AnimationDirection`, `DimensionAnimationEvents`.
- `../../utils/cross-layer/event-bus.ts` — `frame-start` / `frame-end` topics.
