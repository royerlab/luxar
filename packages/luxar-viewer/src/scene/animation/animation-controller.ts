// Animation loop management for the Luxar scene player
//
// This module handles the core animation loop that drives 3D rendering:
// - RequestAnimationFrame-based rendering loop for smooth 60fps
// - Frame pacing that yields the main thread back between very slow frames
// - Intelligent pause/resume system to conserve CPU/GPU when idle
// - Performance monitoring integration for real-time metrics
// - Proper cleanup and resource management

import { ControlsManager } from '../../controls/controls-manager';
import { config } from '../../config';
import { PostProcessingManager } from '../../rendering';
import { AdaptiveDPRManager } from '../../rendering/adaptive-dpr-manager';
import { eventBus } from '../../utils/cross-layer/event-bus';

/**
 * Fraction of a slow frame's own cost inserted as a cooldown before the
 * next frame.
 *
 * Deliberately a FRACTION of the cost rather than a constant, over the band
 * where the fraction is what decides the gap: from
 * `config.animation.pacing.slowFrameMs` (250 ms) up to 1 s, above which
 * `config.animation.pacing.maxCooldownMs` (250 ms) clamps it flat. Across
 * that band the gap tracks the cost — at 0.25 it is a quarter of the frame,
 * so about a fifth of wall-clock (`0.25 / 1.25`) is yielded — whereas a flat
 * gap cannot: a fixed 100 ms over-yields at the bottom of the band and
 * under-yields at the top (after a 1 s frame it is 9 % of wall-clock against
 * the fraction's 20 %).
 *
 * Above the clamp the shipped behaviour is flat too, so the fraction wins
 * nothing there: at 5 s the clamped 250 ms is 4.8 % of wall-clock, the same
 * order as the fixed-100 ms strawman's 2 %. That is the clamp doing its job —
 * it bounds the added latency of an on-demand repaint (see `maxCooldownMs`)
 * — not a failure of the constant. The measured wedge (~1042 ms frames) sits
 * essentially at the clamp already.
 *
 * Any of those shares is only real because the cooldown is armed from the first
 * event-loop turn AFTER the frame's work rather than at the frame's start
 * (see `scheduleNextFrame()`): a timer armed at the start would already be
 * overdue by the time the thread frees, and would insert nothing.
 */
const PACING_COOLDOWN_FRACTION = 0.25;

/**
 * Consecutive frames whose own cost must exceed
 * `config.animation.pacing.slowFrameMs` before a cooldown is inserted.
 *
 * Two rather than one, because the two failure modes are asymmetric:
 * - The wedge pacing exists for is SUSTAINED — every frame costs ~1 s and
 *   never recovers. Requiring a second consecutive slow frame therefore only
 *   delays the first cooldown; it never withholds it.
 * - A single outlier is precisely what must NOT be paced: a GC pause, a
 *   shader compile, one synchronous chunk decode, or any foreign
 *   main-thread task charged to the loop because the measurement is a frame
 *   PERIOD. Pacing one of those inserted a cooldown that adaptive DPR then
 *   read as a 3.75 fps frame rate off a freshly cleared two-sample window
 *   (an unprobed 10 % DPR scale-down), and turned a sub-`gapResetMs`
 *   interval into an over-`gapResetMs` one, inventing a stall gap-reset
 *   that had not happened. With the streak, adaptive DPR never sees a
 *   cooldown that was not preceded by genuinely sustained slowness.
 *
 * An ALTERNATING slow/fast cadence is deliberately not paced either: the
 * fast frames are proof the main thread is already getting slots, which is
 * the only thing a cooldown buys.
 *
 * Two MEASURED slow frames means three frames in one uninterrupted run: the
 * first frame of a run has no predecessor and so measures nothing, the second
 * sets the streak to 1, and the third reaches 2 and arms the first cooldown.
 * And since `startAnimation()` clears the streak on the stopped→running edge,
 * pacing is unreachable from a cold start whenever a frame costs more than
 * `config.animation.idleTimeoutMs / 2` — the idle timer fires before a third
 * frame exists and stops the loop.
 *
 * Neither fact costs anything here. The reported wedge holds
 * `animating=true` continuously — every landing depth-sort reply calls
 * `requestRender()`, which pushes the idle timer out again — so the streak
 * accumulates and pacing engages on the third frame. And in the cold-start
 * case the loop reaching its idle pause IS the outcome pacing exists to
 * enable: the main thread is free either way, so there is nothing to fix.
 */
const PACING_SLOW_FRAME_STREAK = 2;

/**
 * AnimationController manages the main rendering loop and performance optimization
 *
 * Key Features:
 * - RequestAnimationFrame loop for browser-optimized rendering
 * - Frame pacing for pathologically slow frames (see below)
 * - Automatic pause/resume based on user interaction (saves power)
 * - HDR post-processing pipeline with bloom effects
 * - Integrated performance monitoring with stats.js
 * - Proper frame timing and resource cleanup
 *
 * Technical Details:
 * - Uses requestAnimationFrame for 60fps synchronized with display refresh
 * - Pauses after 2 seconds of inactivity to reduce CPU/GPU usage
 * - Integrates Three.js controls.update() and HDR post-processing render
 * - Measures frame timing for performance analysis including post-processing
 *
 * Frame pacing (#1724):
 * Because each frame re-arms `requestAnimationFrame` immediately, a scene
 * whose frames cost ~1 s puts the main thread at a 100 % duty cycle of
 * long tasks, and NOTHING else ever gets a slot — not worker message
 * delivery, not a CDP `Runtime.callFunctionOn`. That is not a rendering
 * inconvenience but a livelock: the depth-sort worker's replies (each
 * 0.1 ms of actual work) were dispatched at ~0.5/s, every landed reply
 * staged an ordering apply that called `requestRender()`, and the loop
 * could therefore never idle — the rendering starved the very hand-off
 * that would have let it stop. Measured on
 * `performance_benchmark_example.luxar.zarr`: ~120 of 200 dispatches
 * still outstanding after 70 s, `page.evaluate` timing out at 15 s
 * throughout.
 *
 * So when a frame's own cost exceeds `config.animation.pacing.slowFrameMs`
 * for `PACING_SLOW_FRAME_STREAK` consecutive frames — sustained slowness,
 * not an isolated hiccup — the next frame is scheduled after a bounded
 * cooldown (`setTimeout` → `setTimeout` → `requestAnimationFrame`) instead
 * of back-to-back. BOTH halves of that matter:
 * - no animation-frame request is outstanding while the frame is drawn, so
 *   the compositor stops driving main frames back-to-back on its own;
 * - and a genuine cooldown follows it. The genuineness is why the cooldown
 *   is armed from a zero-delay hop rather than directly: the expensive part
 *   of a slow frame runs after the rAF callback returns but inside the same
 *   main-thread task, so a timer armed at the frame's START is always
 *   already overdue when the thread frees and inserts no gap at all. The
 *   hop runs at the first event-loop turn after that work; only then is the
 *   real cooldown armed.
 *
 * Frames are DELAYED, never skipped: each one that runs still emits exactly
 * one `frame-start` / `frame-end` pair, and records itself with
 * `adaptiveDPRManager.recordFrame()` whenever that frame does GPU work of its
 * own — the call is gated on the context-lost and render-skip predicates, as
 * it was before pacing existed (see the comment at the call site). Both are
 * on the real clock: the achieved frame rate really is lower and neither the
 * FPS readout nor the DPR control loop may be told otherwise.
 */
export class AnimationController {
  /** Whether the animation loop is currently running */
  private isAnimating = false;

  /** RequestAnimationFrame ID for cancellation */
  private animationId: number = 0;

  /** Timeout ID for auto-pause functionality */
  private idleTimeout: ReturnType<typeof setTimeout> | null = null;

  /**
   * Timeout ID for the pending frame-pacing chain (null = none armed). Holds
   * the zero-delay hop first, then the cooldown that hop arms.
   */
  private pacingTimeout: ReturnType<typeof setTimeout> | null = null;

  /** Start timestamp of the previous frame (null = no frame measured yet) */
  private lastFrameStartTime: number | null = null;

  /** Cooldown we inserted BEFORE the current frame, in ms */
  private appliedPacingDelayMs = 0;

  /** Previous frame's own cost in ms (frame period minus our own cooldown) */
  private lastFrameCostMs = 0;

  /**
   * How many consecutive measured frames have cost more than
   * `config.animation.pacing.slowFrameMs`. Pacing engages only at
   * `PACING_SLOW_FRAME_STREAK`; any fast frame resets it to 0.
   */
  private consecutiveSlowFrames = 0;

  /** Per-frame callbacks for additional updates (keyed by ID for safe add/remove) */
  private perFrameCallbacks: Map<string, { callback: () => void; continuous: boolean }> = new Map();

  /** Adaptive DPR manager for dynamic resolution scaling */
  private adaptiveDPRManager: AdaptiveDPRManager | null = null;

  /**
   * Predicate that returns true while the WebGL context is lost. When
   * set, the animation loop skips `postProcessing.render()` (and any
   * GPU-bound work) so we don't issue draw calls against a dead
   * context — those produce noisy GL errors and waste frame work
   * during the loss window. The renderer is rebuilt by SceneManager
   * on `webgl-context-restored`; until then we keep ticking
   * controls.update() and per-frame callbacks but skip rendering.
   */
  private isContextLost: (() => boolean) | null = null;

  /** Guard for the idle-pause DPR restore (null = always allowed). */
  private canRestoreAtIdle: (() => boolean) | null = null;

  /**
   * Predicate that returns true while some other owner is driving the
   * pipeline itself and the loop's own render would be thrown away.
   * Set during an offline capture, whose every frame runs its own
   * independent pipeline pass into an offscreen target. See
   * {@link setRenderSkipPredicate}.
   */
  private shouldSkipRender: (() => boolean) | null = null;

  /**
   * Predicate that returns true while frame pacing must stay off because
   * some other owner depends on the loop's exact frame cadence. Set for
   * the whole of a recording. See {@link setPacingSuspendPredicate}.
   */
  private isPacingSuspended: (() => boolean) | null = null;

  /**
   * Create animation controller for rendering loop management.
   *
   * Sets up performance monitoring and prepares animation loop. Does not
   * start animation - call startAnimation() to begin rendering.
   *
   * @param controls - Controls manager for camera updates each frame
   * @param postProcessing - Post-processing manager for HDR rendering
   *
   * @example
   * ```typescript
   * const animController = new AnimationController(
   *   controlsManager,
   *   postProcessingManager
   * );
   * animController.startAnimation();  // Begin rendering loop
   * ```
   */
  constructor(
    private controls: ControlsManager,
    private postProcessing: PostProcessingManager
  ) {}

  /**
   * Add a per-frame callback with a unique identifier.
   *
   * Multiple callbacks can be registered simultaneously (unlike setPerFrameCallback).
   * Use unique IDs to allow safe removal without affecting other callbacks.
   *
   * Useful for operations that need to run every frame:
   * - Dynamic clipping plane adjustments
   * - Dimension animations
   * - Camera-based LOD updates
   * - Custom animations or effects
   *
   * The callback is executed after controls.update() but before rendering.
   *
   * @param id - Unique identifier for this callback (for later removal)
   * @param callback - Function to call each frame
   * @param options - Options controlling callback behavior
   * @param options.continuous - If true, this callback prevents the animation loop from
   *   auto-pausing due to idle timeout. Use for callbacks that need every frame (e.g.,
   *   dimension animation, recording). Default: false (on-demand callbacks that only run
   *   when animation is active but don't prevent pausing).
   *
   * @example
   * ```typescript
   * // On-demand callback: runs when animating but doesn't prevent idle pause
   * animController.addPerFrameCallback('dynamic-clipping', () => {
   *   sceneManager.updateDynamicClippingPlanes();
   * });
   *
   * // Continuous callback: keeps animation loop alive
   * animController.addPerFrameCallback('dimension-animation', () => {
   *   animationManager.onFrame();
   * }, { continuous: true });
   * ```
   */
  addPerFrameCallback(id: string, callback: () => void, options?: { continuous?: boolean }): void {
    this.perFrameCallbacks.set(id, { callback, continuous: options?.continuous ?? false });
  }

  /**
   * Remove a per-frame callback by its identifier.
   *
   * @param id - Identifier of the callback to remove
   * @returns true if callback was found and removed, false otherwise
   *
   * @example
   * ```typescript
   * // Remove dimension animation callback
   * animController.removePerFrameCallback('dimension-animation');
   * ```
   */
  removePerFrameCallback(id: string): boolean {
    return this.perFrameCallbacks.delete(id);
  }

  /**
   * Check if a per-frame callback with the given ID exists.
   *
   * @param id - Identifier to check
   * @returns true if callback exists
   */
  hasPerFrameCallback(id: string): boolean {
    return this.perFrameCallbacks.has(id);
  }

  /**
   * Set the adaptive DPR manager for dynamic resolution scaling.
   *
   * The animation loop will call recordFrame() on the manager each frame
   * to track FPS and adjust pixel ratio as needed.
   *
   * @param manager - The AdaptiveDPRManager instance, or null to disable
   */
  setAdaptiveDPRManager(manager: AdaptiveDPRManager | null): void {
    this.adaptiveDPRManager = manager;
  }

  /**
   * Inject a predicate the loop can poll to detect WebGL context
   * loss. When the predicate returns true, the animation loop skips
   * `postProcessing.render()` for that frame; controls and per-frame
   * callbacks still run so user input stays responsive. SceneManager
   * wires this to its own `isWebGLContextLost()`.
   *
   * Pass `null` to disable the guard (useful in tests / embed contexts
   * that can't lose the context).
   */
  setContextLostPredicate(predicate: (() => boolean) | null): void {
    this.isContextLost = predicate;
  }

  /**
   * Inject a predicate consulted before the idle-pause native-DPR
   * restore. When it returns false the resting frame keeps the current
   * DPR — used to protect recordings, whose resolution must stay
   * locked for the whole capture. Mirrors `setContextLostPredicate`.
   * Pass `null` to always allow the restore.
   */
  setIdleRestorePredicate(predicate: (() => boolean) | null): void {
    this.canRestoreAtIdle = predicate;
  }

  /**
   * Inject a predicate the loop polls to decide whether to skip its own
   * `postProcessing.render()`. When it returns true the frame still
   * runs controls.update() and every per-frame callback — the loop has
   * to keep ticking so the depth-sort scheduler and the LOD group
   * selector follow the camera — but issues no draw call of its own.
   *
   * Wired to the offline capture, which renders its own pipeline pass
   * per frame into an offscreen target: the loop's render is pure waste
   * there, and worse, EXR capture holds global mega-shader flags (raw
   * HDR, effects off) across its async readback, so a loop render
   * landing inside that window paints a blown-out frame under the
   * translucent capture overlay. Must stay OFF for the real-time
   * MediaRecorder path, which records the canvas the loop paints.
   * Mirrors `setContextLostPredicate`. Pass `null` to always render.
   */
  setRenderSkipPredicate(predicate: (() => boolean) | null): void {
    this.shouldSkipRender = predicate;
  }

  /**
   * Inject a predicate the loop polls before inserting a frame-pacing
   * cooldown. While it returns true, pacing is disabled and every frame
   * re-arms `requestAnimationFrame` back-to-back exactly as it did
   * before pacing existed.
   *
   * Wired to recording, whose two capture families both depend on the
   * loop's untouched cadence:
   * - the real-time MediaRecorder path records the canvas THIS loop
   *   paints, so a paced gap is a dropped frame in the output video;
   * - the offline capture drives its own `await requestAnimationFrame`
   *   cadence while registering one-shot per-frame orbit callbacks on
   *   this controller, so a paced frame could miss the capture's window
   *   and drop the orbit step.
   *
   * That is why this is keyed on `RecordingPanel.isCurrentlyRecording()`
   * (`session.isAnyCaptureActive()`, i.e. `session.isRecording`: the flag
   * that both the real-time MediaRecorder path and the offline capture set
   * for the whole of their run) rather than the narrower
   * `isLoopRenderSuppressed()` that gates {@link setRenderSkipPredicate} —
   * the breadth is the point here. A plain screenshot does NOT set it and
   * does not need it: it reads the canvas after its own awaited frame
   * rather than depending on the loop's cadence.
   *
   * Mirrors `setContextLostPredicate`. Pass `null` to always allow
   * pacing.
   */
  setPacingSuspendPredicate(predicate: (() => boolean) | null): void {
    this.isPacingSuspended = predicate;
  }

  /**
   * Main animation loop function - the heart of HDR 3D rendering
   *
   * This function is called ~60 times per second (depending on display refresh rate)
   * and handles the complete HDR render pipeline:
   *
   * 1. Measure the previous frame's own cost and update the slow-frame streak
   *    (for pacing)
   * 2. Performance measurement begins (`frame-start`)
   * 3. Record the frame for adaptive DPR — unless the context is lost or
   *    another owner is driving the pipeline, in which case this frame does no
   *    GPU work of its own and must not be recorded
   * 4. Schedule next frame — immediately via requestAnimationFrame, or
   *    after a bounded cooldown once consecutive frames have been
   *    pathologically slow (see `scheduleNextFrame()` below and the class
   *    JSDoc)
   * 5. Update camera controls (handle user input, damping, constraints)
   * 6. Render through HDR post-processing pipeline (scene → bloom → tone mapping)
   * 7. Performance measurement ends (`frame-end`)
   *
   * Uses arrow function to maintain 'this' context when passed as callback.
   * Early return prevents unnecessary work when animation is paused.
   */
  private animate = (): void => {
    // Early exit if animation is paused - prevents unnecessary GPU work
    if (!this.isAnimating) return;

    // Measure the PREVIOUS frame's own cost for the pacing decision.
    //
    // The measurement is the frame PERIOD minus whatever cooldown we
    // ourselves inserted before this frame — not `end - start` of the JS
    // body. In the wedge this fixes, the ~1 s is non-JS main-thread /
    // compositor time: the body's own span reads ~2 ms, so a body-span
    // measurement would never fire. Subtracting our own gap is what stops
    // the measurement chasing its own tail — otherwise the period is
    // cost + cooldown, which stays above the threshold forever and
    // pacing would latch on for the rest of the session.
    //
    // Only a STREAK of slow frames paces (see PACING_SLOW_FRAME_STREAK): the
    // streak is advanced here, where the cost is measured, and read by
    // nextFramePacingDelayMs(). It stays untouched when there was nothing to
    // measure (the first frame of a run), so the rest before a resume cannot
    // become a streak's first link; startAnimation() clears it outright.
    const frameStart = performance.now();
    if (this.lastFrameStartTime !== null) {
      this.lastFrameCostMs = Math.max(
        0,
        frameStart - this.lastFrameStartTime - this.appliedPacingDelayMs
      );
      this.consecutiveSlowFrames =
        this.lastFrameCostMs > config.animation.pacing.slowFrameMs
          ? this.consecutiveSlowFrames + 1
          : 0;
    }
    this.lastFrameStartTime = frameStart;

    // Begin frame timing measurement for performance analysis.
    // Emits on the event bus so subscribers (e.g., the
    // PerformanceMonitor UI panel) can record the start timestamp
    // without animation-controller importing UI code directly.
    eventBus.emit('frame-start', {});

    // Record frame for adaptive DPR - tracks FPS and adjusts pixel
    // ratio. Skipped while the rendering context is lost AND while the
    // render-skip predicate is on: both kinds of frame do no GPU work of
    // their own, so their "speed" would drive bogus scale-ups and
    // falsely settle U-shape probes.
    //
    // A PACED frame is recorded on the RAW clock like any other: the achieved
    // frame rate really is lower, and the FPS readout must not lie. A steady
    // paced cadence is absorbed by the stall detector's outlier test
    // (`rendering/adaptive-dpr/stall-detector.ts` compares against the recent
    // MEDIAN, not an absolute threshold), so it does not manufacture gap
    // resets; and the isolated-hiccup interactions — a paced interval read off
    // a freshly cleared window as a collapsed frame rate, or a
    // just-under-`gapResetMs` frame pushed just over it — cannot arise at all,
    // because pacing requires a STREAK (see PACING_SLOW_FRAME_STREAK) and an
    // isolated slow frame is therefore never paced.
    if (this.adaptiveDPRManager && !this.isContextLost?.() && !this.shouldSkipRender?.()) {
      this.adaptiveDPRManager.recordFrame(performance.now());
    }

    // Schedule the next frame - requestAnimationFrame syncs with display
    // refresh (smooth 60fps on most displays, 120fps on high-refresh
    // monitors), with a pacing cooldown first once consecutive frames have
    // been pathologically slow. Scheduled BEFORE the work below on purpose:
    // the loop then survives an exception thrown by controls, a per-frame
    // callback, or render.
    this.scheduleNextFrame();

    // Update camera controls - processes mouse/touch input and applies damping
    // This must happen before rendering to reflect user interactions
    this.controls.update();

    // Call all registered per-frame callbacks (e.g., dynamic clipping, dimension animation)
    for (const entry of this.perFrameCallbacks.values()) {
      entry.callback();
    }

    // Skip GPU rendering while the WebGL context is lost. The
    // post-processing render() would otherwise issue draw calls
    // against a dead context (noisy GL errors, driver-specific
    // exceptions on some platforms). Controls and per-frame callbacks
    // already ran above so user input stays responsive while the
    // browser drives recovery.
    //
    // The render-skip predicate joins the same early return: an
    // offline capture owns the pipeline for its whole run, so the
    // loop's render would be discarded work drawn between the
    // capture's own passes.
    if (this.isContextLost?.() || this.shouldSkipRender?.()) {
      eventBus.emit('frame-end', {});
      return;
    }

    // Render through HDR post-processing pipeline
    // This executes the complete chain: Scene → HDR buffer → Bloom → Tone mapping → Display
    // Includes vertex shaders, fragment shaders, HDR buffers, bloom blur, ACES tone mapping
    this.postProcessing.render();

    // End frame timing — pair with the frame-start emit above. The
    // PerformanceMonitor UI panel subscribes to both events when
    // visible and feeds them into stats.js for FPS / frame-time
    // readouts.
    eventBus.emit('frame-end', {});
  };

  /**
   * Schedule the next loop iteration, inserting a pacing cooldown once
   * `PACING_SLOW_FRAME_STREAK` consecutive frames have been pathologically
   * slow.
   *
   * With no cooldown this is byte-for-byte the historical behaviour — a
   * bare `requestAnimationFrame(this.animate)`. With one, the rAF is armed
   * from a chain of two `setTimeout`s so the main thread has an actual gap
   * in which the browser can deliver a worker message, a CDP evaluate, or a
   * network callback. Frames are only ever DELAYED here, never dropped.
   */
  private scheduleNextFrame(): void {
    const cooldownMs = this.nextFramePacingDelayMs();

    if (cooldownMs <= 0) {
      this.appliedPacingDelayMs = 0;
      this.animationId = requestAnimationFrame(this.animate);
      return;
    }

    this.appliedPacingDelayMs = cooldownMs;
    // A timer armed HERE would be measured from the frame's start, and the
    // expensive part of a slow frame is browser rendering work that runs
    // after this callback returns but inside the same main-thread task — so
    // it would already be overdue when the thread frees and would insert no
    // gap at all. The zero-delay hop therefore runs at the first event-loop
    // turn AFTER that work, and only then is the real cooldown armed, so the
    // gap is genuine.
    //
    // The cost measurement subtracts the NOMINAL `cooldownMs`, but the real
    // gap is the hop's own latency plus `max(cooldownMs, when the main thread
    // next frees)` plus the post-cooldown rAF alignment (up to one vsync,
    // likewise not subtracted). So while pacing is engaged the next frame's
    // cost is OVER-estimated, and the over-estimate is NOT bounded by a few
    // milliseconds: a cooldown that comes due while a long foreign task is
    // running inflates it by all of that foreign work. (The hop itself is
    // armed from a rAF callback, so timer nesting restarts every frame and the
    // browser's 4 ms nesting clamp is never reached — it contributes nothing.)
    //
    // Unbounded above, but always in the safe direction: a slow frame is never
    // under-measured into the fast path. It also cannot make pacing ENGAGE
    // spuriously — the two frames that build the streak are by definition
    // unpaced, so neither carries a hop or a cooldown and neither is biased.
    // Its steady-state effect is that a session sharing the main thread with
    // sustained foreign work STAYS paced, which is the intended behaviour:
    // yielding to that work is the whole point.
    this.pacingTimeout = setTimeout(() => {
      // The loop may have been stopped (idle pause, tab hide, dispose)
      // while the cooldown was in flight — stopAnimation() clears these
      // timeouts, but a fire that already landed must not resurrect it.
      if (!this.isAnimating) {
        this.pacingTimeout = null;
        return;
      }
      // Re-read the suspend predicate. It was last read at the top of the
      // frame this cooldown was armed for, and a capture can have started
      // since — honouring the stale answer would freeze up to `maxCooldownMs`
      // of duplicate frame into the head of a real-time recording of an
      // already-slow scene. Drop the cooldown instead, and clear the applied
      // delay so the next cost measurement subtracts nothing.
      if (this.pacingSuspended()) {
        this.pacingTimeout = null;
        this.appliedPacingDelayMs = 0;
        this.animationId = requestAnimationFrame(this.animate);
        return;
      }
      this.pacingTimeout = setTimeout(() => {
        this.pacingTimeout = null;
        if (!this.isAnimating) return;
        this.animationId = requestAnimationFrame(this.animate);
      }, cooldownMs);
    }, 0);
  }

  /**
   * Whether pacing is currently suspended, with a throwing predicate read as
   * "not suspended".
   *
   * The try/catch is load-bearing because `scheduleNextFrame()` is the loop's
   * ONLY re-arm point: a throw that escaped it would leave nothing armed while
   * `isAnimating` stayed true, so `startAnimation()` early-returns forever and
   * no `requestRender()` can recover — an unrecoverable freeze. Today's wiring
   * cannot reach that: `RecordingPanel.isCurrentlyRecording()` is a plain flag
   * read, and the session it reads survives the panel's own `dispose()`. This
   * is therefore a guard on the INJECTION POINT rather than on a known
   * thrower — whatever gets wired here next inherits it, and the trade is a
   * paced frame during a capture against the viewer freezing for the rest of
   * the session.
   *
   * Shared by the two places the answer is needed — when the cooldown is
   * armed, and again in the hop callback before the cooldown is committed —
   * so both read it under the same guarantee.
   */
  private pacingSuspended(): boolean {
    try {
      return this.isPacingSuspended?.() === true;
    } catch {
      // Intentionally ignored — see above.
      return false;
    }
  }

  /**
   * Cooldown (ms) to insert before the next frame; 0 means "re-arm
   * requestAnimationFrame immediately", the untouched fast path.
   *
   * @returns Milliseconds to wait before the next `requestAnimationFrame`
   */
  private nextFramePacingDelayMs(): number {
    const pacing = config.animation.pacing;
    if (!pacing.enabled) return 0;

    // Recording owns the frame cadence for the whole capture. Read again in
    // the hop callback, since a capture can start mid-cooldown — see
    // `scheduleNextFrame()`.
    if (this.pacingSuspended()) return 0;

    // Every healthy frame rate lands here, and so does an ISOLATED slow
    // frame: the streak (advanced in animate(), where the cost is measured)
    // is the whole of the threshold test, deliberately not repeated here.
    // `lastFrameCostMs` is a frame PERIOD, so a foreign main-thread task of
    // that size is charged to the loop — one of them cannot pace anything,
    // but a sustained run of them can, which is what we want (see the
    // `slowFrameMs` comment in config/sections/animation/data.ts).
    if (this.consecutiveSlowFrames < PACING_SLOW_FRAME_STREAK) return 0;

    // A bounded fraction of the cost (see PACING_COOLDOWN_FRACTION). A
    // cooldown that rounds to 0 falls through the `cooldownMs <= 0` fast
    // path above, which is right: at a 0.25 fraction and a cost past the
    // threshold that only happens if `maxCooldownMs` is itself 0, i.e.
    // pacing has been configured off.
    return Math.min(
      pacing.maxCooldownMs,
      Math.round(this.lastFrameCostMs * PACING_COOLDOWN_FRACTION)
    );
  }

  /**
   * Check if any features require continuous animation
   * @returns True if animation should continue regardless of user interaction
   */
  private shouldContinueAnimating(): boolean {
    // Check if auto-rotate can move the camera in the active control mode
    const autoRotate = this.controls.isAutoRotateActive();

    // Check if any post-processing effects need continuous updates
    const hasEffects = this.postProcessing.needsContinuousAnimation();

    // Check if any continuous per-frame callbacks are active (e.g., turntable recording, dimension animation)
    // On-demand callbacks (continuous: false) like dynamic-clipping and scale-bar don't prevent idle pause
    const hasContinuousCallbacks = [...this.perFrameCallbacks.values()].some(
      (entry) => entry.continuous
    );

    return autoRotate || hasEffects || hasContinuousCallbacks;
  }

  /**
   * Handle idle timeout - only stop if no continuous effects are active
   */
  private handleIdleTimeout = (): void => {
    // Check if we should continue animating due to effects or auto-rotate
    if (this.shouldContinueAnimating()) {
      // Continuous effects are active, schedule another check
      this.idleTimeout = setTimeout(this.handleIdleTimeout, config.animation.idleTimeoutMs);
    } else {
      // No continuous effects, safe to stop animation
      this.stopAnimation();

      // Idle restore: the static frame the user is about to study
      // should be at full native sharpness — reduced DPR only ever
      // traded quality for interaction smoothness. This lives ONLY in
      // the idle path (never in stopAnimation itself, which also runs
      // on tab-hide and dispose where rendering would be wrong).
      // prepareIdleFrame() returns true only when the DPR actually
      // changed; the resize clears the canvas, so exactly then we
      // render ONE frame directly — NOT via startAnimation(), which
      // would re-arm the idle timer and feed native-DPR frames back
      // into the FPS evaluator.
      //
      // The render-skip predicate is checked here too — this is the
      // loop's OTHER render call site, and the predicate's claim is
      // "nobody but the pipeline's current owner may draw", not "the
      // animate() path may not draw". Today it is redundant (a capture
      // disables adaptive DPR, so isActive() is already false, and the
      // idle-restore predicate is off for the whole capture), but the
      // guard that makes it redundant lives in another file: drop
      // `disableDPR` from the capture's saveRecordingState and this
      // would paint a native-DPR frame through the capture scrim,
      // possibly inside the raw-HDR window. It must come BEFORE
      // prepareIdleFrame(), which RESIZES on its way to returning true
      // — skipping the render after that resize would leave the canvas
      // cleared with nothing to repaint it.
      if (
        this.adaptiveDPRManager?.isActive?.() &&
        this.canRestoreAtIdle?.() !== false &&
        !this.isContextLost?.() &&
        !this.shouldSkipRender?.() &&
        this.adaptiveDPRManager.prepareIdleFrame?.()
      ) {
        this.postProcessing.render();
      }
    }
  };

  /**
   * Start animation loop and reset idle timer for power efficiency
   *
   * This method is called whenever user interaction is detected:
   * - Mouse movement over canvas
   * - Camera control events (start, change)
   * - Keyboard input
   * - Touch events
   * - When continuous effects are enabled (noise, auto-rotate)
   *
   * The idle timer automatically pauses rendering after inactivity to:
   * - Reduce CPU/GPU usage when scene is static
   * - Improve battery life on mobile devices
   * - Lower thermal impact on laptops
   * - Maintain 0% CPU usage when user is not interacting
   *
   * Continuous effects (noise, auto-rotate) will keep animation running.
   *
   * Uses arrow function to maintain 'this' context when used as event handler.
   */
  startAnimation = (): void => {
    // Only start if not already running - prevents duplicate loops
    if (!this.isAnimating) {
      this.isAnimating = true;
      // Resuming from a rest: let the adaptive DPR manager snap back to
      // its remembered operating DPR in one step (stopped→running edge
      // only — this must not fire on every interaction poke).
      this.adaptiveDPRManager?.notifyResumed?.();
      // Forget the frame-cost measurement across the rest. The clock kept
      // running while the loop was stopped, so the gap between the last
      // frame before the pause and the first frame after it is idle time —
      // a two-second rest or a tab-hide would otherwise read as a
      // two-second frame and pace the first frame back for nothing.
      this.lastFrameStartTime = null;
      this.appliedPacingDelayMs = 0;
      this.lastFrameCostMs = 0;
      this.consecutiveSlowFrames = 0;
      // Kick off the first frame - subsequent frames are scheduled by animate()
      this.animate();
    }

    // Reset the idle timeout - this is called on every user interaction
    // Clear any existing timeout to prevent premature stopping
    if (this.idleTimeout !== null) {
      clearTimeout(this.idleTimeout);
    }

    // Set new timeout to check for idle - will continue if continuous effects are active
    // This is the key power-saving optimization for static scenes
    this.idleTimeout = setTimeout(this.handleIdleTimeout, config.animation.idleTimeoutMs);
  };

  /**
   * Stop animation loop and clean up timers
   *
   * This method halts all rendering activity to conserve resources:
   * - Sets flag to prevent further animate() calls
   * - Cancels pending requestAnimationFrame to stop browser scheduling
   * - Clears idle timeout to prevent memory leaks
   *
   * Called automatically after idle timeout (when no continuous effects)
   * or manually for cleanup. Scene remains visible but static until
   * next user interaction or continuous effect activation.
   */
  stopAnimation = (): void => {
    // Set flag to prevent animate() from continuing the loop
    this.isAnimating = false;

    // The FPS window, hysteresis streak, and any in-flight probe are
    // about to go stale across the pause — clear them (session state
    // only; learned floors survive). Method-level optional chaining is
    // deliberate: tests inject bare {recordFrame} manager mocks, and
    // this also runs from dispose() after the manager may be gone.
    this.adaptiveDPRManager?.notifyPaused?.();

    // Cancel any pending requestAnimationFrame call
    // This ensures no more frames are scheduled by the browser
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }

    // Cancel a pending frame-pacing cooldown. Its callback also re-checks
    // `isAnimating`, but leaving the timer armed would keep a stopped
    // controller holding a live timer (and, under fake timers, fire a
    // needless wake).
    if (this.pacingTimeout !== null) {
      clearTimeout(this.pacingTimeout);
      this.pacingTimeout = null;
    }

    // Clear the idle timeout to prevent memory leaks
    if (this.idleTimeout !== null) {
      clearTimeout(this.idleTimeout);
      this.idleTimeout = null;
    }
  };

  /**
   * Get current animation loop state.
   *
   * @returns true if animation loop is running, false if paused
   */
  get isActive(): boolean {
    return this.isAnimating;
  }

  /**
   * Get performance monitor for FPS and timing metrics.
   *
   * Stop animation loop and clean up resources.
   *
   * Stops rendering and cancels timers. The PerformanceMonitor UI
   * panel lives at LuxarApp; this controller emits `frame-start` /
   * `frame-end` on the event bus per frame, which is what the panel
   * listens to.
   *
   * After calling dispose(), the animation controller cannot be reused.
   */
  dispose(): void {
    this.stopAnimation();
    this.perFrameCallbacks.clear();
  }
}
