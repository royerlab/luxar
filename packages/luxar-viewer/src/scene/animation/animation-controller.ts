// Animation loop management for the Luxar scene player
//
// This module handles the core animation loop that drives 3D rendering:
// - The per-frame work (`tick()`): controls, per-frame callbacks, render
// - Scheduling via RafDriver: requestAnimationFrame plus frame pacing
// - Intelligent pause/resume system to conserve CPU/GPU when idle
// - Performance monitoring integration for real-time metrics
// - Proper cleanup and resource management

import { ControlsManager } from '../../controls/controls-manager';
import { config } from '../../config';
import { PostProcessingManager } from '../../rendering';
import { AdaptiveDPRManager } from '../../rendering/adaptive-dpr-manager';
import { eventBus } from '../../utils/cross-layer/event-bus';
import { log, Modules } from '../../utils/log';
import { RafDriver } from './raf-driver';

/**
 * The phases of a frame's per-frame callbacks, run in this order after
 * `controls.update()`:
 *
 * - `camera`: callbacks that MOVE the camera (a flight, the recording
 *   turntable, the offline capture's orbit step). They run first so that
 *   everything reading the camera this frame reads its final pose.
 * - `view`: callbacks that DERIVE state from the camera and the scene
 *   (dynamic clipping planes, depth sort, projected density, LOD selection,
 *   dimension playback). The default.
 * - `pre-render`: work that needs the frame's final view state (the
 *   scene-captured environment).
 * - `ui`: read-only overlays (scale bar, Layers-panel LOD status).
 *
 * Within a phase callbacks run in registration order. Before phases existed
 * everything ran in registration order alone, so a camera writer registered
 * after the view callbacks (every flight is: it registers when it starts)
 * moved the camera after clipping, depth sort and LOD had already read it,
 * and every flight frame rendered with the previous frame's near/far.
 */
export type FramePhase = 'camera' | 'view' | 'pre-render' | 'ui';

/** Phase run order. */
const FRAME_PHASES: readonly FramePhase[] = ['camera', 'view', 'pre-render', 'ui'];

/** A registered per-frame callback. */
interface PerFrameEntry {
  callback: () => void;
  continuous: boolean;
}

/**
 * AnimationController manages the main rendering loop and performance optimization
 *
 * Key Features:
 * - RequestAnimationFrame loop for browser-optimized rendering
 * - Frame pacing for pathologically slow frames (see RafDriver)
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
 * Scheduling vs work:
 * The controller owns WHAT a frame does ({@link AnimationController.tick}) and
 * when the loop may go idle; {@link RafDriver} owns WHEN frames run
 * (requestAnimationFrame, and frame pacing for pathologically slow frames,
 * #1724). Another driver can run the same `tick()`.
 */
export class AnimationController {
  /** Timeout ID for auto-pause functionality */
  private idleTimeout: ReturnType<typeof setTimeout> | null = null;

  /**
   * Per-frame callbacks, one insertion-ordered map per phase (keyed by ID for
   * safe add/remove). Iterating the live maps, rather than a sorted snapshot,
   * keeps the long-standing semantics for a callback that registers or
   * removes another mid-frame: a removed one that has not run yet is skipped,
   * one added to the phase being run (or a later one) runs this frame.
   */
  private readonly callbacksByPhase: Record<FramePhase, Map<string, PerFrameEntry>> = {
    camera: new Map(),
    view: new Map(),
    'pre-render': new Map(),
    ui: new Map(),
  };

  /** Phase each registered callback id lives in. */
  private readonly phaseOf = new Map<string, FramePhase>();

  /** Ids of per-frame callbacks whose failure has already been logged. */
  private readonly failedCallbackIds = new Set<string>();

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

  /** Schedules frames; each runs {@link tick}. */
  private readonly driver = new RafDriver(() => this.tick());

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
   * The callback is executed after controls.update() but before rendering,
   * in its phase's turn (camera → view → pre-render → ui), and within a phase
   * in registration order.
   *
   * @param id - Unique identifier for this callback (for later removal)
   * @param callback - Function to call each frame
   * @param options - Options controlling callback behavior
   * @param options.continuous - If true, this callback prevents the animation loop from
   *   auto-pausing due to idle timeout. Use for callbacks that need every frame (e.g.,
   *   dimension animation, recording). Default: false (on-demand callbacks that only run
   *   when animation is active but don't prevent pausing).
   * @param options.phase - Which {@link FramePhase} the callback runs in.
   *   `'camera'` for callbacks that move the camera, `'pre-render'` for work
   *   that needs the frame's final view state, `'ui'` for read-only overlays.
   *   Default: `'view'`.
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
  addPerFrameCallback(
    id: string,
    callback: () => void,
    options?: { continuous?: boolean; phase?: FramePhase }
  ): void {
    const phase = options?.phase ?? 'view';
    const previous = this.phaseOf.get(id);
    // Re-registering in the SAME phase keeps the callback's position (the
    // map's own overwrite rule); a phase change moves it to the end of its
    // new phase.
    if (previous !== undefined && previous !== phase) this.callbacksByPhase[previous].delete(id);
    this.callbacksByPhase[phase].set(id, { callback, continuous: options?.continuous ?? false });
    this.phaseOf.set(id, phase);
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
    // A later registration under the same id is a new callback: report its
    // first failure too.
    this.failedCallbackIds.delete(id);
    const phase = this.phaseOf.get(id);
    if (phase === undefined) return false;
    this.phaseOf.delete(id);
    return this.callbacksByPhase[phase].delete(id);
  }

  /**
   * Check if a per-frame callback with the given ID exists.
   *
   * @param id - Identifier to check
   * @returns true if callback exists
   */
  hasPerFrameCallback(id: string): boolean {
    return this.phaseOf.has(id);
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
   * Inject a predicate consulted before the idle-pause ceiling-DPR
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
    this.driver.setPacingSuspendPredicate(predicate);
  }

  /**
   * One frame of work — the heart of HDR 3D rendering.
   *
   * Run once per frame by the driver, which has already armed the next frame
   * (so the loop survives an exception thrown here):
   *
   * 1. Performance measurement begins (`frame-start`)
   * 2. Record the frame for adaptive DPR — unless the context is lost or
   *    another owner is driving the pipeline, in which case this frame does no
   *    GPU work of its own and must not be recorded
   * 3. Update camera controls (handle user input, damping, constraints)
   * 4. Run the per-frame callbacks
   * 5. Render through HDR post-processing pipeline (scene → bloom → tone mapping)
   * 6. Performance measurement ends (`frame-end`)
   *
   * Public so a driver other than the window's requestAnimationFrame can run
   * frames. It does not check whether the loop is running: that is the
   * driver's decision.
   */
  tick(): void {
    // Begin frame timing measurement for performance analysis.
    // Emits on the event bus so subscribers (e.g., the
    // PerformanceMonitor UI panel) can record the start timestamp
    // without animation-controller importing UI code directly.
    eventBus.emit('frame-start', {});
    try {
      this.frameWork();
    } finally {
      // End frame timing — pair with the frame-start emit above, even when
      // the work threw: the PerformanceMonitor UI panel subscribes to both
      // events when visible and feeds them into stats.js for FPS /
      // frame-time readouts, and an unpaired start corrupts its timing.
      eventBus.emit('frame-end', {});
    }
  }

  /** {@link tick} between its frame-start and frame-end. */
  private frameWork(): void {
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
    // because pacing requires a STREAK (see PACING_SLOW_FRAME_STREAK in
    // raf-driver.ts) and an isolated slow frame is therefore never paced.
    if (this.adaptiveDPRManager && !this.isContextLost?.() && !this.shouldSkipRender?.()) {
      this.adaptiveDPRManager.recordFrame(performance.now());
    }

    // Update camera controls - processes mouse/touch input and applies damping
    // This must happen before rendering to reflect user interactions
    this.controls.update();

    // Call all registered per-frame callbacks (e.g., dynamic clipping, dimension animation)
    this.runPerFrameCallbacks();

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
    if (this.isContextLost?.() || this.shouldSkipRender?.()) return;

    // Render through HDR post-processing pipeline
    // This executes the complete chain: Scene → HDR buffer → Bloom → Tone mapping → Display
    // Includes vertex shaders, fragment shaders, HDR buffers, bloom blur, ACES tone mapping
    this.postProcessing.render();
  }

  /**
   * Run every per-frame callback, each isolated from the others.
   *
   * A callback that throws is logged (once per id, not once per frame: a
   * callback that fails every frame would otherwise flood the console at the
   * frame rate) and the remaining callbacks and the render still run.
   * Without the isolation one broken subsystem skipped every callback
   * registered after it, and the render, on every frame, freezing the view
   * while the loop kept spinning.
   */
  private runPerFrameCallbacks(): void {
    for (const phase of FRAME_PHASES) {
      for (const [id, entry] of this.callbacksByPhase[phase]) this.runCallback(id, entry);
    }
  }

  /** Run one callback, logging its first failure (see {@link runPerFrameCallbacks}). */
  private runCallback(id: string, entry: PerFrameEntry): void {
    try {
      entry.callback();
    } catch (error) {
      if (!this.failedCallbackIds.has(id)) {
        this.failedCallbackIds.add(id);
        log.error(
          Modules.ANIMATION,
          `Per-frame callback '${id}' threw; skipping it this frame`,
          error
        );
      }
    }
  }

  /**
   * Check if any features require continuous animation
   * @returns True if animation should continue regardless of user interaction
   */
  private shouldContinueAnimating(): boolean {
    // Check if auto-rotate or the auto-dolly can move the camera in the
    // active control mode (the dolly is also alive in ortho, where the
    // turntable is not).
    const autoCamera = this.controls.isAutoRotateActive() || this.controls.isAutoDollyActive();

    // A pointer gesture in progress: button or finger down since `start`, no
    // `end` yet. Without this, a press held still for idleTimeoutMs paused the
    // loop, and the drag that followed fed rotate/pan/zoom deltas into the
    // controls that no update() ever applied — the camera sat frozen until
    // some later event happened to call startAnimation(). Reproduced with a
    // mouse (press, hold 2 s, drag → no rotation) and with a finger alike.
    const gesture = this.controls.isGestureActive();

    // Check if any post-processing effects need continuous updates
    const hasEffects = this.postProcessing.needsContinuousAnimation();

    // Check if any continuous per-frame callbacks are active (e.g., turntable recording, dimension animation)
    // On-demand callbacks (continuous: false) like dynamic-clipping and scale-bar don't prevent idle pause
    const hasContinuousCallbacks = FRAME_PHASES.some((phase) =>
      [...this.callbacksByPhase[phase].values()].some((entry) => entry.continuous)
    );

    return autoCamera || gesture || hasEffects || hasContinuousCallbacks;
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
      // would re-arm the idle timer and feed ceiling-DPR frames back
      // into the FPS evaluator.
      //
      // The render-skip predicate is checked here too — this is the
      // loop's OTHER render call site, and the predicate's claim is
      // "nobody but the pipeline's current owner may draw", not "the
      // tick() path may not draw". Today it is redundant (a capture
      // disables adaptive DPR, so isActive() is already false, and the
      // idle-restore predicate is off for the whole capture), but the
      // guard that makes it redundant lives in another file: drop
      // `captureDPR` from the capture's saveRecordingState and this
      // would paint a ceiling-DPR frame through the capture scrim,
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
    if (!this.driver.isRunning) {
      // Resuming from a rest: let the adaptive DPR manager snap back to
      // its remembered operating DPR in one step (stopped→running edge
      // only — this must not fire on every interaction poke).
      this.adaptiveDPRManager?.notifyResumed?.();
      // Runs the first frame synchronously; later frames are scheduled by
      // the driver.
      this.driver.start();
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
   * - Stops the driver (no further frames; pending rAF and pacing cancelled)
   * - Clears idle timeout to prevent memory leaks
   *
   * Called automatically after idle timeout (when no continuous effects)
   * or manually for cleanup. Scene remains visible but static until
   * next user interaction or continuous effect activation.
   */
  stopAnimation = (): void => {
    // Stop the driver first: no further frame runs, and the pending
    // requestAnimationFrame and any pacing cooldown are cancelled.
    this.driver.stop();

    // The FPS window, hysteresis streak, and any in-flight probe are
    // about to go stale across the pause — clear them (session state
    // only; learned floors survive). Method-level optional chaining is
    // deliberate: tests inject bare {recordFrame} manager mocks, and
    // this also runs from dispose() after the manager may be gone.
    this.adaptiveDPRManager?.notifyPaused?.();

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
    return this.driver.isRunning;
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
    for (const phase of FRAME_PHASES) this.callbacksByPhase[phase].clear();
    this.phaseOf.clear();
  }
}
