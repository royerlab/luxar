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
import { perfCounters } from '../../profiling/perf-counters';
import { RafDriver } from './raf-driver';
import type { RenderAudit } from './render-audit';
import { ViewSignature, type ViewSignatureSource } from './view-signature';

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

// Perf counters (always on; one typed-array store each). `render.count` is
// every scene render this controller issues and equals the sum of the
// `render.byReason.*` counters; `render.once` counts the renderOnce() frames
// among them (which are also loop ticks). `render.ticks` counts every loop
// tick, and `render.skippedTicks` the ticks that drew nothing because nothing
// changed (render-on-change) — ticks skipped for a lost context or an
// offline capture are in neither `render.count` nor `render.skippedTicks`.
const S_RENDER_COUNT = perfCounters.slot('render.count');
const S_RENDER_TICKS = perfCounters.slot('render.ticks');
const S_RENDER_SKIPPED_TICKS = perfCounters.slot('render.skippedTicks');
const S_RENDER_ONCE = perfCounters.slot('render.once');
const S_ADAPTIVE_DPR_SAMPLES = perfCounters.slot('adaptiveDpr.samples');

/**
 * Why a frame was rendered — the `render.byReason.<reason>` counter it lands
 * in. Exactly one per render, taken in this precedence:
 *
 * - `wake` — {@link AnimationController.startAnimation} (any input or state
 *   handler that wakes the loop), `event` — {@link AnimationController.requestRender}
 *   (a commit, a resize, a sort landing …), `once` — {@link AnimationController.renderOnce};
 *   whichever marked the frame dirty FIRST since the last render;
 * - `cb:<id>` — a per-frame callback returned `true` (it changed what is drawn);
 * - `camera` — the view signature (camera matrices, drawing-buffer size)
 *   differs from the last rendered frame's;
 * - `continuous` — something needs every frame drawn (cinematic detector
 *   noise, a real-time recording, `?renderAlways`);
 * - `idleRestore` — the idle pause's one ceiling-DPR frame;
 * - `audit` — a `?renderAudit` frame the scheduler would have skipped.
 */
export type RenderReason =
  'wake' | 'event' | 'once' | 'camera' | 'continuous' | 'idleRestore' | 'audit' | `cb:${string}`;

/** Slot cache for `render.byReason.<reason>` (resolved once per reason). */
const reasonSlots = new Map<string, number>();
function reasonSlot(reason: RenderReason): number {
  let slot = reasonSlots.get(reason);
  if (slot === undefined) {
    slot = perfCounters.slot(`render.byReason.${reason}`);
    reasonSlots.set(reason, slot);
  }
  return slot;
}

/**
 * A per-frame callback. Returning `true` says "I changed what the next render
 * draws" and makes this tick render; `false` says it changed nothing drawn.
 * The return is REQUIRED (not `void | boolean`) so every registration has to
 * state which it is: a callback that mutated drawn state and returned nothing
 * would leave a stale frame on screen until something else redraws, and the
 * compiler now catches the one that forgets. Camera motion needs no `true`
 * (the view signature catches it), nor does a DOM-only overlay.
 */
export type PerFrameCallback = () => boolean;

/** A registered per-frame callback. */
export interface PerFrameEntry {
  callback: PerFrameCallback;
  continuous: boolean;
  /** Draw every tick while registered (a real-time recording). */
  renderEveryFrame: boolean;
}

/** Options of {@link AnimationController.addPerFrameCallback}. */
export interface PerFrameCallbackOptions {
  continuous?: boolean;
  phase?: FramePhase;
  renderEveryFrame?: boolean;
}

/** The stored entry for a registration (defaults applied). */
function perFrameEntry(
  callback: PerFrameCallback,
  options: PerFrameCallbackOptions | undefined
): PerFrameEntry {
  return {
    callback,
    continuous: options?.continuous ?? false,
    renderEveryFrame: options?.renderEveryFrame ?? false,
  };
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
   * Render only when something changed (`config.animation.renderOnChange`;
   * `?renderAlways` turns it off, see {@link setRenderOnChange}). Off = the
   * historical loop: every tick renders.
   */
  private renderOnChange: boolean = config.animation.renderOnChange;

  /**
   * Why the next tick must render, when an explicit request marked it dirty
   * (`wake` / `event` / `once`); null while clean. Cleared just BEFORE the
   * render, so a request made during the render (an `onAfterRender` hook, a
   * sort landing) dirties the NEXT tick instead of being swallowed.
   */
  private dirtyReason: RenderReason | null = null;

  /** Camera + drawing-buffer signature of the last rendered frame (null until wired). */
  private viewSignature: ViewSignature | null = null;

  /** Number of registered callbacks with `renderEveryFrame` (checked per tick). */
  private renderEveryFrameCount = 0;

  /** Whether the previous tick rendered (adaptive-DPR sampling contiguity). */
  private previousTickRendered = false;

  /** `?renderAudit` (debug only): render every tick and verify the skips. */
  private audit: RenderAudit | null = null;

  /** Loop ticks since construction (labels the audit log). */
  private tickCount = 0;

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
   * Multiple callbacks can be registered simultaneously.
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
   * @param options.renderEveryFrame - Draw EVERY tick while this callback is
   *   registered, changed or not. For a consumer of the canvas itself — the
   *   real-time recording, whose MediaRecorder films what the loop paints —
   *   not for state changes (return `true` from the callback for those).
   *   Default: false.
   *
   * The callback's return value is the render-on-change contract: `true` when
   * it changed what the next render draws (the tick then renders), `false`
   * when it did not. Camera motion needs no `true`: the view signature
   * catches it.
   *
   * @example
   * ```typescript
   * // On-demand callback: runs when animating but doesn't prevent idle pause;
   * // returns whether it changed what is drawn
   * animController.addPerFrameCallback('dynamic-clipping', () =>
   *   sceneManager.updateDynamicClippingPlanes()
   * );
   *
   * // Continuous callback: keeps animation loop alive
   * animController.addPerFrameCallback('dimension-animation', () =>
   *   animationManager.onFrame()
   * , { continuous: true });
   * ```
   */
  addPerFrameCallback(
    id: string,
    callback: PerFrameCallback,
    options?: PerFrameCallbackOptions
  ): void {
    const phase = options?.phase ?? 'view';
    this.detachForReregistration(id, phase);
    const entry = perFrameEntry(callback, options);
    if (entry.renderEveryFrame) this.renderEveryFrameCount++;
    this.callbacksByPhase[phase].set(id, entry);
    this.phaseOf.set(id, phase);
  }

  /**
   * Undo an existing registration's bookkeeping before `id` is registered
   * again in `phase`. Re-registering in the SAME phase keeps the callback's
   * position (the map's own overwrite rule); a phase change moves it to the
   * end of its new phase.
   */
  private detachForReregistration(id: string, phase: FramePhase): void {
    const previous = this.phaseOf.get(id);
    if (previous === undefined) return;
    if (this.callbacksByPhase[previous].get(id)?.renderEveryFrame) this.renderEveryFrameCount--;
    if (previous !== phase) this.callbacksByPhase[previous].delete(id);
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
    if (this.callbacksByPhase[phase].get(id)?.renderEveryFrame) this.renderEveryFrameCount--;
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
   * Turn render-on-change on or off. On (the default, from
   * `config.animation.renderOnChange`): a tick renders only when something
   * changed — see {@link RenderReason} for what counts. Off (`?renderAlways`,
   * the kill switch): every tick renders, exactly the historical loop,
   * including adaptive-DPR sampling on every tick.
   *
   * Either way the loop keeps TICKING as before (controls, callbacks, the idle
   * timer); only the redundant re-render of an unchanged state is skipped.
   */
  setRenderOnChange(enabled: boolean): void {
    if (enabled === this.renderOnChange) return;
    this.renderOnChange = enabled;
    // A fresh start either way: no stale "last rendered" view, and the next
    // tick draws.
    this.viewSignature?.invalidate();
    this.previousTickRendered = false;
    this.markDirty('wake');
  }

  /** Whether render-on-change is active (false under `?renderAlways`). */
  get isRenderOnChange(): boolean {
    return this.renderOnChange;
  }

  /**
   * Where the loop reads the live camera and drawing buffer for the camera
   * half of the dirty check (see `view-signature.ts`). Without a source only
   * explicit requests, callbacks and continuous consumers make a tick render
   * — camera input still does through the controls' `change` event, which
   * wakes the loop. The init pipeline wires the scene manager's camera and
   * canvas. Pass null to drop it.
   */
  setViewSignatureSource(source: ViewSignatureSource | null): void {
    this.viewSignature = source ? new ViewSignature(source) : null;
  }

  /**
   * `?renderAudit` (debug only): render every tick and check, on each tick
   * the scheduler would have skipped, that the pixels did not change — see
   * `render-audit.ts`. Pass null to turn it off.
   */
  setRenderAudit(audit: RenderAudit | null): void {
    this.audit = audit;
  }

  /**
   * Keep the loop TICKING without asking for a redraw: controls, per-frame
   * callbacks and the idle timer run as for any wake, but the tick renders
   * only if something then turns out to have changed. For a subsystem that
   * polls per frame for work landing asynchronously (a lazy LOD level
   * loading): the landing itself requests the render.
   */
  requestTick(): void {
    this.resumeIfStopped();
    this.resetIdleTimer();
  }

  /**
   * Something that the next frame draws changed outside the loop (a geometry
   * commit, a resize, a sort result, a material edit): mark the next tick
   * dirty and wake the loop. `detail` names the source for the render audit's
   * log; the frame is tallied as `render.byReason.event`.
   */
  requestRender(detail: string): void {
    this.markDirty('event', detail);
    this.resumeIfStopped();
    this.resetIdleTimer();
  }

  /** Mark the next tick dirty (keeping the first reason since the last render). */
  private markDirty(reason: 'wake' | 'event' | 'once', detail?: string): void {
    this.dirtyReason ??= reason;
    this.audit?.noteRequest(`${this.tickCount}:${detail ?? reason}`);
  }

  /**
   * True while every tick must draw whether or not state changed: cinematic
   * detector noise animates per frame (`needsContinuousAnimation`, which is
   * also what keeps the loop from idling), and a registered `renderEveryFrame`
   * consumer films the canvas.
   */
  private mustRenderEveryTick(): boolean {
    return this.renderEveryFrameCount > 0 || this.postProcessing.needsContinuousAnimation();
  }

  /**
   * One frame of work — the heart of HDR 3D rendering.
   *
   * Run once per frame by the driver, which has already armed the next frame
   * (so the loop survives an exception thrown here):
   *
   * 1. Performance measurement begins (`frame-start`)
   * 2. Record the PREVIOUS frame for adaptive DPR — only when it rendered
   *    (see {@link recordAdaptiveDprSample}), and never while the context is
   *    lost or another owner is driving the pipeline
   * 3. Update camera controls (handle user input, damping, constraints)
   * 4. Run the per-frame callbacks
   * 5. Render through the HDR post-processing pipeline (scene → bloom → tone
   *    mapping) — under render-on-change only when something changed since
   *    the last rendered frame (see {@link RenderReason})
   * 6. Performance measurement ends (`frame-end`, `{ rendered }`)
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
    perfCounters.add(S_RENDER_TICKS);
    this.tickCount++;
    eventBus.emit('frame-start', {});
    let rendered = false;
    try {
      rendered = this.frameWork();
    } finally {
      // End frame timing — pair with the frame-start emit above, even when
      // the work threw: the PerformanceMonitor UI panel subscribes to both
      // events when visible and feeds them into stats.js for FPS /
      // frame-time readouts, and an unpaired start corrupts its timing.
      // `rendered` lets it count only frames that drew (a skipped tick is
      // not a frame the user saw).
      eventBus.emit('frame-end', { rendered });
    }
  }

  /**
   * {@link tick} between its frame-start and frame-end.
   *
   * @returns whether the tick rendered
   */
  private frameWork(): boolean {
    // Sampled only when a manager is wired: the loop's own clock read, taken
    // at the tick's start exactly as before render-on-change.
    const tickStart = this.adaptiveDPRManager ? performance.now() : 0;
    this.recordAdaptiveDprSample(tickStart);

    // Update camera controls - processes mouse/touch input and applies damping
    // This must happen before rendering to reflect user interactions
    this.controls.update();

    // Call all registered per-frame callbacks (e.g., dynamic clipping, dimension
    // animation); the id of the first that changed drawn state, if any.
    const changedBy = this.runPerFrameCallbacks();

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
    // capture's own passes. Whatever was dirty stays dirty.
    if (this.isGpuWorkBlocked()) {
      this.previousTickRendered = false;
      return false;
    }

    const reason = this.renderReason(changedBy);
    if (reason === null) return this.skipTick();
    this.noteStreamResumed(tickStart);
    this.renderFrame(reason);
    return true;
  }

  /**
   * True while the loop must issue no draw of its own: the rendering context
   * is lost, or another owner (an offline capture) is driving the pipeline.
   */
  private isGpuWorkBlocked(): boolean {
    return this.isContextLost?.() === true || this.shouldSkipRender?.() === true;
  }

  /**
   * First render after one or more skipped ticks: the frame-interval clock
   * still reads the last rendered tick, and the gap is idle time, not a slow
   * frame. Re-base it (no sample; nothing learned is forgotten).
   */
  private noteStreamResumed(tickStart: number): void {
    if (!this.renderOnChange || this.previousTickRendered) return;
    this.adaptiveDPRManager?.notifyStreamBreak?.(tickStart);
  }

  /**
   * Record the frame the PREVIOUS tick rendered for adaptive DPR. The sample
   * is the interval between two tick starts, which measures the earlier tick
   * — so it is a render-frame sample only when that tick rendered. A skipped
   * tick costs nothing, and its "speed" would drive bogus scale-ups and
   * falsely settle U-shape probes, exactly like a context-lost or
   * render-skipped frame (which are excluded for the same reason). Under
   * `?renderAlways` every tick renders and every tick records, as it always
   * did.
   *
   * A PACED frame is recorded on the RAW clock like any other: the achieved
   * frame rate really is lower, and the FPS readout must not lie. A steady
   * paced cadence is absorbed by the stall detector's outlier test
   * (`rendering/adaptive-dpr/stall-detector.ts` compares against the recent
   * MEDIAN, not an absolute threshold), so it does not manufacture gap
   * resets; and the isolated-hiccup interactions — a paced interval read off
   * a freshly cleared window as a collapsed frame rate, or a
   * just-under-`gapResetMs` frame pushed just over it — cannot arise at all,
   * because pacing requires a STREAK (see PACING_SLOW_FRAME_STREAK in
   * raf-driver.ts) and an isolated slow frame is therefore never paced.
   */
  private recordAdaptiveDprSample(tickStart: number): void {
    if (!this.adaptiveDPRManager) return;
    if (this.renderOnChange && !this.previousTickRendered) return;
    if (this.isGpuWorkBlocked()) return;
    perfCounters.add(S_ADAPTIVE_DPR_SAMPLES);
    this.adaptiveDPRManager.recordFrame(tickStart);
  }

  /**
   * Why this tick must render, or null when nothing changed since the last
   * rendered frame. Precedence as documented on {@link RenderReason}.
   */
  private renderReason(changedBy: string | null): RenderReason | null {
    if (this.dirtyReason !== null) return this.dirtyReason;
    if (changedBy !== null) return `cb:${changedBy}`;
    if (!this.renderOnChange) return 'continuous';
    if (this.viewSignature?.changed()) return 'camera';
    return this.mustRenderEveryTick() ? 'continuous' : null;
  }

  /**
   * A tick with nothing new to draw. Under `?renderAudit` it is rendered
   * anyway and its pixels checked against the last real frame.
   */
  private skipTick(): boolean {
    if (!this.audit) {
      perfCounters.add(S_RENDER_SKIPPED_TICKS);
      this.previousTickRendered = false;
      return false;
    }
    perfCounters.add(S_RENDER_COUNT);
    perfCounters.add(reasonSlot('audit'));
    this.postProcessing.render();
    this.previousTickRendered = true;
    this.audit.checkSkippedFrame(this.tickCount);
    return true;
  }

  /** Render the frame for `reason` and remember it as the last rendered one. */
  private renderFrame(reason: RenderReason): void {
    // Cleared BEFORE the render: a request made during it dirties the next tick.
    this.dirtyReason = null;
    perfCounters.add(S_RENDER_COUNT);
    perfCounters.add(reasonSlot(reason));
    // Render through HDR post-processing pipeline
    // This executes the complete chain: Scene → HDR buffer → Bloom → Tone mapping → Display
    // Includes vertex shaders, fragment shaders, HDR buffers, bloom blur, ACES tone mapping
    this.postProcessing.render();
    this.previousTickRendered = true;
    if (this.renderOnChange) this.viewSignature?.commit();
    this.audit?.captureReference();
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
   *
   * @returns the id of the first callback that returned `true` (changed what
   *   is drawn), or null. Every callback runs either way.
   */
  private runPerFrameCallbacks(): string | null {
    let changedBy: string | null = null;
    for (const phase of FRAME_PHASES) {
      for (const [id, entry] of this.callbacksByPhase[phase]) {
        if (this.runCallback(id, entry) && changedBy === null) changedBy = id;
      }
    }
    return changedBy;
  }

  /**
   * Run one callback, logging its first failure (see {@link runPerFrameCallbacks}).
   *
   * @returns whether it reported a drawn-state change (a throw reports none)
   */
  private runCallback(id: string, entry: PerFrameEntry): boolean {
    try {
      return entry.callback() === true;
    } catch (error) {
      if (!this.failedCallbackIds.has(id)) {
        this.failedCallbackIds.add(id);
        log.error(
          Modules.ANIMATION,
          `Per-frame callback '${id}' threw; skipping it this frame`,
          error
        );
      }
      return false;
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
      this.renderIdleRestoreFrame();
    }
  };

  /**
   * Idle restore: the static frame the user is about to study should be at
   * full native sharpness — reduced DPR only ever traded quality for
   * interaction smoothness. This lives ONLY in the idle path (never in
   * stopAnimation itself, which also runs on tab-hide and dispose where
   * rendering would be wrong). prepareIdleFrame() returns true only when the
   * DPR actually changed; the resize clears the canvas, so exactly then we
   * render ONE frame directly — NOT via startAnimation(), which would re-arm
   * the idle timer and feed ceiling-DPR frames back into the FPS evaluator.
   *
   * The render-skip predicate is checked here too — this is the loop's OTHER
   * render call site, and the predicate's claim is "nobody but the pipeline's
   * current owner may draw", not "the tick() path may not draw". Today it is
   * redundant (a capture disables adaptive DPR, so isActive() is already
   * false, and the idle-restore predicate is off for the whole capture), but
   * the guard that makes it redundant lives in another file: drop
   * `captureDPR` from the capture's saveRecordingState and this would paint a
   * ceiling-DPR frame through the capture scrim, possibly inside the raw-HDR
   * window. It must come BEFORE prepareIdleFrame(), which RESIZES on its way
   * to returning true — skipping the render after that resize would leave the
   * canvas cleared with nothing to repaint it.
   */
  private renderIdleRestoreFrame(): void {
    if (!this.mayRestoreAtIdle()) return;
    if (this.adaptiveDPRManager?.prepareIdleFrame?.() !== true) return;
    perfCounters.add(S_RENDER_COUNT);
    perfCounters.add(reasonSlot('idleRestore'));
    this.postProcessing.render();
    // The resting frame is now the last rendered one (at the ceiling DPR).
    if (this.renderOnChange) this.viewSignature?.commit();
  }

  /** Whether the idle restore may run at all (checked BEFORE it resizes). */
  private mayRestoreAtIdle(): boolean {
    if (this.adaptiveDPRManager?.isActive?.() !== true) return false;
    if (this.canRestoreAtIdle?.() === false) return false;
    return !this.isGpuWorkBlocked();
  }

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
   * Every call also marks the next tick dirty (`render.byReason.wake`): a
   * caller that wakes the loop gets at least one rendered frame, whatever it
   * changed. Equivalent to `requestRender` + resume + idle-timer reset.
   *
   * On a stopped loop this ARMS the next animation frame; it does not draw
   * synchronously (see RafDriver.start). Use {@link renderOnce} for a frame
   * now.
   *
   * Uses arrow function to maintain 'this' context when used as event handler.
   */
  startAnimation = (): void => {
    this.markDirty('wake');
    // Only start if not already running - prevents duplicate loops
    this.resumeIfStopped();
    this.resetIdleTimer();
  };

  /**
   * Draw a frame NOW if the loop is stopped, then keep it running as
   * {@link startAnimation} does.
   *
   * `startAnimation()` only arms the next animation frame. This is for the
   * few callers that must have a frame on the canvas before they return: a
   * resize made inside an animation-frame callback (arming there would land a
   * frame late and composite one cleared frame), or a video capture that
   * starts recording the canvas in the same turn. When the loop is already
   * running it draws nothing extra: it marks the next tick dirty, and the
   * running loop paints it at the next animation frame.
   */
  renderOnce(): void {
    this.markDirty('once');
    if (this.resumeIfStopped()) {
      perfCounters.add(S_RENDER_ONCE);
      this.tick();
    }
    this.resetIdleTimer();
  }

  /**
   * Bring the view state up to date for the current camera without drawing:
   * `controls.update()` and every per-frame callback, as a frame would run
   * them, but no render and no frame events.
   *
   * For a caller that renders its own pipeline pass outside the loop (an
   * embedder screenshot): after a camera change the per-frame view callbacks
   * (clipping planes, LOD selection, depth sort) have not run yet until the
   * loop's next frame, so a pass drawn before it would use the previous
   * pose's near/far and LOD. A callback that changed drawn state here marks
   * the loop's next tick dirty, so the canvas catches up too.
   */
  prepareFrame(): void {
    this.controls.update();
    if (this.runPerFrameCallbacks() !== null) this.markDirty('event', 'prepareFrame');
  }

  /**
   * Start the driver if it is stopped.
   *
   * @returns true on the stopped→running edge
   */
  private resumeIfStopped(): boolean {
    if (this.driver.isRunning) return false;
    // Resuming from a rest: let the adaptive DPR manager snap back to
    // its remembered operating DPR in one step (stopped→running edge
    // only — this must not fire on every interaction poke).
    this.adaptiveDPRManager?.notifyResumed?.();
    // Arms the first frame for the next animation frame; later frames are
    // scheduled by the driver.
    this.driver.start();
    return true;
  }

  /** Restart the idle countdown (see {@link handleIdleTimeout}). */
  private resetIdleTimer(): void {
    // Reset the idle timeout - this is called on every user interaction
    // Clear any existing timeout to prevent premature stopping
    if (this.idleTimeout !== null) {
      clearTimeout(this.idleTimeout);
    }

    // Set new timeout to check for idle - will continue if continuous effects are active
    // This is the key power-saving optimization for static scenes
    this.idleTimeout = setTimeout(this.handleIdleTimeout, config.animation.idleTimeoutMs);
  }

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
    // The next rendered tick follows a gap: re-based, not sampled.
    this.previousTickRendered = false;

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
    this.renderEveryFrameCount = 0;
  }
}
